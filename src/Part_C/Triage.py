import os
import random
import re
import sys
import time
from collections import Counter

import pandas as pd

sys.path.insert(0, os.getcwd())      # lets us import mock_llm from the repo root
import mock_llm
from mock_llm import TransientLLMError, est_tokens

CATEGORIES = ["billing", "delivery", "technical", "fraud"]
MAX_INPUT_TOKENS = 400
MAX_RETRIES = 3
BASE_DELAY = 0.05
DEMO_BUDGETS = [6000, 5000, 4000]    # first budget that exhausts at least one ticket is used
FINAL_BUDGET = 100000
CLASSIFIED = ["classified_json", "classified_keyword"]
NEEDS_HUMAN = ["skipped_empty", "flagged_long", "flagged_invalid_output", "budget_exhausted"]

# Numbers from the brief. Used ONLY to compare, never to compute.
REFERENCE = {
    "empty tickets": 4, "long tickets": 4,
    "emails": 55, "phones": 91, "order ids": 132, "total redactions": 278,
    "transient errors": 19,
    "classified_json": 135, "classified_keyword": 12, "flagged_invalid_output": 9,
    "classified tickets": 147,
    "zero-shot accuracy %": 85.7, "few-shot accuracy %": 100.0,
}

# The three redaction rules, applied in this order.
REDACTION_RULES = [
    ("emails", re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "[EMAIL]"),
    ("phones", re.compile(r"\b[6-9]\d{9}\b"), "[PHONE]"),
    ("order ids", re.compile(r"ORD-\d{6}"), "[ORDER_ID]"),
]

ZERO_SHOT_PROMPT = (
    "You CLASSIFY customer support tickets into one category: billing, delivery, "
    "technical or fraud. Reply with JSON only, exactly like "
    '{"category": "...", "reason": "..."}.'
)
FEW_SHOT_PROMPT = ZERO_SHOT_PROMPT + "\n" + "\n".join([
    "EXAMPLES:",
    'Ticket: I was charged twice for order [ORDER_ID]. -> {"category": "billing", "reason": "double charge"}',
    'Ticket: The rider has not arrived and delivery is late. -> {"category": "delivery", "reason": "late order"}',
    'Ticket: The app crashes when I open the cart. -> {"category": "technical", "reason": "app problem"}',
    'Ticket: There is an unauthorized transaction on my card. -> {"category": "fraud", "reason": "unauthorized payment"}',
])


# Prints our value next to the brief's reference, with OK or CHECK.
def compare_with_reference(name, actual, tolerance=0.0):
    expected = REFERENCE[name]
    status = "OK" if abs(actual - expected) <= tolerance else "CHECK"
    print(f"{name}: {actual} (reference {expected}) {status}")


# Replaces email, phone and order id (in that order) with placeholders.
# Returns the clean text and how many of each were replaced.
def redact(text):
    counts = Counter()
    for name, pattern, placeholder in REDACTION_RULES:
        text, number = pattern.subn(placeholder, text)
        counts[name] += number
    return text, counts


# Task 2: counts redactions over all tickets, then re-scans the redacted
# texts to prove nothing is left.
def redaction_report(tickets):
    totals = Counter()
    redacted_texts = []
    for text in tickets["text"]:
        clean, counts = redact(text)
        totals.update(counts)
        redacted_texts.append(clean)
    print("=== REDACTION ===")
    compare_with_reference("emails", totals["emails"])
    compare_with_reference("phones", totals["phones"])
    compare_with_reference("order ids", totals["order ids"])
    compare_with_reference("total redactions", sum(totals.values()))
    left = sum(len(pattern.findall(text)) for text in redacted_texts
               for _, pattern, _ in REDACTION_RULES)
    print("patterns still found after redaction:", left, "(reference 0)")
    assert left == 0


# Task 5: reads the model's reply. Valid JSON with an allowed category is
# accepted. Plain text is accepted only if exactly one allowed category word
# appears. Anything else goes to a human; we never guess.
def validate_reply(text):
    import json
    try:
        data = json.loads(text)
    except (TypeError, ValueError):
        words = set(re.findall(r"[a-z]+", (text or "").lower()))
        found = [word for word in CATEGORIES if word in words]
        if len(found) == 1:
            return "classified_keyword", found[0]
        return "flagged_invalid_output", None
    if isinstance(data, dict) and data.get("category") in CATEGORIES:
        return "classified_json", data["category"]
    return "flagged_invalid_output", None


# Task 4: calls the model, retrying temporary failures up to 3 times with
# exponential backoff and jitter. Every retry is logged.
def call_with_retry(messages, ticket_id, run):
    for retry_number in range(MAX_RETRIES + 1):
        try:
            return mock_llm.chat(messages)
        except TransientLLMError:
            run["errors"] += 1
            if retry_number == MAX_RETRIES:
                raise
            wait = BASE_DELAY * 2 ** retry_number * random.uniform(0.9, 1.15)
            run["retry_log"].append((ticket_id, retry_number, wait))
            time.sleep(wait)


# Runs every ticket, in file order, through the guardrails and the model.
# token_budget=None means no budget.
def run_pipeline(tickets, system_prompt, token_budget=None):
    mock_llm.reset()
    random.seed(42)
    run = {"rows": [], "errors": 0, "retry_log": [], "sent": [], "total_tokens": 0}
    for t in tickets.itertuples(index=False):
        status, category, tokens = None, None, 0
        if not t.text.strip():
            status = "skipped_empty"
        elif est_tokens(t.text) > MAX_INPUT_TOKENS:
            status = "flagged_long"
        elif token_budget is not None and run["total_tokens"] >= token_budget:
            status = "budget_exhausted"
        else:
            clean_text, _ = redact(t.text)
            messages = [{"role": "system", "content": system_prompt},
                        {"role": "user", "content": clean_text}]
            run["sent"].append(messages)
            response = call_with_retry(messages, t.ticket_id, run)
            tokens = response.usage.total_tokens
            run["total_tokens"] += tokens
            status, category = validate_reply(response.choices[0].message.content)
        run["rows"].append({"ticket_id": t.ticket_id, "customer_id": t.customer_id,
                            "status": status, "category": category,
                            "tokens_used": tokens})
    return run


# Proves that no email, phone or order id was ever sent to the model.
def assert_no_pii_sent(run):
    for messages in run["sent"]:
        for message in messages:
            for _, pattern, _ in REDACTION_RULES:
                assert not pattern.search(message["content"]), "PII reached the prompt"
    print("no email, phone or order id in any prompt:", True)


# Prints the retry log and checks each wait lies in the allowed jitter range.
def report_retries(run):
    print("\n=== RETRY LOG ===")
    for ticket_id, number, wait in run["retry_log"]:
        low = BASE_DELAY * 2 ** number * 0.9
        high = BASE_DELAY * 2 ** number * 1.15
        assert low <= wait <= high
        print(f"ticket {ticket_id} | retry {number} | wait {wait:.4f}s")
    compare_with_reference("transient errors", run["errors"])
    print("retries logged:", len(run["retry_log"]), "| all recovered (no permanent error)")


# Task 6: status counts, accuracy over classified tickets and token cost.
def summarize(name, run, tickets):
    df = pd.DataFrame(run["rows"])
    truth = tickets.set_index("ticket_id")["true_category"]
    classified = df[df["status"].isin(CLASSIFIED)]
    correct = int((classified["category"] == classified["ticket_id"].map(truth)).sum())
    accuracy = round(correct / len(classified) * 100, 1)
    counts = df["status"].value_counts().to_dict()

    print(f"\n=== {name.upper()} RUN ===")
    print("status counts:", counts)
    compare_with_reference("classified_json", counts.get("classified_json", 0))
    compare_with_reference("classified_keyword", counts.get("classified_keyword", 0))
    compare_with_reference("flagged_invalid_output", counts.get("flagged_invalid_output", 0))
    compare_with_reference("classified tickets", len(classified))
    compare_with_reference(f"{name} accuracy %", accuracy, tolerance=0.5)
    print(f"correct {correct} of {len(classified)}")
    print(f"total tokens {run['total_tokens']} | tokens per classified ticket "
          f"{run['total_tokens'] / len(classified):.1f}")
    return df, classified, truth


# Shows true category (rows) against predicted category (columns).
def confusion_table(classified, truth):
    true_labels = classified["ticket_id"].map(truth)
    print("\nconfusion table (rows = true, columns = predicted):")
    print(pd.crosstab(true_labels, classified["category"]).to_string())


# Task 7a: runs zero-shot with a token budget and saves the demo file.
def budget_demo(tickets):
    for budget in DEMO_BUDGETS:
        run = run_pipeline(tickets, ZERO_SHOT_PROMPT, token_budget=budget)
        df = pd.DataFrame(run["rows"])
        exhausted = int((df["status"] == "budget_exhausted").sum())
        if exhausted > 0:
            break
    os.makedirs("outputs", exist_ok=True)
    df.to_csv("outputs/triage_budget_demo.csv", index=False)
    print("\n=== BUDGET DEMO ===")
    print(f"TOKEN_BUDGET used: {budget} | tokens spent: {run['total_tokens']} "
          f"| budget_exhausted tickets: {exhausted}")


# Task 7b: decides where each ticket goes.
def route_ticket(status, category, is_premium):
    if status in NEEDS_HUMAN:
        return "human_review", "NORMAL"
    if category == "fraud":
        return "security_team", "HIGH"
    if is_premium:
        return "premium_desk", "HIGH"
    return "support_queue", "NORMAL"


# Task 7b: few-shot run with a big budget, routed using the Part B segments.
def final_results(tickets):
    run = run_pipeline(tickets, FEW_SHOT_PROMPT, token_budget=FINAL_BUDGET)
    df = pd.DataFrame(run["rows"])

    segments = pd.read_csv("outputs/customer_segments.csv")
    segment_name = dict(zip(segments["customer_id"], segments["segment_name"]))
    premium_ids = set(segments.loc[
        segments["is_premium"].astype(str).str.lower() == "true", "customer_id"])

    df["segment_name"] = df["customer_id"].map(segment_name)
    routes = [route_ticket(row.status, row.category, row.customer_id in premium_ids)
              for row in df.itertuples()]
    df["route"] = [r[0] for r in routes]
    df["priority"] = [r[1] for r in routes]
    df = df[["ticket_id", "customer_id", "status", "category", "segment_name",
             "priority", "route", "tokens_used"]].fillna("")
    df.to_csv("outputs/triage_results.csv", index=False)

    print("\n=== FINAL ROUTING (few-shot, budget 100000) ===")
    print("rows:", len(df), "(reference 164)")
    print(df["route"].value_counts().to_string())
    print("priority:", df["priority"].value_counts().to_dict())


# Start here.
if __name__ == "__main__":
    tickets = pd.read_csv("data/raw/tickets.csv", keep_default_na=False, dtype=str)

    redaction_report(tickets)

    zero_run = run_pipeline(tickets, ZERO_SHOT_PROMPT)
    df, _, _ = summarize("zero-shot", zero_run, tickets)
    compare_with_reference("empty tickets", int((df["status"] == "skipped_empty").sum()))
    compare_with_reference("long tickets", int((df["status"] == "flagged_long").sum()))
    long_tokens = {est_tokens(t) for t in tickets["text"] if est_tokens(t) > MAX_INPUT_TOKENS}
    print("estimated tokens of the long tickets:", long_tokens, "(reference 734)")
    assert_no_pii_sent(zero_run)
    report_retries(zero_run)
    _, zero_classified, truth = summarize("zero-shot", zero_run, tickets)
    confusion_table(zero_classified, truth)

    few_run = run_pipeline(tickets, FEW_SHOT_PROMPT)
    summarize("few-shot", few_run, tickets)
    assert_no_pii_sent(few_run)

    budget_demo(tickets)
    final_results(tickets)