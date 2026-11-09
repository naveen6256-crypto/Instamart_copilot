import json
import os
import sqlite3
import sys
import time
from datetime import datetime

import pandas as pd

sys.path.insert(0, os.getcwd())      # lets us import mock_llm from the repo root
import mock_llm
from Policy_rag import retrieve      # reuses the retrieval built in tasks 1-3

DB = "data/instamart.db"
LEDGER = "outputs/refund_ledger.jsonl"
RUN_LOG = "outputs/agent_runs.jsonl"
STATE_FOLDER = "state"
LATE_MINUTES = 45

SYSTEM_PROMPT = (
    "You are a refund AGENT. Look up the order, check the refund rules, calculate "
    "the refund, then issue it or escalate it. Finish with a short final answer."
)

# The run id of the run that is executing now. Written into the ledger lines.
CURRENT = {"run_id": ""}


class SimulatedCrash(Exception):
    """Raised on purpose to demonstrate crash and resume."""


# ---------------------------------------------------------------- tools

# Looks up one order in the database from Part A.
def get_order(order_id):
    with sqlite3.connect(DB) as con:
        row = con.execute(
            "SELECT order_id, status, order_amount, delivery_minutes "
            "FROM orders WHERE order_id = ?", (order_id,)).fetchone()
    if row is None:
        return {"error": "order not found"}
    return {"order_id": row[0], "status": row[1],
            "order_amount": row[2], "delivery_minutes": row[3]}


# Searches the policy documents with the retrieval built earlier.
def search_policy(query):
    results = retrieve(query, 3)
    return {"results": [{"source": r["source"], "score": round(r["score"], 3),
                         "text": r["text"]} for r in results]}


# Late (more than 45 minutes) = full refund. On time, or never delivered = 0.
def calculate_refund(order_amount, delivery_minutes):
    if delivery_minutes is None or delivery_minutes <= LATE_MINUTES:
        return {"refund_inr": 0.0}
    return {"refund_inr": round(float(order_amount), 2)}


# Adds one line to the ledger file.
def write_ledger(tool, order_id, refund_inr):
    os.makedirs("outputs", exist_ok=True)
    line = {"timestamp": datetime.now().isoformat(timespec="seconds"),
            "tool": tool, "run_id": CURRENT["run_id"],
            "order_id": order_id, "refund_inr": refund_inr}
    with open(LEDGER, "a", encoding="utf-8") as f:
        f.write(json.dumps(line) + "\n")


# Records an automatic refund (amounts up to INR 1000).
def issue_refund(order_id, refund_inr):
    if refund_inr > 1000:
        return {"error": "refund above INR 1000 must be escalated to a human"}
    write_ledger("issue_refund", order_id, refund_inr)
    return {"refund_inr": refund_inr}


# Records a refund above INR 1000 that a human lead must approve.
def escalate_to_human(order_id, refund_inr):
    write_ledger("escalate_to_human", order_id, refund_inr)
    return {"refund_inr": refund_inr}


# The model asks for a tool by name. Our code looks the name up here.
TOOL_MAP = {
    "get_order": get_order,
    "search_policy": search_policy,
    "calculate_refund": calculate_refund,
    "issue_refund": issue_refund,
    "escalate_to_human": escalate_to_human,
}

# What each tool does and which arguments it takes (JSON-style description).
TOOLS_SPEC = [
    {"name": "get_order",
     "description": "Look up an order by id in the Instamart database.",
     "parameters": {"type": "object",
                    "properties": {"order_id": {"type": "string"}},
                    "required": ["order_id"]}},
    {"name": "search_policy",
     "description": "Search the policy documents and return the best matching passages.",
     "parameters": {"type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"]}},
    {"name": "calculate_refund",
     "description": "Calculate the refund in INR from the order amount and delivery minutes.",
     "parameters": {"type": "object",
                    "properties": {"order_amount": {"type": "number"},
                                   "delivery_minutes": {"type": "number"}},
                    "required": ["order_amount", "delivery_minutes"]}},
    {"name": "issue_refund",
     "description": "Issue an automatic refund of up to INR 1000 and record it in the ledger.",
     "parameters": {"type": "object",
                    "properties": {"order_id": {"type": "string"},
                                   "refund_inr": {"type": "number"}},
                    "required": ["order_id", "refund_inr"]}},
    {"name": "escalate_to_human",
     "description": "Send a refund above INR 1000 to a human support lead and record it.",
     "parameters": {"type": "object",
                    "properties": {"order_id": {"type": "string"},
                                   "refund_inr": {"type": "number"}},
                    "required": ["order_id", "refund_inr"]}},
]


# ---------------------------------------------------------------- checkpoints

# Saves everything needed to continue this run later.
def save_checkpoint(state):
    os.makedirs(STATE_FOLDER, exist_ok=True)
    with open(os.path.join(STATE_FOLDER, f"{state['run_id']}.json"), "w",
              encoding="utf-8") as f:
        json.dump(state, f, indent=2)


# Loads a saved run.
def load_checkpoint(run_id):
    with open(os.path.join(STATE_FOLDER, f"{run_id}.json"), encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------- the loop

# Builds the result dictionary that every run returns.
def make_result(state, status, final_answer):
    return {"run_id": state["run_id"], "scenario_id": state["scenario_id"],
            "status": status, "final_answer": final_answer,
            "steps": state["steps"], "llm_calls": state["llm_calls"],
            "tokens_used": state["tokens_used"],
            "tools_executed": state["tools_executed"]}


# Stops safely: returns a fallback status and a short note for a human.
def fallback(state, status, reason):
    tools = ", ".join(state["tools_executed"]) or "none"
    summary = (f"Handing over to a human agent. Stopped because {reason}. "
               f"Tools run so far: {tools}. Request: {state['messages'][1]['content']}")
    return make_result(state, status, summary)


# The ReAct loop: ask the model, run the tool it asks for, show it the result,
# repeat until it gives a final answer or a limit is reached.
# Used by both run_agent (fresh run) and resume_agent (continue a saved run).
def agent_loop(state, started, crash_after_tools=None):
    CURRENT["run_id"] = state["run_id"]
    while True:
        # limit: wall-clock time (checked before every model call)
        if time.monotonic() - started >= state["time_limit_seconds"]:
            return fallback(state, "FALLBACK_TIMEOUT", "the time limit was reached")
        # limit: number of model calls
        if state["llm_calls"] >= state["max_iterations"]:
            return fallback(state, "FALLBACK_MAX_ITERATIONS",
                            f"{state['max_iterations']} model calls gave no final answer")

        response = mock_llm.chat(state["messages"], tools=TOOLS_SPEC)
        state["llm_calls"] += 1
        state["tokens_used"] += response.usage.total_tokens

        # limit: tokens (stop BEFORE running the tool that was just requested)
        if state["tokens_used"] > state["token_limit"]:
            return fallback(state, "FALLBACK_TOKEN_LIMIT", "the token limit was exceeded")

        message = response.choices[0].message
        if not message.tool_calls:                       # final answer
            answer = message.content
            return make_result(state, answer.split(":")[0].strip(), answer)

        call = message.tool_calls[0]                     # the mock asks for one tool at a time
        state["messages"].append({
            "role": "assistant", "content": None,
            "tool_calls": [{"id": call.id, "name": call.name, "arguments": call.arguments}]})

        function = TOOL_MAP.get(call.name)
        observation = function(**call.arguments) if function else {"error": "unknown tool"}

        state["messages"].append({"role": "tool", "tool_call_id": call.id,
                                  "name": call.name, "content": json.dumps(observation)})
        state["steps"].append({"tool": call.name, "arguments": call.arguments,
                               "observation": observation})
        state["tools_executed"].append(call.name)
        state["step"] = len(state["tools_executed"])
        save_checkpoint(state)                           # saved AFTER the observation is added

        if crash_after_tools and len(state["tools_executed"]) == crash_after_tools:
            raise SimulatedCrash(f"simulated crash after tool {crash_after_tools}")


# Adds one line per finished scenario run to outputs/agent_runs.jsonl.
def log_run(result):
    os.makedirs("outputs", exist_ok=True)
    line = {key: result[key] for key in
            ["scenario_id", "status", "final_answer", "llm_calls", "tokens_used", "steps"]}
    with open(RUN_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(line) + "\n")


# Starts a fresh run for one request.
def run_agent(user_message, scenario_id, max_iterations=6, token_limit=6000,
              time_limit_seconds=10, run_id=None, crash_after_tools=None, log=True):
    state = {
        "run_id": run_id or f"{scenario_id.lower()}_run",
        "scenario_id": scenario_id,
        "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                     {"role": "user", "content": user_message}],
        "tokens_used": 0, "llm_calls": 0, "step": 0,
        "tools_executed": [], "steps": [],
        "max_iterations": max_iterations, "token_limit": token_limit,
        "time_limit_seconds": time_limit_seconds,
    }
    result = agent_loop(state, time.monotonic(), crash_after_tools)
    if log:
        log_run(result)
    return result


# Loads a saved run and continues it. Tools that already ran are not run again,
# because the model only sees the saved messages and asks for the NEXT tool.
def resume_agent(run_id):
    state = load_checkpoint(run_id)
    return agent_loop(state, time.monotonic())


# ---------------------------------------------------------------- scenarios and demos

# The refund (or escalation) amount of a run, taken from its last money step.
def refund_amount(result):
    for step in reversed(result["steps"]):
        if step["tool"] in ("issue_refund", "escalate_to_human"):
            return float(step["observation"]["refund_inr"])
    return 0.0


# Deletes the files from earlier runs so every full run starts clean.
def reset_outputs():
    for path in [LEDGER, RUN_LOG]:
        if os.path.exists(path):
            os.remove(path)
    if os.path.isdir(STATE_FOLDER):
        for name in os.listdir(STATE_FOLDER):
            if name.endswith(".json"):
                os.remove(os.path.join(STATE_FOLDER, name))


# Task 6: runs all 7 scenarios and prints the result table.
def run_scenarios():
    scenarios = pd.read_csv("data/raw/agent_scenarios.csv", keep_default_na=False)
    print("=== SEVEN SCENARIOS ===")
    print("scenario | status | refund INR | model calls | tokens | expected | expected INR | match")
    first_message = None
    for row in scenarios.itertuples(index=False):
        result = run_agent(row.message, row.scenario_id,
                           run_id=f"{row.scenario_id.lower()}_main")
        refund = refund_amount(result)
        expected_refund = float(row.expected_refund_inr)
        status_ok = result["status"].startswith(row.expected_outcome)
        match = status_ok and abs(refund - expected_refund) <= 0.01
        print(f"{row.scenario_id} | {result['status']} | {refund:.2f} | {result['llm_calls']} | "
              f"{result['tokens_used']} | {row.expected_outcome} | {expected_refund:.2f} | "
              f"{'OK' if match else 'CHECK'}")
        if row.scenario_id == "S1":
            first_message = row.message
    return first_message


# Task 7: shows the token-limit stop and the timeout stop on scenario S1.
def stop_safely_demos(s1_message):
    token_run = run_agent(s1_message, "S1", token_limit=200,
                          run_id="s1_token_limit_demo", log=False)
    print("\n=== STOP SAFELY ===")
    print(f"S1 with token_limit=200: {token_run['status']} "
          f"(tokens {token_run['tokens_used']}, model calls {token_run['llm_calls']})")
    print("handoff note:", token_run["final_answer"])

    time_run = run_agent(s1_message, "S1", time_limit_seconds=0,
                         run_id="s1_timeout_demo", log=False)
    print(f"S1 with time_limit_seconds=0: {time_run['status']} "
          f"(model calls {time_run['llm_calls']})")


# Task 8: crashes S1 right after its second tool, then resumes it.
def crash_and_resume_demo(s1_message):
    run_id = "s1_crash_demo"
    try:
        run_agent(s1_message, "S1", run_id=run_id, crash_after_tools=2, log=False)
    except SimulatedCrash as error:
        print("\n=== CRASH AND RESUME ===")
        print("crash:", error)
    before = load_checkpoint(run_id)["tools_executed"]
    print("tools executed before the crash:", before)

    result = resume_agent(run_id)
    after = result["tools_executed"][len(before):]
    print("tools executed after the resume:", after)
    print("all tools across both parts, in order:", result["tools_executed"])
    print("final status:", result["status"], "| refund INR", f"{refund_amount(result):.2f}",
          "| model calls", result["llm_calls"])
    assert result["tools_executed"] == ["get_order", "search_policy",
                                        "calculate_refund", "issue_refund"]
    assert result["status"] == "REFUND" and abs(refund_amount(result) - 128.25) <= 0.01
    print("four tool executions, none repeated: True")


# Start here.
if __name__ == "__main__":
    reset_outputs()
    s1_message = run_scenarios()
    stop_safely_demos(s1_message)
    crash_and_resume_demo(s1_message)