import glob
import os
import sys

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, os.getcwd())      # lets us import mock_llm from the repo root
import mock_llm

POLICY_FOLDER = "data/policies"
CHUNK_WORDS = 40
DEFAULT_OVERLAP = 10

# Numbers from the brief. Used ONLY to compare, never to compute.
EXPECTED_CHUNKS = {"refund_policy.txt": 4, "delivery_policy.txt": 3,
                   "escalation_policy.txt": 3, "payment_policy.txt": 3,
                   "privacy_policy.txt": 2}

SYSTEM_PROMPT = (
    "You are a POLICY ASSISTANT for a grocery delivery company. Answer only from "
    "the supplied context. If the answer is not in the context, say that you could "
    "not find the information."
)


# Task 1: splits every policy file into chunks of 40 words that overlap
# by `overlap` words. Each chunk remembers which file it came from.
def chunk_documents(overlap=DEFAULT_OVERLAP):
    chunks = []
    step = CHUNK_WORDS - overlap
    for path in sorted(glob.glob(os.path.join(POLICY_FOLDER, "*.txt"))):
        words = open(path, encoding="utf-8").read().split()
        start = 0
        while True:
            end = start + CHUNK_WORDS
            chunks.append({"source": os.path.basename(path),
                           "text": " ".join(words[start:end])})
            if end >= len(words):          # this chunk reached the last word
                break
            start += step
    return chunks


# Task 2: turns the chunks into TF-IDF vectors once, then finds the chunks
# most similar to a question using cosine similarity.
class PolicyIndex:
    def __init__(self, overlap=DEFAULT_OVERLAP):
        self.chunks = chunk_documents(overlap)
        self.vectorizer = TfidfVectorizer()
        self.matrix = self.vectorizer.fit_transform([c["text"] for c in self.chunks])

    # Returns the k best chunks as dictionaries with source, score and text.
    def retrieve(self, query, k=3):
        scores = cosine_similarity(self.vectorizer.transform([query]), self.matrix)[0]
        best = scores.argsort()[::-1][:k]
        return [{"source": self.chunks[i]["source"],
                 "score": float(scores[i]),
                 "text": self.chunks[i]["text"]} for i in best]


_default_index = None


# Convenience function for other files (the refund agent uses this one).
def retrieve(query, k=3):
    global _default_index
    if _default_index is None:
        _default_index = PolicyIndex()
    return _default_index.retrieve(query, k)


# Task 3: finds three chunks, sends them with the question to the model,
# and returns the model's text together with the sources used.
def answer(question, index=None):
    results = (index.retrieve(question, 3) if index else retrieve(question, 3))
    context = "\n".join(f"[{r['source']}]\n{r['text']}" for r in results)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"CONTEXT:\n{context}\nQUESTION:\n{question}"},
    ]
    response = mock_llm.chat(messages)
    return {"answer": response.choices[0].message.content,
            "sources": [r["source"] for r in results]}


# Task 1 report: counts chunks per file and compares with the brief.
def chunk_report():
    chunks = chunk_documents(DEFAULT_OVERLAP)
    counts = pd.Series([c["source"] for c in chunks]).value_counts().to_dict()
    print("=== CHUNKS (40 words, overlap 10) ===")
    print("total:", len(chunks), "(reference 15)")
    for source, expected in EXPECTED_CHUNKS.items():
        got = counts.get(source, 0)
        print(f"{source}: {got} (reference {expected}) {'OK' if got == expected else 'CHECK'}")


# Task 2 report: the sample query from the brief should hit only refund_policy.txt.
def retrieval_report():
    results = retrieve("refund policy late delivery", 3)
    print("\n=== RETRIEVAL: 'refund policy late delivery' ===")
    for r in results:
        print(f"{r['source']} | score {r['score']:.3f}")
    print("all three from refund_policy.txt:",
          all(r["source"] == "refund_policy.txt" for r in results))


# Task 4: runs the 14 evaluation questions and prints the three scores.
# Answerable questions are checked for the right document and the right fact.
# Unanswerable ones must get the "I could not find" fallback.
def evaluate(overlap):
    index = PolicyIndex(overlap)
    questions = pd.read_csv("data/raw/policy_eval.csv", keep_default_na=False)
    answerable = unanswerable = 0
    doc_hits = fact_hits = fallback_hits = 0
    failures = []
    for row in questions.itertuples(index=False):
        result = answer(row.question, index)
        if row.expected_doc == "NONE":
            unanswerable += 1
            if result["answer"].startswith("I could not find"):
                fallback_hits += 1
            else:
                failures.append((row.question, "should have said it could not find"))
        else:
            answerable += 1
            doc_ok = row.expected_doc in result["sources"]
            fact_ok = row.expected_fact in result["answer"]
            doc_hits += doc_ok
            fact_hits += fact_ok
            if not (doc_ok and fact_ok):
                failures.append((row.question,
                                 f"doc found: {doc_ok}, fact in answer: {fact_ok} "
                                 f"(wanted '{row.expected_fact}')"))
    print(f"\n=== EVALUATION: {len(index.chunks)} chunks, overlap {overlap} ===")
    print(f"expected document in top 3: {doc_hits}/{answerable} (reference 12/12 at overlap 10)")
    print(f"expected fact in answer:    {fact_hits}/{answerable} (reference 12/12 at overlap 10)")
    print(f"fallback for unanswerable:  {fallback_hits}/{unanswerable} (reference 2/2)")
    for question, reason in failures:
        print("FAILED:", question, "->", reason)


# Start here.
if __name__ == "__main__":
    chunk_report()
    retrieval_report()
    evaluate(overlap=10)
    evaluate(overlap=0)