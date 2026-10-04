import json
from pathlib import Path
from store import get_collection

QUESTIONS_PATH = Path(__file__).resolve().parent.parent / "data" / "eval" / "questions.json"

def norm(s):
    return " ".join(s.lower().split())

questions = json.load(open(QUESTIONS_PATH))
collection = get_collection()
MISSES = ["demand", "export restrictions", "security and execution"]  # substrings of the 3 questions

for q in questions:
    if not any(m in q["question"] for m in MISSES):
        continue
    where = {"$and": [{"ticker": q["ticker"]}, {"fiscal_year": q["fiscal_year"]}]}
    res = collection.query(query_texts=[q["question"]], n_results=30, where=where)
    rank = None
    for i, doc in enumerate(res["documents"][0], start=1):
        if norm(q["evidence"]) in norm(doc):
            rank = i
            break
    print("\n", q["question"])
    print("  evidence chunk rank in top 30:", rank)
    for i in range(5):
        m = res["metadatas"][0][i]
        print(f"  {i+1}. Item {m['item']} | {res['documents'][0][i][:150]!r}")