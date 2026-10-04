import json
from pathlib import Path
from store import get_collection
from retrieve import make_vector_retriever

QUESTIONS_PATH = Path(__file__).resolve().parent.parent / "data" / "eval" / "questions.json"

def norm(s):
    return " ".join(s.lower().split())

def first_hit_rank(results, q):
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    for rank, (doc, meta) in enumerate(zip(docs, metas), start=1):
        if (meta["ticker"] == q["ticker"] and 
                meta["fiscal_year"] == q["fiscal_year"] and 
                norm(q["evidence"]) in norm(doc)):
            return rank

    return None


def evaluate(retrieve, questions, k=5, use_filter=False):
    ranks = []
    for q in questions:
        where = None
        if use_filter:
            where = {"$and" : [{"ticker": q["ticker"]}, {"fiscal_year": q["fiscal_year"]}]}
        results = retrieve(q["question"], k, where)
        ranks.append(first_hit_rank(results, q))
    hits = [r for r in ranks if r]
    recall = len(hits) / len(questions)
    mrr = sum(1 / r for r in hits) / len(questions)
    return recall, mrr, ranks

def main():
    questions = json.load(open(QUESTIONS_PATH))
    collection = get_collection()
    vector_retriever = make_vector_retriever(collection)
    for k in (5, 10):
        for use_filter in (False, True):
            recall, mrr, ranks = evaluate(vector_retriever, questions, k=k, use_filter=use_filter)
            print(f"filter={use_filter}: recall@{k}={recall:.2f} ({len([r for r in ranks if r])}/{len(questions)})  MRR={mrr:.2f}")
            for q, r in zip(questions, ranks):
                if r is None:
                    print("  MISS:", q["question"])

if __name__ == "__main__":
    main()