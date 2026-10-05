import json
from pathlib import Path
from store import get_collection
from retrieve import make_vector_retriever, make_bm25_retriever, make_hybrid_retriever, make_reranker_retriever

CHUNKS_PATH = Path(__file__).resolve().parent.parent / "data" / "chunked" / "chunked.json"
QUESTIONS_PATH = Path(__file__).resolve().parent.parent / "data" / "eval" / "questions.json"

ENCODER_MODEL = "BAAI/bge-reranker-base"

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
    with open(QUESTIONS_PATH, "r", encoding="utf-8") as f:
        questions = json.load(f)

    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    vector_retriever = make_vector_retriever(get_collection())
    bm25_retriever = make_bm25_retriever(chunks)
    retrievers = {
        "vector" : vector_retriever,
        "bm25" : bm25_retriever,
        "hybrid_w1-1": make_hybrid_retriever(vector_retriever, bm25_retriever, weights=(1, 1)),
        "hybrid_w1-2": make_hybrid_retriever(vector_retriever, bm25_retriever, weights=(1, 2)),
        "bm25+rerank": make_reranker_retriever(bm25_retriever, ENCODER_MODEL),
    }

    ks_for = {
        "bm25+rerank": (5, )
    }

    print(f"{'method':<12} {'filter':<7} {'k':<3} {'recall':<10} {'MRR':<5}")
    all_ranks = {}
    for name, retriever in retrievers.items():
        for k in ks_for.get(name, (5, 10, 30)):
            for use_filter in (False, True):
                recall, mrr, ranks = evaluate(retriever, questions, k=k, use_filter=use_filter)
                hits = sum(r is not None for r in ranks)
                print(f"{name:<12} {str(use_filter):<7} {k:<3} {hits}/{len(questions)} ({recall:.2f}) {mrr:.2f}")
                all_ranks[(name, use_filter, k)] = ranks
    
    print("\nBM25 vs rerank (filtered, k=5), questions that changed:")
    b = all_ranks[("bm25", True, 5)]
    r = all_ranks[("bm25+rerank", True, 5)]
    for q, x, y in zip(questions, b, r):
        if x != y:
            print(f"  bm25={x} rerank={y}  {q['question'][:70]}")

    print("\nMisses (filtered, k=5)")
    for name in retrievers:
        ranks = all_ranks[(name, True, 5)]
        missed = [q["question"][:70] for q, r in zip(questions, ranks) if r is None]
        print(f" {name}: {len(missed)}")
        for m in missed:
            print(f"    -{m}")

if __name__ == "__main__":
    main()