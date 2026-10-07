import json
from pathlib import Path
from chunk import build_chunks
from check_questions import check_questions
from embed import run_embedding
from store import CHROMADB_PATH, get_collection
from retrieve import make_vector_retriever, make_bm25_retriever
from eval import evaluate

ROOT = Path(__file__).resolve().parent.parent / "data"
CHUNK_DIR = ROOT / "chunked"
MANUAL_PATH = ROOT / "processed" / "filings.json"
QUESTIONS_PATH = ROOT / "eval" / "questions.json"

OVERLAP_SIZE = 75

CONFIG = [
    ("bge_small_t250", "BAAI/bge-small-en-v1.5", 250),
    ("bge_small_t350", "BAAI/bge-small-en-v1.5", 350),      # baseline
    ("gte_small_t250", "thenlper/gte-small", 250),
    ("gte_small_t350", "thenlper/gte-small", 350),
    # ("all_miniLM_t90", "sentence-transformers/all-MiniLM-L12-v2", 90),
    # ("all_miniLM_t150", "sentence-transformers/all-MiniLM-L12-v2", 150),
    # ("mdbr_leaf_ir_t250", "MongoDB/mdbr-leaf-ir", 250),
    # ("mdbr_leaf_ir_t350", "MongoDB/mdbr-leaf-ir", 350),
    # ("mdbr_leaf_mt_t250", "MongoDB/mdbr-leaf-mt", 250),
    # ("mdbr_leaf_mt_t350", "MongoDB/mdbr-leaf-mt", 350),
]

COLS = ["config", "n_chunks", "vector_r5", "vector_mrr5", "vector_r30",
            "bm25_r5", "bm25_mrr5", "bm25_r30"]

def chunk_path(size):
     return CHUNK_DIR / f"t{size}.json"

def print_row(row):
      print(" | ".join(f"{str(row[c]):<18}" for c in COLS), flush=True)

def main():
    results = []

    with open(MANUAL_PATH, "r", encoding="utf-8") as f:
            filings = json.load(f)
    with open(QUESTIONS_PATH, "r", encoding="utf-8") as f:
                    questions = json.load(f)

    print(" | ".join(f"{c:<18}" for c in COLS))
    for name, model, target_size in CONFIG:
        path = chunk_path(target_size)
        if path.exists():
             with open(path, "r", encoding="utf-8") as f:
                  chunks = json.load(f)
        else:
            chunks = build_chunks(filings, target_size, OVERLAP_SIZE)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                 json.dump(chunks, f, indent=2)

        problems = check_questions(questions, chunks, verbose=False)
        print(name, "key problems:", problems or "none")

        collection = get_collection(model, name)
        if collection.count() != len(chunks):
            run_embedding(model, name, path, CHROMADB_PATH)
        assert collection.count() == len(chunks)
        
        retrievers = {
            "vector": make_vector_retriever(collection),
            "bm25": make_bm25_retriever(chunks) 
        }

        row = {"config": name, "n_chunks": len(chunks)}
        for method, r in retrievers.items():
             for k in (5, 30):
                recall, mrr, ranks = evaluate(r, questions, k=k, use_filter=True)
                row[f"{method}_r{k}"] = f"{sum(x is not None for x in ranks)}/{len(questions)}"
                row[f"{method}_mrr{k}"] = f"{mrr:.2f}"
        results.append(row)
        print_row(row)

    print("\nFinal table")
    print(" | ".join(f"{c:<18}" for c in COLS))
    for row in results:
        print_row(row)


if __name__ == "__main__":
     main()