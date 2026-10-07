import json
import os

from pathlib import Path
from dotenv import load_dotenv
from google import genai

from embed import run_embedding
from answer import answer_question
from retrieve import (
    make_vector_retriever,
    make_bm25_retriever,
    make_hybrid_retriever,
    make_reranker_retriever,
)
from store import CHROMADB_PATH, COLLECTION_NAME, MODEL_NAME, get_collection

load_dotenv()

CHUNKS_PATH = Path(__file__).resolve().parent.parent / "data" / "chunked" / "chunked.json"
RERANKER_MODEL = "BAAI/bge-reranker-base"

def ensure_index():
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    collection = get_collection()
    if collection.count() != len(chunks):
        run_embedding(MODEL_NAME, COLLECTION_NAME, CHUNKS_PATH, CHROMADB_PATH)
    return chunks, collection

def load_resources():
    chunks, collection = ensure_index()
    vector = make_vector_retriever(collection)
    bm25 = make_bm25_retriever(chunks)

    return {
        "chunks": chunks,
        "retrievers": {
            "vector": vector,
            "bm25": bm25,
            "hybrid": make_hybrid_retriever(vector, bm25),
            "bm25+rerank": make_reranker_retriever(bm25, RERANKER_MODEL)
        },
        "client": genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    }

def years_by_ticker(chunks):
    out = {}
    for c in chunks:
        out.setdefault(c["ticker"], set()).add(c["fiscal_year"])
    return {t: sorted(ys) for t, ys in sorted(out.items())} 

def ask(question, resources, method="bm25", ticker=None, fiscal_year=None, k=5):
    answer, results = answer_question(
        question,
        resources["retrievers"][method],
        resources["client"],
        k=k,
        ticker=ticker,
        fiscal_year=fiscal_year,
    )
    chunks = [
        {
            "id": id_,
            "label": f"{m['ticker']} FY{m['fiscal_year']}, {m['section_name']}",
            "text": doc,
        }
        for id_, doc, m in zip(
            results["ids"][0], results["documents"][0], results["metadatas"][0]
        )
    ]
    return {"answer": answer, "chunks": chunks}

if __name__ == "__main__":
    res = load_resources()
    out = ask("What are Apple's net sales by product category?", res, ticker="AAPL", fiscal_year="2025")
    print(out["answer"])
    print([c["label"] for c in out["chunks"]])