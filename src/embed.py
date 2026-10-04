import json
from pathlib import Path
from store import get_collection, MODEL_NAME, COLLECTION_NAME, CHROMADB_PATH

CHUNKS_PATH = Path(__file__).resolve().parent.parent / "data" / "chunked" / "chunked.json"

def embed_and_save(collection, chunks):
    ids, documents, metadatas = [], [], []
    for chunk in chunks:
        ids.append(chunk["chunk_id"])
        documents.append(chunk["text"])

        metadata = {
            "ticker": chunk["ticker"],
            "fiscal_year": chunk["fiscal_year"],
            "item": chunk["item"],
            "section_name": chunk["section_name"],
        }
        metadatas.append(metadata)

    collection.upsert(
        ids=ids,
        documents=documents,
        metadatas=metadatas
    )
    return collection

def run_embedding(model_name, collection_name, chunks_path, db_path):
    collection = get_collection(model_name, collection_name, db_path)

    with open(CHUNKS_PATH, "r") as f:
        chunks = json.load(f)

    return embed_and_save(collection, chunks)

def main():
    run_embedding(MODEL_NAME, COLLECTION_NAME, CHUNKS_PATH, CHROMADB_PATH)

    # results = collection.query(
    #     query_texts=["What is NVDIA's main sources of revenue?"],
    #     n_results=2
    # )
    # for i, document in enumerate(results["documents"][0]):
    #     print(f"\n--- Result {i + 1} ---")
    #     print(document)
    #     print("Distance:", results["distances"][0][i])



if __name__ == "__main__":
    main()