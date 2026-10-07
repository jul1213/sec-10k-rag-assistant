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
    
    BATCH = 100
    for i in range(0, len(ids), BATCH):
        collection.upsert(
            ids=ids[i:i+BATCH],
            documents=documents[i:i+BATCH],
            metadatas=metadatas[i:i+BATCH]
        )
        print(f"{min(i + BATCH, len(ids))}/{len(ids)}", flush=True)
    return collection

def run_embedding(model_name, collection_name, chunks_path, db_path):
    collection = get_collection(model_name, collection_name, db_path)

    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    return embed_and_save(collection, chunks)

def main():
    run_embedding(MODEL_NAME, COLLECTION_NAME, CHUNKS_PATH, CHROMADB_PATH)



if __name__ == "__main__":
    main()