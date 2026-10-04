import chromadb
from pathlib import Path
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

CHROMADB_PATH = Path(__file__).resolve().parent.parent / "data" / "chroma_db"
COLLECTION_NAME = "bge_small_target350"
MODEL_NAME = "BAAI/bge-small-en-v1.5"   

def get_collection(model_name=MODEL_NAME, collection_name=COLLECTION_NAME, db_path=CHROMADB_PATH):
    ef = SentenceTransformerEmbeddingFunction(model_name=model_name)
    client = chromadb.PersistentClient(path=str(db_path))
    return client.get_or_create_collection(name=collection_name, embedding_function=ef)