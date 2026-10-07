import json
from pathlib import Path
from sentence_transformers import SentenceTransformer

CHUNK_DIR = Path(__file__).resolve().parent.parent / "data" / "chunked"

CHECKS = [
    ("BAAI/bge-small-en-v1.5", 250),
    ("BAAI/bge-small-en-v1.5", 350),
    ("MongoDB/mdbr-leaf-mt", 350),
]

for model_name, size in CHECKS:
    m = SentenceTransformer(model_name)
    with open(CHUNK_DIR / f"t{size}.json", encoding="utf-8") as f:
        chunks = json.load(f)
    lens = [len(m.tokenizer(c["text"], truncation=False)["input_ids"]) for c in chunks]
    over = sum(l > m.max_seq_length for l in lens)
    print(model_name, size, "limit:", m.max_seq_length, "max:", max(lens), f"over: {over}/{len(lens)}")