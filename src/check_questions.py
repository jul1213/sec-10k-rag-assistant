import json
from pathlib import Path
from collections import Counter

root = Path(__file__).resolve().parent.parent / "data"
chunks = json.load(open(root / "chunked" / "chunked.json"))
questions = json.load(open(root / "eval" / "questions.json"))

def norm(s):
    return " ".join(s.lower().split())

for q in questions:
    ev = norm(q["evidence"])
    found = Counter((c["ticker"], str(c["fiscal_year"])) for c in chunks if ev in norm(c["text"]))
    print(q["ticker"], q["fiscal_year"], dict(found) or "NOT FOUND")