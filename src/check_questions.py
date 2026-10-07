import json
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parent.parent / "data"

def norm(s):
    return " ".join(s.lower().split())

def check_questions(questions, chunks, verbose=True):
    normed = [(c["ticker"], str(c["fiscal_year"]), norm(c["text"])) for c in chunks]
    problems = []

    for q in questions:
        ev = norm(q["evidence"])
        found = Counter((t, y) for t, y, text in normed if ev in text)
        n = found.get((q["ticker"], str(q["fiscal_year"])), 0)
        status = "ZERO" if n == 0 else "HIGH" if n > 5 else "ok"
        if status != "ok":
            problems.append((q["question"][:60], n, status))
        if verbose:
            print(f'{status:4} {q["ticker"]} {q["fiscal_year"]} n={n} all={dict(found)}')
    return problems

def check_containment(chunks):
    n = sum(1 for a, b in zip(chunks, chunks[1:])
        if (a["ticker"], a["fiscal_year"], a["item"]) == (b["ticker"], b["fiscal_year"], b["item"])
        and b["text"] in a["text"])

    return n


def main():
    with open(ROOT / "chunked" / "chunked.json", "r", encoding="utf-8") as f:
        chunks = json.load(f)

    with open(ROOT / "eval" / "questions.json", "r", encoding="utf-8") as f:
        questions = json.load(f)

    print(check_questions(questions, chunks))


if __name__ == "__main__":
    main()