from dotenv import load_dotenv

load_dotenv()

PROMPT_TEMPLATE = """
Answer the question using ONLY the excerpts below.
Cite each claim like [TICKER FYYEAR, Section].
If the excerpts do not contain the answer, say so explicitly instead of guessing.

{excerpts}

Question: {question}
"""

def format_excerpts(results):
    lines = []
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    for i, (doc, meta) in enumerate(zip(docs, metas), start=1):
        label = f"{meta['ticker']} FY{meta['fiscal_year']}, {meta['section_name']}"
        lines.append(f"Excerpt {i} [{label}]:\n{doc}")
    return "\n\n".join(lines)

def answer_question(question, retrieve, client, k=5, ticker=None, fiscal_year=None):
    conditions = []
    if ticker:
        conditions.append({"ticker": ticker})
    if fiscal_year:
        conditions.append({"fiscal_year": fiscal_year})
    where = None
    if len(conditions) == 1:
        where = conditions[0]
    elif len(conditions) > 1:
        where = {"$and": conditions}
    
    results = retrieve(question, k, where)
    prompt = PROMPT_TEMPLATE.format(
        excerpts=format_excerpts(results),
        question=question
    )
    interaction = client.interactions.create(
        model="gemini-3.5-flash-lite",
        input=prompt
    )
    return interaction.output_text, results