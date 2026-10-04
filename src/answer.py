import os
from google import genai
from dotenv import load_dotenv
from store import CHROMADB_PATH, MODEL_NAME, COLLECTION_NAME, get_collection
from retrieve import make_vector_retriever

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

def main():
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    collection = get_collection(MODEL_NAME, COLLECTION_NAME, CHROMADB_PATH)
    vector_retriever = make_vector_retriever(collection)
    
    question = "What are the main sources of revenue?"
    answer, results = answer_question(question, vector_retriever, client)
    print(answer)
    print([(m["ticker"], m["item"]) for m in results["metadatas"][0]])

    question_aapl = "What are the main sources of revenue?"
    answer_aapl, results_aapl = answer_question(question_aapl, collection, client, ticker="AAPL")
    print(answer_aapl)
    print([(m["ticker"], m["item"]) for m in results_aapl["metadatas"][0]])

    question_nvda = "What are the main sources of revenue?"
    answer_nvda, results_nvda = answer_question(question_nvda, collection, client, ticker="NVDA")
    print(answer_nvda)
    print([(m["ticker"], m["item"]) for m in results_nvda["metadatas"][0]])



if __name__ == "__main__":
    main()