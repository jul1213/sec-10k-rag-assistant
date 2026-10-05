import re
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

def make_vector_retriever(collection):
    def retrieve(question, k, where):
        return collection.query(
            query_texts=[question], 
            n_results=k, 
            where=where
        )
    return retrieve

# used to throw away chunks whose metadata doesn't fit the filter
def matches(meta, where):
    if where is None:
        return True
    if "$and" in where:
        return all(matches(meta, c) for c in where["$and"])
    return all(meta.get(key) == value for key, value in where.items())

def tokenise(text):
    return re.findall(r"\w+", text.lower())

def make_bm25_retriever(chunks):
    texts = [chunk["text"] for chunk in chunks]
    metadatas = [
        {
            "ticker": chunk["ticker"],
            "fiscal_year": chunk["fiscal_year"],
            "item": chunk["item"],
            "section_name": chunk["section_name"],
        }
        for chunk in chunks
    ]

    bm25 = BM25Okapi([tokenise(text) for text in texts])

    def retrieve(question, k, where):
        scores = bm25.get_scores(tokenise(question))

        candidates = [
            (scores[i], i)
            for i in range(len(chunks))
            if matches(metadatas[i], where)
        ]
        candidates.sort(reverse=True)
        top=candidates[:k]

        return {
            "ids": [[chunks[i]["chunk_id"] for _, i in top]],
            "documents": [[texts[i] for _, i in top]],
            "metadatas":[[metadatas[i] for _, i in top]],
        }

    return retrieve


def make_hybrid_retriever(vector, bm25, pool=30, rrf_k=60, weights=(1.0, 1.0)):
    def retrieve(question, k, where):
        scores, items = {}, {}
        for r, w in zip((vector, bm25), weights):
            res = r(question, pool, where)
            for rank, (id_, doc, meta) in enumerate(
                zip(res["ids"][0], res["documents"][0], res["metadatas"][0]),
                start=1,
            ):
                scores[id_] = scores.get(id_, 0) + w / (rrf_k + rank)
                items[id_] = (doc, meta)
        
        top = sorted(scores, key=scores.get, reverse=True)[:k]
        return {
            "ids": [top],
            "documents": [[items[i][0] for i in top]],
            "metadatas": [[items[i][1] for i in top]]
        }
    return retrieve

def make_reranker_retriever(retriever, encoder_model, pool=30):
    model = CrossEncoder(encoder_model)

    def retrieve(question, k, where):
        res = retriever(question, pool, where)
        ids = res["ids"][0]
        docs = res["documents"][0]
        metas = res["metadatas"][0]
        
        scores = model.predict([[question, d] for d in docs])

        top = sorted(range(len(docs)), key=lambda i: scores[i], reverse=True)[:k]
        return {
            "ids": [[ids[i] for i in top]],
            "documents": [[docs[i] for i in top]],
            "metadatas": [[metas[i] for i in top]]
        }
    return retrieve