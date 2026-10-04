def make_vector_retriever(collection):
    def retrieve(question, k, where):
        return collection.query(
            query_texts=[question], 
            n_results=k, 
            where=where
        )
    return retrieve