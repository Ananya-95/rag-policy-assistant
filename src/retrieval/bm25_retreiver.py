from rank_bm25 import BM25Okapi

from rank_bm25 import BM25Okapi

class BM25Retriever:
    def __init__(self, docs):
        self.docs = docs
        self.tokenized_corpus = [
            doc.page_content.lower().split()
            for doc in docs
        ]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def retrieve(self, query, top_k=10):
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)

        ranked = sorted(
            zip(self.docs, scores),
            key=lambda x: x[1],
            reverse=True
        )

        return [doc for doc, score in ranked[:top_k]]