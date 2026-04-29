from __future__ import annotations
import os, logging
from collections import defaultdict
from typing import List
from langchain.schema import Document
from src.retrieval.bm25_retreiver import BM25Retriever
from src.retrieval.retriever import Retriever
logger = logging.getLogger(__name__)

def reciprocal_rank_fusion(ranked_lists: List[List[Document]], k: int = 60) -> List[Document]:
    scores: dict = defaultdict(float)
    doc_map: dict = {}
    for ranked in ranked_lists:
        for rank, doc in enumerate(ranked, start=1):
            key = doc.page_content
            scores[key] += 1.0 / (k + rank)
            if key not in doc_map:
                doc_map[key] = doc
    sorted_keys = sorted(scores, key=lambda x: scores[x], reverse=True)
    return [doc_map[k] for k in sorted_keys]

class HybridRetriever:
    def __init__(self, chunks: List[Document], bm25_top_k: int = 10, dense_top_k: int = 10,
                 use_cohere_rerank: bool = True, rerank_top_n: int = 5) -> None:
        self.bm25_top_k = bm25_top_k
        self.dense_top_k = dense_top_k
        self.rerank_top_n = rerank_top_n
        logger.info("Building BM25 index over %d chunks", len(chunks))
        self.bm25 = BM25Retriever(chunks)
        logger.info("Loading FAISS index")
        self.dense = Retriever()
        self._cohere_client = None
        if use_cohere_rerank:
            api_key = os.getenv("COHERE_API_KEY", "")
            if api_key:
                try:
                    import cohere
                    self._cohere_client = cohere.Client(api_key)
                    logger.info("Cohere reranker enabled")
                except ImportError:
                    logger.warning("cohere package not installed. pip install cohere")
            else:
                logger.info("COHERE_API_KEY not set - reranking disabled")

    def retrieve(self, query: str, top_k: int = 5) -> List[Document]:
        bm25_results = self.bm25.retrieve(query, top_k=self.bm25_top_k)
        dense_results = self.dense.retrieve(query)
        fused = reciprocal_rank_fusion([bm25_results, dense_results])
        candidates = fused[:max(top_k * 2, 10)]
        if self._cohere_client and candidates:
            candidates = self._cohere_rerank(query, candidates, top_n=top_k)
        else:
            candidates = candidates[:top_k]
        return candidates

    def _cohere_rerank(self, query: str, docs: List[Document], top_n: int) -> List[Document]:
        try:
            texts = [d.page_content for d in docs]
            response = self._cohere_client.rerank(query=query, documents=texts,
                model="rerank-english-v3.0", top_n=top_n)
            return [docs[r.index] for r in response.results]
        except Exception as exc:
            logger.warning("Cohere rerank failed (%s) - using RRF order", exc)
            return docs[:top_n]