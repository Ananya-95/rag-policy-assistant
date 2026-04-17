"""
Policy RAG orchestration: ingestion, embeddings, FAISS, retrieval, and Groq generation.

Typical flow
------------
1. Place PDFs under ``data/Docs`` (see ``config.settings``).
2. Run indexing (:meth:`RAGPipeline.build_index` from ``main.py`` or your own script).
3. Ask questions via :meth:`RAGPipeline.answer` (e.g. from ``main.py``).
"""
import logging
from typing import Any, List, Union

from src.embedding.embedder import Embedder
from src.ingestion.ingest import PDFIngester
from src.llm.groq_client import GroqClient
from src.retrieval.retriever import Retriever
from src.vectorstore.faiss_store import FAISSStore

logger = logging.getLogger(__name__)


class RAGPipeline:
    """Composes all services needed for offline indexing and online Q&A."""

    def __init__(self):
        # Shared embedding model for indexing; same model id must be used when loading the index for search.
        self.embedder = Embedder()
        self.ingester = PDFIngester()
        self.faiss_store = FAISSStore(embedding_model=self.embedder.get_model())
        # Retriever uses its own Embedder + FAISSStore but loads the same on-disk index path.
        self.retriever = Retriever()
        self.llm_client = GroqClient()

    def build_index(self) -> None:
        """
        Offline path: load PDFs, split into chunks, optionally dump JSON for debugging,
        build the FAISS vector index, and save it under ``settings.FAISS_INDEX_PATH``.
        """
        documents = self.ingester.ingest()
        chunks = self.ingester.chunk(
            documents, embeddings=self.embedder.get_model()
        )
        self.ingester.save_chunks(chunks)

        self.faiss_store.create_index(chunks)
        self.faiss_store.save_index()
        logger.info("Index saved.")

    @staticmethod
    def _documents_to_context(docs: Union[List[Any], Any]) -> str:
        """Turn retriever output (one or many LangChain Documents) into a single context block."""
        if docs is None:
            return ""
        items = docs if isinstance(docs, list) else [docs]
        parts = []
        for i, doc in enumerate(items, start=1):
            text = getattr(doc, "page_content", None) or str(doc)
            parts.append(f"[{i}] {text}")
        return "\n\n".join(parts)

    def answer(self, question: str) -> str:
        """
        Retrieve top-k chunks for ``question``, then ask Groq to answer using only that context.

        Requires a built index on disk and ``GROQ_API_KEY`` in the environment.
        """
        retrieved = self.retriever.retrieve(question)
        context = self._documents_to_context(retrieved)
        if not context.strip():
            return (
                "No relevant passages were retrieved. "
                "Build the index first (`main.py index`) and ensure PDFs exist under data/Docs."
            )
        prompt = (
            "You are a policy assistant. Answer using only the context below. "
            "If the answer is not in the context, say that you do not have enough information.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n\n"
            "Answer:"
        )
        return self.llm_client.invoke(prompt)
