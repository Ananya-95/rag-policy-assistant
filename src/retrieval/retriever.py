"""
Query-time retrieval: loads the saved FAISS index and returns top-k chunks for a question.

Implements the "R" in RAG; combine with :class:`src.llm.groq_client.GroqClient` for generation.
"""
from src.embedding.embedder import Embedder
from src.vectorstore.faiss_store import FAISSStore


class Retriever:
    """Thin wrapper around a LangChain retriever backed by on-disk FAISS."""

    def __init__(self):
        # Must use the same embedding model that built the index (same ``EMBEDDING_MODEL``).
        embedder = Embedder().get_model()
        self.store = FAISSStore(embedding_model=embedder)

    def get_retriever(self):
        """Return the LangChain retriever (loads FAISS from disk if the in-memory index is empty)."""
        return self.store.get_retriever()

    def retrieve(self, query: str):
        """Run similarity search for ``query`` and return retrieved documents (top-k per settings)."""
        return self.get_retriever().invoke(query)
