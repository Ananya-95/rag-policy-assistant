"""
FAISS vector store: build an index from chunked documents, persist it, and expose a retriever.

The index directory is ``settings.FAISS_INDEX_PATH``; ``load_index`` uses pickle-based load
(``allow_dangerous_deserialization``) — only load indexes you created yourself.
"""
import os

from langchain_community.vectorstores import FAISS

from config.settings import settings


class FAISSStore:
    """Wraps LangChain ``FAISS`` for create / save / load / ``as_retriever``."""

    def __init__(self, embedding_model):
        self.embedding_model = embedding_model
        self.index = None

    def create_index(self, chunks: list) -> FAISS:
        """Embed ``chunks`` and build an in-memory FAISS index (does not write to disk yet)."""
        self.index = FAISS.from_documents(chunks, self.embedding_model)
        return self.index

    def save_index(self) -> None:
        """Serialize the current index under ``settings.FAISS_INDEX_PATH``."""
        if self.index is None:
            raise ValueError("No index to save. Run create_index() first.")
        os.makedirs(settings.FAISS_INDEX_PATH, exist_ok=True)
        self.index.save_local(settings.FAISS_INDEX_PATH)

    def load_index(self) -> FAISS:
        """Load a previously saved index from disk into ``self.index``."""
        self.index = FAISS.load_local(
            settings.FAISS_INDEX_PATH,
            self.embedding_model,
            allow_dangerous_deserialization=True,
        )
        return self.index

    def get_retriever(self):
        """
        LangChain retriever for similarity search; loads from disk if ``create_index`` was not run this session.
        """
        if self.index is None:
            self.load_index()
        return self.index.as_retriever(
            search_kwargs={"k": settings.TOP_K},
        )
