"""
Local embedding model wrapper (Hugging Face) used for chunking, FAISS indexing, and query vectors.

Keeps one :class:`~langchain_huggingface.HuggingFaceEmbeddings` instance so all stages share
the same model weights and dimensions.
"""
from langchain_huggingface import HuggingFaceEmbeddings

from config.settings import settings


class Embedder:
    """Loads ``settings.EMBEDDING_MODEL`` and exposes LangChain ``Embeddings`` APIs."""

    def __init__(self, model_name: str = settings.EMBEDDING_MODEL):
        self.model_name = model_name
        self.embedding_model = HuggingFaceEmbeddings(model_name=model_name)

    def get_model(self) -> HuggingFaceEmbeddings:
        """Return the shared embeddings object for FAISS, SemanticChunker, etc."""
        return self.embedding_model

    def embed_documents(self, documents: list):
        """Embed a batch of document strings (used by vector stores and tooling)."""
        return self.embedding_model.embed_documents(documents)

    def embed_query(self, query: str):
        """Embed a single user query string for similarity search."""
        return self.embedding_model.embed_query(query)
