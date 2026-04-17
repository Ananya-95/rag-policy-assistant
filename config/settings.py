"""
Central configuration for paths, embedding model, chunking, retrieval, and Groq.

Loaded once at import; ``load_dotenv()`` pulls ``.env`` so ``GROQ_API_KEY`` is available locally.
"""
import os

from dotenv import load_dotenv

load_dotenv()


class Settings:
    """Default paths and hyperparameters for the policy RAG stack."""

    # PDFs to index (place files here before running ``RAGPipeline.build_index``).
    DATA_PATH = "data/Docs"
    # Where FAISS persists its index (must match between index build and query).
    FAISS_INDEX_PATH = "data/Faiss_Index"

    # Hugging Face sentence-transformer id; must stay consistent across index and query.
    EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

    # Second-stage splitter after SemanticChunker (character-based windows).
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 100

    # Number of chunks to return per query (FAISS retriever ``k``).
    TOP_K = 5

    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    GROQ_MODEL = "llama-3.1-70b-versatile"


settings = Settings()
