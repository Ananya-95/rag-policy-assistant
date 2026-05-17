"""
PDF ingestion and chunking for the policy corpus.

Flow: ``ingest`` loads raw pages → ``chunk`` applies semantic then character splitting →
``save_chunks`` writes a JSON sidecar for inspection (the FAISS index is built elsewhere).
"""
import json
import os
from typing import Optional

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.settings import settings


class PDFIngester:
    """Loads ``.pdf`` files from a directory and produces LangChain ``Document`` chunks."""

    def __init__(self, data_path: str = settings.DATA_PATH):
        self.data_path = data_path

    def ingest(self, data_path: str | None = None) -> list:
        """
        Read every ``*.pdf`` under ``data_path`` (or ``self.data_path``) and return documents.
        """
        root = data_path or self.data_path
        if not os.path.isdir(root):
            raise FileNotFoundError(f"PDF directory not found: {root}")

        documents = []
        for file in sorted(os.listdir(root)):
            if file.endswith(".pdf"):
                file_path = os.path.join(root, file)
                loader = PyMuPDFLoader(file_path)
                documents.extend(loader.load())
        return documents

    def chunk(
        self,
        documents: list,
        embeddings: Optional[Embeddings] = None,
    ):
        return RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            separators=["\n\n", "\n", " ", ""],
        ).split_documents(documents)

    def save_chunks(self, chunks: list, path: str = "data/chunks.json") -> None:
        """
        Persist chunk text and metadata as JSON for debugging (not required for FAISS search).
        """
        out_dir = os.path.dirname(path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        serializable = [
            {"page_content": doc.page_content, "metadata": dict(doc.metadata)}
            for doc in chunks
        ]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2, ensure_ascii=False, default=str)
