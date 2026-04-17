"""
PDF ingestion and chunking for the policy corpus.

Flow: ``ingest`` loads raw pages → ``chunk`` applies semantic then character splitting →
``save_chunks`` writes a JSON sidecar for inspection (the FAISS index is built elsewhere).
"""
import json
import os
from typing import Optional

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.embeddings import Embeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings

from config.settings import settings


class PDFIngester:
    """Loads ``.pdf`` files from a directory and produces LangChain ``Document`` chunks."""

    def __init__(self, data_path: str = settings.DATA_PATH):
        self.data_path = data_path

    def ingest(self):
        """
        Read every ``*.pdf`` under ``data_path`` and return concatenated LangChain documents.
        """
        documents = []
        for file in os.listdir(self.data_path):
            if file.endswith(".pdf"):
                file_path = os.path.join(self.data_path, file)
                loader = PyPDFLoader(file_path)
                documents.extend(loader.load())
        return documents

    def chunk(
        self,
        documents: list,
        embeddings: Optional[Embeddings] = None,
    ):
        """
        Split documents for retrieval: semantic boundaries first, then fixed-size windows.

        Pass the same ``Embeddings`` as used for FAISS so chunk boundaries align with search.
        If omitted, a new ``HuggingFaceEmbeddings`` is created (extra model load).
        """
        if embeddings is None:
            embeddings = HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL)

        semantic_text_splitter = SemanticChunker(embeddings)
        semantic_chunks = semantic_text_splitter.split_documents(documents)

        final_chunks = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            separators=["\n\n", "\n", " ", ""],
        ).split_documents(semantic_chunks)

        return final_chunks

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
