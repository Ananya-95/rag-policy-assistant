from __future__ import annotations

import logging
import os
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional, Union

from langchain.schema import Document

from config.settings import settings
from src.embedding.embedder import Embedder
from src.ingestion.ingest import PDFIngester
from src.llm.groq_client import GroqClient
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.retriever import Retriever
from src.vectorstore.faiss_store import FAISSStore

logger = logging.getLogger(__name__)


@dataclass
class SourceCitation:
    filename: str
    page: Optional[int]
    snippet: str


@dataclass
class RAGResponse:
    answer: str
    sources: List[SourceCitation] = field(default_factory=list)


# ---------------------------------------------------------------------------
# ConversationBufferWindowMemory  (last k turns = 2*k messages)
# ---------------------------------------------------------------------------

class ConversationBufferWindowMemory:
    """Keeps the last *k* human/AI turn-pairs in a fixed-size deque."""

    def __init__(self, k: int = 5) -> None:
        self.k = k
        self._buffer: deque = deque(maxlen=max(k * 2, 0))

    def save_context(self, human: str, ai: str) -> None:
        if self.k <= 0:
            return
        self._buffer.append({"role": "user", "content": human})
        self._buffer.append({"role": "assistant", "content": ai})

    def load_memory(self) -> List[dict]:
        return list(self._buffer)

    def clear(self) -> None:
        self._buffer.clear()

    def as_text(self) -> str:
        lines = []
        for msg in self._buffer:
            role = "Human" if msg["role"] == "user" else "Assistant"
            lines.append(f"{role}: {msg['content']}")
        return "\n".join(lines)

    def __len__(self) -> int:
        return len(self._buffer)

    def __bool__(self) -> bool:
        return len(self._buffer) > 0


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

class RAGPipeline:
    """Orchestrates ingestion, hybrid retrieval, query rewriting, memory, and generation."""

    def __init__(
        self,
        use_hybrid: bool = True,
        use_query_rewrite: bool = True,
        memory_k: int = 5,
    ) -> None:
        self.ingester = PDFIngester()
        self.embedder = Embedder()
        self.faiss_store = FAISSStore(embedding_model=self.embedder.get_model())
        self.llm_client = GroqClient()
        self.use_hybrid = use_hybrid
        self.use_query_rewrite = use_query_rewrite
        self.memory = ConversationBufferWindowMemory(k=memory_k)
        self._retriever: Optional[Union[Retriever, HybridRetriever]] = None
        self._try_load_retriever()

    # ------------------------------------------------------------------
    # Index management
    # ------------------------------------------------------------------

    def build_index(self, pdf_dir: str | None = None) -> int:
        """Load PDFs → chunk → save FAISS index + chunks.json. Returns chunk count."""
        if pdf_dir:
            self.ingester = PDFIngester(data_path=pdf_dir)

        documents = self.ingester.ingest()
        if not documents:
            raise ValueError(
                f"No PDFs found in {self.ingester.data_path}. "
                "Add policy PDFs and run again."
            )

        chunks = self.ingester.chunk(documents, embeddings=self.embedder.get_model())
        self.ingester.save_chunks(chunks)
        self.faiss_store.build(chunks)

        if self.use_hybrid:
            self._retriever = HybridRetriever(chunks)
        else:
            self._retriever = Retriever()

        logger.info("Index built (%d chunks from %d pages)", len(chunks), len(documents))
        return len(chunks)

    def _try_load_retriever(self) -> None:
        if os.path.exists(settings.FAISS_INDEX_PATH):
            try:
                self.faiss_store.load_index()
                if self.use_hybrid:
                    chunks = self._load_chunks_from_json()
                    if chunks:
                        self._retriever = HybridRetriever(chunks)
                        logger.info("HybridRetriever loaded (%d chunks)", len(chunks))
                    else:
                        self._retriever = Retriever()
                else:
                    self._retriever = Retriever()
            except Exception as exc:
                logger.warning("Could not load existing index: %s", exc)

    def _load_chunks_from_json(self) -> List[Document]:
        import json

        path = "data/chunks.json"
        if not os.path.exists(path):
            return []
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
        return [
            Document(page_content=c.get("page_content", ""), metadata=c.get("metadata", {}))
            for c in raw
        ]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _doc_to_citation(doc: Document) -> SourceCitation:
        meta = getattr(doc, "metadata", None) or {}
        source = meta.get("source") or meta.get("file_path") or "unknown"
        filename = os.path.basename(str(source))
        page = meta.get("page")
        if page is None:
            page = meta.get("page_number")
        if page is not None:
            try:
                page = int(page) + 1  # PyMuPDF uses 0-based pages
            except (TypeError, ValueError):
                pass
        text = getattr(doc, "page_content", "") or ""
        snippet = text[:400] + ("…" if len(text) > 400 else "")
        return SourceCitation(filename=filename, page=page, snippet=snippet)

    def _format_docs(self, docs: List[Document]) -> str:
        parts = []
        for i, doc in enumerate(docs, start=1):
            cite = self._doc_to_citation(doc)
            page_str = f", p.{cite.page}" if cite.page is not None else ""
            parts.append(f"[{i}] ({cite.filename}{page_str})\n{cite.snippet}")
        return "\n\n".join(parts)

    # ------------------------------------------------------------------
    # Step 1 – LLM-based standalone query rewriter
    # ------------------------------------------------------------------

    def _rewrite_query(self, question: str) -> str:
        if not self.memory:
            return question

        history_text = self.memory.as_text()
        if not history_text.strip():
            return question

        prompt = (
            "Given the conversation history below, rewrite the follow-up question as a "
            "fully standalone question that does NOT require the history to understand.\n"
            "Return ONLY the rewritten question with no extra commentary.\n\n"
            f"History:\n{history_text}\n\n"
            f"Follow-up question: {question}\n\n"
            "Standalone question:"
        )
        try:
            rewritten = self.llm_client.invoke(prompt).strip()
            logger.info("Query rewritten: %r -> %r", question, rewritten)
            return rewritten if rewritten else question
        except Exception as exc:
            logger.warning("Query rewrite failed: %s", exc)
            return question

    # ------------------------------------------------------------------
    # Step 2 – Multi-query retrieval with deduplication & reranking
    # ------------------------------------------------------------------

    def _generate_paraphrases(self, question: str) -> List[str]:
        prompt = (
            "Generate 3 different paraphrases of the following search query. "
            "Each paraphrase should capture the same information need but use different wording. "
            "Output ONLY the 3 paraphrases, one per line, with no numbering or extra text.\n\n"
            f"Query: {question}"
        )
        try:
            raw = self.llm_client.invoke(prompt).strip()
            return [p.strip() for p in raw.splitlines() if p.strip()][:3]
        except Exception as exc:
            logger.warning("Paraphrase generation failed: %s", exc)
            return []

    def _multi_query_retrieve(self, question: str, k: int = 4) -> List[Document]:
        queries = [question] + self._generate_paraphrases(question)
        seen_content: dict = {}

        for q in queries:
            try:
                if self.use_hybrid and isinstance(self._retriever, HybridRetriever):
                    docs = self._retriever.retrieve(q, top_k=k)
                elif self._retriever is not None:
                    docs = self._retriever.retrieve(q)
                else:
                    docs = self.faiss_store.search(q, k=k)
            except Exception as exc:
                logger.warning("Retrieval failed for query %r: %s", q, exc)
                docs = []

            for doc in docs:
                key = getattr(doc, "page_content", str(doc))
                if key in seen_content:
                    seen_content[key] = (seen_content[key][0], seen_content[key][1] + 1)
                else:
                    seen_content[key] = (doc, 1)

        reranked = sorted(seen_content.values(), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in reranked[: k * 2]]

    # ------------------------------------------------------------------
    # Main answer method
    # ------------------------------------------------------------------

    def answer(self, question: str, *, store_memory: bool = True) -> RAGResponse:
        if self._retriever is None:
            return RAGResponse(
                answer="No index loaded. Run `python main.py index` first.",
                sources=[],
            )

        standalone = (
            self._rewrite_query(question)
            if self.use_query_rewrite
            else question
        )

        docs = self._multi_query_retrieve(standalone, k=4)
        sources = [self._doc_to_citation(d) for d in docs]
        context = self._format_docs(docs)

        history_text = self.memory.as_text()
        history_section = (
            f"\nConversation history:\n{history_text}\n" if history_text else ""
        )
        prompt = (
            "You are a helpful policy assistant. Answer the question based ONLY on the "
            "provided context. Cite sources inline using the format [filename, p.N] "
            "when you use a passage. If the answer is not in the context, say you do not "
            "have enough information — do not guess.\n"
            f"{history_section}"
            f"\nContext:\n{context}\n\n"
            f"Question: {question}\n\n"
            "Answer:"
        )

        reply = self.llm_client.invoke(prompt)

        if store_memory:
            self.memory.save_context(question, reply)

        return RAGResponse(answer=reply, sources=sources)

    def get_history(self) -> List[dict]:
        return self.memory.load_memory()

    def clear_history(self) -> None:
        self.memory.clear()
