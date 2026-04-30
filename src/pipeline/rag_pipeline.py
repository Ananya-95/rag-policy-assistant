from __future__ import annotations
import logging
from collections import deque
from typing import List, Optional, Union, Any

from langchain.schema import Document
from src.ingestion.ingest import PDFIngester
from src.embedding.embedder import Embedder
from src.vectorstore.faiss_store import FAISSStore
from src.retrieval.retriever import Retriever
from src.retrieval.hybrid import HybridRetriever
from src.llm.groq_client import GroqClient
from config.settings import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ConversationBufferWindowMemory  (last k turns = 2*k messages)
# ---------------------------------------------------------------------------

class ConversationBufferWindowMemory:
    """Keeps the last *k* human/AI turn-pairs in a fixed-size deque."""

    def __init__(self, k: int = 5) -> None:
        self.k = k
        # Each element: {"role": "user"|"assistant", "content": str}
        self._buffer: deque = deque(maxlen=k * 2)

    # -- public interface ---------------------------------------------------

    def save_context(self, human: str, ai: str) -> None:
        self._buffer.append({"role": "user", "content": human})
        self._buffer.append({"role": "assistant", "content": ai})

    def load_memory(self) -> List[dict]:
        """Return a plain list copy (chronological order)."""
        return list(self._buffer)

    def clear(self) -> None:
        self._buffer.clear()

    def as_text(self) -> str:
        """Compact text representation for prompt injection."""
        lines = []
        for msg in self._buffer:
            role = "Human" if msg["role"] == "user" else "Assistant"
            lines.append(f"{role}: {msg['content']}")
        return "\n".join(lines)

    def __len__(self) -> int:
        return len(self._buffer)


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

        # ConversationBufferWindowMemory – last `memory_k` turns
        self.memory = ConversationBufferWindowMemory(k=memory_k)

        self._retriever: Optional[Union[Retriever, HybridRetriever]] = None
        self._try_load_retriever()

    # ------------------------------------------------------------------
    # Index management
    # ------------------------------------------------------------------

    def build_index(self, pdf_dir: str = "data/Docs") -> None:
        chunks = self.ingester.ingest(pdf_dir)
        self.faiss_store.build(chunks)
        if self.use_hybrid:
            self._retriever = HybridRetriever(chunks)
        else:
            self._retriever = Retriever()
        logger.info("Index built (%d chunks)", len(chunks))

    def _try_load_retriever(self) -> None:
        import os
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

    def _load_chunks_from_json(self):
        import json, os
        from langchain.schema import Document as LCDoc
        path = "data/chunks.json"
        if not os.path.exists(path):
            return []
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
        return [
            LCDoc(page_content=c.get("page_content", ""), metadata=c.get("metadata", {}))
            for c in raw
        ]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _format_docs(self, docs) -> str:
        if docs is None:
            return ""
        items = docs if isinstance(docs, list) else [docs]
        parts = []
        for i, doc in enumerate(items, start=1):
            text = getattr(doc, "page_content", None) or str(doc)
            parts.append(f"[{i}] {text}")
        return "\n\n".join(parts)

    # ------------------------------------------------------------------
    # Step 1 – LLM-based standalone query rewriter
    # ------------------------------------------------------------------

    def _rewrite_query(self, question: str) -> str:
        """Rewrite a follow-up question into a standalone query using chat history."""
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
        """Ask the LLM for 3 paraphrases of the query."""
        prompt = (
            "Generate 3 different paraphrases of the following search query. "
            "Each paraphrase should capture the same information need but use different wording. "
            "Output ONLY the 3 paraphrases, one per line, with no numbering or extra text.\n\n"
            f"Query: {question}"
        )
        try:
            raw = self.llm_client.invoke(prompt).strip()
            paraphrases = [p.strip() for p in raw.splitlines() if p.strip()][:3]
            logger.info("Generated %d paraphrases", len(paraphrases))
            return paraphrases
        except Exception as exc:
            logger.warning("Paraphrase generation failed: %s", exc)
            return []

    def _multi_query_retrieve(self, question: str, k: int = 4) -> List[Document]:
        """Retrieve for original + paraphrases, deduplicate, rerank by frequency."""
        queries = [question] + self._generate_paraphrases(question)
        seen_content: dict = {}   # page_content -> (doc, freq)

        for q in queries:
            try:
                if self.use_hybrid and isinstance(self._retriever, HybridRetriever):
                    docs = self._retriever.retrieve(q, k=k)
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

        # Rerank: sort by frequency descending, take top k*2 unique docs
        reranked = sorted(seen_content.values(), key=lambda x: x[1], reverse=True)
        top_docs = [doc for doc, _ in reranked[: k * 2]]
        logger.info(
            "Multi-query: %d queries -> %d unique docs (top %d kept)",
            len(queries),
            len(seen_content),
            len(top_docs),
        )
        return top_docs

    # ------------------------------------------------------------------
    # Main answer method
    # ------------------------------------------------------------------

    def answer(self, question: str) -> str:
        # 1. Ensure retriever is available
        if self._retriever is None:
            return "No index loaded. Run `python main.py index` first."

        # 2. Rewrite query using conversation history
        if self.use_query_rewrite:
            standalone = self._rewrite_query(question)
        else:
            standalone = question

        # 3. Multi-query retrieval with dedup + rerank
        docs = self._multi_query_retrieve(standalone, k=4)
        context = self._format_docs(docs)

        # 4. Build generation prompt with windowed memory
        history_text = self.memory.as_text()
        history_section = (
            f"\nConversation history:\n{history_text}\n" if history_text else ""
        )
        prompt = (
            "You are a helpful policy assistant. Answer the question based ONLY on the "
            "provided context. If the answer is not in the context, say you do not have "
            "enough information.\n"
            f"{history_section}"
            f"\nContext:\n{context}\n\n"
            f"Question: {question}\n\n"
            "Answer:"
        )

        # 5. Generate
        reply = self.llm_client.invoke(prompt)

        # 6. Store in window memory
        self.memory.save_context(question, reply)

        return reply

    # ------------------------------------------------------------------
    # Public helpers for Streamlit
    # ------------------------------------------------------------------

    def get_history(self) -> List[dict]:
        """Return current window memory as a list of message dicts."""
        return self.memory.load_memory()

    def clear_history(self) -> None:
        """Reset the conversation window."""
        self.memory.clear()
