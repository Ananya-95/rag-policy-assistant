"""Unit tests for retrieval helpers (no Groq API required)."""
from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from langchain.schema import Document

from src.retrieval.bm25_retreiver import BM25Retriever
from src.retrieval.hybrid import reciprocal_rank_fusion


class TestBM25Retriever(unittest.TestCase):
    def setUp(self) -> None:
        self.docs = [
            Document(page_content="RAASB was established in 2024", metadata={"source": "a.pdf"}),
            Document(page_content="NISM Series XV certification required", metadata={"source": "b.pdf"}),
            Document(page_content="unrelated leave policy text", metadata={"source": "c.pdf"}),
        ]
        self.retriever = BM25Retriever(self.docs)

    def test_retrieve_returns_top_k(self) -> None:
        results = self.retriever.retrieve("RAASB establishment", top_k=2)
        self.assertEqual(len(results), 2)
        self.assertIn("RAASB", results[0].page_content)

    def test_retrieve_ranks_keyword_match(self) -> None:
        results = self.retriever.retrieve("NISM certification", top_k=1)
        self.assertIn("NISM", results[0].page_content)


class TestReciprocalRankFusion(unittest.TestCase):
    def test_fusion_prefers_docs_in_both_lists(self) -> None:
        d1 = Document(page_content="doc one")
        d2 = Document(page_content="doc two")
        d3 = Document(page_content="doc three")
        fused = reciprocal_rank_fusion([[d1, d2], [d2, d3]], k=60)
        self.assertEqual(fused[0].page_content, "doc two")

    def test_fusion_empty_lists(self) -> None:
        self.assertEqual(reciprocal_rank_fusion([]), [])


class TestRetrieverLoad(unittest.TestCase):
    @patch("src.retrieval.retriever.FAISSStore")
    @patch("src.retrieval.retriever.Embedder")
    def test_retriever_invoke(self, mock_embedder: MagicMock, mock_store_cls: MagicMock) -> None:
        mock_embedder.return_value.get_model.return_value = MagicMock()
        mock_store = mock_store_cls.return_value
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = [Document(page_content="test")]
        mock_store.get_retriever.return_value = mock_retriever

        from src.retrieval.retriever import Retriever

        r = Retriever()
        out = r.retrieve("query")
        self.assertEqual(len(out), 1)
        mock_retriever.invoke.assert_called_once_with("query")


if __name__ == "__main__":
    unittest.main()
