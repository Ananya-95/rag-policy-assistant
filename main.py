#!/usr/bin/env python3
"""CLI: ``python main.py index`` | ``python main.py ask "question"``"""
import logging
import sys
from pathlib import Path

import typer

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.pipeline.rag_pipeline import RAGPipeline

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
app = typer.Typer(help="Policy RAG Assistant CLI")


@app.command("index")
def build_index(
    pdf_dir: str = typer.Option("data/Docs", help="Folder containing PDF files"),
) -> None:
    """Build FAISS index from PDFs in ``pdf_dir``."""
    pipeline = RAGPipeline()
    n = pipeline.build_index(pdf_dir=pdf_dir)
    typer.echo(f"Index built: {n} chunks saved to {pipeline.ingester.data_path}")


@app.command("ask")
def ask(
    question: str = typer.Argument(..., help="Question to ask the policy assistant"),
    hybrid: bool = typer.Option(True, help="Use hybrid BM25+FAISS retrieval"),
) -> None:
    """Ask one question (prints answer and source list)."""
    pipeline = RAGPipeline(use_hybrid=hybrid, use_query_rewrite=False, memory_k=0)
    result = pipeline.answer(question, store_memory=False)
    typer.echo(result.answer)
    if result.sources:
        typer.echo("\n--- Sources ---")
        for i, s in enumerate(result.sources, 1):
            page = f", p.{s.page}" if s.page is not None else ""
            typer.echo(f"[{i}] {s.filename}{page}")


if __name__ == "__main__":
    app()
