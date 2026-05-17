#!/usr/bin/env python3
"""CLI: ``python main.py index``"""
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
def build_index(pdf_dir: str = typer.Option("data/Docs", help="Folder containing PDF files")) -> None:
    pipeline = RAGPipeline()
    n = pipeline.build_index(pdf_dir=pdf_dir)
    typer.echo(f"Index built: {n} chunks → data/Faiss_Index/")
if __name__ == "__main__":
    app()
