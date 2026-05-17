#!/usr/bin/env python3
"""
Standalone ingestion entry point (bootcamp checklist).

Usage:
    python ingest.py
    python ingest.py --pdf-dir data/Docs
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.pipeline.rag_pipeline import RAGPipeline

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build FAISS index from policy PDFs")
    parser.add_argument(
        "--pdf-dir",
        default="data/Docs",
        help="Directory containing .pdf files (default: data/Docs)",
    )
    args = parser.parse_args()
    pipeline = RAGPipeline()
    n = pipeline.build_index(pdf_dir=args.pdf_dir)
    print(f"Done: {n} chunks → data/Faiss_Index/ and data/chunks.json")


if __name__ == "__main__":
    main()
