#!/usr/bin/env python3
"""Run from ``rag-policy-assistant/``: ``python main.py``. Edit the calls at the bottom as needed."""
import logging
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.pipeline.rag_pipeline import RAGPipeline

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

if __name__ == "__main__":
    pipeline = RAGPipeline()
    pipeline.build_index()
    # print(pipeline.answer("Your question here?"))
