#!/usr/bin/env python3
"""
Run retrieval + generation evals on evals/golden.json.

Examples:
    python evals/run_eval.py --mode dense --limit 10 -o evals/dense_results.json
    python evals/run_eval.py --mode hybrid --limit 10 -o evals/hybrid_results.json
    python evals/run_eval.py --mode hybrid --limit 5 --ragas
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv

load_dotenv(_ROOT / ".env")

from src.pipeline.rag_pipeline import RAGPipeline


def load_golden(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def run_eval(mode: str, limit: int | None) -> list[dict]:
    use_hybrid = mode == "hybrid"
    pipeline = RAGPipeline(
        use_hybrid=use_hybrid,
        use_query_rewrite=False,
        memory_k=0,
    )
    items = load_golden(_ROOT / "evals" / "golden.json")
    if limit:
        items = items[:limit]

    results = []
    for item in items:
        q = item["question"]
        print(f"[{mode}] {item.get('id', '?')}: {q[:60]}…")
        try:
            response = pipeline.answer(q, store_memory=False)
            answer = response.answer
            sources = [
                {"filename": s.filename, "page": s.page, "snippet": s.snippet[:200]}
                for s in response.sources
            ]
        except Exception as exc:
            answer = f"ERROR: {exc}"
            sources = []

        results.append(
            {
                "id": item.get("id"),
                "question": q,
                "ideal_answer": item.get("ideal_answer"),
                "ground_truth_source": item.get("ground_truth_source"),
                "answer": answer,
                "sources": sources,
                "mode": mode,
            }
        )
    return results


def run_ragas(results: list[dict]) -> dict:
    """Optional RAGAS faithfulness / answer relevancy (requires GROQ_API_KEY)."""
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import answer_relevancy, faithfulness

    rows = {
        "question": [r["question"] for r in results],
        "answer": [r["answer"] for r in results],
        "contexts": [
            [s["snippet"] for s in r.get("sources", [])] or [""]
            for r in results
        ],
        "ground_truth": [r.get("ideal_answer", "") for r in results],
    }
    ds = Dataset.from_dict(rows)
    scores = evaluate(ds, metrics=[faithfulness, answer_relevancy])
    return dict(scores)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate RAG pipeline on golden.json")
    parser.add_argument("--mode", choices=["dense", "hybrid"], default="hybrid")
    parser.add_argument("--limit", type=int, default=10, help="Max questions (default 10)")
    parser.add_argument("-o", "--output", help="Write JSON results to this path")
    parser.add_argument("--ragas", action="store_true", help="Run RAGAS metrics after eval")
    args = parser.parse_args()

    if not os.getenv("GROQ_API_KEY"):
        print("Warning: GROQ_API_KEY not set — generation will fail.")

    results = run_eval(args.mode, args.limit)

    if args.ragas:
        try:
            ragas_scores = run_ragas(results)
            print("\nRAGAS scores:", ragas_scores)
        except Exception as exc:
            print(f"RAGAS skipped: {exc}")

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Wrote {len(results)} results → {out}")
    else:
        print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
