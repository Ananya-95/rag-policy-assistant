# Week 2 Plan — Policy RAG Assistant

## Goals

1. **RAGAS eval at scale** — run all 25 golden questions; track faithfulness, answer relevancy, context precision
2. **Latency & cost baseline** — log Groq tokens + retrieval ms per query
3. **QLoRA fine-tune (optional)** — Colab T4: adapter on Llama-3.1-8B for domain tone
4. **Corpus expansion** — 10–15 SEBI/RBI PDFs; rebuild index weekly

## Week 2 tasks

| Day | Focus | Deliverable |
|-----|--------|-------------|
| Mon | RAGAS pipeline | `evals/run_eval.py --ragas --limit 25` + scores in `evals/ragas_report.json` |
| Tue | Failure analysis | Top 5 failure modes from manual + RAGAS runs |
| Wed | Retrieval tuning | Try Cohere rerank on; compare hybrid_vs_dense v2 |
| Thu | Latency | Add timing logs; p50/p95 in `evals/latency.md` |
| Fri | Fine-tune spike | Colab notebook: QLoRA on 50–100 (Q, A) pairs from golden.json |
| Sat | Deploy v0.4 | HF Space green build; demo GIF in README |
| Sun | Applications | 25 more apps; LinkedIn post on eval results |

## Metrics to hit

- RAGAS faithfulness ≥ 0.75 on golden subset (n=25)
- Retrieval p95 < 2s on CPU (FAISS + BM25)
- Live HF URL uptime (fix secrets: `GROQ_API_KEY`, optional `COHERE_API_KEY`)

## Blockers from Week 1 (resolved in repo)

- [x] `build_index()` wired end-to-end
- [x] Streamlit sources panel + citations in prompt
- [x] `evals/run_eval.py`, `manual_run.md`, `PLAN.md`
- [ ] **You:** add 10–15 real PDFs to `data/Docs/` (see `data/Docs/README.md`)
- [ ] **You:** HF Space Secrets → `GROQ_API_KEY`; rebuild Space
- [ ] **You:** run manual_run + hybrid A/B; fill result tables

## Target companies (continue applications)

Tier 1 (no DSA): Deloitte GenAI, Sarvam AI, CoRover, EY/PwC/KPMG AI practices  
Tier 2 (easy DSA): Fractal, Tiger Analytics, Haptik, Walmart GTI, Krutrim, Adobe India
