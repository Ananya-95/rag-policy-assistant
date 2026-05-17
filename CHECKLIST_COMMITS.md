# Bootcamp checklist → Git commits

Use this map when reviewing your GitHub history. **Push:** `git push origin main --tags`

## Already on `main` (before day-1 patch commits)

| Day | Commit | Message |
|-----|--------|---------|
| Mon–Tue base | `762cedb` | feat: add modular RAG project structure |
| Tue pipeline | `044bafb` | RAG PIPELINE1_V2 |
| **Wed** | `5157719` | feat: hybrid retrieval — BM25 + FAISS + RRF |
| **Thu** | `caf79cf` | feat: conversational memory + query rewrite — **tag `v0.2-advanced-retrieval`** |
| **Fri** | `a685461` | deploy-ready — Groq, README, HF config |
| Fri | `7eb8fe9` | Dockerfile for HF Spaces |
| Fri | `dc9998f` | Git LFS for index/PDFs |
| **Sat** | `3182071` | golden.json (25 Q&A) + ragas/deepeval |
| Fri | `a63333a` | HF README short_description fix |

## New commits (checklist completion)

| Day | Commit | Message |
|-----|--------|---------|
| **Mon** | `b9e8d9e` | feat(day-1): pdf ingestion + faiss index build |
| **Tue** | `8ad049f` | feat(day-2): Streamlit sources + citations — **tag `v0.1-mvp`** |
| **Wed** | `46dec1c` | docs(day-3): hybrid vs dense eval + Cohere |
| **Thu** | _(see `caf79cf` above)_ | memory + multi-query already shipped |
| **Fri** | `7583b64` | chore(day-5): LICENSE + README URLs |
| **Sat** | `9a0605e` | feat(day-6): evals/run_eval.py |
| **Sun** | `d22f88f` | chore(day-7): PLAN.md + tests + corpus guide |

## Tags

| Tag | Points to | Day |
|-----|-----------|-----|
| `v0.1-mvp` | `8ad049f` | Tue |
| `v0.2-advanced-retrieval` | `caf79cf` | Thu |

## Verify locally

```bash
git log --oneline --decorate -20
python main.py index          # Mon
streamlit run streamlit_app.py  # Tue
python evals/run_eval.py --mode hybrid --limit 3  # Sat
python -m unittest tests.test_retriever -v       # Sun
```
