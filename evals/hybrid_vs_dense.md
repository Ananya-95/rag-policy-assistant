# A/B Eval: Hybrid vs Dense Retrieval

**Date:** 2026-05-02  
**Model:** llama-3.3-70b-versatile (Groq)  
**Embeddings:** BAAI/bge-small-en-v1.5  
**Dense baseline:** FAISS top-5  
**Hybrid system:** BM25 top-10 + FAISS top-10 → RRF → top-5 → (optional) Cohere rerank  

---

## Results

| # | Question | Dense Answer (summary) | Hybrid Answer (summary) | Improved? | Notes |
|---|----------|----------------------|------------------------|-----------|-------|
| 1 | | | | ⬜ TBD | |
| 2 | | | | ⬜ TBD | |
| 3 | | | | ⬜ TBD | |
| 4 | | | | ⬜ TBD | |
| 5 | | | | ⬜ TBD | |
| 6 | | | | ⬜ TBD | |
| 7 | | | | ⬜ TBD | |
| 8 | | | | ⬜ TBD | |
| 9 | | | | ⬜ TBD | |
| 10 | | | | ⬜ TBD | |

**Goal:** ✅ improved on ≥ 3 of 10 questions  
**Actual:** _TBD_

---

## Observations

_Fill in after running eval_

- BM25 wins when:
- Dense wins when:
- Reranker impact:

---

## How to run

```bash
# Run the eval script (to be built on Saturday)
python evals/run_eval.py --mode dense   > evals/dense_results.json
python evals/run_eval.py --mode hybrid  > evals/hybrid_results.json
```
