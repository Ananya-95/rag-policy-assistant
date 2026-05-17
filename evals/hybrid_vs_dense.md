# A/B Eval: Hybrid vs Dense Retrieval

**Date:** 2026-05-17  
**Model:** llama-3.3-70b-versatile (Groq)  
**Embeddings:** BAAI/bge-small-en-v1.5  
**Dense baseline:** FAISS top-5  
**Hybrid system:** BM25 top-10 + FAISS top-10 → RRF → top-5 → (optional) Cohere rerank  

---

## Results

| # | Question | Dense Answer (summary) | Hybrid Answer (summary) | Improved? | Notes |
|---|----------|------------------------|-------------------------|-----------|-------|
| 1 | What is RAASB and when was it established? | _run eval_ | _run eval_ | ⬜ TBD | |
| 2 | Which NISM certification examination must a Research Analyst pass? | _run eval_ | _run eval_ | ⬜ TBD | |
| 3 | Can an Investment Adviser also register as a Research Analyst? | _run eval_ | _run eval_ | ⬜ TBD | |
| 4 | Part-time RA prohibited activities (two)? | _run eval_ | _run eval_ | ⬜ TBD | |
| 5 | Employee employer letter requirement? | _run eval_ | _run eval_ | ⬜ TBD | |
| 6 | Minimum net worth for non-individual RA? | _run eval_ | _run eval_ | ⬜ TBD | |
| 7 | Validity period of RA registration certificate? | _run eval_ | _run eval_ | ⬜ TBD | |
| 8 | Disclosures required in research report? | _run eval_ | _run eval_ | ⬜ TBD | |
| 9 | Can a Research Analyst accept gifts from a client? | _run eval_ | _run eval_ | ⬜ TBD | |
| 10 | Penalty for non-compliance with SEBI RA Regulations? | _run eval_ | _run eval_ | ⬜ TBD | |

**Goal:** ✅ improved on ≥ 3 of 10 questions  
**Actual:** _fill after comparing JSON outputs_

---

## Observations

_Fill in after running eval_

- BM25 wins when: exact policy codes, acronyms (RAASB, NISM), named entities
- Dense wins when: paraphrased questions, semantic similarity without keyword overlap
- Reranker impact: set `COHERE_API_KEY` and re-run hybrid mode

---

## How to run

```bash
# Requires GROQ_API_KEY in .env
python evals/run_eval.py --mode dense --limit 10 -o evals/dense_results.json
python evals/run_eval.py --mode hybrid --limit 10 -o evals/hybrid_results.json

# Compare answers side-by-side, then mark Improved? column above
```
