# Policy corpus — add your PDFs here

The bootcamp checklist expects **10–15 real policy PDFs** in this folder.

## Recommended sources (free, public)

1. **SEBI** — Research Analyst regulations, circulars  
   https://www.sebi.gov.in/legal/circulars.html

2. **RBI** — Master directions, FAQs (HR/compliance style policies work too)

3. **Company HR** — employee handbooks, leave policies (PDF exports)

## Steps

1. Download 10–15 PDFs into this directory (`data/Docs/`).
2. Rebuild the index:
   ```bash
   python ingest.py
   # or
   python main.py index
   ```
3. Commit PDFs + index (Git LFS is configured for `*.pdf`, `*.faiss`, `*.pkl`).
4. Re-deploy Hugging Face Space so the live app uses the new corpus.

## Current state

There is a small sample PDF (`1770375507051.pdf`) used for the golden eval set. Replace or supplement it with full SEBI/RBI documents for production-quality answers.
