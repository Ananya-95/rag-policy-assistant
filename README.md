# Policy RAG Assistant

Retrieve passages from policy PDFs with FAISS, then answer questions with **Groq** (Llama). Run indexing or Q&A by editing and running **`main.py`** (or import `RAGPipeline` elsewhere).

## Where execution starts

| How you run it | What happens |
|----------------|----------------|
| **`python main.py`** | Runs the calls at the bottom of `main.py` (typically `build_index()` and/or `answer(...)`). Adds the project root to `sys.path` so `src.*` and `config` resolve. |
| **Your own script / notebook** | `from src.pipeline.rag_pipeline import RAGPipeline`, then `RAGPipeline().build_index()` or `.answer("...")`. Run with **`PYTHONPATH`** set to this project root, or run the script from this directory with the same path setup as `main.py`. |
| **Importing in another project** | Install the package in editable mode (if you add a `pyproject.toml`) or extend `PYTHONPATH` to include this folder so `src` and `config` resolve. |

End-to-end flow:

1. Put **PDFs** in `data/Docs/`.
2. **Index** (embed + FAISS): in `main.py`, call `pipeline.build_index()`, then run `python main.py` → writes `data/Faiss_Index/` and optional `data/chunks.json`.
3. **Ask**: set `GROQ_API_KEY`, uncomment or add `print(pipeline.answer("..."))` in `main.py`, run `python main.py` again → retrieves top-k chunks and calls Groq.

Configuration lives in **`config/settings.py`** (paths, embedding model, chunk sizes, `TOP_K`, Groq model).

## Setup

1. **Python 3.10+** recommended (3.9 may work with compatible LangChain versions).

2. Create a virtual environment and install dependencies:

   ```bash
   cd rag-policy-assistant
   python3 -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Environment variables** — create a `.env` file in the project root (optional but convenient; loaded by `config/settings.py`):

   ```env
   GROQ_API_KEY=your_groq_api_key_here
   ```

   Without a key, asking questions / `answer()` will fail when calling Groq.

4. **Data directories** — create if missing:

   ```bash
   mkdir -p data/Docs
   ```

   Add your policy PDFs under `data/Docs/` before indexing.

## Running via `main.py`

From **`rag-policy-assistant/`**, open **`main.py`** and set which methods you need, for example:

```python
pipeline = RAGPipeline()
pipeline.build_index()
# print(pipeline.answer("What is the vacation policy?"))
```

Then:

```bash
python main.py
```

Comment or remove `build_index()` when you only want to query an existing index.

## Programmatic usage

```python
from src.pipeline.rag_pipeline import RAGPipeline

p = RAGPipeline()
p.build_index()           # offline: PDFs → FAISS
answer = p.answer("What does the policy say about PTO?")  # retrieve + Groq
print(answer)
```

Run this from a context where **`rag-policy-assistant`** is the working directory and imports resolve (e.g. `PYTHONPATH=. python your_script.py`).

## Layout

- `main.py` — thin entry: constructs `RAGPipeline` and calls its methods (edit as needed).
- `config/settings.py` — paths and hyperparameters.
- `src/pipeline/rag_pipeline.py` — orchestrates ingest, index, retrieve, **answer**.
- `src/ingestion/` — PDF load and chunking.
- `src/embedding/` — Hugging Face embeddings.
- `src/vectorstore/` — FAISS save/load.
- `src/retrieval/` — query-time retrieval.
- `src/llm/` — Groq client.

## Notes

- First indexing run **downloads** the embedding model (`BAAI/bge-small-en-v1.5`) and may take a while.
- The FAISS index is loaded with `allow_dangerous_deserialization=True` — only load indexes **you** created.
- `Retriever` currently instantiates its own `Embedder`; indexing and querying share the same **model name** and **index path** from settings.
