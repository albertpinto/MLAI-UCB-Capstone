# Capstone RAG: semantic search over support documents

Ingest support documents (pdf, docx, txt), enrich them with an LLM, and search them by meaning.

```
file / pasted text ─► clean ─► chunk (no LLM) ─► LLM: Q&A per chunk, topics + summary per document
                                                   │
        React UI ◄─ FastAPI ◄─ Qdrant (cosine) ◄───┘   embeddings: all-MiniLM-L6-v2 (or your fine-tuned model)
```

One Qdrant collection holds four kinds of points, told apart by `type`: `chunk`, `qa_pair`, `topic`, `summary`.
Every point carries `document_id`, `title`, `citation`, `file_hash` and `topics`. Search is retrieval only: the
"best answer" is the top-ranked Q&A pair or summary, with its source.

## Run it
```bash
cp .env.example .env            # then put your GROQ_API_KEY in it
./start.sh                      # Qdrant + API + UI in one go. Or run the three lines below by hand:
docker compose up -d            # Qdrant on localhost:6335
uv run uvicorn capstone.main:app --reload          # API on :8000 (first start downloads the embedding model)
cd frontend && npm install && npm run dev          # UI on http://localhost:5173
```
`data/docs/` holds 36 sample Brandcil help articles for the "Ingest the docs folder" button.

| Page | Use |
|---|---|
| **Ask** `/` | Ask a question, filter by result type, see the best answer and ranked results with sources. |
| **Ingest** `/ingest` | Drop files, or ingest everything in `DOCS_DIR`. Re-ingesting the same text shows `skipped`. |
| **Add document** `/add` | Paste a title and text. Also lists stored documents so you can delete one. |

## Configuration
Everything is in `.env` (see `.env.example`, read only by `src/capstone/config.py`). To use the fine-tuned model, set
`EMBED_MODEL` to its folder. **Changing the embedding model changes the vectors: use a new `COLLECTION` name (or delete
the old one) and re-ingest.**

## Evaluate retrieval
```bash
uv run --group eval python eval/eval.py --k 5
```
Compares BM25, stock MiniLM and `EMBED_MODEL` (if different) by Recall@k, MRR and nDCG@k over `eval/questions.json`
(16 starter questions; add more, ideally written by someone who has not read the documents). It scores raw chunks only.
Do not evaluate on the pairs you fine-tuned with.

## Tests
```bash
uv run pytest        # fakes for embeddings and the LLM, in-memory Qdrant: no network needed
```

## Layout
`src/capstone/`: `config` · `extract` · `chunking` · `llm` · `store` · `ingest` · `schemas` · `main` (API).
`frontend/`: Vite + React, three pages. `eval/`: the comparison script. `tests/`.

## Notes
- Ingestion runs one document at a time in the background; progress is polled from `/api/jobs/{id}`. Jobs are kept
  in memory and are lost on restart; stored documents are not.
- Each chunk costs one LLM call, so large documents take a while and may hit Groq rate limits (calls retry with backoff;
  a chunk that still fails is skipped and the document continues).
- Fine-tuning is not part of this project yet; the old script is `rag-product/backend/src/compute/gte_model_training.py`.
