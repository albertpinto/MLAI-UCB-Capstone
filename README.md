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
The **Dashboard** page (menu: Insights) runs and charts the evaluation: Recall@k, MRR and nDCG@k for BM25, the un-tuned
MiniLM and, once you have one, the fine-tuned model, plus a Recall@k curve, a per-question table (CSV export) and index statistics.
The same code runs from the command line:
```bash
uv run python eval/eval.py --k 5 --split test
```
Questions live in `eval/questions.json`. Each has `evidence`, a phrase copied from the text that answers it; a section that
contains it is the right answer, so labels survive a re-chunk. Each question is `train` or `test`, split by Part of the
document. The shipped 62 were drafted by reading the Constitution, so treat them as a first set: add questions written by someone who
has not read the text. Retrieval is scored on the chunks stored in the `COLLECTION` in `.env`.

## Fine-tune the embedding model (CPU is enough for MiniLM)
```bash
uv run python eval/finetune.py --epochs 2          # writes models/finetuned-minilm
# then put FINE_TUNED_MODEL=models/finetuned-minilm in .env and run the evaluation again
```
Training pairs are the generated Q&A questions with a hard negative (the most similar chunk from another Part), trained with
`MultipleNegativesRankingLoss`. Parts that hold a `test` question are left out of training entirely (positives and negatives), so the test
numbers are not leaked. To *search* with the tuned model, set `EMBED_MODEL` to it, use a new `COLLECTION` and re-ingest.

## Tests
```bash
uv run pytest        # fakes for embeddings and the LLM, in-memory Qdrant: no network needed
```

## Layout
`src/capstone/`: `config` · `extract` · `parsing` (Docling parents, Chonkie children) · `enrich` (LLM per chunk) · `ingest` · `store` · `answer` · `evaluation` · `finetune` · `schemas` · `main` (API).
`frontend/`: Vite + React + Tailwind; pages Ask, Documents, Dashboard (Plotly), API docs. `eval/`: command-line wrappers for the evaluation and fine-tuning, and the labelled questions. `tests/`.

## Notes
- Ingestion runs one document at a time in the background; progress is polled from `/api/jobs/{id}`. Jobs are kept
  in memory and are lost on restart; stored documents are not.
- Each chunk costs one LLM call, so large documents take a while and may hit Groq rate limits (calls retry with backoff;
  a chunk that still fails is skipped and the document continues).
- The first evaluation is a baseline, not a final result: the questions were drafted from the text, and the Part labels Docling gives are coarse (a Part heading that appears on few pages is not always picked up), which makes the hold-out wider than the test Parts alone.
