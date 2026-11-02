# Capstone RAG: semantic retrieval over documentation

Ask questions in everyday language and get grounded, cited answers from your own documents. The default corpus is the
Constitution of India (PDF); any pdf, docx or txt works. The project question: how can we reliably retrieve the most relevant
passage for a natural-language question, better than keyword search?

```
file ─► Docling: parent chunks (a topic section, ~half a page)
          └─► Chonkie: child chunks (~65 words, split at meaning boundaries)
                └─► LLM per child: resolve references ("It" -> "The President"), Q&A pairs, factoids, triplets
document ─► LLM: topics (glossary) + summary
                          │
   React UI ◄─ FastAPI ◄─ Qdrant (cosine) ◄─ embeddings: all-MiniLM-L6-v2        triplets ─► knowledge graph (NetworkX)
```

Ask: embed the question → dense search over children → (optional) knowledge-graph candidates → cross-encoder re-order →
the matching children's **parent sections** go to the LLM as numbered context → a short answer with `[1]` citations, or an
explicit "I couldn't find this" when nothing relevant was retrieved (small-to-big retrieval, grounded answer, abstention).

## Run it
```bash
cp .env.example .env            # then put your GROQ_API_KEY in it
./start.sh                      # Qdrant (docker) + API :8000 + UI :5173. Or by hand:
docker compose up -d            # Qdrant on localhost:6335
uv run uvicorn capstone.main:app --reload          # API on :8000 (the first start downloads the models)
cd frontend && npm install && npm run dev          # UI on http://localhost:5173
```
`./stop.sh` stops the API and UI started from this folder (Qdrant keeps running; `./stop.sh --all` stops it too, data is kept).
Always run Python through `uv run`. `start.sh` is a bash script and needs Docker for Qdrant.
**Without Docker (Windows, or a machine where Docker is a burden):** set `QDRANT_PATH=data/qdrant` in `.env`. Qdrant then runs inside the API and keeps the
index in that folder (`start.ps1` / `stop.ps1` are the PowerShell versions of the scripts and skip Docker when it is set). To fill the folder from an index you
already have in the Docker Qdrant: `uv run python -m capstone.export_index data/qdrant` (about 30 seconds for 18,000 points); the folder, `data/graphs/` and
`data/enrichment_cache.jsonl` can be zipped and moved to another computer so it needs no ingest and no LLM calls. The PowerShell scripts and a Windows run have **not been tried**;
the folder mode was tried on Linux (the API answered a real question from a copied index).
`data/docs/` holds the PDF and sample help articles for the "Ingest the docs folder" button. Ingesting a large PDF with real
Groq calls takes tens of minutes (Groq's output-token rate limit is the bottleneck); LLM results are cached in `ENRICH_CACHE`, so a
repeated or resumed ingest is much faster.

| Page | Use |
|---|---|
| **Ask** `/` | Ask a question; filter by result type; optional **Knowledge graph** toggle. Grounded answer with citations, source cards linking to the full passage, "via graph" tags on passages the graph contributed. |
| **Documents** `/documents` | Upload files, paste text, ingest `DOCS_DIR`, list and delete documents. Re-ingesting identical text is skipped. |
| **Dashboard** `/dashboard` | Run the retrieval evaluation and chart it (Plotly): Recall@k, MRR, nDCG@k per retriever, Recall-vs-k curve, per-question ranks with CSV export, index statistics. |
| **Questions** `/questions` | Manage the labelled evaluation questions: add, edit, delete, train/test, import/export JSON, and see which section each question's evidence is found in. |
| **Knowledge graph** `/graph` | Search an entity, see it and its neighbours as a network, read its triplets with the source passage of each. |
| **Quality checks** `/quality` | Grade the answers (faithfulness, relevance, abstention) and check a sample of the generated Q&A pairs and factoids against their source chunks; mark each correct or incorrect and read the error rates. |
| **User guide** `/guide` | Illustrated usage documentation, with screenshots (regenerate them with `scripts/screenshots.sh`). |
| **API docs** `/api-docs` | The API reference (also at `:8000/docs`). |

## What is stored
One Qdrant collection (`COLLECTION`), one point per item, told apart by `type`:

| type | what | in a default search |
|---|---|---|
| `chunk` | a child chunk with references resolved; carries its parent's text, section and page | yes |
| `qa_pair` | question and answer generated from a chunk | yes |
| `factoid` | a short standalone fact from a chunk | yes |
| `topic` | glossary term and definition (per document) | yes |
| `summary` | abstract of the document | yes |
| `triplet` | "subject relation object" from a chunk (embedded) | only when asked for |
| `graph_summary` | one deterministic text per document describing its graph (top entities, common relations) | only when asked for |

Every point has `document_id`, `title`, `citation`, `file_hash` and `topics`; chunk-derived points also have `parent_id`, `chunk_id`,
`section` and `page`, which is how a small match leads back to its parent section. Dedup is by hash of the cleaned text. Every point also carries the document's `version`, `ingested_at` and `expires_at`.
LLM output is checked before it is stored: a reference rewrite must keep every number, facts must share most of their words with
the source, triplet entities must appear in the text. Anything failing a check is dropped.

## Configuration
Everything is in `.env` (see `.env.example`; read only by `src/capstone/config.py`). **Changing the embedding model or the point
schema needs a new `COLLECTION` name and a re-ingest.** Old collections are never deleted by the app.

## Retrieval and evaluation
Retrievers compared, on section-level relevance (a section is right if it contains the question's `evidence` phrase):
BM25 · un-tuned MiniLM · fine-tuned MiniLM · hybrid (BM25 + dense, reciprocal rank fusion) · hybrid + cross-encoder rerank ·
base + rerank · base + knowledge graph · base + knowledge graph + rerank.

Latest run on the **28 held-out test questions** (k = 5; `COLLECTION=constitution_india_v2`, 2,358 chunks, 524 sections):

| Retriever | Recall@5 | MRR | nDCG@5 |
|---|---|---|---|
| BM25 | 0.482 | 0.348 | 0.358 |
| Base MiniLM (un-tuned) | 0.893 | 0.701 | 0.731 |
| Fine-tuned MiniLM | 0.839 | 0.619 | 0.650 |
| Hybrid (BM25 + dense) | 0.714 | 0.559 | 0.568 |
| Hybrid + rerank | 0.929 | 0.766 | 0.801 |
| **Base + rerank** (what Ask uses) | **0.964** | **0.798** | **0.836** |
| Base + knowledge graph | 0.893 | 0.701 | 0.732 |
| Base + knowledge graph + rerank | 0.964 | 0.798 | 0.836 |

What this shows, and what it does not:
- **Dense + cross-encoder rerank is best.** BM25 is weak because users do not use the documents' words.
- **Fine-tuning did not beat the base model** here (two settings tried). The training questions are LLM-generated from the same text,
  and the base model is already strong on this corpus.
- **Hybrid RRF hurt**: BM25's weak ranking dilutes the dense one.
- **The knowledge graph is neutral.** At full fusion weight it hurt badly (Recall@5 0.554 without rerank); the weight was swept on the
  *train* questions only and set to 0.1, where it neither helps nor hurts. So `GRAPH_SEARCH` is off by default.
- **Caveats:** only 28 test questions, so differences of one or two questions are noise; all 62 questions were drafted by reading the text,
  so they are not an independent evaluation (add questions written by someone who has not read the documents); the rerank depth (30) came from
  a small sweep.

```bash
uv run python eval/eval.py --k 5 --split test      # the same code as the Dashboard's Run button
```
`eval/questions.json` holds the labelled questions (`id`, `question`, `evidence[]`, `split`, `note`). Evidence phrases survive a re-chunk or
re-ingest. The **Questions** page edits this file; a question whose evidence matches no section is flagged, because an evaluation leaves it out.

## Reranking
The Ask page re-orders its top `RERANK_CANDIDATES` (30) dense results with a cross-encoder (`RERANK_MODEL`, default
`cross-encoder/ms-marco-MiniLM-L-6-v2`). Only the order changes: the cosine score and the abstention rule (`ANSWER_MIN_SCORE`) are unchanged.
Turn it off with `RERANK_SEARCH=false`.

## Knowledge graph
Built from the `triplet` points the LLM extracted per chunk, so it needs no extra LLM calls and no re-ingest.
- **Entities are normalised** (case, plurals, possessives, leading articles, optional aliases from `GRAPH_ALIASES_FILE`), so "the Governors"
  and "Governor" are one node. Every edge remembers its source chunk and document.
- **NetworkX graph, saved to disk** at `GRAPH_DIR/<COLLECTION>.json`; rebuilt automatically after each ingest or delete, or via
  `POST /api/graph/rebuild` (or the button on the Knowledge graph page). Triplets are also searchable as `triplet` points, and one
  `graph_summary` point per document is stored.
- **Graph-assisted retrieval** (`GRAPH_SEARCH=true`, or the Ask toggle, or `"graph": true` in the request): entities named in the question are linked to
  nodes, the chunks their edges came from join the candidates, and the two rankings are fused with reciprocal rank fusion at weight `GRAPH_WEIGHT`
  (0.1, in `graph.py`). Edges of very common entities ("President") count only when the question names the relation or a second entity.
- It is deliberately **not Graph RAG**: no community detection, no LLM-written graph summaries, no query router, no separate graph LLM.
- The graph is as good as the LLM's triplets: many entities are noisy phrases ("both Houses assembled together"), and the graph is sparse
  (5,212 entities for 5,416 relations), which is part of why it adds nothing to retrieval here. It is useful for exploring the corpus.
- **Not built:** a second graph evaluation variant for BM25/hybrid/fine-tuned (only the base and base + rerank pipelines have graph twins).

## Documents: versions, expiry
Storing the same title with changed text is an update: the new version is built first (so a failed update leaves the old one), then the old version is deleted and
the new one stored as `version + 1`. `DOCUMENT_TTL_DAYS` (default 0 = never) sets an expiry on documents stored from then on; expired documents are deleted by a flush that
runs every `FLUSH_INTERVAL_HOURS` while the API is up, or on demand with `POST /api/documents/flush`. This is a loop inside the API, not an operating-system cron job, so
nothing is flushed while the API is stopped. The first Constitution ingest predates versions and counts as version 1 without an expiry.
The document summary and glossary are written from four slices spread over the whole text (not just the first 12,000 characters).

## Quality checks
`src/capstone/quality.py`, results in `QUALITY_FILE`, page `/quality`.
- **Answers:** each labelled test question is answered as Ask does; an LLM grades the answer against the passages it was given (faithful, relevant); questions in
  `eval/unanswerable.json` (off-topic) should be refused. Latest run (28 test questions, 8 off-topic, Groq `llama-3.3-70b-versatile` as writer **and** judge):
  faithfulness 100%, relevance 100%, wrongly abstained 0%, correctly abstained 100%. A judge from the same model family tends to be lenient, so read this as an upper bound.
- **Generated artifacts:** a random sample (30 per kind in the latest run) is checked by a second LLM pass against its source chunk. It flagged 2 of 30 Q&A pairs and 2 of 30 factoids (6.7% each).
  Looking at the four: one factoid is a real error (it says the Sixth Schedule where the source says the Seventh), one dropped half of "Union or State Commission", and two Q&A flags
  are context from the section heading that the chunk itself lacks. **The error rate to report is the one from a person's review** (buttons on the page); none has been recorded yet,
  and 30 items is a small sample.

## Fine-tune the embedding model (CPU is enough for MiniLM)
```bash
uv run python eval/finetune.py --epochs 2          # writes models/finetuned-minilm (gitignored)
# then FINE_TUNED_MODEL=models/finetuned-minilm in .env, and run the evaluation again
```
Training pairs are the generated Q&A questions, each with a hard negative (the most similar chunk from another Part), trained with
`MultipleNegativesRankingLoss`. **No leakage:** every Part that holds a `test` question is excluded from training entirely (positives and
negatives). To *search* with the tuned model, set `EMBED_MODEL` to it, use a new `COLLECTION` and re-ingest.

## Tests
```bash
uv run pytest        # 76 tests; fakes for embeddings, reranker and LLM, in-memory Qdrant: no network, no model download
```
Frontend: `cd frontend && npm run build`. The UI has been checked in headless Chrome in light mode only (dark mode is wired but not looked at).

## Layout
`src/capstone/`: `config` · `extract` · `chunking` · `parsing` (Docling parents, Chonkie children) · `enrich` (LLM per chunk) · `ingest` · `store` ·
`graph` (knowledge graph) · `answer` (grounded answer) · `quality` (answer and artifact checks) · `export_index` · `evaluation` · `finetune` · `schemas` · `main` (API).
`frontend/src/pages/`: Ask, Documents, Dashboard, Questions, Graph, Quality, Guide, ApiDocs, Passage. `scripts/screenshots.sh`: regenerates the guide's images. `eval/`: command-line evaluation and fine-tuning, labelled questions.
`tests/`. `data/graphs/`, `models/`, `eval/results.json` and `.env` are not committed.

## Not done yet
- An independent, human-written question set; a larger test set. All 62 questions were drafted by reading the text, so the evaluation is not independent.
- A person's review of the generated Q&A pairs and factoids (the page is built; the error rate that counts is yours). The LLM-judged numbers above use the same model family as the writer.
- Windows: `start.ps1` / `stop.ps1` and the Docker-free folder mode are untested on Windows.
- Dark mode is wired but has not been looked at; the Quality, User guide and Questions pages were only checked in headless Chrome, light mode, without clicking through every control.
- About 80 chunks start mid-sentence and about 30 are oversized (Chonkie cannot split inside a sentence).
- The Part labels Docling gives are coarse and inconsistent in the existing index (the parser fix applies to a re-ingest), which makes the fine-tuning hold-out wider than the test Parts alone.
- Expiry only applies to documents stored after `DOCUMENT_TTL_DAYS` is set, and the flush runs inside the API (see above).
