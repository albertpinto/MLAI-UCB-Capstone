### Capstone RAG: semantic retrieval over documentation

**Author**: albertpinto (UCB MLAI capstone)

#### Executive summary

Ask a question in everyday words and get a short, cited answer from your own documents, or an explicit "I couldn't find this" when the documents do not say.
The system splits documents into parent sections and small child chunks, has an LLM add Q&A pairs, facts and a knowledge graph, stores everything in Qdrant, and answers with a
dense search followed by a cross-encoder reranker. On 28 held-out test questions about the Constitution of India (k = 5), **dense search plus reranking found the right section for
91% to 96% of questions (Recall@5 0.911 to 0.964, MRR about 0.78)**, against 52% for BM25 keyword search (Recall@5 0.518, MRR 0.359); that gap is statistically clear. Fine-tuning the embedding model,
hybrid BM25 + dense search and the knowledge graph showed no benefit over the simple base model with a reranker on this corpus. It is a FastAPI + React app that runs locally; Groq or OpenAI supplies the LLM.

**The data analysis is in the notebook: [notebooks/capstone_analysis.ipynb](notebooks/capstone_analysis.ipynb)** (pandas, seaborn, bootstrap confidence intervals, a grid search and cross-validation grouped by Part).

#### Rationale

People search support and policy documentation with the words they use ("Why didn't my post go out?") while the documents use formal terms, so keyword search misses. Finding the right
passage quickly saves time, and an answer that is grounded in the documents, with citations and the ability to say "not found", is more trustworthy than one a language model makes up.
The same retrieval problem appears for any large body of text: policies, manuals, legal text. The Constitution of India is the default corpus because it is long, public, formal and full of
overlapping topics, which is a hard case for keyword search.

#### Research Question

How can we reliably retrieve the most relevant passage for a natural-language question, better than keyword search?
Sub-questions, each measured against the same labelled questions:
- Does a dense model beat BM25, and does a cross-encoder reranker add to it?
- Does fine-tuning the embedding model on the corpus help? Does hybrid BM25 + dense search? Does a knowledge graph?
- Can the system abstain when nothing relevant is retrieved, and is its answer faithful to the passages it was given?
- How often is the LLM-generated material (Q&A pairs, facts) wrong?

#### Data Sources

- **Corpus:** the Constitution of India (2024), a 402-page PDF in `data/docs/` (the one data file kept in git). Any `pdf`, `docx` or `txt` can be ingested. Ingested as 2,358 child chunks in 524 sections.
- **Derived data**, made by an LLM per chunk and stored as typed points: resolved chunk text (references such as "It" replaced by the name), Q&A pairs, factoids, knowledge-graph triplets (5,416 relations over 5,212 entities), plus a glossary and a summary per document.
- **Labelled evaluation set:** `eval/questions.json`, 62 questions each with an `evidence` phrase copied from the answering passage and a `split` (train or test, divided by Part of the Constitution so fine-tuning cannot leak). 28 are test questions. `eval/unanswerable.json` has 8 off-topic questions that should be refused.
- **Data challenges, handled explicitly:** one long document with many overlapping topics; LLM-generated content that can be wrong (every item is checked, see Methodology); noise from PDF extraction (running headers, footnotes); and questions drafted by reading the text, so the evaluation is not independent (see Next steps).

#### Methodology

##### Pipeline

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

- **Three-level chunking:** Docling gives parents; Chonkie splits each into children; one LLM call per child resolves references so the chunk can be read alone.
- **Reranking:** the Ask page re-orders its top `RERANK_CANDIDATES` (30) dense results with a cross-encoder (`RERANK_MODEL`, default `cross-encoder/ms-marco-MiniLM-L-6-v2`). Only the order changes: the cosine score and the abstention rule (`ANSWER_MIN_SCORE`) are unchanged. Turn it off with `RERANK_SEARCH=false`.
- **Checks on LLM output before it is stored:** a reference rewrite must keep every number, facts must share most of their words with the source, triplet entities must appear in the text. Anything failing a check is dropped.

##### What is stored

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

Every point has `document_id`, `title`, `citation`, `file_hash` and `topics`; chunk-derived points also have `parent_id`, `chunk_id`, `section` and `page`, which is how a small match leads back to its parent section.
Dedup is by hash of the cleaned text. Every point also carries the document's `version`, `ingested_at` and `expires_at`.

##### Configuration

Everything is in `.env` (see `.env.example`; read only by `src/capstone/config.py`, and `.env` wins over variables already exported in the shell). **Changing the embedding model or the point
schema needs a new `COLLECTION` name and a re-ingest.** Old collections are never deleted by the app.

##### LLM provider and speed

`LLM_PROVIDER=groq` (default) or `openai`, with `GROQ_API_KEY`/`GROQ_MODEL` or `OPENAI_API_KEY`/`OPENAI_MODEL`. Calls are throttled (`LLM_MAX_CONCURRENT`, `LLM_MIN_INTERVAL`), a 429 waits for the time the
provider asks, and errors that can never succeed (wrong key or model) fail at once. LLM results are cached per model in `ENRICH_CACHE`, so a repeated or resumed ingest is much faster; switching model enriches again.
`uv run python scripts/bench_ingest.py` times the enrichment on the same pages of your PDF for whichever provider is set. Measured on 213 chunks, scaled to the whole Constitution (an estimate):

| Provider and model | Parallel calls | Time for 213 chunks | Whole Constitution (estimate) |
|---|---|---|---|
| Groq `qwen/qwen3.8-27b` | 4 | 91.6 s | about 17 min (more in a long run: Groq caps output tokens per minute) |
| Groq `qwen/qwen3.8-27b` | 8 | 89.3 s (162 rate-limit retries) | about 16 min |
| OpenAI `gpt-4o-mini` | 4 | 168.9 s | about 31 min |
| OpenAI `gpt-4o-mini` | 12 | 63.0 s | about 12 min |
| OpenAI `gpt-4o-mini` | 24 | 39.4 s | about 7 min |
| OpenAI `gpt-4o-mini` | 48 | 29.8 s | about 5 min |

Use `LLM_WORKERS` = `LLM_MAX_CONCURRENT` = 4 for Groq and up to 24 for OpenAI. This compares speed only, not the quality of what each model writes.

##### Retrieval and evaluation

Retrievers compared, on section-level relevance (a section is right if it contains the question's `evidence` phrase):
BM25 · un-tuned MiniLM · fine-tuned MiniLM · hybrid (BM25 + dense, reciprocal rank fusion) · hybrid + cross-encoder rerank ·
base + rerank · base + knowledge graph · base + knowledge graph + rerank.

```bash
uv run python eval/eval.py --k 5 --split test      # the same code as the Dashboard's Run button
```
`eval/questions.json` holds the labelled questions (`id`, `question`, `evidence[]`, `split`, `note`). Evidence phrases survive a re-chunk or
re-ingest. The **Questions** page edits this file; a question whose evidence matches no section is flagged, because an evaluation leaves it out.

The statistical analysis lives in [the notebook](notebooks/capstone_analysis.ipynb): bootstrap confidence intervals and paired tests for every comparison, a **grid search** of BM25 (`k1`, `b`), the hybrid's fusion
constant and the reranker's depth, and **5-fold cross-validation grouped by Part**, so a setting is never tuned on a Part it is judged on. The metrics (Recall@5, MRR, nDCG@5) and why they fit this problem are explained there.
```bash
uv run --group eval --group notebook python eval/grid_search.py     # scores every setting on every question (about 4 minutes), writes notebooks/data/
uv run --group notebook jupyter lab notebooks/capstone_analysis.ipynb
```

##### Fine-tuning the embedding model (CPU is enough for MiniLM)

```bash
uv run python eval/finetune.py --epochs 2          # writes models/finetuned-minilm (gitignored)
# then FINE_TUNED_MODEL=models/finetuned-minilm in .env, and run the evaluation again
```
Training pairs are the generated Q&A questions, each with a hard negative (the most similar chunk from another Part), trained with
`MultipleNegativesRankingLoss`. **No leakage:** every Part that holds a `test` question is excluded from training entirely (positives and
negatives). To *search* with the tuned model, set `EMBED_MODEL` to it, use a new `COLLECTION` and re-ingest.

##### Knowledge graph

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

##### Documents: versions, expiry

Storing the same title with changed text is an update: the new version is built first (so a failed update leaves the old one), then the old version is deleted and
the new one stored as `version + 1`. `DOCUMENT_TTL_DAYS` (default 0 = never) sets an expiry on documents stored from then on; expired documents are deleted by a flush that
runs every `FLUSH_INTERVAL_HOURS` while the API is up, or on demand with `POST /api/documents/flush`. This is a loop inside the API, not an operating-system cron job, so
nothing is flushed while the API is stopped. The document summary and glossary are written from four slices spread over the whole text (not just the first 12,000 characters).

##### Quality checks

`src/capstone/quality.py`, results in `QUALITY_FILE`, page `/quality`.
- **Answers:** each labelled test question is answered as Ask does; an LLM grades the answer against the passages it was given (faithful, relevant); questions in
  `eval/unanswerable.json` (off-topic) should be refused.
- **Generated artifacts:** a random sample is checked by a second LLM pass against its source chunk; a person can then mark each item correct or incorrect on the page, and the page shows both error rates.

#### Results

Latest run on the **28 held-out test questions** (k = 5; `COLLECTION=constitution_india_v2`, searching the Constitution's 2,358 chunks in 524 sections; the index was re-ingested since the first version of this table, so the numbers moved slightly):

| Retriever | Recall@5 | MRR | nDCG@5 |
|---|---|---|---|
| BM25 | 0.518 | 0.359 | 0.379 |
| Base MiniLM (un-tuned) | 0.875 | 0.696 | 0.720 |
| Fine-tuned MiniLM | 0.911 | 0.644 | 0.696 |
| Hybrid (BM25 + dense) | 0.786 | 0.562 | 0.592 |
| Hybrid + rerank | **0.964** | 0.771 | **0.815** |
| **Base + rerank** (what Ask uses) | 0.911 | **0.780** | 0.795 |
| Base + knowledge graph | 0.875 | 0.681 | 0.711 |
| Base + knowledge graph + rerank | 0.911 | 0.777 | 0.792 |

What this shows, and what it does not (confidence intervals, tests and cross-validation are in [the notebook](notebooks/capstone_analysis.ipynb)):
- **Meaning-based search beats keyword search, and the gap is real.** BM25 answers 15 of 28 questions in its top 5; the base model 25, and the rerank pipelines 26 to 27. The paired bootstrap interval for the MRR gain over BM25 excludes 0 (+0.34 for the base model, +0.42 with rerank), on the test questions and on all 62.
- **Reranking puts the right section first more often** (68% of questions against 57% for the base model), but with 28 questions the MRR gain (+0.08) is **not statistically shown** (95% interval -0.07 to 0.25). About 200 independent questions would be needed to confirm a gap that size.
- **Fine-tuning showed no benefit.** Recall@5 is one question higher (0.911 against 0.875) but MRR is lower (0.644 against 0.696); the difference is within the noise. The training questions are LLM-generated from the same text, and the base model is already strong on this corpus.
- **Hybrid RRF hurt**: BM25's weak ranking dilutes the dense one. Not shown on the 28 test questions, but clear on all 62 (MRR -0.14, interval -0.24 to -0.05).
- **The knowledge graph is neutral** (MRR 0.681 against 0.696, within the noise). In an earlier run, at full fusion weight it hurt badly (Recall@5 0.554 without rerank); the weight was swept on the *train* questions only and set to 0.1. So `GRAPH_SEARCH` is off by default.
- **Settings barely matter.** In a grid search with cross-validation grouped by Part, tuned BM25 settings, the fusion constant and the rerank depth (20 against the default 30) each moved MRR by less than 0.03, and no gain was statistically shown.
- **Caveats:** only 28 test questions, so differences of one or two questions are noise; all 62 questions were drafted by reading the text, so they are not an independent evaluation.

**Answer quality** (28 test questions, 8 off-topic, Groq `llama-3.3-70b-versatile` as writer **and** judge): faithfulness 100%, relevance 100%, wrongly abstained 0%, correctly abstained 100%.
A judge from the same model family tends to be lenient, so read this as an upper bound.

**Generated material:** a second LLM pass flagged 2 of 30 sampled Q&A pairs and 2 of 30 sampled factoids (6.7% each). Of the four, one factoid is a real error (it says the Sixth Schedule where the source says the
Seventh), one dropped half of "Union or State Commission", and two Q&A flags are context from the section heading that the chunk itself lacks. **The error rate to report is the one from a person's review**
(buttons on the Quality page); none has been recorded yet, and 30 items is a small sample.

#### Next steps

- An independent, human-written question set, and a larger test set (about 200 questions to confirm a re-ranking gain of 0.08): all 62 questions were drafted by reading the text, so the evaluation is not independent. This is the most important gap.
- A person's review of 50 to 100 generated Q&A pairs and factoids, so the error rate reported is a human one (the page is built).
- Compare the quality of OpenAI's and Groq's output (only speed has been compared), and rerun the quality checks with a judge from a different model family than the writer.
- An ablation of the derived artifacts (chunks only against chunks plus Q&A, glossary, summary and factoids) and graph variants for BM25, hybrid and fine-tuned (only the base and base + rerank pipelines have graph twins).
- Windows: `start.ps1` / `stop.ps1` and the Docker-free folder mode are untested on Windows. Dark mode is wired but has not been looked at; the Quality, User guide and Questions pages were only checked in headless Chrome, light mode, without clicking through every control.
- About 80 chunks start mid-sentence and about 30 are oversized (Chonkie cannot split inside a sentence). The Part labels Docling gives are coarse and inconsistent in the existing index (the parser fix applies to a re-ingest), which makes the fine-tuning hold-out wider than the test Parts alone.
- Expiry only applies to documents stored after `DOCUMENT_TTL_DAYS` is set, and the flush runs inside the API.

#### Outline of project

- [**Analysis notebook**](notebooks/capstone_analysis.ipynb) (executed, with charts) and its data in [`notebooks/data/`](notebooks/data): business understanding, data cleaning, descriptive statistics, the retriever comparison, confidence intervals and tests, the grid search with grouped cross-validation, and findings with recommendations for a non-technical reader
- [User guide (PDF)](Capstone_User_Guide.pdf) and [startup steps (PDF)](Capstone_Startup_Steps.pdf) / [PowerPoint](Capstone_Startup_Steps.pptx)
- [`src/capstone/`](src/capstone): the app. `config` · `extract` · `chunking` · `parsing` (Docling parents, Chonkie children) · `enrich` (LLM per chunk) · `llm` (Groq or OpenAI) · `ingest` · `store` · `graph` (knowledge graph) · `answer` (grounded answer) · `quality` · `export_index` · `evaluation` · `finetune` · `schemas` · `main` (API)
- [`eval/`](eval): command-line evaluation, grid search and fine-tuning (`eval.py`, `grid_search.py`, `finetune.py`), the labelled questions (`questions.json`, `unanswerable.json`)
- [`frontend/src/pages/`](frontend/src/pages): Ask, Documents, Dashboard, Questions, Graph, Quality, Guide, ApiDocs, Passage
- [`scripts/`](scripts): `bench_ingest.py` (time the LLM providers), `screenshots.sh` and `guide_pdf.sh` (rebuild the user guide), `check_llm_key.py` (used by the start scripts)
- [`tests/`](tests): `uv run pytest` (89 tests; fakes for embeddings, reranker and LLM, in-memory Qdrant: no network, no model download). Frontend: `cd frontend && npm run build`.

##### Run it

```bash
cp .env.example .env            # then put your GROQ_API_KEY (or OPENAI_API_KEY with LLM_PROVIDER=openai) in it
./start.sh                      # Qdrant (docker) + API :8000 + UI :5173. Or by hand:
docker compose up -d            # Qdrant on localhost:6335
uv run uvicorn capstone.main:app --reload          # API on :8000 (the first start downloads the models)
cd frontend && npm install && npm run dev          # UI on http://localhost:5173
```
`./stop.sh` stops the API and UI started from this folder (Qdrant keeps running; `./stop.sh --all` stops it too, data is kept).
Always run Python through `uv run`. `start.sh` is a bash script and needs Docker for Qdrant, and it checks for the key of the provider set in `LLM_PROVIDER`.
**Without Docker (Windows, or a machine where Docker is a burden):** set `QDRANT_PATH=data/qdrant` in `.env`. Qdrant then runs inside the API and keeps the
index in that folder (`start.ps1` / `stop.ps1` are the PowerShell versions of the scripts and skip Docker when it is set). To fill the folder from an index you
already have in the Docker Qdrant: `uv run python -m capstone.export_index data/qdrant` (about 30 seconds for 18,000 points); the folder, `data/graphs/` and
`data/enrichment_cache.jsonl` can be zipped and moved to another computer so it needs no ingest and no LLM calls. The PowerShell scripts and a Windows run have **not been tried**;
the folder mode was tried on Linux (the API answered a real question from a copied index).
`data/docs/` holds the PDF for the "Ingest the docs folder" button. Ingesting a large PDF with real LLM calls takes from several minutes (OpenAI, many parallel calls) to tens of minutes (Groq, whose
output-token rate limit is the bottleneck), and the Documents page shows live progress while it runs. `data/graphs/`, `models/`, `eval/results.json` and `.env` are not committed.

##### The app

| Page | Use |
|---|---|
| **Ask** `/` | Ask a question; filter by result type; optional **Knowledge graph** toggle. Grounded answer with citations, source cards linking to the full passage, "via graph" tags on passages the graph contributed. |
| **Documents** `/documents` | Upload files, paste text, ingest `DOCS_DIR`, list and delete documents, with live ingestion progress. Re-ingesting identical text is skipped. |
| **Dashboard** `/dashboard` | Run the retrieval evaluation and chart it (Plotly): Recall@k, MRR, nDCG@k per retriever, Recall-vs-k curve, per-question ranks with CSV export, index statistics. |
| **Questions** `/questions` | Manage the labelled evaluation questions: add, edit, delete, train/test, import/export JSON, and see which section each question's evidence is found in. |
| **Knowledge graph** `/graph` | Search an entity, see it and its neighbours as a network, read its triplets with the source passage of each. |
| **Quality checks** `/quality` | Grade the answers (faithfulness, relevance, abstention) and check a sample of the generated Q&A pairs and factoids against their source chunks; mark each correct or incorrect and read the error rates. |
| **User guide** `/guide` | Illustrated usage documentation with a **Download as PDF** button (`frontend/public/user-guide.pdf`, also copied to `Capstone_User_Guide.pdf`; rebuild the images with `scripts/screenshots.sh`, then the PDF with `scripts/guide_pdf.sh`). |
| **API docs** `/api-docs` | The API reference (also at `:8000/docs`). |

##### Contact and Further Information

- Repository: https://github.com/albertpinto/MLAI-UCB-Capstone (GitHub: [albertpinto](https://github.com/albertpinto))
- Usage and troubleshooting: the User guide page in the app, or [Capstone_User_Guide.pdf](Capstone_User_Guide.pdf)
- Questions or problems: open an issue on the repository
