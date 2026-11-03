import asyncio
import json
import logging
import time
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

from fastapi import BackgroundTasks, FastAPI, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from .config import Settings, get_settings
from .extract import SUPPORTED
from .answer import answer_question
from .ingest import IngestResult, ingest_text
from .llm import LLM
from .parsing import parents_from_file
from . import graph as kg
from .evaluation import RETRIEVERS, default_reranker, available_retrievers, build_scorers, load_corpus, load_questions, relevant_sections, run_evaluation
from . import quality
from .schemas import AnswerRun, AskRequest, EvalQuestions, EvalRun, QualityRun, Review, SearchRequest, TextDocument
from .store import Embedder, Store

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def create_app(settings: Settings | None = None, store: Store | None = None, llm=None, embedder_factory=None, reranker_factory=None) -> FastAPI:
    """Build the app. Store, LLM and the evaluation's embedders are created once at startup unless injected (tests do)."""
    settings = settings or get_settings()
    jobs: dict[str, dict] = {}
    ingest_lock = threading.Lock()  # one ingestion at a time: the embedding model is shared
    eval_jobs: dict[str, dict] = {}
    quality_jobs: dict[str, dict] = {}
    quality_lock = threading.Lock()  # the quality file is read, changed and written back by jobs and by reviews
    embedders: dict[str, object] = {}
    aliases = kg.load_aliases(settings.graph_aliases_file)

    def embedder_for(name: str):
        if name not in embedders:
            embedders[name] = embedder_factory(name) if embedder_factory else Embedder(name, settings.embed_device, settings.embed_threads)
        return embedders[name]

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if store is None:
            reranker = (reranker_factory or default_reranker)(settings.rerank_model) if settings.rerank_search and settings.rerank_model else None
            store_ = Store(settings.qdrant_path or settings.qdrant_url, settings.collection, Embedder(settings.embed_model, settings.embed_device, settings.embed_threads), reranker, settings.rerank_candidates)
        app.state.store = store or store_
        app.state.llm = llm or LLM(*(settings.openai_api_key, settings.openai_model) if settings.llm_provider == "openai" else (settings.groq_api_key, settings.groq_model), settings.llm_max_concurrent, settings.llm_min_interval, settings.llm_provider)
        app.state.graph = kg.load(settings.graph_path)
        flusher = asyncio.create_task(flush_loop()) if settings.document_ttl_days else None
        yield
        if flusher:
            flusher.cancel()

    app = FastAPI(
        title="Capstone RAG API",
        version="0.1.0",
        description="Semantic retrieval over your documents: ingest documents, search them in everyday language, manage what is stored.",
        openapi_url="/api/openapi.json",  # under /api so the UI's dev proxy can reach it
        lifespan=lifespan,
    )

    def rebuild_graph() -> dict:
        """Build the graph from the stored triplets, save it, and store one searchable summary point per document."""
        store_ = app.state.store
        graph = kg.build(store_.all_payloads("triplet"), aliases)
        kg.save(graph, settings.graph_path)
        app.state.graph = graph
        summaries = [(text, {"document_id": d["document_id"], "title": d["title"], "citation": d["citation"], "type": "graph_summary", "content": text, "topics": []})
                     for d in store_.documents() if (text := kg.summary_text(graph, d["title"], d["document_id"]))]
        store_.replace_graph_summaries(summaries)
        return kg.stats(graph)

    def flush_expired() -> list[dict]:
        """Delete every document whose time-to-live has run out (the nightly flush), then rebuild the graph without them."""
        with ingest_lock:
            gone = app.state.store.expired(time.time())
            for d in gone:
                app.state.store.delete_document(d["document_id"])
            if gone and app.state.graph is not None:
                rebuild_graph()
        return gone

    async def flush_loop() -> None:
        while True:
            try:
                for d in await asyncio.to_thread(flush_expired):
                    logging.getLogger(__name__).info("Flushed expired document %s (version %s)", d["title"], d["version"])
            except Exception:
                logging.getLogger(__name__).exception("Flushing expired documents failed")
            await asyncio.sleep(settings.flush_interval_hours * 3600)

    def run_job(job_id: str, docs: list[tuple[IngestResult, str, bytes | str]]) -> None:
        """docs: (result row, title, raw bytes to extract from, or text). Parsing the next file overlaps the LLM work on the current one."""
        def parse(title: str, content: bytes | str):
            parents = None if isinstance(content, str) else parents_from_file(title, content, settings)
            return parents, content if parents is None else "\n\n".join(p.text for p in parents)

        with ingest_lock, ThreadPoolExecutor(1) as parser:
            parsed = [parser.submit(parse, title, content) for _, title, content in docs]
            for (result, title, _), future in zip(docs, parsed):
                result.status, result.stage = "running", "reading"
                try:
                    parents, text = future.result()
                except Exception as exc:
                    result.status, result.error = "failed", str(exc)
                    continue
                ingest_text(app.state.store, app.state.llm, settings, title, text, result, parents)
            if any(r.status == "processed" for r, _, _ in docs):
                try:
                    rebuild_graph()
                except Exception:
                    logging.getLogger(__name__).exception("Rebuilding the knowledge graph failed")
        jobs[job_id]["status"] = "done"

    def start_job(background: BackgroundTasks, docs: list[tuple[str, bytes | str]]) -> dict:
        rows = [(IngestResult(name=title), title, content) for title, content in docs]
        job_id = str(uuid.uuid4())
        jobs[job_id] = {"id": job_id, "status": "running", "items": [r for r, _, _ in rows]}
        background.add_task(run_job, job_id, rows)
        return {"job_id": job_id}

    @app.get("/api/health", tags=["System"], summary="Health check", responses={200: {"content": {"application/json": {"example": {"status": "ok"}}}}})
    def health():
        """Returns `ok` while the API is running. The UI polls this to show the API status light in the header."""
        return {"status": "ok"}

    @app.get("/api/branding", tags=["System"], summary="Company branding", responses={200: {"content": {"application/json": {"example": {"name": "Bakasura", "tagline": "Ask your documents anything", "logo_url": "/logo.svg"}}}}})
    def branding():
        """Company name, tagline and logo URL shown in the UI header. Set with `COMPANY_NAME`, `COMPANY_TAGLINE` and `LOGO_URL` in `.env`."""
        return {"name": settings.company_name, "tagline": settings.company_tagline, "logo_url": settings.logo_url}

    @app.get("/api/suggestions", tags=["Search"], summary="Suggested questions", responses={200: {"content": {"application/json": {"example": {
        "suggestions": [{"topic": "The President", "question": "How is the President of India elected?"}]}}}}})
    def suggestions():
        """A pick list of example questions for the Ask page, read from the JSON file at `SUGGESTED_QUESTIONS_FILE`
        (a list of `{"topic", "question"}`). Returns an empty list if the file does not exist.
        """
        path = settings.suggested_questions_file
        return {"suggestions": json.loads(path.read_text()) if path.is_file() else []}

    @app.post("/api/ingest/files", tags=["Ingest"], summary="Upload documents", responses={200: {"content": {"application/json": {"example": {"job_id": "b59a6883-bef9-4a34-9cbd-f1a76b447e50"}}}}})
    async def ingest_files(files: list[UploadFile], background: BackgroundTasks):
        """Upload one or more `.pdf`, `.docx` or `.txt` files as multipart form data (field name `files`).

        Each file is cleaned, split into chunks, and enriched by the LLM with Q&A pairs, key topics and a summary, then embedded and stored.
        Ingestion runs in the background: the call returns a `job_id` immediately; poll `GET /api/jobs/{job_id}` for progress.
        Text already stored (same content hash) is skipped. Any unsupported file type rejects the whole request with 400.
        """
        docs = []
        for f in files:
            if not f.filename or "." + f.filename.rsplit(".", 1)[-1].lower() not in SUPPORTED:
                raise HTTPException(400, f"Unsupported file: {f.filename}. Use {', '.join(sorted(SUPPORTED))}")
            docs.append((f.filename, await f.read()))
        return start_job(background, docs)

    @app.post("/api/ingest/folder", tags=["Ingest"], summary="Ingest the docs folder", responses={200: {"content": {"application/json": {"example": {"job_id": "b59a6883-bef9-4a34-9cbd-f1a76b447e50"}}}}})
    def ingest_folder(background: BackgroundTasks):
        """Ingests every supported file in the server's `DOCS_DIR` folder, as a background job. Use it to load a whole corpus at once.

        Returns 400 if the folder does not exist or holds no `.pdf`, `.docx` or `.txt` files.
        """
        folder = settings.docs_dir
        if not folder.is_dir():
            raise HTTPException(400, f"DOCS_DIR does not exist: {folder.resolve()}")
        paths = sorted(p for p in folder.iterdir() if p.suffix.lower() in SUPPORTED)
        if not paths:
            raise HTTPException(400, f"No {', '.join(sorted(SUPPORTED))} files in {folder.resolve()}")
        return start_job(background, [(p.name, p.read_bytes()) for p in paths])

    @app.post("/api/ingest/text", tags=["Ingest"], summary="Add a document from pasted text", responses={200: {"content": {"application/json": {"example": {"job_id": "b59a6883-bef9-4a34-9cbd-f1a76b447e50"}}}}})
    def ingest_manual(doc: TextDocument, background: BackgroundTasks):
        """Adds a single document from text sent as JSON. The `title` becomes the citation shown with search results.
        Runs as a background job, like the file uploads. Returns 400 if the title or text is blank.
        """
        if not doc.title.strip() or not doc.text.strip():
            raise HTTPException(400, "Title and text are required")
        return start_job(background, [(doc.title.strip(), doc.text)])

    @app.get("/api/jobs/{job_id}", tags=["Ingest"], summary="Check an ingestion job", responses={200: {"content": {"application/json": {"example": {"id": "b59a6883-…", "status": "done", "items": [{"name": "Billing.txt", "status": "processed", "document_id": "56a545b7-…", "chunks": 4, "qa_pairs": 17, "topics": 5, "error": None}]}}}}})
    def get_job(job_id: str):
        """Progress of an ingestion job. `status` is `running` until every document is finished, then `done`.
        Each item reports its own status (`queued`, `running`, `processed`, `skipped` or `failed`) with counts of chunks, Q&A pairs and topics created.
        Returns 404 for an unknown job id. Jobs are kept in memory, so they are lost when the API restarts.
        """
        job = jobs.get(job_id)
        if not job:
            raise HTTPException(404, "Unknown job")
        return {"id": job["id"], "status": job["status"], "items": [i.to_dict() for i in job["items"]]}

    @app.get("/api/documents", tags=["Documents"], summary="List stored documents", responses={200: {"content": {"application/json": {"example": {"documents": [{"document_id": "56a545b7-…", "title": "Billing.txt"}]}}}}})
    def list_documents():
        """Every document currently stored, with its id, title, `version` (1 for the first copy; storing changed text under the same title replaces the document with version + 1), and `ingested_at` / `expires_at` (Unix seconds; `expires_at` is null when it never expires)."""
        return {"documents": app.state.store.documents()}

    @app.post("/api/documents/flush", tags=["Documents"], summary="Delete expired documents", responses={200: {"content": {"application/json": {"example": {"flushed": [{"document_id": "56a545b7-…", "title": "Old policy.txt", "version": 1}]}}}}})
    def flush_documents():
        """Deletes every document whose time-to-live (`DOCUMENT_TTL_DAYS`, counted from when it was stored) has run out, with everything derived from it. The API also does this every `FLUSH_INTERVAL_HOURS` while a TTL is set. Documents stored without an expiry are never flushed."""
        return {"flushed": [{k: d[k] for k in ("document_id", "title", "version")} for d in flush_expired()]}

    @app.get("/api/documents/{document_id}/passages/{parent_id}", tags=["Documents"], summary="Read one passage", responses={200: {"content": {"application/json": {"example": {
        "document_id": "7cd5ca08-…", "title": "Constitution of India (2024).pdf", "citation": "Constitution of India (2024).pdf", "section": "Part III › Right to Freedom",
        "page": 41, "parent_id": 120, "total": 640, "text": "32. Remedies for enforcement of rights conferred by this Part.—(1) The right to move the Supreme Court…"}}}}})
    def get_passage(document_id: str, parent_id: int):
        """The original text of one passage (a parent chunk: a section of about half a page), with its heading path and page, and how many passages the document has.
        Search results and answer sources link here; the small chunk that matched is part of this passage. `parent_id` counts from 0, so use ±1 for the neighbours.
        Returns 404 if there is no such passage.
        """
        found = app.state.store.passage(document_id, parent_id)
        if not found:
            raise HTTPException(404, "Unknown passage")
        return found

    @app.delete("/api/documents/{document_id}", tags=["Documents"], summary="Delete a document", responses={200: {"content": {"application/json": {"example": {"deleted": "56a545b7-…"}}}}})
    def delete_document(document_id: str):
        """Permanently deletes a document and everything derived from it: its chunks, Q&A pairs, topics and summary. This cannot be undone."""
        app.state.store.delete_document(document_id)
        if app.state.graph is not None:
            rebuild_graph()
        return {"deleted": document_id}

    def graph_for(requested: bool | None) -> tuple:
        """(graph, aliases) for a search, or (None, None) when graph-assisted retrieval is off or no graph is built."""
        use = settings.graph_search if requested is None else requested
        return (app.state.graph, aliases) if use and app.state.graph is not None else (None, None)

    @app.post("/api/search", tags=["Search"], summary="Semantic search", responses={200: {"content": {"application/json": {"example": {"answer": {"text": "Q: Who do I contact…?\nA: support@example.com", "type": "qa_pair", "citation": "Post Failures.txt", "score": 0.41}, "results": ["… same shape plus document_id, file_hash, topics …"]}}}}})
    def search(req: SearchRequest):
        """Finds the most relevant stored content for a question written in everyday language.

        The query is embedded and matched against all stored points by cosine similarity. Optional `types` (`chunk`, `qa_pair`, `topic`, `summary`) and `topics` filters narrow the search.
        `answer` is the best-ranked Q&A pair or summary (or the best chunk if none), meant to be shown first; `results` is the full ranked list with similarity scores.
        `answer` is `null` when nothing is stored. Returns 400 for a blank query.
        """
        if not req.query.strip():
            raise HTTPException(400, "Query is required")
        results = app.state.store.search(req.query, req.limit, req.types, req.topics, *graph_for(req.graph))
        # Immediate answer: the best-ranked Q&A pair or summary, else the best chunk.
        best = next((r for r in results if r["type"] in ("qa_pair", "summary")), None) or next(
            (r for r in results if r["type"] == "chunk"), None
        )
        answer = {k: best[k] for k in ("text", "type", "citation", "score")} if best else None
        return {"answer": answer, "results": results}

    @app.post("/api/ask", tags=["Search"], summary="Ask a question (grounded answer)", responses={200: {"content": {"application/json": {"example": {
        "answer": "Check that your account is connected and the post is not scheduled for later [1].",
        "abstained": False,
        "sources": [{"n": 1, "text": "Q: Why did my post fail?…", "type": "qa_pair", "citation": "Post Failures.txt", "score": 0.52}],
        "matches": [{"n": 2, "answer": "Posts also fail when the media file is too large.", "type": "chunk", "citation": "Post Failures.txt", "score": 0.48}]}}}}})
    def ask(req: AskRequest):
        """Answers a question in natural language, using only your documents.

        Retrieval runs first (same as `POST /api/search`); the best passages are then given to the LLM as numbered context, and it writes a short answer citing them as `[1]`, `[2]`.
        `sources` lists the cited passages with their document. `matches` are up to 10 other retrieved passages, each rewritten by the LLM as a short readable answer (passages that do not help are left out). If nothing relevant is retrieved (every score below `ANSWER_MIN_SCORE`), or the passages do not contain the answer,
        `abstained` is `true` and the answer says so instead of guessing. Returns 400 for a blank query and 500 if the LLM call fails.
        """
        if not req.query.strip():
            raise HTTPException(400, "Query is required")
        results = app.state.store.with_parents(app.state.store.search(req.query, settings.answer_results, req.types, req.topics, *graph_for(req.graph)))
        return answer_question(app.state.llm, req.query, results, settings.answer_min_score)

    @app.get("/api/topics", tags=["Search"], summary="List topics", responses={200: {"content": {"application/json": {"example": {"topics": ["Scheduling", "Billing"]}}}}})
    def topics():
        """The glossary of key topics the LLM extracted from all documents. Pass these names in the `topics` filter of `POST /api/search`."""
        return {"topics": app.state.store.topics()}

    @app.get("/api/stats", tags=["Evaluation"], summary="Index statistics", responses={200: {"content": {"application/json": {"example": {
        "points": {"chunk": 2362, "qa_pair": 4506, "factoid": 5975, "triplet": 5435, "topic": 7, "summary": 1}, "documents": 1, "parents": 547,
        "chunks_per_part": [{"part": "Part III – Fundamental Rights", "chunks": 120}], "child_words": [{"from": 20, "to": 29, "count": 410}]}}}}})
    def stats():
        """Numbers behind the dashboard's index charts: how many points of each type are stored, how many chunks each Part of the document has, and how big the child chunks are (words, in buckets of 10)."""
        return app.state.store.stats()

    @app.get("/api/eval/questions", tags=["Evaluation"], summary="List labelled questions", responses={200: {"content": {"application/json": {"example": {"questions": [
        {"id": "q-article-21", "question": "Can the government take away my life?", "evidence": ["procedure established by law"], "split": "test", "note": ""}], "retrievers": [{"id": "bm25", "label": "BM25"}]}}}}})
    def eval_questions():
        """The labelled evaluation questions from `EVAL_QUESTIONS_FILE`. Each carries `evidence`: phrases copied from the text that answers it. A section that contains one of them is the right answer, so labels still work after the document is re-chunked. `retrievers` lists what `POST /api/eval/run` can score (the fine-tuned model appears once `FINE_TUNED_MODEL` is set)."""
        return {"questions": load_questions(settings.eval_questions_file), "retrievers": [{"id": n, "label": RETRIEVERS[n]} for n in available_retrievers(settings)]}

    @app.put("/api/eval/questions", tags=["Evaluation"], summary="Replace the labelled questions", responses={200: {"content": {"application/json": {"example": {"saved": 60}}}}})
    def save_eval_questions(body: EvalQuestions):
        """Replaces the whole question list (this is also how you import a file). Ids must be unique. Mark each question `train` or `test`, splitting by Part of the document, so fine-tuning never sees a test question."""
        ids = [q.id for q in body.questions]
        if len(ids) != len(set(ids)):
            raise HTTPException(400, "Question ids must be unique")
        settings.eval_questions_file.parent.mkdir(parents=True, exist_ok=True)
        settings.eval_questions_file.write_text(json.dumps([q.model_dump() for q in body.questions], indent=1, ensure_ascii=False))
        return {"saved": len(ids)}

    @app.post("/api/eval/questions/check", tags=["Evaluation"], summary="Check where each question's evidence is found", responses={200: {"content": {"application/json": {"example": {"checks": {"q-article-21": ["Part III – Fundamental Rights › Right to Life"]}}}}}})
    def check_eval_questions(body: EvalQuestions):
        """Works on a draft, before it is saved. For each question, the sections of the index whose text contains one of its `evidence` phrases. An empty list means the evidence matches nothing, so the question would be left out of an evaluation."""
        corpus = load_corpus(app.state.store)
        return {"checks": {q.id: [corpus.sections[k]["section"] or k for k in sorted(relevant_sections(corpus, q.model_dump()))] for q in body.questions}}

    @app.post("/api/eval/run", tags=["Evaluation"], summary="Run the retrieval evaluation", responses={200: {"content": {"application/json": {"example": {"job_id": "9f0c1c0e-…"}}}}})
    def eval_run(body: EvalRun, background: BackgroundTasks):
        """Starts a background job that scores the chosen retrievers (BM25, the un-tuned MiniLM, the fine-tuned model when `FINE_TUNED_MODEL` is set, hybrid BM25 + dense with reciprocal rank fusion, and a cross-encoder reranker over the hybrid results) on the labelled questions, using Recall@k, MRR and nDCG@k at section level. Poll `GET /api/eval/jobs/{id}`; the finished result is stored and returned by `GET /api/eval/results`. Returns 400 for an unknown retriever or when no questions exist."""
        names = body.retrievers or available_retrievers(settings)
        unknown = [n for n in names if n not in RETRIEVERS or n not in available_retrievers(settings)]
        if unknown:
            raise HTTPException(400, f"Unknown or unavailable retriever: {', '.join(unknown)}")
        if not load_questions(settings.eval_questions_file):
            raise HTTPException(400, "No labelled questions yet")
        job_id = str(uuid.uuid4())
        eval_jobs[job_id] = {"id": job_id, "status": "running", "done": 0, "total": len(names), "message": "Loading the index", "error": None}

        def work():
            job = eval_jobs[job_id]
            try:
                corpus = load_corpus(app.state.store)
                job["message"] = "Embedding the chunks (first run only)"
                scorers = build_scorers(settings, corpus, names, embedder_for, reranker_factory or default_reranker)
                def progress(done, total, label):
                    job.update(done=done, total=total, message=f"Scored {label}")
                result = run_evaluation(corpus, load_questions(settings.eval_questions_file), scorers, body.k, body.split, progress)
                result["collection"] = settings.collection
                settings.eval_results_file.parent.mkdir(parents=True, exist_ok=True)
                settings.eval_results_file.write_text(json.dumps(result, ensure_ascii=False))
                job["status"] = "done"
            except Exception as exc:
                logging.getLogger(__name__).exception("Evaluation failed")
                job.update(status="failed", error=str(exc))

        background.add_task(work)
        return {"job_id": job_id}

    @app.get("/api/eval/jobs/{job_id}", tags=["Evaluation"], summary="Evaluation job progress", responses={200: {"content": {"application/json": {"example": {"id": "9f0c1c0e-…", "status": "running", "done": 1, "total": 2, "message": "Scored BM25", "error": None}}}}})
    def eval_job(job_id: str):
        """Progress of an evaluation started with `POST /api/eval/run`: `status` is `running`, `done` or `failed` (then `error` says why). Returns 404 for an unknown id."""
        if job_id not in eval_jobs:
            raise HTTPException(404, "Unknown job")
        return eval_jobs[job_id]

    @app.get("/api/eval/results", tags=["Evaluation"], summary="Last evaluation results", responses={200: {"content": {"application/json": {"example": {
        "k": 5, "split": "test", "questions": 40, "retrievers": {"bm25": {"label": "BM25", "recall": 0.55, "mrr": 0.41, "ndcg": 0.46, "curve": [0.3, 0.45]}},
        "per_question": [{"id": "q-article-21", "question": "…", "ranks": {"bm25": 2}, "top": {"bm25": [{"section": "Part III", "page": 42, "hit": True}]}}]}}}}})
    def eval_results():
        """The most recent finished evaluation, or `{}` if none has run yet. `retrievers` holds the metrics and the Recall@k curve (k = 1 to 20) for each retriever; `per_question` has the rank each retriever gave the right section (`null` = not found) and its top results."""
        path = settings.eval_results_file
        return json.loads(path.read_text()) if path.is_file() else {}

    @app.exception_handler(Exception)
    async def unhandled(_, exc: Exception):
        logging.getLogger(__name__).exception("Unhandled error")
        return JSONResponse(status_code=500, content={"detail": str(exc)})

    def quality_job(body_work) -> tuple[dict, callable]:
        """A job row, and the function that runs `body_work(job)` and records how it ended (pass it to BackgroundTasks)."""
        job_id = str(uuid.uuid4())
        job = quality_jobs[job_id] = {"id": job_id, "status": "running", "done": 0, "total": 0, "error": None}

        def work():
            try:
                body_work(job)
                job["status"] = "done"
            except Exception as exc:
                logging.getLogger(__name__).exception("Quality job failed")
                job.update(status="failed", error=str(exc))
        return job, work

    def progress_into(job):
        return lambda done, total: job.update(done=done, total=total)

    @app.post("/api/quality/artifacts/run", tags=["Quality"], summary="Check generated Q&A pairs and factoids", responses={200: {"content": {"application/json": {"example": {"job_id": "9f0c1c0e-…"}}}}})
    def quality_artifacts_run(body: QualityRun, background: BackgroundTasks):
        """Background job: a random sample of the generated items is checked by a second LLM pass against the chunk each came from (supported or not). Replaces the previous check but keeps a person's reviews of items that are checked again (same sample). Poll `GET /api/quality/jobs/{id}`."""
        def body_work(job):
            items = quality.check_artifacts(app.state.llm, app.state.store, body.kinds, body.sample, settings.llm_workers, progress=progress_into(job))
            with quality_lock:
                data = quality.load(settings.quality_file)
                kept = {i["id"]: i["human"] for i in data.get("artifacts", {}).get("items", [])}
                for i in items:
                    i["human"] = kept.get(i["id"])
                quality.save(settings.quality_file, {**data, "artifacts": {"checked_at": time.time(), "items": items}})
        job, work = quality_job(body_work)
        background.add_task(work)
        return {"job_id": job["id"]}

    @app.post("/api/quality/answers/run", tags=["Quality"], summary="Evaluate the answers", responses={200: {"content": {"application/json": {"example": {"job_id": "9f0c1c0e-…"}}}}})
    def quality_answers_run(body: AnswerRun, background: BackgroundTasks):
        """Background job: answers every labelled question of the split the way the Ask page does, and an LLM grades each answer against the context it was given (faithful? relevant?). Questions in `EVAL_UNANSWERABLE_FILE` should be answered with an abstention. Returns 400 when there are no labelled questions."""
        questions = [q for q in load_questions(settings.eval_questions_file) if body.split == "all" or q.get("split", "test") == body.split]
        if not questions:
            raise HTTPException(400, "No labelled questions for that split")
        unanswerable = json.loads(settings.eval_unanswerable_file.read_text()) if settings.eval_unanswerable_file.is_file() else []

        def retrieve(question):
            return app.state.store.with_parents(app.state.store.search(question, settings.answer_results, None, None, *graph_for(None)))

        def body_work(job):
            result = quality.evaluate_answers(app.state.llm, retrieve, questions, unanswerable, settings.answer_min_score, settings.llm_workers, progress_into(job))
            with quality_lock:
                quality.save(settings.quality_file, {**quality.load(settings.quality_file), "answers": result})
        job, work = quality_job(body_work)
        background.add_task(work)
        return {"job_id": job["id"]}

    @app.get("/api/quality/jobs/{job_id}", tags=["Quality"], summary="Quality job progress", responses={200: {"content": {"application/json": {"example": {"id": "9f0c1c0e-…", "status": "running", "done": 12, "total": 100, "error": None}}}}})
    def quality_job_status(job_id: str):
        """`status` is `running`, `done` or `failed` (then `error` says why). Returns 404 for an unknown id."""
        if job_id not in quality_jobs:
            raise HTTPException(404, "Unknown job")
        return quality_jobs[job_id]

    @app.get("/api/quality", tags=["Quality"], summary="Last quality results", responses={200: {"content": {"application/json": {"example": {
        "artifacts": {"checked_at": 1760000000, "summary": {"qa_pair": {"items": 50, "llm_error_rate": 0.04, "human_error_rate": None}}, "items": ["…"]},
        "answers": {"faithfulness": 0.96, "relevance": 0.93, "wrongly_abstained": 0.04, "correctly_abstained": 1.0, "rows": ["…"]}}}}}})
    def quality_results():
        """The last artifact check (each item with its source chunk, the LLM's verdict and any review by a person, plus error rates per kind) and the last answer evaluation. Empty parts are missing."""
        with quality_lock:
            data = quality.load(settings.quality_file)
        if "artifacts" in data:
            data["artifacts"]["summary"] = quality.artifact_summary(data["artifacts"]["items"])
        return data

    @app.put("/api/quality/artifacts/{item_id}/review", tags=["Quality"], summary="Mark a generated item correct or incorrect", responses={200: {"content": {"application/json": {"example": {"reviewed": "3f9a1c0d2b7e", "correct": False}}}}})
    def quality_review(item_id: str, body: Review):
        """Records a person's verdict on one checked item (`null` clears it). The error rate by review is the one to report; the LLM check only helps find the wrong ones. Returns 404 for an unknown item."""
        with quality_lock:
            if not quality.review(settings.quality_file, item_id, body.correct):
                raise HTTPException(404, "Unknown item")
        return {"reviewed": item_id, "correct": body.correct}

    @app.post("/api/graph/rebuild", tags=["Knowledge graph"], summary="Build the knowledge graph", responses={200: {"content": {"application/json": {"example": {"entities": 5212, "relations": 5416, "documents": 1}}}}})
    def graph_rebuild():
        """Builds the graph from every stored `triplet` point (entities are normalised so spellings merge), saves it to `GRAPH_DIR`, and stores one `graph_summary` point per document so the graph is searchable. It also runs after every ingestion and deletion. Returns the counts."""
        return rebuild_graph()

    @app.get("/api/graph/stats", tags=["Knowledge graph"], summary="Knowledge graph statistics", responses={200: {"content": {"application/json": {"example": {"built": True, "entities": 5212, "relations": 5416, "documents": 1,
        "top_entities": [{"id": "governor", "label": "Governor", "degree": 213}], "top_relations": [{"relation": "includes", "count": 155}]}}}}})
    def graph_stats():
        """Size of the graph, its most-connected entities and its most common relations. `{"built": false}` until the graph has been built."""
        return {"built": True, **kg.stats(app.state.graph)} if app.state.graph is not None else {"built": False}

    @app.get("/api/graph/entities", tags=["Knowledge graph"], summary="Find entities", responses={200: {"content": {"application/json": {"example": {"entities": [{"id": "governor", "label": "Governor", "degree": 213}]}}}}})
    def graph_entities(q: str):
        """Entities whose normalised name contains `q`, the most connected first."""
        return {"entities": kg.search_entities(app.state.graph, q) if app.state.graph is not None else []}

    @app.get("/api/graph/entity", tags=["Knowledge graph"], summary="One entity and its neighbours", responses={200: {"content": {"application/json": {"example": {
        "id": "governor", "label": "Governor", "nodes": [{"id": "governor", "label": "Governor", "degree": 213, "x": 0, "y": 0, "centre": True}], "edges": [{"source": "governor", "target": "chief minister", "relations": ["appoints"], "count": 1}],
        "neighbours_total": 120, "facts": [{"subject": "Governor", "relation": "appoints", "object": "Chief Minister", "section": "Part VI › Council of Ministers", "page": 120, "document_id": "7cd5ca08-…", "parent_id": 4, "text": "164. …"}]}}}}})
    def graph_entity(id: str):
        """The entity, up to 20 of its most-connected neighbours with the relations between them (positions are computed server-side for drawing), and its triplets with the section, page and text of the chunk each came from. Returns 404 for an unknown entity."""
        graph = app.state.graph
        if graph is None or id not in graph:
            raise HTTPException(404, "Unknown entity")
        rows = kg.facts(graph, id)
        chunks = app.state.store.chunk_payloads(list({(r["document_id"], r["chunk_id"]) for r in rows}))
        for r in rows:
            c = chunks.get((r["document_id"], r["chunk_id"]), {})
            r.update(section=c.get("section", ""), page=c.get("page"), text=c.get("content", ""))
        return {"id": id, "label": graph.nodes[id]["label"], **kg.neighbourhood(graph, id), "facts": rows}

    return app


app = create_app()

