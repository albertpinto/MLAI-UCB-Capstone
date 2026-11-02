import json
import logging
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
from .evaluation import RETRIEVERS, available_retrievers, build_scorers, load_corpus, load_questions, run_evaluation
from .schemas import AskRequest, EvalQuestions, EvalRun, SearchRequest, TextDocument
from .store import Embedder, Store

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def create_app(settings: Settings | None = None, store: Store | None = None, llm=None, embedder_factory=None) -> FastAPI:
    """Build the app. Store, LLM and the evaluation's embedders are created once at startup unless injected (tests do)."""
    settings = settings or get_settings()
    jobs: dict[str, dict] = {}
    ingest_lock = threading.Lock()  # one ingestion at a time: the embedding model is shared
    eval_jobs: dict[str, dict] = {}
    embedders: dict[str, object] = {}

    def embedder_for(name: str):
        if name not in embedders:
            embedders[name] = embedder_factory(name) if embedder_factory else Embedder(name, settings.embed_device, settings.embed_threads)
        return embedders[name]

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.store = store or Store(settings.qdrant_url, settings.collection, Embedder(settings.embed_model, settings.embed_device, settings.embed_threads))
        app.state.llm = llm or LLM(settings.groq_api_key, settings.groq_model)
        yield

    app = FastAPI(
        title="Capstone RAG API",
        version="0.1.0",
        description="Semantic retrieval over your documents: ingest documents, search them in everyday language, manage what is stored.",
        openapi_url="/api/openapi.json",  # under /api so the UI's dev proxy can reach it
        lifespan=lifespan,
    )

    def run_job(job_id: str, docs: list[tuple[IngestResult, str, bytes | str]]) -> None:
        """docs: (result row, title, raw bytes to extract from, or text). Parsing the next file overlaps the LLM work on the current one."""
        def parse(title: str, content: bytes | str):
            parents = None if isinstance(content, str) else parents_from_file(title, content, settings)
            return parents, content if parents is None else "\n\n".join(p.text for p in parents)

        with ingest_lock, ThreadPoolExecutor(1) as parser:
            parsed = [parser.submit(parse, title, content) for _, title, content in docs]
            for (result, title, _), future in zip(docs, parsed):
                try:
                    parents, text = future.result()
                except Exception as exc:
                    result.status, result.error = "failed", str(exc)
                    continue
                ingest_text(app.state.store, app.state.llm, settings, title, text, result, parents)
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
        """Every document currently stored, with its id and title."""
        return {"documents": app.state.store.documents()}

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
        return {"deleted": document_id}

    @app.post("/api/search", tags=["Search"], summary="Semantic search", responses={200: {"content": {"application/json": {"example": {"answer": {"text": "Q: Who do I contact…?\nA: support@example.com", "type": "qa_pair", "citation": "Post Failures.txt", "score": 0.41}, "results": ["… same shape plus document_id, file_hash, topics …"]}}}}})
    def search(req: SearchRequest):
        """Finds the most relevant stored content for a question written in everyday language.

        The query is embedded and matched against all stored points by cosine similarity. Optional `types` (`chunk`, `qa_pair`, `topic`, `summary`) and `topics` filters narrow the search.
        `answer` is the best-ranked Q&A pair or summary (or the best chunk if none), meant to be shown first; `results` is the full ranked list with similarity scores.
        `answer` is `null` when nothing is stored. Returns 400 for a blank query.
        """
        if not req.query.strip():
            raise HTTPException(400, "Query is required")
        results = app.state.store.search(req.query, req.limit, req.types, req.topics)
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
        results = app.state.store.with_parents(app.state.store.search(req.query, settings.answer_results, req.types, req.topics))
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

    @app.post("/api/eval/run", tags=["Evaluation"], summary="Run the retrieval evaluation", responses={200: {"content": {"application/json": {"example": {"job_id": "9f0c1c0e-…"}}}}})
    def eval_run(body: EvalRun, background: BackgroundTasks):
        """Starts a background job that scores the chosen retrievers (BM25, the un-tuned MiniLM, and the fine-tuned model when `FINE_TUNED_MODEL` is set) on the labelled questions, using Recall@k, MRR and nDCG@k at section level. Poll `GET /api/eval/jobs/{id}`; the finished result is stored and returned by `GET /api/eval/results`. Returns 400 for an unknown retriever or when no questions exist."""
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
                scorers = build_scorers(settings, corpus, names, embedder_for)
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

    return app


app = create_app()
