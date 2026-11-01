import logging
import threading
import uuid
from contextlib import asynccontextmanager

from fastapi import BackgroundTasks, FastAPI, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from .config import Settings, get_settings
from .extract import SUPPORTED, extract_text
from .answer import answer_question
from .ingest import IngestResult, ingest_text
from .llm import LLM
from .schemas import AskRequest, SearchRequest, TextDocument
from .store import Embedder, Store

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def create_app(settings: Settings | None = None, store: Store | None = None, llm=None) -> FastAPI:
    """Build the app. Store and LLM are created once at startup unless injected (tests do)."""
    settings = settings or get_settings()
    jobs: dict[str, dict] = {}
    ingest_lock = threading.Lock()  # one ingestion at a time: the embedding model is shared

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
        """docs: (result row, title, raw bytes to extract from, or text)."""
        with ingest_lock:
            for result, title, content in docs:
                try:
                    text = content if isinstance(content, str) else extract_text(title, content)
                except Exception as exc:
                    result.status, result.error = "failed", str(exc)
                    continue
                ingest_text(app.state.store, app.state.llm, settings, title, text, result)
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

    @app.get("/api/branding", tags=["System"], summary="Company branding", responses={200: {"content": {"application/json": {"example": {"name": "Bakasura", "tagline": "Knowledge base", "logo_url": "/logo.svg"}}}}})
    def branding():
        """Company name, tagline and logo URL shown in the UI header. Set with `COMPANY_NAME`, `COMPANY_TAGLINE` and `LOGO_URL` in `.env`."""
        return {"name": settings.company_name, "tagline": settings.company_tagline, "logo_url": settings.logo_url}

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
        "sources": [{"n": 1, "text": "Q: Why did my post fail?…", "type": "qa_pair", "citation": "Post Failures.txt", "score": 0.52}]}}}}})
    def ask(req: AskRequest):
        """Answers a question in natural language, using only your documents.

        Retrieval runs first (same as `POST /api/search`); the best passages are then given to the LLM as numbered context, and it writes a short answer citing them as `[1]`, `[2]`.
        `sources` lists the cited passages with their document. If nothing relevant is retrieved (every score below `ANSWER_MIN_SCORE`), or the passages do not contain the answer,
        `abstained` is `true` and the answer says so instead of guessing. Returns 400 for a blank query and 500 if the LLM call fails.
        """
        if not req.query.strip():
            raise HTTPException(400, "Query is required")
        results = app.state.store.search(req.query, settings.answer_results, req.types, req.topics)
        return answer_question(app.state.llm, req.query, results, settings.answer_min_score)

    @app.get("/api/topics", tags=["Search"], summary="List topics", responses={200: {"content": {"application/json": {"example": {"topics": ["Scheduling", "Billing"]}}}}})
    def topics():
        """The glossary of key topics the LLM extracted from all documents. Pass these names in the `topics` filter of `POST /api/search`."""
        return {"topics": app.state.store.topics()}

    @app.exception_handler(Exception)
    async def unhandled(_, exc: Exception):
        logging.getLogger(__name__).exception("Unhandled error")
        return JSONResponse(status_code=500, content={"detail": str(exc)})

    return app


app = create_app()
