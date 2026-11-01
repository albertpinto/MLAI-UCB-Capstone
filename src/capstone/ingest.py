import hashlib
import logging
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass

from .chunking import chunk_text, clean_text
from .config import Settings
from .store import Store

logger = logging.getLogger(__name__)

QA_SYSTEM = (
    "You write question-answer pairs for a knowledge base. Write 3 to 5 questions the way a user "
    "would ask them in everyday language, each answered only from the text. "
    'Reply as JSON: {"qa_pairs": [{"question": "...", "answer": "..."}]}'
)
DOC_SYSTEM = (
    "You analyse a document. Extract 5 to 10 key topics (a short term and a one-sentence "
    "definition) and write a 3 to 5 sentence summary. "
    'Reply as JSON: {"topics": [{"term": "...", "definition": "..."}], "summary": "..."}'
)
DOC_INPUT_CHARS = 12000


@dataclass
class IngestResult:
    name: str
    status: str = "queued"  # queued | running | processed | skipped | failed
    document_id: str | None = None
    chunks: int = 0
    qa_pairs: int = 0
    topics: int = 0
    error: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def _qa_pairs(llm, chunk: str) -> list[dict]:
    try:
        pairs = llm.json_call(QA_SYSTEM, f"Text:\n{chunk}").get("qa_pairs", [])
    except Exception as exc:  # one bad chunk must not abort the document
        logger.error("Q&A generation failed: %s", exc)
        return []
    return [p for p in pairs if isinstance(p, dict) and isinstance(p.get("question"), str) and isinstance(p.get("answer"), str)]


def _topics_and_summary(llm, text: str) -> tuple[list[dict], str]:
    try:
        result = llm.json_call(DOC_SYSTEM, f"Document:\n{text[:DOC_INPUT_CHARS]}")
    except Exception as exc:
        logger.error("Topic/summary generation failed: %s", exc)
        return [], ""
    topics = [t for t in result.get("topics", []) if isinstance(t, dict) and t.get("term") and t.get("definition")]
    summary = result.get("summary") if isinstance(result.get("summary"), str) else ""
    return topics, summary


def _topics_in(text: str, terms: list[str]) -> list[str]:
    lowered = text.lower()
    return [term for term in terms if term.lower() in lowered]


def ingest_text(store: Store, llm, settings: Settings, title: str, text: str, result: IngestResult | None = None) -> IngestResult:
    """Chunk, enrich with the LLM, embed and store one document. Skips text already stored."""
    result = result or IngestResult(name=title)
    result.status = "running"
    try:
        text = clean_text(text)
        if not text:
            raise ValueError("Document has no text")
        file_hash = hashlib.sha256(text.encode()).hexdigest()
        if store.has_hash(file_hash):
            result.status = "skipped"
            return result

        document_id = str(uuid.uuid4())
        base = {"document_id": document_id, "title": title, "citation": title, "file_hash": file_hash}
        chunks = chunk_text(text, settings.chunk_chars, settings.chunk_overlap)

        with ThreadPoolExecutor(settings.llm_workers) as pool:
            qa_per_chunk = list(pool.map(lambda c: _qa_pairs(llm, c), chunks))
        topics, summary = _topics_and_summary(llm, text)
        terms = [t["term"] for t in topics]

        items = []
        for chunk_id, (chunk, pairs) in enumerate(zip(chunks, qa_per_chunk)):
            chunk_topics = _topics_in(chunk, terms)
            items.append((chunk, {**base, "type": "chunk", "chunk_id": chunk_id, "content": chunk, "topics": chunk_topics}))
            for pair in pairs:
                qa_topics = sorted(set(chunk_topics) | set(_topics_in(pair["question"] + " " + pair["answer"], terms)))
                items.append((
                    f"Question: {pair['question']} Answer: {pair['answer']}",
                    {**base, "type": "qa_pair", "chunk_id": chunk_id, "question": pair["question"], "answer": pair["answer"], "topics": qa_topics},
                ))
        for topic in topics:
            items.append((f"{topic['term']}: {topic['definition']}", {**base, "type": "topic", "term": topic["term"], "definition": topic["definition"], "topics": [topic["term"]]}))
        if summary:
            items.append((summary, {**base, "type": "summary", "content": summary, "topics": _topics_in(summary, terms)}))

        store.add(items)
        result.status = "processed"
        result.document_id = document_id
        result.chunks = len(chunks)
        result.qa_pairs = sum(len(p) for p in qa_per_chunk)
        result.topics = len(topics)
    except Exception as exc:
        logger.exception("Ingest failed for %s", title)
        result.status = "failed"
        result.error = str(exc)
    return result
