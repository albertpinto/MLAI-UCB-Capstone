import hashlib
import logging
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass

from .chunking import clean_text
from .config import Settings
from .enrich import EnrichmentCache, enrich_child
from .parsing import Parent, parents_from_text, split_children
from .store import Store

logger = logging.getLogger(__name__)

DOC_SYSTEM = (
    "You analyse a document. Extract 5 to 10 key topics (a short term and a one-sentence "
    "definition) and write a 3 to 5 sentence summary. "
    'Reply as JSON: {"topics": [{"term": "...", "definition": "..."}], "summary": "..."}'
)
DOC_INPUT_CHARS = 12000
DOC_SAMPLES = 4  # a longer document is shown to the LLM as this many slices spread over the whole text, not just its first pages


@dataclass
class IngestResult:
    name: str
    status: str = "queued"  # queued | running | processed | skipped | failed
    document_id: str | None = None
    version: int = 1  # 2 or more: this replaced an earlier version of the document with the same title
    parents: int = 0
    chunks: int = 0
    qa_pairs: int = 0
    factoids: int = 0
    triplets: int = 0
    topics: int = 0
    error: str | None = None
    stage: str = ""  # what it is doing now: reading | splitting | enriching | summarising | storing
    enriched: int = 0  # chunks the LLM has finished, out of `chunks`; the counts above grow while this runs

    def to_dict(self) -> dict:
        return asdict(self)


def _sample(text: str) -> str:
    """The whole text if it fits, else DOC_SAMPLES slices spread evenly from start to end."""
    if len(text) <= DOC_INPUT_CHARS:
        return text
    size = DOC_INPUT_CHARS // DOC_SAMPLES
    starts = [(len(text) - size) * i // (DOC_SAMPLES - 1) for i in range(DOC_SAMPLES)]
    return "\n[...]\n".join(text[s:s + size] for s in starts)


def _topics_and_summary(llm, text: str) -> tuple[list[dict], str]:
    try:
        result = llm.json_call(DOC_SYSTEM, f"Document:\n{_sample(text)}")
    except Exception as exc:
        logger.error("Topic/summary generation failed: %s", exc)
        return [], ""
    topics = [t for t in result.get("topics", []) if isinstance(t, dict) and t.get("term") and t.get("definition")]
    summary = result.get("summary") if isinstance(result.get("summary"), str) else ""
    return topics, summary


def _topics_in(text: str, terms: list[str]) -> list[str]:
    lowered = text.lower()
    return [term for term in terms if term.lower() in lowered]


def ingest_text(store: Store, llm, settings: Settings, title: str, text: str, result: IngestResult | None = None, parents: list[Parent] | None = None) -> IngestResult:
    """Chunk (parents, then children), enrich with the LLM, embed and store one document. Skips text already stored.

    A document with the same title but different text is an update: the new version is built first, then the old one is
    deleted and the new one stored with the next version number, so a failed update leaves the old version in place.

    `parents` come from the document layout when the caller has it (PDF, DOCX); otherwise they are cut from `text`.
    """
    result = result or IngestResult(name=title)
    result.status = "running"
    try:
        text = clean_text(text)
        if not text:
            raise ValueError("Document has no text")
        file_hash = hashlib.sha256(text.encode()).hexdigest()
        if store.has_hash(file_hash):
            result.status, result.stage = "skipped", ""
            return result

        document_id = str(uuid.uuid4())
        now = time.time()
        old = store.document_by_title(title)
        version = old["version"] + 1 if old else 1
        expires_at = now + settings.document_ttl_days * 86400 if settings.document_ttl_days else None  # None = never
        base = {"document_id": document_id, "title": title, "citation": title, "file_hash": file_hash, "version": version, "ingested_at": now, "expires_at": expires_at}
        result.stage = "splitting"
        parents = parents or parents_from_text(text, settings.parent_chars)
        children = [(pid, parent, child) for pid, parent in enumerate(parents) for child in split_children(parent.text, settings)]
        result.parents, result.chunks = len(parents), len(children)

        cache = EnrichmentCache(settings.enrich_cache) if settings.enrich_cache else None
        result.stage = "enriching"
        tally = threading.Lock()

        def enrich(c):
            e = enrich_child(llm, settings, c[1], c[2], cache)
            with tally:  # the UI reads these while the job runs
                result.enriched += 1
                result.qa_pairs += len(e.qa_pairs)
                result.factoids += len(e.factoids)
                result.triplets += len(e.triplets)
            return e

        with ThreadPoolExecutor(settings.llm_workers) as pool:
            enriched = list(pool.map(enrich, children))
        result.stage = "summarising"
        topics, summary = _topics_and_summary(llm, text)
        result.topics = len(topics)
        result.stage = "storing"
        terms = [t["term"] for t in topics]

        items = []
        for chunk_id, ((pid, parent, _), e) in enumerate(zip(children, enriched)):
            chunk_topics = _topics_in(e.content, terms)
            link = {"parent_id": pid, "chunk_id": chunk_id, "section": parent.section, "page": parent.page}
            heading = f"{parent.section}. " if parent.section else ""  # the section name helps a short chunk match
            items.append((
                heading + e.content,
                {**base, **link, "type": "chunk", "content": e.content, "original": e.original, "parent_text": parent.text, "parents_total": len(parents), "topics": chunk_topics},
            ))
            for pair in e.qa_pairs:
                topics_here = sorted(set(chunk_topics) | set(_topics_in(pair["question"] + " " + pair["answer"], terms)))
                items.append((f"Question: {pair['question']} Answer: {pair['answer']}", {**base, **link, "type": "qa_pair", "question": pair["question"], "answer": pair["answer"], "topics": topics_here}))
            for fact in e.factoids:
                items.append((fact, {**base, **link, "type": "factoid", "content": fact, "topics": sorted(set(chunk_topics) | set(_topics_in(fact, terms)))}))
            for t in e.triplets:
                items.append((f"{t['subject']} {t['relation']} {t['object']}", {**base, **link, "type": "triplet", **t, "topics": chunk_topics}))
        for topic in topics:
            items.append((f"{topic['term']}: {topic['definition']}", {**base, "type": "topic", "term": topic["term"], "definition": topic["definition"], "topics": [topic["term"]]}))
        if summary:
            items.append((summary, {**base, "type": "summary", "content": summary, "topics": _topics_in(summary, terms)}))

        if old:
            store.delete_document(old["document_id"])
        store.add(items)
        result.status, result.stage = "processed", "done"
        result.document_id = document_id
        result.version = version
        result.parents = len(parents)
        result.chunks = len(children)
        result.qa_pairs = sum(len(e.qa_pairs) for e in enriched)
        result.factoids = sum(len(e.factoids) for e in enriched)
        result.triplets = sum(len(e.triplets) for e in enriched)
        result.topics = len(topics)
    except Exception as exc:
        logger.exception("Ingest failed for %s", title)
        result.status = "failed"
        result.error = str(exc)
    return result
