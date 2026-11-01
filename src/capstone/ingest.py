import hashlib
import logging
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass

from .chunking import clean_text
from .config import Settings
from .parsing import Parent, parents_from_text, split_children
from .store import Store

logger = logging.getLogger(__name__)

CHILD_SYSTEM = (
    "You prepare one small passage (CHILD) of a larger section (PARENT) for a search index. Reply as JSON: "
    '{"context_passage": "...", "qa_pairs": [{"question": "...", "answer": "..."}]}. '
    "context_passage: the CHILD text made to stand on its own. Keep its wording and facts, but replace vague references "
    "(it, this, they, the Act, the President, such person) with the thing they name, taking the name from the PARENT, and add at most one short "
    "opening sentence saying what the passage is about (for example the article or topic). Add nothing that is not in the PARENT. "
    "qa_pairs: 2 to 4 question-answer pairs a user would ask in everyday language, answered only from the CHILD text. "
    "Every question must be self-contained: name its subject, and never say 'the text', 'this passage', 'the document' or 'the section'."
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
    parents: int = 0
    chunks: int = 0
    qa_pairs: int = 0
    topics: int = 0
    error: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def _enrich_child(llm, settings: Settings, parent: Parent, child: str) -> tuple[str, list[dict]]:
    """One LLM call per child: (the child with its missing context restored, Q&A pairs). Falls back to the raw child."""
    try:
        reply = llm.json_call(CHILD_SYSTEM, f"SECTION: {parent.section or '-'}\n\nPARENT:\n{parent.text}\n\nCHILD:\n{child}")
        passage = reply.get("context_passage")
        content = passage.strip() if isinstance(passage, str) and 0 < len(passage.split()) <= settings.child_words * 3 else child
        pairs = reply.get("qa_pairs", [])
        pairs = [p for p in pairs if isinstance(p, dict) and isinstance(p.get("question"), str) and isinstance(p.get("answer"), str)] if isinstance(pairs, list) else []
    except Exception as exc:  # one bad chunk must not abort the document
        logger.error("Chunk enrichment failed: %s", exc)
        return child, []
    return content, pairs


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


def ingest_text(store: Store, llm, settings: Settings, title: str, text: str, result: IngestResult | None = None, parents: list[Parent] | None = None) -> IngestResult:
    """Chunk (parents, then children), enrich with the LLM, embed and store one document. Skips text already stored.

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
            result.status = "skipped"
            return result

        document_id = str(uuid.uuid4())
        base = {"document_id": document_id, "title": title, "citation": title, "file_hash": file_hash}
        parents = parents or parents_from_text(text, settings.parent_chars)
        children = [(pid, parent, child) for pid, parent in enumerate(parents) for child in split_children(parent.text, settings)]

        with ThreadPoolExecutor(settings.llm_workers) as pool:
            enriched = list(pool.map(lambda c: _enrich_child(llm, settings, c[1], c[2]), children))
        topics, summary = _topics_and_summary(llm, text)
        terms = [t["term"] for t in topics]

        items = []
        for chunk_id, ((pid, parent, _), (content, pairs)) in enumerate(zip(children, enriched)):
            chunk_topics = _topics_in(content, terms)
            link = {"parent_id": pid, "section": parent.section, "page": parent.page}
            heading = f"{parent.section}. " if parent.section else ""  # the section name helps a short chunk match
            items.append((
                heading + content,
                {**base, **link, "type": "chunk", "chunk_id": chunk_id, "content": content, "parent_text": parent.text, "parents_total": len(parents), "topics": chunk_topics},
            ))
            for pair in pairs:
                qa_topics = sorted(set(chunk_topics) | set(_topics_in(pair["question"] + " " + pair["answer"], terms)))
                items.append((
                    f"Question: {pair['question']} Answer: {pair['answer']}",
                    {**base, **link, "type": "qa_pair", "chunk_id": chunk_id, "question": pair["question"], "answer": pair["answer"], "topics": qa_topics},
                ))
        for topic in topics:
            items.append((f"{topic['term']}: {topic['definition']}", {**base, "type": "topic", "term": topic["term"], "definition": topic["definition"], "topics": [topic["term"]]}))
        if summary:
            items.append((summary, {**base, "type": "summary", "content": summary, "topics": _topics_in(summary, terms)}))

        store.add(items)
        result.status = "processed"
        result.document_id = document_id
        result.parents = len(parents)
        result.chunks = len(children)
        result.qa_pairs = sum(len(pairs) for _, pairs in enriched)
        result.topics = len(topics)
    except Exception as exc:
        logger.exception("Ingest failed for %s", title)
        result.status = "failed"
        result.error = str(exc)
    return result
