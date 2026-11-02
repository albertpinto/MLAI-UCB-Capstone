import logging
import uuid
from pathlib import Path

from qdrant_client import QdrantClient, models

from .chunking import is_contents, reflow

logger = logging.getLogger(__name__)

TYPES = ("chunk", "qa_pair", "factoid", "triplet", "topic", "summary")
DEFAULT_TYPES = [t for t in TYPES if t != "triplet"]  # triplets are short and numerous; they only join a search when asked for
BATCH = 64
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class Embedder:
    """Sentence-transformers model, loaded once. Falls back to the stock model if a local path is missing."""

    def __init__(self, name: str, device: str = "", threads: int = 0):
        import torch
        from sentence_transformers import SentenceTransformer

        if threads:
            torch.set_num_threads(threads)

        if name.startswith(("/", ".", "~")):  # a local path rather than a Hugging Face id
            path = Path(name).expanduser()
            if path.exists():
                name = str(path)
            else:
                logger.warning("Embedding model %s not found, using %s", name, DEFAULT_MODEL)
                name = DEFAULT_MODEL
        logger.info("Loading embedding model %s", name)
        self.model = SentenceTransformer(name, device=device or None)
        self.dim = self.model.get_sentence_embedding_dimension()

    def encode(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts, normalize_embeddings=True, batch_size=32).tolist()


def _match(key: str, values: list[str]) -> models.FieldCondition:
    return models.FieldCondition(key=key, match=models.MatchAny(any=values))


def _equals(key: str, value) -> models.FieldCondition:
    return models.FieldCondition(key=key, match=models.MatchValue(value=value))


def display_text(payload: dict) -> str:
    kind = payload.get("type")
    if kind == "qa_pair":
        return f"Q: {payload['question']}\nA: {payload['answer']}"
    if kind == "topic":
        return f"{payload['term']}: {payload['definition']}"
    if kind == "triplet":
        return f"{payload['subject']} → {payload['relation']} → {payload['object']}"
    return reflow(payload.get("content", ""))


class Store:
    """One Qdrant collection holding chunk, qa_pair, topic and summary points."""

    def __init__(self, url: str, collection: str, embedder):
        self.client = QdrantClient(location=url)  # ":memory:" works too (used by tests)
        self.collection = collection
        self.embedder = embedder
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        if self.client.collection_exists(self.collection):
            return
        self.client.create_collection(
            self.collection,
            vectors_config=models.VectorParams(size=self.embedder.dim, distance=models.Distance.COSINE),
        )
        for key in ("type", "document_id", "file_hash", "topics"):
            self.client.create_payload_index(self.collection, key, models.PayloadSchemaType.KEYWORD)
        self.client.create_payload_index(self.collection, "parent_id", models.PayloadSchemaType.INTEGER)

    def has_hash(self, file_hash: str) -> bool:
        flt = models.Filter(must=[_equals("file_hash", file_hash)])
        return self.client.count(self.collection, count_filter=flt, exact=True).count > 0

    def add(self, items: list[tuple[str, dict]]) -> None:
        """items: (text to embed, payload). Embeds and upserts in batches."""
        for start in range(0, len(items), BATCH):
            batch = items[start : start + BATCH]
            vectors = self.embedder.encode([text for text, _ in batch])
            points = [
                models.PointStruct(id=str(uuid.uuid4()), vector=vector, payload=payload)
                for vector, (_, payload) in zip(vectors, batch)
            ]
            self.client.upsert(self.collection, points=points)

    def search(self, query: str, limit: int = 10, types: list[str] | None = None, topics: list[str] | None = None) -> list[dict]:
        must = [_match("type", types or DEFAULT_TYPES)]
        if topics:
            must.append(_match("topics", topics))
        response = self.client.query_points(
            self.collection,
            query=self.embedder.encode([query])[0],
            limit=limit,
            query_filter=models.Filter(must=must) if must else None,
            with_payload=True,
        )
        results = [
            {"score": hit.score, "text": display_text(hit.payload), **{k: v for k, v in hit.payload.items() if k != "parent_text"}}  # parents are fetched by with_parents
            for hit in response.points
        ]
        return [r for r in results if not (r["type"] == "chunk" and is_contents(r["content"]))]  # a contents page answers nothing

    def _parent_payload(self, document_id: str, parent_id: int) -> dict | None:
        """Any child of the parent: they all carry the parent's text, section and page."""
        found = self._scroll([_equals("document_id", document_id), _equals("type", "chunk"), _equals("parent_id", parent_id)], limit=1)
        return found[0] if found else None

    def with_parents(self, results: list[dict]) -> list[dict]:
        """Small-to-big: add `context`, the parent's full text, to each result that came from a child chunk.

        Results from the same parent are merged (the best-ranked one is kept) so the LLM never reads a section twice.
        """
        expanded, seen = [], set()
        for r in results:
            key = (r["document_id"], r.get("parent_id"))
            if r.get("parent_id") is None:
                expanded.append(r)
            elif key not in seen:
                seen.add(key)
                parent = self._parent_payload(*key)
                expanded.append({**r, "context": parent["parent_text"] if parent else r["text"]})
        return expanded

    def passage(self, document_id: str, parent_id: int) -> dict | None:
        """A parent chunk: the original section text with its heading and page, or None."""
        payload = self._parent_payload(document_id, parent_id)
        if not payload:
            return None
        return {
            "document_id": document_id, "title": payload["title"], "citation": payload["citation"], "section": payload["section"],
            "page": payload["page"], "parent_id": parent_id, "total": payload["parents_total"], "text": payload["parent_text"],
        }

    def _scroll(self, must: list, limit: int = 1000) -> list[dict]:
        points, _ = self.client.scroll(
            self.collection, scroll_filter=models.Filter(must=must), limit=limit, with_payload=True
        )
        return [p.payload for p in points]

    def all_payloads(self, kind: str) -> list[dict]:
        """Every point of one type (without vectors), paged through the collection."""
        out, offset = [], None
        flt = models.Filter(must=[_equals("type", kind)])
        while True:
            points, offset = self.client.scroll(self.collection, scroll_filter=flt, limit=1000, offset=offset, with_payload=True)
            out += [p.payload for p in points]
            if offset is None:
                return out

    def stats(self) -> dict:
        """Numbers for the dashboard: points per type, chunks per top-level section (Part), and child-chunk sizes in words."""
        counts = {t: self.client.count(self.collection, count_filter=models.Filter(must=[_equals("type", t)]), exact=True).count for t in TYPES}
        chunks = self.all_payloads("chunk")
        per_part: dict[str, int] = {}
        sizes: dict[int, int] = {}
        for c in chunks:
            part = (c.get("section") or "(no heading)").split(" › ")[0]
            per_part[part] = per_part.get(part, 0) + 1
            bucket = len(c.get("content", "").split()) // 10 * 10
            sizes[bucket] = sizes.get(bucket, 0) + 1
        return {
            "points": counts,
            "documents": len({c["document_id"] for c in chunks}),
            "parents": len({(c["document_id"], c["parent_id"]) for c in chunks}),
            "chunks_per_part": sorted(({"part": k, "chunks": v} for k, v in per_part.items()), key=lambda r: -r["chunks"]),
            "child_words": [{"from": k, "to": k + 9, "count": v} for k, v in sorted(sizes.items())],
        }

    def documents(self) -> list[dict]:
        """One entry per document, taken from its first chunk."""
        firsts = self._scroll([_equals("type", "chunk"), _equals("chunk_id", 0)])
        return [{k: p[k] for k in ("document_id", "title", "citation")} for p in firsts]

    def topics(self) -> list[dict]:
        seen = {}
        for p in self._scroll([_equals("type", "topic")]):
            seen.setdefault(p["term"].lower(), {"term": p["term"], "definition": p["definition"]})
        return sorted(seen.values(), key=lambda t: t["term"].lower())

    def delete_document(self, document_id: str) -> None:
        self.client.delete(
            self.collection,
            points_selector=models.FilterSelector(filter=models.Filter(must=[_equals("document_id", document_id)])),
        )
