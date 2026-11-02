"""Copy the index from the Qdrant server into a data folder, so the app can run without Docker (QDRANT_PATH).

    uv run python -m capstone.export_index data/qdrant
"""
import sys
from pathlib import Path

from qdrant_client import QdrantClient, models

from .config import get_settings

BATCH = 256


def export_index(source: QdrantClient, collection: str, folder: Path) -> int:
    """Copy every point (with its vector) of `collection` into a local Qdrant at `folder`. Returns how many points were copied."""
    target = QdrantClient(path=str(folder))
    size = source.get_collection(collection).config.params.vectors.size
    if target.collection_exists(collection):
        raise SystemExit(f"{folder} already holds the collection {collection}; remove the folder first or choose another")
    target.create_collection(collection, vectors_config=models.VectorParams(size=size, distance=models.Distance.COSINE))
    copied, offset = 0, None
    while True:
        points, offset = source.scroll(collection, limit=BATCH, offset=offset, with_payload=True, with_vectors=True)
        target.upsert(collection, points=[models.PointStruct(id=p.id, vector=p.vector, payload=p.payload) for p in points])
        copied += len(points)
        if offset is None:
            target.close()
            return copied


if __name__ == "__main__":
    settings = get_settings()
    folder = Path(sys.argv[1] if len(sys.argv) > 1 else "data/qdrant")
    print(f"Copied {export_index(QdrantClient(location=settings.qdrant_url), settings.collection, folder)} points to {folder}")
