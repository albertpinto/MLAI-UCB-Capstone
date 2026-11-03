"""Time the LLM enrichment on the same text with the provider set in .env (or LLM_PROVIDER=openai on the command line).

    uv run python scripts/bench_ingest.py [chunks] [workers]

Takes `chunks` chunks (default 100) from the live index, ingests them into an in-memory index with no cache, and prints the time.
Nothing is written to the real index or the cache. Run it once per provider and compare.
"""
import sys
import time

from qdrant_client import QdrantClient

from capstone.config import Settings
from capstone.ingest import ingest_text
from capstone.llm import LLM
from capstone.store import Embedder, Store

count = int(sys.argv[1]) if len(sys.argv) > 1 else 100
workers = int(sys.argv[2]) if len(sys.argv) > 2 else 4
s = Settings()
points, _ = QdrantClient(location=s.qdrant_url).scroll(s.collection, scroll_filter={"must": [{"key": "type", "match": {"value": "chunk"}}]}, limit=count + 100, with_payload=["content"], with_vectors=False)
text = "\n\n".join(p.payload["content"] for p in points[100 : 100 + count])

cfg = Settings(qdrant_url=":memory:", enrich_cache=None, llm_workers=workers)
openai = cfg.llm_provider == "openai"
llm = LLM(cfg.openai_api_key if openai else cfg.groq_api_key, cfg.openai_model if openai else cfg.groq_model, workers, cfg.llm_min_interval, cfg.llm_provider)
store = Store(":memory:", "bench", Embedder(cfg.embed_model, cfg.embed_device, cfg.embed_threads))
start = time.time()
result = ingest_text(store, llm, cfg, "Benchmark", text)
seconds = time.time() - start
print(f"{cfg.llm_provider} {llm.model}: {result.chunks} chunks, {result.qa_pairs} Q&A, {result.factoids} facts, {result.triplets} triplets "
      f"in {seconds:.1f}s ({seconds / max(result.chunks, 1):.2f}s per chunk; the whole Constitution would take about {seconds / max(result.chunks, 1) * 2358 / 60:.0f} min) status={result.status}")
