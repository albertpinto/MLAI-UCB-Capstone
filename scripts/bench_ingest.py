"""Time the LLM enrichment on the same text with the provider set in .env (or LLM_PROVIDER=openai on the command line).

    uv run python scripts/bench_ingest.py [pages] [workers]

Takes the plain text of `pages` pages (default 25) from the first PDF in DOCS_DIR, ingests it into an in-memory index with no cache, and prints
the time. Nothing is written to the real index or the cache. Run it once per provider and compare.
"""
import sys
import time

from pypdf import PdfReader

from capstone.config import Settings
from capstone.ingest import ingest_text
from capstone.llm import LLM
from capstone.store import Embedder, Store

pages = int(sys.argv[1]) if len(sys.argv) > 1 else 25
workers = int(sys.argv[2]) if len(sys.argv) > 2 else 4
s = Settings()
pdf = next(s.docs_dir.glob("*.pdf"))
text = "\n\n".join(page.extract_text() for page in PdfReader(pdf).pages[60 : 60 + pages])  # from page 60, past the contents and preamble

cfg = Settings(qdrant_url=":memory:", enrich_cache=None, llm_workers=workers)
openai = cfg.llm_provider == "openai"
llm = LLM(cfg.openai_api_key if openai else cfg.groq_api_key, cfg.openai_model if openai else cfg.groq_model, workers, cfg.llm_min_interval, cfg.llm_provider)
store = Store(":memory:", "bench", Embedder(cfg.embed_model, cfg.embed_device, cfg.embed_threads))
start = time.time()
result = ingest_text(store, llm, cfg, "Benchmark", text)
seconds = time.time() - start
print(f"{cfg.llm_provider} {llm.model}: {result.chunks} chunks, {result.qa_pairs} Q&A, {result.factoids} facts, {result.triplets} triplets "
      f"in {seconds:.1f}s ({seconds / max(result.chunks, 1):.2f}s per chunk; the whole Constitution would take about {seconds / max(result.chunks, 1) * 2358 / 60:.0f} min) status={result.status}")
