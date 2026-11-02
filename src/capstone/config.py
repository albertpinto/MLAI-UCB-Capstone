from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All configuration lives here; values come from the environment or .env."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"

    qdrant_url: str = "http://localhost:6335"
    collection: str = "constitution_india"
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embed_device: str = ""  # "" = auto (GPU if available), or "cpu" / "cuda"
    embed_threads: int = 4  # CPU threads torch may use; the default is every core, which can overload the machine
    company_name: str = "Bakasura"
    company_tagline: str = "Ask your documents anything"
    logo_url: str = "/logo.svg"  # a URL, or a file in frontend/public such as /logo.png; "" shows the first letter instead
    docs_dir: Path = Path("data/docs")
    suggested_questions_file: Path = Path("data/suggested_questions.json")  # pick list shown on the Ask page

    answer_results: int = 10  # passages given to the LLM as context
    answer_min_score: float = 0.25  # below this cosine score a passage is not relevant enough; with none left the answer abstains

    parent_chars: int = 1800  # size of a parent chunk (a topic section, about half a page); children are matched, parents are read
    child_words: int = 65  # target size of a child chunk; Chonkie splits a parent at meaning boundaries up to this size
    child_min_words: int = 20  # a smaller piece is merged into its neighbour, because tiny chunks embed thinly
    enrich_cache: Path | None = Path("data/enrichment_cache.jsonl")  # LLM results per child, so a crashed ingest resumes; None disables
    docling_threads: int = 4  # CPU threads for Docling's layout model
    chunk_chars: int = 1000  # the old single-level chunker, still used by eval/eval.py for its raw-chunk baseline
    chunk_overlap: int = 150
    llm_workers: int = 4

    eval_questions_file: Path = Path("eval/questions.json")  # labelled questions for the retrieval evaluation
    eval_results_file: Path = Path("eval/results.json")  # the last evaluation run, so the dashboard opens instantly
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # a real cross-encoder (not the bi-encoder); "" leaves reranking out
    fine_tuned_model: str = ""  # path to the fine-tuned embedding model; "" leaves it out of the evaluation


@lru_cache
def get_settings() -> Settings:
    return Settings()
