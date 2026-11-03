from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All configuration lives here; values come from the environment or .env."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    @classmethod
    def settings_customise_sources(cls, settings_cls, init_settings, env_settings, dotenv_settings, file_secret_settings):
        """.env wins over the shell's environment: a key left exported by another project (for example in ~/.bashrc) must not override this project's."""
        return init_settings, dotenv_settings, env_settings, file_secret_settings

    groq_api_key: str = ""
    groq_model: str = "openai/gpt-oss-120b"
    llm_provider: str = "groq"  # groq | openai: which service writes the chunk enrichment and the answers
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    llm_max_concurrent: int = 4
    llm_min_interval: float = 0.0

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

    document_ttl_days: int = 0  # a document stored from now on expires after this many days and is flushed; 0 = never
    flush_interval_hours: int = 24  # how often the API flushes expired documents (only when DOCUMENT_TTL_DAYS is set); POST /api/documents/flush does it on demand
    qdrant_path: Path | None = None  # a folder: Qdrant runs inside the API process and keeps its data there, so Docker is not needed (QDRANT_URL is then ignored)

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
    eval_unanswerable_file: Path = Path("eval/unanswerable.json")  # JSON list of questions the documents cannot answer; the answer evaluation checks the system abstains
    quality_file: Path = Path("data/quality.json")  # the last artifact check (with a person's reviews) and answer evaluation
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # a real cross-encoder (not the bi-encoder); "" leaves reranking out
    rerank_search: bool = True  # the Ask page re-orders its dense results with the cross-encoder (measured: better MRR and nDCG)
    rerank_candidates: int = 30  # results the cross-encoder re-scores; 20 to 30 measured about the same, more adds noise
    graph_dir: Path = Path("data/graphs")  # the knowledge graph is saved here as <COLLECTION>.json, so a different index never loads another's graph
    graph_aliases_file: Path | None = None  # optional JSON {"alias": "canonical entity"} to merge names the normaliser cannot, e.g. {"SC": "Supreme Court"}
    graph_search: bool = False  # Ask/search also use chunks tied to the question's entities in the graph (off until the evaluation shows it helps)
    fine_tuned_model: str = ""  # path to the fine-tuned embedding model; "" leaves it out of the evaluation

    @property
    def graph_path(self) -> Path:
        return self.graph_dir / f"{self.collection}.json"


@lru_cache
def get_settings() -> Settings:
    return Settings()
