from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All configuration lives here; values come from the environment or .env."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"

    qdrant_url: str = "http://localhost:6335"
    collection: str = "capstone_india"
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embed_device: str = ""  # "" = auto (GPU if available), or "cpu" / "cuda"
    embed_threads: int = 4  # CPU threads torch may use; the default is every core, which can overload the machine
    company_name: str = "Bakasura"
    company_tagline: str = "Knowledge base"
    logo_url: str = "/logo.svg"  # a URL, or a file in frontend/public such as /logo.png; "" shows the first letter instead
    docs_dir: Path = Path("data/docs")

    answer_results: int = 5  # passages given to the LLM as context
    answer_min_score: float = 0.25  # below this cosine score a passage is not relevant enough; with none left the answer abstains

    chunk_chars: int = 1000
    chunk_overlap: int = 150
    llm_workers: int = 4


@lru_cache
def get_settings() -> Settings:
    return Settings()
