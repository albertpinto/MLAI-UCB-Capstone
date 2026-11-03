"""Exit 0 if the key for the provider chosen by LLM_PROVIDER is set (in .env or the environment), else print what is missing and exit 1. Used by start.sh and start.ps1."""
import sys

from capstone.config import Settings

settings = Settings()
if settings.llm_provider not in ("groq", "openai"):
    sys.exit(f"LLM_PROVIDER must be groq or openai, not {settings.llm_provider!r}. Fix it in .env")
name = "OPENAI_API_KEY" if settings.llm_provider == "openai" else "GROQ_API_KEY"
if not (settings.openai_api_key if settings.llm_provider == "openai" else settings.groq_api_key):
    sys.exit(f"Missing {name} (LLM_PROVIDER={settings.llm_provider}). Run: cp .env.example .env, then set {name} in .env")
