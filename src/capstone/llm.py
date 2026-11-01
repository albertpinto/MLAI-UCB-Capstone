import json
import logging
import time

from groq import Groq

logger = logging.getLogger(__name__)


class LLM:
    """Thin Groq wrapper: one method that returns parsed JSON."""

    def __init__(self, api_key: str, model: str):
        if not api_key:
            raise ValueError("GROQ_API_KEY is not set (see .env.example)")
        self.client = Groq(api_key=api_key)
        self.model = model

    def json_call(self, system: str, user: str, retries: int = 5) -> dict:
        for attempt in range(retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                    response_format={"type": "json_object"},
                    temperature=0.3,
                )
                data = json.loads(response.choices[0].message.content)
                if not isinstance(data, dict):  # callers read fields from it; a list or string is a bad reply, so retry
                    raise ValueError(f"Expected a JSON object, got {type(data).__name__}")
                return data
            except Exception as exc:
                logger.warning("LLM call failed (attempt %d/%d): %s", attempt + 1, retries, exc)
                if attempt == retries - 1:
                    raise
                time.sleep(2**attempt)
