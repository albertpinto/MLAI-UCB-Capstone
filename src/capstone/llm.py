import json
import logging
import threading
import time

from groq import APIStatusError, Groq

logger = logging.getLogger(__name__)

# A wrong model name, key or request will fail the same way every time, so these are not retried.
NOT_RETRYABLE = {400, 401, 403, 404, 422}


class LLM:
    """Thin Groq wrapper: one method that returns parsed JSON. Calls are throttled (at most `max_concurrent` in flight,
    at least `min_interval` seconds between starts) and a 429 waits for the time Groq asks for."""

    def __init__(self, api_key: str, model: str, max_concurrent: int = 4, min_interval: float = 0.0):
        if not api_key:
            raise ValueError("GROQ_API_KEY is not set (see .env.example)")
        self.client = Groq(api_key=api_key, max_retries=0)
        self.model = model
        self.slots = threading.Semaphore(max(1, max_concurrent))
        self.min_interval = min_interval
        self.next_start = 0.0
        self.pace = threading.Lock()

    def _wait_turn(self) -> None:
        with self.pace:
            now = time.monotonic()
            start = max(now, self.next_start)
            self.next_start = start + self.min_interval
        if start > now:
            time.sleep(start - now)

    def json_call(self, system: str, user: str, retries: int = 5) -> dict:
        for attempt in range(retries):
            delay = 2**attempt
            try:
                self._wait_turn()
                with self.slots:
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
                if isinstance(exc, APIStatusError):
                    if exc.status_code in NOT_RETRYABLE:
                        raise
                    try:
                        delay = max(delay, float(exc.response.headers.get("retry-after", 0)))
                    except (TypeError, ValueError):
                        pass
                if attempt == retries - 1:
                    raise
                time.sleep(delay)
