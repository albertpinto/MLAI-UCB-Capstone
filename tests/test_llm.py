import json
from types import SimpleNamespace

import httpx
import pytest
from groq import APIStatusError

from capstone.llm import LLM


class Client:
    """Stands in for the Groq client: replies with each of `contents` in turn."""

    def __init__(self, *contents):
        self.contents, self.calls = list(contents), 0
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        content = self.contents[min(self.calls, len(self.contents) - 1)]
        self.calls += 1
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


@pytest.fixture
def llm(monkeypatch):
    monkeypatch.setattr("capstone.llm.time.sleep", lambda s: None)
    return LLM("key", "model")


def test_a_reply_that_is_not_an_object_is_retried(llm):
    llm.client = Client(json.dumps([1, 2]), json.dumps("text"), json.dumps({"ok": True}))
    assert llm.json_call("s", "u") == {"ok": True} and llm.client.calls == 3


def test_gives_up_after_the_retries(llm):
    llm.client = Client(json.dumps([1]))
    with pytest.raises(ValueError):
        llm.json_call("s", "u", retries=3)
    assert llm.client.calls == 3


def status_error(code, headers=None):
    response = httpx.Response(code, headers=headers or {}, request=httpx.Request("POST", "https://x"))
    return APIStatusError("boom", response=response, body=None)


class Failing(Client):
    def __init__(self, error, then=None):
        super().__init__(json.dumps({"ok": True}))
        self.error, self.then = error, then

    def create(self, **kwargs):
        if self.calls < (self.then if self.then is not None else 10**9):
            self.calls += 1
            raise self.error
        return super().create(**kwargs)


def test_a_missing_model_fails_at_once(llm):
    llm.client = Failing(status_error(404))
    with pytest.raises(APIStatusError):
        llm.json_call("s", "u")
    assert llm.client.calls == 1


def test_a_rate_limit_waits_as_long_as_asked_then_succeeds(monkeypatch):
    sleeps = []
    monkeypatch.setattr("capstone.llm.time.sleep", sleeps.append)
    llm = LLM("key", "model")
    llm.client = Failing(status_error(429, {"retry-after": "7"}), then=1)
    assert llm.json_call("s", "u") == {"ok": True} and 7 in sleeps


def test_calls_are_spaced_by_the_minimum_interval(monkeypatch):
    sleeps = []
    monkeypatch.setattr("capstone.llm.time.sleep", sleeps.append)
    llm = LLM("key", "model", min_interval=5)
    llm.client = Client(json.dumps({"ok": True}))
    llm.json_call("s", "u")
    llm.json_call("s", "u")
    assert len(sleeps) == 1 and 4 < sleeps[0] <= 5
