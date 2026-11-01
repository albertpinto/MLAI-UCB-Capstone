import json
from types import SimpleNamespace

import pytest

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
