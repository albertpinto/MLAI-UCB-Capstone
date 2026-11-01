import hashlib

import pytest
from fastapi.testclient import TestClient

from capstone.config import Settings
from capstone.main import create_app
from capstone.store import Store


class FakeEmbedder:
    """Deterministic bag-of-words vectors: texts sharing words are close. No model download."""

    dim = 64

    def encode(self, texts):
        vectors = []
        for text in texts:
            v = [0.0] * self.dim
            for word in text.lower().split():
                v[int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dim] += 1.0
            norm = sum(x * x for x in v) ** 0.5 or 1.0
            vectors.append([x / norm for x in v])
        return vectors


class FakeLLM:
    def __init__(self):
        self.prompts = []  # user messages of grounded-answer calls

    def json_call(self, system, user, retries=3):
        if "numbered context" in system:
            self.prompts.append(user)
            return {"answer": "Click Add Brand [1].", "sources": [1]}
        if "question-answer" in system:
            return {"qa_pairs": [{"question": "How do I add a brand?", "answer": "Click Add Brand."}]}
        return {"topics": [{"term": "brand", "definition": "A company identity."}], "summary": "How to manage brands."}


@pytest.fixture
def settings(tmp_path):
    return Settings(qdrant_url=":memory:", collection="test", docs_dir=tmp_path, groq_api_key="x")


@pytest.fixture
def store():
    return Store(":memory:", "test", FakeEmbedder())


@pytest.fixture
def llm():
    return FakeLLM()


@pytest.fixture
def client(settings, store, llm):
    with TestClient(create_app(settings, store, llm)) as c:
        yield c
