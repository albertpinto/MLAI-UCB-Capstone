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


class FakeReranker:
    """Scores a (query, text) pair by the words they share. No model download."""

    def predict(self, pairs):
        return [float(len(set(q.lower().split()) & set(t.lower().split()))) for q, t in pairs]


@pytest.fixture(autouse=True)
def offline_splitter(monkeypatch):
    """Children are cut at blank lines or single lines, so tests never load the Chonkie model."""
    monkeypatch.setattr("capstone.ingest.split_children", lambda text, settings: [p.strip() for p in text.split("\n\n") if p.strip()] or [text])


class FakeLLM:
    def __init__(self):
        self.prompts = []  # user messages of grounded-answer calls
        self.supported = True  # what the quality checks say about generated items and answers

    def json_call(self, system, user, retries=3):
        if "check generated material" in system:
            return {"supported": self.supported, "reason": "checked"}
        if "grade an answer" in system:
            return {"faithful": self.supported, "relevant": True, "reason": "graded"}
        if "readable answers" in system:
            return {"matches": [{"n": 1, "answer": "A readable version."}, {"n": 2, "answer": "Another readable version."}]}
        if "numbered context" in system:
            self.prompts.append(user)
            return {"answer": "Click Add Brand [1].", "sources": [1]}
        if "resolve references" in system:
            return {"replacements": [], "resolved": user.split("CHILD:\n", 1)[1]}
        if "search index" in system:
            return {
                "qa_pairs": [{"question": "How do I add a brand?", "answer": "Click Add Brand."}],
                "factoids": ["The Add Brand button creates a brand."],
                "triplets": [{"subject": "Add Brand button", "relation": "creates", "object": "a brand"}],
            }
        return {"topics": [{"term": "brand", "definition": "A company identity."}], "summary": "How to manage brands."}


@pytest.fixture
def settings(tmp_path):
    return Settings(qdrant_url=":memory:", collection="test", docs_dir=tmp_path, groq_api_key="x", enrich_cache=None, eval_questions_file=tmp_path / "q.json", eval_results_file=tmp_path / "r.json", fine_tuned_model="", graph_dir=tmp_path / "graphs", quality_file=tmp_path / "quality.json", eval_unanswerable_file=tmp_path / "unanswerable.json")


@pytest.fixture
def store():
    return Store(":memory:", "test", FakeEmbedder())


@pytest.fixture
def llm():
    return FakeLLM()


@pytest.fixture
def client(settings, store, llm):
    with TestClient(create_app(settings, store, llm, embedder_factory=lambda name: FakeEmbedder(), reranker_factory=lambda name: FakeReranker())) as c:
        yield c
