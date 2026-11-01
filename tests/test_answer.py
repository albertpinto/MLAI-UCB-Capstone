import pytest

from capstone.answer import NO_ANSWER, answer_question
from test_api import ingest


class ScriptedLLM:
    def __init__(self, reply):
        self.reply, self.calls = reply, 0

    def json_call(self, system, user, retries=3):
        self.calls += 1
        return self.reply


def passage(score, text="Click Add Brand.", citation="Guide"):
    return {"score": score, "text": text, "type": "chunk", "citation": citation}


def test_abstains_without_calling_the_llm_when_nothing_is_relevant():
    llm = ScriptedLLM({"answer": "made up"})
    assert answer_question(llm, "q", [], 0.25)["abstained"]
    result = answer_question(llm, "q", [passage(0.1)], 0.25)
    assert result == {"answer": NO_ANSWER, "abstained": True, "sources": []}
    assert llm.calls == 0


def test_returns_only_the_cited_passages():
    llm = ScriptedLLM({"answer": "Do X [2].", "sources": [2, 9, "x"]})
    result = answer_question(llm, "q", [passage(0.9, "one"), passage(0.8, "two")], 0.25)
    assert [(s["n"], s["text"]) for s in result["sources"]] == [(2, "two")]
    assert not result["abstained"]


def test_shows_all_passages_when_the_model_cites_none():
    llm = ScriptedLLM({"answer": "Do X.", "sources": []})
    assert len(answer_question(llm, "q", [passage(0.9), passage(0.8)], 0.25)["sources"]) == 2


def test_model_can_abstain_itself():
    llm = ScriptedLLM({"answer": NO_ANSWER, "sources": [1]})
    assert answer_question(llm, "q", [passage(0.9)], 0.25) == {"answer": NO_ANSWER, "abstained": True, "sources": []}


def test_empty_llm_answer_is_an_error():
    with pytest.raises(ValueError):
        answer_question(ScriptedLLM({"sources": []}), "q", [passage(0.9)], 0.25)


def test_ask_endpoint_gives_the_llm_only_retrieved_context(client, llm):
    ingest(client)
    body = client.post("/api/ask", json={"query": "How do I add a brand?"}).json()
    assert body["answer"] == "Click Add Brand [1]." and not body["abstained"]
    assert body["sources"][0]["citation"] == "Guide"
    assert "Question: How do I add a brand?" in llm.prompts[0] and "[1] (Guide)" in llm.prompts[0]


def test_ask_endpoint_abstains_on_an_empty_store_and_validates(client, llm):
    assert client.post("/api/ask", json={"query": "anything"}).json()["abstained"]
    assert llm.prompts == []
    assert client.post("/api/ask", json={"query": " "}).status_code == 400
