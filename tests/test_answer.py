import pytest

from capstone.answer import NO_ANSWER, answer_question
from test_api import ingest


class ScriptedLLM:
    def __init__(self, reply):
        self.reply, self.calls = reply, 0

    def json_call(self, system, user, retries=3):
        self.calls += 1
        return self.reply


class RoutedLLM:
    """Answers the grounded-answer call and the readable-matches call differently."""

    def __init__(self, answer, matches):
        self.answer, self.matches = answer, matches

    def json_call(self, system, user, retries=3):
        reply = self.matches if "readable answers" in system else self.answer
        if isinstance(reply, Exception):
            raise reply
        return reply


def passage(score, text="Click Add Brand.", citation="Guide"):
    return {"score": score, "text": text, "type": "chunk", "citation": citation}


def test_abstains_without_calling_the_llm_when_nothing_is_relevant():
    llm = ScriptedLLM({"answer": "made up"})
    assert answer_question(llm, "q", [], 0.25)["abstained"]
    result = answer_question(llm, "q", [passage(0.1)], 0.25)
    assert result == {"answer": NO_ANSWER, "abstained": True, "sources": [], "matches": []}
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
    assert answer_question(llm, "q", [passage(0.9)], 0.25) == {"answer": NO_ANSWER, "abstained": True, "sources": [], "matches": []}


def test_empty_llm_answer_is_an_error():
    with pytest.raises(ValueError):
        answer_question(ScriptedLLM({"sources": []}), "q", [passage(0.9)], 0.25)


def test_ask_endpoint_gives_the_llm_only_retrieved_context(client, llm):
    ingest(client)
    body = client.post("/api/ask", json={"query": "How do I add a brand?"}).json()
    assert body["answer"] == "Click Add Brand [1]." and not body["abstained"]
    assert body["sources"][0]["citation"] == "Guide"
    first = body["sources"][0]
    assert client.get(f"/api/documents/{first['document_id']}/passages/{first['parent_id']}").status_code == 200  # every source links to a passage
    assert "Question: How do I add a brand?" in llm.prompts[0] and "[1] (Guide)" in llm.prompts[0]


def test_ask_endpoint_abstains_on_an_empty_store_and_validates(client, llm):
    assert client.post("/api/ask", json={"query": "anything"}).json()["abstained"]
    assert llm.prompts == []
    assert client.post("/api/ask", json={"query": " "}).status_code == 400


def test_matches_are_readable_rewrites_flagged_when_cited():
    llm = RoutedLLM(
        {"answer": "Do X [1][3].", "sources": [1, 3]},
        {"matches": [{"n": 1, "answer": "Readable one."}, {"n": 2, "answer": " Readable two. "}, {"n": 3, "answer": None}, {"n": 9, "answer": "no such passage"}, {"n": 4, "answer": ""}]},
    )
    passages = [passage(0.9, "one"), passage(0.8, "two", "Doc B"), passage(0.7, "three"), passage(0.6, "four")]
    matches = answer_question(llm, "q", passages, 0.25)["matches"]
    assert [(m["n"], m["answer"], m["cited"]) for m in matches] == [(1, "Readable one.", True), (2, "Readable two.", False), (3, None, True)]
    assert "text" not in matches[0] and matches[2]["text"] == "three"  # cited but not rewritten: the original text stands in
    assert matches[1]["citation"] == "Doc B" and "text" not in matches[1]  # the original text stays behind the passage link


def test_matches_are_capped_at_ten():
    many = [passage(0.9 - i / 100, f"p{i}") for i in range(14)]
    llm = RoutedLLM({"answer": "A [1].", "sources": [1]}, {"matches": [{"n": i, "answer": f"a{i}"} for i in range(1, 15)]})
    assert len(answer_question(llm, "q", many, 0.25)["matches"]) == 10


def test_a_failed_matches_call_still_returns_the_answer():
    llm = RoutedLLM({"answer": "Do X [1].", "sources": [1]}, RuntimeError("rate limited"))
    result = answer_question(llm, "q", [passage(0.9), passage(0.8)], 0.25)
    assert result["answer"] == "Do X [1]." and [m["n"] for m in result["matches"]] == [1]  # the cited passage still has a card
    assert result["matches"][0]["answer"] is None and result["matches"][0]["text"]
