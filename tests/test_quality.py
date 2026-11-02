import json
import time

from capstone.quality import answer_summary, artifact_summary
from test_api import DOC, ingest


def wait_quality(client, job_id):
    for _ in range(100):
        job = client.get(f"/api/quality/jobs/{job_id}").json()
        if job["status"] != "running":
            return job
        time.sleep(0.1)
    raise AssertionError("quality job did not finish")


def test_artifact_check_review_and_error_rates(client, llm):
    ingest(client, "Guide", DOC)
    assert client.get("/api/quality").json() == {}
    llm.supported = False  # the second pass finds every item unsupported
    job = wait_quality(client, client.post("/api/quality/artifacts/run", json={"sample": 5}).json()["job_id"])
    assert job["status"] == "done" and job["done"] == job["total"] >= 2

    data = client.get("/api/quality").json()["artifacts"]
    items = data["items"]
    assert {i["type"] for i in items} == {"qa_pair", "factoid"} and all(i["source"] and i["llm_supported"] is False for i in items)
    assert data["summary"]["qa_pair"]["llm_error_rate"] == 1.0 and data["summary"]["qa_pair"]["human_error_rate"] is None

    qa = next(i for i in items if i["type"] == "qa_pair")
    assert client.put(f"/api/quality/artifacts/{qa['id']}/review", json={"correct": True}).json()["correct"] is True
    assert client.put("/api/quality/artifacts/nope/review", json={"correct": True}).status_code == 404
    summary = client.get("/api/quality").json()["artifacts"]["summary"]["qa_pair"]
    assert summary["human_error_rate"] == 0.0 and summary["reviewed"] == 1 and summary["agreement"] == 0.0  # the person disagrees with the LLM

    llm.supported = True  # checking again keeps the person's verdict
    wait_quality(client, client.post("/api/quality/artifacts/run", json={"sample": 5}).json()["job_id"])
    again = client.get("/api/quality").json()["artifacts"]
    assert next(i for i in again["items"] if i["id"] == qa["id"])["human"] is True and again["summary"]["qa_pair"]["llm_error_rate"] == 0.0


def test_answer_evaluation_grades_answers_and_abstentions(client, settings, llm):
    ingest(client, "Guide", DOC)
    assert client.post("/api/quality/answers/run", json={}).status_code == 400  # no labelled questions
    settings.eval_questions_file.write_text(json.dumps([
        {"id": "q1", "question": "How do I add a brand?", "evidence": ["add brand"], "split": "test"},
        {"id": "q2", "question": "Train question", "evidence": ["x"], "split": "train"}]))
    settings.eval_unanswerable_file.write_text(json.dumps(["zzzz qqqq xxxx"]))  # shares no word with the document
    job = wait_quality(client, client.post("/api/quality/answers/run", json={}).json()["job_id"])
    assert job["status"] == "done" and job["total"] == 2

    answers = client.get("/api/quality").json()["answers"]
    assert (answers["answerable"], answers["unanswerable"], answers["failed"]) == (1, 1, 0)
    assert answers["faithfulness"] == 1.0 and answers["relevance"] == 1.0
    assert answers["wrongly_abstained"] == 0.0 and answers["correctly_abstained"] == 1.0  # the off-topic question retrieved nothing relevant


def test_summaries_ignore_failed_checks():
    items = [{"type": "qa_pair", "llm_supported": True, "human": None}, {"type": "qa_pair", "llm_supported": False, "human": False},
             {"type": "qa_pair", "llm_supported": None, "human": None}]
    qa = artifact_summary(items)["qa_pair"]
    assert (qa["items"], qa["checked"], qa["llm_error_rate"], qa["human_error_rate"], qa["agreement"]) == (3, 2, 0.5, 1.0, 1.0)
    rows = [{"answerable": True, "abstained": None, "faithful": None, "relevant": None}]
    assert answer_summary(rows)["failed"] == 1 and answer_summary(rows)["faithfulness"] is None
