import json
import time

import pytest

from capstone.evaluation import metrics, normalise, relevant_sections, Corpus, section_ranking
from test_api import ingest

DOC = "Billing\n\nInvoices are sent on the first day of every month.\n\nRefunds are paid within 14 days of the request.\n\nPasswords\n\nTo reset a password click the Forgot password link."


def test_metrics_with_one_and_several_relevant_sections():
    assert metrics(["a", "b", "c"], {"a"}, 3) == (1.0, 1.0, 1.0)
    recall, mrr, ndcg = metrics(["a", "b", "c"], {"c"}, 3)
    assert (recall, round(mrr, 3), round(ndcg, 3)) == (1.0, 0.333, 0.5)
    assert metrics(["a", "b", "c"], {"c"}, 2) == (0.0, 1 / 3, 0.0)  # found, but beyond k
    assert metrics(["a"], {"zzz"}, 5) == (0.0, 0.0, 0.0)
    recall, mrr, ndcg = metrics(["a", "x", "b"], {"a", "b"}, 3)
    assert recall == 1.0 and mrr == 1.0 and 0.9 < ndcg < 1.0  # two relevant, one at rank 3


def test_sections_rank_by_their_best_chunk_and_evidence_ignores_case_and_punctuation():
    assert section_ranking([0.1, 0.9, 0.5, 0.8], ["s1", "s2", "s1", "s3"]) == ["s2", "s3", "s1"]
    corpus = Corpus([], [], {"s1": {"section": "A", "page": 1, "text": "", "norm": normalise("Refunds are PAID within 14 days.")}})
    assert relevant_sections(corpus, {"evidence": ["paid within 14 days"]}) == {"s1"}
    assert relevant_sections(corpus, {"evidence": ["nothing like it"]}) == set()


def wait_eval(client, job_id):
    for _ in range(100):
        job = client.get(f"/api/eval/jobs/{job_id}").json()
        if job["status"] != "running":
            return job
        time.sleep(0.1)
    raise AssertionError("evaluation did not finish")


def test_questions_run_and_results_over_the_api(client):
    assert client.get("/api/eval/results").json() == {}
    assert client.post("/api/eval/run", json={}).status_code == 400  # no questions yet
    ingest(client, "Guide", DOC)
    questions = [
        {"id": "q1", "question": "When are invoices sent?", "evidence": ["first day of every month"], "split": "test"},
        {"id": "q2", "question": "How do I reset a password?", "evidence": ["forgot password link"], "split": "test"},
        {"id": "q3", "question": "A train question", "evidence": ["paid within 14 days"], "split": "train"},
        {"id": "q4", "question": "Evidence missing from the text", "evidence": ["zebra"], "split": "test"},
    ]
    assert client.put("/api/eval/questions", json={"questions": questions}).json() == {"saved": 4}
    listed = client.get("/api/eval/questions").json()
    assert [q["id"] for q in listed["questions"]] == ["q1", "q2", "q3", "q4"] and [r["id"] for r in listed["retrievers"]] == ["bm25", "base"]
    assert client.put("/api/eval/questions", json={"questions": questions + questions[:1]}).status_code == 400  # duplicate id

    assert client.post("/api/eval/run", json={"retrievers": ["tuned"]}).status_code == 400  # no fine-tuned model configured
    job = wait_eval(client, client.post("/api/eval/run", json={"k": 3}).json()["job_id"])
    assert job["status"] == "done" and job["done"] == job["total"] == 2

    result = client.get("/api/eval/results").json()
    assert set(result["retrievers"]) == {"bm25", "base"} and result["k"] == 3
    assert result["questions"] == 2 and result["unresolved"] == ["q4"]  # test split only; q4's evidence is nowhere
    assert result["retrievers"]["bm25"]["recall"] == 1.0 and len(result["retrievers"]["bm25"]["curve"]) == 20
    assert {r["id"] for r in result["per_question"]} == {"q1", "q2"} and all(r["ranks"]["bm25"] == 1 for r in result["per_question"])
    all_split = wait_eval(client, client.post("/api/eval/run", json={"split": "all"}).json()["job_id"])
    assert all_split["status"] == "done" and client.get("/api/eval/results").json()["questions"] == 3


def test_stats_describe_the_index(client):
    ingest(client, "Guide", DOC)
    stats = client.get("/api/stats").json()
    assert stats["documents"] == 1 and stats["points"]["chunk"] >= 2 and stats["points"]["qa_pair"] >= 1
    assert stats["chunks_per_part"][0]["chunks"] >= 1 and sum(b["count"] for b in stats["child_words"]) == stats["points"]["chunk"]


def test_fine_tuning_data_never_touches_the_held_out_parts(store):
    from capstone.finetune import batches, build_triplets, held_out_parts
    from capstone.evaluation import load_corpus

    rows = [("Part A", "Invoices are sent monthly.", "When are invoices sent?"), ("Part A", "Refunds take 14 days.", "How long do refunds take?"),
            ("Part B", "Passwords reset by email link.", "How do I reset my password?"), ("Part C", "Brands need an admin approval.", "Who approves a brand?")]
    items = []
    for i, (part, content, question) in enumerate(rows):
        link = {"document_id": "d", "chunk_id": i, "parent_id": i, "section": part, "page": 1, "title": "T", "citation": "T", "file_hash": "h", "topics": []}
        items.append((content, {**link, "type": "chunk", "content": content, "original": content, "parent_text": content, "parents_total": 4}))
        items.append((question, {**link, "type": "qa_pair", "question": question, "answer": content}))
    store.add(items)
    questions = [{"id": "t", "question": "Reset?", "evidence": ["reset by email link"], "split": "test"},
                 {"id": "x", "question": "Brands?", "evidence": ["admin approval"], "split": "train"}]
    assert held_out_parts(load_corpus(store), questions) == {"Part B"}  # only test questions hold a Part out

    triplets, held = build_triplets(store, questions, store.embedder)
    assert held == {"Part B"} and triplets
    for question, positive, negative in triplets:
        assert "reset" not in question.lower() and "Passwords" not in positive + negative  # the held-out Part is neither a positive nor a negative
        assert positive.split(". ", 1)[0] != negative.split(". ", 1)[0]  # the negative comes from another Part

    pairs = [("q1", "same chunk", "n"), ("q2", "same chunk", "n"), ("q3", "other chunk", "n")]
    assert all(len({t[1] for t in b}) == len(b) for b in batches(pairs, 3, 0))  # no chunk twice in one batch
    assert sum(len(b) for b in batches(pairs, 3, 0)) == 3


def test_training_runs_and_saves_a_loadable_model(tmp_path):
    from capstone.finetune import train
    from capstone.store import DEFAULT_MODEL

    triplets = [(f"question {i}", f"the chunk about topic {i}", f"an unrelated chunk number {i}") for i in range(6)]
    logged = []
    train(triplets, DEFAULT_MODEL, str(tmp_path / "m"), epochs=1, batch_size=3, threads=2, log=logged.append)
    from sentence_transformers import SentenceTransformer

    assert SentenceTransformer(str(tmp_path / "m"), device="cpu").encode(["hello"]).shape == (1, 384)
    assert logged and "loss" in logged[-1]
