"""Quality checks on what the LLM produced.

1. Generated artifacts (Q&A pairs, factoids): a second LLM pass checks each one against the chunk it came from; a person can
   then mark items correct or incorrect, and the error rate of both is reported.
2. Answers: for labelled questions, is the answer faithful to the retrieved context and does it answer the question, and for
   questions the documents cannot answer, does the system abstain? Used by /api/quality/*.
"""
import hashlib
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .answer import _numbered, answer_question

ARTIFACT_KINDS = ("qa_pair", "factoid")
CHECK_SYSTEM = (
    "You check generated material against the source text it was made from. Judge ONLY from the source text. "
    "The item is supported if the source text states it or clearly implies it, and it adds nothing the source does not say. "
    "It is not supported if it contradicts the source, changes a number, name or condition, or states something the source does not. "
    'Reply as JSON: {"supported": true or false, "reason": "one short sentence"}'
)
JUDGE_SYSTEM = (
    "You grade an answer given by a question-answering system. Judge it ONLY against the numbered context passages it was given. "
    "faithful: every claim in the answer is supported by the context (an answer that adds outside facts is not faithful). "
    "relevant: the answer addresses what the question asks (it is not off-topic and not an empty deflection). "
    'Reply as JSON: {"faithful": true or false, "relevant": true or false, "reason": "one short sentence"}'
)


def item_id(payload: dict) -> str:
    text = payload.get("question", "") + "|" + payload.get("answer", payload.get("content", ""))
    return hashlib.sha1(f"{payload['type']}|{payload['document_id']}|{payload['chunk_id']}|{text}".encode()).hexdigest()[:12]


def load(path: Path) -> dict:
    return json.loads(path.read_text()) if path.is_file() else {}


def save(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False))


def check_artifacts(llm, store, kinds: list[str], sample: int, workers: int = 4, seed: int = 0, progress=None) -> list[dict]:
    """Check a random sample (per kind) of generated items against their source chunk. Items whose chunk is missing are skipped."""
    picked = []
    for kind in kinds:
        payloads = store.all_payloads(kind)
        picked += random.Random(seed).sample(payloads, min(sample, len(payloads)))
    chunks = store.chunk_payloads([(p["document_id"], p["chunk_id"]) for p in picked])
    work = [(p, chunks[(p["document_id"], p["chunk_id"])]) for p in picked if (p["document_id"], p["chunk_id"]) in chunks]
    done = [0]

    def check(pair):
        payload, chunk = pair
        item = payload["question"] + " -> " + payload["answer"] if payload["type"] == "qa_pair" else payload["content"]
        try:
            reply = llm.json_call(CHECK_SYSTEM, f"Source text:\n{chunk['content']}\n\nGenerated {payload['type']}:\n{item}")
            supported, reason = bool(reply.get("supported")), str(reply.get("reason", ""))
        except Exception as exc:  # a failed check is recorded, not counted as right or wrong
            supported, reason = None, f"check failed: {exc}"
        done[0] += 1
        if progress:
            progress(done[0], len(work))
        return {"id": item_id(payload), "type": payload["type"], "document_id": payload["document_id"], "chunk_id": payload["chunk_id"], "parent_id": payload["parent_id"],
                "section": payload.get("section", ""), "page": payload.get("page"), "question": payload.get("question"), "answer": payload.get("answer", payload.get("content")),
                "source": chunk["content"], "original": chunk.get("original", ""), "llm_supported": supported, "reason": reason, "human": None}

    with ThreadPoolExecutor(workers) as pool:
        return list(pool.map(check, work))


def artifact_summary(items: list[dict]) -> dict:
    """Error rates per kind: by the LLM check, and by the person's review. `agreement` compares the two on the items reviewed."""
    out = {}
    for kind in ARTIFACT_KINDS:
        rows = [i for i in items if i["type"] == kind]
        checked = [i for i in rows if i["llm_supported"] is not None]
        reviewed = [i for i in rows if i["human"] is not None]
        both = [i for i in reviewed if i["llm_supported"] is not None]
        llm_errors, human_errors = sum(not i["llm_supported"] for i in checked), sum(not i["human"] for i in reviewed)
        out[kind] = {
            "items": len(rows), "checked": len(checked), "llm_errors": llm_errors, "llm_error_rate": llm_errors / len(checked) if checked else None,
            "reviewed": len(reviewed), "human_errors": human_errors, "human_error_rate": human_errors / len(reviewed) if reviewed else None,
            "agreement": sum(i["llm_supported"] == i["human"] for i in both) / len(both) if both else None,
        }
    return out


def review(path: Path, item: str, correct: bool | None) -> bool:
    """Record a person's verdict on one item (None clears it). False if the item is unknown."""
    data = load(path)
    found = next((i for i in data.get("artifacts", {}).get("items", []) if i["id"] == item), None)
    if found is None:
        return False
    found["human"] = correct
    save(path, data)
    return True


def evaluate_answers(llm, retrieve, answerable: list[dict], unanswerable: list[str], min_score: float, workers: int = 4, progress=None) -> dict:
    """`retrieve(question)` gives the results the Ask page would give the LLM. Answerable questions are graded for faithfulness and relevance
    (an abstention on one of them counts as a missed answer); unanswerable ones should end in an abstention."""
    questions = [(q["id"], q["question"], True) for q in answerable] + [(f"u{i + 1}", q, False) for i, q in enumerate(unanswerable)]
    done = [0]

    def grade(entry):
        qid, question, expected = entry
        row = {"id": qid, "question": question, "answerable": expected, "answer": "", "abstained": None, "faithful": None, "relevant": None, "reason": ""}
        try:
            results = retrieve(question)
            reply = answer_question(llm, question, results, min_score)
            row.update(answer=reply["answer"], abstained=reply["abstained"])
            if not reply["abstained"]:
                context = _numbered([r for r in results if r["score"] >= min_score])
                verdict = llm.json_call(JUDGE_SYSTEM, f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer: {reply['answer']}")
                row.update(faithful=bool(verdict.get("faithful")), relevant=bool(verdict.get("relevant")), reason=str(verdict.get("reason", "")))
        except Exception as exc:  # a failed question is shown as failed and left out of the rates
            row.update(abstained=None, reason=f"failed: {exc}")
        done[0] += 1
        if progress:
            progress(done[0], len(questions))
        return row

    with ThreadPoolExecutor(workers) as pool:
        rows = list(pool.map(grade, questions))
    return {"checked_at": time.time(), "rows": rows, **answer_summary(rows)}


def answer_summary(rows: list[dict]) -> dict:
    rate = lambda hits, total: hits / total if total else None
    ok = [r for r in rows if r["abstained"] is not None]
    answerable = [r for r in ok if r["answerable"]]
    answered = [r for r in answerable if not r["abstained"]]
    unanswerable = [r for r in ok if not r["answerable"]]
    return {
        "answerable": len(answerable), "unanswerable": len(unanswerable), "failed": len(rows) - len(ok),
        "faithfulness": rate(sum(r["faithful"] for r in answered), len(answered)),
        "relevance": rate(sum(r["relevant"] for r in answered), len(answered)),
        "wrongly_abstained": rate(sum(r["abstained"] for r in answerable), len(answerable)),
        "correctly_abstained": rate(sum(r["abstained"] for r in unanswerable), len(unanswerable)),
    }
