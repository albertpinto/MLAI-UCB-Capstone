import logging
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

NO_ANSWER = "I couldn't find this in the documentation."
ANSWER_SYSTEM = (
    "You answer questions for a knowledge base using ONLY the numbered context passages provided. "
    "Give a complete, well-organised answer: start with one sentence that answers the question directly, "
    "then list the details as short bullet points (lines starting with \"- \"), covering everything the passages say that is relevant. "
    "Name articles, parts or section numbers when the passages state them. Cite the passages you used like [1] or [2] after each point. "
    "Do not use outside knowledge. Passages may be Q&A pairs, topic definitions, summaries or raw text; combine them. "
    "If the passages cover only part of the question, answer that part and say what is not covered. "
    "If they contain nothing relevant, set \"answer\" to exactly "
    f"\"{NO_ANSWER}\" and \"sources\" to []. "
    'Reply as JSON: {"answer": "...", "sources": [1, 2]}'
)
MATCHES_SYSTEM = (
    "You turn retrieved passages into short, readable answers. For EACH numbered passage, write a clear answer to the question "
    "using only that passage: one or two plain sentences, or a few bullet lines starting with \"- \" when it lists things. "
    "Name articles or sections when the passage states them. Do not mention \"the passage\" and do not add outside knowledge. "
    "Keep a passage only if it directly helps answer the question; being on the same topic is not enough. Otherwise set its answer to null. "
    'Reply as JSON: {"matches": [{"n": 1, "answer": "..."}, {"n": 2, "answer": null}]}'
)
MAX_MATCHES = 10
SOURCE_KEYS = ("text", "type", "citation", "score", "document_id", "parent_id", "section", "page")


def _numbered(passages: list[dict]) -> str:
    """The LLM reads the parent section (`context`) when the match was a small child chunk."""
    return "\n\n".join(f"[{i}] ({r['citation']}{' · ' + r['section'] if r.get('section') else ''}) {r.get('context') or r['text']}" for i, r in enumerate(passages, 1))


def _answer(llm, query: str, passages: list[dict]) -> dict:
    reply = llm.json_call(ANSWER_SYSTEM, f"Context:\n{_numbered(passages)}\n\nQuestion: {query}")
    answer = reply.get("answer")
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("The LLM returned no answer")

    cited = sorted({n for n in reply.get("sources", []) if isinstance(n, int) and 1 <= n <= len(passages)})
    abstained = answer.strip() == NO_ANSWER
    if not cited and not abstained:
        cited = list(range(1, len(passages) + 1))  # the model did not say which; show everything it was given
    sources = [] if abstained else [{"n": n, **{k: passages[n - 1].get(k) for k in SOURCE_KEYS}} for n in cited]
    return {"answer": answer.strip(), "abstained": abstained, "sources": sources}


def _readable_matches(llm, query: str, passages: list[dict]) -> dict[int, str]:
    """One LLM call that rewrites every passage as a short readable answer; passages that do not help are left out."""
    reply = llm.json_call(MATCHES_SYSTEM, f"Passages:\n{_numbered(passages)}\n\nQuestion: {query}")
    return {
        m["n"]: m["answer"].strip()
        for m in reply.get("matches", [])
        if isinstance(m, dict) and isinstance(m.get("n"), int) and 1 <= m["n"] <= len(passages) and isinstance(m.get("answer"), str) and m["answer"].strip()
    }


def answer_question(llm, query: str, results: list[dict], min_score: float) -> dict:
    """Grounded answer: the retrieved passages are the only context the LLM sees.

    Abstains without calling the LLM when nothing relevant was retrieved. Returns
    {"answer", "abstained", "sources", "matches"}. Sources are the passages the answer cites, numbered as in the answer.
    Matches are the retrieved passages (at most MAX_MATCHES), numbered as in the answer, each rewritten as a readable
    answer and flagged `cited` if the answer used it. A cited passage always has a match, with its original `text` if
    it could not be rewritten. They come from a second LLM call that runs alongside the first;
    if it fails the answer is still returned, with no matches.
    """
    passages = [r for r in results if r["score"] >= min_score]
    if not passages:
        return {"answer": NO_ANSWER, "abstained": True, "sources": [], "matches": []}

    with ThreadPoolExecutor(2) as pool:
        answer_job = pool.submit(_answer, llm, query, passages)
        matches_job = pool.submit(_readable_matches, llm, query, passages)
        result = answer_job.result()  # an answer failure is the caller's error
        try:
            readable = matches_job.result()
        except Exception as exc:
            logger.error("Readable matches failed: %s", exc)
            readable = {}

    cited = {s["n"] for s in result["sources"]}
    # every cited passage gets a card, so each [n] in the answer has something to open; one the rewrite call
    # dropped has no readable answer, so its original text is sent instead
    result["matches"] = [
        {"n": n, "answer": readable.get(n), "cited": n in cited, **{k: passages[n - 1].get(k) for k in SOURCE_KEYS if k != "text" or n not in readable}}
        for n in range(1, len(passages) + 1)
        if n in readable or n in cited
    ][:MAX_MATCHES]
    return result
