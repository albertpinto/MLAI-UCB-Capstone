import logging

logger = logging.getLogger(__name__)

NO_ANSWER = "I couldn't find this in the documentation."
ANSWER_SYSTEM = (
    "You answer questions for a knowledge base using ONLY the numbered context passages provided. "
    "Write a short, direct answer in plain language and cite the passages you used like [1] or [2]. "
    "Do not use outside knowledge. If the passages do not contain the answer, set \"answer\" to exactly "
    f"\"{NO_ANSWER}\" and \"sources\" to []. "
    'Reply as JSON: {"answer": "...", "sources": [1, 2]}'
)


def answer_question(llm, query: str, results: list[dict], min_score: float) -> dict:
    """Grounded answer: the retrieved passages are the only context the LLM sees.

    Abstains without calling the LLM when nothing relevant was retrieved. Returns
    {"answer", "abstained", "sources"}; sources are the passages the answer cites, numbered as in the answer.
    """
    passages = [r for r in results if r["score"] >= min_score]
    if not passages:
        return {"answer": NO_ANSWER, "abstained": True, "sources": []}

    context = "\n\n".join(f"[{i}] ({r['citation']}) {r['text']}" for i, r in enumerate(passages, 1))
    reply = llm.json_call(ANSWER_SYSTEM, f"Context:\n{context}\n\nQuestion: {query}")
    answer = reply.get("answer")
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("The LLM returned no answer")

    cited = sorted({n for n in reply.get("sources", []) if isinstance(n, int) and 1 <= n <= len(passages)})
    abstained = answer.strip() == NO_ANSWER
    if not cited and not abstained:
        cited = list(range(1, len(passages) + 1))  # the model did not say which; show everything it was given
    sources = [] if abstained else [{"n": n, **{k: passages[n - 1][k] for k in ("text", "type", "citation", "score")}} for n in cited]
    return {"answer": answer.strip(), "abstained": abstained, "sources": sources}
