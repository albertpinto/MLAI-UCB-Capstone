"""Retrieval evaluation: BM25 against dense retrievers, on labelled questions. Used by eval/eval.py and by /api/eval/*.

A question is labelled with `evidence`: phrases copied from the text that answers it. A section (parent chunk) is relevant
if its text contains any of them, so labels survive a re-chunk or a re-ingest. Retrievers rank child chunks; a section is
ranked by its best chunk, and the metrics are computed on that section ranking.
"""
import json
import math
import re
from dataclasses import dataclass

import numpy as np
from rank_bm25 import BM25Okapi

from .store import DEFAULT_MODEL

RETRIEVERS = {"bm25": "BM25", "base": "Base MiniLM (un-tuned)", "tuned": "Fine-tuned MiniLM"}
MAX_CURVE_K = 20
TOP_SHOWN = 5


@dataclass
class Corpus:
    texts: list[str]  # what is searched, per chunk: section heading + resolved text (as embedded at ingest)
    keys: list[str]  # the section each chunk belongs to, "document_id:parent_id"
    sections: dict[str, dict]  # key -> {section, page, text}


def normalise(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def load_corpus(store) -> Corpus:
    texts, keys, sections = [], [], {}
    for c in store.all_payloads("chunk"):
        key = f"{c['document_id']}:{c['parent_id']}"
        texts.append((f"{c['section']}. " if c.get("section") else "") + c["content"])
        keys.append(key)
        sections.setdefault(key, {"section": c.get("section", ""), "page": c.get("page"), "text": c["parent_text"], "norm": normalise(c["parent_text"])})
    return Corpus(texts, keys, sections)


def relevant_sections(corpus: Corpus, question: dict) -> set[str]:
    phrases = [normalise(e) for e in question.get("evidence", []) if normalise(e)]
    return {k for k, s in corpus.sections.items() if any(p in s["norm"] for p in phrases)}


def metrics(ranked: list[str], relevant: set[str], k: int) -> tuple[float, float, float]:
    """Recall@k (share of the relevant sections found), MRR (first relevant section) and nDCG@k with binary relevance."""
    hits = [i for i, key in enumerate(ranked) if key in relevant]
    recall = len([i for i in hits if i < k]) / len(relevant)
    mrr = 1 / (hits[0] + 1) if hits else 0.0
    dcg = sum(1 / math.log2(i + 2) for i in hits if i < k)
    ideal = sum(1 / math.log2(i + 2) for i in range(min(len(relevant), k)))
    return recall, mrr, dcg / ideal


def section_ranking(scores, keys: list[str]) -> list[str]:
    """Best-first list of distinct sections, each ranked by its best chunk."""
    seen: dict[str, None] = {}
    for i in np.argsort(-np.asarray(scores)):
        seen.setdefault(keys[i], None)
    return list(seen)


def bm25_scorer(texts: list[str]):
    tokenize = lambda s: re.findall(r"\w+", s.lower())
    index = BM25Okapi([tokenize(t) for t in texts])
    return lambda q: index.get_scores(tokenize(q))


_vectors: dict[tuple[str, int], np.ndarray] = {}  # chunk vectors per (model, corpus), so repeated runs do not re-embed


def dense_scorer(model: str, embedder, texts: list[str]):
    key = (model, hash(tuple(texts)))
    if key not in _vectors:
        _vectors[key] = np.array(embedder.encode(texts))
    return lambda q: _vectors[key] @ np.array(embedder.encode([q])[0])


def build_scorers(settings, corpus: Corpus, names: list[str], embedder_factory) -> dict:
    """`embedder_factory(model_name)` gives an object with encode(list[str]) -> vectors."""
    scorers = {}
    for name in names:
        if name == "bm25":
            scorers[name] = bm25_scorer(corpus.texts)
        elif name == "base":
            scorers[name] = dense_scorer(DEFAULT_MODEL, embedder_factory(DEFAULT_MODEL), corpus.texts)
        elif name == "tuned" and settings.fine_tuned_model:
            scorers[name] = dense_scorer(settings.fine_tuned_model, embedder_factory(settings.fine_tuned_model), corpus.texts)
    return scorers


def available_retrievers(settings) -> list[str]:
    return [n for n in RETRIEVERS if n != "tuned" or settings.fine_tuned_model]


def load_questions(path) -> list[dict]:
    return json.loads(path.read_text()) if path.is_file() else []


def run_evaluation(corpus: Corpus, questions: list[dict], scorers: dict, k: int = 5, split: str = "test", progress=None) -> dict:
    """Score every retriever on the questions of `split` ("test", "train" or "all"). Questions whose evidence matches no section
    are reported as unresolved and left out."""
    chosen = [q for q in questions if split == "all" or q.get("split", "test") == split]
    resolved = [(q, relevant_sections(corpus, q)) for q in chosen]
    unresolved = [q["id"] for q, rel in resolved if not rel]
    resolved = [(q, rel) for q, rel in resolved if rel]

    per_question = [{"id": q["id"], "question": q["question"], "split": q.get("split", "test"),
                     "relevant": [corpus.sections[r]["section"] or r for r in sorted(rel)], "ranks": {}, "top": {}} for q, rel in resolved]
    summary = {}
    for step, (name, score) in enumerate(scorers.items()):
        rows, curve = [], np.zeros(MAX_CURVE_K)
        for row, (q, rel) in zip(per_question, resolved):
            ranked = section_ranking(score(q["question"]), corpus.keys)
            rows.append(metrics(ranked, rel, k))
            curve += [metrics(ranked, rel, j)[0] for j in range(1, MAX_CURVE_K + 1)]
            row["ranks"][name] = next((i + 1 for i, key in enumerate(ranked) if key in rel), None)
            row["top"][name] = [{"section": corpus.sections[key]["section"], "page": corpus.sections[key]["page"], "hit": key in rel} for key in ranked[:TOP_SHOWN]]
        recall, mrr, ndcg = np.mean(rows, axis=0) if rows else (0.0, 0.0, 0.0)
        summary[name] = {"label": RETRIEVERS[name], "recall": float(recall), "mrr": float(mrr), "ndcg": float(ndcg),
                         "curve": (curve / max(len(rows), 1)).tolist()}
        if progress:
            progress(step + 1, len(scorers), RETRIEVERS[name])
    return {"k": k, "split": split, "chunks": len(corpus.texts), "sections": len(corpus.sections), "questions": len(resolved),
            "unresolved": unresolved, "retrievers": summary, "per_question": per_question}
