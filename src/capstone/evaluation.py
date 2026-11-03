"""Retrieval evaluation: BM25 against dense retrievers, on labelled questions. Used by eval/eval.py and by /api/eval/*.

A question is labelled with `evidence`: phrases copied from the text that answers it. A section (parent chunk) is relevant
if its text contains any of them, so labels survive a re-chunk or a re-ingest. Retrievers rank child chunks; a section is
ranked by its best chunk, and the metrics are computed on that section ranking.
"""
import json
import math
import re
from dataclasses import dataclass, field

import numpy as np
from rank_bm25 import BM25Okapi

from . import graph as kg
from .store import DEFAULT_MODEL

RETRIEVERS = {
    "bm25": "BM25", "base": "Base MiniLM (un-tuned)", "tuned": "Fine-tuned MiniLM",
    "hybrid": "Hybrid (BM25 + base, RRF)", "rerank": "Hybrid + cross-encoder rerank", "base_rerank": "Base + cross-encoder rerank",
    "base_graph": "Base + knowledge graph", "base_rerank_graph": "Base + knowledge graph + cross-encoder rerank",
}
NEEDS_GRAPH = ("base_graph", "base_rerank_graph")
RRF_K = 60  # reciprocal rank fusion constant
RERANK_TOP = 30  # chunks the cross-encoder re-scores
MAX_CURVE_K = 20
TOP_SHOWN = 5


@dataclass
class Corpus:
    texts: list[str]  # what is searched, per chunk: section heading + resolved text (as embedded at ingest)
    keys: list[str]  # the section each chunk belongs to, "document_id:parent_id"
    sections: dict[str, dict]  # key -> {section, page, text}
    ids: list[tuple[str, int]] = field(default_factory=list)  # (document_id, chunk_id) per chunk, to find the chunks the graph points at


def corpus_of(corpus: Corpus, document_ids: set[str]) -> Corpus:
    """Only the chunks and sections of the given documents, so another document in the same index does not act as a distractor."""
    keep = [i for i, (document_id, _) in enumerate(corpus.ids) if document_id in document_ids]
    keys = [corpus.keys[i] for i in keep]
    return Corpus([corpus.texts[i] for i in keep], keys, {k: corpus.sections[k] for k in dict.fromkeys(keys)}, [corpus.ids[i] for i in keep])


def normalise(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def load_corpus(store) -> Corpus:
    texts, keys, sections, ids = [], [], {}, []
    for c in store.all_payloads("chunk"):
        ids.append((c["document_id"], c["chunk_id"]))
        key = f"{c['document_id']}:{c['parent_id']}"
        texts.append((f"{c['section']}. " if c.get("section") else "") + c["content"])
        keys.append(key)
        sections.setdefault(key, {"section": c.get("section", ""), "page": c.get("page"), "text": c["parent_text"], "norm": normalise(c["parent_text"])})
    return Corpus(texts, keys, sections, ids)


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


def positions(scores) -> np.ndarray:
    """Rank of every item (0 = best) under `scores`."""
    order = np.argsort(-np.asarray(scores))
    ranks = np.empty(len(order))
    ranks[order] = np.arange(len(order))
    return ranks


def rrf_scorer(scorers: list):
    """Reciprocal rank fusion: each retriever votes 1 / (60 + rank) for every chunk."""
    return lambda q: sum(1.0 / (RRF_K + positions(score(q))) for score in scorers)


def graph_scorer(first, graph, corpus: Corpus, aliases=None):
    """Graph-assisted retrieval: reciprocal rank fusion of `first`'s ranking with the chunks the question's entities lead to in the graph."""
    at = {chunk: i for i, chunk in enumerate(corpus.ids)}

    def score(q):
        out = 1.0 / (kg.RRF_K + positions(first(q)))
        for rank, (chunk, _) in enumerate(kg.graph_chunks(graph, kg.link_entities(graph, q, aliases), q)):
            if chunk in at:
                out[at[chunk]] += kg.GRAPH_WEIGHT / (kg.RRF_K + rank)
        return out
    return score


def rerank_scorer(first, reranker, texts: list[str], top: int = RERANK_TOP):
    """A cross-encoder re-scores the first stage's top chunks; the rest keep the first stage's order, below them."""
    def score(q):
        base = np.asarray(first(q), dtype=float)
        best = np.argsort(-base)[:top]
        out = base - base.max() - 1e6
        out[best] = reranker.predict([(q, texts[i]) for i in best])
        return out
    return score


def default_reranker(name: str):
    from sentence_transformers import CrossEncoder

    return CrossEncoder(name, device="cpu")


def build_scorers(settings, corpus: Corpus, names: list[str], embedder_factory, reranker_factory=default_reranker) -> dict:
    """`embedder_factory(model_name)` gives an object with encode(list[str]) -> vectors; `reranker_factory(name)` one with predict(pairs)."""
    built = {}
    graph, aliases = kg.load(settings.graph_path), kg.load_aliases(settings.graph_aliases_file)

    def get(name):
        if name not in built:
            if name == "bm25":
                built[name] = bm25_scorer(corpus.texts)
            elif name == "base":
                built[name] = dense_scorer(DEFAULT_MODEL, embedder_factory(DEFAULT_MODEL), corpus.texts)
            elif name == "tuned":
                built[name] = dense_scorer(settings.fine_tuned_model, embedder_factory(settings.fine_tuned_model), corpus.texts)
            elif name == "hybrid":
                built[name] = rrf_scorer([get("bm25"), get("base")])
            elif name == "rerank":
                built[name] = rerank_scorer(get("hybrid"), reranker_factory(settings.rerank_model), corpus.texts, settings.rerank_candidates)
            elif name == "base_rerank":  # what the Ask page does: dense search, then a cross-encoder over its top results
                built[name] = rerank_scorer(get("base"), reranker_factory(settings.rerank_model), corpus.texts, settings.rerank_candidates)
            elif name == "base_graph":
                built[name] = graph_scorer(get("base"), graph, corpus, aliases)
            elif name == "base_rerank_graph":  # the graph adds candidates before the cross-encoder re-orders them
                built[name] = rerank_scorer(get("base_graph"), reranker_factory(settings.rerank_model), corpus.texts, settings.rerank_candidates)
        return built[name]

    return {name: get(name) for name in names if name in available_retrievers(settings)}


def available_retrievers(settings) -> list[str]:
    return [n for n in RETRIEVERS if (n != "tuned" or settings.fine_tuned_model) and (n not in ("rerank", "base_rerank", "base_rerank_graph") or settings.rerank_model)
            and (n not in NEEDS_GRAPH or settings.graph_path.is_file())]


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
