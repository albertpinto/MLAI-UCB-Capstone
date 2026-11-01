"""Compare BM25, the stock MiniLM and EMBED_MODEL on questions.json.

Corpus: every chunk of every document in DOCS_DIR (same chunker as ingestion).
A question counts as answered where its labelled document first appears in the ranking.
Raw chunks only: the Q&A / topic / summary points are not part of this comparison.

Run: uv run --group eval python eval/eval.py [--k 5]
"""
import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from capstone.chunking import chunk_text
from capstone.config import get_settings
from capstone.extract import SUPPORTED, extract_text
from capstone.store import Embedder

BASE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def load_corpus(docs_dir: Path, size: int, overlap: int):
    texts, owners = [], []
    for path in sorted(docs_dir.iterdir()):
        if path.suffix.lower() in SUPPORTED:
            for chunk in chunk_text(extract_text(path.name, path.read_bytes()), size, overlap):
                texts.append(chunk)
                owners.append(path.name)
    return texts, owners


def doc_ranking(scores, owners) -> list[str]:
    """Best-first list of distinct documents, ranked by their best chunk."""
    seen = []
    for i in np.argsort(-np.asarray(scores)):
        if owners[i] not in seen:
            seen.append(owners[i])
    return seen


def metrics(ranked: list[str], relevant: str, k: int) -> tuple[float, float, float]:
    rank = ranked.index(relevant) + 1 if relevant in ranked else None
    recall = 1.0 if rank and rank <= k else 0.0
    mrr = 1.0 / rank if rank else 0.0
    ndcg = 1.0 / math.log2(rank + 1) if rank and rank <= k else 0.0  # one relevant doc, so ideal DCG = 1
    return recall, mrr, ndcg


def bm25_scorer(texts):
    tokenize = lambda s: re.findall(r"\w+", s.lower())
    index = BM25Okapi([tokenize(t) for t in texts])
    return lambda q: index.get_scores(tokenize(q))


def dense_scorer(model_name, texts):
    embedder = Embedder(model_name)
    vectors = np.array(embedder.encode(texts))
    return lambda q: vectors @ np.array(embedder.encode([q])[0])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--questions", default=str(Path(__file__).with_name("questions.json")))
    args = parser.parse_args()

    settings = get_settings()
    questions = json.loads(Path(args.questions).read_text())
    texts, owners = load_corpus(settings.docs_dir, settings.chunk_chars, settings.chunk_overlap)
    print(f"{len(questions)} questions, {len(texts)} chunks from {len(set(owners))} documents\n")

    scorers = {"BM25": bm25_scorer(texts), f"base ({BASE_MODEL})": dense_scorer(BASE_MODEL, texts)}
    if settings.embed_model != BASE_MODEL:
        scorers[f"tuned ({settings.embed_model})"] = dense_scorer(settings.embed_model, texts)

    k = args.k
    print(f"{'retriever':<60} {f'Recall@{k}':>9} {'MRR':>7} {f'nDCG@{k}':>8}")
    for name, score in scorers.items():
        rows = [metrics(doc_ranking(score(q["question"]), owners), q["document"], k) for q in questions]
        recall, mrr, ndcg = np.mean(rows, axis=0)
        print(f"{name[:60]:<60} {recall:>9.3f} {mrr:>7.3f} {ndcg:>8.3f}")


if __name__ == "__main__":
    main()
