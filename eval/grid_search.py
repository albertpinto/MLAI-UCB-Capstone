"""Grid search of the retrieval settings on all labelled questions, saved for the analysis notebook.

Runs the retrievers of eval/eval.py on every labelled question (train and test), then sweeps the settings that are not learned from data:
BM25 (k1, b), the hybrid's reciprocal-rank-fusion constant, and how many chunks the cross-encoder re-scores. Writes
  notebooks/data/eval_results.json   the evaluation of the named retrievers: per-question ranks for all questions, and the summary on the test split
  notebooks/data/grid_results.csv    one row per setting and question
Only one document is searched (the largest in the index, or --title), so other documents in the index are not distractors.

Run: uv run --group eval python eval/grid_search.py [--title "Constitution of India (2024).pdf"]
"""
import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi

from capstone import evaluation as ev
from capstone.chunking import part_key
from capstone.config import get_settings
from capstone.store import Embedder, Store

OUT = Path("notebooks/data")
K = 5
BM25_K1, BM25_B = (0.9, 1.2, 1.5, 2.0), (0.5, 0.75, 0.9)
RRF_KS = (10, 30, 60, 100)
RERANK_DEPTHS = (10, 20, 30, 50, 100)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--title", default="", help="document to search; default is the one with the most chunks")
    args = parser.parse_args()

    settings = get_settings()
    embedders = {}
    factory = lambda name: embedders.setdefault(name, Embedder(name, settings.embed_device, settings.embed_threads))
    store = Store(settings.qdrant_path or settings.qdrant_url, settings.collection, factory(settings.embed_model))
    corpus = ev.load_corpus(store)
    titles = {d["document_id"]: d["title"] for d in store.documents()}
    chosen = {i for i, t in titles.items() if t == args.title} if args.title else {Counter(d for d, _ in corpus.ids).most_common(1)[0][0]}
    corpus = ev.corpus_of(corpus, chosen)
    print(f"{titles[next(iter(chosen))]}: {len(corpus.texts)} chunks in {len(corpus.sections)} sections")

    questions = ev.load_questions(settings.eval_questions_file)
    resolved = [(q, ev.relevant_sections(corpus, q)) for q in questions]
    resolved = [(q, rel) for q, rel in resolved if rel]
    print(f"{len(resolved)} of {len(questions)} questions have their evidence in the text")

    # The named retrievers, as the Dashboard compares them.
    scorers = ev.build_scorers(settings, corpus, ev.available_retrievers(settings), factory)
    result = ev.run_evaluation(corpus, questions, scorers, K, "all")
    for row in result["per_question"]:
        row.pop("top")  # the notebook needs ranks, not the top results shown in the UI
    test = ev.run_evaluation(corpus, questions, scorers, K, "test")  # the headline table: Recall, MRR and nDCG on the held-out questions
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "eval_results.json").write_text(json.dumps({**result, "test_summary": test["retrievers"], "document": titles[next(iter(chosen))]}))

    # The grid: every setting scored on every question.
    base = scorers["base"]
    reranker = ev.default_reranker(settings.rerank_model)
    tokens = [re.findall(r"\w+", t.lower()) for t in corpus.texts]
    bm25 = lambda k1, b: (lambda index: lambda q: index.get_scores(re.findall(r"\w+", q.lower())))(BM25Okapi(tokens, k1=k1, b=b))
    configs = [("base", {}, base)]
    configs += [("bm25", {"k1": k1, "b": b}, bm25(k1, b)) for k1 in BM25_K1 for b in BM25_B]
    configs += [("hybrid", {"rrf_k": r}, ev.rrf_scorer([bm25(1.5, 0.75), base])) for r in RRF_KS]
    configs += [("base_rerank", {"depth": d}, ev.rerank_scorer(base, reranker, corpus.texts, d)) for d in RERANK_DEPTHS]

    rows = []
    for family, params, scorer in configs:
        ev.RRF_K = params.get("rrf_k", 60)  # the hybrid reads this constant each time it scores
        for q, rel in resolved:
            ranked = ev.section_ranking(scorer(q["question"]), corpus.keys)
            recall, mrr, ndcg = ev.metrics(ranked, rel, K)
            first = next((i + 1 for i, key in enumerate(ranked) if key in rel), np.nan)
            part = part_key(corpus.sections[sorted(rel)[0]]["section"])
            rows.append({"family": family, **{p: params.get(p, np.nan) for p in ("k1", "b", "rrf_k", "depth")}, "id": q["id"], "split": q.get("split", "test"), "part": part,
                         "rank": first, "recall": recall, "mrr": mrr, "ndcg": ndcg})
    ev.RRF_K = 60
    grid = pd.DataFrame(rows)
    grid.to_csv(OUT / "grid_results.csv", index=False)

    summary = grid.groupby(["family", "k1", "b", "rrf_k", "depth"], dropna=False)[["recall", "mrr", "ndcg"]].mean().round(3)
    print(summary.to_string())


if __name__ == "__main__":
    main()
