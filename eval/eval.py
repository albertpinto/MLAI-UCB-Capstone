"""Compare BM25, the un-tuned MiniLM and (if FINE_TUNED_MODEL is set) the fine-tuned model on the labelled questions.

Runs on the chunks stored in Qdrant (the COLLECTION in .env), through the same functions as the dashboard's /api/eval/run.
A question counts as answered where a section containing its evidence first appears in the ranking.

Run: uv run python eval/eval.py [--k 5] [--split test|train|all]
"""
import argparse

from capstone.config import get_settings
from capstone.evaluation import RETRIEVERS, available_retrievers, build_scorers, load_corpus, load_questions, run_evaluation
from capstone.store import Embedder, Store


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--split", default="test", choices=["test", "train", "all"])
    args = parser.parse_args()

    settings = get_settings()
    embedders = {}
    factory = lambda name: embedders.setdefault(name, Embedder(name, settings.embed_device, settings.embed_threads))
    corpus = load_corpus(Store(settings.qdrant_url, settings.collection, factory(settings.embed_model)))
    scorers = build_scorers(settings, corpus, available_retrievers(settings), factory)
    result = run_evaluation(corpus, load_questions(settings.eval_questions_file), scorers, args.k, args.split)

    print(f"{result['questions']} {args.split} questions, {result['chunks']} chunks in {result['sections']} sections", end="")
    print(f" ({len(result['unresolved'])} questions skipped: evidence not found in the index)\n" if result["unresolved"] else "\n")
    print(f"{'retriever':<46} {f'Recall@{args.k}':>9} {'MRR':>7} {f'nDCG@{args.k}':>8}")
    for r in result["retrievers"].values():
        print(f"{r['label']:<46} {r['recall']:>9.3f} {r['mrr']:>7.3f} {r['ndcg']:>8.3f}")


if __name__ == "__main__":
    main()
