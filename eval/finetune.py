"""Fine-tune the embedding model on the generated questions, holding out the Parts of the test questions.

Run: uv run python eval/finetune.py [--epochs 2] [--out models/finetuned-minilm]
Then set FINE_TUNED_MODEL=models/finetuned-minilm in .env and run the evaluation. The indexed vectors are not changed; to *search* with the
tuned model, set EMBED_MODEL to it, use a new COLLECTION name and re-ingest.
"""
import argparse

from capstone.config import get_settings
from capstone.evaluation import load_questions
from capstone.finetune import build_triplets, train
from capstone.store import DEFAULT_MODEL, Embedder, Store


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--max-pairs", type=int, default=None, help="use only this many pairs (a quick trial)")
    parser.add_argument("--out", default="models/finetuned-minilm")
    args = parser.parse_args()

    settings = get_settings()
    base = Embedder(DEFAULT_MODEL, settings.embed_device, settings.embed_threads)
    store = Store(settings.qdrant_url, settings.collection, base)
    triplets, held = build_triplets(store, load_questions(settings.eval_questions_file), base, args.max_pairs)
    print(f"{len(triplets)} training triplets; held out for testing: {', '.join(sorted(held)) or 'nothing'}")
    train(triplets, DEFAULT_MODEL, args.out, args.epochs, args.batch, device=settings.embed_device, threads=settings.embed_threads)
    print(f"Saved to {args.out}. Set FINE_TUNED_MODEL={args.out} to include it in the evaluation.")


if __name__ == "__main__":
    main()
