"""Fine-tune the embedding model on the corpus's own generated questions.

Training examples are (question, the chunk it was generated from, a hard negative). The negative is the most similar chunk
from a different Part of the document, so the model learns to tell neighbouring topics apart.

No leakage: the Parts that hold a `test` evaluation question are held out. Their chunks are neither positives nor negatives, so the
model never sees the text the test questions are answered from.
"""
import random

import numpy as np

from .chunking import part_key as part_of
from .evaluation import load_corpus, relevant_sections


def held_out_parts(corpus, questions: list[dict]) -> set[str]:
    parts = set()
    for q in questions:
        if q.get("split", "test") == "test":
            parts |= {part_of(corpus.sections[k]["section"]) for k in relevant_sections(corpus, q)}
    return parts


def build_triplets(store, questions: list[dict], embedder, max_pairs: int | None = None) -> tuple[list[tuple[str, str, str]], set[str]]:
    """Returns ((question, chunk text, hard-negative chunk text) list, the held-out Parts)."""
    corpus = load_corpus(store)
    held = held_out_parts(corpus, questions)
    chunks = store.all_payloads("chunk")
    text = {(c["document_id"], c["chunk_id"]): (f"{c['section']}. " if c.get("section") else "") + c["content"] for c in chunks}
    usable = [(k, c) for k, c in ((k, c) for (k, c) in zip(text, chunks)) if part_of(c.get("section", "")) not in held]
    keys = [k for k, _ in usable]
    parts = np.array([part_of(c.get("section", "")) for _, c in usable])
    chunk_vectors = np.array(embedder.encode([text[k] for k in keys]))
    index = {k: i for i, k in enumerate(keys)}

    pairs = [qa for qa in store.all_payloads("qa_pair") if (qa["document_id"], qa["chunk_id"]) in index]
    if max_pairs:
        random.Random(0).shuffle(pairs)
        pairs = pairs[:max_pairs]
    question_vectors = np.array(embedder.encode([qa["question"] for qa in pairs]))
    triplets = []
    for qa, vector in zip(pairs, question_vectors):
        own = index[(qa["document_id"], qa["chunk_id"])]
        scores = chunk_vectors @ vector
        scores[parts == parts[own]] = -1.0  # a negative must come from another Part
        negative = int(np.argmax(scores))
        if scores[negative] > -1.0:
            triplets.append((qa["question"], text[keys[own]], text[keys[negative]]))
    return triplets, held


def batches(triplets: list[tuple[str, str, str]], size: int, seed: int):
    """Shuffled batches in which no chunk appears twice as a positive, so in-batch negatives are never the right answer."""
    rng = random.Random(seed)
    pool = triplets[:]
    rng.shuffle(pool)
    while pool:
        batch, seen, rest = [], set(), []
        for t in pool:
            if t[1] not in seen and len(batch) < size:
                batch.append(t)
                seen.add(t[1])
            else:
                rest.append(t)
        yield batch
        pool = rest


def train(triplets, base_model: str, out_dir: str, epochs: int = 2, batch_size: int = 32, lr: float = 2e-5, device: str = "", threads: int = 4, log=print) -> None:
    import torch
    from sentence_transformers import SentenceTransformer, losses

    if threads:
        torch.set_num_threads(threads)
    model = SentenceTransformer(base_model, device=device or None)
    model.max_seq_length = 128
    loss_fn = losses.MultipleNegativesRankingLoss(model)  # anchor vs every positive and hard negative in the batch
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    steps = epochs * -(-len(triplets) // batch_size)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: min(1.0, (s + 1) / max(steps // 10, 1)) * max(0.0, 1 - s / steps))
    model.train()
    step = 0
    for epoch in range(epochs):
        for batch in batches(triplets, batch_size, seed=epoch):
            features = [{k: v.to(model.device) if hasattr(v, "to") else v for k, v in model.tokenize(list(column)).items()} for column in zip(*batch)]
            loss = loss_fn(features, None)
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            step += 1
            if step % 10 == 0 or step == steps:
                log(f"epoch {epoch + 1}/{epochs} step {step}/{steps} loss {loss.item():.4f}")
    model.eval()
    model.save(out_dir)
