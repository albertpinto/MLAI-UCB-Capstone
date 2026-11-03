"""The statistics behind the retrieval evaluation, for the Analysis page (/api/analysis).

Reads what eval/grid_search.py wrote: `eval_results.json` (for every question, the rank at which each retriever first shows a correct section) and
`grid_results.csv` (every setting of BM25, the hybrid and the reranker, scored on every question). The same analysis, with pandas and charts, is in
notebooks/capstone_analysis.ipynb; resampling here uses a fixed seed, so intervals can differ from the notebook's in the third decimal.
"""
import csv
import json
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.model_selection import GroupKFold

K = 5
RESAMPLES = 10_000
FOLDS = 5
LABELS = {"bm25": "BM25", "base": "Base", "tuned": "Fine-tuned", "hybrid": "Hybrid", "rerank": "Hybrid + rerank", "base_rerank": "Base + rerank",
          "base_graph": "Base + graph", "base_rerank_graph": "Base + graph + rerank"}
TEST_PAIRS = [("base_rerank", "bm25"), ("base", "bm25"), ("base_rerank", "base"), ("tuned", "base"), ("base_graph", "base"), ("hybrid", "base")]
# The fine-tuned model was trained on the train Parts, so it is compared on the test Parts only; the others can be compared on every question.
ALL_PAIRS = [("base_rerank", "bm25"), ("base", "bm25"), ("base_rerank", "base"), ("base_graph", "base"), ("hybrid", "base")]
DEFAULTS = {"bm25": "k1=1.5, b=0.75", "hybrid": "rrf_k=60", "base_rerank": "depth=30", "base": "as downloaded"}  # the settings the application uses
PARAMS = ("k1", "b", "rrf_k", "depth")
METRICS = ("recall", "mrr", "ndcg")


def bootstrap(values, seed: int = 0) -> list[float]:
    """95% interval of the mean, by resampling the questions with replacement."""
    values = np.asarray(values, dtype=float)
    means = np.random.default_rng(seed).choice(values, size=(RESAMPLES, len(values))).mean(axis=1)
    return [float(x) for x in np.percentile(means, [2.5, 97.5])]


def paired(a: str, b: str, rr: dict[str, dict[str, float]], ids: list[str]) -> dict:
    """Retriever a minus retriever b by reciprocal rank on the same questions: mean difference, bootstrap interval, Wilcoxon signed-rank p-value."""
    diff = np.array([rr[a][i] - rr[b][i] for i in ids])
    low, high = bootstrap(diff)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # ties among the differences make scipy fall back to an approximation
        p = float(stats.wilcoxon(diff).pvalue) if np.any(diff != 0) else 1.0
    return {"first": a, "second": b, "questions": len(ids), "difference": float(diff.mean()), "ci": [low, high], "p": p, "real": low > 0 or high < 0}


def _comparisons(pairs, rr, ids) -> list[dict]:
    rows = [paired(a, b, rr, ids) for a, b in pairs]
    for row in rows:
        row["p_adjusted"] = min(1.0, row["p"] * len(rows))  # Bonferroni: several comparisons, so a lone small p-value means less
    return rows


def _number(text: str):
    return float(text) if text not in ("", "nan") else None


def _config(row: dict) -> str:
    return ", ".join(f"{p}={row[p]:g}" for p in PARAMS if row[p] is not None) or "as downloaded"


def load_grid(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for row in csv.DictReader(f):
            row = {**row, **{p: _number(row[p]) for p in PARAMS}, **{m: float(row[m]) for m in METRICS}}
            row["config"] = _config(row)
            rows.append(row)
    return rows


def _mean(rows: list[dict]) -> dict:
    return {m: float(np.mean([r[m] for r in rows])) for m in METRICS}


def grid_summary(rows: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str], list[dict]] = {}
    for r in rows:
        groups.setdefault((r["family"], r["config"]), []).append(r)
    return [{"family": family, "config": config, "params": {p: g[0][p] for p in PARAMS if g[0][p] is not None}, "questions": len(g), **_mean(g)}
            for (family, config), g in groups.items()]


def cross_validate(rows: list[dict]) -> dict:
    """Grid search inside cross-validation, grouped by Part: the setting with the best mean MRR on the training folds is scored on the held-out fold,
    so a setting is never tuned on a Part it is judged on. Every question is held out exactly once."""
    questions = [(r["id"], r["part"]) for r in rows if r["family"] == "base"]
    ids = [q for q, _ in questions]
    parts = [p for _, p in questions]
    by_cell = {(r["family"], r["config"], r["id"]): r for r in rows}
    configs = {f: sorted({r["config"] for r in rows if r["family"] == f}) for f in {r["family"] for r in rows}}

    held_out = {f: {} for f in configs}  # family -> question id -> metrics of the chosen setting
    chosen = {f: [] for f in configs}
    fold_mrr = {f: [] for f in configs}
    held_parts = []
    for train_at, test_at in GroupKFold(n_splits=min(FOLDS, len(set(parts)))).split(ids, groups=parts):
        train, test = [ids[i] for i in train_at], [ids[i] for i in test_at]
        held_parts.append(sorted({parts[i] for i in test_at}))
        for family, names in configs.items():
            best, best_mrr = None, -1.0
            for config in names:  # sorted, and the first of equal scores wins
                mrr = float(np.mean([by_cell[(family, config, i)]["mrr"] for i in train]))
                if mrr > best_mrr:
                    best, best_mrr = config, mrr
            chosen[family].append(best)
            for i in test:
                held_out[family][i] = by_cell[(family, best, i)]
            fold_mrr[family].append(float(np.mean([by_cell[(family, best, i)]["mrr"] for i in test])))

    families = []
    for family in ("bm25", "hybrid", "base_rerank", "base"):
        if family not in configs:
            continue
        default = {i: by_cell[(family, DEFAULTS[family], i)] for i in ids}
        diff = [held_out[family][i]["mrr"] - default[i]["mrr"] for i in ids]
        low, high = bootstrap(diff)
        families.append({"family": family, "label": LABELS[family], "default_setting": DEFAULTS[family], "default": _mean(list(default.values())),
                         "tuned": _mean(list(held_out[family].values())), "gain": {"mrr": float(np.mean(diff)), "ci": [low, high], "real": low > 0},
                         "chosen": chosen[family], "fold_mrr": fold_mrr[family]})
    return {"folds": len(held_parts), "held_out_parts": held_parts, "families": families}


def build(results_file: Path, grid_file: Path) -> dict:
    results = json.loads(results_file.read_text())
    per_question = results["per_question"]
    rr = {name: {q["id"]: (1 / q["ranks"][name] if q["ranks"].get(name) else 0.0) for q in per_question if name in q["ranks"]} for name in results["retrievers"]}
    test_ids = [q["id"] for q in per_question if q["split"] == "test"]

    retrievers = []
    for name, summary in results["test_summary"].items():
        hit = [float(q["ranks"][name] is not None and q["ranks"][name] <= K) for q in per_question if q["id"] in test_ids]
        first = [float(q["ranks"][name] == 1) for q in per_question if q["id"] in test_ids]
        retrievers.append({"id": name, "label": LABELS.get(name, summary["label"]), "full_label": summary["label"], "recall": summary["recall"], "mrr": summary["mrr"],
                           "ndcg": summary["ndcg"], "hit": float(np.mean(hit)), "hit_ci": bootstrap(hit), "mrr_ci": bootstrap([rr[name][i] for i in test_ids]),
                           "first": float(np.mean(first))})

    grid = load_grid(grid_file)
    every_id = [q["id"] for q in per_question]
    return {
        "document": results.get("document", ""), "chunks": results["chunks"], "sections": results["sections"], "k": K,
        "questions": {"total": len(per_question), "train": len(per_question) - len(test_ids), "test": len(test_ids)},
        "generated": datetime.fromtimestamp(grid_file.stat().st_mtime, timezone.utc).isoformat(timespec="seconds"),
        "retrievers": retrievers,
        "comparisons": {
            "test": _comparisons([p for p in TEST_PAIRS if p[0] in rr and p[1] in rr], rr, test_ids),
            "all": _comparisons([p for p in ALL_PAIRS if p[0] in rr and p[1] in rr], rr, every_id),
        },
        "grid": grid_summary(grid),
        "cross_validation": cross_validate(grid),
    }
