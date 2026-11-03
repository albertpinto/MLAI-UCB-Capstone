import csv
import json
from pathlib import Path

import numpy as np
import pytest

from capstone import analysis

REAL = Path(__file__).resolve().parent.parent / "notebooks" / "data"  # the data the notebook was run on


def test_bootstrap_interval_contains_the_mean_and_is_repeatable():
    values = [0.0, 1.0, 0.5, 0.25, 1.0, 0.0, 0.5, 1.0]
    low, high = analysis.bootstrap(values)
    assert low < np.mean(values) < high and analysis.bootstrap(values) == [low, high]


def test_paired_finds_a_consistent_gap_and_ignores_none():
    rr = {"a": {str(i): 1.0 for i in range(20)}, "b": {str(i): 0.5 for i in range(20)}}
    row = analysis.paired("a", "b", rr, list(rr["a"]))
    assert row["difference"] == pytest.approx(0.5) and row["real"] and row["p"] < 0.001
    same = analysis.paired("a", "a", rr, list(rr["a"]))
    assert same["difference"] == 0 and not same["real"] and same["p"] == 1.0


def write_data(folder: Path, parts=("Part I", "Part II", "Part III", "Part IV", "Part V", "Part VI")):
    """Six Parts with two questions each; 'good' always ranks the answer first, 'poor' fifth; the reranker's depth 20 is best."""
    questions = [(f"q{n}", parts[n // 2], "test" if n % 2 else "train") for n in range(12)]
    results = {"chunks": 100, "sections": 10, "document": "Doc.pdf", "retrievers": {"bm25": {}, "base": {}},
               "per_question": [{"id": q, "split": split, "ranks": {"bm25": 5, "base": 1}} for q, _, split in questions],
               "test_summary": {name: {"label": name, "recall": 0.5, "mrr": 0.5, "ndcg": 0.5} for name in ("bm25", "base")}}
    (folder / "eval_results.json").write_text(json.dumps(results))
    with (folder / "grid_results.csv").open("w", newline="") as f:
        out = csv.writer(f)
        out.writerow(["family", "k1", "b", "rrf_k", "depth", "id", "split", "part", "rank", "recall", "mrr", "ndcg"])
        for q, part, split in questions:
            out.writerow(["base", "", "", "", "", q, split, part, 1, 1, 1.0, 1.0])
            for depth, mrr in ((10, 0.5), (20, 0.9), (30, 0.6)):
                out.writerow(["base_rerank", "", "", "", depth, q, split, part, 1, 1, mrr, mrr])
            for k1 in (1.5, 2.0):
                out.writerow(["bm25", k1, 0.75, "", "", q, split, part, 5, 1, 0.2, 0.2])


def test_build_reports_every_part_of_the_analysis(tmp_path):
    write_data(tmp_path)
    report = analysis.build(tmp_path / "eval_results.json", tmp_path / "grid_results.csv")
    assert report["questions"] == {"total": 12, "train": 6, "test": 6} and report["document"] == "Doc.pdf"
    by_id = {r["id"]: r for r in report["retrievers"]}
    assert by_id["base"]["hit"] == 1.0 and by_id["bm25"]["hit"] == 1.0 and by_id["base"]["first"] == 1.0 and by_id["bm25"]["first"] == 0.0
    assert by_id["base"]["mrr_ci"] == [1.0, 1.0]
    gap = report["comparisons"]["test"][0]  # base vs BM25: the only pair both of whose retrievers exist here
    assert gap["difference"] == pytest.approx(0.8) and gap["real"]
    assert {g["config"] for g in report["grid"] if g["family"] == "base_rerank"} == {"depth=10", "depth=20", "depth=30"}


def test_cross_validation_chooses_the_best_setting_without_seeing_the_held_out_part(tmp_path):
    write_data(tmp_path)
    cv = analysis.build(tmp_path / "eval_results.json", tmp_path / "grid_results.csv")["cross_validation"]
    rerank = next(f for f in cv["families"] if f["family"] == "base_rerank")
    assert cv["folds"] == 5 and sorted(sum(cv["held_out_parts"], [])) == sorted(f"Part {r}" for r in ("I", "II", "III", "IV", "V", "VI"))  # each Part is held out once
    assert rerank["chosen"] == ["depth=20"] * 5 and rerank["tuned"]["mrr"] == pytest.approx(0.9) and rerank["default"]["mrr"] == pytest.approx(0.6)
    assert rerank["gain"]["mrr"] == pytest.approx(0.3) and rerank["gain"]["real"]


@pytest.mark.skipif(not (REAL / "grid_results.csv").is_file(), reason="the notebook's data is not in this checkout")
def test_the_app_reproduces_the_numbers_in_the_notebook():
    report = analysis.build(REAL / "eval_results.json", REAL / "grid_results.csv")
    test = {(c["first"], c["second"]): c for c in report["comparisons"]["test"]}
    assert test[("base_rerank", "bm25")]["difference"] == pytest.approx(0.421, abs=0.001) and test[("base_rerank", "bm25")]["real"]
    assert test[("base_rerank", "base")]["difference"] == pytest.approx(0.084, abs=0.001) and not test[("base_rerank", "base")]["real"]
    cv = {f["family"]: f for f in report["cross_validation"]["families"]}
    assert cv["base_rerank"]["tuned"]["mrr"] == pytest.approx(0.791, abs=0.001) and cv["base_rerank"]["default"]["mrr"] == pytest.approx(0.774, abs=0.001)
    assert cv["bm25"]["tuned"]["mrr"] == pytest.approx(0.414, abs=0.001) and cv["hybrid"]["tuned"]["mrr"] == pytest.approx(0.581, abs=0.001)
    assert cv["base_rerank"]["chosen"] == ["depth=20"] * 5 and cv["base"]["tuned"]["mrr"] == pytest.approx(0.728, abs=0.001)


def test_the_api_serves_the_report_and_the_notebook(client, settings):
    assert client.get("/api/analysis").status_code == 404 and client.get("/api/analysis/notebook").status_code == 404
    settings.analysis_dir.mkdir()
    write_data(settings.analysis_dir)
    settings.notebook_file.write_text("{}")
    body = client.get("/api/analysis").json()
    assert body["questions"]["total"] == 12 and body["cross_validation"]["folds"] == 5
    assert client.get("/api/analysis/notebook").content == b"{}"
