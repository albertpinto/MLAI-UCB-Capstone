import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("eval_script", Path(__file__).parents[1] / "eval" / "eval.py")
try:
    ev = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ev)
except ModuleNotFoundError:  # rank-bm25 lives in the optional eval group
    ev = None

import pytest


@pytest.mark.skipif(ev is None, reason="eval group not installed")
def test_metrics():
    assert ev.metrics(["a", "b", "c"], "a", 3) == (1.0, 1.0, 1.0)
    recall, mrr, ndcg = ev.metrics(["a", "b", "c"], "c", 3)
    assert (recall, round(mrr, 3), round(ndcg, 3)) == (1.0, 0.333, 0.5)
    assert ev.metrics(["a", "b", "c"], "c", 2) == (0.0, 1 / 3, 0.0)
    assert ev.metrics(["a"], "zzz", 5) == (0.0, 0.0, 0.0)
