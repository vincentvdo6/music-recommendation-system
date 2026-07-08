"""
Offline A/B over the held-out eval sample exported by the training notebook.

Compares, on identical candidate sets and features:
  - the shipped LightGBM v2 ranker (if present)
  - the LinearFallbackRanker (what serves when the model is absent)
  - seed-cosine-only ordering (equivalent to the old deployed 1-feature model)
  - raw retrieval order

Usage: python scripts/evaluate.py [evaluation/eval_sample.parquet]
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from services.recommendation.features import FEATURE_NAMES  # noqa: E402
from services.recommendation.ranker import LightGBMRanker, LinearFallbackRanker  # noqa: E402

RANKER_PATH = ROOT / "models" / "ranker" / "lightgbm_ranker_v2.txt"


def group_metrics(scores: np.ndarray, labels: np.ndarray, ks=(10, 50)) -> dict:
    order = np.argsort(-scores)
    rel = labels[order]
    out = {}
    for k in ks:
        topk = rel[:k]
        out[f"recall@{k}"] = float(topk.sum() / labels.sum())
        dcg = float((topk / np.log2(np.arange(2, len(topk) + 2))).sum())
        ideal = np.sort(labels)[::-1][:k]
        idcg = float((ideal / np.log2(np.arange(2, len(ideal) + 2))).sum())
        out[f"ndcg@{k}"] = dcg / idcg if idcg > 0 else 0.0
    return out


def main() -> int:
    sample_path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "evaluation" / "eval_sample.parquet"
    if not sample_path.exists():
        print(f"{sample_path} not found — run the Kaggle notebook and scripts/install_artifacts.py first")
        return 1

    df = pd.read_parquet(sample_path)
    fallback = LinearFallbackRanker()

    scorers = {
        "linear-fallback": lambda X: fallback.predict(X),
        "seed-cosine-only": lambda X: X["seed_i2v_cos"].to_numpy(),
        "retrieval-order": lambda X: -np.arange(len(X), dtype=np.float64),
    }
    if RANKER_PATH.exists():
        v2 = LightGBMRanker(str(RANKER_PATH))
        scorers = {"lightgbm-v2": lambda X: v2.predict(X), **scorers}
    else:
        print(f"note: {RANKER_PATH} not present — comparing baselines only\n")

    results = {name: [] for name in scorers}
    for _, group in df.groupby("qid"):
        X, y = group[FEATURE_NAMES], group["label"].to_numpy()
        if y.sum() == 0:
            continue
        for name, fn in scorers.items():
            results[name].append(group_metrics(np.asarray(fn(X), dtype=np.float64), y))

    n_groups = len(next(iter(results.values())))
    table = pd.DataFrame({
        name: {k: np.mean([m[k] for m in ms]) for k in ms[0]}
        for name, ms in results.items()
    }).T.round(4)
    print(f"{n_groups} evaluation groups\n")
    print(table.to_string())

    if "lightgbm-v2" in table.index:
        beats = (table.loc["lightgbm-v2"] >= table.loc["seed-cosine-only"]).all()
        print("\ngate:", "PASS — v2 >= seed-cosine baseline on all metrics"
              if beats else "FAIL — keep the linear fallback and iterate on the notebook")
        return 0 if beats else 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
