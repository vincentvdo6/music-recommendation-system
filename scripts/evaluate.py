"""
Offline A/B over the held-out eval sample exported by the training notebook.

Compares, on identical candidate sets and features:
  - the shipped LightGBM ranker, raw AND through the exact served policy
    (shared ScoringPolicy: standardization + dials + eligible-first floor)
  - the LinearFallbackRanker through the same served policy
  - seed-cosine-only ordering (equivalent to the old deployed 1-feature model)
  - raw retrieval order

Caveats vs production: the service's Spotify-availability losses and final
one-per-artist rule are not simulated here (the eval sample carries no
candidate identities); the training notebook's manifest covers those.

Usage: python scripts/evaluate.py [evaluation/eval_sample.parquet]
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from services.recommendation.features import FEATURE_NAMES, MOOD_DIMS, effective_seed_cos  # noqa: E402
from services.recommendation.policy import load_policy  # noqa: E402
from services.recommendation.ranker import LightGBMRanker, LinearFallbackRanker  # noqa: E402

RANKER_PATH = ROOT / "models" / "ranker" / "lightgbm_ranker_v2.txt"
POLICY = load_policy()  # frozen artifact policy + the same env overrides serving honors

# "Subsystem absent" conventions per feature (see features.build_matrix).
# Pre-v4 eval samples lack the channel indicators: those candidates all came
# from the i2v channel, so has_i2v defaults to 1 and audio_seed_rr to 0.
MISSING_DEFAULTS = {**{f"{d}_diff": 0.5 for d in MOOD_DIMS}, "mood_sim": 0.5, "duration_diff": 1.0,
                    "has_i2v": 1.0, "audio_seed_rr": 0.0}


def ordered_metrics(order: np.ndarray, labels: np.ndarray, seed_cos: np.ndarray,
                    audio_cos: np.ndarray = None, ks=(10, 50)) -> dict:
    """Metrics for an explicit candidate ORDER (so floor/selection effects count)."""
    rel = labels[order]
    out = {}
    for k in ks:
        topk = rel[:k]
        out[f"recall@{k}"] = float(topk.sum() / labels.sum())
        dcg = float((topk / np.log2(np.arange(2, len(topk) + 2))).sum())
        ideal = np.sort(labels)[::-1][:k]
        idcg = float((ideal / np.log2(np.arange(2, len(ideal) + 2))).sum())
        out[f"ndcg@{k}"] = dcg / idcg if idcg > 0 else 0.0
    out["hit@1"] = float(rel[0] > 0) if len(rel) else 0.0
    pos = np.nonzero(rel > 0)[0]
    out["mrr"] = float(1.0 / (pos[0] + 1)) if len(pos) else 0.0
    # Conditional vibe diagnostics (not quality bars, never coverage-scaled).
    out["seed_cos@10"] = float(seed_cos[order][:10].mean())
    if audio_cos is not None:
        out["audio_cos@10"] = float(audio_cos[order][:10].mean())
    return out


def raw_order(scores: np.ndarray) -> np.ndarray:
    return np.argsort(-np.asarray(scores, dtype=np.float64))


def served_order(model_scores, X: pd.DataFrame) -> np.ndarray:
    """Exactly what engine.recommend ships (full-length order for metrics)."""
    seed_cos = effective_seed_cos(X)
    blended = POLICY.blend(np.asarray(model_scores, dtype=np.float64),
                           seed_cos, X["log_pop"].to_numpy(), has_seed=True)
    return POLICY.select(blended, seed_cos, limit=len(X), has_seed=True)


def main() -> int:
    sample_path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "evaluation" / "eval_sample.parquet"
    if not sample_path.exists():
        print(f"{sample_path} not found — run the Kaggle notebook and scripts/install_artifacts.py first")
        return 1

    df = pd.read_parquet(sample_path)
    fallback = LinearFallbackRanker()

    orderers = {
        "fallback-served": lambda X: served_order(fallback.predict(X), X),
        "seed-cosine-only": lambda X: raw_order(X["seed_i2v_cos"].to_numpy()),
        "retrieval-order": lambda X: np.arange(len(X)),
    }
    try:
        v2 = LightGBMRanker(str(RANKER_PATH))
        orderers = {
            "lightgbm-raw": lambda X: raw_order(v2.predict(X)),
            "lightgbm-served": lambda X: served_order(v2.predict(X), X),
            **orderers,
        }
    except (FileNotFoundError, ValueError) as exc:
        print(f"note: LightGBM ranker unavailable ({exc}) — comparing baselines only\n")

    results = {name: [] for name in orderers}
    for _, group in df.groupby("qid"):
        # Older eval samples may predate newly added features; absent columns
        # take the same defaults serving uses when a subsystem is missing.
        X = group.reindex(columns=FEATURE_NAMES)
        for col, default in MISSING_DEFAULTS.items():
            if X[col].isna().all():
                X[col] = default
        X = X.fillna(0.0)
        y = (group["label"].to_numpy() > 0).astype(float)
        if y.sum() == 0:
            continue
        seed_cos = group["seed_i2v_cos"].to_numpy()
        audio_cos = group["audio_cos_seed"].to_numpy() if "audio_cos_seed" in group.columns else None
        for name, fn in orderers.items():
            results[name].append(ordered_metrics(fn(X), y, seed_cos, audio_cos))

    n_groups = len(next(iter(results.values())))
    if n_groups == 0:
        print("no evaluation groups contain positive labels")
        return 1
    exact = {name: {k: float(np.mean([m[k] for m in ms])) for k in ms[0]}
             for name, ms in results.items()}
    table = pd.DataFrame(exact).T
    print(f"{n_groups} evaluation groups (survivor-conditional; see notebook manifest for true e2e)\n")
    print(table.round(4).to_string())

    if "lightgbm-served" in exact:
        # Gate the SERVED policy on unrounded values against EVERY baseline
        # ordering; *_cos@10 columns are trade-off dials, not quality bars.
        quality = [c for c in table.columns if "_cos@" not in c]
        served = exact["lightgbm-served"]
        failures = [
            f"{c} vs {base}"
            for base in ("seed-cosine-only", "fallback-served")
            if base in exact
            for c in quality
            if served[c] < exact[base][c]
        ]
        print("\ngate:", "PASS — served policy >= all baselines on all ranking metrics"
              if not failures else f"FAIL — served policy loses on: {', '.join(failures)}")
        return 0 if not failures else 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
