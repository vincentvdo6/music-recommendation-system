"""
Offline A/B over the held-out eval sample exported by the training notebook.

Compares, on identical candidate sets and features:
  - the shipped LightGBM ranker, raw AND through the exact served policy
    (shared ScoringPolicy: standardization + dials + eligible-first floor)
  - the LinearFallbackRanker through the same served policy
  - seed-cosine-only ordering (equivalent to the old deployed 1-feature model)
  - raw retrieval order

Caveats vs production: Spotify-availability losses are not simulated here.
When the eval sample carries ``artist`` (v13+), the exact one-per-artist
serving constraint is applied before every cutoff.

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


def serving_pool_size(limit: int) -> int:
    """Mirror MusicService's bounded deep-overfetch contract."""
    return min(200, max(50, int(limit) * 10))


def artist_dedup_order(order: np.ndarray, artists=None, pool_limit=None) -> np.ndarray:
    """Apply the production one-per-known-artist constraint to a candidate pool."""
    if pool_limit is not None:
        order = order[:pool_limit]
    if artists is None:
        return order
    artists = np.asarray(artists, dtype=object)
    kept = []
    seen = set()
    for idx in order:
        artist = artists[idx]
        # Missing metadata should not collapse unrelated tracks together.
        if artist is None or (isinstance(artist, float) and np.isnan(artist)) or not str(artist).strip():
            kept.append(idx)
            continue
        key = str(artist).strip().casefold()
        if key in seen:
            continue
        seen.add(key)
        kept.append(idx)
    return np.asarray(kept, dtype=np.int64)


def ordered_metrics(order: np.ndarray, labels: np.ndarray, seed_cos: np.ndarray,
                    audio_cos: np.ndarray = None, artists=None, ks=(10, 50)) -> dict:
    """Metrics for an explicit candidate ORDER (so floor/selection effects count)."""
    out = {}
    served = {}
    for k in ks:
        top = artist_dedup_order(order, artists, serving_pool_size(k))[:k]
        served[k] = top
        topk = labels[top]
        out[f"recall@{k}"] = float(topk.sum() / labels.sum())
        dcg = float((topk / np.log2(np.arange(2, len(topk) + 2))).sum())
        ideal = np.sort(labels)[::-1][:k]
        idcg = float((ideal / np.log2(np.arange(2, len(ideal) + 2))).sum())
        out[f"ndcg@{k}"] = dcg / idcg if idcg > 0 else 0.0
    top10 = served[10]
    rel10 = labels[top10]
    out["hit@1"] = float(rel10[0] > 0) if len(rel10) else 0.0
    pos = np.nonzero(rel10 > 0)[0]
    out["mrr"] = float(1.0 / (pos[0] + 1)) if len(pos) else 0.0
    # Conditional vibe diagnostics (not quality bars, never coverage-scaled).
    out["seed_cos@10"] = float(seed_cos[top10].mean()) if len(top10) else np.nan
    if audio_cos is not None:
        out["audio_cos@10"] = float(audio_cos[top10].mean()) if len(top10) else np.nan
    return out


def raw_order(scores: np.ndarray) -> np.ndarray:
    return np.argsort(-np.asarray(scores, dtype=np.float64))


def served_order(model_scores, X: pd.DataFrame, policy=POLICY) -> np.ndarray:
    """Exactly what engine.recommend ships (full-length order for metrics)."""
    seed_cos = effective_seed_cos(X)
    blended = policy.blend(np.asarray(model_scores, dtype=np.float64),
                           seed_cos, X["log_pop"].to_numpy(), has_seed=True)
    return policy.select(blended, seed_cos, limit=len(X), has_seed=True)


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
        model = LightGBMRanker(str(RANKER_PATH))

        orderers = {
            "lightgbm-raw": lambda X: raw_order(model.predict(X)),
            "lightgbm-served": lambda X: served_order(model.predict(X), X),
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
        artists = group["artist"].to_numpy() if "artist" in group.columns else None
        for name, fn in orderers.items():
            results[name].append(ordered_metrics(fn(X), y, seed_cos, audio_cos, artists))

    n_groups = len(next(iter(results.values())))
    if n_groups == 0:
        print("no evaluation groups contain positive labels")
        return 1
    exact = {name: {k: float(np.mean([m[k] for m in ms])) for k in ms[0]}
             for name, ms in results.items()}
    table = pd.DataFrame(exact).T
    print(f"{n_groups} evaluation groups (survivor-conditional; see notebook manifest for true e2e)\n")
    print(table.round(4).to_string())

    conditional_gate_ok = True
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
        conditional_gate_ok = not failures
        print("\nconditional baseline gate:",
              "PASS — served policy >= simple baselines on all ranking metrics"
              if conditional_gate_ok else f"FAIL — served policy loses on: {', '.join(failures)}")

    # The artifact manifest owns the actual release decision: a paired,
    # end-to-end comparison with the previous production system.  Do not let a
    # survivor-conditional PASS above mask a failed production gate.
    metrics_path = ROOT / "models" / "metrics.json"
    release_gate_ok = True
    if metrics_path.exists():
        try:
            import json

            manifest = json.loads(metrics_path.read_text())
            gate = manifest.get("decision_gate") or {}
            if gate.get("pass") is False:
                release_gate_ok = False
                print(
                    "release gate: FAIL — paired production comparison "
                    f"delta={gate.get('ndcg_diff')} CI={gate.get('ndcg_ci')}"
                )
            elif gate.get("pass") is True:
                print("release gate: PASS — paired production comparison")
            else:
                print("release gate: UNKNOWN — artifact has no completed paired decision")
                release_gate_ok = False
        except (OSError, ValueError, TypeError) as exc:
            print(f"release gate: UNKNOWN — could not read {metrics_path} ({exc})")
            release_gate_ok = False

    if not release_gate_ok:
        return 3
    return 0 if conditional_gate_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
