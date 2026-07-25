"""Compare request-level recommendation profiles on the installed eval sample.

This measures the profile's scoring-policy trade-offs on identical candidates.
Session affinity requires live request context and is reported by serving
telemetry rather than approximated here.

Usage: python scripts/evaluate_profiles.py [evaluation/eval_sample.parquet]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.evaluate import (  # noqa: E402
    MISSING_DEFAULTS,
    POLICY,
    RANKER_PATH,
    ordered_metrics,
    served_order,
)
from services.recommendation.features import FEATURE_NAMES  # noqa: E402
from services.recommendation.profiles import PROFILES  # noqa: E402
from services.recommendation.ranker import LightGBMRanker  # noqa: E402


def main() -> int:
    sample = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "evaluation" / "eval_sample.parquet"
    if not sample.exists():
        print(f"not found: {sample}")
        return 1

    model = LightGBMRanker(str(RANKER_PATH))
    results = {name: [] for name in PROFILES}
    frame = pd.read_parquet(sample)
    for _, group in frame.groupby("qid"):
        X = group.reindex(columns=FEATURE_NAMES)
        for column, default in MISSING_DEFAULTS.items():
            if X[column].isna().all():
                X[column] = default
        X = X.fillna(0.0)
        labels = (group["label"].to_numpy() > 0).astype(float)
        if labels.sum() == 0:
            continue
        scores = model.predict(X)
        seed_cos = group["seed_i2v_cos"].to_numpy()
        audio_cos = group["audio_cos_seed"].to_numpy() if "audio_cos_seed" in group else None
        artists = group["artist"].to_numpy() if "artist" in group else None
        for name, profile in PROFILES.items():
            order = served_order(scores, X, profile.scoring_policy(POLICY))
            results[name].append(ordered_metrics(order, labels, seed_cos, audio_cos, artists))

    if not any(results.values()):
        print("no evaluation groups contain positives")
        return 1
    summary = {
        name: {metric: float(np.mean([row[metric] for row in rows])) for metric in rows[0]}
        for name, rows in results.items()
    }
    table = pd.DataFrame(summary).T
    print(f"{len(next(iter(results.values())))} groups; policy-only profile comparison\n")
    print(table.round(4).to_string())
    print("\nNote: session affinity is evaluated from live telemetry.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
