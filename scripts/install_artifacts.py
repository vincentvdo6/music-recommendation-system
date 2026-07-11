"""
Install training artifacts from the Kaggle notebook into models/.

Usage: python scripts/install_artifacts.py path/to/artifacts.zip
"""

import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DESTINATIONS = {
    "lightgbm_ranker_v2.txt": ROOT / "models" / "ranker",
    "ncf_item_v2.pt": ROOT / "models" / "ncf",
    "track_meta.parquet": ROOT / "models" / "meta",
    "metrics.json": ROOT / "models",
    "feature_names.json": ROOT / "models",
    "eval_sample.parquet": ROOT / "evaluation",
}

OPTIONAL_DESTINATIONS = {
    "audio_emb.parquet": ROOT / "models" / "audio",   # present from contract v3 onward
    "group_manifest.parquet": ROOT / "evaluation",    # v11+: dropped-group identities
    "query_manifest.parquet": ROOT / "evaluation",    # v11+: exact seeds/contexts/positives
    "policy.json": ROOT / "models",                   # v11+: validation-frozen serving policy
}


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 1

    zip_path = Path(sys.argv[1])
    if not zip_path.exists():
        print(f"not found: {zip_path}")
        return 1

    with zipfile.ZipFile(zip_path) as zf:
        names = set(zf.namelist())
        missing = set(DESTINATIONS) - names
        if missing:
            print(f"artifacts.zip is missing: {sorted(missing)}")
            return 1

        # Refuse artifacts trained against a different feature contract.
        sys.path.insert(0, str(ROOT))
        from services.recommendation.features import FEATURE_NAMES

        trained = json.loads(zf.read("feature_names.json"))
        if trained != FEATURE_NAMES:
            print("feature contract mismatch — retrain with the current features_spec.py:")
            print(f"  trained:  {trained}")
            print(f"  serving:  {FEATURE_NAMES}")
            return 1

        for name, dest_dir in DESTINATIONS.items():
            dest_dir.mkdir(parents=True, exist_ok=True)
            (dest_dir / name).write_bytes(zf.read(name))
            print(f"  {name} -> {dest_dir / name}")

        for name, dest_dir in OPTIONAL_DESTINATIONS.items():
            if name in names:
                dest_dir.mkdir(parents=True, exist_ok=True)
                (dest_dir / name).write_bytes(zf.read(name))
                print(f"  {name} -> {dest_dir / name}")

    metrics = json.loads((ROOT / "models" / "metrics.json").read_text())
    print("\noffline metrics:")
    for key, values in metrics.get("metrics", {}).items():
        if isinstance(values, dict):
            flat = "  ".join(f"{k}={v:.4f}" for k, v in values.items() if isinstance(v, (int, float)))
            print(f"  {key:>18}: {flat}")
        elif isinstance(values, (int, float)):
            print(f"  {key:>18}: {values:.4f}")
    if metrics.get("frozen_policy"):
        print("  frozen policy:", metrics["frozen_policy"])
    print("\nnext: python scripts/evaluate.py && python -m pytest -m slow")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
