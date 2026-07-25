"""
Install training artifacts from the Kaggle notebook into models/.

Usage: python scripts/install_artifacts.py path/to/artifacts.zip

Artifacts whose paired production-baseline decision gate did not pass are
refused before any serving file is written.
"""

import json
import shutil
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
    "extension_tracks.parquet": ROOT / "models" / "extension",     # catalog expansion
    "extension_audio_emb.parquet": ROOT / "models" / "extension",  # catalog expansion
}


def install_member(zf: zipfile.ZipFile, name: str, destination: Path) -> None:
    """Stream one member to an atomic per-file replacement."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".installing")
    try:
        with zf.open(name) as source, partial.open("wb") as target:
            shutil.copyfileobj(source, target, length=1 << 20)
        partial.replace(destination)
    finally:
        partial.unlink(missing_ok=True)


def main() -> int:
    args = sys.argv[1:]
    if len(args) != 1:
        print(__doc__)
        return 1

    zip_path = Path(args[0])
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

        # Validate the release decision before writing a single file.  The
        # survivor-conditional local evaluator can beat simple baselines while
        # a candidate still loses to the exact production system end-to-end.
        metrics = json.loads(zf.read("metrics.json"))
        gate = metrics.get("decision_gate") or {}
        if gate.get("pass") is not True:
            diff = gate.get("ndcg_diff")
            ci = gate.get("ndcg_ci")
            print("refusing artifact: paired production decision gate did not pass")
            if diff is not None:
                print(f"  ndcg@10 delta: {diff:+.6f}")
            if ci is not None:
                print(f"  95% CI: {ci}")
            return 2

        for name, dest_dir in DESTINATIONS.items():
            install_member(zf, name, dest_dir / name)
            print(f"  {name} -> {dest_dir / name}")

        for name, dest_dir in OPTIONAL_DESTINATIONS.items():
            if name in names:
                install_member(zf, name, dest_dir / name)
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
