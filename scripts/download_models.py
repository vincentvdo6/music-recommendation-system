"""
Download model artifacts from the GitHub release into models/.

The repo ships no model weights; this fetches them (~1.4 GB total, the ANN
index dominating) so the app can serve real recommendations.

Usage: python scripts/download_models.py [--tag v1.0.0]
"""

import argparse
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = "vincentvdo6/music-recommendation-system"

ARTIFACTS = {
    "item2vec.wordvectors": "models/item2vec",
    "item2vec.wordvectors.vectors.npy": "models/item2vec",
    "metadata.json": "models/item2vec",
    "annoy.index": "models/item2vec/ann_index",
    "id_map.json": "models/item2vec/ann_index",
    "mood_predictor.pkl": "models/mood_predictor",
    "lightgbm_ranker_v2.txt": "models/ranker",
    "ncf_item_v2.pt": "models/ncf",
    "track_meta.parquet": "models/meta",
    "audio_emb.parquet": "models/audio",
    "policy.json": "models",
    "metrics.json": "models",
    "feature_names.json": "models",
}


def download(url: str, dest: Path) -> None:
    def report(blocks, block_size, total):
        done = blocks * block_size
        if total > 0:
            sys.stdout.write(f"\r  {dest.name}: {done / 1e6:,.0f} / {total / 1e6:,.0f} MB")
            sys.stdout.flush()

    partial = dest.with_name(dest.name + ".part")
    try:
        urllib.request.urlretrieve(url, partial, reporthook=report)
        partial.replace(dest)
        print()
    finally:
        partial.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="latest")
    args = parser.parse_args()

    base = (
        f"https://github.com/{REPO}/releases/latest/download"
        if args.tag == "latest"
        else f"https://github.com/{REPO}/releases/download/{args.tag}"
    )

    for name, rel_dir in ARTIFACTS.items():
        dest_dir = ROOT / rel_dir
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / name
        if dest.exists():
            print(f"  {name}: already present, skipping")
            continue
        try:
            download(f"{base}/{name}", dest)
        except Exception as exc:
            print(f"  {name}: FAILED ({exc}) — the app degrades gracefully without it")

    print("\ndone — start the server with: python run_local.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
