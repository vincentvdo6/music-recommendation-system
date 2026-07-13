"""
Merge embed-fleet outputs into the extension-catalog artifacts.

The catalog-expansion embedding queue is worked by several concurrent Kaggle
kernels (a sequential head session plus sharded workers). Each produces
ext_ckpt_*.npz (1280-d fp16 embeddings) and ext_misses*.parquet. This merges
them, projects through the FROZEN PCA (the same 256-d space as
models/audio/audio_emb.parquet), and rebuilds:

    extension_audio_emb.parquet   (track_id, embedding)  -> models/extension/
    extension_tracks.parquet      (catalog metadata)     -> models/extension/

plus a dataset directory ready for `kaggle datasets version` so the next
round of kernels resumes from the merged state.

Usage:
    python scripts/merge_extension.py OUT_DIR INPUT_DIR [INPUT_DIR ...]

One of the input dirs must contain embed_queue.parquet and pca_audio.npz.
Pass --install to also copy the two parquets into models/extension/.
"""

import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def find_one(dirs, name):
    for d in dirs:
        hit = d / name
        if hit.exists():
            return hit
    raise SystemExit(f"{name} not found in any input dir")


def main() -> int:
    args = [a for a in sys.argv[1:] if a != "--install"]
    install = "--install" in sys.argv
    if len(args) < 2:
        print(__doc__)
        return 1
    out_dir = Path(args[0])
    in_dirs = [Path(a) for a in args[1:]]
    out_dir.mkdir(parents=True, exist_ok=True)

    ids_seen, id_chunks, emb_chunks = set(), [], []
    n_ckpts = 0
    for src_i, d in enumerate(in_dirs):
        for f in sorted(d.glob("ext_ckpt_*.npz")):
            z = np.load(f, allow_pickle=True)
            fresh = np.array([tid not in ids_seen for tid in z["ids"]])
            ids_seen.update(z["ids"].tolist())
            if fresh.any():
                id_chunks.append(z["ids"][fresh])
                emb_chunks.append(z["embs"][fresh])
            # unique name, still matching the kernels' ext_ckpt_*.npz glob
            shutil.copy(f, out_dir / f"ext_ckpt_m{src_i}_{f.stem.split('ext_ckpt_')[1]}.npz")
            n_ckpts += 1
    all_ids = np.concatenate(id_chunks) if id_chunks else np.array([], dtype=object)
    all_embs = (np.concatenate([e.astype(np.float32) for e in emb_chunks])
                if emb_chunks else np.zeros((0, 1280), np.float32))
    print(f"{n_ckpts} checkpoints from {len(in_dirs)} dirs -> {len(all_ids):,} unique embeddings")

    misses = set()
    for d in in_dirs:
        for f in d.glob("ext_misses*.parquet"):
            misses |= set(pd.read_parquet(f)["id"])
    pd.DataFrame({"id": sorted(misses)}).to_parquet(out_dir / "ext_misses.parquet", index=False)

    queue = pd.read_parquet(find_one(in_dirs, "embed_queue.parquet"))
    pca = np.load(find_one(in_dirs, "pca_audio.npz"))
    reduced = ((all_embs - pca["mean"]) @ pca["components"].T).astype(np.float16)
    pd.DataFrame({"track_id": all_ids, "embedding": [r.tobytes() for r in reduced]}
                 ).to_parquet(out_dir / "extension_audio_emb.parquet", index=False)

    got = set(all_ids.tolist())
    ext = queue[queue["kind"] != "backfill"].copy()
    ext["embedded"] = ext["id"].isin(got)
    ext["duration_ms"] = ext["duration_s"] * 1000
    ext[["id", "deezer_id", "title", "artist_deezer", "artist_mpd", "duration_ms",
         "pos", "stage", "kind", "matched_track_id", "embedded"]].to_parquet(
        out_dir / "extension_tracks.parquet", index=False)
    shutil.copy(find_one(in_dirs, "embed_queue.parquet"), out_dir / "embed_queue.parquet")

    print(f"embedded by kind:\n{queue[queue['id'].isin(got)].groupby('kind').size().to_string()}")
    print(f"permanent misses: {len(misses):,} | queue remaining: "
          f"{len(queue) - len(got & set(queue['id'])) - len(misses):,}")

    if install:
        dest = ROOT / "models" / "extension"
        dest.mkdir(parents=True, exist_ok=True)
        for name in ("extension_audio_emb.parquet", "extension_tracks.parquet"):
            shutil.copy(out_dir / name, dest / name)
            print(f"  installed {name} -> {dest / name}")

    print(f"\nnext round: kaggle datasets version -p {out_dir} -m 'round merge'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
