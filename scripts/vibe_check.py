"""
Vibe check: how does the engine see a seed song and a set of target songs?

Given a seed and the songs you WISH it recommended, reports for each target:
whether the model knows it, its co-listen cosine to the seed, where it sits in
the seed's retrieval neighborhood, mood distance, and popularity — then shows
what the engine actually returns for that seed. Pinpoints whether a miss is a
retrieval problem, a ranking problem, or a vocabulary problem.

Usage:
    python scripts/vibe_check.py "Seed Song - Artist" "Target 1 - Artist" ["Target 2 - Artist" ...]

Requires the dev server running on :8000 (for Spotify search) and local model
artifacts.
"""

import sys
from pathlib import Path

import httpx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from services.recommendation.factory import get_engine  # noqa: E402

API = "http://127.0.0.1:8000"


def search(query: str) -> dict:
    resp = httpx.get(f"{API}/api/v1/search", params={"q": query, "limit": 1}, timeout=30)
    resp.raise_for_status()
    results = resp.json()["results"]
    if not results:
        raise SystemExit(f"no Spotify result for {query!r}")
    return results[0]["track"]


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 1

    engine = get_engine()
    if engine is None:
        raise SystemExit("engine failed to load — check model artifacts")
    wv = engine.embeddings.item2vec

    seed = search(sys.argv[1])
    targets = [search(q) for q in sys.argv[2:]]

    seed_id, proxied = engine._resolve_seed(seed["id"], seed["artist"])
    print(f"\nSEED: {seed['name']} — {seed['artist']}  (id {seed['id']})")
    if seed_id is None:
        print("  !! seed not in model and no artist proxy — engine is blind here")
        return 2
    print(f"  model seed: {seed_id}" + ("  (same-artist proxy)" if proxied else "  (direct)"))

    seed_vec = wv.vector(seed_id)
    seed_unit = seed_vec / np.linalg.norm(seed_vec)
    neighbors = engine.embeddings.get_neighbors(track_id=seed_id, k=1000)
    neighbor_rank = {tid: i for i, tid in enumerate(neighbors)}

    seed_mood = engine.mood.predict(seed_vec.reshape(1, -1))[0] if engine.mood else None

    print(f"\n{'TARGET':44s} {'in-model':>8s} {'seed-cos':>8s} {'ANN-rank':>8s} {'mood-d':>6s} {'pop':>4s}")
    for t in targets:
        label = f"{t['name'][:28]} — {t['artist'][:12]}"
        if not wv.has_vector(t["id"]):
            # maybe reachable via its artist's other tracks
            proxy = None
            if engine.track_meta:
                for cand in engine.track_meta.tracks_by_artist(t["artist"], limit=20):
                    if wv.has_vector(cand):
                        proxy = cand
                        break
            note = f"artist known via {proxy}" if proxy else "artist unknown too"
            print(f"{label:44s} {'NO':>8s}   -> unreachable by co-occurrence ({note})")
            continue
        vec = wv.vector(t["id"])
        cos = float(vec / np.linalg.norm(vec) @ seed_unit)
        rank = neighbor_rank.get(t["id"])
        rank_s = str(rank) if rank is not None else ">1000"
        mood_d = "-"
        if seed_mood is not None and engine.mood:
            tm = engine.mood.predict(vec.reshape(1, -1))[0]
            mood_d = f"{np.abs(tm - seed_mood).mean():.3f}"
        print(f"{label:44s} {'yes':>8s} {cos:8.3f} {rank_s:>8s} {mood_d:>6s} {t['popularity']:>4d}")

    print("\nWHAT THE ENGINE RETURNS (seed-only mode):")
    resp = httpx.post(f"{API}/api/v1/playlist/recommendations",
                      json={"tracks": [], "seed": f"spotify:track:{seed['id']}", "limit": 10},
                      timeout=120)
    resp.raise_for_status()
    for i, rec in enumerate(resp.json()["recommendations"], 1):
        why = " · ".join(rec["explanation"]["top_factors"][:2])
        print(f"  {i:2d}. {rec['name'][:34]:34s} — {rec['artist'][:18]:18s} "
              f"sim={rec['similarity_score']:.3f} pop={rec['popularity']:>3d}  [{why}]")

    print("""
READING THE TABLE
  in-model=NO      -> co-occurrence can't reach it (post-2017 or too obscure);
                      only the audio pipeline or newer training data fixes this
  ANN-rank >1000   -> ranking never sees it: a RETRIEVAL gap
  ANN-rank <1000,
  not in top 10    -> retrieved but out-ranked: a RANKING/weights gap
  mood-d           -> mean |mood diff| (valence/energy/acoustic/dance), lower = closer vibe""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
