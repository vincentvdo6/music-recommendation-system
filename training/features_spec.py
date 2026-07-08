"""
Ranking feature contract (v2).

FEATURE_NAMES is the single source of truth for the ranker feature vector.
The Kaggle training notebook mirrors this list (training/features_spec.py);
the shipped LightGBM model's feature names are asserted against it at load
time, so serving and training cannot silently drift apart.

Every feature here is computable at serving time from:
  - item2vec embeddings (always present for retrieved candidates)
  - ANN retrieval ranks
  - the item-NCF model (optional)
  - the MPD-derived track metadata table (optional)
  - the embedding-based mood predictor (optional)

No deprecated Spotify audio-features API anywhere. When an optional
subsystem is missing, its features collapse to a constant for the whole
candidate set (0 for scores/indicators, 0.5 for mood diffs), which leaves
within-query ranking unaffected by that subsystem.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

FEATURE_NAMES: List[str] = [
    "seed_i2v_cos",         # cosine(candidate, seed-or-proxy vector)
    "playlist_i2v_cos",     # cosine(candidate, playlist mean vector)
    "playlist_i2v_max",     # max cosine over sampled playlist vectors
    "i2v_seed_rr",          # 1/(rank+1) in the seed ANN neighbor list
    "i2v_playlist_rr",      # 1/(rank+1) in the playlist-mean ANN neighbor list
    "ncf_score",            # item-NCF fold-in score (sigmoid)
    "has_ncf",              # 1 if candidate scored by NCF with >=3 context tracks
    "log_pop",              # log1p(MPD playlist count) / log1p(max count)
    "same_artist_as_seed",  # binary, real artist names from track_meta
    "artist_in_playlist",   # binary
    "artist_playlist_share",# fraction of playlist tracks by candidate's artist
    "valence_diff",         # |mood(candidate) - target mood| per dimension
    "energy_diff",
    "acousticness_diff",
    "danceability_diff",
    "mood_sim",             # 1 - mean of the four mood diffs
    "duration_diff",        # |candidate - playlist mean| in minutes, capped at 5
]

# Canonical order of mood-predictor output dimensions.
MOOD_DIMS: List[str] = ["valence", "energy", "acousticness", "danceability"]


def _unit_rows(matrix: np.ndarray) -> np.ndarray:
    """Row-normalize, leaving zero rows as zeros."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def _unit(vec: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if vec is None:
        return None
    norm = np.linalg.norm(vec)
    if norm == 0:
        return None
    return vec / norm


@dataclass
class RankingContext:
    """Query-level context shared by all candidates of one request."""

    seed_vec: Optional[np.ndarray] = None           # (d,) raw seed/proxy vector
    playlist_mean_vec: Optional[np.ndarray] = None  # (d,)
    playlist_vecs: Optional[np.ndarray] = None      # (m, d) sample, m <= 100
    seed_rank: Dict[str, int] = field(default_factory=dict)      # id -> 0-based ANN rank
    playlist_rank: Dict[str, int] = field(default_factory=dict)  # id -> 0-based ANN rank
    seed_artist: str = ""                            # normalized lowercase, "" if unknown
    playlist_artist_share: Dict[str, float] = field(default_factory=dict)
    target_mood: Optional[np.ndarray] = None         # (4,) in MOOD_DIMS order
    playlist_mean_duration_ms: float = 0.0


def build_matrix(
    ids: List[str],
    vectors: np.ndarray,
    artists: List[str],
    log_pop: np.ndarray,
    durations_ms: np.ndarray,
    moods: Optional[np.ndarray],
    ncf_scores: Optional[np.ndarray],
    ncf_mask: Optional[np.ndarray],
    ctx: RankingContext,
) -> pd.DataFrame:
    """
    Build the (n, len(FEATURE_NAMES)) feature matrix for one query.

    Args:
        ids: candidate track ids, length n
        vectors: (n, d) item2vec vectors, zero rows where missing
        artists: normalized lowercase artist per candidate, "" if unknown
        log_pop: (n,) popularity prior already normalized to [0, 1]
        durations_ms: (n,) track durations, 0 if unknown
        moods: (n, 4) mood-predictor outputs in MOOD_DIMS order, or None
        ncf_scores: (n,) item-NCF scores, or None if NCF unavailable
        ncf_mask: (n,) 1.0 where the candidate was actually scored, or None
        ctx: query-level context

    Returns:
        DataFrame with columns exactly FEATURE_NAMES, in order.
    """
    n = len(ids)
    unit_vecs = _unit_rows(np.asarray(vectors, dtype=np.float32))
    out: Dict[str, np.ndarray] = {}

    seed_unit = _unit(ctx.seed_vec)
    out["seed_i2v_cos"] = unit_vecs @ seed_unit if seed_unit is not None else np.zeros(n, dtype=np.float32)

    mean_unit = _unit(ctx.playlist_mean_vec)
    out["playlist_i2v_cos"] = unit_vecs @ mean_unit if mean_unit is not None else np.zeros(n, dtype=np.float32)

    if ctx.playlist_vecs is not None and len(ctx.playlist_vecs):
        playlist_units = _unit_rows(np.asarray(ctx.playlist_vecs, dtype=np.float32))
        out["playlist_i2v_max"] = (unit_vecs @ playlist_units.T).max(axis=1)
    else:
        out["playlist_i2v_max"] = np.zeros(n, dtype=np.float32)

    out["i2v_seed_rr"] = np.array(
        [1.0 / (ctx.seed_rank[i] + 1) if i in ctx.seed_rank else 0.0 for i in ids], dtype=np.float32
    )
    out["i2v_playlist_rr"] = np.array(
        [1.0 / (ctx.playlist_rank[i] + 1) if i in ctx.playlist_rank else 0.0 for i in ids], dtype=np.float32
    )

    if ncf_scores is not None:
        out["ncf_score"] = np.asarray(ncf_scores, dtype=np.float32)
        out["has_ncf"] = (
            np.asarray(ncf_mask, dtype=np.float32) if ncf_mask is not None else np.ones(n, dtype=np.float32)
        )
    else:
        out["ncf_score"] = np.zeros(n, dtype=np.float32)
        out["has_ncf"] = np.zeros(n, dtype=np.float32)

    out["log_pop"] = np.asarray(log_pop, dtype=np.float32)

    known = np.array([bool(a) for a in artists])
    out["same_artist_as_seed"] = np.where(
        known & (np.array(artists, dtype=object) == ctx.seed_artist) & bool(ctx.seed_artist), 1.0, 0.0
    ).astype(np.float32)
    out["artist_in_playlist"] = np.array(
        [1.0 if a and a in ctx.playlist_artist_share else 0.0 for a in artists], dtype=np.float32
    )
    out["artist_playlist_share"] = np.array(
        [ctx.playlist_artist_share.get(a, 0.0) if a else 0.0 for a in artists], dtype=np.float32
    )

    if moods is not None and ctx.target_mood is not None:
        diffs = np.abs(np.asarray(moods, dtype=np.float32) - np.asarray(ctx.target_mood, dtype=np.float32))
        for dim_idx, dim in enumerate(MOOD_DIMS):
            out[f"{dim}_diff"] = diffs[:, dim_idx]
        out["mood_sim"] = 1.0 - diffs.mean(axis=1)
    else:
        # Neutral constants: a missing mood subsystem must not reorder candidates.
        for dim in MOOD_DIMS:
            out[f"{dim}_diff"] = np.full(n, 0.5, dtype=np.float32)
        out["mood_sim"] = np.full(n, 0.5, dtype=np.float32)

    durations = np.asarray(durations_ms, dtype=np.float32)
    if ctx.playlist_mean_duration_ms > 0:
        diff_min = np.abs(durations - ctx.playlist_mean_duration_ms) / 60000.0
        out["duration_diff"] = np.where(durations > 0, np.minimum(diff_min, 5.0), 1.0).astype(np.float32)
    else:
        out["duration_diff"] = np.full(n, 1.0, dtype=np.float32)

    return pd.DataFrame(out, columns=FEATURE_NAMES)
