"""
Recommendation engine: the one and only serving pipeline.

    ANN retrieval (70% seed / 30% playlist, or the whole pool from the
      seed when there is no playlist; plus K_AUDIO audio-ANN candidates
      when the seed has an audio vector)
      -> artist pre-dedup (no-op while ARTIST_PRECAP is None)
      -> mood filter (no-op while MOOD_KEEP_PCT is 1.0)
      -> vectorized feature matrix (features.FEATURE_NAMES)
      -> ranker (LightGBM LambdaRank, or linear fallback)
      -> top-N with per-track explanations

Both hard filters stay in the call path but are disabled by default; the
constants below carry the funnel measurements that turned them off.

The searched song ("seed") dominates: 70% of the co-listen pool comes from
its neighbors, its mood carries 80% of the target-mood blend, and
seed_i2v_cos is the primary ranking feature. Display metadata (album art,
preview URLs, live popularity) is attached later by the service layer via
the Spotify metadata API — the engine itself never performs I/O.
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from services.recommendation import features as F
from services.recommendation.audio_store import AudioStore
from services.recommendation.embeddings import EmbeddingService
from services.recommendation.mood import MoodPredictor
from services.recommendation.ncf import ItemNCFScorer
from services.recommendation.track_meta import TrackMetaStore

logger = logging.getLogger(__name__)

RETRIEVAL_POOL = 1000
# Extra audio-ANN candidates when the seed has an audio vector. Sized from the
# calibration set: human-picked scene-swappable tracks sit at audio rank
# ~3.5-4K of 65K (a-cos ~0.42) — a channel that can't reach that depth can't
# surface them. Candidate features are vectorized; 5K rows rank in ~ms.
K_AUDIO = 4000
SEED_SHARE = 0.70         # fraction of the pool retrieved from the seed's neighbors
SEED_MOOD_WEIGHT = 0.8    # seed weight in the target-mood blend
# Hard filters are deliberately soft: mood features earn ~2% of model gain,
# so a mood veto mostly destroyed candidate recall (funnel measurements);
# the ranker's mood_sim feature handles coherence. Artist pre-caps destroyed
# recall the same way (funnel ablation), so the ranker sees everything and
# the service enforces 1-per-artist after enrichment. Mirrored in the
# training notebook — retrain if these change.
MOOD_KEEP_PCT = 1.0       # 1.0 = no hard mood cut
ARTIST_PRECAP: Optional[int] = None  # candidates per artist entering the ranker; None = uncapped
PLAYLIST_VEC_SAMPLE = 100
PLAYLIST_MOOD_SAMPLE = 50

# The serving score policy (dials, standardization, floor) lives in
# policy.ScoringPolicy so evaluation measures exactly what serving ships.
# Constants re-exported for backward compatibility.
from services.recommendation.policy import (  # noqa: E402
    DEFAULT_DISCOVERY,
    DEFAULT_SEED_AFFINITY,
    SEED_COS_FLOOR,
    ScoringPolicy,
)

__all_policy__ = (DEFAULT_DISCOVERY, DEFAULT_SEED_AFFINITY, SEED_COS_FLOOR)  # keep re-exports import-visible

FEATURE_LABELS = {
    "seed_i2v_cos": "often played alongside the seed track",
    "playlist_i2v_cos": "fits the playlist as a whole",
    "playlist_i2v_max": "very close to one of your playlist tracks",
    "i2v_seed_rr": "a top neighbor of the seed track",
    "i2v_playlist_rr": "a top neighbor of your playlist",
    "ncf_score": "strong neural collaborative-filtering match",
    "has_ncf": "scored by the neural model",
    "log_pop": "widely playlisted",
    "same_artist_as_seed": "same artist as the seed",
    "artist_in_playlist": "artist already in your playlist",
    "artist_playlist_share": "artist prominent in your playlist",
    "valence_diff": "similar positivity",
    "energy_diff": "similar energy",
    "acousticness_diff": "similar acousticness",
    "danceability_diff": "similar danceability",
    "mood_sim": "matches the mood",
    "duration_diff": "similar length",
    "audio_cos_seed": "sounds like the seed track",
    "audio_cos_playlist": "sounds like your playlist",
    "has_audio": "audio analyzed",
}


class RecommendationEngine:
    """item2vec retrieval + learned ranking over runtime-real features."""

    def __init__(
        self,
        embeddings: EmbeddingService,
        ranker,
        mood: Optional[MoodPredictor] = None,
        ncf: Optional[ItemNCFScorer] = None,
        track_meta: Optional[TrackMetaStore] = None,
        audio: Optional[AudioStore] = None,
        seed_affinity: float = DEFAULT_SEED_AFFINITY,
        discovery: float = DEFAULT_DISCOVERY,
    ):
        if embeddings is None or embeddings.item2vec.wv is None:
            raise ValueError("RecommendationEngine requires a loaded EmbeddingService")
        self.embeddings = embeddings
        self.ranker = ranker
        self.mood = mood if mood and mood.available else None
        self.ncf = ncf
        self.track_meta = track_meta if track_meta and track_meta.available else None
        self.audio = audio if audio and audio.available else None
        self.seed_affinity = seed_affinity
        self.discovery = discovery
        self.policy = ScoringPolicy(seed_affinity=seed_affinity, discovery=discovery)

    # ------------------------------------------------------------------ API

    def recommend(
        self,
        playlist_tracks: List[str],
        input_track_id: Optional[str] = None,
        input_artist_name: Optional[str] = None,
        input_track_name: Optional[str] = None,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """Recommend tracks driven by the searched song; the playlist (optional) adds flavor."""
        seed_id, seed_proxied = self._resolve_seed(input_track_id, input_artist_name)
        if input_track_id and not seed_id:
            logger.info(
                "Seed %s ('%s' by '%s') not in model and no proxy found — playlist-only retrieval",
                input_track_id, input_track_name, input_artist_name,
            )
        if not playlist_tracks and not seed_id:
            return []

        # The seed's own row in the playlist would leak into playlist-side
        # features ("fits the playlist" partly restating "is the seed");
        # playlist context is everything EXCEPT the seed.
        playlist_tracks = [t for t in playlist_tracks if t not in (input_track_id, seed_id)]

        seed_audio_vec = self._seed_audio(seed_id, input_artist_name, input_track_id)
        ids, ctx = self._retrieve(playlist_tracks, seed_id, input_track_id, seed_audio_vec)
        if not ids:
            logger.warning("No candidates retrieved (playlist outside model vocabulary)")
            return []

        vectors = self._vectors(ids)
        artists_norm = self.track_meta.artists(ids) if self.track_meta else [""] * len(ids)

        ids, vectors, artists_norm = self._artist_dedup(ids, vectors, artists_norm)

        moods = self.mood.predict(vectors) if self.mood else None
        ctx.target_mood = self._target_mood(playlist_tracks, seed_id)
        ids, vectors, artists_norm, moods = self._mood_filter(ids, vectors, artists_norm, moods, ctx.target_mood, limit)

        ctx.seed_artist = (input_artist_name or "").lower().strip()
        ctx.playlist_artist_share = self._playlist_artist_share(playlist_tracks)
        ctx.playlist_mean_duration_ms = self._playlist_mean_duration(playlist_tracks)

        ncf_scores, ncf_mask = self._ncf_scores(playlist_tracks, seed_id, ids)
        log_pop = self.track_meta.log_pop(ids) if self.track_meta else np.zeros(len(ids), dtype=np.float32)
        durations = self.track_meta.durations_ms(ids) if self.track_meta else np.zeros(len(ids), dtype=np.float32)

        audio_vecs = None
        if self.audio:
            audio_vecs, _ = self.audio.matrix_for(ids)
            # ctx.seed_audio_vec was set during retrieval (same vector feeds
            # the audio channel and the audio_cos_seed feature).
            ctx.playlist_audio_mean = self.audio.mean_vector(playlist_tracks[:PLAYLIST_VEC_SAMPLE])

        X = F.build_matrix(ids, vectors, artists_norm, log_pop, durations, moods, ncf_scores, ncf_mask, ctx,
                           audio_vecs=audio_vecs)
        seed_cos = X["seed_i2v_cos"].to_numpy()
        has_seed = ctx.seed_vec is not None

        scores = self.policy.blend(self.ranker.predict(X), seed_cos, X["log_pop"].to_numpy(), has_seed)
        order = self.policy.select(scores, seed_cos, limit, has_seed)
        explanations = self.ranker.explain(X.iloc[order])

        meta = self.track_meta.lookup([ids[i] for i in order]) if self.track_meta else None
        results = []
        for out_pos, i in enumerate(order):
            tid = ids[i]
            row = X.iloc[i]
            name, artist = tid, ""
            duration = int(durations[i])
            if meta is not None:
                meta_row = meta.iloc[out_pos]
                if isinstance(meta_row.get("name"), str):
                    name = meta_row["name"]
                if isinstance(meta_row.get("artist"), str):
                    artist = meta_row["artist"]
            results.append({
                "id": tid,
                "name": name,
                "artist": artist,
                "duration_ms": duration,
                "rank_score": float(scores[i]),
                "similarity_score": float(row["seed_i2v_cos"]),
                "recommendation": {
                    "score": float(scores[i]),
                    "components": {
                        "seed_similarity": float(row["seed_i2v_cos"]),
                        "playlist_similarity": float(row["playlist_i2v_cos"]),
                        "mood_match": float(row["mood_sim"]),
                        "popularity_prior": float(row["log_pop"]),
                    },
                },
                "why": [FEATURE_LABELS.get(feat, feat) for feat, _ in explanations[out_pos]],
            })

        logger.info(
            "Recommended %d/%d candidates (ranker=%s, seed=%s%s)",
            len(results), len(ids), self.ranker.name, seed_id, " via proxy" if seed_proxied else "",
        )
        return results

    def coverage(self, playlist_tracks: List[str], input_track_id: Optional[str] = None) -> Dict[str, Any]:
        """How much of the request the model can actually see."""
        in_model = sum(1 for t in playlist_tracks if self.embeddings.item2vec.has_vector(t))
        return {
            "tracks_in_model": in_model,
            "playlist_size": len(playlist_tracks),
            "seed_in_model": bool(input_track_id and self.embeddings.item2vec.has_vector(input_track_id)),
        }

    def popular_fallback(self, limit: int) -> List[Dict[str, Any]]:
        """Deterministic cold-start: most-playlisted tracks from the MPD table."""
        if not self.track_meta:
            return []
        ids = self.track_meta.top_popular(limit)
        meta = self.track_meta.lookup(ids)
        pop = self.track_meta.log_pop(ids)
        durations = self.track_meta.durations_ms(ids)
        return [
            {
                "id": tid,
                "name": meta.iloc[i]["name"],
                "artist": meta.iloc[i]["artist"],
                "duration_ms": int(durations[i]),
                "rank_score": float(pop[i]),
                "similarity_score": 0.0,
                "recommendation": {"score": float(pop[i]), "components": {"popularity_prior": float(pop[i])}},
                "why": ["widely playlisted"],
            }
            for i, tid in enumerate(ids)
        ]

    # ------------------------------------------------------------ internals

    def _seed_audio(self, seed_id: Optional[str], input_artist_name: Optional[str],
                    input_track_id: Optional[str] = None):
        """Seed's preview embedding. The SEARCHED track's own audio comes first:
        an out-of-vocab seed (post-2017 release, deep cut) may still have a real
        audio vector from the extension harvest — that, not the i2v proxy's
        sound, is what the audio channel should chase. Then the resolved seed,
        then any same-artist track with audio."""
        if not self.audio:
            return None
        for tid in (input_track_id, seed_id):
            if tid is not None:
                vec = self.audio.vector(tid)
                if vec is not None:
                    return vec
        if input_artist_name and self.track_meta:
            for cand in self.track_meta.tracks_by_artist(input_artist_name, limit=20):
                vec = self.audio.vector(cand)
                if vec is not None:
                    return vec
        return None

    def _resolve_seed(
        self, input_track_id: Optional[str], input_artist_name: Optional[str]
    ) -> Tuple[Optional[str], bool]:
        """The seed vector source: the searched track, or a same-artist proxy."""
        if input_track_id and self.embeddings.item2vec.has_vector(input_track_id):
            return input_track_id, False
        if input_artist_name and self.track_meta:
            for candidate in self.track_meta.tracks_by_artist(input_artist_name, limit=20):
                if self.embeddings.item2vec.has_vector(candidate):
                    logger.info("Using proxy seed %s for artist '%s'", candidate, input_artist_name)
                    return candidate, True
        return None, False

    def _retrieve(
        self,
        playlist_tracks: List[str],
        seed_id: Optional[str],
        input_track_id: Optional[str],
        seed_audio_vec: Optional[np.ndarray] = None,
    ) -> Tuple[List[str], F.RankingContext]:
        ctx = F.RankingContext()
        seed_neighbors: List[str] = []
        playlist_neighbors: List[str] = []
        audio_neighbors: List[str] = []
        if seed_id:
            # Seed-only mode gets the entire pool from the seed's neighbors.
            k_seed = RETRIEVAL_POOL if not playlist_tracks else int(RETRIEVAL_POOL * SEED_SHARE)
            seed_neighbors = self.embeddings.get_neighbors(track_id=seed_id, k=k_seed)
            if playlist_tracks:
                playlist_neighbors = self.embeddings.get_playlist_neighbors(
                    playlist_tracks, k=RETRIEVAL_POOL - k_seed
                )
            ctx.seed_vec = self.embeddings.item2vec.vector(seed_id)
        elif playlist_tracks:
            playlist_neighbors = self.embeddings.get_playlist_neighbors(playlist_tracks, k=RETRIEVAL_POOL)

        # Second retrieval channel: what SOUNDS like the seed, age-blind.
        # Reaches tracks with no co-occurrence vector at all (extension catalog).
        if self.audio and seed_audio_vec is not None:
            audio_neighbors = self.audio.nearest(seed_audio_vec, k=K_AUDIO)
        ctx.seed_audio_vec = seed_audio_vec

        ctx.seed_rank = {tid: rank for rank, tid in enumerate(seed_neighbors)}
        ctx.playlist_rank = {tid: rank for rank, tid in enumerate(playlist_neighbors)}
        ctx.playlist_mean_vec = self.embeddings.item2vec.mean_vector(playlist_tracks)

        playlist_vecs = [
            self.embeddings.item2vec.vector(t)
            for t in playlist_tracks[:PLAYLIST_VEC_SAMPLE]
            if self.embeddings.item2vec.has_vector(t)
        ]
        ctx.playlist_vecs = np.stack(playlist_vecs) if playlist_vecs else None

        exclude = set(playlist_tracks) | {seed_id, input_track_id, None}
        seen = set()
        ids = []
        for tid in [*seed_neighbors, *playlist_neighbors, *audio_neighbors]:
            if tid not in exclude and tid not in seen:
                seen.add(tid)
                ids.append(tid)
        return ids, ctx

    def _vectors(self, ids: List[str]) -> np.ndarray:
        wv = self.embeddings.item2vec.wv
        dim = self.embeddings.item2vec.dim
        return np.stack([wv[t] if t in wv else np.zeros(dim, dtype=np.float32) for t in ids])

    @staticmethod
    def _artist_dedup(
        ids: List[str], vectors: np.ndarray, artists_norm: List[str]
    ) -> Tuple[List[str], np.ndarray, List[str]]:
        """Cap best-retrieved tracks per known artist; unknown artists pass through."""
        if ARTIST_PRECAP is None:
            return ids, vectors, artists_norm
        counts: Dict[str, int] = {}
        keep = []
        for i, artist in enumerate(artists_norm):
            if artist:
                if counts.get(artist, 0) >= ARTIST_PRECAP:
                    continue
                counts[artist] = counts.get(artist, 0) + 1
            keep.append(i)
        if len(keep) == len(ids):
            return ids, vectors, artists_norm
        return [ids[i] for i in keep], vectors[keep], [artists_norm[i] for i in keep]

    def _target_mood(self, playlist_tracks: List[str], seed_id: Optional[str]) -> Optional[np.ndarray]:
        if not self.mood:
            return None
        playlist_vecs = [
            self.embeddings.item2vec.vector(t)
            for t in playlist_tracks[:PLAYLIST_MOOD_SAMPLE]
            if self.embeddings.item2vec.has_vector(t)
        ]
        playlist_mood = None
        if playlist_vecs:
            preds = self.mood.predict(np.stack(playlist_vecs))
            if preds is not None:
                playlist_mood = preds.mean(axis=0)

        seed_mood = None
        if seed_id:
            seed_vec = self.embeddings.item2vec.vector(seed_id)
            if seed_vec is not None:
                preds = self.mood.predict(seed_vec.reshape(1, -1))
                if preds is not None:
                    seed_mood = preds[0]

        if seed_mood is not None and playlist_mood is not None:
            return SEED_MOOD_WEIGHT * seed_mood + (1 - SEED_MOOD_WEIGHT) * playlist_mood
        return seed_mood if seed_mood is not None else playlist_mood

    @staticmethod
    def _mood_filter(
        ids: List[str],
        vectors: np.ndarray,
        artists_norm: List[str],
        moods: Optional[np.ndarray],
        target_mood: Optional[np.ndarray],
        limit: int,
    ):
        if moods is None or target_mood is None or len(ids) <= limit:
            return ids, vectors, artists_norm, moods
        mood_sim = 1.0 - np.abs(moods - target_mood).mean(axis=1)
        keep_n = max(min(len(ids), limit * 3), int(len(ids) * MOOD_KEEP_PCT))
        keep = np.sort(np.argsort(-mood_sim)[:keep_n])  # preserve retrieval order
        return [ids[i] for i in keep], vectors[keep], [artists_norm[i] for i in keep], moods[keep]

    def _playlist_artist_share(self, playlist_tracks: List[str]) -> Dict[str, float]:
        if not self.track_meta or not playlist_tracks:
            return {}
        counts = Counter(a for a in self.track_meta.artists(playlist_tracks) if a)
        total = len(playlist_tracks)
        return {artist: count / total for artist, count in counts.items()}

    def _playlist_mean_duration(self, playlist_tracks: List[str]) -> float:
        if not self.track_meta:
            return 0.0
        durations = self.track_meta.durations_ms(playlist_tracks)
        known = durations[durations > 0]
        return float(known.mean()) if len(known) else 0.0

    def _ncf_scores(
        self, playlist_tracks: List[str], seed_id: Optional[str], candidate_ids: List[str]
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if not self.ncf:
            return None, None
        context = list(playlist_tracks)
        if seed_id and seed_id not in context:
            context.append(seed_id)
        result = self.ncf.score(context, candidate_ids)
        if result is None:
            return None, None
        return result
