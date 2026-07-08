"""Shared fixtures: fake embedding/metadata/mood services and a fake Spotify client."""

import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from services.recommendation.track_meta import TrackMetaStore  # noqa: E402

DIM = 8


class FakeItem2Vec:
    def __init__(self, vectors: Dict[str, np.ndarray]):
        self.wv = vectors
        self.dim = DIM

    def has_vector(self, track_id: str) -> bool:
        return track_id in self.wv

    def vector(self, track_id: str) -> Optional[np.ndarray]:
        return self.wv.get(track_id)

    def mean_vector(self, track_ids: List[str]) -> Optional[np.ndarray]:
        vecs = [self.wv[t] for t in track_ids if t in self.wv]
        return np.mean(vecs, axis=0) if vecs else None


class FakeEmbeddingService:
    """Brute-force cosine neighbors over a small in-memory vocabulary."""

    def __init__(self, vectors: Dict[str, np.ndarray]):
        self.item2vec = FakeItem2Vec(vectors)
        self.ann_index = None

    def _neighbors_for_vec(self, query: np.ndarray, k: int, exclude: Optional[str] = None) -> List[str]:
        scores = []
        qn = query / (np.linalg.norm(query) or 1.0)
        for tid, vec in self.item2vec.wv.items():
            if tid == exclude:
                continue
            vn = vec / (np.linalg.norm(vec) or 1.0)
            scores.append((tid, float(qn @ vn)))
        scores.sort(key=lambda x: -x[1])
        return [tid for tid, _ in scores[:k]]

    def get_neighbors(self, track_id=None, embedding=None, k=200):
        if track_id is not None:
            vec = self.item2vec.vector(track_id)
            if vec is None:
                return []
            return self._neighbors_for_vec(vec, k, exclude=track_id)
        if embedding is not None:
            return self._neighbors_for_vec(embedding, k)
        return []

    def get_playlist_neighbors(self, track_ids, k=500):
        mean = self.item2vec.mean_vector(track_ids)
        if mean is None:
            return []
        return self._neighbors_for_vec(mean, k)


class DummyMoodModel:
    """Deterministic embedding -> mood mapping (picklable)."""

    def predict(self, X):
        X = np.asarray(X)
        return np.abs(X[:, :4]) % 1.0


class FakeSpotify:
    demo_mode = False

    def __init__(self, tracks: Optional[Dict[str, dict]] = None):
        self.tracks = tracks or {}
        self.bulk_calls: List[List[str]] = []

    async def start(self):
        pass

    async def close(self):
        pass

    async def search_tracks(self, query: str, limit: int = 10):
        return []

    async def get_tracks_bulk(self, track_ids):
        self.bulk_calls.append(list(track_ids))
        return {tid: dict(self.tracks[tid]) for tid in track_ids if tid in self.tracks}


class FakeApple:
    async def start(self):
        pass

    async def close(self):
        pass

    async def search_tracks(self, query: str, limit: int = 10):
        return []

    async def enrich_track(self, track):
        return track


def spotify_track(track_id: str, name: str, artist: str, popularity: int = 60) -> dict:
    return {
        "id": track_id,
        "provider": "spotify",
        "name": name,
        "artist": artist,
        "album": f"{name} album",
        "duration_ms": 210000,
        "popularity": popularity,
        "preview_url": "https://p.scdn.co/preview",
        "external_urls": {"spotify": f"https://open.spotify.com/track/{track_id}"},
        "uri": f"spotify:track:{track_id}",
        "release_date": "2020-01-01",
        "image_url": "https://i.scdn.co/image/x",
        "metadata": {},
    }


@pytest.fixture
def vocab():
    """Two clusters of tracks: t00-t29 near axis 0, u00-u29 near axis 1."""
    rng = np.random.default_rng(7)
    vectors: Dict[str, np.ndarray] = {}
    for i in range(30):
        base = np.zeros(DIM)
        base[0] = 1.0
        vectors[f"t{i:02d}"] = (base + rng.normal(0, 0.15, DIM)).astype(np.float32)
    for i in range(30):
        base = np.zeros(DIM)
        base[1] = 1.0
        vectors[f"u{i:02d}"] = (base + rng.normal(0, 0.15, DIM)).astype(np.float32)
    return vectors


@pytest.fixture
def embeddings(vocab):
    return FakeEmbeddingService(vocab)


@pytest.fixture
def track_meta(vocab, tmp_path):
    ids = sorted(vocab.keys())
    df = pd.DataFrame({
        "track_id": ids,
        "name": [f"Song {tid}" for tid in ids],
        # Two tracks per artist so the engine's artist dedup has work to do.
        "artist": [f"Artist {i // 2}" for i in range(len(ids))],
        "duration_ms": [200000 + 1000 * i for i in range(len(ids))],
        "playlist_count": list(range(1, len(ids) + 1)),
    })
    path = tmp_path / "track_meta.parquet"
    df.to_parquet(path)
    store = TrackMetaStore(str(path))
    assert store.available
    return store


@pytest.fixture
def mood_predictor(tmp_path):
    from services.recommendation.mood import MoodPredictor

    path = tmp_path / "mood.pkl"
    with open(path, "wb") as f:
        pickle.dump(
            {"model": DummyMoodModel(), "feature_names": ["valence", "energy", "acousticness", "danceability"]},
            f,
        )
    predictor = MoodPredictor(str(path))
    assert predictor.available
    return predictor


@pytest.fixture
def engine(embeddings, track_meta, mood_predictor):
    from services.recommendation.engine import RecommendationEngine
    from services.recommendation.ranker import LinearFallbackRanker

    return RecommendationEngine(
        embeddings=embeddings,
        ranker=LinearFallbackRanker(),
        mood=mood_predictor,
        ncf=None,
        track_meta=track_meta,
    )
