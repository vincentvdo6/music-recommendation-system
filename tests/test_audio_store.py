"""AudioStore: parquet loading, lookups, zero-fill semantics."""

import numpy as np

from services.recommendation.audio_store import AudioStore


def test_vectors_are_unit_normalized(audio_store):
    vec = audio_store.vector("t00")
    assert vec is not None
    assert np.isclose(np.linalg.norm(vec), 1.0, atol=1e-3)


def test_unknown_track_returns_none(audio_store):
    assert audio_store.vector("t01") is None      # odd tracks have no audio
    assert audio_store.vector("nope") is None


def test_matrix_for_zero_fills_and_masks(audio_store):
    vecs, mask = audio_store.matrix_for(["t00", "t01", "u02"])
    assert vecs.shape == (3, audio_store.dim)
    assert mask.tolist() == [1.0, 0.0, 1.0]
    assert np.abs(vecs[1]).sum() == 0.0


def test_mean_vector_skips_unknown(audio_store):
    mean = audio_store.mean_vector(["t00", "t01", "t02"])
    assert mean is not None
    assert audio_store.mean_vector(["t01", "t03"]) is None


def test_clusters_are_acoustically_separated(audio_store):
    t, u = audio_store.vector("t00"), audio_store.vector("u00")
    assert float(t @ audio_store.vector("t02")) > 0.8
    assert float(t @ u) < 0.5


def test_unavailable_store():
    store = AudioStore("does/not/exist.parquet")
    assert not store.available
    assert store.vector("t00") is None
