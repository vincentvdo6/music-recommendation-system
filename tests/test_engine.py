"""Engine pipeline tests over the fake embedding/metadata/mood stack."""

import numpy as np

from services.recommendation.engine import RecommendationEngine
from services.recommendation.ranker import LinearFallbackRanker


def playlist_of(prefix, count):
    return [f"{prefix}{i:02d}" for i in range(count)]


def test_recommendations_have_expected_shape(engine):
    recs = engine.recommend(playlist_of("t", 5), input_track_id="t10", limit=5)
    assert recs
    for rec in recs:
        assert rec["id"] not in playlist_of("t", 5)
        assert rec["id"] != "t10"
        assert isinstance(rec["rank_score"], float)
        assert 0.0 <= rec["recommendation"]["components"]["mood_match"] <= 1.0
        assert rec["why"]
        assert rec["name"].startswith("Song ")


def test_seed_dominates_retrieval(engine):
    # Playlist in cluster t, seed in cluster u: candidates should lean toward u.
    recs = engine.recommend(playlist_of("t", 5), input_track_id="u00", limit=8)
    assert recs
    u_count = sum(1 for r in recs if r["id"].startswith("u"))
    assert u_count > len(recs) / 2


def test_different_seeds_produce_different_recommendations(engine):
    playlist = playlist_of("t", 5) + playlist_of("u", 5)
    recs_a = [r["id"] for r in engine.recommend(playlist, input_track_id="t20", limit=5)]
    recs_b = [r["id"] for r in engine.recommend(playlist, input_track_id="u20", limit=5)]
    assert recs_a != recs_b


def test_artist_dedup_one_track_per_artist(engine, track_meta):
    recs = engine.recommend(playlist_of("t", 5), input_track_id="t10", limit=10)
    artists = [track_meta.lookup([r["id"]])["artist_norm"].iloc[0] for r in recs]
    assert len(artists) == len(set(artists))


def test_proxy_seed_used_when_track_not_in_model(engine):
    # Unknown track id, but the artist exists in track_meta -> proxy retrieval.
    seed_id, proxied = engine._resolve_seed("unknown_track", "Artist 0")
    assert proxied
    assert seed_id in {"t00", "t01"}  # Artist 0's tracks (sorted ids t00, t01)


def test_no_seed_no_playlist_coverage_returns_empty(embeddings, track_meta):
    engine = RecommendationEngine(
        embeddings=embeddings, ranker=LinearFallbackRanker(), track_meta=track_meta
    )
    assert engine.recommend(["not_in_vocab_1", "not_in_vocab_2"]) == []


def test_popular_fallback_is_deterministic(engine):
    a = [r["id"] for r in engine.popular_fallback(5)]
    b = [r["id"] for r in engine.popular_fallback(5)]
    assert a == b
    assert len(a) == 5


def test_coverage_reports_model_visibility(engine):
    cov = engine.coverage(["t00", "t01", "missing"], input_track_id="u00")
    assert cov == {"tracks_in_model": 2, "playlist_size": 3, "seed_in_model": True}


def test_engine_works_without_optional_subsystems(embeddings):
    engine = RecommendationEngine(embeddings=embeddings, ranker=LinearFallbackRanker())
    recs = engine.recommend(playlist_of("t", 5), input_track_id="t10", limit=5)
    assert recs
    # Without track_meta there are no artist names; ids pass through as names.
    assert all(r["name"] == r["id"] for r in recs)


def test_similarity_score_is_seed_cosine(engine, vocab):
    recs = engine.recommend(playlist_of("t", 5), input_track_id="t10", limit=5)
    seed_unit = vocab["t10"] / np.linalg.norm(vocab["t10"])
    for rec in recs:
        vec = vocab[rec["id"]]
        expected = float(vec / np.linalg.norm(vec) @ seed_unit)
        assert abs(rec["similarity_score"] - expected) < 1e-5
