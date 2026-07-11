"""Engine pipeline tests over the fake embedding/metadata/mood stack."""

import numpy as np
import pandas as pd

from services.recommendation.engine import RecommendationEngine
from services.recommendation.features import FEATURE_NAMES
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


def test_empty_fallback_weights_are_respected():
    ranker = LinearFallbackRanker({})
    features = pd.DataFrame(np.ones((2, len(FEATURE_NAMES))), columns=FEATURE_NAMES)
    assert np.array_equal(ranker.predict(features), np.zeros(2))


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


def test_artist_precap_limits_candidates_per_artist(engine, track_meta):
    from services.recommendation.engine import ARTIST_PRECAP

    recs = engine.recommend(playlist_of("t", 5), input_track_id="t10", limit=10)
    artists = [track_meta.lookup([r["id"]])["artist_norm"].iloc[0] for r in recs]
    counts = {a: artists.count(a) for a in artists}
    assert max(counts.values()) <= ARTIST_PRECAP
    # final 1-per-artist diversity is the service layer's job post-enrichment


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


def _mean_seed_cos(recs):
    return np.mean([r["similarity_score"] for r in recs])


def test_seed_affinity_pulls_results_toward_seed(embeddings, track_meta, mood_predictor):
    # Mixed playlist, seed from cluster u: a stronger blend must not lower the
    # average seed cosine of what gets recommended.
    playlist = playlist_of("t", 8) + playlist_of("u", 2)
    results = {}
    for affinity in (0.0, 25.0):
        engine = RecommendationEngine(
            embeddings=embeddings, ranker=LinearFallbackRanker(),
            mood=mood_predictor, track_meta=track_meta, seed_affinity=affinity,
        )
        results[affinity] = engine.recommend(playlist, input_track_id="u20", limit=8)
    assert _mean_seed_cos(results[25.0]) >= _mean_seed_cos(results[0.0])


def test_seed_only_mode_works_without_playlist(engine):
    recs = engine.recommend([], input_track_id="t10", limit=5)
    assert recs
    assert all(r["id"] != "t10" for r in recs)
    assert all(r["id"].startswith("t") for r in recs)  # pool is 100% seed neighbors


def test_discovery_dampens_popularity(embeddings, track_meta, mood_predictor):
    # track_meta assigns playlist_count = 1..N over sorted ids, so u-cluster
    # tracks are the "popular" ones. A strong discovery weight must not raise
    # the average popularity of what gets recommended.
    def mean_pop(recs):
        return np.mean([track_meta.lookup([r["id"]])["playlist_count"].iloc[0] for r in recs])

    results = {}
    for discovery in (0.0, 25.0):
        engine = RecommendationEngine(
            embeddings=embeddings, ranker=LinearFallbackRanker(),
            mood=mood_predictor, track_meta=track_meta, discovery=discovery,
        )
        results[discovery] = engine.recommend(playlist_of("t", 5), input_track_id="t10", limit=8)
    assert mean_pop(results[25.0]) <= mean_pop(results[0.0])


def test_audio_features_boost_sound_alike_candidates(embeddings, track_meta, mood_predictor, audio_store):
    # Even-numbered tracks carry audio embeddings; with the audio store wired,
    # candidates that SOUND like the seed earn extra score, so the share of
    # audio-covered tracks in the top results must not drop.
    def audio_share(recs):
        return np.mean([int(r["id"][1:]) % 2 == 0 for r in recs])

    kwargs = dict(embeddings=embeddings, ranker=LinearFallbackRanker(),
                  mood=mood_predictor, track_meta=track_meta)
    without = RecommendationEngine(**kwargs).recommend(playlist_of("t", 5), input_track_id="t10", limit=8)
    audio_engine = RecommendationEngine(**kwargs, audio=audio_store)
    with_audio = audio_engine.recommend(playlist_of("t", 5), input_track_id="t10", limit=8)
    assert audio_share(with_audio) >= audio_share(without)

    # Seed audio resolution: direct for covered tracks, same-artist proxy otherwise.
    assert audio_engine._seed_audio("t10", None) is not None
    assert audio_engine._seed_audio("t01", "Artist 0") is not None  # t01 has no audio; t00 (Artist 0) does
    assert audio_engine._seed_audio("t01", None) is None


def test_seed_cos_floor_filters_dissimilar_candidates(embeddings, track_meta, mood_predictor):
    # Playlist entirely in cluster t, seed in cluster u: playlist-side
    # candidates are near-orthogonal to the seed and should be floored out
    # as long as enough seed-side candidates remain.
    engine = RecommendationEngine(
        embeddings=embeddings, ranker=LinearFallbackRanker(),
        mood=mood_predictor, track_meta=track_meta,
    )
    recs = engine.recommend(playlist_of("t", 5), input_track_id="u00", limit=6)
    assert recs
    from services.recommendation.engine import SEED_COS_FLOOR
    assert all(r["similarity_score"] >= SEED_COS_FLOOR for r in recs)
