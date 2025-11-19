"""Tests for the playlist-personalised recommendation engine."""

import asyncio

import pytest

from services.recommendation.catalogue import TrackCatalogue
from services.recommendation.contextual_engine import ContextualRecommendationEngine


def test_engine_requires_user_profile() -> None:
    catalogue = TrackCatalogue()
    engine = ContextualRecommendationEngine(catalogue)

    with pytest.raises(ValueError):
        asyncio.run(engine.get_recommendations(limit=3))


def test_playlist_personalisation_excludes_original_tracks() -> None:
    catalogue = TrackCatalogue()
    engine = ContextualRecommendationEngine(catalogue)

    playlist_entries = catalogue.all_tracks()[:4]
    profile = engine.build_user_profile(playlist_entries)

    assert "audio_features" in profile and profile["audio_features"], "Profile should include averaged audio features"
    assert profile["context"].get("moods"), "Profile should include derived mood preferences"

    seed_entry = playlist_entries[0]

    recommendations = asyncio.run(
        engine.get_recommendations(
            seed_track_id=seed_entry["id"],
            seed_metadata={"name": seed_entry["name"], "artist": seed_entry["artist"]},
            seed_features=seed_entry.get("audio_features"),
            context=None,
            user_profile=profile,
            limit=5,
        )
    )

    assert recommendations, "Engine should return personalised recommendations"

    playlist_ids = {entry["id"] for entry in playlist_entries}
    assert all(track["id"] not in playlist_ids for track in recommendations), "Results should exclude playlist tracks"

    playlist_pairs = {
        (entry.get("name", "").casefold(), entry.get("artist", "").casefold())
        for entry in playlist_entries
    }
    recommended_pairs = {
        (track.get("name", "").casefold(), track.get("artist", "").casefold())
        for track in recommendations
    }
    assert playlist_pairs.isdisjoint(recommended_pairs), "Playlist songs should not be repeated in recommendations"


def test_context_normalisation_adds_adjacent_genres_from_moods() -> None:
    catalogue = TrackCatalogue()
    engine = ContextualRecommendationEngine(catalogue)

    normalised = engine._normalise_context({"moods": ["melancholic"]})

    assert normalised["adjacent_genres"], "Moods should expand into adjacent genres"
    assert "indie" in normalised["adjacent_genres"], "Melancholic mood should include indie-leaning genres"


def test_adjacent_genres_boost_context_alignment() -> None:
    catalogue = TrackCatalogue()
    engine = ContextualRecommendationEngine(catalogue)

    context = engine._normalise_context({"moods": ["melancholic"]})

    candidate_adjacent = {"tags": {"moods": [], "genres": ["indie"]}}
    candidate_unrelated = {"tags": {"moods": [], "genres": ["dance"]}}

    adjacent_score = engine._context_alignment(candidate_adjacent, context, None)
    unrelated_score = engine._context_alignment(candidate_unrelated, context, None)

    assert adjacent_score > unrelated_score, "Adjacent genres should provide a higher context score even without mood tags"
