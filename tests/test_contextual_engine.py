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
