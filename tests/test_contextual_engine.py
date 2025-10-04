"""Tests for the context-aware recommendation engine."""

import asyncio

from services.recommendation.catalogue import TrackCatalogue
from services.recommendation.contextual_engine import ContextualRecommendationEngine


def test_contextual_engine_handles_context_without_seed():
    catalogue = TrackCatalogue()
    engine = ContextualRecommendationEngine(catalogue)

    recommendations = asyncio.run(
        engine.get_recommendations(
            seed_track_id=None,
            seed_metadata=None,
            context={"mood": "energetic", "activity": "workout", "time_of_day": "evening"},
            limit=3,
        )
    )

    assert recommendations, "Engine should return results for context-only queries"

    top = recommendations[0]
    assert "recommendation" in top and "components" in top["recommendation"]
    components = top["recommendation"]["components"]
    assert components["context"] >= 0.6
    assert "workout" in {tag.casefold() for tag in top.get("tags", {}).get("activities", [])}


def test_contextual_engine_skips_seed_track():
    catalogue = TrackCatalogue()
    engine = ContextualRecommendationEngine(catalogue)

    seed_name = "Still Water"
    seed_artist = "Eira"

    recommendations = asyncio.run(
        engine.get_recommendations(
            seed_track_id=None,
            seed_metadata={"name": seed_name, "artist": seed_artist},
            context={"mood": "calm"},
            limit=5,
        )
    )

    assert recommendations, "Engine should find calming tracks"
    assert all(track["id"] != "catalog:track:still_water" for track in recommendations)
    assert any(
        "ambient" in {tag.casefold() for tag in track.get("tags", {}).get("genres", [])}
        for track in recommendations
    )
