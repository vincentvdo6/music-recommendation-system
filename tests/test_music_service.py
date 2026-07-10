"""Service orchestration tests: seed resolution, enrichment, dedup, blacklist."""

from typing import List

import pytest

from services.music.service import MusicService
from tests.conftest import FakeApple, FakeSpotify, spotify_track


class RecordingEngine:
    """Fake engine that records calls and returns canned recommendations."""

    def __init__(self, rec_ids: List[str]):
        self.rec_ids = rec_ids
        self.calls: List[dict] = []

    def recommend(self, playlist_tracks, input_track_id=None, input_artist_name=None,
                  input_track_name=None, limit=20):
        self.calls.append({
            "playlist_tracks": list(playlist_tracks),
            "input_track_id": input_track_id,
            "input_artist_name": input_artist_name,
            "limit": limit,
        })
        return [
            {"id": tid, "name": tid, "artist": "", "duration_ms": 0,
             "rank_score": 1.0 - 0.01 * i, "similarity_score": 0.9 - 0.01 * i,
             "recommendation": {"score": 1.0, "components": {}}, "why": ["test"]}
            for i, tid in enumerate(self.rec_ids)
        ]

    def popular_fallback(self, limit):
        return []

    def coverage(self, playlist_tracks, input_track_id=None):
        return {"tracks_in_model": len(playlist_tracks), "seed_in_model": True}


def make_service(rec_ids, catalog=None):
    catalog = catalog or {}
    spotify = FakeSpotify(tracks=catalog)
    engine = RecordingEngine(rec_ids)
    service = MusicService(spotify=spotify, apple=FakeApple(), engine=engine)
    return service, spotify, engine


PLAYLIST = [
    {"spotify_id": "p1", "name": "Track One", "artist": "A"},
    {"spotify_id": "p2", "name": "Track Two", "artist": "B"},
]


def playlist_catalog():
    return {
        "p1": spotify_track("p1", "Track One", "A"),
        "p2": spotify_track("p2", "Track Two", "B"),
        "r1": spotify_track("r1", "Rec One", "X"),
        "r2": spotify_track("r2", "Rec Two", "Y"),
        "r3": spotify_track("r3", "Rec Three", "X"),  # same artist as r1
        "seed1": spotify_track("seed1", "Seed Song", "Seeder"),
    }


async def test_seed_uri_is_resolved_and_passed_to_engine():
    service, _, engine = make_service(["r1", "r2"], playlist_catalog())
    await service.recommend_from_playlist(PLAYLIST, seed="spotify:track:seed1", limit=2)

    assert engine.calls[0]["input_track_id"] == "seed1"
    assert engine.calls[0]["input_artist_name"] == "Seeder"
    assert set(engine.calls[0]["playlist_tracks"]) == {"p1", "p2"}


async def test_different_seeds_reach_engine():
    service, _, engine = make_service(["r1"], playlist_catalog())
    await service.recommend_from_playlist(PLAYLIST, seed="spotify:track:p1", limit=1)
    await service.recommend_from_playlist(PLAYLIST, seed="spotify:track:p2", limit=1)
    assert [c["input_track_id"] for c in engine.calls] == ["p1", "p2"]


async def test_recommendations_are_enriched_with_spotify_metadata():
    service, _, _ = make_service(["r1", "r2"], playlist_catalog())
    recs, source, info = await service.recommend_from_playlist(
        PLAYLIST, seed="spotify:track:p1", limit=2
    )

    assert source == "ml-playlist"
    assert [r["id"] for r in recs] == ["r1", "r2"]
    assert recs[0]["album"] == "Rec One album"      # real Spotify metadata attached
    assert recs[0]["similarity_score"] == 0.9        # engine scores preserved
    assert info["tracks_in_model"] == 2


async def test_artist_dedup_after_enrichment():
    # r1 and r3 share artist X; only the higher-ranked r1 survives.
    service, _, _ = make_service(["r1", "r3", "r2"], playlist_catalog())
    recs, _, _ = await service.recommend_from_playlist(PLAYLIST, seed="spotify:track:p1", limit=3)
    assert [r["id"] for r in recs] == ["r1", "r2"]


async def test_blacklisted_content_is_filtered():
    catalog = playlist_catalog()
    catalog["k1"] = spotify_track("k1", "Wheels on the Bus", "Kids Music Club")
    service, _, _ = make_service(["k1", "r1"], catalog)
    recs, _, _ = await service.recommend_from_playlist(PLAYLIST, seed="spotify:track:p1", limit=2)
    assert [r["id"] for r in recs] == ["r1"]


async def test_unknown_tracks_dropped_by_enrichment():
    service, _, _ = make_service(["ghost", "r1"], playlist_catalog())
    recs, _, _ = await service.recommend_from_playlist(PLAYLIST, seed="spotify:track:p1", limit=2)
    assert [r["id"] for r in recs] == ["r1"]


async def test_name_only_playlist_search_does_not_include_none():
    class QuerySpotify(FakeSpotify):
        def __init__(self):
            super().__init__()
            self.queries = []

        async def search_tracks(self, query: str, limit: int = 10):
            self.queries.append(query)
            return [spotify_track("resolved", "Track Name", "Artist")]

    spotify = QuerySpotify()
    service = MusicService(spotify=spotify, apple=FakeApple(), engine=RecordingEngine([]))
    resolved = await service._resolve_playlist_tracks([{"name": "Track Name", "artist": None}])

    assert spotify.queries == ["Track Name"]
    assert list(resolved) == ["resolved"]


async def test_empty_playlist_and_no_seed_short_circuits():
    service, _, engine = make_service(["r1"], playlist_catalog())
    recs, source, info = await service.recommend_from_playlist([], seed=None, limit=5)
    assert recs == []
    assert engine.calls == []
    assert info["playlist_size"] == 0


async def test_seed_only_mode_without_playlist():
    service, _, engine = make_service(["r1", "r2"], playlist_catalog())
    recs, source, info = await service.recommend_from_playlist(
        [], seed="spotify:track:seed1", limit=2
    )
    assert source == "ml-seed"
    assert [r["id"] for r in recs] == ["r1", "r2"]
    assert engine.calls[0]["playlist_tracks"] == []
    assert engine.calls[0]["input_track_id"] == "seed1"
    assert engine.calls[0]["input_artist_name"] == "Seeder"
    assert info["playlist_size"] == 0


async def test_engine_missing_raises_runtime_error():
    service = MusicService(spotify=FakeSpotify(playlist_catalog()), apple=FakeApple(), engine=None)
    with pytest.raises(RuntimeError, match="engine unavailable"):
        await service.recommend_from_playlist(PLAYLIST, seed=None, limit=5)


def test_extract_spotify_id_variants():
    extract = MusicService._extract_spotify_id
    assert extract("spotify:track:abc123") == "abc123"
    assert extract("https://open.spotify.com/track/abc123?si=xyz") == "abc123"
    assert extract("4uLU6hMCjMI75M1A2tKUQC") == "4uLU6hMCjMI75M1A2tKUQC"
    assert extract("not a track") is None
    assert extract(None) is None
