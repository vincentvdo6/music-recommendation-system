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
                  input_track_name=None, limit=20, serve_limit=None,
                  profile="balanced", session_feedback=None, exclude_track_ids=None):
        self.calls.append({
            "playlist_tracks": list(playlist_tracks),
            "input_track_id": input_track_id,
            "input_artist_name": input_artist_name,
            "limit": limit,
            "serve_limit": serve_limit,
            "profile": profile,
            "session_feedback": list(session_feedback or []),
            "exclude_track_ids": list(exclude_track_ids or []),
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
    service, _, engine = make_service(["r1", "r2"], playlist_catalog())
    recs, source, info = await service.recommend_from_playlist(
        PLAYLIST, seed="spotify:track:p1", limit=2
    )

    assert source == "ml-playlist"
    assert [r["id"] for r in recs] == ["r1", "r2"]
    assert recs[0]["album"] == "Rec One album"      # real Spotify metadata attached
    assert recs[0]["similarity_score"] == 0.9        # engine scores preserved
    assert info["tracks_in_model"] == 2
    assert engine.calls[0]["limit"] == 50
    assert info["engine_candidates"] == 2
    assert info["playable_candidates"] == 2
    assert info["engine_tracks_served"] == 2


async def test_profile_and_session_signals_reach_the_ranker():
    class SignalStore:
        def recent_session_signals(self, session_id):
            assert session_id == "session-12345678"
            return [("r1", 1.0), ("r2", -0.6)]

        def recent_session_exclusions(self, session_id):
            assert session_id == "session-12345678"
            return ["seen-1", "seen-2"]

    service, _, engine = make_service(["r1", "r2"], playlist_catalog())
    service.store = SignalStore()
    await service.recommend_from_playlist(
        PLAYLIST,
        seed="spotify:track:p1",
        limit=2,
        session_id="session-12345678",
        profile="explorer",
    )

    assert engine.calls[0]["serve_limit"] == 2
    assert engine.calls[0]["profile"] == "explorer"
    assert engine.calls[0]["session_feedback"] == [("r1", 1.0), ("r2", -0.6)]
    assert engine.calls[0]["exclude_track_ids"] == ["seen-1", "seen-2"]


async def test_artist_dedup_after_enrichment():
    # r1 and r3 share artist X; only the higher-ranked r1 survives.
    service, _, _ = make_service(["r1", "r3", "r2"], playlist_catalog())
    recs, _, _ = await service.recommend_from_playlist(
        PLAYLIST, seed="spotify:track:p1", limit=3, profile="balanced"
    )
    assert [r["id"] for r in recs] == ["r1", "r2"]


async def test_closest_profile_keeps_two_highly_ranked_tracks_from_same_artist():
    service, _, _ = make_service(["r1", "r3", "r2"], playlist_catalog())
    recs, _, _ = await service.recommend_from_playlist(
        PLAYLIST, seed="spotify:track:p1", limit=3, profile="familiar"
    )
    assert [r["id"] for r in recs] == ["r1", "r3", "r2"]


async def test_blacklisted_content_is_filtered():
    catalog = playlist_catalog()
    catalog["k1"] = spotify_track("k1", "Wheels on the Bus", "Kids Music Club")
    service, _, _ = make_service(["k1", "r1"], catalog)
    recs, _, _ = await service.recommend_from_playlist(PLAYLIST, seed="spotify:track:p1", limit=2)
    assert [r["id"] for r in recs] == ["r1"]


async def test_unknown_tracks_dropped_then_refilled_from_fallback():
    # "ghost" is unknown to Spotify; the popular fallback refills the slot
    # AFTER enrichment instead of returning a short (or empty) list.
    service, _, engine = make_service(["ghost", "r1"], playlist_catalog())
    engine.popular_fallback = lambda limit: [
        {"id": "r2", "name": "r2", "artist": "", "duration_ms": 0, "rank_score": 0.5,
         "similarity_score": 0.0, "recommendation": {"score": 0.5, "components": {}}, "why": ["popular"]}
    ]
    recs, _, _ = await service.recommend_from_playlist(PLAYLIST, seed="spotify:track:p1", limit=2)
    assert [r["id"] for r in recs] == ["r1", "r2"]


async def test_all_recs_unknown_to_spotify_still_returns_fallback():
    service, _, engine = make_service(["ghost1", "ghost2"], playlist_catalog())
    engine.popular_fallback = lambda limit: [
        {"id": "r1", "name": "r1", "artist": "", "duration_ms": 0, "rank_score": 0.5,
         "similarity_score": 0.0, "recommendation": {"score": 0.5, "components": {}}, "why": ["popular"]}
    ]
    recs, _, _ = await service.recommend_from_playlist(PLAYLIST, seed="spotify:track:p1", limit=2)
    assert [r["id"] for r in recs] == ["r1"]


async def test_popularity_fallback_does_not_repeat_recent_session_tracks():
    class ExposureStore:
        def recent_session_signals(self, session_id):
            return []

        def recent_session_exclusions(self, session_id):
            return ["r1"]

    service, _, engine = make_service([], playlist_catalog())
    service.store = ExposureStore()
    engine.popular_fallback = lambda limit: [
        {"id": track_id, "name": track_id, "artist": "", "duration_ms": 0,
         "rank_score": 0.5, "similarity_score": 0.0,
         "recommendation": {"score": 0.5, "components": {}}, "why": ["popular"]}
        for track_id in ("r1", "r2")
    ]

    recs, source, _ = await service.recommend_from_playlist(
        PLAYLIST, seed="spotify:track:p1", limit=2, session_id="session-12345678"
    )

    assert source == "popular-fallback"
    assert [rec["id"] for rec in recs] == ["r2"]


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


class ExtensionEngine(RecordingEngine):
    """Engine whose recs include an extension-catalog (dz:) track."""

    def recommend(self, playlist_tracks, input_track_id=None, input_artist_name=None,
                  input_track_name=None, limit=20, serve_limit=None,
                  profile="balanced", session_feedback=None, exclude_track_ids=None):
        super().recommend(playlist_tracks, input_track_id=input_track_id,
                          input_artist_name=input_artist_name,
                          input_track_name=input_track_name, limit=limit,
                          serve_limit=serve_limit, profile=profile,
                          session_feedback=session_feedback,
                          exclude_track_ids=exclude_track_ids)
        base = {"rank_score": 1.0, "similarity_score": 0.5,
                "recommendation": {"score": 1.0, "components": {}}, "why": ["test"]}
        return [
            {"id": "r1", "name": "Rec One", "artist": "X", "duration_ms": 210000, **base},
            {"id": "dz:900", "name": "fake prophet", "artist": "Tai Verdes",
             "duration_ms": 169000, **base},
        ]


class SearchableSpotify(FakeSpotify):
    def __init__(self, tracks, search_results):
        super().__init__(tracks)
        self.search_results = search_results
        self.search_calls: List[str] = []

    async def search_tracks(self, query: str, limit: int = 10):
        self.search_calls.append(query)
        return self.search_results


async def test_extension_dz_track_resolved_by_search():
    hit = spotify_track("sp_fp", "fake prophet", "Tai Verdes")
    hit["duration_ms"] = 168500
    spotify = SearchableSpotify(playlist_catalog(), [hit])
    service = MusicService(spotify=spotify, apple=FakeApple(), engine=ExtensionEngine([]))

    recs, _, _ = await service.recommend_from_playlist([], seed="spotify:track:seed1", limit=5)
    ids = [r["id"] for r in recs]
    assert "sp_fp" in ids            # dz:900 served under its Spotify identity
    assert not any(i.startswith("dz:") for i in ids)
    resolved = next(r for r in recs if r["id"] == "sp_fp")
    assert resolved["rank_score"] == 1.0  # engine scores survive the merge
    # the dz: id never entered a /tracks bulk chunk (it would 400 the batch)
    assert all("dz:900" not in call for call in spotify.bulk_calls)


async def test_extension_dz_track_unresolvable_is_dropped_and_cached():
    wrong = spotify_track("sp_x", "completely different", "Someone Else")
    spotify = SearchableSpotify(playlist_catalog(), [wrong])
    service = MusicService(spotify=spotify, apple=FakeApple(), engine=ExtensionEngine([]))

    recs, _, _ = await service.recommend_from_playlist([], seed="spotify:track:seed1", limit=5)
    assert not any(r["id"].startswith(("dz:", "sp_x")) for r in recs)

    await service.recommend_from_playlist([], seed="spotify:track:seed1", limit=5)
    assert len(spotify.search_calls) == 1  # negative result cached per process


async def test_extension_duration_mismatch_rejected():
    # right name and artist, but a 3x-length version (live/extended) — reject
    hit = spotify_track("sp_long", "fake prophet", "Tai Verdes")
    hit["duration_ms"] = 500000
    spotify = SearchableSpotify(playlist_catalog(), [hit])
    service = MusicService(spotify=spotify, apple=FakeApple(), engine=ExtensionEngine([]))

    recs, _, _ = await service.recommend_from_playlist([], seed="spotify:track:seed1", limit=5)
    assert not any(r["id"] == "sp_long" for r in recs)


# ---------------------------------------------------------------- era spread


def _dated_track(i, artist, release_date):
    return {"id": f"t{i}", "name": f"Song {i}", "artist": artist, "release_date": release_date}


def test_era_soft_cap_spreads_decades():
    service, _, _ = make_service([])
    pool = (
        [_dated_track(i, f"A{i}", "2015-01-01") for i in range(20)]
        + [_dated_track(20 + i, f"B{i}", "1985-06-01") for i in range(5)]
        + [_dated_track(25 + i, f"C{i}", "1999-11-01") for i in range(5)]
    )
    picked = service._deduplicate_by_artist(pool, max_per_artist=1, limit=10)
    assert len(picked) == 10
    decades = [service._release_decade(t) for t in picked]
    assert decades.count(2010) == 5  # seed's decade capped at half the list
    assert 1980 in decades  # lower-ranked cross-era candidates surface


def test_era_cap_is_soft_when_pool_is_one_decade():
    service, _, _ = make_service([])
    pool = [_dated_track(i, f"A{i}", "2015-01-01") for i in range(15)]
    picked = service._deduplicate_by_artist(pool, max_per_artist=1, limit=10)
    assert len(picked) == 10  # capped-out tracks refill rather than run short
    assert [t["id"] for t in picked] == [f"t{i}" for i in range(10)]  # rank order kept
