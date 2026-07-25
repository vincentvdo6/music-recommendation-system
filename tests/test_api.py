"""API tests with an injected fake MusicService (no model loading, no network)."""

import pytest
from httpx import ASGITransport, AsyncClient

pytest.importorskip("fastapi")

from api.main import app  # noqa: E402
from tests.conftest import spotify_track  # noqa: E402


class FakeMusicService:
    def __init__(self):
        self.recommendation_calls = []

    async def search_tracks(self, query, *, limit=10):
        return [spotify_track("s1", f"Result for {query}", "Some Artist")], "spotify"

    async def recommend_from_playlist(
        self, tracks_payload, *, seed=None, limit=5, session_id=None, profile="familiar"
    ):
        self.recommendation_calls.append({"session_id": session_id, "profile": profile})
        recs = [
            {
                **spotify_track("r1", "Rec One", "X"),
                "similarity_score": 0.87,
                "rank_score": 0.91,
                "recommendation": {"score": 0.91, "components": {"seed_similarity": 0.87}},
                "why": ["often played alongside the seed track"],
            }
        ]
        info = {"playlist_size": len(tracks_payload), "resolved_tracks": len(tracks_payload),
                "tracks_in_model": 1, "seed_in_model": True}
        return recs[:limit], "ml-playlist", info

    async def import_playlist_from_url(self, url, *, limit=300):
        tracks = [{"raw": "Song - Artist", "spotify_id": "p1", "name": "Song",
                   "artist": "Artist", "seed": True}]
        summary = {"id": "pl1", "name": "My Playlist", "total_tracks": 1, "loaded_tracks": 1}
        return tracks, summary


@pytest.fixture
async def client():
    previous_music = getattr(app.state, "music", None)
    previous_feedback = getattr(app.state, "feedback_store", None)
    app.state.music = FakeMusicService()
    app.state.feedback_store = None
    transport = ASGITransport(app=app)
    try:
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            yield ac
    finally:
        app.state.music = previous_music
        app.state.feedback_store = previous_feedback


async def test_health(client):
    resp = await client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"


async def test_root_serves_html(client):
    resp = await client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]


async def test_search(client):
    resp = await client.get("/api/v1/search", params={"q": "ivy"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["count"] == 1
    assert data["results"][0]["track"]["name"] == "Result for ivy"


async def test_search_failure_does_not_expose_internal_error(client):
    class FailingMusicService:
        async def search_tracks(self, query, *, limit=10):
            raise RuntimeError("sensitive provider detail")

    app.state.music = FailingMusicService()
    resp = await client.get("/api/v1/search", params={"q": "ivy"})
    assert resp.status_code == 500
    assert resp.json() == {"detail": "Search failed"}


async def test_search_rate_limit_returns_429(client):
    transport = ASGITransport(app=app, client=("rate-limit-test", 1234))
    async with AsyncClient(transport=transport, base_url="http://test") as rate_client:
        # /search allows 120/minute (autocomplete: one request per keystroke)
        responses = [
            await rate_client.get("/api/v1/search", params={"q": "ivy"})
            for _ in range(121)
        ]

    assert responses[-2].status_code == 200
    assert responses[-1].status_code == 429
    assert responses[-1].headers["retry-after"] == "60"


async def test_playlist_recommendations(client):
    payload = {
        "tracks": [{"spotify_id": "p1", "name": "Song", "artist": "Artist"}],
        "seed": "spotify:track:p1",
        "limit": 5,
        "session_id": "session-12345678",
        "profile": "explorer",
    }
    resp = await client.post("/api/v1/playlist/recommendations", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 1
    hit = data["recommendations"][0]
    assert hit["similarity_score"] == 0.87
    assert hit["explanation"]["top_factors"] == ["often played alongside the seed track"]
    assert data["playlist"]["tracks_in_model"] == 1
    assert data["impression_id"]
    assert data["profile"] == "explorer"


async def test_feedback_accepts_only_tracks_from_the_impression(client, tmp_path):
    from services.storage import LocalStore

    store = LocalStore(tmp_path / "feedback.sqlite3")
    app.state.feedback_store = store
    response = await client.post(
        "/api/v1/playlist/recommendations",
        json={"tracks": [], "seed": "spotify:track:p1", "limit": 5},
    )
    impression_id = response.json()["impression_id"]

    accepted = await client.post(
        "/api/v1/feedback",
        json={
            "impression_id": impression_id,
            "track_id": "r1",
            "event": "like",
            "position": 0,
        },
    )
    unknown = await client.post(
        "/api/v1/feedback",
        json={
            "impression_id": impression_id,
            "track_id": "not-shown",
            "event": "like",
            "position": 1,
        },
    )

    assert accepted.status_code == 200
    assert accepted.json() == {"accepted": True}
    assert unknown.status_code == 404
    store.close()
    app.state.feedback_store = None


async def test_playlist_import(client):
    resp = await client.post("/api/v1/playlist/import", json={"url": "https://open.spotify.com/playlist/x"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["summary"]["name"] == "My Playlist"
    assert data["tracks"][0]["spotify_id"] == "p1"


async def test_dead_endpoints_are_gone(client):
    assert (await client.get("/api/v1/recommendations")).status_code in (404, 405)
    assert (await client.post("/api/v1/track-interaction")).status_code in (404, 405)
    assert (await client.get("/api")).status_code == 404
