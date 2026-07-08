"""API tests with an injected fake MusicService (no model loading, no network)."""

import pytest
from httpx import ASGITransport, AsyncClient

pytest.importorskip("fastapi")

from api.main import app  # noqa: E402
from tests.conftest import spotify_track  # noqa: E402


class FakeMusicService:
    async def search_tracks(self, query, *, limit=10):
        return [spotify_track("s1", f"Result for {query}", "Some Artist")], "spotify"

    async def recommend_from_playlist(self, tracks_payload, *, seed=None, limit=5):
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
    app.state.music = FakeMusicService()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


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


async def test_playlist_recommendations(client):
    payload = {
        "tracks": [{"spotify_id": "p1", "name": "Song", "artist": "Artist"}],
        "seed": "spotify:track:p1",
        "limit": 5,
    }
    resp = await client.post("/api/v1/playlist/recommendations", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 1
    hit = data["recommendations"][0]
    assert hit["similarity_score"] == 0.87
    assert hit["explanation"]["top_factors"] == ["often played alongside the seed track"]
    assert data["playlist"]["tracks_in_model"] == 1


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
