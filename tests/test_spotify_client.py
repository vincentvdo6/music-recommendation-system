"""Spotify client fallback behavior without external network calls."""

from unittest.mock import AsyncMock

import httpx

from services.spotify.client import SpotifyClient


async def test_demo_search_returns_no_synthetic_tracks(monkeypatch):
    monkeypatch.delenv("SPOTIFY_CLIENT_ID", raising=False)
    monkeypatch.delenv("SPOTIFY_CLIENT_SECRET", raising=False)
    client = SpotifyClient()

    assert client.demo_mode
    assert await client.search_tracks("anything") == []


async def test_http_failure_returns_empty_for_provider_fallback():
    client = SpotifyClient(client_id="client", client_secret="secret")
    client._request = AsyncMock(side_effect=httpx.ConnectError("network unavailable"))

    assert await client.search_tracks("anything") == []
