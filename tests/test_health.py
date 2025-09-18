"""Basic smoke tests for the API."""

import pytest
from httpx import AsyncClient
from api.main import app


@pytest.mark.asyncio
async def test_health():
    """Test health endpoint returns 200."""
    async with AsyncClient(app=app, base_url="http://test") as ac:
        r = await ac.get("/health")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "healthy"
        assert "version" in data
        assert "timestamp" in data


@pytest.mark.asyncio
async def test_root():
    """Test root endpoint serves HTML."""
    async with AsyncClient(app=app, base_url="http://test") as ac:
        r = await ac.get("/")
        assert r.status_code == 200
        assert "text/html" in r.headers["content-type"]


@pytest.mark.asyncio
async def test_api_info():
    """Test API info endpoint."""
    async with AsyncClient(app=app, base_url="http://test") as ac:
        r = await ac.get("/api")
        assert r.status_code == 200
        data = r.json()
        assert data["name"] == "Music Recommendation API"
        assert "endpoints" in data