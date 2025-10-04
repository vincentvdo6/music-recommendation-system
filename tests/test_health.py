"""Basic smoke tests for the API."""

from contextlib import asynccontextmanager

import pytest
from httpx import ASGITransport, AsyncClient

pytest.importorskip("fastapi")

from api.main import app


@asynccontextmanager
async def lifespan_client():
      """Yield an AsyncClient bound to the FastAPI app via ASGI transport."""
      await app.router.startup()
      try:
          transport = ASGITransport(app=app)
          async with AsyncClient(transport=transport, base_url="http://test") as client:
              yield client
      finally:
          await app.router.shutdown()

@pytest.mark.asyncio
async def test_health():
    """Test health endpoint returns 200."""
    async with lifespan_client() as async_client:
        resp = await async_client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert "version" in data
        assert "timestamp" in data


@pytest.mark.asyncio
async def test_root():
    """Test root endpoint serves HTML."""
    async with lifespan_client() as async_client:
        resp = await async_client.get("/")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]


@pytest.mark.asyncio
async def test_api_info():
    """Test API info endpoint."""
    async with lifespan_client() as async_client:
        resp = await async_client.get("/api")
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "Music Recommendation API"
        assert "endpoints" in data
