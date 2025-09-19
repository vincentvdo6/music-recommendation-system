"""Spotify Web API client with proper security and concurrency patterns."""

import asyncio
import base64
import logging
import os
import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

import httpx
from cachetools import TTLCache

logger = logging.getLogger(__name__)


class SpotifyClient:
    """Client for Spotify Web API using Client Credentials flow."""

    def __init__(self, client_id: Optional[str] = None, client_secret: Optional[str] = None):
        # Use environment variables for credentials - NEVER hardcode secrets!
        self.client_id = client_id or os.getenv("SPOTIFY_CLIENT_ID")
        self.client_secret = client_secret or os.getenv("SPOTIFY_CLIENT_SECRET")

        # Fallback mode if no credentials provided
        self.demo_mode = not (self.client_id and self.client_secret)

        if self.demo_mode:
            logger.warning("Spotify in DEMO mode (no credentials) — images/previews will be empty.")

        # HTTP client and token management
        self._client: Optional[httpx.AsyncClient] = None
        self._token_lock = asyncio.Lock()
        self._access_token: Optional[str] = None
        self._token_expiry: Optional[datetime] = None

        # Caching with thread-safe TTL cache
        self._search_cache = TTLCache(maxsize=500, ttl=60)
        self._features_cache = TTLCache(maxsize=5000, ttl=3600)
        self._cache_lock = asyncio.Lock()

        # Rate limiting
        self._sem = asyncio.Semaphore(10)

    async def start(self):
        """Initialize the HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(10.0, connect=5.0),
                limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
                headers={"Accept": "application/json"}
            )

    async def close(self):
        """Clean shutdown of HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def _cache_get(self, cache, key):
        async with self._cache_lock:
            return cache.get(key)

    async def _cache_set(self, cache, key, value):
        async with self._cache_lock:
            cache[key] = value

    async def _get_access_token(self) -> str:
        """Get access token using Client Credentials flow with proper locking."""
        if self.demo_mode:
            return "fallback_token"

        # Fast path: valid token
        if self._access_token and self._token_expiry and self._token_expiry > datetime.utcnow():
            return self._access_token

        async with self._token_lock:
            # Check again after acquiring the lock
            if self._access_token and self._token_expiry and self._token_expiry > datetime.utcnow():
                return self._access_token

            assert self._client, "Call start() before using the client"

            creds = f"{self.client_id}:{self.client_secret}".encode()
            b64 = base64.b64encode(creds).decode()
            try:
                resp = await self._client.post(
                    "https://accounts.spotify.com/api/token",
                    headers={"Authorization": f"Basic {b64}"},
                    data={"grant_type": "client_credentials"},
                )

                if resp.status_code != 200:
                    logger.error(f"Spotify token request failed: {resp.status_code} {resp.text}")
                    raise httpx.HTTPStatusError(f"Token request failed: {resp.status_code}", request=resp.request, response=resp)

                data = resp.json()

                if "access_token" not in data:
                    logger.error(f"Spotify token response missing access_token: {data}")
                    raise ValueError("Token response missing access_token")

                self._access_token = data["access_token"]
                expires_in = int(data.get("expires_in", 3600)) - 60
                self._token_expiry = datetime.utcnow() + timedelta(seconds=max(expires_in, 60))
                logger.info("Successfully obtained Spotify access token")
                return self._access_token

            except httpx.HTTPError as e:
                logger.error(f"Spotify token request HTTP error: {e}")
                raise
            except Exception as e:
                logger.error(f"Spotify token request failed: {e}")
                raise

    async def _request(self, method: str, url: str, *, params=None) -> httpx.Response:
        """Make a request with retry logic and rate limiting."""
        assert self._client, "Call start() before using the client"

        async with self._sem:  # Rate limiting
            token = await self._get_access_token()
            headers = {"Authorization": f"Bearer {token}"}

            # Basic retry with 429/backoff + transient network errors
            backoff = 0.5
            for attempt in range(5):
                resp = await self._client.request(method, url, params=params, headers=headers)

                if resp.status_code == 401 and not self.demo_mode:
                    # Token expired unexpectedly – force refresh and retry once
                    async with self._token_lock:
                        self._access_token = None
                        self._token_expiry = None
                    token = await self._get_access_token()
                    headers["Authorization"] = f"Bearer {token}"
                    continue

                if resp.status_code == 429:
                    ra = resp.headers.get("Retry-After")
                    sleep_s = float(ra) if ra else backoff
                    await asyncio.sleep(sleep_s)
                    backoff = min(backoff * 2, 8.0)
                    continue

                if resp.status_code >= 500:
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 2, 8.0)
                    continue

                return resp

            return resp  # Give last response to caller to handle

    async def search_tracks(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search for tracks on Spotify with caching."""
        if self.demo_mode:
            return self._fallback_search_results(query, limit)

        # Create cache key
        cache_key = (query.strip().lower(), limit)

        # Check cache first
        cached_result = await self._cache_get(self._search_cache, cache_key)
        if cached_result is not None:
            return cached_result

        try:
            resp = await self._request("GET", "https://api.spotify.com/v1/search",
                                     params={"q": query, "type": "track", "limit": limit})
            resp.raise_for_status()
            data = resp.json()
            tracks = [self._parse_spotify_track(t) for t in data.get("tracks", {}).get("items", [])]

            # Cache the result
            await self._cache_set(self._search_cache, cache_key, tracks)
            return tracks

        except httpx.HTTPError:
            # Graceful degradation to fallback
            return self._fallback_search_results(query, limit)

    async def get_track_features(self, track_id: str) -> Optional[Dict[str, Any]]:
        """Get audio features for a track with caching."""
        if self.demo_mode:
            return self._fallback_audio_features()

        # Check cache first
        cached_features = await self._cache_get(self._features_cache, track_id)
        if cached_features is not None:
            return cached_features

        try:
            resp = await self._request("GET", f"https://api.spotify.com/v1/audio-features/{track_id}")

            if resp.status_code == 404:
                return None  # Track not found

            resp.raise_for_status()
            features = resp.json()

            # Cache the result
            await self._cache_set(self._features_cache, track_id, features)
            return features

        except httpx.HTTPError:
            return self._fallback_audio_features()

    async def get_tracks_features_bulk(self, track_ids: List[str]) -> Dict[str, Any]:
        """Get audio features for multiple tracks."""
        if self.demo_mode:
            return {track_id: self._fallback_audio_features() for track_id in track_ids}

        # For simplicity, get features one by one (could be optimized with batch API)
        features_dict = {}
        for track_id in track_ids:
            features = await self.get_track_features(track_id)
            if features:
                features_dict[track_id] = features

        return features_dict

    async def get_recommendations(self, seed_tracks: List[str], limit: int = 10) -> List[Dict[str, Any]]:
        """Get recommendations based on seed tracks."""
        if self.demo_mode:
            return self._fallback_recommendations(seed_tracks, limit)

        try:
            # Use first seed track for recommendations
            seed_track = seed_tracks[0] if seed_tracks else ""
            params = {
                "seed_tracks": seed_track,
                "limit": limit,
                "market": "US"
            }

            resp = await self._request("GET", "https://api.spotify.com/v1/recommendations", params=params)
            resp.raise_for_status()
            data = resp.json()

            return [self._parse_spotify_track(t) for t in data.get("tracks", [])]

        except httpx.HTTPError as exc:
            logger.warning("Spotify recommendations API failed (%s), allowing Apple Music fallback", exc)
            return []

    def _parse_spotify_track(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Parse a Spotify track item into our format."""
        # Get highest resolution image
        image_url = None
        if item.get("album", {}).get("images"):
            image_url = item["album"]["images"][0]["url"]

        metadata = {
            "spotify_album_id": item.get("album", {}).get("id"),
            "spotify_artist_ids": [
                artist.get("id") for artist in item.get("artists", []) if artist.get("id")
            ],
        }

        return {
            "id": item["id"],
            "provider": "spotify",
            "name": item["name"],
            "artist": item["artists"][0]["name"] if item.get("artists") else "Unknown",
            "album": item.get("album", {}).get("name", "Unknown"),
            "duration_ms": item.get("duration_ms", 0),
            "popularity": item.get("popularity", 0),
            "preview_url": item.get("preview_url"),
            "external_urls": item.get("external_urls", {}),
            "uri": item.get("uri", ""),
            "release_date": item.get("album", {}).get("release_date", ""),
            "image_url": image_url,
            "metadata": {k: v for k, v in metadata.items() if v},
        }

    def _fallback_search_results(self, query: str, limit: int) -> List[Dict[str, Any]]:
        """Generate fallback search results when API is unavailable."""
        # Popular tracks for fallback responses
        fallback_tracks = [
            {"name": "Blinding Lights", "artist": "The Weeknd", "album": "After Hours", "popularity": 100},
            {"name": "Shape of You", "artist": "Ed Sheeran", "album": "÷", "popularity": 98},
            {"name": "Someone Like You", "artist": "Adele", "album": "21", "popularity": 95},
            {"name": "Bohemian Rhapsody", "artist": "Queen", "album": "A Night at the Opera", "popularity": 97},
            {"name": "Hotel California", "artist": "Eagles", "album": "Hotel California", "popularity": 94},
            {"name": "Imagine", "artist": "John Lennon", "album": "Imagine", "popularity": 93},
            {"name": "Billie Jean", "artist": "Michael Jackson", "album": "Thriller", "popularity": 96},
            {"name": "Stairway to Heaven", "artist": "Led Zeppelin", "album": "Led Zeppelin IV", "popularity": 95},
            {"name": "Sweet Caroline", "artist": "Neil Diamond", "album": "Brother Love's Travelling Salvation Show", "popularity": 89},
            {"name": "Don't Stop Believin'", "artist": "Journey", "album": "Escape", "popularity": 92}
        ]

        # Filter by query or return random selection
        query_lower = query.lower()
        matching_tracks = [
            track for track in fallback_tracks
            if query_lower in track["name"].lower() or query_lower in track["artist"].lower()
        ]

        if not matching_tracks:
            matching_tracks = random.sample(fallback_tracks, min(len(fallback_tracks), limit))

        results = []
        for i, track in enumerate(matching_tracks[:limit]):
            results.append({
                "id": f"track_{i}_{hash(track['name']) % 10000}",
                "name": track["name"],
                "artist": track["artist"],
                "album": track["album"],
                "duration_ms": random.randint(180000, 300000),  # 3-5 minutes
                "popularity": track["popularity"],
                "preview_url": None,
                "external_urls": {"spotify": f"https://open.spotify.com/track/fallback_{i}"},
                "uri": f"spotify:track:fallback_{i}",
                "release_date": f"{random.randint(1970, 2024)}-{random.randint(1, 12):02d}-{random.randint(1, 28):02d}",
                "image_url": None,
                "provider": "spotify-fallback",
                "metadata": {},
            })

        return results

    def _fallback_audio_features(self) -> Dict[str, Any]:
        """Generate fallback audio features."""
        return {
            "acousticness": random.uniform(0.0, 1.0),
            "danceability": random.uniform(0.0, 1.0),
            "energy": random.uniform(0.0, 1.0),
            "instrumentalness": random.uniform(0.0, 1.0),
            "liveness": random.uniform(0.0, 1.0),
            "loudness": random.uniform(-60.0, 0.0),
            "speechiness": random.uniform(0.0, 1.0),
            "tempo": random.uniform(60.0, 200.0),
            "valence": random.uniform(0.0, 1.0),
            "key": random.randint(0, 11),
            "mode": random.randint(0, 1),
            "time_signature": random.choice([3, 4, 5])
        }

    def _fallback_recommendations(self, seed_tracks: List[str], limit: int) -> List[Dict[str, Any]]:
        """Generate fallback recommendations."""
        recommendations = [
            {"name": "Levitating", "artist": "Dua Lipa", "album": "Future Nostalgia", "popularity": 94},
            {"name": "Bad Habits", "artist": "Ed Sheeran", "album": "=", "popularity": 92},
            {"name": "Stay", "artist": "The Kid LAROI, Justin Bieber", "album": "F*CK LOVE 3: OVER YOU", "popularity": 96},
            {"name": "Good 4 U", "artist": "Olivia Rodrigo", "album": "SOUR", "popularity": 89},
            {"name": "Watermelon Sugar", "artist": "Harry Styles", "album": "Fine Line", "popularity": 88},
        ]

        selected = random.sample(recommendations, min(len(recommendations), limit))

        results = []
        for i, track in enumerate(selected):
            results.append({
                "id": f"rec_{i}_{hash(track['name']) % 10000}",
                "name": track["name"],
                "artist": track["artist"],
                "album": track["album"],
                "duration_ms": random.randint(180000, 300000),
                "popularity": track["popularity"],
                "preview_url": None,
                "external_urls": {"spotify": f"https://open.spotify.com/track/rec_{i}"},
                "uri": f"spotify:track:rec_{i}",
                "release_date": f"{random.randint(2020, 2024)}-{random.randint(1, 12):02d}-{random.randint(1, 28):02d}",
                "image_url": None,
                "provider": "spotify-fallback",
                "metadata": {},
            })

        return results