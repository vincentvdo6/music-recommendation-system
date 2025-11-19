"""Spotify Web API client with proper security and concurrency patterns."""

import asyncio
import base64
import json
import logging
import os
import random
import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Sequence

import httpx
from cachetools import TTLCache
from dotenv import load_dotenv

# Ensure .env is loaded before SpotifyClient is instantiated
load_dotenv()

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
        else:
            logger.info(f"Spotify initialized with credentials (Client ID: {self.client_id[:8]}...)")

        # HTTP client and token management
        self._client: Optional[httpx.AsyncClient] = None
        self._token_lock = asyncio.Lock()
        self._access_token: Optional[str] = None
        self._token_expiry: Optional[datetime] = None

        # Caching with thread-safe TTL cache
        self._search_cache = TTLCache(maxsize=2000, ttl=300)  # 2k searches, 5min
        self._features_cache = TTLCache(maxsize=10000, ttl=7200)  # 10k features, 2hrs
        self._artist_cache = TTLCache(maxsize=5000, ttl=86400)  # Artist metadata, 24hrs
        self._cache_lock = asyncio.Lock()

        # Rate limiting
        self._sem = asyncio.Semaphore(10)

        self._playlist_data_pattern = re.compile(
            r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>',
            re.S,
        )

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
        """Get audio features for a track with caching and retry logic."""
        if self.demo_mode:
            return self._fallback_audio_features()

        # Check cache first
        cached_features = await self._cache_get(self._features_cache, track_id)
        if cached_features is not None:
            return cached_features

        # Retry logic for rate limiting (403/429)
        max_retries = 3
        base_delay = 0.5  # Start with 500ms delay

        for attempt in range(max_retries):
            try:
                resp = await self._request("GET", f"https://api.spotify.com/v1/audio-features/{track_id}")

                if resp.status_code == 404:
                    return None  # Track not found

                if resp.status_code in [403, 429]:
                    # Rate limited - wait and retry
                    if attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt)  # Exponential backoff
                        logger.warning(f"Audio features rate limited (403/429), retrying in {delay}s... (attempt {attempt + 1}/{max_retries})")
                        await asyncio.sleep(delay)
                        continue
                    else:
                        logger.error(f"Audio features still rate limited after {max_retries} retries, returning fallback")
                        return self._fallback_audio_features()

                resp.raise_for_status()
                features = resp.json()

                # Cache the result
                await self._cache_set(self._features_cache, track_id, features)
                return features

            except httpx.HTTPError:
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    await asyncio.sleep(delay)
                    continue
                else:
                    logger.error(f"HTTP error fetching audio features: {track_id}")
                    return self._fallback_audio_features()

        # If we exhausted retries
        return self._fallback_audio_features()

    async def get_tracks_features_bulk(self, track_ids: List[str]) -> Dict[str, Any]:
        """Get audio features for multiple tracks using batch API."""
        if self.demo_mode:
            return {track_id: self._fallback_audio_features() for track_id in track_ids}

        if not track_ids:
            return {}

        features_dict = {}

        # Check cache first
        uncached_ids = []
        for track_id in track_ids:
            cached = await self._cache_get(self._features_cache, track_id)
            if cached is not None:
                features_dict[track_id] = cached
            else:
                uncached_ids.append(track_id)

        if not uncached_ids:
            return features_dict

        # Batch fetch uncached features (API supports up to 100 IDs)
        # Add retry logic for rate limiting
        max_retries = 3
        base_delay = 1.0  # Start with 1s delay for bulk requests

        try:
            for i in range(0, len(uncached_ids), 100):
                batch = uncached_ids[i:i + 100]
                ids_param = ",".join(batch)

                # Retry loop for this batch
                for attempt in range(max_retries):
                    resp = await self._request("GET", "https://api.spotify.com/v1/audio-features",
                                              params={"ids": ids_param})

                    print(f"[DEBUG SPOTIFY] Audio features response status: {resp.status_code}", flush=True)

                    if resp.status_code == 200:
                        data = resp.json()
                        print(f"[DEBUG SPOTIFY] Got {len(data.get('audio_features', []))} audio features in response", flush=True)
                        for features in data.get("audio_features", []):
                            if features:  # API returns null for tracks without features
                                track_id = features.get("id")
                                features_dict[track_id] = features
                                await self._cache_set(self._features_cache, track_id, features)
                        break  # Success - move to next batch

                    elif resp.status_code in [403, 429]:
                        # Rate limited
                        if attempt < max_retries - 1:
                            delay = base_delay * (2 ** attempt)
                            logger.warning(f"🔄 Bulk audio features rate limited (403/429), waiting {delay}s before retry {attempt + 2}/{max_retries}...")
                            print(f"[DEBUG SPOTIFY] Rate limited, retrying in {delay}s...", flush=True)
                            await asyncio.sleep(delay)
                            continue
                        else:
                            logger.error(f"❌ Bulk audio features still rate limited after {max_retries} retries - this batch failed")
                            print(f"[DEBUG SPOTIFY] Max retries reached, batch failed", flush=True)
                            break
                    else:
                        print(f"[DEBUG SPOTIFY] Non-200 response: {resp.text[:200]}", flush=True)
                        break  # Other error - don't retry

        except httpx.HTTPError as exc:
            logger.warning("Batch features fetch failed (%s), falling back to individual fetches", exc)
            # Fallback to individual fetches for remaining uncached
            for track_id in uncached_ids:
                if track_id not in features_dict:
                    features = await self.get_track_features(track_id)
                    if features:
                        features_dict[track_id] = features

        return features_dict

    async def get_artists(self, artist_ids: Sequence[str]) -> Dict[str, Dict[str, Any]]:
        """Fetch artist metadata (genres, etc.) with caching."""
        if self.demo_mode or not artist_ids:
            return {}

        results: Dict[str, Dict[str, Any]] = {}
        unique: List[str] = []

        for artist_id in artist_ids:
            if not artist_id:
                continue
            cached = await self._cache_get(self._artist_cache, artist_id)
            if cached is not None:
                results[artist_id] = cached
            elif artist_id not in unique:
                unique.append(artist_id)

        if not unique:
            return results

        for idx in range(0, len(unique), 50):
            batch = unique[idx : idx + 50]
            params = {"ids": ",".join(batch)}

            try:
                resp = await self._request("GET", "https://api.spotify.com/v1/artists", params=params)
                resp.raise_for_status()
            except httpx.HTTPError as exc:
                logger.warning("Spotify artists lookup failed for batch starting %s (%s)", batch[0], exc)
                continue

            payload = resp.json()
            for item in payload.get("artists", []) or []:
                artist_id = item.get("id")
                if not artist_id:
                    continue
                results[artist_id] = item
                await self._cache_set(self._artist_cache, artist_id, item)

        return results

    async def get_playlist_tracks(
        self,
        playlist_id: str,
        *,
        limit: int = 100,
        max_tracks: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Return parsed track entries from a Spotify playlist."""
        if self.demo_mode:
            raise RuntimeError("SpotifyClient is in demo mode; playlist tracks unavailable.")

        assert self._client, "Call start() before using the client"

        collected: List[Dict[str, Any]] = []
        page_size = max(1, min(limit, 100))
        offset = 0

        while True:
            params = {
                "limit": page_size,
                "offset": offset,
                "market": "US",
            }

            resp = await self._request(
                "GET",
                f"https://api.spotify.com/v1/playlists/{playlist_id}/tracks",
                params=params,
            )

            if resp.status_code == 404:
                logger.warning("Playlist %s not found or inaccessible", playlist_id)
                break

            resp.raise_for_status()
            payload = resp.json()

            for item in payload.get("items", []) or []:
                track = item.get("track")
                if not track or track.get("is_local"):
                    continue
                parsed = self._parse_spotify_track(track)
                collected.append(parsed)

                if max_tracks and len(collected) >= max_tracks:
                    return collected[:max_tracks]

            next_url = payload.get("next")
            if not next_url:
                break

            offset += page_size

        return collected

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

    async def get_tracks_bulk(self, track_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        """Fetch full track metadata for a batch of Spotify track IDs."""
        if self.demo_mode or not track_ids:
            return {}

        # Preserve order but avoid duplicate network calls
        seen: set[str] = set()
        unique_ids = [tid for tid in track_ids if tid and not (tid in seen or seen.add(tid))]
        if not unique_ids:
            return {}

        results: Dict[str, Dict[str, Any]] = {}

        for idx in range(0, len(unique_ids), 50):
            chunk = unique_ids[idx:idx + 50]
            params = {"ids": ",".join(chunk)}

            try:
                resp = await self._request("GET", "https://api.spotify.com/v1/tracks", params=params)
                resp.raise_for_status()
            except httpx.HTTPError as exc:
                logger.warning("Spotify tracks lookup failed for chunk starting %s (%s)", chunk[0], exc)
                continue

            data = resp.json()
            for item in data.get("tracks", []) or []:
                if not item or not item.get("id"):
                    continue
                parsed = self._parse_spotify_track(item)
                results[item["id"]] = parsed

        return results

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

    @staticmethod
    def _extract_playlist_id(reference: str) -> Optional[str]:
        if not reference:
            return None

        ref = reference.strip()
        if not ref:
            return None

        if ref.startswith("spotify:playlist:"):
            return ref.split(":")[-1]

        if "open.spotify.com/playlist/" in ref:
            fragment = ref.split("/playlist/")[-1]
            fragment = fragment.split("?")[0].split("#")[0]
            return fragment or None

        ref = ref.split("?")[0].split("#")[0]
        if len(ref) == 22 and ref.replace("-", "").isalnum():
            return ref

        return None

    async def fetch_playlist_tracks_open(
        self,
        playlist_ref: str,
        *,
        limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Resolve a Spotify playlist URL/ID to track entries for client-side curation."""

        playlist_id = self._extract_playlist_id(playlist_ref)
        if not playlist_id:
            raise ValueError("Invalid Spotify playlist reference")

        html = await self._fetch_open_playlist_html(playlist_id)
        next_data = self._parse_next_data(html)
        playlist_payload = self._extract_playlist_payload(next_data)

        if not playlist_payload:
            raise ValueError("Failed to parse playlist data from Spotify share page")

        tracks = self._build_tracks_from_payload(playlist_payload, limit)
        metadata = self._build_playlist_metadata(playlist_payload, playlist_id, tracks)

        if not tracks:
            raise ValueError("Playlist appears to be empty or unsupported")

        return {"playlist": metadata, "tracks": tracks}

    async def _fetch_open_playlist_html(self, playlist_id: str) -> str:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://open.spotify.com/",
        }

        async with httpx.AsyncClient(timeout=httpx.Timeout(10.0, connect=5.0)) as client:
            resp = await client.get(
                f"https://open.spotify.com/embed/playlist/{playlist_id}",
                headers=headers,
            )

        if resp.status_code == 404:
            raise ValueError("Playlist not found or not publicly accessible")

        resp.raise_for_status()
        return resp.text

    def _parse_next_data(self, html: str) -> Dict[str, Any]:
        match = self._playlist_data_pattern.search(html)
        if not match:
            raise ValueError("Unable to locate Spotify playlist JSON payload")

        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError as exc:  # pragma: no cover - depends on external payload
            raise ValueError("Spotify playlist payload is malformed") from exc

    @staticmethod
    def _extract_playlist_payload(next_data: Dict[str, Any]) -> Dict[str, Any]:
        props = next_data.get("props") or {}
        page_props = props.get("pageProps") or {}

        state = page_props.get("state") or {}
        entity = state.get("data", {}).get("entity")
        if isinstance(entity, dict) and entity:
            return entity

        candidates = [
            page_props.get("playlist"),
            page_props.get("page"),
            page_props.get("data"),
            page_props.get("layout", {}).get("playlistPage", {}).get("data"),
        ]

        for candidate in candidates:
            if isinstance(candidate, dict) and candidate:
                return candidate

        return {}

    def _build_tracks_from_payload(self, payload: Dict[str, Any], limit: Optional[int]) -> List[Dict[str, Any]]:
        sections = [
            payload.get("tracks"),
            payload.get("trackList"),
            payload.get("items"),
        ]

        items = []
        for section in sections:
            if isinstance(section, dict) and isinstance(section.get("items"), list):
                items = section.get("items")
                break
            if isinstance(section, list):
                items = section
                break

        tracks: List[Dict[str, Any]] = []
        if not isinstance(items, list):
            return tracks

        for item in items:
            if limit is not None and len(tracks) >= limit:
                break

            parsed = self._parse_playlist_item(item)
            if parsed:
                tracks.append(parsed)

        return tracks

    def _parse_playlist_item(self, item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not isinstance(item, dict):
            return None

        container = item
        track_candidate = item.get("track")
        if isinstance(track_candidate, dict):
            track = track_candidate
        else:
            inner = item.get("item")
            if isinstance(inner, dict):
                nested_track = inner.get("track")
                if isinstance(nested_track, dict):
                    track = nested_track
                    container = inner
                elif inner.get("entityType") == "track":
                    track = inner
                    container = inner
                else:
                    track = None
            else:
                track = None

        if not isinstance(track, dict):
            if item.get("entityType") == "track" or item.get("uri"):
                track = item
                container = item
            else:
                return None

        uri_value = track.get("uri") or container.get("uri") or track.get("uid") or container.get("uid")
        track_id = None
        if isinstance(uri_value, str) and uri_value.startswith("spotify:track:"):
            track_id = uri_value.split(":")[-1]
        elif isinstance(track.get("id"), str):
            track_id = track["id"]

        if not track_id:
            return None

        name = track.get("name") or track.get("title") or container.get("title") or ""

        album_info = track.get("album") if isinstance(track.get("album"), dict) else {}
        album_name = album_info.get("name") or album_info.get("title") or ""

        cover_sources = album_info.get("coverArt", {}).get("sources") if isinstance(album_info, dict) else None
        image_url = None
        if isinstance(cover_sources, list) and cover_sources:
            image_url = cover_sources[0].get("url")
        else:
            images = album_info.get("images") if isinstance(album_info, dict) else None
            if isinstance(images, list) and images:
                image_url = images[0].get("url")

        duration_ms = track.get("duration_ms") or track.get("durationMs")
        if not duration_ms:
            duration_field = track.get("duration")
            if isinstance(duration_field, dict):
                duration_ms = duration_field.get("totalMilliseconds")
            elif isinstance(duration_field, (int, float)):
                duration_ms = duration_field
        if not duration_ms:
            duration_field = container.get("duration")
            if isinstance(duration_field, dict):
                duration_ms = duration_field.get("totalMilliseconds")
            elif isinstance(duration_field, (int, float)):
                duration_ms = duration_field
        if not duration_ms:
            duration_ms = 0

        release_date = ""
        date_info = album_info.get("date") if isinstance(album_info, dict) else None
        if isinstance(date_info, dict) and date_info.get("year"):
            year = date_info.get("year")
            month = date_info.get("month") or 1
            day = date_info.get("day") or 1
            try:
                release_date = f"{int(year):04d}-{int(month):02d}-{int(day):02d}"
            except (TypeError, ValueError):  # pragma: no cover - defensive
                release_date = str(year)

        artist_name = ""
        artist_ids: List[str] = []
        artists_field = track.get("artists")
        if isinstance(artists_field, dict):
            items = artists_field.get("items")
            if isinstance(items, list):
                for artist in items:
                    if not isinstance(artist, dict):
                        continue
                    artist_name = artist_name or artist.get("profile", {}).get("name") or artist.get("name")
                    uri = artist.get("uri")
                    if isinstance(uri, str) and uri.startswith("spotify:artist:"):
                        artist_ids.append(uri.split(":")[-1])
        elif isinstance(artists_field, list):
            for artist in artists_field:
                if not isinstance(artist, dict):
                    continue
                artist_name = artist_name or artist.get("name") or artist.get("profile", {}).get("name")
                uri = artist.get("uri")
                if isinstance(uri, str) and uri.startswith("spotify:artist:"):
                    artist_ids.append(uri.split(":")[-1])

        if not artist_name:
            artist_name = track.get("artist") or track.get("subtitle") or container.get("subtitle") or ""

        album_uri = album_info.get("uri") if isinstance(album_info, dict) else None
        album_id = None
        if isinstance(album_uri, str) and album_uri.startswith("spotify:album:"):
            album_id = album_uri.split(":")[-1]

        metadata: Dict[str, Any] = {}
        if album_id:
            metadata["spotify_album_id"] = album_id
        if artist_ids:
            metadata["spotify_artist_ids"] = artist_ids

        preview_url = track.get("preview_url") or track.get("previewUrl")
        if not preview_url and isinstance(track.get("audioPreview"), dict):
            preview_url = track["audioPreview"].get("url")

        external_urls = {"spotify": f"https://open.spotify.com/track/{track_id}"}

        return {
            "id": track_id,
            "provider": "spotify",
            "name": name,
            "artist": artist_name,
            "album": album_name,
            "duration_ms": int(duration_ms) if isinstance(duration_ms, (int, float)) else 0,
            "popularity": track.get("popularity", 0) or 0,
            "preview_url": preview_url or "",
            "external_urls": external_urls,
            "uri": f"spotify:track:{track_id}",
            "release_date": release_date,
            "image_url": image_url or "",
            "metadata": {k: v for k, v in metadata.items() if v},
        }

    @staticmethod
    def _build_playlist_metadata(
        playlist_payload: Dict[str, Any],
        playlist_id: str,
        tracks: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        owner = playlist_payload.get("owner") or {}
        if isinstance(owner, dict):
            owner_name = owner.get("displayName") or owner.get("name") or owner.get("username")
        else:
            owner_name = None

        if not owner_name:
            owner_name = playlist_payload.get("subtitle") or ""

        track_list = playlist_payload.get("trackList")
        track_list_count = None
        if isinstance(track_list, dict):
            track_list_count = track_list.get("totalCount")
        elif isinstance(track_list, list):
            track_list_count = len(track_list)

        total = (
            playlist_payload.get("tracks", {}).get("totalCount")
            or playlist_payload.get("totalTracks")
            or playlist_payload.get("tracks", {}).get("total")
            or track_list_count
            or len(tracks)
        )

        images = playlist_payload.get("images")
        cover_art = playlist_payload.get("coverArt", {}).get("sources")
        image_url = None
        if isinstance(cover_art, list) and cover_art:
            image_url = cover_art[0].get("url")
        elif isinstance(images, list) and images:
            image_url = images[0].get("url")

        followers = None
        followers_data = playlist_payload.get("followers")
        if isinstance(followers_data, dict):
            followers = followers_data.get("totalCount") or followers_data.get("total")

        return {
            "id": playlist_id,
            "name": playlist_payload.get("name") or playlist_payload.get("title") or "",
            "owner": owner_name or "",
            "total_tracks": int(total) if isinstance(total, (int, float)) else len(tracks),
            "image_url": image_url or "",
            "followers": int(followers) if isinstance(followers, (int, float)) else None,
        }





