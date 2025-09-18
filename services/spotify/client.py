"""Spotify Web API client for real music data."""

import asyncio
import base64
import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta

import httpx

logger = logging.getLogger(__name__)


class SpotifyClient:
    """Client for Spotify Web API using Client Credentials flow."""
    
    def __init__(self, client_id: Optional[str] = None, client_secret: Optional[str] = None):
        # Use environment variables for credentials - NEVER hardcode secrets!
        import os
        self.client_id = client_id or os.getenv("SPOTIFY_CLIENT_ID")
        self.client_secret = client_secret or os.getenv("SPOTIFY_CLIENT_SECRET")
        self.access_token = None
        self.token_expires_at = None
        self.base_url = "https://api.spotify.com/v1"
        
        # Demo mode if no credentials provided
        self.demo_mode = not (self.client_id and self.client_secret)
        
        # Lazy-initialized persistent HTTP client
        self._client: Optional[httpx.AsyncClient] = None
        self._max_retries = 3
        
        # TTL caches for performance
        self._search_cache = TTLCache(ttl_seconds=60, maxsize=500)  # 1 min search cache
        self._features_cache = TTLCache(ttl_seconds=3600, maxsize=5000)  # 1 hour features cache
        
        # In-flight request deduplication
        self._inflight_search: Dict[Any, asyncio.Task] = {}
        self._inflight_features: Dict[Any, asyncio.Task] = {}
        
    def _ensure_client(self) -> None:
        """Ensure a single persistent AsyncClient exists."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(connect=2.0, read=5.0, write=5.0, pool=5.0),
                limits=httpx.Limits(max_keepalive_connections=20, max_connections=100),
            )

    async def _get_access_token(self) -> str:
        """Get access token using Client Credentials flow."""
        if self.access_token and self.token_expires_at and datetime.now() < self.token_expires_at:
            return self.access_token
            
        # Check if we have real credentials
        if self.demo_mode:
            logger.info("Using demo mode - no real Spotify API calls")
            return "demo_token"
        
        try:
            # Ensure HTTP client exists before token request
            self._ensure_client()
            # Encode credentials
            credentials = f"{self.client_id}:{self.client_secret}"
            encoded_credentials = base64.b64encode(credentials.encode()).decode()
            
            response = await self._client.post(
                "https://accounts.spotify.com/api/token",
                headers={
                    "Authorization": f"Basic {encoded_credentials}",
                    "Content-Type": "application/x-www-form-urlencoded"
                },
                data={"grant_type": "client_credentials"}
            )
            
            if response.status_code == 200:
                token_data = response.json()
                self.access_token = token_data["access_token"]
                expires_in = token_data.get("expires_in", 3600)
                # Refresh token 60 seconds before expiry
                self.token_expires_at = datetime.now() + timedelta(seconds=expires_in - 60)
                return self.access_token
            else:
                logger.error(f"Failed to get Spotify token: {response.status_code}")
                return "demo_token"
                    
        except Exception as e:
            logger.error(f"Error getting Spotify token: {e}")
            return "demo_token"
            
    async def close(self):
        """Close the HTTP client and clean up resources."""
        if self._client:
            await self._client.aclose()
    
    async def search_tracks(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Search for tracks on Spotify with caching and request coalescing."""
        if self.demo_mode:
            return self._mock_search_results(query, limit)

        # Create cache key
        cache_key = (query.strip().lower(), limit)
        
        # Check cache first
        cached_result = self._search_cache.get(cache_key)
        if cached_result is not None:
            return cached_result

        # Coalesce identical in-flight searches
        if cache_key in self._inflight_search:
            return await self._inflight_search[cache_key]

        async def _perform_search():
            try:
                token = await self._get_access_token()
                # Ensure client exists (lazy init)
                self._ensure_client()
                response = await self._client.get(
                    f"{self.base_url}/search",
                    headers={"Authorization": f"Bearer {token}"},
                    params={
                        "q": query,
                        "type": "track",
                        "limit": limit,
                        "market": "US"
                    }
                )
                
                if response.status_code == 200:
                    data = response.json()
                    tracks = [self._parse_spotify_track(item) 
                             for item in data.get("tracks", {}).get("items", [])]
                    
                    # Cache the result
                    self._search_cache.set(cache_key, tracks)
                    return tracks
                else:
                    logger.error(f"Spotify search failed: {response.status_code}")
                    return self._mock_search_results(query, limit)
                    
            except Exception as e:
                logger.error(f"Error searching Spotify: {e}")
                return self._mock_search_results(query, limit)

        # Create and track the search task
        task = asyncio.create_task(_perform_search())
        self._inflight_search[cache_key] = task
        
        try:
            return await task
        finally:
            # Clean up the inflight task
            self._inflight_search.pop(cache_key, None)
    
    async def get_track_features(self, track_id: str) -> Dict[str, Any]:
        """Get audio features for a track."""
        token = await self._get_access_token()
        
        if token == "demo_token":
            return self._mock_audio_features()
        
        try:
            self._ensure_client()
            response = await self._client.get(
                f"{self.base_url}/audio-features/{track_id}",
                headers={"Authorization": f"Bearer {token}"}
            )
                
            if response.status_code == 200:
                return response.json()
            else:
                return self._mock_audio_features()
                    
        except Exception as e:
            logger.error(f"Error getting track features: {e}")
            return self._mock_audio_features()
            
    async def get_tracks_features_bulk(self, track_ids: List[str]) -> Dict[str, Any]:
        """Get audio features for multiple tracks efficiently (max 100 at a time)."""
        if self.demo_mode:
            return {track_id: self._mock_audio_features() for track_id in track_ids}
        
        results: Dict[str, Any] = {}
        to_fetch: List[str] = []
        
        # Check cache first
        for track_id in track_ids:
            cached_features = self._features_cache.get(track_id)
            if cached_features is not None:
                results[track_id] = cached_features
            else:
                to_fetch.append(track_id)
        
        if not to_fetch:
            return results
        
        # Coalesce by sorted key to avoid duplicate requests
        cache_key = tuple(sorted(to_fetch))
        if cache_key in self._inflight_features:
            fetched_features = await self._inflight_features[cache_key]
        else:
            async def _fetch_features():
                try:
                    token = await self._get_access_token()
                    self._ensure_client()
                    fetched: Dict[str, Any] = {}
                    
                    # Process in chunks of 100 (Spotify API limit)
                    for i in range(0, len(to_fetch), 100):
                        chunk = to_fetch[i:i+100]
                        response = await self._client.get(
                            f"{self.base_url}/audio-features",
                            headers={"Authorization": f"Bearer {token}"},
                            params={"ids": ",".join(chunk)}
                        )
                        
                        if response.status_code == 200:
                            data = response.json() or {}
                            for features in (data.get("audio_features") or []):
                                if features and features.get("id"):
                                    fetched[features["id"]] = features
                        
                    return fetched
                except Exception as e:
                    logger.error(f"Error fetching bulk features: {e}")
                    return {}
            
            # Create and track the task
            task = asyncio.create_task(_fetch_features())
            self._inflight_features[cache_key] = task
            
            try:
                fetched_features = await task
            finally:
                self._inflight_features.pop(cache_key, None)
        
        # Cache the results and merge with existing
        for track_id, features in fetched_features.items():
            self._features_cache.set(track_id, features)
            results[track_id] = features
        
        return results
    
    async def get_recommendations(self, seed_tracks: List[str], limit: int = 20) -> List[Dict[str, Any]]:
        """Get track recommendations based on seed tracks."""
        token = await self._get_access_token()
        
        if token == "demo_token":
            return self._mock_recommendations(seed_tracks, limit)
        
        try:
            self._ensure_client()
            response = await self._client.get(
                f"{self.base_url}/recommendations",
                headers={"Authorization": f"Bearer {token}"},
                params={
                    "seed_tracks": ",".join(seed_tracks[:5]),  # Max 5 seeds
                    "limit": limit,
                    "market": "US"
                }
            )
                
            if response.status_code == 200:
                data = response.json()
                recommendations = []
                
                for item in data.get("tracks", []):
                    track = self._parse_spotify_track(item)
                    recommendations.append(track)
                
                return recommendations
            else:
                logger.error(f"Spotify recommendations failed: {response.status_code}")
                return self._mock_recommendations(seed_tracks, limit)
                    
        except Exception as e:
            logger.error(f"Error getting recommendations: {e}")
            return self._mock_recommendations(seed_tracks, limit)
    
    def _parse_spotify_track(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Parse Spotify track data into our format."""
        artists = [artist["name"] for artist in item.get("artists", [])]
        
        return {
            "id": item.get("id", ""),
            "name": item.get("name", "Unknown Track"),
            "artist": ", ".join(artists) if artists else "Unknown Artist",
            "album": item.get("album", {}).get("name", "Unknown Album"),
            "duration_ms": item.get("duration_ms", 0),
            "popularity": item.get("popularity", 0),
            "preview_url": item.get("preview_url"),
            "external_urls": item.get("external_urls", {}),
            "uri": item.get("uri", ""),
            "release_date": item.get("album", {}).get("release_date"),
            "image_url": self._get_image_url(item.get("album", {}))
        }
    
    def _get_image_url(self, album: Dict[str, Any]) -> Optional[str]:
        """Extract album image URL."""
        images = album.get("images", [])
        if images:
            # Sort by size and return the best quality image
            # Spotify usually provides [640x640, 300x300, 64x64]
            sorted_images = sorted(images, key=lambda x: x.get("height", 0), reverse=True)
            
            # Prefer medium-high quality (300x300 or higher)
            for image in sorted_images:
                if image.get("height", 0) >= 300:
                    return image.get("url")
            
            # Fallback to any available image
            return sorted_images[0].get("url")
        return None
    
    def _mock_search_results(self, query: str, limit: int) -> List[Dict[str, Any]]:
        """Generate mock search results for demo mode."""
        import random
        
        # Sample tracks that match common search terms
        mock_tracks = [
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
            track for track in mock_tracks 
            if query_lower in track["name"].lower() or query_lower in track["artist"].lower()
        ]
        
        if not matching_tracks:
            matching_tracks = random.sample(mock_tracks, min(len(mock_tracks), limit))
        
        results = []
        for i, track in enumerate(matching_tracks[:limit]):
            results.append({
                "id": f"mock_{i}_{hash(track['name']) % 10000}",
                "name": track["name"],
                "artist": track["artist"],
                "album": track["album"],
                "duration_ms": random.randint(180000, 300000),  # 3-5 minutes
                "popularity": track["popularity"],
                "preview_url": f"https://example.com/preview_{i}.mp3",
                "external_urls": {"spotify": f"https://open.spotify.com/track/mock_{i}"},
                "uri": f"spotify:track:mock_{i}",
                "release_date": f"{random.randint(1970, 2024)}-{random.randint(1, 12):02d}-{random.randint(1, 28):02d}",
                "image_url": "https://via.placeholder.com/300x300/1DB954/FFFFFF?text=♪"
            })
        
        return results
    
    def _mock_audio_features(self) -> Dict[str, Any]:
        """Generate mock audio features."""
        import random
        
        return {
            "acousticness": round(random.uniform(0.0, 1.0), 3),
            "danceability": round(random.uniform(0.0, 1.0), 3),
            "energy": round(random.uniform(0.0, 1.0), 3),
            "instrumentalness": round(random.uniform(0.0, 1.0), 3),
            "liveness": round(random.uniform(0.0, 1.0), 3),
            "loudness": round(random.uniform(-60.0, 0.0), 3),
            "speechiness": round(random.uniform(0.0, 1.0), 3),
            "tempo": round(random.uniform(60.0, 200.0), 1),
            "valence": round(random.uniform(0.0, 1.0), 3),
            "key": random.randint(0, 11),
            "mode": random.choice([0, 1]),
            "time_signature": random.choice([3, 4, 5])
        }
    
    def _mock_recommendations(self, seed_tracks: List[str], limit: int) -> List[Dict[str, Any]]:
        """Generate mock recommendations."""
        # Use different tracks than search results
        recommendations = [
            {"name": "As It Was", "artist": "Harry Styles", "album": "Harry's House", "popularity": 98},
            {"name": "Heat Waves", "artist": "Glass Animals", "album": "Dreamland", "popularity": 95},
            {"name": "Levitating", "artist": "Dua Lipa", "album": "Future Nostalgia", "popularity": 94},
            {"name": "Good 4 U", "artist": "Olivia Rodrigo", "album": "SOUR", "popularity": 97},
            {"name": "Stay", "artist": "The Kid LAROI, Justin Bieber", "album": "F*CK LOVE 3: OVER YOU", "popularity": 96},
            {"name": "Industry Baby", "artist": "Lil Nas X, Jack Harlow", "album": "MONTERO", "popularity": 93},
            {"name": "Bad Habits", "artist": "Ed Sheeran", "album": "=", "popularity": 92},
            {"name": "Peaches", "artist": "Justin Bieber", "album": "Justice", "popularity": 91},
            {"name": "Drivers License", "artist": "Olivia Rodrigo", "album": "SOUR", "popularity": 89},
            {"name": "Watermelon Sugar", "artist": "Harry Styles", "album": "Fine Line", "popularity": 88}
        ]
        
        import random
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
                "preview_url": f"https://example.com/preview_rec_{i}.mp3",
                "external_urls": {"spotify": f"https://open.spotify.com/track/rec_{i}"},
                "uri": f"spotify:track:rec_{i}",
                "release_date": f"{random.randint(2020, 2024)}-{random.randint(1, 12):02d}-{random.randint(1, 28):02d}",
                "image_url": "https://via.placeholder.com/300x300/1DB954/FFFFFF?text=♪"
            })
        
        return results


# Global Spotify client instance (initialized after env loading)
spotify_client = None

def get_spotify_client():
    """Get or create the global Spotify client instance."""
    global spotify_client
    if spotify_client is None:
        spotify_client = SpotifyClient()
    return spotify_client


class TTLCache:
    """Simple TTL cache with a max size. Not a strict LRU, but keeps recent inserts first.
    get(key) returns None on miss/expired. set(key, value) stores with expiry.
    """
    def __init__(self, ttl_seconds: int, maxsize: int = 1024):
        self.ttl = ttl_seconds
        self.maxsize = maxsize
        self._store: Dict[Any, Any] = {}
        self._timestamps: Dict[Any, float] = {}

    def get(self, key):
        import time as _t
        if key not in self._store:
            return None
        if (_t.time() - self._timestamps.get(key, 0)) > self.ttl:
            # Expired
            self._store.pop(key, None)
            self._timestamps.pop(key, None)
            return None
        return self._store[key]

    def set(self, key, value):
        import time as _t
        # Evict oldest if over capacity
        if len(self._store) >= self.maxsize:
            try:
                # Find oldest by timestamp
                oldest_key = min(self._timestamps, key=self._timestamps.get)
                self._store.pop(oldest_key, None)
                self._timestamps.pop(oldest_key, None)
            except ValueError:
                # Empty; ignore
                pass
        self._store[key] = value
        self._timestamps[key] = _t.time()
