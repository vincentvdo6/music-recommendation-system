"""Client for the Apple Music / iTunes Search API."""

import asyncio
import logging
from typing import Any, Dict, Iterable, List, Optional

import httpx

logger = logging.getLogger(__name__)


class AppleMusicClient:
    """Wraps the public iTunes Search API to fetch tracks and previews."""

    BASE_URL = "https://itunes.apple.com"

    def __init__(self, country: Optional[str] = None):
        self.country = (country or "US").upper()
        self._client: Optional[httpx.AsyncClient] = None
        self._sem = asyncio.Semaphore(6)

    async def start(self) -> None:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(10.0, connect=5.0),
                headers={"Accept": "application/json"},
            )

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    async def search_tracks(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search for tracks using a free Apple Music endpoint."""
        if not query or not query.strip():
            return []

        params = {
            "term": query,
            "entity": "song",
            "limit": min(max(limit, 1), 50),
            "country": self.country,
            "media": "music",
        }
        payload = await self._fetch_json("/search", params)
        return self._extract_tracks(payload, desired=limit)

    async def lookup_track(self, track_id: str) -> Optional[Dict[str, Any]]:
        """Lookup a specific track by its Apple identifier."""
        if not track_id:
            return None

        params = {"id": track_id, "entity": "song", "country": self.country}
        payload = await self._fetch_json("/lookup", params)
        tracks = self._extract_tracks(payload, desired=1)
        return tracks[0] if tracks else None

    async def get_related_tracks(
        self,
        *,
        track_id: Optional[str] = None,
        track_name: Optional[str] = None,
        artist_name: Optional[str] = None,
        limit: int = 5,
        exclude_ids: Optional[Iterable[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Return tracks that feel related based on artist/metadata."""
        exclude: set[str] = {str(i) for i in (exclude_ids or [])}
        candidates: List[Dict[str, Any]] = []

        track: Optional[Dict[str, Any]] = None
        if track_id:
            track = await self.lookup_track(track_id)

        if track is None and track_name:
            track = await self.find_best_match(track_name, artist_name)

        if track:
            exclude.add(track["id"])
            artist_id = track.get("metadata", {}).get("apple_artist_id")
            if artist_id:
                artist_tracks = await self._lookup_artist_tracks(int(artist_id), limit + 5)
                candidates.extend(artist_tracks)

        if not candidates and artist_name:
            artist_results = await self.search_tracks(artist_name, limit + 5)
            artist_lower = artist_name.lower()
            candidates.extend(t for t in artist_results if t.get("artist", "").lower() == artist_lower)

        if not candidates and track_name:
            query = f"{track_name} {artist_name or ''}".strip()
            candidates.extend(await self.search_tracks(query, limit + 5))

        unique: List[Dict[str, Any]] = []
        seen: set[str] = set()
        for candidate in candidates:
            cid = candidate["id"]
            if cid in exclude or cid in seen:
                continue
            seen.add(cid)
            unique.append(candidate)
            if len(unique) >= limit:
                break

        return unique

    async def find_best_match(self, track_name: str, artist_name: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Find the most likely Apple track for the given metadata."""
        query = f"{track_name} {artist_name or ''}".strip()
        results = await self.search_tracks(query, limit=5)
        if not results:
            return None

        if artist_name:
            target = artist_name.lower()
            for track in results:
                if track.get("artist", "").lower() == target:
                    return track

        return results[0]

    async def enrich_track(self, track: Dict[str, Any]) -> Dict[str, Any]:
        """Fill missing preview/image values for a track using Apple data."""
        if track.get("provider") == "apple":
            return track

        has_preview = bool(track.get("preview_url"))
        has_image = bool(track.get("image_url"))
        if has_preview and has_image:
            return track

        best = await self.find_best_match(track.get("name", ""), track.get("artist"))
        if not best:
            return track

        if not has_preview and best.get("preview_url"):
            track["preview_url"] = best["preview_url"]
        if not has_image and best.get("image_url"):
            track["image_url"] = best["image_url"]

        metadata = best.get("metadata")
        if metadata:
            track.setdefault("metadata", {}).update(metadata)

        return track

    async def _lookup_artist_tracks(self, artist_id: int, limit: int) -> List[Dict[str, Any]]:
        params = {
            "id": artist_id,
            "entity": "song",
            "limit": min(max(limit, 1), 50),
            "country": self.country,
        }
        payload = await self._fetch_json("/lookup", params)
        return self._extract_tracks(payload, desired=limit)

    async def _fetch_json(self, endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
        assert self._client, "Call start() before using AppleMusicClient"
        try:
            async with self._sem:
                response = await self._client.get(f"{self.BASE_URL}{endpoint}", params=params)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as exc:
            logger.warning("Apple Music call failed: %s", exc)
            return {}

    def _extract_tracks(self, payload: Dict[str, Any], desired: int) -> List[Dict[str, Any]]:
        items = payload.get("results", []) if isinstance(payload, dict) else []
        tracks: List[Dict[str, Any]] = []
        for idx, item in enumerate(items):
            parsed = self._parse_track(item, rank=idx)
            if parsed:
                tracks.append(parsed)
                if desired and len(tracks) >= desired:
                    break
        return tracks

    def _parse_track(self, item: Dict[str, Any], *, rank: int) -> Optional[Dict[str, Any]]:
        if item.get("wrapperType") != "track":
            return None

        track_id = item.get("trackId")
        if track_id is None:
            return None

        image_url = item.get("artworkUrl100") or item.get("artworkUrl60")
        if image_url:
            image_url = image_url.replace("100x100bb", "512x512bb").replace("100x100", "512x512")

        preview_url = item.get("previewUrl")
        metadata = {
            "apple_artist_id": item.get("artistId"),
            "primary_genre": item.get("primaryGenreName"),
            "collection_id": item.get("collectionId"),
        }

        popularity = self._score_from_rank(rank)

        return {
            "id": str(track_id),
            "provider": "apple",
            "name": item.get("trackName") or item.get("collectionName") or "Unknown",
            "artist": item.get("artistName", "Unknown"),
            "album": item.get("collectionName") or "",
            "duration_ms": int(item.get("trackTimeMillis") or 0),
            "popularity": popularity,
            "preview_url": preview_url,
            "external_urls": {
                "itunes": item.get("trackViewUrl"),
                "appleMusic": item.get("trackViewUrl"),
            },
            "uri": f"itunes:track:{track_id}",
            "release_date": item.get("releaseDate") or "",
            "image_url": image_url,
            "metadata": {k: v for k, v in metadata.items() if v is not None},
        }

    @staticmethod
    def _score_from_rank(rank: int) -> int:
        base = 95 - (rank * 5)
        return max(20, min(base, 100))
