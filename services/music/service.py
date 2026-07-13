"""High level orchestration for music data providers."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple

from services.apple.client import AppleMusicClient
from services.recommendation.factory import get_engine
from services.spotify.client import SpotifyClient

logger = logging.getLogger(__name__)

# Children's music and low-quality content excluded from recommendations.
ARTIST_BLACKLIST_KEYWORDS = [
    "kids", "children", "bambini", "enfants", "kinder", "niños",
    "baby", "toddler", "nursery", "lullaby", "kinderlieder",
    "canzoni per bambini", "musicclub", "evviva", "famiglia",
]

_UNSET = object()


class MusicService:
    """Coordinates the recommendation engine with Spotify/Apple metadata providers."""

    def __init__(
        self,
        *,
        spotify: Optional[SpotifyClient] = None,
        apple: Optional[AppleMusicClient] = None,
        engine: Any = _UNSET,
    ) -> None:
        self.spotify = spotify
        self.apple = apple or AppleMusicClient()
        self.engine = get_engine() if engine is _UNSET else engine
        # dz:<id> -> Spotify track dict (or None): extension-catalog tracks
        # resolved lazily at serving time; negative results cached too.
        self._ext_resolutions: Dict[str, Optional[Dict[str, Any]]] = {}

        if self.engine:
            logger.info("MusicService initialized with recommendation engine")
        else:
            logger.error("MusicService initialized WITHOUT a recommendation engine (models missing)")

    async def start(self) -> None:
        tasks = []
        if self.spotify:
            tasks.append(self.spotify.start())
        if self.apple:
            tasks.append(self.apple.start())
        if tasks:
            await asyncio.gather(*tasks)

    async def close(self) -> None:
        tasks = []
        if self.spotify:
            tasks.append(self.spotify.close())
        if self.apple:
            tasks.append(self.apple.close())
        if tasks:
            await asyncio.gather(*tasks)

    async def search_tracks(
        self,
        query: str,
        *,
        limit: int = 10,
    ) -> Tuple[List[Dict[str, Any]], str]:
        """Return tracks and the source provider."""
        tracks: List[Dict[str, Any]] = []
        source = "apple"

        if self.spotify and not self.spotify.demo_mode:
            try:
                tracks = await self.spotify.search_tracks(query, limit=limit)
                source = "spotify"
                tracks = await self._ensure_media_assets(tracks)
            except Exception as exc:  # pragma: no cover - defensive fallback
                logger.warning("Spotify search failed (%s), falling back to Apple", exc)
                tracks = []

        if not tracks:
            tracks = await self.apple.search_tracks(query, limit=limit)
            source = "apple"

        return tracks, source

    async def recommend_from_playlist(
        self,
        tracks_payload: List[Dict[str, Any]],
        *,
        seed: Optional[str] = None,
        limit: int = 5,
    ) -> Tuple[List[Dict[str, Any]], str, Dict[str, Any]]:
        """
        Generate recommendations driven by the searched song ("seed").

        The seed is the primary driver: most of the candidate pool comes from
        its neighbors and ranking is weighted toward it. The playlist is
        optional flavor — seed-only requests are fully supported.
        """
        if not tracks_payload and not seed:
            return [], "playlist", {"playlist_size": 0, "resolved_tracks": 0}

        if not self.spotify or self.spotify.demo_mode:
            raise RuntimeError("Spotify credentials required for recommendations")

        if not self.engine:
            raise RuntimeError("Recommendation engine unavailable — model artifacts are missing")

        normalised_inputs = self._normalise_playlist_inputs(tracks_payload)
        tracks_map = await self._resolve_playlist_tracks(normalised_inputs) if normalised_inputs else {}
        if tracks_payload and not tracks_map:
            raise RuntimeError("Could not resolve any tracks from supplied playlist data")

        playlist_track_ids = list(tracks_map.keys())
        input_track_id = self._resolve_input_track_id(seed, normalised_inputs, tracks_map)
        if not input_track_id:
            raise RuntimeError("Provide a seed track or a playlist with resolvable tracks")

        # The engine needs the seed's artist name for proxy lookup when the
        # track itself isn't in the model vocabulary.
        input_metadata = tracks_map.get(input_track_id)
        if not input_metadata:
            fetched = await self.spotify.get_tracks_bulk([input_track_id])
            input_metadata = fetched.get(input_track_id)

        recommendations = self.engine.recommend(
            playlist_tracks=playlist_track_ids,
            input_track_id=input_track_id,
            input_artist_name=(input_metadata or {}).get("artist"),
            input_track_name=(input_metadata or {}).get("name"),
            limit=limit * 3,  # headroom for enrichment losses + artist dedup
        )
        source = "ml-playlist" if playlist_track_ids else "ml-seed"

        if not recommendations:
            logger.warning("Engine returned no recommendations — using popularity fallback")
            recommendations = self.engine.popular_fallback(limit * 3)
            source = "popular-fallback"

        recommendations = await self._enrich_with_spotify_metadata(recommendations)
        recommendations = await self._ensure_media_assets(recommendations)
        recommendations = self._deduplicate_by_artist(recommendations, max_per_artist=1, limit=limit)

        if len(recommendations) < limit:
            # Spotify drops and artist dedup can leave us short — refill from
            # the popularity fallback AFTER enrichment, not only before it.
            n_engine = len(recommendations)
            exclude = {r["id"] for r in recommendations} | set(playlist_track_ids) | {input_track_id}
            extras = [t for t in self.engine.popular_fallback(limit * 3) if t["id"] not in exclude]
            if extras:
                extras = await self._enrich_with_spotify_metadata(extras)
                extras = await self._ensure_media_assets(extras)
                merged = self._deduplicate_by_artist(recommendations + extras, max_per_artist=1, limit=limit)
                if len(merged) > len(recommendations):
                    logger.info("Refilled %d slots from popularity fallback", len(merged) - len(recommendations))
                    recommendations = merged
            if n_engine == 0 and recommendations:
                source = "popular-fallback"  # nothing the engine picked survived

        playlist_info: Dict[str, Any] = {
            "playlist_size": len(tracks_map),
            "resolved_tracks": len(tracks_map),
        }
        playlist_info.update(self.engine.coverage(playlist_track_ids, input_track_id))

        return recommendations, source, playlist_info

    async def import_playlist_from_url(
        self,
        playlist_ref: str,
        *,
        limit: int = 300,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Resolve a Spotify playlist URL/ID into playlist entries for the UI."""

        if not self.spotify:
            raise RuntimeError("Spotify client not configured for playlist import")

        try:
            result = await self.spotify.fetch_playlist_tracks_open(playlist_ref, limit=limit)
        except ValueError as exc:
            raise RuntimeError(str(exc)) from exc

        tracks: List[Dict[str, Any]] = []
        for idx, track in enumerate(result.get("tracks", [])):
            spotify_id = track.get("id")
            if not spotify_id:
                continue

            raw = track.get("name") or ""
            artist = track.get("artist") or ""
            if raw and artist:
                raw = f"{raw} - {artist}"

            tracks.append({
                "raw": raw or spotify_id,
                "spotify_id": spotify_id,
                "name": track.get("name"),
                "artist": artist or None,
                "album": track.get("album"),
                "image_url": track.get("image_url"),
                "uri": f"spotify:track:{spotify_id}",
                "url": (track.get("external_urls") or {}).get("spotify"),
                "seed": idx == 0,
            })

        if not tracks:
            raise RuntimeError("Playlist does not contain any playable tracks")

        summary = result.get("playlist") or {}
        summary.setdefault("total_tracks", len(tracks))
        summary["loaded_tracks"] = len(tracks)

        return tracks, summary

    # ------------------------------------------------------------ internals

    async def _resolve_playlist_tracks(
        self, normalised_inputs: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """Resolve playlist entries to Spotify tracks (ids directly, names via search)."""
        spotify_ids: List[str] = []
        search_descriptors: List[Dict[str, Any]] = []

        for item in normalised_inputs:
            spotify_id = item.get("spotify_id")
            if spotify_id:
                if spotify_id not in spotify_ids:
                    spotify_ids.append(spotify_id)
                continue
            if item.get("name"):
                search_descriptors.append(item)

        tracks_map: Dict[str, Dict[str, Any]] = {}
        if spotify_ids:
            tracks_map.update(await self.spotify.get_tracks_bulk(spotify_ids))

        search_cache: Dict[Tuple[str, str], Optional[Dict[str, Any]]] = {}
        for descriptor in search_descriptors:
            name = descriptor.get("name", "").strip()
            artist = descriptor.get("artist")
            cache_key = (name.casefold(), (artist or "").casefold())

            track = search_cache.get(cache_key)
            if track is None and cache_key not in search_cache:
                query = f"{name} {artist or ''}".strip()
                results = await self.spotify.search_tracks(query, limit=5)
                track = None
                if artist:
                    artist_norm = artist.casefold()
                    for candidate in results:
                        if candidate.get("artist", "").casefold() == artist_norm:
                            track = candidate
                            break
                if not track and results:
                    track = results[0]
                search_cache[cache_key] = track

            if track:
                descriptor["spotify_id"] = track["id"]
                tracks_map.setdefault(track["id"], track)

        return tracks_map

    def _deduplicate_by_artist(
        self,
        tracks: List[Dict[str, Any]],
        max_per_artist: int = 1,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """Artist diversity + content blacklist, applied after enrichment."""
        artist_counts: Dict[str, int] = {}
        diverse_tracks: List[Dict[str, Any]] = []

        for track in tracks:
            artist_lower = (track.get("artist") or "Unknown").casefold()
            track_name = (track.get("name") or "").casefold()

            if any(
                keyword in value
                for value in (artist_lower, track_name)
                for keyword in ARTIST_BLACKLIST_KEYWORDS
            ):
                continue

            count = artist_counts.get(artist_lower, 0)
            if count < max_per_artist:
                diverse_tracks.append(track)
                artist_counts[artist_lower] = count + 1

            if len(diverse_tracks) >= limit:
                break

        return diverse_tracks

    @staticmethod
    def _match_norm(s: str) -> str:
        return "".join(ch for ch in (s or "").casefold() if ch.isalnum() or ch == " ").strip()

    async def _resolve_extension_track(self, track: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extension-catalog tracks carry Deezer-derived ids (dz:...) — find
        their Spotify identity by search, at most once per process. Only
        served tracks pay this cost (a handful per request, cached)."""
        key = track["id"]
        if key in self._ext_resolutions:
            return self._ext_resolutions[key]
        real = None
        name, artist = track.get("name") or "", track.get("artist") or ""
        if name and artist:
            try:
                candidates = await self.spotify.search_tracks(f"{name} {artist}", limit=5)
            except Exception as exc:
                logger.warning("Extension resolution search failed for %s: %s", key, exc)
                candidates = []
            want_artist, want_title = self._match_norm(artist), self._match_norm(name)
            for cand in candidates:
                got_artist = self._match_norm(cand.get("artist") or "")
                got_title = self._match_norm(cand.get("name") or "")
                if want_artist not in (got_artist, "") and got_artist not in (want_artist, ""):
                    continue
                if want_title not in got_title and got_title not in want_title:
                    continue
                dur, cand_dur = track.get("duration_ms") or 0, cand.get("duration_ms") or 0
                if dur and cand_dur and abs(dur - cand_dur) > 10_000:
                    continue
                real = cand
                break
        self._ext_resolutions[key] = real
        if real is None:
            logger.info("Extension track %s (%s — %s) has no Spotify identity", key, name, artist)
        return real

    async def _enrich_with_spotify_metadata(self, tracks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Attach live Spotify metadata (album, artwork, popularity, links) to
        engine results. Tracks Spotify no longer knows are dropped — they
        cannot be rendered or played.
        """
        if not tracks or not self.spotify:
            return tracks

        # Extension-catalog ids are not Spotify ids — one in a /tracks batch
        # would invalidate the whole chunk. They resolve by search instead.
        ids = [t["id"] for t in tracks if t.get("id") and not t["id"].startswith("dz:")]
        spotify_tracks = await self.spotify.get_tracks_bulk(ids)
        for track in tracks:
            if (track.get("id") or "").startswith("dz:"):
                real = await self._resolve_extension_track(track)
                if real:
                    spotify_tracks[track["id"]] = real

        enriched: List[Dict[str, Any]] = []
        for track in tracks:
            real = spotify_tracks.get(track.get("id"))
            if not real:
                continue
            merged = dict(real)
            for key in ("recommendation", "rank_score", "similarity_score", "why"):
                if key in track:
                    merged[key] = track[key]
            enriched.append(merged)

        if len(enriched) < len(tracks):
            logger.info("Dropped %d recommendations unknown to Spotify", len(tracks) - len(enriched))
        return enriched

    async def _ensure_media_assets(self, tracks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Fill missing preview/artwork from Apple Music."""
        if not tracks or not self.apple:
            return tracks

        tasks = []
        for track in tracks:
            if track.get("provider") == "apple":
                continue
            if track.get("preview_url") and track.get("image_url"):
                continue
            tasks.append(self.apple.enrich_track(track))

        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, Exception):
                    logger.warning("Apple enrichment task failed: %s", result)
        return tracks

    def _resolve_input_track_id(
        self,
        input_track_uri: Optional[str],
        normalised_inputs: List[Dict[str, Any]],
        tracks_map: Dict[str, Dict[str, Any]],
    ) -> Optional[str]:
        """
        Resolve the INPUT TRACK ID (the searched song driving recommendations).

        Priority: explicit seed URI from the request > playlist entry marked
        seed=True > first playlist track.
        """
        input_id = self._extract_spotify_id(input_track_uri) if input_track_uri else None
        if input_id:
            return input_id

        for item in normalised_inputs:
            if item.get("seed") and item.get("spotify_id"):
                return item["spotify_id"]

        if tracks_map:
            fallback = next(iter(tracks_map.keys()))
            logger.warning("No input track specified — falling back to first playlist track %s", fallback)
            return fallback

        return None

    @staticmethod
    def _extract_spotify_id(value: Optional[str]) -> Optional[str]:
        if not value:
            return None

        candidate = value.strip()
        if not candidate:
            return None

        if candidate.startswith("spotify:track:"):
            return candidate.split(":")[-1]

        if "open.spotify.com/track/" in candidate:
            fragment = candidate.split("/track/")[-1]
            fragment = fragment.split("?")[0].split("#")[0]
            return fragment or None

        candidate = candidate.split("?")[0].split("#")[0]
        if len(candidate) == 22 and candidate.replace("-", "").isalnum():
            return candidate

        return None

    @staticmethod
    def _split_track_line(raw: str) -> Tuple[Optional[str], Optional[str]]:
        text = raw.strip()
        if not text:
            return None, None

        separators = [" – ", " — ", " - ", " | ", " : ", " by "]
        for sep in separators:
            if sep in text:
                left, right = text.split(sep, 1)
                return left.strip() or None, right.strip() or None

        if "-" in text:
            left, right = text.split("-", 1)
            return left.strip() or None, right.strip() or None

        return text, None

    def _normalise_playlist_inputs(self, tracks_payload: List[Any]) -> List[Dict[str, Any]]:
        normalised: List[Dict[str, Any]] = []

        for item in tracks_payload:
            if isinstance(item, str):
                descriptor: Dict[str, Any] = {"raw": item}
            elif isinstance(item, dict):
                descriptor = dict(item)
            else:
                continue

            raw_value = descriptor.get("raw")
            spotify_id = (
                descriptor.get("spotify_id")
                or descriptor.get("id")
                or self._extract_spotify_id(descriptor.get("uri"))
                or self._extract_spotify_id(descriptor.get("url"))
            )

            if not spotify_id and isinstance(raw_value, str):
                spotify_id = self._extract_spotify_id(raw_value)

            if spotify_id:
                descriptor["spotify_id"] = spotify_id

            name = descriptor.get("name")
            artist = descriptor.get("artist")

            if isinstance(raw_value, str):
                guessed_name, guessed_artist = self._split_track_line(raw_value)
                name = name or guessed_name
                artist = artist or guessed_artist

            descriptor["name"] = name
            descriptor["artist"] = artist
            descriptor["seed"] = bool(descriptor.get("seed"))

            normalised.append(descriptor)

        return normalised
