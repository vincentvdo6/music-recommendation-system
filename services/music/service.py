"""High level orchestration for music data providers."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple

from services.apple.client import AppleMusicClient
from services.spotify.client import SpotifyClient
from services.recommendation.catalogue import TrackCatalogue
from services.recommendation.contextual_engine import ContextualRecommendationEngine
from services.recommendation.playlist_builder import build_entries

logger = logging.getLogger(__name__)


class MusicService:
    """Coordinates Spotify and Apple providers to keep media complete."""

    def __init__(
        self,
        *,
        spotify: Optional[SpotifyClient] = None,
        apple: Optional[AppleMusicClient] = None,
    ) -> None:
        self.spotify = spotify
        self.apple = apple or AppleMusicClient()
        self.catalogue: Optional[TrackCatalogue] = None
        self.context_engine: Optional[ContextualRecommendationEngine] = None

        try:
            self.catalogue = TrackCatalogue()
            self.context_engine = ContextualRecommendationEngine(self.catalogue)
            logger.info(
                "✓ Contextual recommendation engine initialized with %d curated tracks",
                self.catalogue.size(),
            )
        except (FileNotFoundError, ValueError) as exc:
            logger.error(
                "✗ Failed to initialise contextual engine (%s) — falling back to provider heuristics",
                exc,
            )

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
        include_features: bool = False,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any], str]:
        """Return tracks, optional feature map, and source identifier."""
        tracks: List[Dict[str, Any]] = []
        features: Dict[str, Any] = {}
        source = "apple"

        if self.spotify and not self.spotify.demo_mode:
            try:
                tracks = await self.spotify.search_tracks(query, limit=limit)
                source = "spotify"
                if include_features and tracks:
                    ids = [track["id"] for track in tracks]
                    features = await self.spotify.get_tracks_features_bulk(ids)
                tracks = await self._ensure_media_assets(tracks)
            except Exception as exc:  # pragma: no cover - defensive fallback
                logger.warning("Spotify search failed (%s), falling back to Apple", exc)
                tracks = []

        if not tracks:
            tracks = await self.apple.search_tracks(query, limit=limit)
            source = "apple"

        return tracks, features, source

    async def get_recommendations(
        self,
        seed: str,
        *,
        limit: int = 5,
        track_name: Optional[str] = None,
        artist_name: Optional[str] = None,
        user_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[Dict[str, Any]], str]:
        """Get recommendation list plus the provider source."""
        raise RuntimeError(
            "Playlist personalization required. Submit a playlist via the playlist recommendations endpoint to tailor results."
        )

    async def recommend_from_playlist(
        self,
        tracks_payload: List[Dict[str, Any]],
        *,
        seed: Optional[str] = None,
        limit: int = 5,
        context: Optional[Dict[str, Any]] = None,
        min_popularity: int = 0,
    ) -> Tuple[List[Dict[str, Any]], str, Dict[str, Any]]:
        """Generate recommendations from a user-supplied playlist."""

        if not tracks_payload:
            return [], "playlist", {"playlist_size": 0}

        if not self.spotify or self.spotify.demo_mode:
            raise RuntimeError("Spotify credentials required for playlist-based recommendations")

        normalised_inputs = self._normalise_playlist_inputs(tracks_payload)

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
            fetched = await self.spotify.get_tracks_bulk(spotify_ids)
            tracks_map.update(fetched)

        search_cache: Dict[Tuple[str, str], Optional[Dict[str, Any]]] = {}
        for descriptor in search_descriptors:
            name = descriptor.get("name", "").strip()
            artist = descriptor.get("artist")
            cache_key = (name.casefold(), (artist or "").casefold())

            track: Optional[Dict[str, Any]] = search_cache.get(cache_key)
            if track is None:
                query = f"{name} {artist}".strip()
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
                if track["id"] not in tracks_map:
                    tracks_map[track["id"]] = track

        if not tracks_map:
            raise RuntimeError("Could not resolve any tracks from supplied playlist data")

        track_ids = list(tracks_map.keys())
        features_map = await self.spotify.get_tracks_features_bulk(track_ids)

        artist_ids: List[str] = []
        for track in tracks_map.values():
            artist_ids.extend(track.get("metadata", {}).get("spotify_artist_ids", []))

        artist_map = await self.spotify.get_artists(artist_ids)

        entries = build_entries(
            tracks_map,
            features_map,
            artist_map,
            min_popularity=min_popularity,
        )

        # Fallback: enrich with curated catalogue matches when Spotify features are missing
        if self.context_engine and self.catalogue:
            seen_ids: set[str] = {entry["id"] for entry in entries}
            for item in normalised_inputs:
                candidate = None
                spotify_id = item.get("spotify_id")
                if spotify_id:
                    candidate = self.catalogue.get_by_id(spotify_id)
                if not candidate:
                    candidate = self.catalogue.find_best_match(item.get("name"), item.get("artist"))
                if candidate and candidate["id"] not in seen_ids:
                    entries.append(candidate)
                    seen_ids.add(candidate["id"])
                    tracks_map.setdefault(candidate["id"], candidate)

        if not entries:
            raise RuntimeError(
                "Playlist tracks were filtered out (missing features or below popularity threshold)"
            )

        if not self.context_engine:
            raise RuntimeError("Curated engine unavailable; cannot personalise recommendations")

        user_profile = self.context_engine.build_user_profile(entries)

        seed_id = self._extract_spotify_id(seed) if seed else None
        if not seed_id:
            for item in normalised_inputs:
                if item.get("seed") and item.get("spotify_id"):
                    seed_id = item["spotify_id"]
                    break

        seed_metadata = tracks_map.get(seed_id) if seed_id else None
        if not seed_metadata and tracks_map:
            seed_metadata = next(iter(tracks_map.values()))
            seed_id = seed_metadata.get("id")

        seed_features = features_map.get(seed_id) if seed_id else None

        recommendations = await self.context_engine.get_recommendations(
            seed_track_id=seed_id,
            seed_metadata=seed_metadata,
            seed_features=seed_features,
            context=context,
            user_profile=user_profile,
            limit=limit,
        )

        # Re-enrich recommendations with Spotify/Apple assets if available
        recommendations = await self._ensure_media_assets(recommendations)

        profile_context = user_profile.get("context", {})
        resolved_count = len(tracks_map) or len(entries)
        playlist_info = {
            "playlist_size": len(entries),
            "resolved_tracks": resolved_count,
            "top_moods": profile_context.get("moods", [])[:3],
            "top_activities": profile_context.get("activities", [])[:3],
            "top_genres": profile_context.get("genres", [])[:5],
            "regions": profile_context.get("regions", [])[:3],
            "time_of_day": profile_context.get("time_of_day"),
            "energy_level": profile_context.get("energy_level"),
            "tempo": profile_context.get("tempo"),
            "era": profile_context.get("era"),
        }

        return recommendations, "playlist", playlist_info

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

            entry = {
                "raw": raw or spotify_id,
                "spotify_id": spotify_id,
                "name": track.get("name"),
                "artist": artist or None,
                "album": track.get("album"),
                "image_url": track.get("image_url"),
                "uri": f"spotify:track:{spotify_id}",
                "url": (track.get("external_urls") or {}).get("spotify"),
                "seed": idx == 0,
            }
            tracks.append(entry)

        if not tracks:
            raise RuntimeError("Playlist does not contain any playable tracks")

        summary = result.get("playlist") or {}
        summary.setdefault("total_tracks", len(tracks))
        summary["loaded_tracks"] = len(tracks)

        return tracks, summary

    def track_user_interaction(
        self,
        user_id: str,
        track_id: str,
        interaction_type: str,
        track_data: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Interactions are logged for analytics; recommendations are context-driven only."""
        logger.debug(
            "User %s performed %s on %s", user_id, interaction_type, track_id
        )

    def save_recommendation_state(self) -> None:
        """Save recommendation engine state."""
        logger.debug("Contextual engine is stateless; nothing to persist")

    async def _ensure_media_assets(self, tracks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
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
            await asyncio.gather(*tasks)
        return tracks

    @staticmethod
    def _parse_seed(seed: str) -> Tuple[str, str]:
        if not seed:
            return "", ""

        parts = seed.split(":")
        if len(parts) >= 2:
            scheme = parts[0].lower()
            identifier = parts[-1]
            if scheme == "spotify":
                return "spotify", identifier
            if scheme in {"itunes", "apple"}:
                return "apple", identifier
        return "", seed

    @staticmethod
    def _post_process_recommendations(
        tracks: List[Dict[str, Any]],
        *,
        seed_track_id: Optional[str],
        seed_name: Optional[str],
        seed_artist: Optional[str],
        limit: int,
    ) -> List[Dict[str, Any]]:
        """Deduplicate results and remove the seed track itself."""
        if not tracks:
            return []

        normalized_seed_name = seed_name.lower().strip() if seed_name else None
        normalized_seed_artist = seed_artist.lower().strip() if seed_artist else None
        seed_track_id = seed_track_id or ""

        unique: List[Dict[str, Any]] = []
        seen_ids: set[str] = set()

        for track in tracks:
            track_id = str(track.get("id")) if track.get("id") is not None else ""
            if not track_id:
                continue

            if seed_track_id and track_id == seed_track_id:
                continue

            if track_id in seen_ids:
                continue

            name = track.get("name", "").lower().strip()
            artist = track.get("artist", "").lower().strip()

            if normalized_seed_name:
                same_name = name == normalized_seed_name
                same_artist = normalized_seed_artist and artist == normalized_seed_artist
                if same_name and (not normalized_seed_artist or same_artist):
                    # Skip exact match to the seed
                    continue

            unique.append(track)
            seen_ids.add(track_id)

            if len(unique) >= limit:
                break

        return unique

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

        separators = [
            " – ",
            " — ",
            " - ",
            " | ",
            " : ",
            " by ",
        ]

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
                descriptor = {k: v for k, v in item.items()}
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
