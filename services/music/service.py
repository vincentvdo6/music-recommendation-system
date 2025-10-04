"""High level orchestration for music data providers."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple

from services.apple.client import AppleMusicClient
from services.spotify.client import SpotifyClient
from services.recommendation.catalogue import TrackCatalogue
from services.recommendation.contextual_engine import ContextualRecommendationEngine

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
        provider, track_id = self._parse_seed(seed)
        source = "contextual"
        recommendations: List[Dict[str, Any]] = []

        seed_features: Optional[Dict[str, Any]] = None
        if track_id and provider == "spotify" and self.spotify and not self.spotify.demo_mode:
            try:
                seed_features = await self.spotify.get_track_features(track_id)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Failed to fetch Spotify audio features for %s (%s)", track_id, exc)

        seed_metadata: Dict[str, Any] = {
            "id": track_id,
            "name": track_name,
            "artist": artist_name,
            "audio_features": seed_features,
        }

        if self.context_engine:
            try:
                recommendations = await self.context_engine.get_recommendations(
                    seed_track_id=track_id,
                    seed_metadata=seed_metadata,
                    seed_features=seed_features,
                    context=context,
                    limit=limit,
                )
            except Exception as exc:  # pragma: no cover - log and proceed with fallbacks
                logger.error("Contextual engine failed (%s); falling back to providers", exc)
                recommendations = []

        if recommendations:
            recommendations = await self._ensure_media_assets(recommendations)
            recommendations = self._post_process_recommendations(
                recommendations,
                seed_track_id=track_id,
                seed_name=track_name,
                seed_artist=artist_name,
                limit=limit,
            )
            return recommendations, source

        source = "spotify"
        if provider == "spotify" and self.spotify and not self.spotify.demo_mode:
            try:
                recommendations = await self.spotify.get_recommendations([track_id], limit=limit)
                recommendations = await self._ensure_media_assets(recommendations)
                recommendations = self._post_process_recommendations(
                    recommendations,
                    seed_track_id=track_id,
                    seed_name=track_name,
                    seed_artist=artist_name,
                    limit=limit,
                )
            except Exception as exc:  # pragma: no cover
                logger.warning("Spotify recommendations failed (%s); using Apple fallback", exc)
                recommendations = []
        if not recommendations:
            source = "apple"
            if provider == "spotify" and (track_name or artist_name):
                recommendations = await self.apple.get_related_tracks(
                    track_id=None,
                    track_name=track_name,
                    artist_name=artist_name,
                    limit=limit,
                )
            elif provider == "apple" and track_id:
                recommendations = await self.apple.get_related_tracks(
                    track_id=track_id,
                    track_name=track_name,
                    artist_name=artist_name,
                    limit=limit,
                    exclude_ids=[track_id],
                )

        if not recommendations and self.apple:
            fallback_query = artist_name or track_name
            if fallback_query:
                source = "apple"
                recommendations = await self.apple.search_tracks(fallback_query, limit=limit)
                if provider == "apple" and track_id:
                    recommendations = [t for t in recommendations if t.get("id") != track_id]
                recommendations = recommendations[:limit]

        if not recommendations and self.apple and (track_name or artist_name):
            source = "apple"
            genre_query = (
                f"{artist_name} similar artists" if artist_name else f"{track_name} similar songs"
            )
            recommendations = await self.apple.search_tracks(genre_query, limit=limit)

        recommendations = self._post_process_recommendations(
            recommendations,
            seed_track_id=track_id,
            seed_name=track_name,
            seed_artist=artist_name,
            limit=limit,
        )

        return recommendations, source

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

