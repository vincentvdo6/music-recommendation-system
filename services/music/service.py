"""High level orchestration for music data providers."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple

from services.apple.client import AppleMusicClient
from services.spotify.client import SpotifyClient
from services.recommendation.hybrid_engine import HybridRecommendationEngine

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
        self.hybrid_engine = HybridRecommendationEngine(spotify) if spotify else None

        if self.hybrid_engine:
            logger.info("✓ Hybrid recommendation engine initialized")
        else:
            logger.warning("✗ Hybrid engine disabled (no Spotify client)")

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
        source = "apple"
        recommendations: List[Dict[str, Any]] = []

        # Try hybrid personalized recommendations if user_id provided
        if user_id and self.hybrid_engine:
            logger.info("Using hybrid engine for user %s (seed: %s)", user_id[:12], seed[:30])
            try:
                # Get seed track features if available
                seed_features = None
                if track_id and self.spotify and not self.spotify.demo_mode:
                    seed_features = await self.spotify.get_track_features(track_id)
                    logger.info("Fetched seed track features for hybrid engine")

                recommendations = await self.hybrid_engine.get_personalized_recommendations(
                    user_id=user_id,
                    seed_track_id=track_id if provider == "spotify" else None,
                    seed_features=seed_features,
                    limit=limit,
                    context=context,
                )
                recommendations = await self._ensure_media_assets(recommendations)
                source = "hybrid"
                logger.info("✓ Hybrid engine returned %d recommendations", len(recommendations))
            except Exception as exc:
                logger.warning("Hybrid recommendations failed (%s), falling back", exc)
                recommendations = []

        # Fallback to basic Spotify recommendations
        if not recommendations and provider == "spotify" and self.spotify and not self.spotify.demo_mode:
            try:
                recommendations = await self.spotify.get_recommendations([track_id], limit=limit)
                recommendations = await self._ensure_media_assets(recommendations)
                source = "spotify"
            except Exception as exc:  # pragma: no cover
                logger.warning("Spotify recommendations failed (%s); using Apple fallback", exc)
                recommendations = []

        # Fallback to Apple Music
        if not recommendations:
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
                )
            source = "apple"

        if not recommendations and self.apple:
            fallback_query = artist_name or track_name
            if fallback_query:
                recommendations = await self.apple.search_tracks(fallback_query, limit=limit)
                if provider == "apple" and track_id:
                    recommendations = [t for t in recommendations if t.get("id") != track_id]
                recommendations = recommendations[:limit]
                source = "apple"

        if not recommendations and self.apple and (track_name or artist_name):
            genre_query = f"{artist_name} similar artists" if artist_name else f"{track_name} similar songs"
            recommendations = await self.apple.search_tracks(genre_query, limit=limit)
            source = "apple"

        return recommendations, source

    def track_user_interaction(
        self,
        user_id: str,
        track_id: str,
        interaction_type: str,
        track_data: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Track user interactions for personalization."""
        if self.hybrid_engine:
            self.hybrid_engine.track_interaction(user_id, track_id, interaction_type, track_data)

    def save_recommendation_state(self) -> None:
        """Save recommendation engine state."""
        if self.hybrid_engine:
            self.hybrid_engine.save_state()

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

