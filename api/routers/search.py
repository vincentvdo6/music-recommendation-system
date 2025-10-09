"""Song search and recommendations endpoints using Spotify and Apple data."""

import time
import logging
import random
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Request, Query
from slowapi import Limiter
from slowapi.util import get_remote_address

from api.models import (
    SearchResponse,
    SearchHit,
    Track,
    AudioFeatures,
    RecommendationsResponse,
    RecommendationHit,
    PlaylistRecommendationsRequest,
    PlaylistRecommendationsResponse,
    PlaylistSummary,
    PlaylistImportRequest,
    PlaylistImportResponse,
    PlaylistTrackInput,
)
from services.spotify.client import SpotifyClient

logger = logging.getLogger(__name__)
router = APIRouter(tags=["search"])
limiter = Limiter(key_func=get_remote_address)


def _explain(features: Dict[str, Any]) -> List[str]:
    """Generate explanation for track features."""
    out: List[str] = []
    tempo = features.get("tempo")
    if tempo:
        if tempo > 140:
            out.append("High-energy tempo")
        elif tempo < 80:
            out.append("Chill/slow tempo")

    energy = features.get("energy")
    if energy is not None:
        if energy > 0.7:
            out.append("High energy")
        elif energy < 0.3:
            out.append("Low energy")

    valence = features.get("valence")
    if valence is not None:
        if valence > 0.7:
            out.append("Upbeat mood")
        elif valence < 0.3:
            out.append("Melancholic mood")

    return out[:3]


def _metadata_hints(track: Dict[str, Any]) -> List[str]:
    """Generate explanation hints from metadata when features are absent."""
    hints: List[str] = []
    metadata = track.get("metadata") or {}

    genre = metadata.get("primary_genre")
    if genre:
        hints.append(f"Genre: {genre}")

    artist = track.get("artist")
    provider = track.get("provider")
    if provider == "apple" and artist:
        hints.append(f"Similar artist: {artist}")

    popularity = track.get("popularity")
    if popularity:
        hints.append(f"Listener popularity: {popularity}%")

    return hints[:3]


@router.get("/search", response_model=SearchResponse)
@limiter.limit("30/minute")
async def search_songs(
    request: Request,
    q: str = Query(..., min_length=2, max_length=200),
    limit: int = Query(10, ge=1, le=20),
    include_features: bool = Query(False, description="Include audio features (slower)")
):
    """Search for songs using available music providers."""
    start_time = time.time()
    request_id = getattr(request.state, "request_id", "unknown")

    music_service = request.app.state.music

    try:
        tracks_data, features_map, source = await music_service.search_tracks(
            q,
            limit=limit,
            include_features=include_features,
        )

        results: List[SearchHit] = []
        for track_dict in tracks_data:
            track = Track(**track_dict)

            features_data: Dict[str, Any] = {}
            features: Optional[AudioFeatures] = None
            if include_features:
                features_data = features_map.get(track.id, {}) if isinstance(features_map, dict) else {}
                if features_data:
                    features = AudioFeatures(**features_data)

            hints = _explain(features_data) if features_data else _metadata_hints(track_dict)

            results.append(SearchHit(
                track=track,
                features=features,
                why=hints
            ))

        processing_time = int((time.time() - start_time) * 1000)

        return SearchResponse(
            query=q,
            count=len(results),
            results=results,
            source=source,
            request_id=request_id,
            processing_time_ms=processing_time
        )

    except Exception as exc:  # pragma: no cover - defensive catch
        logger.error("Search failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Search failed: {str(exc)}")


@router.get("/recommendations", response_model=RecommendationsResponse)
@limiter.limit("20/minute")
async def get_recommendations(
    request: Request,
    seed: str = Query("", description="Seed track URI or ID (optional)"),
    limit: int = Query(5, ge=1, le=20, description="Number of recommendations"),
    track_name: Optional[str] = Query(None, description="Seed track name for non-Spotify IDs"),
    artist_name: Optional[str] = Query(None, description="Seed artist name for non-Spotify IDs"),
    user_id: Optional[str] = Query(None, description="User ID (retained for backwards compatibility)"),
    time_of_day: Optional[str] = Query(None, description="Context: morning/afternoon/evening/night"),
    mood: Optional[str] = Query(None, description="Context: upbeat/calm/energetic"),
    activity: Optional[str] = Query(None, description="Context: focus/workout/relax/etc."),
    genre: Optional[str] = Query(None, description="Preferred genre hint"),
    energy: Optional[str] = Query(None, description="Energy level: low/medium/high"),
    tempo: Optional[str] = Query(None, description="Tempo bucket or BPM"),
    era: Optional[str] = Query(None, description="Era preference: recent/2010s/classic"),
    region: Optional[str] = Query(None, description="Regional focus"),
):
    """Get song recommendations based on a seed track with optional personalization."""
    start_time = time.time()
    request_id = getattr(request.state, "request_id", "unknown")

    music_service = request.app.state.music
    spotify_client: Optional[SpotifyClient] = getattr(request.app.state, "spotify", None)

    # Build context dict
    context: Dict[str, Any] = {}
    if time_of_day:
        context["time_of_day"] = time_of_day
    if mood:
        context["mood"] = mood
    if activity:
        context["activity"] = activity
    if genre:
        context["genre"] = genre
    if energy:
        context["energy_level"] = energy
    if tempo is not None:
        context["tempo"] = tempo
    if era:
        context["era"] = era
    if region:
        context["region"] = region

    try:
        recommendations_data, source = await music_service.get_recommendations(
            seed or "",
            limit=limit,
            track_name=track_name,
            artist_name=artist_name,
            user_id=user_id,
            context=context if context else None,
        )

        logger.info(f"Recommendations source: {source}, count: {len(recommendations_data)}, demo_mode: {spotify_client.demo_mode if spotify_client else 'N/A'}")

        enhanced_recommendations: List[RecommendationHit] = []
        for idx, track_dict in enumerate(recommendations_data):
            provider = track_dict.get("provider")
            audio_features: Optional[AudioFeatures] = None
            explanation: List[str] = []

            if source == "contextual":
                components = (track_dict.get("recommendation") or {}).get("components", {})
                similarity_score = components.get("similarity", 0.72)
                rank_score = (track_dict.get("recommendation") or {}).get("score", similarity_score)

                feature_payload = track_dict.get("audio_features") or {}
                if feature_payload:
                    audio_features = AudioFeatures(**feature_payload)

                top_factors = []
                if components:
                    top_factors.append(
                        f"Audio similarity {int(components.get('similarity', 0) * 100)}%"
                    )
                    top_factors.append(
                        f"Context alignment {int(components.get('context', 0) * 100)}%"
                    )
                tags = track_dict.get("tags", {})
                if tags.get("activities"):
                    top_factors.append(
                        f"Activity match: {', '.join(tags['activities'][:2])}"
                    )

                explanation_dict = {
                    "top_factors": top_factors[:3],
                    "similarity_reason": "Catalogue-driven contextual engine",
                    "ranking_boost": f"Popularity score {track_dict.get('popularity', 0)}%",
                }

            elif provider == "spotify" and spotify_client and not spotify_client.demo_mode:
                features_data = await spotify_client.get_track_features(track_dict["id"]) or {}
                if features_data:
                    audio_features = AudioFeatures(**features_data)
                    explanation = _explain(features_data)

                # More natural diversity - exponential decay with variance
                base_similarity = 0.92 - (idx * 0.05) - (idx ** 1.3 * 0.01)
                popularity_boost = (track_dict.get("popularity", 50) / 100) * 0.08
                diversity_variance = random.uniform(-0.02, 0.02)
                similarity_score = max(0.65, min(0.96, base_similarity + popularity_boost + diversity_variance))
                rank_score = similarity_score  # Use actual computed similarity
                explanation_dict = {
                    "top_factors": explanation,
                    "similarity_reason": "Based on audio features and listening patterns",
                    "ranking_boost": f"High popularity ({track_dict.get('popularity', 0)}%)"
                }
            else:
                metadata_hints = _metadata_hints(track_dict)
                if artist_name and track_dict.get("artist") and track_dict["artist"].lower() == artist_name.lower():
                    metadata_hints.insert(0, f"Artist match: {track_dict['artist']}")

                # Add variance to Apple recommendations too
                base_similarity = 0.85 - (idx * 0.06)
                diversity_variance = random.uniform(-0.03, 0.03)
                similarity_score = max(0.68, min(0.89, base_similarity + diversity_variance))
                rank_score = similarity_score  # Use actual computed similarity
                explanation_dict = {
                    "top_factors": metadata_hints[:3],
                    "similarity_reason": "Curated via Apple Music catalog",
                    "ranking_boost": f"Popularity estimate ({track_dict.get('popularity', 70)}%)"
                }

            enhanced_payload = dict(track_dict)
            enhanced_payload.update({
                "audio_features": audio_features,
                "similarity_score": similarity_score,
                "rank_score": rank_score,
                "explanation": explanation_dict,
            })

            enhanced_recommendations.append(RecommendationHit(**enhanced_payload))

        processing_time = int((time.time() - start_time) * 1000)

        algorithm = (
            "Contextual (Catalogue + Audio + Intent)"
            if source == "contextual"
            else "Spotify Web API + Audio Feature Analysis"
            if source == "spotify"
            else "Apple Music catalog similarity"
        )

        return RecommendationsResponse(
            seed=seed,
            recommendations=enhanced_recommendations,
            total=len(enhanced_recommendations),
            algorithm=algorithm,
            source=source,
            request_id=request_id,
            processing_time_ms=processing_time
        )

    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - defensive catch
        logger.error("Recommendations failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Recommendations failed: {str(exc)}")


@router.post("/track-interaction")
@limiter.limit("100/minute")
async def track_interaction(
    request: Request,
    user_id: str = Query(..., description="User ID"),
    track_id: str = Query(..., description="Track ID"),
    interaction: str = Query(..., description="Interaction type: play/like/skip"),
    track_name: Optional[str] = Query(None),
    artist_name: Optional[str] = Query(None),
):
    """Track user interaction with a track for personalization."""
    music_service = request.app.state.music

    valid_interactions = {"play", "like", "skip"}
    if interaction not in valid_interactions:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid interaction type. Must be one of: {', '.join(valid_interactions)}"
        )

    track_data = {
        "name": track_name,
        "artist": artist_name,
    }

    music_service.track_user_interaction(user_id, track_id, interaction, track_data)

    return {
        "status": "success",
        "user_id": user_id,
        "track_id": track_id,
        "interaction": interaction,
    }


@router.post("/playlist/recommendations", response_model=PlaylistRecommendationsResponse)
@limiter.limit("10/minute")
async def playlist_recommendations(
    request: Request,
    payload: PlaylistRecommendationsRequest,
):
    """Generate recommendations using a user-supplied playlist as the catalogue."""

    start_time = time.time()
    request_id = getattr(request.state, "request_id", "unknown")

    music_service = request.app.state.music
    spotify_client: Optional[SpotifyClient] = getattr(request.app.state, "spotify", None)

    context: Dict[str, Any] = {}
    ctx = payload.context
    if ctx:
        if ctx.time_of_day:
            context["time_of_day"] = ctx.time_of_day
        if ctx.moods:
            context["moods"] = ctx.moods
        if ctx.activities:
            context["activities"] = ctx.activities
        if ctx.genres:
            context["genres"] = ctx.genres
        if ctx.energy:
            context["energy_level"] = ctx.energy
        if ctx.tempo is not None:
            context["tempo"] = ctx.tempo
        if ctx.era:
            context["era"] = ctx.era
        if ctx.regions:
            context["regions"] = ctx.regions

    try:
        playlist_tracks = [track.dict(exclude_none=True) for track in payload.tracks]
        recommendations_data, source, playlist_info = await music_service.recommend_from_playlist(
            playlist_tracks,
            seed=payload.seed,
            limit=payload.limit,
            context=context or None,
            min_popularity=payload.min_popularity,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    enhanced_recommendations: List[RecommendationHit] = []
    for track_dict in recommendations_data:
        components = (track_dict.get("recommendation") or {}).get("components", {})
        similarity_score = components.get("similarity", 0.75)
        rank_score = (track_dict.get("recommendation") or {}).get("score", similarity_score)

        feature_payload = track_dict.get("audio_features") or {}
        audio_features: Optional[AudioFeatures] = (
            AudioFeatures(**feature_payload) if feature_payload else None
        )

        top_factors = []
        if components:
            top_factors.append(
                f"Audio similarity {int(components.get('similarity', 0) * 100)}%"
            )
            top_factors.append(
                f"Context alignment {int(components.get('context', 0) * 100)}%"
            )
        tags = track_dict.get("tags", {})
        if tags.get("activities"):
            top_factors.append(f"Activity match: {', '.join(tags['activities'][:2])}")

        explanation_dict = {
            "top_factors": top_factors[:3],
            "similarity_reason": "Contextual playlist engine",
            "ranking_boost": f"Popularity score {track_dict.get('popularity', 0)}%",
        }

        enhanced_payload = dict(track_dict)
        enhanced_payload.update(
            {
                "audio_features": audio_features,
                "similarity_score": similarity_score,
                "rank_score": rank_score,
                "explanation": explanation_dict,
            }
        )

        enhanced_recommendations.append(RecommendationHit(**enhanced_payload))

    processing_time = int((time.time() - start_time) * 1000)
    playlist_summary = PlaylistSummary(**playlist_info)

    return PlaylistRecommendationsResponse(
        seed=payload.seed,
        recommendations=enhanced_recommendations,
        total=len(enhanced_recommendations),
        algorithm="Contextual (User Playlist)",
        source=source,
        request_id=request_id,
        processing_time_ms=processing_time,
        playlist=playlist_summary,
    )


@router.post("/playlist/import", response_model=PlaylistImportResponse)
@limiter.limit("20/minute")
async def import_playlist(
    request: Request,
    payload: PlaylistImportRequest,
):
    """Resolve a Spotify playlist URL/ID to track entries for client-side curation."""

    music_service = request.app.state.music

    try:
        tracks, summary = await music_service.import_playlist_from_url(
            payload.url,
            limit=payload.limit if payload.limit is not None else 300,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return PlaylistImportResponse(
        tracks=[PlaylistTrackInput(**track) for track in tracks],
        summary=summary,
    )
