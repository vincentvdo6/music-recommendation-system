"""Song search and playlist recommendation endpoints."""

import logging
import time
import uuid
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query, Request
from slowapi import Limiter
from slowapi.util import get_remote_address

from api.models import (
    FeedbackRequest,
    FeedbackResponse,
    PlaylistImportRequest,
    PlaylistImportResponse,
    PlaylistRecommendationsRequest,
    PlaylistRecommendationsResponse,
    PlaylistSummary,
    PlaylistTrackInput,
    RecommendationHit,
    SearchHit,
    SearchResponse,
    Track,
)
from services.recommendation.profiles import get_profile

logger = logging.getLogger(__name__)
router = APIRouter(tags=["search"])
limiter = Limiter(key_func=get_remote_address)


def _metadata_hints(track: Dict[str, Any]) -> List[str]:
    """Human-readable hints from track metadata."""
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
@limiter.limit("120/minute")  # autocomplete: every fresh keystroke is one request
async def search_songs(
    request: Request,
    q: str = Query(..., min_length=2, max_length=200),
    limit: int = Query(10, ge=1, le=20),
):
    """Search for songs using available music providers."""
    start_time = time.perf_counter()
    request_id = getattr(request.state, "request_id", "unknown")

    music_service = request.app.state.music

    try:
        tracks_data, source = await music_service.search_tracks(q, limit=limit)

        results = [
            SearchHit(track=Track(**track_dict), why=_metadata_hints(track_dict))
            for track_dict in tracks_data
        ]

        return SearchResponse(
            query=q,
            count=len(results),
            results=results,
            source=source,
            request_id=request_id,
            processing_time_ms=int((time.perf_counter() - start_time) * 1000),
        )

    except Exception as exc:  # pragma: no cover - defensive catch
        logger.exception("Search failed")
        raise HTTPException(status_code=500, detail="Search failed") from exc


@router.post("/playlist/recommendations", response_model=PlaylistRecommendationsResponse)
@limiter.limit("10/minute")
async def playlist_recommendations(
    request: Request,
    payload: PlaylistRecommendationsRequest,
):
    """Generate recommendations from the user's playlist, driven by the searched song."""

    start_time = time.perf_counter()
    request_id = getattr(request.state, "request_id", "unknown")

    music_service = request.app.state.music
    effective_profile = get_profile(payload.profile).name

    try:
        playlist_tracks = [track.model_dump(exclude_none=True) for track in payload.tracks]
        recommendations_data, source, playlist_info = await music_service.recommend_from_playlist(
            playlist_tracks,
            seed=payload.seed,
            limit=payload.limit,
            session_id=payload.session_id,
            profile=effective_profile,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    hits: List[RecommendationHit] = []
    for track_dict in recommendations_data:
        components = (track_dict.get("recommendation") or {}).get("components", {})
        similarity_score: Optional[float] = track_dict.get("similarity_score")

        explanation = {
            "top_factors": (track_dict.get("why") or _metadata_hints(track_dict))[:3],
            "components": components,
            "ranking_method": source,
        }

        payload_dict = dict(track_dict)
        payload_dict.update({
            "similarity_score": similarity_score if similarity_score is not None else 0.0,
            "rank_score": track_dict.get("rank_score", 0.0),
            "explanation": explanation,
        })
        hits.append(RecommendationHit(**payload_dict))

    impression_id = str(uuid.uuid4())
    candidate_slate = playlist_info.pop("_candidate_slate", recommendations_data)
    provenance = playlist_info.pop("_provenance", {"profile": effective_profile})
    provenance["requested_profile"] = payload.profile
    feedback_store = getattr(request.app.state, "feedback_store", None)
    if feedback_store is not None:
        playlist_ids = [
            track.get("spotify_id") or track.get("id")
            for track in playlist_tracks
        ]
        try:
            feedback_store.record_impression(
                impression_id=impression_id,
                request_id=request_id,
                seed_id=payload.seed,
                mode="assisted" if playlist_tracks else "seed_only",
                playlist_ids=playlist_ids,
                source=source,
                recommendations=recommendations_data,
                candidates=candidate_slate,
                session_id=payload.session_id,
                profile=effective_profile,
                provenance=provenance,
            )
        except Exception:
            logger.exception("Could not record recommendation impression %s", impression_id)

    return PlaylistRecommendationsResponse(
        seed=payload.seed,
        recommendations=hits,
        total=len(hits),
        algorithm="item2vec retrieval + learned ranking",
        source=source,
        request_id=request_id,
        processing_time_ms=int((time.perf_counter() - start_time) * 1000),
        playlist=PlaylistSummary(**playlist_info),
        impression_id=impression_id,
        profile=effective_profile,
    )


@router.post("/feedback", response_model=FeedbackResponse)
@limiter.limit("120/minute")
async def recommendation_feedback(
    request: Request,
    payload: FeedbackRequest,
):
    """Record privacy-minimal feedback for an item that was actually shown."""
    feedback_store = getattr(request.app.state, "feedback_store", None)
    if feedback_store is None:
        raise HTTPException(status_code=503, detail="Feedback collection is unavailable")
    try:
        accepted = feedback_store.record_feedback(**payload.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not accepted:
        raise HTTPException(status_code=404, detail="Unknown recommendation impression")
    return FeedbackResponse(accepted=True)


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
