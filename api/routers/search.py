"""Song search and playlist recommendation endpoints."""

import logging
import time
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query, Request
from slowapi import Limiter
from slowapi.util import get_remote_address

from api.models import (
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
@limiter.limit("30/minute")
async def search_songs(
    request: Request,
    q: str = Query(..., min_length=2, max_length=200),
    limit: int = Query(10, ge=1, le=20),
):
    """Search for songs using available music providers."""
    start_time = time.time()
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
            processing_time_ms=int((time.time() - start_time) * 1000),
        )

    except Exception as exc:  # pragma: no cover - defensive catch
        logger.error("Search failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Search failed: {str(exc)}")


@router.post("/playlist/recommendations", response_model=PlaylistRecommendationsResponse)
@limiter.limit("10/minute")
async def playlist_recommendations(
    request: Request,
    payload: PlaylistRecommendationsRequest,
):
    """Generate recommendations from the user's playlist, driven by the searched song."""

    start_time = time.time()
    request_id = getattr(request.state, "request_id", "unknown")

    music_service = request.app.state.music

    try:
        playlist_tracks = [track.model_dump(exclude_none=True) for track in payload.tracks]
        recommendations_data, source, playlist_info = await music_service.recommend_from_playlist(
            playlist_tracks,
            seed=payload.seed,
            limit=payload.limit,
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

    return PlaylistRecommendationsResponse(
        seed=payload.seed,
        recommendations=hits,
        total=len(hits),
        algorithm="item2vec retrieval + learned ranking",
        source=source,
        request_id=request_id,
        processing_time_ms=int((time.time() - start_time) * 1000),
        playlist=PlaylistSummary(**playlist_info),
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
