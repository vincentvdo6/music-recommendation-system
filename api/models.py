from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class Track(BaseModel):
    id: str
    name: str
    artist: str
    album: str
    duration_ms: int
    popularity: int
    preview_url: Optional[str] = None
    external_urls: dict
    uri: str
    release_date: str
    image_url: Optional[str] = None


class SearchHit(BaseModel):
    track: Track
    why: List[str] = Field(default_factory=list)


class SearchResponse(BaseModel):
    query: str
    count: int
    results: List[SearchHit]
    source: str = "spotify"
    request_id: str
    processing_time_ms: int


class RecommendationHit(Track):
    similarity_score: float
    rank_score: float
    explanation: dict
    discovery: bool = False


class HealthResponse(BaseModel):
    status: str
    version: str
    timestamp: float


class PlaylistTrackInput(BaseModel):
    raw: Optional[str] = None
    spotify_id: Optional[str] = None
    uri: Optional[str] = None
    url: Optional[str] = None
    name: Optional[str] = None
    artist: Optional[str] = None
    seed: bool = False
    album: Optional[str] = None
    image_url: Optional[str] = None


class PlaylistSummary(BaseModel):
    playlist_size: int
    resolved_tracks: int
    tracks_in_model: Optional[int] = None
    seed_in_model: Optional[bool] = None
    engine_candidates: Optional[int] = None
    playable_candidates: Optional[int] = None
    engine_tracks_served: Optional[int] = None
    fallback_tracks_served: Optional[int] = None


class PlaylistRecommendationsRequest(BaseModel):
    tracks: List[PlaylistTrackInput] = Field(default_factory=list)  # optional: seed-only mode
    seed: Optional[str] = None
    limit: int = Field(5, ge=1, le=20)
    session_id: Optional[str] = Field(None, min_length=8, max_length=64)
    profile: Literal["familiar", "balanced", "explorer"] = "familiar"


class PlaylistRecommendationsResponse(BaseModel):
    seed: Optional[str]
    recommendations: List[RecommendationHit]
    total: int
    algorithm: str
    source: str = "playlist"
    request_id: str
    processing_time_ms: int
    playlist: PlaylistSummary
    impression_id: str
    profile: Literal["familiar", "balanced", "explorer"] = "familiar"


class FeedbackRequest(BaseModel):
    impression_id: str = Field(min_length=8, max_length=64)
    track_id: str = Field(min_length=1, max_length=200)
    event: Literal[
        "view",
        "dismiss",
        "preview_start",
        "preview_complete",
        "like",
        "neutral",
        "dislike",
        "open_spotify",
        "open_apple",
        "skip",
        "replay",
        "save",
        "more_like_this",
        "not_similar_enough",
    ]
    position: Optional[int] = Field(None, ge=0, le=1000)
    dwell_ms: Optional[int] = Field(None, ge=0, le=3_600_000)


class FeedbackResponse(BaseModel):
    accepted: bool


class PlaylistImportRequest(BaseModel):
    url: str
    limit: Optional[int] = Field(300, ge=1, le=500)


class PlaylistImportSummary(BaseModel):
    id: str
    name: Optional[str] = None
    owner: Optional[str] = None
    total_tracks: int
    loaded_tracks: int
    followers: Optional[int] = None
    image_url: Optional[str] = None


class PlaylistImportResponse(BaseModel):
    tracks: List[PlaylistTrackInput]
    summary: PlaylistImportSummary
