from typing import List, Optional

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


class RecommendationHit(BaseModel):
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
    similarity_score: float
    rank_score: float
    explanation: dict


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


class PlaylistRecommendationsRequest(BaseModel):
    tracks: List[PlaylistTrackInput] = Field(default_factory=list)  # optional: seed-only mode
    seed: Optional[str] = None
    limit: int = Field(5, ge=1, le=20)


class PlaylistRecommendationsResponse(BaseModel):
    seed: Optional[str]
    recommendations: List[RecommendationHit]
    total: int
    algorithm: str
    source: str = "playlist"
    request_id: str
    processing_time_ms: int
    playlist: PlaylistSummary


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
