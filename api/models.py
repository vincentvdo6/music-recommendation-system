from pydantic import BaseModel, Field
from typing import List, Optional


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


class AudioFeatures(BaseModel):
    acousticness: Optional[float] = None
    danceability: Optional[float] = None
    energy: Optional[float] = None
    instrumentalness: Optional[float] = None
    liveness: Optional[float] = None
    loudness: Optional[float] = None
    speechiness: Optional[float] = None
    tempo: Optional[float] = None
    valence: Optional[float] = None
    key: Optional[int] = None
    mode: Optional[int] = None
    time_signature: Optional[int] = None


class SearchHit(BaseModel):
    track: Track
    features: Optional[AudioFeatures] = None
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
    audio_features: Optional[AudioFeatures] = None
    similarity_score: float
    rank_score: float
    explanation: dict


class RecommendationsResponse(BaseModel):
    seed: str
    recommendations: List[RecommendationHit]
    total: int
    algorithm: str
    source: str = "spotify"
    request_id: str
    processing_time_ms: int


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


class PlaylistContext(BaseModel):
    time_of_day: Optional[str] = None
    moods: Optional[List[str]] = None
    activities: Optional[List[str]] = None
    genres: Optional[List[str]] = None
    energy: Optional[str] = None
    tempo: Optional[str] = None
    era: Optional[str] = None
    regions: Optional[List[str]] = None


class PlaylistSummary(BaseModel):
    playlist_size: int
    resolved_tracks: int
    top_moods: List[str] = Field(default_factory=list)
    top_activities: List[str] = Field(default_factory=list)
    top_genres: List[str] = Field(default_factory=list)
    regions: List[str] = Field(default_factory=list)
    time_of_day: Optional[str] = None
    energy_level: Optional[str] = None
    tempo: Optional[str] = None
    era: Optional[str] = None


class PlaylistRecommendationsRequest(BaseModel):
    tracks: List[PlaylistTrackInput]
    seed: Optional[str] = None
    limit: int = Field(5, ge=1, le=20)
    min_popularity: int = Field(0, ge=0, le=100)
    context: Optional[PlaylistContext] = None


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
