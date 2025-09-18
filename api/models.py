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