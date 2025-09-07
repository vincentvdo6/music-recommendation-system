"""SQLAlchemy models for the music recommendation system."""

from datetime import datetime
from typing import Dict, List, Optional, Any

from sqlalchemy import (
    Column, Integer, String, Text, Float, DateTime, Boolean, JSON,
    ForeignKey, Index, func
)
from sqlalchemy.dialects.postgresql import UUID, ARRAY
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()


class Track(Base):
    """Music track metadata."""
    __tablename__ = 'tracks'
    
    id = Column(Integer, primary_key=True)
    canonical_uid = Column(String(128), unique=True, nullable=False)
    isrc = Column(String(32), nullable=True)
    mbid = Column(UUID, nullable=True)
    acoustid = Column(String(36), nullable=True)
    title = Column(Text, nullable=False)
    artist = Column(Text, nullable=False)
    album = Column(Text, nullable=True)
    duration_ms = Column(Integer, nullable=True)
    release_year = Column(Integer, nullable=True)
    genre = Column(String(100), nullable=True)
    popularity_score = Column(Float, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    
    # Relationships
    audio_features = relationship("AudioFeature", back_populates="track", cascade="all, delete-orphan")
    seed_judgments = relationship("Judgment", foreign_keys="[Judgment.seed_track_id]", back_populates="seed_track")
    candidate_judgments = relationship("Judgment", foreign_keys="[Judgment.candidate_track_id]", back_populates="candidate_track")
    
    __table_args__ = (
        Index('idx_tracks_canonical_uid', 'canonical_uid'),
        Index('idx_tracks_isrc', 'isrc'),
        Index('idx_tracks_mbid', 'mbid'),
        Index('idx_tracks_artist', 'artist'),
        Index('idx_tracks_created_at', 'created_at'),
    )


class AudioFeature(Base):
    """Audio embeddings and features for tracks."""
    __tablename__ = 'audio_features'
    
    id = Column(Integer, primary_key=True)
    track_id = Column(Integer, ForeignKey('tracks.id', ondelete='CASCADE'), nullable=False)
    embedding_model = Column(String(50), nullable=False)
    embedding_version = Column(String(20), nullable=False)
    embedding = Column(ARRAY(Float), nullable=False)
    embedding_vector = Column(Text, nullable=True)  # For pgvector when needed
    tempo = Column(Float, nullable=True)
    key_signature = Column(String(10), nullable=True)
    energy = Column(Float, nullable=True)
    valence = Column(Float, nullable=True)
    danceability = Column(Float, nullable=True)
    segment_start_ms = Column(Integer, nullable=True)
    segment_end_ms = Column(Integer, nullable=True)
    provenance = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    
    # Relationships
    track = relationship("Track", back_populates="audio_features")
    
    __table_args__ = (
        Index('idx_audio_features_track_model', 'track_id', 'embedding_model', 'embedding_version', unique=True),
        Index('idx_audio_features_model_version', 'embedding_model', 'embedding_version'),
    )


class AnnManifest(Base):
    """Metadata for ANN index versions."""
    __tablename__ = 'ann_manifests'
    
    id = Column(Integer, primary_key=True)
    index_version = Column(String(50), unique=True, nullable=False)
    embedding_model = Column(String(50), nullable=False)
    embedding_version = Column(String(20), nullable=False)
    index_type = Column(String(20), nullable=False)  # HNSW32, IVF-PQ
    index_params = Column(JSON, nullable=True)
    num_tracks = Column(Integer, nullable=False)
    dimension = Column(Integer, nullable=False)
    recall_stats = Column(JSON, nullable=True)
    build_metadata = Column(JSON, nullable=True)
    storage_path = Column(String(500), nullable=False)
    status = Column(String(20), nullable=False, default='building')  # building, ready, deprecated
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    
    __table_args__ = (
        Index('idx_ann_manifests_version', 'index_version'),
        Index('idx_ann_manifests_status', 'status'),
        Index('idx_ann_manifests_model', 'embedding_model', 'embedding_version'),
    )


class Judgment(Base):
    """Relevance judgments for evaluation."""
    __tablename__ = 'judgments'
    
    id = Column(Integer, primary_key=True)
    seed_track_id = Column(Integer, ForeignKey('tracks.id', ondelete='CASCADE'), nullable=False)
    candidate_track_id = Column(Integer, ForeignKey('tracks.id', ondelete='CASCADE'), nullable=False)
    relevance_score = Column(Float, nullable=False)  # 0-1 or 1-5 scale
    judgment_type = Column(String(20), nullable=False)  # explicit, implicit, crowdsourced
    user_id = Column(String(128), nullable=True)
    session_id = Column(String(128), nullable=True)
    context = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    
    # Relationships
    seed_track = relationship("Track", foreign_keys=[seed_track_id], back_populates="seed_judgments")
    candidate_track = relationship("Track", foreign_keys=[candidate_track_id], back_populates="candidate_judgments")
    
    __table_args__ = (
        Index('idx_judgments_seed', 'seed_track_id'),
        Index('idx_judgments_candidate', 'candidate_track_id'),
        Index('idx_judgments_user_session', 'user_id', 'session_id'),
        Index('idx_judgments_created_at', 'created_at'),
    )


class ConsentLog(Base):
    """Audit log for user consent and data processing."""
    __tablename__ = 'consent_log'
    
    id = Column(Integer, primary_key=True)
    consent_id = Column(String(128), unique=True, nullable=False)
    user_hash = Column(String(128), nullable=False)  # HMAC(salt, user/ip)
    audio_hash = Column(String(128), nullable=False)  # SHA256(audio)
    terms_version = Column(String(20), nullable=False)
    retention_days = Column(Integer, nullable=False)
    preview_duration_ms = Column(Integer, nullable=False)
    expires_at = Column(DateTime(timezone=True), nullable=False)
    revoked_at = Column(DateTime(timezone=True), nullable=True)
    metadata = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    
    __table_args__ = (
        Index('idx_consent_log_consent_id', 'consent_id'),
        Index('idx_consent_log_user_hash', 'user_hash'),
        Index('idx_consent_log_audio_hash', 'audio_hash'),
        Index('idx_consent_log_expires_at', 'expires_at'),
        Index('idx_consent_log_created_at', 'created_at'),
    )