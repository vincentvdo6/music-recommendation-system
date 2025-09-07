"""Initial database schema

Revision ID: 0001
Revises: 
Create Date: 2024-01-01 12:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = '0001'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Enable pgvector extension
    op.execute('CREATE EXTENSION IF NOT EXISTS vector')
    
    # Create tracks table
    op.create_table(
        'tracks',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('canonical_uid', sa.String(128), nullable=False),
        sa.Column('isrc', sa.String(32), nullable=True),
        sa.Column('mbid', postgresql.UUID, nullable=True),
        sa.Column('acoustid', sa.String(36), nullable=True),
        sa.Column('title', sa.Text(), nullable=False),
        sa.Column('artist', sa.Text(), nullable=False),
        sa.Column('album', sa.Text(), nullable=True),
        sa.Column('duration_ms', sa.Integer(), nullable=True),
        sa.Column('release_year', sa.Integer(), nullable=True),
        sa.Column('genre', sa.String(100), nullable=True),
        sa.Column('popularity_score', sa.Float(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.Index('idx_tracks_canonical_uid', 'canonical_uid', unique=True),
        sa.Index('idx_tracks_isrc', 'isrc'),
        sa.Index('idx_tracks_mbid', 'mbid'),
        sa.Index('idx_tracks_artist', 'artist'),
        sa.Index('idx_tracks_created_at', 'created_at')
    )
    
    # Create audio_features table  
    op.create_table(
        'audio_features',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('track_id', sa.Integer(), nullable=False),
        sa.Column('embedding_model', sa.String(50), nullable=False),
        sa.Column('embedding_version', sa.String(20), nullable=False),
        sa.Column('embedding', postgresql.ARRAY(sa.Float()), nullable=False),
        sa.Column('embedding_vector', postgresql.TEXT, nullable=True),  # For pgvector
        sa.Column('tempo', sa.Float(), nullable=True),
        sa.Column('key_signature', sa.String(10), nullable=True),
        sa.Column('energy', sa.Float(), nullable=True),
        sa.Column('valence', sa.Float(), nullable=True),
        sa.Column('danceability', sa.Float(), nullable=True),
        sa.Column('segment_start_ms', sa.Integer(), nullable=True),
        sa.Column('segment_end_ms', sa.Integer(), nullable=True),
        sa.Column('provenance', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['track_id'], ['tracks.id'], ondelete='CASCADE'),
        sa.Index('idx_audio_features_track_model', 'track_id', 'embedding_model', 'embedding_version', unique=True),
        sa.Index('idx_audio_features_model_version', 'embedding_model', 'embedding_version')
    )
    
    # Create ann_manifests table
    op.create_table(
        'ann_manifests',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('index_version', sa.String(50), nullable=False),
        sa.Column('embedding_model', sa.String(50), nullable=False),
        sa.Column('embedding_version', sa.String(20), nullable=False),
        sa.Column('index_type', sa.String(20), nullable=False),  # HNSW32, IVF-PQ
        sa.Column('index_params', sa.JSON(), nullable=True),
        sa.Column('num_tracks', sa.Integer(), nullable=False),
        sa.Column('dimension', sa.Integer(), nullable=False),
        sa.Column('recall_stats', sa.JSON(), nullable=True),
        sa.Column('build_metadata', sa.JSON(), nullable=True),
        sa.Column('storage_path', sa.String(500), nullable=False),
        sa.Column('status', sa.String(20), nullable=False, server_default='building'),  # building, ready, deprecated
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.Index('idx_ann_manifests_version', 'index_version', unique=True),
        sa.Index('idx_ann_manifests_status', 'status'),
        sa.Index('idx_ann_manifests_model', 'embedding_model', 'embedding_version')
    )
    
    # Create judgments table for evaluation
    op.create_table(
        'judgments',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('seed_track_id', sa.Integer(), nullable=False),
        sa.Column('candidate_track_id', sa.Integer(), nullable=False),
        sa.Column('relevance_score', sa.Float(), nullable=False),  # 0-1 or 1-5 scale
        sa.Column('judgment_type', sa.String(20), nullable=False),  # explicit, implicit, crowdsourced
        sa.Column('user_id', sa.String(128), nullable=True),
        sa.Column('session_id', sa.String(128), nullable=True),
        sa.Column('context', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['seed_track_id'], ['tracks.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['candidate_track_id'], ['tracks.id'], ondelete='CASCADE'),
        sa.Index('idx_judgments_seed', 'seed_track_id'),
        sa.Index('idx_judgments_candidate', 'candidate_track_id'),
        sa.Index('idx_judgments_user_session', 'user_id', 'session_id'),
        sa.Index('idx_judgments_created_at', 'created_at')
    )
    
    # Create consent_log table
    op.create_table(
        'consent_log',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('consent_id', sa.String(128), nullable=False),
        sa.Column('user_hash', sa.String(128), nullable=False),  # HMAC(salt, user/ip)
        sa.Column('audio_hash', sa.String(128), nullable=False),  # SHA256(audio)
        sa.Column('terms_version', sa.String(20), nullable=False),
        sa.Column('retention_days', sa.Integer(), nullable=False),
        sa.Column('preview_duration_ms', sa.Integer(), nullable=False),
        sa.Column('expires_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('revoked_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('metadata', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.Index('idx_consent_log_consent_id', 'consent_id', unique=True),
        sa.Index('idx_consent_log_user_hash', 'user_hash'),
        sa.Index('idx_consent_log_audio_hash', 'audio_hash'),
        sa.Index('idx_consent_log_expires_at', 'expires_at'),
        sa.Index('idx_consent_log_created_at', 'created_at')
    )


def downgrade() -> None:
    op.drop_table('consent_log')
    op.drop_table('judgments')
    op.drop_table('ann_manifests')
    op.drop_table('audio_features')
    op.drop_table('tracks')
    op.execute('DROP EXTENSION IF EXISTS vector')