"""Audio analysis endpoint for generating embeddings."""

import time
from typing import Dict, Any, Optional

import structlog
from fastapi import APIRouter, HTTPException, Request, File, UploadFile, Form
from pydantic import BaseModel, Field, validator

from api.middleware.timing import track_component_time
from services.consent.manager import ConsentManager

logger = structlog.get_logger()

router = APIRouter(tags=["analyze"])


class ConsentRequest(BaseModel):
    """User consent information."""
    terms_version: str = Field(..., description="Terms of service version")
    retention_days: Optional[int] = Field(None, description="Data retention period in days")
    
    @validator('retention_days')
    def validate_retention_days(cls, v):
        if v is not None and (v < 1 or v > 365):
            raise ValueError("Retention days must be between 1 and 365")
        return v


class AnalyzeResponse(BaseModel):
    """Response from audio analysis."""
    canonical_uid: str = Field(..., description="Canonical track identifier")
    embedding: Dict[str, Any] = Field(..., description="Audio embedding information")
    features: Dict[str, float] = Field(..., description="Extracted audio features")
    provenance: Dict[str, Any] = Field(..., description="Model and processing metadata")
    request_id: str = Field(..., description="Request identifier for tracking")


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze_audio(
    request: Request,
    audio: UploadFile = File(..., description="Audio file (max 30 seconds preview)"),
    consent: str = Form(..., description="JSON consent information"),
    link: Optional[str] = Form(None, description="Optional URL to audio source")
):
    """
    Analyze audio content and generate embeddings.
    
    This endpoint processes audio previews (≤30 seconds) to generate:
    - High-dimensional embeddings for similarity matching
    - Traditional audio features (tempo, key, energy, etc.)
    - Metadata about the analysis process
    
    Requires valid user consent and JWT authentication.
    """
    
    start_time = time.time()
    request_id = getattr(request.state, "request_id", "unknown")
    user_id = getattr(request.state, "user_id", "anonymous")
    
    logger.info(
        "Audio analysis started",
        request_id=request_id,
        user_id=user_id,
        filename=audio.filename,
        content_type=audio.content_type
    )
    
    try:
        # Parse consent information
        import json
        try:
            consent_data = json.loads(consent)
            consent_request = ConsentRequest(**consent_data)
        except (json.JSONDecodeError, ValueError) as e:
            raise HTTPException(status_code=400, detail=f"Invalid consent data: {str(e)}")
        
        # Read audio content
        audio_content = await audio.read()
        
        # Validate consent and store
        consent_start = time.time()
        consent_manager = ConsentManager()
        
        # Get audio duration (stub - would use actual audio analysis)
        duration_ms = await _get_audio_duration(audio_content)
        
        # For now, we'll create a database session (in real implementation)
        # db = get_database_session()
        
        consent_id = "stub_consent_id"  # Would call: await consent_manager.store_consent(...)
        
        track_component_time(request, "consent", time.time() - consent_start)
        
        # Identity resolution (stub)
        identity_start = time.time()
        canonical_uid = await _resolve_identity(audio_content, link)
        track_component_time(request, "identity", time.time() - identity_start)
        
        # Embedding generation (stub)
        embedding_start = time.time()
        embedding_result = await _generate_embedding(audio_content, canonical_uid)
        track_component_time(request, "embedding_compute", time.time() - embedding_start)
        
        # Feature extraction (stub)
        features_start = time.time()
        audio_features = await _extract_features(audio_content)
        track_component_time(request, "features", time.time() - features_start)
        
        # Build response
        response = AnalyzeResponse(
            canonical_uid=canonical_uid,
            embedding={
                "model": embedding_result["model"],
                "version": embedding_result["version"], 
                "dimension": embedding_result["dimension"],
                "vector_id": embedding_result["vector_id"]
            },
            features=audio_features,
            provenance={
                "models_used": embedding_result["models_used"],
                "processing_time_ms": int((time.time() - start_time) * 1000),
                "consent_id": consent_id,
                "segment": embedding_result.get("segment"),
                "preview_adapter_used": False  # Feature flag controlled
            },
            request_id=request_id
        )
        
        logger.info(
            "Audio analysis completed",
            request_id=request_id,
            canonical_uid=canonical_uid,
            duration_ms=int((time.time() - start_time) * 1000)
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Audio analysis failed",
            request_id=request_id,
            error=str(e),
            duration_ms=int((time.time() - start_time) * 1000)
        )
        raise HTTPException(status_code=500, detail="Internal server error")


async def _get_audio_duration(audio_content: bytes) -> int:
    """Get audio duration in milliseconds (stub)."""
    # In real implementation, would use librosa or similar
    # For now, assume reasonable preview length
    return min(len(audio_content) // 1000, 30000)  # Max 30 seconds


async def _resolve_identity(audio_content: bytes, link: Optional[str]) -> str:
    """Resolve track identity from audio and metadata (stub)."""
    # In real implementation, would:
    # 1. Generate acoustid fingerprint
    # 2. Query MusicBrainz for MBID
    # 3. Look up ISRC if available
    # 4. Create canonical UID from best match
    
    import hashlib
    audio_hash = hashlib.sha256(audio_content).hexdigest()[:16]
    return f"track:{audio_hash}"


async def _generate_embedding(audio_content: bytes, canonical_uid: str) -> Dict[str, Any]:
    """Generate audio embedding using configured model (stub)."""
    # In real implementation, would:
    # 1. Check cache for existing embedding
    # 2. If not found, run MERT/CLMR model
    # 3. Apply preview adapter if enabled
    # 4. Store embedding in database/S3
    
    import numpy as np
    
    # Simulate embedding generation
    embedding_vector = np.random.normal(0, 1, 256).astype(np.float32)
    embedding_vector = embedding_vector / np.linalg.norm(embedding_vector)  # L2 normalize
    
    return {
        "model": "mert",
        "version": "v2",
        "dimension": 256,
        "vector_id": f"emb_{canonical_uid}",
        "models_used": ["mert_v2"],
        "segment": {"start_ms": 5000, "end_ms": 25000}  # Chorus detection stub
    }


async def _extract_features(audio_content: bytes) -> Dict[str, float]:
    """Extract traditional audio features (stub)."""
    # In real implementation, would use librosa
    import random
    
    return {
        "tempo": round(random.uniform(60, 180), 1),
        "key_signature": random.choice(["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]),
        "energy": round(random.uniform(0.1, 1.0), 3),
        "valence": round(random.uniform(0.1, 1.0), 3),  
        "danceability": round(random.uniform(0.1, 1.0), 3),
        "acousticness": round(random.uniform(0.1, 1.0), 3),
        "instrumentalness": round(random.uniform(0.0, 1.0), 3),
        "liveness": round(random.uniform(0.0, 1.0), 3),
        "speechiness": round(random.uniform(0.0, 1.0), 3),
    }