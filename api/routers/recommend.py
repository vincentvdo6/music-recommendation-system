"""Music recommendation endpoint with neural ranking and diversity."""

import time
from typing import List, Dict, Any, Optional

import structlog
from fastapi import APIRouter, HTTPException, Request, Query
from pydantic import BaseModel, Field, validator

from api.middleware.timing import track_component_time

logger = structlog.get_logger()

router = APIRouter(tags=["recommend"])


class TrackRecommendation(BaseModel):
    """Individual track recommendation with scores and explanation."""
    canonical_uid: str = Field(..., description="Track identifier")
    title: str = Field(..., description="Track title")
    artist: str = Field(..., description="Artist name")
    album: Optional[str] = Field(None, description="Album name")
    similarity_score: float = Field(..., ge=0, le=1, description="Cosine similarity to seed")
    rank_score: float = Field(..., ge=0, le=1, description="Learned ranking score")
    popularity_score: Optional[float] = Field(None, description="Track popularity")
    features: Dict[str, float] = Field(..., description="Audio features")
    explanation: Dict[str, Any] = Field(..., description="Why this track was recommended")


class RecommendationResponse(BaseModel):
    """Response from recommendation endpoint."""
    seed: str = Field(..., description="Seed track canonical UID")
    recommendations: List[TrackRecommendation] = Field(..., description="Recommended tracks")
    pipeline: Dict[str, Any] = Field(..., description="Pipeline configuration used")
    request_id: str = Field(..., description="Request identifier")
    total_candidates: int = Field(..., description="Total tracks considered")
    timing_ms: Dict[str, int] = Field(..., description="Component timing breakdown")


@router.get("/recommend", response_model=RecommendationResponse)
async def get_recommendations(
    request: Request,
    seed: str = Query(..., description="Canonical UID of seed track"),
    k: int = Query(20, ge=1, le=100, description="Number of recommendations to return"),
    diversity_lambda: Optional[float] = Query(None, ge=0, le=1, description="MMR diversity parameter"),
    flow_optimization: Optional[bool] = Query(None, description="Enable flow optimization"),
    explanation: bool = Query(True, description="Include explanations"),
):
    """
    Get music recommendations based on a seed track.
    
    This endpoint provides personalized music recommendations using:
    - Neural embeddings for similarity matching
    - Learned ranking with XGBoost
    - MMR diversity optimization
    - Flow-based reordering for smooth transitions
    - SHAP-based explanations
    
    The pipeline stages:
    1. Lookup/generate seed embedding
    2. ANN search for top-200 candidates  
    3. Feature engineering (similarity, popularity, graph distance, etc.)
    4. Neural ranking with XGBoost
    5. MMR diversity selection
    6. Flow optimization for transitions
    7. Explanation generation
    """
    
    start_time = time.time()
    request_id = getattr(request.state, "request_id", "unknown")
    user_id = getattr(request.state, "user_id", "anonymous")
    
    logger.info(
        "Recommendation request started",
        request_id=request_id,
        user_id=user_id,
        seed=seed,
        k=k,
        diversity_lambda=diversity_lambda
    )
    
    component_timings = {}
    
    try:
        # Consent verification (stub)
        consent_start = time.time()
        await _verify_seed_consent(seed)
        component_timings["consent"] = int((time.time() - consent_start) * 1000)
        
        # Embedding lookup/generation
        embedding_start = time.time()
        seed_embedding = await _get_seed_embedding(seed)
        component_timings["embedding_lookup"] = int((time.time() - embedding_start) * 1000)
        
        # ANN search for candidates
        ann_start = time.time()
        candidates = await _ann_search(seed_embedding, top_k=200)
        component_timings["ann_search"] = int((time.time() - ann_start) * 1000)
        
        # Feature engineering
        features_start = time.time()
        candidate_features = await _compute_features(seed, seed_embedding, candidates)
        component_timings["features"] = int((time.time() - features_start) * 1000)
        
        # Neural ranking
        ranking_start = time.time()
        ranked_candidates = await _rank_candidates(candidate_features)
        component_timings["ranking"] = int((time.time() - ranking_start) * 1000)
        
        # MMR diversity + flow optimization
        diversity_start = time.time()
        final_recs = await _apply_diversity_and_flow(
            ranked_candidates, 
            k=k,
            diversity_lambda=diversity_lambda,
            flow_optimization=flow_optimization
        )
        component_timings["mmr_flow"] = int((time.time() - diversity_start) * 1000)
        
        # Generate explanations
        if explanation:
            explain_start = time.time()
            final_recs = await _add_explanations(seed, final_recs)
            component_timings["explanations"] = int((time.time() - explain_start) * 1000)
        
        # Track component times in middleware
        for component, timing_ms in component_timings.items():
            track_component_time(request, component, timing_ms / 1000.0)
        
        # Build response
        response = RecommendationResponse(
            seed=seed,
            recommendations=final_recs,
            pipeline={
                "embedding_model": "mert_v2",
                "index_version": "ann_v1", 
                "ranker_version": "xgb_v1",
                "diversity_lambda": diversity_lambda or 0.7,
                "flow_optimization": flow_optimization or False,
                "feature_flags": await _get_active_flags()
            },
            request_id=request_id,
            total_candidates=len(candidates),
            timing_ms=component_timings
        )
        
        logger.info(
            "Recommendation request completed",
            request_id=request_id,
            seed=seed,
            num_recommendations=len(final_recs),
            total_duration_ms=int((time.time() - start_time) * 1000)
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Recommendation request failed",
            request_id=request_id,
            seed=seed,
            error=str(e),
            duration_ms=int((time.time() - start_time) * 1000)
        )
        raise HTTPException(status_code=500, detail="Internal server error")


async def _verify_seed_consent(seed: str):
    """Verify that we have consent to use this seed track (stub)."""
    # In real implementation, would check consent_log table
    pass


async def _get_seed_embedding(seed: str) -> List[float]:
    """Get or generate embedding for seed track (stub)."""
    # In real implementation:
    # 1. Check cache (Redis)
    # 2. Check database (audio_features table)  
    # 3. If not found, generate from audio
    
    import numpy as np
    embedding = np.random.normal(0, 1, 256).astype(np.float32)
    embedding = embedding / np.linalg.norm(embedding)
    return embedding.tolist()


async def _ann_search(seed_embedding: List[float], top_k: int = 200) -> List[Dict[str, Any]]:
    """Search for similar tracks using ANN index (stub)."""
    # In real implementation, would call FAISS service
    
    import random
    candidates = []
    for i in range(top_k):
        candidates.append({
            "canonical_uid": f"track:{i:06d}",
            "title": f"Song {i}",
            "artist": f"Artist {i // 10}",
            "album": f"Album {i // 20}" if i % 3 == 0 else None,
            "distance": random.uniform(0.1, 0.9),
            "similarity_score": 1.0 - random.uniform(0.1, 0.9)
        })
    
    # Sort by similarity (descending)
    candidates.sort(key=lambda x: x["similarity_score"], reverse=True)
    return candidates


async def _compute_features(
    seed: str, 
    seed_embedding: List[float], 
    candidates: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Compute ranking features for candidates (stub)."""
    # In real implementation, would compute:
    # - Cosine similarity
    # - Popularity score + penalty  
    # - Track age features
    # - Graph distance (co-playlist, co-listen)
    # - Audio feature deltas (tempo, energy, key)
    # - Camelot key distance
    
    import random
    
    for candidate in candidates:
        candidate.update({
            "pop_score": random.uniform(0.0, 1.0),
            "pop_penalty": random.uniform(0.0, 0.2),
            "track_age_days": random.randint(1, 3650),
            "graph_distance": random.uniform(0.0, 1.0),
            "tempo_delta": random.uniform(-30, 30),
            "energy_delta": random.uniform(-0.5, 0.5),
            "camelot_distance": random.randint(0, 6),
            "key_compatibility": random.choice([True, False])
        })
    
    return candidates


async def _rank_candidates(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Rank candidates using learned model (stub)."""
    # In real implementation, would:
    # 1. Extract feature vectors
    # 2. Load XGBoost model
    # 3. Predict ranking scores
    # 4. Sort by rank score
    
    import random
    
    for candidate in candidates:
        candidate["rank_score"] = random.uniform(0.0, 1.0)
    
    # Sort by rank score (descending)
    candidates.sort(key=lambda x: x["rank_score"], reverse=True)
    return candidates


async def _apply_diversity_and_flow(
    candidates: List[Dict[str, Any]],
    k: int,
    diversity_lambda: Optional[float] = None,
    flow_optimization: Optional[bool] = None
) -> List[TrackRecommendation]:
    """Apply MMR diversity and flow optimization (stub)."""
    # In real implementation, would:
    # 1. Apply MMR algorithm with lambda parameter
    # 2. Optimize for smooth transitions (tempo, key, energy)
    # 3. Ensure no duplicate artists in top positions
    
    diversity_lambda = diversity_lambda or 0.7
    flow_optimization = flow_optimization or False
    
    # Simple diverse selection (stub)
    selected = []
    used_artists = set()
    
    for candidate in candidates:
        if len(selected) >= k:
            break
        
        # Simple artist diversity
        if candidate["artist"] not in used_artists or len(selected) == 0:
            selected.append(candidate)
            used_artists.add(candidate["artist"])
    
    # Convert to response format
    recommendations = []
    for candidate in selected:
        recommendations.append(TrackRecommendation(
            canonical_uid=candidate["canonical_uid"],
            title=candidate["title"],
            artist=candidate["artist"],
            album=candidate.get("album"),
            similarity_score=candidate["similarity_score"],
            rank_score=candidate["rank_score"],
            popularity_score=candidate.get("pop_score"),
            features={
                "tempo": candidate.get("tempo", 120.0),
                "energy": candidate.get("energy", 0.5),
                "valence": candidate.get("valence", 0.5),
                "danceability": candidate.get("danceability", 0.5)
            },
            explanation={}  # Will be filled in by explanation step
        ))
    
    return recommendations


async def _add_explanations(seed: str, recommendations: List[TrackRecommendation]) -> List[TrackRecommendation]:
    """Add SHAP-based explanations (stub)."""
    # In real implementation, would:
    # 1. Use SHAP to explain ranking decisions
    # 2. Fall back to heuristic explanations
    # 3. Format for user consumption
    
    explanations = [
        "Similar tempo and energy",
        "Same key signature", 
        "Popular among similar listeners",
        "Matching genre and mood",
        "Complementary audio characteristics"
    ]
    
    import random
    
    for rec in recommendations:
        rec.explanation = {
            "top_factors": random.sample(explanations, k=min(3, len(explanations))),
            "similarity_reason": "High cosine similarity in embedding space",
            "ranking_boost": random.choice([
                "Strong collaborative filtering signal",
                "Trending in your music taste cluster", 
                "Perfect tempo matching",
                "Harmonic key compatibility"
            ])
        }
    
    return recommendations


async def _get_active_flags() -> Dict[str, Any]:
    """Get currently active feature flags (stub)."""
    return {
        "embedding_model": "mert_v2",
        "index_version": "ann_v1",
        "ranker_version": "xgb_v1", 
        "preview_adapter_enabled": False,
        "flow_optimizer_enabled": False,
        "cache_enabled": True
    }