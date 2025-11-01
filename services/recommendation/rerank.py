"""
Maximal Marginal Relevance (MMR) for diversity-aware reranking.

MMR balances relevance and diversity by iteratively selecting tracks
that are similar to the query but dissimilar to already selected tracks.

Reference: "The Use of MMR, Diversity-Based Reranking for Reordering Documents
and Producing Summaries" (Carbonell & Goldstein, 1998)
"""

import logging
import numpy as np
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    dot = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return float(dot / (norm1 * norm2))


def mmr_rerank(
    candidates: List[Dict[str, Any]],
    query_vec: Optional[np.ndarray] = None,
    lambda_param: float = 0.7,
    k: int = 30,
    embedding_key: str = "embedding",
    score_key: str = "rank_score",
) -> List[Dict[str, Any]]:
    """
    Rerank candidates using Maximal Marginal Relevance.

    MMR Score = λ × Relevance - (1-λ) × MaxSimilarity

    Where:
    - Relevance: Similarity to query (or ML model score)
    - MaxSimilarity: Maximum similarity to already selected items
    - λ: Trade-off parameter (0=max diversity, 1=max relevance)

    Args:
        candidates: List of candidate dicts with embeddings and scores
        query_vec: Query embedding (optional, if None uses rank_score for relevance)
        lambda_param: Relevance vs diversity trade-off (default 0.7)
        k: Number of tracks to select
        embedding_key: Key in candidate dict for embedding vector
        score_key: Key in candidate dict for relevance score

    Returns:
        Reranked list of up to k candidates
    """
    if not candidates:
        return []

    # Limit k to available candidates
    k = min(k, len(candidates))

    # Separate candidates with embeddings from those without
    with_embeddings = [c for c in candidates if embedding_key in c]
    without_embeddings = [c for c in candidates if embedding_key not in c]

    if not with_embeddings:
        logger.warning(f"No candidates contain '{embedding_key}', skipping MMR")
        return candidates[:k]

    selected = []
    pool = with_embeddings.copy()

    while len(selected) < k and pool:
        # Compute MMR score for each candidate in pool
        mmr_scores = []

        for candidate in pool:
            # Relevance component
            if query_vec is not None and embedding_key in candidate:
                # Use cosine similarity to query
                relevance = cosine_similarity(query_vec, candidate[embedding_key])
            elif score_key in candidate:
                # Use pre-computed ML score
                relevance = candidate[score_key]
            else:
                # Fallback: no relevance
                relevance = 0.0

            # Diversity component (max similarity to selected)
            if not selected:
                max_sim = 0.0
            else:
                similarities = [
                    cosine_similarity(candidate[embedding_key], s[embedding_key])
                    for s in selected
                    if embedding_key in s
                ]
                max_sim = max(similarities) if similarities else 0.0

            # MMR formula
            mmr = lambda_param * relevance - (1 - lambda_param) * max_sim
            mmr_scores.append(mmr)

        # Select candidate with highest MMR score
        best_idx = int(np.argmax(mmr_scores))
        best_candidate = pool[best_idx]

        selected.append(best_candidate)
        pool.pop(best_idx)

    # If we still need more items, append candidates that lacked embeddings
    if len(selected) < k and without_embeddings:
        remaining = k - len(selected)
        selected.extend(without_embeddings[:remaining])

    return selected


def artist_diversity_rerank(
    candidates: List[Dict[str, Any]],
    max_per_artist: int = 2,
    artist_key: str = "artist",
) -> List[Dict[str, Any]]:
    """
    Simple artist diversity filter.

    Limits the number of tracks per artist to avoid repetition.

    Args:
        candidates: List of candidate tracks
        max_per_artist: Maximum tracks allowed per artist
        artist_key: Key in candidate dict for artist name

    Returns:
        Filtered list with artist diversity constraint
    """
    artist_counts = {}
    filtered = []

    for candidate in candidates:
        artist = candidate.get(artist_key, "Unknown")

        # Check artist count
        count = artist_counts.get(artist, 0)

        if count < max_per_artist:
            filtered.append(candidate)
            artist_counts[artist] = count + 1

    return filtered


def combined_diversity_rerank(
    candidates: List[Dict[str, Any]],
    query_vec: Optional[np.ndarray] = None,
    lambda_param: float = 0.7,
    k: int = 30,
    max_per_artist: int = 2,
    embedding_key: str = "embedding",
    score_key: str = "rank_score",
    artist_key: str = "artist",
) -> List[Dict[str, Any]]:
    """
    Combined MMR and artist diversity reranking.

    First applies MMR for embedding-level diversity,
    then enforces artist constraints.

    Args:
        candidates: List of candidate tracks
        query_vec: Query embedding for MMR
        lambda_param: MMR trade-off parameter
        k: Number of tracks to select
        max_per_artist: Maximum tracks per artist
        embedding_key: Key for embedding vector
        score_key: Key for relevance score
        artist_key: Key for artist name

    Returns:
        Reranked and filtered list
    """
    # Apply MMR first (may return more than k to allow artist filtering)
    mmr_candidates = mmr_rerank(
        candidates=candidates,
        query_vec=query_vec,
        lambda_param=lambda_param,
        k=k * 2,  # Get 2x to have buffer for artist filtering
        embedding_key=embedding_key,
        score_key=score_key,
    )

    # Apply artist diversity
    diverse_candidates = artist_diversity_rerank(
        candidates=mmr_candidates,
        max_per_artist=max_per_artist,
        artist_key=artist_key,
    )

    # Truncate to k
    return diverse_candidates[:k]
