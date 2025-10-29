"""
Hybrid recommendation engine with two-stage retrieval and learned ranking.

Architecture:
1. Retrieval Stage: Fast candidate generation using item2vec ANN + catalogue filtering
2. Ranking Stage: Learned model combines multiple signals (item2vec, audio, popularity, etc.)
3. Reranking Stage: MMR for diversity without sacrificing relevance

This follows the YouTube/Spotify production pattern for scalable recommendations.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Optional, Set

from services.recommendation.contextual_engine import ContextualRecommendationEngine
from services.recommendation.embeddings import EmbeddingService
from services.recommendation.ranker import FeatureExtractor, LearnedRanker, SimpleRanker
from services.recommendation.rerank import combined_diversity_rerank
from services.recommendation.audio_similarity import AudioSimilarityEngine

logger = logging.getLogger(__name__)


class HybridRecommendationEngine(ContextualRecommendationEngine):
    """
    Enhanced recommendation engine with two-stage retrieval and learned ranking.

    Extends ContextualRecommendationEngine with:
    - Item2vec collaborative filtering
    - ANN-based fast retrieval
    - Learned ranker (LightGBM)
    - MMR diversity reranking
    """

    def __init__(
        self,
        catalogue,
        embedding_service: Optional[EmbeddingService] = None,
        ranker: Optional[LearnedRanker] = None,
        use_mmr: bool = True,
        **kwargs
    ):
        """
        Initialize hybrid engine.

        Args:
            catalogue: TrackCatalogue instance
            embedding_service: Item2vec embedding service (optional)
            ranker: Learned ranker (optional, falls back to SimpleRanker)
            use_mmr: Whether to apply MMR diversity reranking
            **kwargs: Additional args for base engine
        """
        super().__init__(catalogue, **kwargs)

        self.embedding_service = embedding_service
        self.use_mmr = use_mmr

        # Initialize ranker
        if ranker:
            self.ranker = ranker
        else:
            # Fallback to simple weighted ranker
            logger.info("No learned ranker provided, using SimpleRanker")
            self.ranker = SimpleRanker()

        # Initialize feature extractor
        self.feature_extractor = FeatureExtractor(self.audio_similarity)

    async def get_recommendations(
        self,
        *,
        seed_track_id: Optional[str] = None,
        seed_metadata: Optional[Dict[str, Any]] = None,
        seed_features: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
        user_profile: Optional[Dict[str, Any]] = None,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """
        Generate recommendations using two-stage retrieval + learned ranking.

        Stage 1 (Retrieval):
        - Union of item2vec neighbors (from seed + playlist)
        - Union of catalogue tracks filtered by context tags
        - Fast ANN search for ~1-2k candidates

        Stage 2 (Ranking):
        - Extract features for each candidate
        - Score with learned ranker or simple weighted combination
        - Sort by score descending

        Stage 3 (Reranking):
        - Apply MMR for diversity
        - Enforce artist constraints
        - Return top-K

        Args:
            seed_track_id: Optional seed track ID
            seed_metadata: Optional seed track metadata
            seed_features: Optional seed audio features
            context: Contextual hints (mood, activity, etc.)
            user_profile: Playlist-derived user profile
            limit: Number of recommendations to return

        Returns:
            List of recommended tracks with scores and explanations
        """
        if not user_profile:
            raise ValueError(
                "user_profile is required. Build it from a playlist before requesting recommendations."
            )

        limit = max(1, min(limit, 50))

        # Merge and normalize context
        merged_context = self._merge_context(user_profile.get("context") or {}, context or {})
        normalised_context = self._normalise_context(merged_context)

        # Resolve seed entry
        seed_entry = self._resolve_seed_entry(seed_track_id, seed_metadata)
        if not seed_features and seed_metadata and seed_metadata.get("audio_features"):
            seed_features = seed_metadata["audio_features"]
        if not seed_features and seed_entry:
            seed_features = seed_entry.get("audio_features")

        # Build target profile
        target_profile = self._build_target_profile(
            seed_features,
            merged_context,
            seed_entry,
            user_profile=user_profile,
        )

        # === STAGE 1: RETRIEVAL ===
        candidates = self._retrieve_candidates(
            user_profile=user_profile,
            seed_track_id=seed_track_id,
            normalised_context=normalised_context,
            limit=limit * 50,  # Retrieve 50x candidates for ranking
        )

        logger.info(f"Retrieved {len(candidates)} candidates")

        # === STAGE 2: RANKING ===
        ranked_candidates = self._rank_candidates(
            candidates=candidates,
            user_profile=user_profile,
            seed_entry=seed_entry,
            target_profile=target_profile,
        )

        logger.info(f"Ranked {len(ranked_candidates)} candidates")

        # === STAGE 3: RERANKING (Diversity) ===
        if self.use_mmr and len(ranked_candidates) > limit:
            final_recommendations = self._apply_diversity(
                candidates=ranked_candidates,
                limit=limit,
                user_profile=user_profile,
            )
        else:
            # Just apply artist diversity
            final_recommendations = self._apply_diversity_simple(
                ranked_candidates, limit
            )

        logger.info(f"Final recommendations: {len(final_recommendations)}")

        # Add explanations
        for rec in final_recommendations:
            rec["explanation"] = self._build_explanation(rec, target_profile)

        return final_recommendations

    def _retrieve_candidates(
        self,
        user_profile: Dict[str, Any],
        seed_track_id: Optional[str],
        normalised_context: Dict[str, Any],
        limit: int = 1000,
    ) -> List[Dict[str, Any]]:
        """
        Stage 1: Fast candidate retrieval.

        Uses multiple retrieval strategies:
        1. Item2vec neighbors of seed track
        2. Item2vec neighbors of playlist (mean pooling)
        3. Catalogue tracks filtered by context tags

        Returns union of all strategies.
        """
        candidate_ids: Set[str] = set()

        # Strategy 1: Item2vec neighbors of seed
        if seed_track_id and self.embedding_service:
            seed_neighbors = self.embedding_service.get_neighbors(
                track_id=seed_track_id,
                k=min(500, limit // 2),
            )
            candidate_ids.update(seed_neighbors)
            logger.info(f"Retrieved {len(seed_neighbors)} seed neighbors")

        # Strategy 2: Item2vec neighbors of playlist
        if self.embedding_service:
            playlist_track_ids = list(user_profile.get("track_ids", []))
            if playlist_track_ids:
                playlist_neighbors = self.embedding_service.get_playlist_neighbors(
                    track_ids=playlist_track_ids,
                    k=min(1000, limit),
                )
                candidate_ids.update(playlist_neighbors)
                logger.info(f"Retrieved {len(playlist_neighbors)} playlist neighbors")

        # Strategy 3: Catalogue filtering by context
        context_candidates = self._get_context_filtered_candidates(
            normalised_context, limit=min(500, limit // 2)
        )
        candidate_ids.update(c["id"] for c in context_candidates if "id" in c)
        logger.info(f"Retrieved {len(context_candidates)} context-filtered candidates")

        # Remove playlist tracks
        playlist_track_ids = user_profile.get("track_ids", set())
        candidate_ids -= playlist_track_ids

        # Lookup full track data
        candidates = []
        for track_id in candidate_ids:
            track = self.catalogue.get_track_by_id(track_id)
            if track:
                # Add item2vec embedding if available
                if self.embedding_service:
                    emb = self.embedding_service.item2vec.vector(track_id)
                    if emb is not None:
                        track["i2v_embedding"] = emb

                candidates.append(track)

        return candidates

    def _get_context_filtered_candidates(
        self,
        normalised_context: Dict[str, Any],
        limit: int = 500,
    ) -> List[Dict[str, Any]]:
        """Get candidates filtered by context tags (from catalogue)."""
        # This is a simplified version - in practice you'd do tag-based filtering
        all_tracks = self.catalogue.all_tracks()

        # Filter by genres if specified
        genres = normalised_context.get("genres", [])
        if genres:
            filtered = [
                track for track in all_tracks
                if any(
                    genre in track.get("tags", {}).get("genres", [])
                    for genre in genres
                )
            ]
        else:
            filtered = all_tracks

        # Take top by popularity
        filtered = sorted(filtered, key=lambda t: t.get("popularity", 0), reverse=True)
        return filtered[:limit]

    def _rank_candidates(
        self,
        candidates: List[Dict[str, Any]],
        user_profile: Dict[str, Any],
        seed_entry: Optional[Dict[str, Any]],
        target_profile: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Stage 2: Rank candidates using learned model or simple ranker.
        """
        # Prepare playlist profile for feature extraction
        playlist_profile = {
            "audio_features": user_profile.get("audio_features", {}),
            "artists": set(
                track.get("artist", "")
                for track in user_profile.get("tracks", [])
            ),
            "genres": user_profile.get("context", {}).get("genres", []),
            "avg_year": self._get_avg_year(user_profile),
        }

        # Add item2vec embedding (mean of playlist)
        if self.embedding_service:
            playlist_track_ids = list(user_profile.get("track_ids", []))
            playlist_emb = self.embedding_service.item2vec.mean_vector(playlist_track_ids)
            if playlist_emb is not None:
                playlist_profile["i2v_embedding"] = playlist_emb

        # Rank using learned or simple ranker
        ranked = self.ranker.rank_candidates(
            candidates=candidates,
            feature_extractor=self.feature_extractor,
            playlist_profile=playlist_profile,
            seed_track=seed_entry,
        )

        return ranked

    def _apply_diversity(
        self,
        candidates: List[Dict[str, Any]],
        limit: int,
        user_profile: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Stage 3: Apply MMR diversity reranking."""
        # Compute query vector (playlist mean embedding)
        query_vec = None
        if self.embedding_service:
            playlist_track_ids = list(user_profile.get("track_ids", []))
            query_vec = self.embedding_service.item2vec.mean_vector(playlist_track_ids)

        # Apply combined MMR + artist diversity
        diverse_candidates = combined_diversity_rerank(
            candidates=candidates,
            query_vec=query_vec,
            lambda_param=0.7,  # 70% relevance, 30% diversity
            k=limit,
            max_per_artist=2,
            embedding_key="i2v_embedding",
            score_key="rank_score",
            artist_key="artist",
        )

        return diverse_candidates

    def _apply_diversity_simple(
        self,
        candidates: List[Dict[str, Any]],
        limit: int,
    ) -> List[Dict[str, Any]]:
        """Simple artist diversity (no MMR)."""
        artist_counts = {}
        diverse = []

        for candidate in candidates:
            artist = candidate.get("artist", "Unknown")
            count = artist_counts.get(artist, 0)

            if count < 2:  # Max 2 per artist
                diverse.append(candidate)
                artist_counts[artist] = count + 1

            if len(diverse) >= limit:
                break

        return diverse

    def _build_explanation(
        self,
        recommendation: Dict[str, Any],
        target_profile: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Build explanation for recommendation."""
        factors = []

        # Item2vec similarity
        if "rank_score" in recommendation:
            factors.append(f"Relevance score: {recommendation['rank_score']:.2f}")

        # Audio similarity
        audio_sim = recommendation.get("audio_similarity", 0)
        if audio_sim > 0:
            factors.append(f"Audio similarity: {audio_sim:.0%}")

        # Popularity
        pop = recommendation.get("popularity", 0)
        if pop > 70:
            factors.append(f"Popular track ({pop}/100)")

        return {
            "top_factors": factors[:3],
            "similarity_reason": "Hybrid: Item2vec + Audio + Context",
            "ranking_method": "Learned ranker" if isinstance(self.ranker, LearnedRanker) else "Weighted features",
        }

    def _get_avg_year(self, user_profile: Dict[str, Any]) -> int:
        """Extract average release year from profile."""
        # Try to compute from tracks if available
        tracks = user_profile.get("tracks", [])
        years = [t.get("release_year") for t in tracks if t.get("release_year")]

        if years:
            return int(sum(years) / len(years))

        return 2020  # Default

    def recommend(
        self,
        playlist_tracks: List[str],
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """
        Simplified sync method for playlist continuation (for evaluation).

        Args:
            playlist_tracks: List of track IDs in the playlist
            limit: Number of recommendations to return

        Returns:
            List of recommended tracks
        """
        if not playlist_tracks:
            logger.warning("No playlist tracks provided")
            return []

        # Stage 1: Retrieve candidates using item2vec
        candidates = self._retrieve_candidates_simple(playlist_tracks, limit * 50)
        logger.info(f"Stage 1: Retrieved {len(candidates)} candidates")

        if not candidates:
            logger.warning("No candidates retrieved")
            return []

        # Stage 2: Rank candidates
        ranked = self._rank_candidates_simple(
            candidates=candidates,
            playlist_tracks=playlist_tracks,
        )
        logger.info(f"Stage 2: Ranked {len(ranked)} candidates")

        # Stage 3: Apply simple diversity
        final = self._apply_diversity_simple(ranked, limit)
        logger.info(f"Stage 3: Final {len(final)} recommendations")

        return final

    def _retrieve_candidates_simple(
        self,
        playlist_tracks: List[str],
        limit: int = 1000,
    ) -> List[Dict[str, Any]]:
        """Simple candidate retrieval for evaluation."""
        candidate_ids: Set[str] = set()

        # Get item2vec neighbors
        if self.embedding_service:
            neighbors = self.embedding_service.get_playlist_neighbors(
                track_ids=playlist_tracks,
                k=limit,
            )
            logger.info(f"get_playlist_neighbors returned {len(neighbors)} neighbors")
            candidate_ids.update(neighbors)
        else:
            logger.warning("No embedding service available")

        logger.info(f"Candidate IDs before filtering: {len(candidate_ids)}")

        # Remove playlist tracks
        candidate_ids -= set(playlist_tracks)
        logger.info(f"Candidate IDs after filtering playlist tracks: {len(candidate_ids)}")

        # Lookup full track data (or create minimal stub)
        candidates = []
        for track_id in candidate_ids:
            track = self.catalogue.get_by_id(track_id)

            if not track:
                # Create minimal track stub for tracks not in catalogue
                # Use unique artist ID to avoid diversity filter issues
                track = {
                    "id": track_id,
                    "name": track_id,
                    "artist": f"Artist_{hash(track_id) % 10000}",  # Unique artist per track
                    "popularity": 50,
                    "audio_features": {},
                    "tags": {"genres": [], "moods": []},
                }

            # Add item2vec embedding if available
            if self.embedding_service:
                emb = self.embedding_service.item2vec.vector(track_id)
                if emb is not None:
                    track["i2v_embedding"] = emb

            # Add track to candidates
            candidates.append(track)

        return candidates

    def _rank_candidates_simple(
        self,
        candidates: List[Dict[str, Any]],
        playlist_tracks: List[str],
    ) -> List[Dict[str, Any]]:
        """Simple ranking for evaluation."""
        # Build playlist profile
        playlist_profile = {
            "artists": set(),
            "genres": [],
            "avg_year": 2020,
        }

        # Add item2vec embedding
        if self.embedding_service:
            playlist_emb = self.embedding_service.item2vec.mean_vector(playlist_tracks)
            if playlist_emb is not None:
                playlist_profile["i2v_embedding"] = playlist_emb

        # Rank using ranker
        ranked = self.ranker.rank_candidates(
            candidates=candidates,
            feature_extractor=self.feature_extractor,
            playlist_profile=playlist_profile,
            seed_track=None,
        )

        return ranked
