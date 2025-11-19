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
import pickle
from pathlib import Path
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
        ncf_recommender: Optional[Any] = None,
        use_mmr: bool = True,
        mood_predictor_path: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize hybrid engine.

        Args:
            catalogue: TrackCatalogue instance
            embedding_service: Item2vec embedding service (optional)
            ranker: Learned ranker (optional, falls back to SimpleRanker)
            ncf_recommender: Neural Collaborative Filtering recommender (optional, Phase 2)
            use_mmr: Whether to apply MMR diversity reranking
            mood_predictor_path: Path to mood predictor model (optional)
            **kwargs: Additional args for base engine
        """
        super().__init__(catalogue, **kwargs)

        self.embedding_service = embedding_service
        self.ncf_recommender = ncf_recommender
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

        # Load mood predictor if available
        self.mood_predictor = None
        if mood_predictor_path and Path(mood_predictor_path).exists():
            try:
                with open(mood_predictor_path, 'rb') as f:
                    data = pickle.load(f)
                self.mood_predictor = data
                logger.info(f"✅ Loaded mood predictor from {mood_predictor_path}")
            except Exception as e:
                logger.warning(f"Failed to load mood predictor: {e}")

    def _get_or_predict_audio_features(self, track: Dict[str, Any]) -> Dict[str, float]:
        """
        Get audio features for a track, predicting them if not available.

        Args:
            track: Track dictionary with optional audio_features and i2v_embedding

        Returns:
            Dict with valence, energy, acousticness, danceability
        """
        # Check if track already has audio features
        audio_features = track.get("audio_features", {})
        if audio_features and "valence" in audio_features:
            return audio_features

        # Try to predict from item2vec embedding if mood predictor is available
        if self.mood_predictor and "i2v_embedding" in track:
            try:
                model = self.mood_predictor.get('model')
                feature_names = self.mood_predictor.get('feature_names', ['valence', 'energy', 'acousticness', 'danceability'])

                if model is not None:
                    import numpy as np
                    embedding = track["i2v_embedding"]

                    # Reshape for prediction
                    X = np.array(embedding).reshape(1, -1)
                    predictions = model.predict(X)[0]

                    # Clip to valid range [0, 1]
                    predictions = np.clip(predictions, 0, 1)

                    # Build predicted features dict
                    predicted_features = {
                        name: float(pred)
                        for name, pred in zip(feature_names, predictions)
                    }

                    return predicted_features
            except Exception as e:
                logger.debug(f"Failed to predict audio features: {e}")

        # Return defaults if no features available
        return {
            'valence': 0.5,
            'energy': 0.5,
            'acousticness': 0.5,
            'danceability': 0.5
        }

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
                # Fetch ALL tracks for maximum diversity after filtering
                playlist_neighbors = self.embedding_service.get_playlist_neighbors(
                    track_ids=playlist_track_ids,
                    k=333,  # Get all tracks from item2vec
                )
                candidate_ids.update(playlist_neighbors)
                logger.info(f"Retrieved {len(playlist_neighbors)} playlist neighbors")

        # Strategy 3: Catalogue filtering by context
        context_candidates = self._get_context_filtered_candidates(
            normalised_context, limit=min(500, limit // 2)
        )
        candidate_ids.update(c["id"] for c in context_candidates if "id" in c)
        logger.info(f"Retrieved {len(context_candidates)} context-filtered candidates")

        # Remove playlist tracks (convert to set for efficient filtering)
        playlist_track_ids_set = set(user_profile.get("track_ids", []))
        candidate_ids -= playlist_track_ids_set
        logger.info(f"After filtering playlist tracks: {len(candidate_ids)} candidates remain")

        # Lookup full track data (or create stubs for ML tracks not in catalogue)
        candidates = []
        for track_id in candidate_ids:
            track = self.catalogue.get_by_id(track_id)

            if not track:
                # Create minimal track stub for ML-known tracks not in catalogue
                track = {
                    "id": track_id,
                    "name": track_id,
                    "artist": f"Artist_{hash(track_id) % 10000}",
                    "popularity": 50,
                    "audio_features": {},
                    "tags": {"genres": [], "moods": []},
                }

            # Add item2vec embedding if available
            if self.embedding_service:
                emb = self.embedding_service.item2vec.vector(track_id)
                if emb is not None:
                    track["i2v_embedding"] = emb

            candidates.append(track)

        logger.info(f"Final candidates: {len(candidates)} (with embeddings: {sum(1 for c in candidates if 'i2v_embedding' in c)})")

        # IMPORTANT: Return both candidates AND their track IDs for Spotify enrichment
        # This allows the service layer to fetch real metadata before diversity
        return candidates

    def _get_context_filtered_candidates(
        self,
        normalised_context: Dict[str, Any],
        limit: int = 500,
    ) -> List[Dict[str, Any]]:
        """Get candidates filtered by context tags (from catalogue)."""
        # Filter by both genre AND audio features (mood/vibe)
        all_tracks = self.catalogue.all_tracks()

        # Get target audio profile from user_profile
        target_audio = user_profile.get("audio_features", {})

        # Filter by genres if specified
        genres = normalised_context.get("genres", [])
        adjacent_genres = normalised_context.get("adjacent_genres", [])
        genre_filters = genres or adjacent_genres

        if genre_filters:
            filtered = [
                track for track in all_tracks
                if any(
                    genre in track.get("tags", {}).get("genres", [])
                    for genre in genre_filters
                )
            ]
        else:
            filtered = all_tracks

        # ENHANCED: Filter by audio features (mood/vibe) if we have target audio
        if target_audio and filtered:
            # Calculate audio similarity for each track
            scored_tracks = []
            for track in filtered:
                # Get or predict audio features
                track_audio = self._get_or_predict_audio_features(track)

                # Calculate mood similarity (focus on valence, energy, acousticness)
                mood_score = self._calculate_mood_similarity(track_audio, target_audio)
                scored_tracks.append((track, mood_score))

            # Sort by mood similarity (descending) and take top candidates
            scored_tracks.sort(key=lambda x: x[1], reverse=True)

            # Blend mood similarity (60%) with popularity (40%)
            final_scored = []
            for track, mood_score in scored_tracks:
                popularity_score = track.get("popularity", 50) / 100.0
                blended_score = 0.6 * mood_score + 0.4 * popularity_score
                final_scored.append((track, blended_score))

            final_scored.sort(key=lambda x: x[1], reverse=True)
            filtered = [track for track, _ in final_scored[:limit]]
        else:
            # No audio profile - just sort by popularity
            filtered = sorted(filtered, key=lambda t: t.get("popularity", 0), reverse=True)
            filtered = filtered[:limit]

        return filtered

    def _calculate_mood_similarity(self, track_audio: Dict, target_audio: Dict) -> float:
        """
        Calculate mood similarity between track and target audio features.

        Focuses on key mood indicators:
        - Valence (happiness/sadness): HIGH WEIGHT
        - Energy (intensity): HIGH WEIGHT
        - Acousticness (intimacy): MEDIUM WEIGHT
        - Danceability: LOW WEIGHT
        """
        # Extract features with defaults
        track_valence = track_audio.get("valence", 0.5)
        track_energy = track_audio.get("energy", 0.5)
        track_acoustic = track_audio.get("acousticness", 0.5)
        track_dance = track_audio.get("danceability", 0.5)

        target_valence = target_audio.get("valence", 0.5)
        target_energy = target_audio.get("energy", 0.5)
        target_acoustic = target_audio.get("acousticness", 0.5)
        target_dance = target_audio.get("danceability", 0.5)

        # Calculate weighted differences (smaller = more similar)
        valence_diff = abs(track_valence - target_valence) * 2.0  # Most important for mood
        energy_diff = abs(track_energy - target_energy) * 1.5     # Very important
        acoustic_diff = abs(track_acoustic - target_acoustic) * 1.2  # Medium importance
        dance_diff = abs(track_dance - target_dance) * 0.8       # Less important

        # Total weighted difference
        total_diff = valence_diff + energy_diff + acoustic_diff + dance_diff
        max_diff = 2.0 + 1.5 + 1.2 + 0.8  # Sum of weights

        # Convert to similarity score (0-1, higher = more similar)
        similarity = 1.0 - (total_diff / max_diff)

        return max(0.0, min(1.0, similarity))

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
        # Note: max_per_artist is generous because tracks are enriched with real data later
        diverse_candidates = combined_diversity_rerank(
            candidates=candidates,
            query_vec=query_vec,
            lambda_param=0.6,  # 60% relevance, 40% diversity (more diversity)
            k=limit * 3,  # Get 3x more to account for post-enrichment deduplication
            max_per_artist=10,  # Allow more per artist (fake artists will be replaced)
            embedding_key="i2v_embedding",
            score_key="rank_score",
            artist_key="artist",
        )

        # Return extra candidates so service layer can dedupe after enrichment
        return diverse_candidates[:limit * 2]  # 2x the requested amount

    def _apply_diversity_simple(
        self,
        candidates: List[Dict[str, Any]],
        limit: int,
    ) -> List[Dict[str, Any]]:
        """Simple artist diversity (no MMR) - AGGRESSIVE FILTERING."""
        artist_counts = {}
        diverse = []

        for candidate in candidates:
            artist = candidate.get("artist", "Unknown").lower()  # Use lowercase for consistency

            count = artist_counts.get(artist, 0)

            # AGGRESSIVE: Only 1 track per artist for maximum variety
            if count < 1:
                diverse.append(candidate)
                artist_counts[artist] = count + 1

            if len(diverse) >= limit * 3:  # Get 3x more for post-enrichment deduplication
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

    def _build_playlist_audio_profile(
        self,
        playlist_tracks: List[str],
        seed_track_id: Optional[str] = None,
        seed_weight: float = 0.6
    ) -> Optional[Dict[str, float]]:
        """
        Build average audio profile for playlist tracks using mood predictor.

        If a seed track is provided, blend its mood with the playlist average.
        This gives recommendations that match both the playlist vibe and the specific seed track.

        Args:
            playlist_tracks: List of track IDs in the playlist
            seed_track_id: Optional seed track to emphasize (blended with playlist avg)
            seed_weight: How much to weight seed track (0.0-1.0, default 0.6 = 60% seed, 40% playlist)

        Returns:
            Dict with average valence, energy, acousticness, danceability
        """
        if not self.embedding_service:
            return None

        audio_features = []

        # Get seed track features first if provided
        seed_features = None
        if seed_track_id:
            emb = self.embedding_service.item2vec.vector(seed_track_id)
            if emb is not None:
                track = {'id': seed_track_id, 'i2v_embedding': emb}
                seed_features = self._get_or_predict_audio_features(track)

        # Get playlist features
        for track_id in playlist_tracks[:50]:  # Sample up to 50 tracks to avoid slowness
            # Create track dict with embedding
            emb = self.embedding_service.item2vec.vector(track_id)
            if emb is None:
                continue

            track = {'id': track_id, 'i2v_embedding': emb}

            # Get or predict audio features
            features = self._get_or_predict_audio_features(track)
            audio_features.append(features)

        if not audio_features:
            return None

        # Compute playlist average
        import numpy as np
        playlist_avg = {
            'valence': float(np.mean([f['valence'] for f in audio_features])),
            'energy': float(np.mean([f['energy'] for f in audio_features])),
            'acousticness': float(np.mean([f['acousticness'] for f in audio_features])),
            'danceability': float(np.mean([f['danceability'] for f in audio_features])),
        }

        # Blend with seed track if provided (60% seed, 40% playlist by default)
        if seed_features:
            blended = {
                'valence': seed_weight * seed_features['valence'] + (1 - seed_weight) * playlist_avg['valence'],
                'energy': seed_weight * seed_features['energy'] + (1 - seed_weight) * playlist_avg['energy'],
                'acousticness': seed_weight * seed_features['acousticness'] + (1 - seed_weight) * playlist_avg['acousticness'],
                'danceability': seed_weight * seed_features['danceability'] + (1 - seed_weight) * playlist_avg['danceability'],
            }
            return blended

        return playlist_avg

    def _filter_candidates_by_mood(
        self,
        candidates: List[Dict[str, Any]],
        target_audio: Dict[str, float],
        keep_top_pct: float = 0.4
    ) -> List[Dict[str, Any]]:
        """
        Filter candidates by mood similarity to target audio profile.

        Args:
            candidates: List of candidate tracks
            target_audio: Target audio profile (playlist average)
            keep_top_pct: Percentage of candidates to keep (0.0-1.0)

        Returns:
            Filtered list of candidates with mood_similarity scores attached
        """
        # Score each candidate by mood similarity
        scored_candidates = []

        for candidate in candidates:
            # Get or predict audio features
            candidate_audio = self._get_or_predict_audio_features(candidate)

            # Calculate mood similarity
            mood_score = self._calculate_mood_similarity(candidate_audio, target_audio)

            # Attach mood score to candidate for ranker to use
            candidate['mood_similarity'] = mood_score

            scored_candidates.append((candidate, mood_score))

        # Sort by mood similarity (descending)
        scored_candidates.sort(key=lambda x: x[1], reverse=True)

        # Keep top percentage
        keep_count = max(1, int(len(scored_candidates) * keep_top_pct))
        filtered = [candidate for candidate, score in scored_candidates[:keep_count]]

        # Log mood range of filtered candidates
        if filtered:
            min_mood = min(c['mood_similarity'] for c in filtered)
            max_mood = max(c['mood_similarity'] for c in filtered)
            logger.info(f"Mood similarity range: {min_mood:.3f} to {max_mood:.3f}")

        return filtered

    def recommend(
        self,
        playlist_tracks: List[str],
        seed_track_id: Optional[str] = None,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """
        Simplified sync method for playlist continuation (for evaluation).

        Args:
            playlist_tracks: List of track IDs in the playlist
            seed_track_id: Optional seed track to emphasize in mood filtering
            limit: Number of recommendations to return

        Returns:
            List of recommended tracks
        """
        if not playlist_tracks:
            logger.warning("No playlist tracks provided")
            return []

        # Stage 1: Retrieve candidates using item2vec with mood filtering
        candidates = self._retrieve_candidates_simple(
            playlist_tracks,
            seed_track_id=seed_track_id,
            limit=limit * 50
        )
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
        seed_track_id: Optional[str] = None,
        limit: int = 1000,
    ) -> List[Dict[str, Any]]:
        """Simple candidate retrieval for evaluation with mood filtering."""
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

        # Build playlist audio profile for mood filtering (blended with seed if provided)
        playlist_audio_profile = self._build_playlist_audio_profile(
            playlist_tracks,
            seed_track_id=seed_track_id,
            seed_weight=0.6  # 60% seed track, 40% playlist
        )
        if playlist_audio_profile:
            seed_indicator = " (with seed blend)" if seed_track_id else ""
            logger.info(f"Target mood{seed_indicator}: V={playlist_audio_profile.get('valence', 0.5):.2f} "
                       f"E={playlist_audio_profile.get('energy', 0.5):.2f} "
                       f"A={playlist_audio_profile.get('acousticness', 0.5):.2f}")

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

        # CRITICAL FIX: Apply artist diversity BEFORE mood filtering!
        # Otherwise mood filtering can create a feedback loop where one artist dominates
        if len(candidates) > 100:  # Always apply if we have enough candidates
            logger.info(f"Pre-filtering for artist diversity on {len(candidates)} candidates...")
            candidates = self._apply_diversity_simple(candidates, limit=len(candidates))
            logger.info(f"After artist pre-filtering: {len(candidates)} candidates (max 1 per artist)")

        # Apply mood-based filtering if we have a playlist audio profile
        if playlist_audio_profile and len(candidates) > limit // 2:
            logger.info(f"Applying mood filtering to {len(candidates)} candidates...")
            candidates = self._filter_candidates_by_mood(
                candidates,
                playlist_audio_profile,
                keep_top_pct=0.7  # Keep top 70% by mood (less aggressive to preserve artist variety)
            )
            logger.info(f"After mood filtering: {len(candidates)} candidates remain")

        return candidates

    def _rank_candidates_simple(
        self,
        candidates: List[Dict[str, Any]],
        playlist_tracks: List[str],
    ) -> List[Dict[str, Any]]:
        """Simple ranking for evaluation with optional NCF blending."""
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

        # NOTE: NCF disabled in simple path - it's designed for playlist→track, not track→track
        # The recommend_for_tracks method incorrectly uses track IDs as playlist IDs
        # causing "index out of range" errors. NCF is only used in the full recommendation path.

        # Rank using ranker
        ranked = self.ranker.rank_candidates(
            candidates=candidates,
            feature_extractor=self.feature_extractor,
            playlist_profile=playlist_profile,
            seed_track=None,
        )

        return ranked
