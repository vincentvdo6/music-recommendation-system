"""Hybrid recommendation engine combining collaborative filtering and content-based methods."""

import logging
import random
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from services.recommendation.audio_similarity import AudioSimilarityEngine
from services.recommendation.user_profile import UserProfile
from services.spotify.client import SpotifyClient

logger = logging.getLogger(__name__)


class HybridRecommendationEngine:
    """
    Combines multiple recommendation strategies:
    1. Collaborative Filtering - what similar users like
    2. Content-Based - audio feature similarity
    3. Context-Aware - time of day, listening patterns
    """

    def __init__(self, spotify_client: SpotifyClient):
        self.spotify = spotify_client
        self.user_profile = UserProfile()
        self.audio_engine = AudioSimilarityEngine()

        # Weights for hybrid scoring (tuned based on research)
        self.COLLABORATIVE_WEIGHT = 0.4
        self.CONTENT_WEIGHT = 0.4
        self.POPULARITY_WEIGHT = 0.1
        self.DIVERSITY_WEIGHT = 0.1

    async def get_personalized_recommendations(
        self,
        user_id: str,
        seed_track_id: Optional[str] = None,
        seed_features: Optional[Dict] = None,
        limit: int = 20,
        context: Optional[Dict] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get personalized recommendations using hybrid approach.

        Two-stage architecture:
        1. Candidate Generation - gather candidates from multiple sources
        2. Ranking - score and re-rank candidates using hybrid scoring
        """
        # Stage 1: Candidate Generation
        candidates = await self._generate_candidates(
            user_id, seed_track_id, seed_features, context
        )

        if not candidates:
            logger.warning("No candidates generated for user %s", user_id)
            return []

        # Stage 2: Hybrid Scoring & Ranking
        scored_candidates = await self._score_and_rank_candidates(
            user_id, candidates, seed_features, context
        )

        # Add diversity to prevent filter bubble
        final_recommendations = self._add_diversity(scored_candidates, limit)

        return final_recommendations[:limit]

    async def _generate_candidates(
        self,
        user_id: str,
        seed_track_id: Optional[str],
        seed_features: Optional[Dict],
        context: Optional[Dict],
    ) -> List[Dict]:
        """
        Stage 1: Generate candidate tracks from multiple sources.
        Uses retrieval methods that are fast but less precise.
        """
        candidates = []
        seen_ids = set()

        # Source 1: Collaborative Filtering (similar users' tracks)
        collab_candidates = self.user_profile.get_collaborative_recommendations(user_id, limit=30)
        for track_id in collab_candidates:
            if track_id not in seen_ids:
                candidates.append({"id": track_id, "source": "collaborative"})
                seen_ids.add(track_id)

        # Source 2: Content-Based (similar to seed track)
        if seed_track_id and self.spotify and not self.spotify.demo_mode:
            try:
                spotify_recs = await self.spotify.get_recommendations([seed_track_id], limit=20)
                for track in spotify_recs:
                    if track["id"] not in seen_ids:
                        track["source"] = "content_spotify"
                        candidates.append(track)
                        seen_ids.add(track["id"])
            except Exception as exc:
                logger.warning("Spotify recommendations failed: %s", exc)

        # Source 3: Popular tracks as fallback
        popular = self.user_profile.get_popular_tracks(limit=20)
        for track_id in popular:
            if track_id not in seen_ids and len(candidates) < 100:
                candidates.append({"id": track_id, "source": "popular"})
                seen_ids.add(track_id)

        logger.info("Generated %d candidates for user %s", len(candidates), user_id)
        return candidates

    async def _score_and_rank_candidates(
        self,
        user_id: str,
        candidates: List[Dict],
        seed_features: Optional[Dict],
        context: Optional[Dict],
    ) -> List[Tuple[Dict, float]]:
        """
        Stage 2: Score candidates using hybrid approach and rank them.
        This is more computationally expensive but only runs on filtered candidates.
        """
        scored = []

        # Get user's audio taste profile
        user_taste_profile = self._get_user_taste_profile(user_id)

        for candidate in candidates:
            # Calculate hybrid score
            score = await self._calculate_hybrid_score(
                user_id, candidate, user_taste_profile, seed_features, context
            )

            if score > 0.0:
                scored.append((candidate, score))

        # Sort by hybrid score (highest first)
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    async def _calculate_hybrid_score(
        self,
        user_id: str,
        candidate: Dict,
        user_taste_profile: Dict,
        seed_features: Optional[Dict],
        context: Optional[Dict],
    ) -> float:
        """
        Calculate hybrid recommendation score combining multiple signals.

        Score components:
        - Collaborative filtering score
        - Content similarity score
        - Popularity score
        - Context match score
        """
        track_id = candidate.get("id")
        if not track_id:
            return 0.0

        total_score = 0.0

        # Component 1: Collaborative Filtering Score
        collab_score = self._get_collaborative_score(user_id, track_id)
        total_score += collab_score * self.COLLABORATIVE_WEIGHT

        # Component 2: Content-Based Score
        content_score = await self._get_content_score(
            candidate, user_taste_profile, seed_features
        )
        total_score += content_score * self.CONTENT_WEIGHT

        # Component 3: Popularity Score (avoid too obscure tracks)
        popularity_score = self._get_popularity_score(track_id, candidate)
        total_score += popularity_score * self.POPULARITY_WEIGHT

        # Component 4: Context Match (time of day, mood)
        if context:
            context_score = self._get_context_score(candidate, context)
            total_score += context_score * 0.1

        return total_score

    def _get_collaborative_score(self, user_id: str, track_id: str) -> float:
        """Score based on collaborative filtering signals."""
        # Check if similar users liked this track
        similar_users = self.user_profile.get_similar_users(user_id, limit=20)
        if not similar_users:
            return 0.0

        score = 0.0
        for similar_user_id, similarity in similar_users:
            # Check if this similar user played the track
            user_tracks = {p["track_id"] for p in self.user_profile.user_plays[similar_user_id]}
            if track_id in user_tracks:
                score += similarity

        # Normalize by number of similar users
        if similar_users:
            score /= len(similar_users)

        return min(1.0, score)

    async def _get_content_score(
        self,
        candidate: Dict,
        user_taste_profile: Dict,
        seed_features: Optional[Dict],
    ) -> float:
        """Score based on audio feature similarity."""
        candidate_features = candidate.get("audio_features")

        # Fetch features if not present
        if not candidate_features and self.spotify and not self.spotify.demo_mode:
            try:
                candidate_features = await self.spotify.get_track_features(candidate["id"])
                if candidate_features:
                    candidate["audio_features"] = candidate_features
            except Exception:
                pass

        if not candidate_features:
            return 0.5  # Neutral score if no features

        score = 0.0
        count = 0

        # Match to user's taste profile
        if user_taste_profile:
            taste_score = self.audio_engine.match_to_user_profile(
                user_taste_profile, candidate_features
            )
            score += taste_score
            count += 1

        # Match to seed track
        if seed_features:
            seed_score = self.audio_engine.calculate_similarity(
                seed_features, candidate_features
            )
            score += seed_score
            count += 1

        return score / count if count > 0 else 0.5

    def _get_popularity_score(self, track_id: str, candidate: Dict) -> float:
        """Score based on track popularity."""
        # Use platform popularity if available
        popularity = candidate.get("popularity", 50)
        normalized = popularity / 100.0

        # Also consider play count in our system
        play_count = self.user_profile.track_plays.get(track_id, 0)
        if play_count > 0:
            # Logarithmic scaling for play count
            import math
            play_score = min(1.0, math.log10(play_count + 1) / 2.0)
            normalized = (normalized + play_score) / 2.0

        return normalized

    def _get_context_score(self, candidate: Dict, context: Dict) -> float:
        """Score based on contextual match (time, mood, etc.)."""
        features = candidate.get("audio_features")
        if not features:
            return 0.5

        score = 0.0
        count = 0

        # Time-based matching
        time_of_day = context.get("time_of_day")
        if time_of_day:
            energy = features.get("energy", 0.5)

            # Morning: prefer energetic
            if time_of_day == "morning" and energy > 0.6:
                score += 0.8
                count += 1
            # Night: prefer calmer
            elif time_of_day == "night" and energy < 0.5:
                score += 0.8
                count += 1
            # Evening: moderate energy
            elif time_of_day == "evening" and 0.4 < energy < 0.7:
                score += 0.7
                count += 1

        # Mood-based matching
        desired_mood = context.get("mood")
        if desired_mood:
            valence = features.get("valence", 0.5)

            if desired_mood == "upbeat" and valence > 0.6:
                score += 0.9
                count += 1
            elif desired_mood == "calm" and valence < 0.4:
                score += 0.9
                count += 1

        return score / count if count > 0 else 0.5

    def _get_user_taste_profile(self, user_id: str) -> Dict:
        """Build user's audio taste profile from listening history."""
        plays = self.user_profile.user_plays.get(user_id, [])
        if not plays:
            return {}

        # Get recent plays with features
        recent_tracks = []
        for play in plays[-30:]:  # Last 30 tracks
            features = play.get("audio_features")
            if features:
                recent_tracks.append({"audio_features": features})

        if not recent_tracks:
            return {}

        return self.audio_engine.calculate_user_audio_profile(recent_tracks)

    def _add_diversity(
        self,
        scored_candidates: List[Tuple[Dict, float]],
        limit: int,
    ) -> List[Dict]:
        """
        Add diversity to recommendations to prevent filter bubble.
        Uses a greedy algorithm to balance relevance and diversity.
        """
        if not scored_candidates:
            return []

        final = []
        remaining = scored_candidates.copy()

        # First, take top-scoring candidate
        if remaining:
            top_candidate, top_score = remaining.pop(0)
            final.append(top_candidate)

        # Then alternate between high-scoring and diverse picks
        while len(final) < limit and remaining:
            # 70% chance: take next highest scoring
            # 30% chance: take a diverse pick from middle range
            if random.random() < 0.7 or len(final) < 3:
                candidate, score = remaining.pop(0)
            else:
                # Pick from middle range for diversity
                diverse_idx = min(len(remaining) - 1, random.randint(3, min(10, len(remaining) - 1)))
                candidate, score = remaining.pop(diverse_idx)

            final.append(candidate)

        return final

    def track_interaction(
        self,
        user_id: str,
        track_id: str,
        interaction_type: str,
        track_data: Optional[Dict] = None,
    ) -> None:
        """Track user interactions for improving recommendations."""
        if interaction_type == "play":
            self.user_profile.track_play(user_id, track_id, track_data or {})
        elif interaction_type == "like":
            self.user_profile.track_like(user_id, track_id)
        elif interaction_type == "skip":
            self.user_profile.track_skip(user_id, track_id)

    def save_state(self) -> None:
        """Persist recommendation engine state."""
        self.user_profile.save_profiles()
