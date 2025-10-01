"""Audio feature-based content similarity using Spotify's audio analysis."""

import logging
import math
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class AudioSimilarityEngine:
    """Content-based filtering using audio features like danceability, energy, valence."""

    # Feature weights based on research showing their importance for similarity
    FEATURE_WEIGHTS = {
        "valence": 1.2,  # Mood is very important
        "energy": 1.1,
        "danceability": 1.0,
        "acousticness": 0.9,
        "instrumentalness": 0.8,
        "tempo": 0.6,  # Normalized tempo has lower weight
        "loudness": 0.4,
        "speechiness": 0.7,
    }

    def calculate_similarity(self, features1: Dict, features2: Dict) -> float:
        """
        Calculate similarity between two tracks based on audio features.
        Returns similarity score between 0.0 (completely different) and 1.0 (identical).
        """
        if not features1 or not features2:
            return 0.0

        total_distance = 0.0
        total_weight = 0.0

        # Calculate weighted Euclidean distance for each feature
        for feature, weight in self.FEATURE_WEIGHTS.items():
            val1 = features1.get(feature)
            val2 = features2.get(feature)

            if val1 is None or val2 is None:
                continue

            # Normalize tempo (BPM) to 0-1 range
            if feature == "tempo":
                val1 = self._normalize_tempo(val1)
                val2 = self._normalize_tempo(val2)
            # Normalize loudness (dB) to 0-1 range
            elif feature == "loudness":
                val1 = self._normalize_loudness(val1)
                val2 = self._normalize_loudness(val2)

            # Squared difference weighted by importance
            distance = (val1 - val2) ** 2 * weight
            total_distance += distance
            total_weight += weight

        if total_weight == 0:
            return 0.0

        # Convert distance to similarity score
        # Using exponential decay for more natural similarity distribution
        normalized_distance = math.sqrt(total_distance / total_weight)
        similarity = math.exp(-2.0 * normalized_distance)

        return similarity

    def find_similar_tracks(self, seed_features: Dict, candidates: List[Dict],
                           limit: int = 20) -> List[Tuple[Dict, float]]:
        """
        Find tracks similar to a seed track based on audio features.
        Returns list of (track, similarity_score) tuples, sorted by similarity.
        """
        if not seed_features:
            return []

        similarities = []
        for candidate in candidates:
            candidate_features = candidate.get("audio_features") or candidate.get("features")
            if not candidate_features:
                continue

            similarity = self.calculate_similarity(seed_features, candidate_features)
            if similarity > 0.0:
                similarities.append((candidate, similarity))

        # Sort by similarity score (highest first)
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:limit]

    def calculate_user_audio_profile(self, user_tracks: List[Dict]) -> Dict:
        """
        Build an audio feature profile for a user based on their listening history.
        Returns average feature values representing the user's taste.
        """
        if not user_tracks:
            return {}

        feature_sums = {}
        feature_counts = {}

        for track in user_tracks:
            features = track.get("audio_features") or track.get("features")
            if not features:
                continue

            for feature in self.FEATURE_WEIGHTS.keys():
                value = features.get(feature)
                if value is not None:
                    if feature not in feature_sums:
                        feature_sums[feature] = 0.0
                        feature_counts[feature] = 0

                    feature_sums[feature] += value
                    feature_counts[feature] += 1

        # Calculate averages
        user_profile = {}
        for feature, total in feature_sums.items():
            if feature_counts[feature] > 0:
                user_profile[feature] = total / feature_counts[feature]

        return user_profile

    def match_to_user_profile(self, user_profile: Dict, track_features: Dict) -> float:
        """
        Calculate how well a track matches a user's audio preference profile.
        Returns match score between 0.0 and 1.0.
        """
        if not user_profile or not track_features:
            return 0.5  # Neutral score if no data

        return self.calculate_similarity(user_profile, track_features)

    def categorize_track(self, features: Dict) -> Dict[str, str]:
        """
        Categorize track based on audio features for explainability.
        Returns human-readable descriptions of the track's characteristics.
        """
        if not features:
            return {}

        categories = {}

        # Energy level
        energy = features.get("energy", 0.5)
        if energy > 0.8:
            categories["energy"] = "high-energy"
        elif energy < 0.3:
            categories["energy"] = "calm"
        else:
            categories["energy"] = "moderate-energy"

        # Mood (valence)
        valence = features.get("valence", 0.5)
        if valence > 0.7:
            categories["mood"] = "upbeat"
        elif valence < 0.3:
            categories["mood"] = "melancholic"
        else:
            categories["mood"] = "neutral"

        # Danceability
        dance = features.get("danceability", 0.5)
        if dance > 0.7:
            categories["dance"] = "very danceable"
        elif dance < 0.4:
            categories["dance"] = "not danceable"

        # Acousticness
        acoustic = features.get("acousticness", 0.5)
        if acoustic > 0.7:
            categories["style"] = "acoustic"
        elif acoustic < 0.3:
            categories["style"] = "electronic"

        # Tempo
        tempo = features.get("tempo", 120)
        if tempo > 140:
            categories["tempo"] = "fast"
        elif tempo < 80:
            categories["tempo"] = "slow"

        return categories

    @staticmethod
    def _normalize_tempo(bpm: float) -> float:
        """Normalize tempo from BPM to 0-1 range."""
        # Most music is 60-180 BPM
        min_bpm = 60.0
        max_bpm = 180.0
        normalized = (bpm - min_bpm) / (max_bpm - min_bpm)
        return max(0.0, min(1.0, normalized))

    @staticmethod
    def _normalize_loudness(db: float) -> float:
        """Normalize loudness from dB to 0-1 range."""
        # Spotify loudness typically ranges from -60 to 0 dB
        min_db = -60.0
        max_db = 0.0
        normalized = (db - min_db) / (max_db - min_db)
        return max(0.0, min(1.0, normalized))
