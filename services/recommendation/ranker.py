"""
Learned ranking model for combining recommendation signals.

Uses LightGBM to learn optimal weights for features like:
- Item2vec similarity
- Audio feature similarity
- Popularity
- Genre overlap
- Artist match
- Era gap

Trained on playlist continuation task with held-out tracks.
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional, Set

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """Extract features for ranking candidates."""

    def __init__(self, audio_similarity_engine):
        self.audio_sim = audio_similarity_engine

    def extract_features(
        self,
        candidate: Dict[str, Any],
        playlist_profile: Dict[str, Any],
        seed_track: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        """
        Extract ranking features for a candidate track.

        Args:
            candidate: Candidate track dict
            playlist_profile: Aggregated playlist profile
            seed_track: Optional seed track for single-seed recs

        Returns:
            Dictionary of feature values
        """
        features = {}

        # 1. Item2vec similarity (collaborative signal)
        if "i2v_embedding" in candidate and "i2v_embedding" in playlist_profile:
            features["i2v_cosine"] = self._cosine_sim(
                candidate["i2v_embedding"],
                playlist_profile["i2v_embedding"]
            )
        else:
            features["i2v_cosine"] = 0.0

        # 2. Audio feature similarity (content signal)
        if "audio_features" in candidate and "audio_features" in playlist_profile:
            features["audio_similarity"] = self.audio_sim.calculate_similarity(
                candidate["audio_features"],
                playlist_profile["audio_features"]
            )
        else:
            features["audio_similarity"] = 0.0

        # 3. Popularity (normalized 0-1)
        features["popularity"] = candidate.get("popularity", 50) / 100.0

        # 4. Artist match (binary: is artist in playlist?)
        playlist_artists = playlist_profile.get("artists", set())
        candidate_artist = candidate.get("artist", "")
        features["artist_in_playlist"] = float(candidate_artist in playlist_artists)

        # 5. Genre overlap (Jaccard similarity)
        playlist_genres = set(playlist_profile.get("genres", []))
        candidate_genres = set(candidate.get("genres", []))
        features["genre_jaccard"] = self._jaccard(playlist_genres, candidate_genres)

        # 6. Era gap (normalized)
        playlist_year = playlist_profile.get("avg_year", 2020)
        candidate_year = candidate.get("release_year", 2020)
        features["era_gap"] = abs(candidate_year - playlist_year) / 50.0  # normalize

        # 7. Seed similarity (if seed provided)
        if seed_track:
            if "i2v_embedding" in candidate and "i2v_embedding" in seed_track:
                features["seed_i2v_cosine"] = self._cosine_sim(
                    candidate["i2v_embedding"],
                    seed_track["i2v_embedding"]
                )
            else:
                features["seed_i2v_cosine"] = 0.0

            if "audio_features" in candidate and "audio_features" in seed_track:
                features["seed_audio_sim"] = self.audio_sim.calculate_similarity(
                    candidate["audio_features"],
                    seed_track["audio_features"]
                )
            else:
                features["seed_audio_sim"] = 0.0
        else:
            features["seed_i2v_cosine"] = 0.0
            features["seed_audio_sim"] = 0.0

        # 8. Valence/energy similarity (key mood features)
        if "audio_features" in candidate and "audio_features" in playlist_profile:
            cand_af = candidate["audio_features"]
            prof_af = playlist_profile["audio_features"]

            features["valence_diff"] = abs(
                cand_af.get("valence", 0.5) - prof_af.get("valence", 0.5)
            )
            features["energy_diff"] = abs(
                cand_af.get("energy", 0.5) - prof_af.get("energy", 0.5)
            )
        else:
            features["valence_diff"] = 0.5
            features["energy_diff"] = 0.5

        return features

    @staticmethod
    def _cosine_sim(vec1, vec2) -> float:
        """Cosine similarity between two vectors."""
        if vec1 is None or vec2 is None:
            return 0.0

        vec1 = np.array(vec1)
        vec2 = np.array(vec2)

        dot = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot / (norm1 * norm2))

    @staticmethod
    def _jaccard(set1: Set, set2: Set) -> float:
        """Jaccard similarity between two sets."""
        if not set1 and not set2:
            return 0.0

        intersection = len(set1 & set2)
        union = len(set1 | set2)

        if union == 0:
            return 0.0

        return intersection / union


class LearnedRanker:
    """LightGBM ranker for combining recommendation signals."""

    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.feature_names = None

        if model_path and Path(model_path).exists():
            self.load(model_path)

    def load(self, model_path: str):
        """Load trained LightGBM model."""
        try:
            import lightgbm as lgb
            self.model = lgb.Booster(model_file=model_path)
            self.feature_names = self.model.feature_name()
            logger.info(f"Loaded ranker from {model_path}")
            logger.info(f"Features: {self.feature_names}")
        except ImportError:
            logger.error("LightGBM not installed. Run: pip install lightgbm")
            raise
        except Exception as e:
            logger.error(f"Failed to load ranker: {e}")
            raise

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """
        Predict ranking scores for candidates.

        Args:
            features: DataFrame with feature columns

        Returns:
            Array of scores (higher = better)
        """
        if self.model is None:
            logger.warning("No model loaded, returning zeros")
            return np.zeros(len(features))

        # Ensure features are in correct order
        if self.feature_names:
            missing = set(self.feature_names) - set(features.columns)
            if missing:
                logger.warning(f"Missing features: {missing}, filling with zeros")
                for feat in missing:
                    features[feat] = 0.0

            features = features[self.feature_names]

        return self.model.predict(features)

    def rank_candidates(
        self,
        candidates: List[Dict[str, Any]],
        feature_extractor: FeatureExtractor,
        playlist_profile: Dict[str, Any],
        seed_track: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Rank candidates using learned model.

        Args:
            candidates: List of candidate tracks
            feature_extractor: FeatureExtractor instance
            playlist_profile: Aggregated playlist profile
            seed_track: Optional seed track

        Returns:
            Candidates sorted by rank score (descending)
        """
        if not candidates:
            return []

        # Extract features for all candidates
        features_list = []
        for candidate in candidates:
            feats = feature_extractor.extract_features(
                candidate, playlist_profile, seed_track
            )
            features_list.append(feats)

        features_df = pd.DataFrame(features_list)

        # Predict scores
        scores = self.predict(features_df)

        # Attach scores to candidates
        for candidate, score in zip(candidates, scores):
            candidate["rank_score"] = float(score)

        # Sort by score descending
        ranked = sorted(candidates, key=lambda c: c["rank_score"], reverse=True)

        return ranked


class SimpleRanker:
    """
    Simple weighted ranker (fallback when no learned model).

    Uses hand-tuned weights for combining signals.
    """

    DEFAULT_WEIGHTS = {
        "i2v_cosine": 0.35,
        "audio_similarity": 0.30,
        "popularity": 0.15,
        "genre_jaccard": 0.10,
        "seed_i2v_cosine": 0.10,
    }

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.weights = weights or self.DEFAULT_WEIGHTS

    def rank_candidates(
        self,
        candidates: List[Dict[str, Any]],
        feature_extractor: FeatureExtractor,
        playlist_profile: Dict[str, Any],
        seed_track: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Rank candidates using weighted feature combination.

        Args:
            candidates: List of candidate tracks
            feature_extractor: FeatureExtractor instance
            playlist_profile: Aggregated playlist profile
            seed_track: Optional seed track

        Returns:
            Candidates sorted by rank score (descending)
        """
        if not candidates:
            return []

        # Extract features and compute weighted score
        for candidate in candidates:
            feats = feature_extractor.extract_features(
                candidate, playlist_profile, seed_track
            )

            # Weighted sum
            score = sum(
                feats.get(feat, 0.0) * weight
                for feat, weight in self.weights.items()
            )

            candidate["rank_score"] = float(score)

        # Sort by score descending
        ranked = sorted(candidates, key=lambda c: c["rank_score"], reverse=True)

        return ranked
