"""
Train LightGBM ranking model on playlist continuation task.

Generates training data from held-out playlist tracks and trains
a learning-to-rank model using LambdaMART.

Based on RecSys 2018 Automatic Playlist Continuation task.
"""

import json
import logging
import random
import os
import sys
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RankerTrainingDataGenerator:
    """Generate training data for ranker from playlist corpus."""

    def __init__(
        self,
        item2vec_service,
        catalogue,
        audio_sim_engine,
        min_playlist_length: int = 10,
        n_negatives_per_positive: int = 10,
    ):
        self.item2vec = item2vec_service
        self.catalogue = catalogue
        self.audio_sim = audio_sim_engine
        self.min_playlist_length = min_playlist_length
        self.n_negatives = n_negatives_per_positive

    def generate_training_examples(
        self,
        playlists: List[List[str]],
        max_examples: int = 50000,
    ) -> Tuple[pd.DataFrame, List[int]]:
        """
        Generate training examples from playlist continuation task.

        For each playlist:
        1. Split into train (first 60-80%) and test (last 20-40%)
        2. Build profile from train tracks
        3. Held-out test tracks are positives (label=1)
        4. Random tracks (not in playlist) are negatives (label=0)

        Returns:
            features_df: DataFrame with features for each example
            groups: List of group sizes for listwise ranking
        """
        logger.info(f"Generating training data from {len(playlists):,} playlists")

        all_features = []
        all_labels = []
        groups = []

        n_processed = 0

        for playlist in playlists:
            if len(playlist) < self.min_playlist_length:
                continue

            # Split playlist
            split_idx = int(len(playlist) * 0.7)  # 70% train, 30% test
            train_tracks = playlist[:split_idx]
            test_tracks = playlist[split_idx:]

            if not test_tracks:
                continue

            # Build playlist profile
            profile = self._build_profile(train_tracks)

            # Track actual examples added for this group
            group_start = len(all_features)

            # Generate positive examples (held-out tracks)
            for test_track in test_tracks:
                track_data = self._get_track_data(test_track)
                if track_data is None:
                    continue

                features = self._extract_features(track_data, profile)
                all_features.append(features)
                all_labels.append(1)  # Positive

            n_positives = len(test_tracks)

            # Generate negative examples (random tracks not in playlist)
            negatives = self._sample_negatives(
                playlist, n_samples=n_positives * self.n_negatives
            )

            for neg_track in negatives:
                track_data = self._get_track_data(neg_track)
                if track_data is None:
                    continue

                features = self._extract_features(track_data, profile)
                all_features.append(features)
                all_labels.append(0)  # Negative

            # Record ACTUAL group size for this playlist (for LambdaMART)
            group_size = len(all_features) - group_start
            if group_size > 0:  # Only add if we actually added examples
                groups.append(group_size)

            n_processed += 1

            if n_processed >= max_examples:
                break

            if n_processed % 1000 == 0:
                logger.info(f"Processed {n_processed} playlists")

        logger.info(f"Generated {len(all_features):,} training examples")
        logger.info(f"  Positives: {sum(all_labels):,}")
        logger.info(f"  Negatives: {len(all_labels) - sum(all_labels):,}")

        features_df = pd.DataFrame(all_features)
        features_df['label'] = all_labels

        return features_df, groups

    def _build_profile(self, track_ids: List[str]) -> Dict:
        """Build aggregated profile from playlist tracks."""
        # Get embeddings
        i2v_vec = self.item2vec.mean_vector(track_ids)

        # Get audio features (mock for now - integrate with actual catalogue)
        # In practice, fetch from Spotify or catalogue
        audio_features = {}

        # Get artists and genres
        artists = set()
        genres = set()
        years = []

        for track_id in track_ids:
            track_data = self._get_track_data(track_id)
            if track_data:
                artists.add(track_data.get("artist", ""))
                genres.update(track_data.get("genres", []))
                if "release_year" in track_data:
                    years.append(track_data["release_year"])

        avg_year = int(np.mean(years)) if years else 2020

        return {
            "i2v_embedding": i2v_vec,
            "audio_features": audio_features,
            "artists": artists,
            "genres": genres,
            "avg_year": avg_year,
        }

    def _get_track_data(self, track_id: str) -> Dict:
        """Get track data (from catalogue or mock)."""
        # Try catalogue first
        track = self.catalogue.get_by_id(track_id)

        if track:
            return {
                "id": track_id,
                "artist": track.get("artist", ""),
                "genres": track.get("tags", {}).get("genres", []),
                "release_year": track.get("release_year", 2020),
                "popularity": track.get("popularity", 50),
                "audio_features": track.get("audio_features", {}),
                "i2v_embedding": self.item2vec.vector(track_id),
            }

        # Fallback: item2vec only
        if self.item2vec.has_vector(track_id):
            return {
                "id": track_id,
                "artist": "",
                "genres": [],
                "release_year": 2020,
                "popularity": 50,
                "audio_features": {},
                "i2v_embedding": self.item2vec.vector(track_id),
            }

        return None

    def _extract_features(self, track_data: Dict, profile: Dict) -> Dict:
        """Extract features for a candidate track."""
        features = {}

        # Item2vec cosine
        if track_data.get("i2v_embedding") is not None and profile.get("i2v_embedding") is not None:
            features["i2v_cosine"] = self._cosine_sim(
                track_data["i2v_embedding"],
                profile["i2v_embedding"]
            )
        else:
            features["i2v_cosine"] = 0.0

        # Audio similarity (placeholder)
        features["audio_similarity"] = 0.0

        # Popularity
        features["popularity"] = track_data.get("popularity", 50) / 100.0

        # Artist match
        features["artist_in_playlist"] = float(
            track_data.get("artist", "") in profile.get("artists", set())
        )

        # Genre overlap
        track_genres = set(track_data.get("genres", []))
        profile_genres = set(profile.get("genres", []))
        features["genre_jaccard"] = self._jaccard(track_genres, profile_genres)

        # Era gap
        features["era_gap"] = abs(
            track_data.get("release_year", 2020) - profile.get("avg_year", 2020)
        ) / 50.0

        # Seed similarity (set to 0 for playlist task)
        features["seed_i2v_cosine"] = 0.0
        features["seed_audio_sim"] = 0.0

        # Valence/energy diff (placeholder)
        features["valence_diff"] = 0.5
        features["energy_diff"] = 0.5

        return features

    def _sample_negatives(self, playlist: List[str], n_samples: int) -> List[str]:
        """Sample negative tracks not in playlist."""
        playlist_set = set(playlist)

        # Get all tracks from vocabulary
        all_tracks = list(self.item2vec.wv.index_to_key)

        # Filter out playlist tracks
        candidates = [t for t in all_tracks if t not in playlist_set]

        # Sample negatives
        n_samples = min(n_samples, len(candidates))
        return random.sample(candidates, n_samples)

    @staticmethod
    def _cosine_sim(vec1, vec2) -> float:
        """Cosine similarity."""
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
    def _jaccard(set1, set2) -> float:
        """Jaccard similarity."""
        if not set1 and not set2:
            return 0.0

        intersection = len(set1 & set2)
        union = len(set1 | set2)

        return intersection / union if union > 0 else 0.0


def train_lightgbm_ranker(
    features_df: pd.DataFrame,
    groups: List[int],
    output_path: str,
):
    """
    Train LightGBM LambdaMART ranker.

    Args:
        features_df: DataFrame with features and 'label' column
        groups: List of group sizes for listwise ranking
        output_path: Path to save trained model
    """
    try:
        import lightgbm as lgb
    except ImportError:
        logger.error("LightGBM not installed. Run: pip install lightgbm")
        return

    logger.info("Training LightGBM ranker")

    # Split features and labels
    y = features_df['label'].values
    X = features_df.drop(columns=['label'])

    # Train-test split (by playlist groups)
    split_idx = int(len(groups) * 0.8)
    train_groups = groups[:split_idx]
    test_groups = groups[split_idx:]

    train_size = sum(train_groups)
    test_size = sum(test_groups)

    X_train = X.iloc[:train_size]
    y_train = y[:train_size]

    X_test = X.iloc[train_size:train_size + test_size]
    y_test = y[train_size:train_size + test_size]

    logger.info(f"Train size: {len(X_train):,} ({len(train_groups)} playlists)")
    logger.info(f"Test size: {len(X_test):,} ({len(test_groups)} playlists)")

    # Train LightGBM ranker
    train_data = lgb.Dataset(X_train, label=y_train, group=train_groups)
    test_data = lgb.Dataset(X_test, label=y_test, group=test_groups, reference=train_data)

    params = {
        'objective': 'lambdarank',
        'metric': 'ndcg',
        'ndcg_eval_at': [5, 10, 30],
        'num_leaves': 31,
        'learning_rate': 0.1,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': 1,
    }

    logger.info("Training model...")
    model = lgb.train(
        params,
        train_data,
        num_boost_round=100,
        valid_sets=[test_data],
        valid_names=['test'],
    )

    # Save model
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model.save_model(str(output_path))
    logger.info(f"Saved model to {output_path}")

    # Feature importance
    importance = model.feature_importance(importance_type='gain')
    feature_names = X_train.columns.tolist()

    logger.info("Feature importance:")
    for name, imp in sorted(zip(feature_names, importance), key=lambda x: x[1], reverse=True):
        logger.info(f"  {name}: {imp:.2f}")


def main():
    """Main training pipeline."""
    from services.recommendation.embeddings import EmbeddingService
    from services.recommendation.catalogue import TrackCatalogue
    from services.recommendation.audio_similarity import AudioSimilarityEngine

    # Configuration
    item2vec_path = os.getenv("ITEM2VEC_PATH", "models/item2vec/item2vec.wordvectors")
    playlist_file = os.getenv("PLAYLIST_FILE", "data/smpd/processed/playlists.json")
    output_path = os.getenv("RANKER_OUTPUT", "models/ranker/lightgbm_ranker.txt")

    # Check dependencies
    if not Path(item2vec_path).exists():
        logger.error(f"Item2vec model not found: {item2vec_path}")
        return

    if not Path(playlist_file).exists():
        logger.error(f"Playlist file not found: {playlist_file}")
        return

    # Load services
    logger.info("Loading embedding service")
    embedding_service = EmbeddingService(item2vec_path=item2vec_path)

    logger.info("Loading catalogue")
    catalogue = TrackCatalogue()

    audio_sim = AudioSimilarityEngine()

    # Load playlists
    logger.info("Loading playlists")
    with open(playlist_file, 'r') as f:
        playlists = json.load(f)

    # Sample for faster training
    playlists = random.sample(playlists, min(10000, len(playlists)))

    # Generate training data
    generator = RankerTrainingDataGenerator(
        embedding_service.item2vec,
        catalogue,
        audio_sim,
    )

    features_df, groups = generator.generate_training_examples(playlists, max_examples=10000)

    # Train ranker
    train_lightgbm_ranker(features_df, groups, output_path)

    logger.info("Training complete!")


if __name__ == "__main__":
    main()
