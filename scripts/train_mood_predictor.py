"""
Train a mood predictor that estimates audio features from item2vec embeddings.

This allows us to predict moods for any track without needing Spotify's audio-features API!

Strategy:
1. Load curated mood seeds (30 tracks with known audio features)
2. Use item2vec to find similar tracks to each seed
3. Propagate mood labels through similarity graph
4. Train Random Forest to predict: embedding → audio_features
"""

import json
import logging
import os
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# Add parent to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from services.recommendation.embeddings import EmbeddingService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MoodPredictor:
    """Predicts audio features from item2vec embeddings."""

    def __init__(self):
        self.model = None
        self.feature_names = ['valence', 'energy', 'acousticness', 'danceability']

    def train(self, X: np.ndarray, y: np.ndarray):
        """
        Train Random Forest to predict audio features.

        Args:
            X: (n_samples, embedding_dim) - item2vec embeddings
            y: (n_samples, 4) - [valence, energy, acousticness, danceability]
        """
        try:
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.multioutput import MultiOutputRegressor
        except ImportError:
            logger.error("scikit-learn not installed. Run: pip install scikit-learn")
            return False

        logger.info(f"Training mood predictor on {len(X)} samples...")

        # Multi-output Random Forest
        base_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )

        self.model = MultiOutputRegressor(base_model)
        self.model.fit(X, y)

        logger.info(" Model trained successfully!")
        return True

    def predict(self, embedding: np.ndarray) -> Dict[str, float]:
        """
        Predict audio features for a track embedding.

        Args:
            embedding: (embedding_dim,) item2vec embedding

        Returns:
            Dict with valence, energy, acousticness, danceability
        """
        if self.model is None:
            logger.warning("Model not trained, returning defaults")
            return {
                'valence': 0.5,
                'energy': 0.5,
                'acousticness': 0.5,
                'danceability': 0.5
            }

        # Reshape for prediction
        X = embedding.reshape(1, -1)
        predictions = self.model.predict(X)[0]

        # Clip to valid range [0, 1]
        predictions = np.clip(predictions, 0, 1)

        return {
            name: float(pred)
            for name, pred in zip(self.feature_names, predictions)
        }

    def save(self, path: str):
        """Save model to disk."""
        if self.model is None:
            logger.error("No model to save!")
            return

        Path(path).parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'feature_names': self.feature_names
            }, f)

        logger.info(f" Model saved to {path}")

    def load(self, path: str):
        """Load model from disk."""
        with open(path, 'rb') as f:
            data = pickle.load(f)

        self.model = data['model']
        self.feature_names = data['feature_names']

        logger.info(f" Model loaded from {path}")


def load_mood_seeds(seed_file: str) -> List[Dict]:
    """Load curated mood seeds."""
    with open(seed_file, 'r') as f:
        data = json.load(f)

    logger.info(f"Loaded {len(data['seeds'])} mood seeds")
    return data['seeds']


def propagate_moods_via_similarity(
    seeds: List[Dict],
    embedding_service: EmbeddingService,
    k_neighbors: int = 10,
    min_similarity: float = 0.3
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Propagate mood labels to similar tracks using item2vec.

    For each seed:
    1. Find K nearest neighbors
    2. Assign them weighted audio features based on similarity
    3. Collect all embeddings and labels for training

    Args:
        seeds: List of seed tracks with audio_features
        embedding_service: Item2vec service
        k_neighbors: Number of neighbors to propagate to
        min_similarity: Minimum similarity threshold

    Returns:
        (embeddings, audio_features) for training
    """
    all_embeddings = []
    all_features = []

    for seed in seeds:
        track_id = seed['spotify_id']

        # Get seed embedding
        seed_embedding = embedding_service.item2vec.vector(track_id)
        if seed_embedding is None:
            logger.warning(f"Seed {seed['name']} not in item2vec, skipping")
            continue

        seed_audio = seed['audio_features']

        # Add seed itself
        all_embeddings.append(seed_embedding)
        all_features.append([
            seed_audio['valence'],
            seed_audio['energy'],
            seed_audio['acousticness'],
            seed_audio['danceability']
        ])

        # Find similar tracks
        neighbors = embedding_service.item2vec.neighbors(track_id, k=k_neighbors)

        for neighbor_id, similarity in neighbors:
            if similarity < min_similarity:
                continue

            neighbor_embedding = embedding_service.item2vec.vector(neighbor_id)
            if neighbor_embedding is None:
                continue

            # Weighted propagation: closer neighbors get features more similar to seed
            propagated_features = [
                seed_audio['valence'],  # Keep seed features (could blend if multiple seeds affect same track)
                seed_audio['energy'],
                seed_audio['acousticness'],
                seed_audio['danceability']
            ]

            all_embeddings.append(neighbor_embedding)
            all_features.append(propagated_features)

    logger.info(f"Propagated moods to {len(all_embeddings)} total tracks")

    return np.array(all_embeddings), np.array(all_features)


def main():
    """Main training pipeline."""
    logger.info("="*80)
    logger.info("Training Mood Predictor")
    logger.info("="*80)

    # Paths
    seed_file = "data/mood_seeds.json"
    item2vec_path = "models/item2vec/item2vec.wordvectors"
    output_path = "models/mood_predictor/mood_predictor.pkl"

    # Check dependencies
    if not Path(seed_file).exists():
        logger.error(f"Mood seeds not found: {seed_file}")
        return

    if not Path(item2vec_path).exists():
        logger.error(f"Item2vec not found: {item2vec_path}")
        return

    # Load item2vec
    logger.info("Loading item2vec embeddings...")
    embedding_service = EmbeddingService(item2vec_path=item2vec_path)

    # Load mood seeds
    seeds = load_mood_seeds(seed_file)

    # Propagate moods to similar tracks
    logger.info("\nPropagating moods via item2vec similarity...")
    X_train, y_train = propagate_moods_via_similarity(
        seeds,
        embedding_service,
        k_neighbors=20,  # Use top 20 neighbors per seed
        min_similarity=0.3  # Only propagate to reasonably similar tracks
    )

    if len(X_train) == 0:
        logger.error("No training data generated! Seeds may not be in item2vec.")
        return

    logger.info("\nTraining dataset:")
    logger.info(f"  Samples: {len(X_train)}")
    logger.info(f"  Embedding dim: {X_train.shape[1]}")
    logger.info("  Features: valence, energy, acousticness, danceability")

    # Train model
    logger.info("\n" + "="*80)
    predictor = MoodPredictor()
    success = predictor.train(X_train, y_train)

    if not success:
        logger.error("Training failed!")
        return

    # Save model
    predictor.save(output_path)

    # Test predictions on a few seeds
    logger.info("\n" + "="*80)
    logger.info("Testing predictions on seed tracks:")
    logger.info("="*80)

    for seed in seeds[:5]:  # Test first 5
        track_id = seed['spotify_id']
        embedding = embedding_service.item2vec.vector(track_id)

        if embedding is None:
            continue

        predicted = predictor.predict(embedding)
        actual = seed['audio_features']

        logger.info(f"\n{seed['name']} - {seed['artist']}")
        logger.info(f"  Actual:    V={actual['valence']:.2f} E={actual['energy']:.2f} "
                   f"A={actual['acousticness']:.2f} D={actual['danceability']:.2f}")
        logger.info(f"  Predicted: V={predicted['valence']:.2f} E={predicted['energy']:.2f} "
                   f"A={predicted['acousticness']:.2f} D={predicted['danceability']:.2f}")

    logger.info("\n" + "="*80)
    logger.info("Training complete!")
    logger.info(f"Model saved to: {output_path}")
    logger.info("="*80)


if __name__ == "__main__":
    main()
