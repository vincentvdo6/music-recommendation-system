"""
Train item2vec embeddings on playlist data.

Uses Word2Vec skip-gram with playlist co-occurrence as supervision.
Based on "Item2Vec: Neural Item Embedding for Collaborative Filtering" (2016)
and RecSys 2018 Million Playlist Challenge best practices.
"""

import json
import logging
import os
from pathlib import Path
from typing import List

from gensim.models import Word2Vec

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Item2VecTrainer:
    """Train item2vec embeddings on playlist corpus."""

    def __init__(
        self,
        vector_size: int = 100,
        window: int = 50,  # Large window for playlists (unordered sets)
        min_count: int = 2,  # Filter rare tracks (adjusted for small dataset)
        sg: int = 1,  # Skip-gram (not CBOW)
        negative: int = 20,  # Negative sampling
        workers: int = 8,
        epochs: int = 10,
    ):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.sg = sg
        self.negative = negative
        self.workers = workers
        self.epochs = epochs

    def load_playlists(self, playlist_file: str) -> List[List[str]]:
        """Load preprocessed playlist sequences."""
        logger.info(f"Loading playlists from {playlist_file}")

        with open(playlist_file, 'r') as f:
            playlists = json.load(f)

        logger.info(f"Loaded {len(playlists):,} playlists")
        return playlists

    def train(self, playlists: List[List[str]]) -> Word2Vec:
        """Train Word2Vec model on playlist corpus."""
        logger.info("Training item2vec model")
        logger.info(f"Parameters:")
        logger.info(f"  vector_size: {self.vector_size}")
        logger.info(f"  window: {self.window}")
        logger.info(f"  min_count: {self.min_count}")
        logger.info(f"  negative: {self.negative}")
        logger.info(f"  epochs: {self.epochs}")
        logger.info(f"  workers: {self.workers}")

        model = Word2Vec(
            sentences=playlists,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=self.workers,
            sg=self.sg,
            negative=self.negative,
            epochs=self.epochs,
            compute_loss=True,  # Track training loss
        )

        logger.info(f"Training complete!")
        logger.info(f"Vocabulary size: {len(model.wv):,} tracks")
        logger.info(f"Final loss: {model.get_latest_training_loss():.4f}")

        return model

    def evaluate(self, model: Word2Vec):
        """Run basic sanity checks on the trained model."""
        logger.info("Running evaluation")

        # Check a few example tracks
        vocab = list(model.wv.index_to_key)[:100]

        if vocab:
            example_track = vocab[0]
            logger.info(f"Example track: {example_track}")

            # Get neighbors
            try:
                neighbors = model.wv.most_similar(example_track, topn=5)
                logger.info(f"Top 5 neighbors:")
                for track, score in neighbors:
                    logger.info(f"  {track}: {score:.3f}")
            except KeyError:
                logger.warning(f"Track {example_track} not in vocabulary")

    def save_model(self, model: Word2Vec, output_dir: str):
        """Save trained model and vocabulary."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save full model
        model_path = output_path / "item2vec.model"
        model.save(str(model_path))
        logger.info(f"Saved model to {model_path}")

        # Save word vectors only (smaller, faster to load for inference)
        wv_path = output_path / "item2vec.wordvectors"
        model.wv.save(str(wv_path))
        logger.info(f"Saved word vectors to {wv_path}")

        # Save metadata
        metadata = {
            "n_tracks": len(model.wv),
            "vector_size": self.vector_size,
            "window": self.window,
            "min_count": self.min_count,
            "epochs": self.epochs,
        }

        metadata_path = output_path / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Saved metadata to {metadata_path}")


def main():
    """Main entry point."""
    # Configuration
    playlist_file = os.getenv("PLAYLIST_FILE", "data/smpd/processed/playlists.json")
    output_dir = os.getenv("MODEL_OUTPUT_DIR", "models/item2vec")

    # Check if playlist file exists
    if not Path(playlist_file).exists():
        logger.error(f"Playlist file not found: {playlist_file}")
        logger.info("Please run scripts/download_smpd.py first to preprocess the dataset")
        return

    # Train
    trainer = Item2VecTrainer(
        vector_size=100,
        window=50,
        min_count=2,
        epochs=10,
        workers=8,
    )

    playlists = trainer.load_playlists(playlist_file)
    model = trainer.train(playlists)
    trainer.evaluate(model)
    trainer.save_model(model, output_dir)

    logger.info("Training complete!")


if __name__ == "__main__":
    main()
