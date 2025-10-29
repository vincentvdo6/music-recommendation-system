"""
Build ANN index from trained item2vec embeddings.

This creates an Annoy index for fast nearest neighbor search
during recommendation serving.
"""

import logging
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from services.recommendation.embeddings import EmbeddingService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Build and save ANN index."""
    # Configuration
    item2vec_path = os.getenv("ITEM2VEC_PATH", "models/item2vec/item2vec.wordvectors")
    output_path = os.getenv("ANN_INDEX_PATH", "models/item2vec/ann_index")
    n_trees = int(os.getenv("ANN_N_TREES", "50"))

    # Check if model exists
    if not Path(item2vec_path).exists():
        logger.error(f"Item2vec model not found: {item2vec_path}")
        logger.info("Please run scripts/train_item2vec.py first")
        return

    # Load embeddings
    logger.info("Loading embedding service")
    embedding_service = EmbeddingService(item2vec_path=item2vec_path)

    # Build index
    logger.info(f"Building ANN index with {n_trees} trees")
    ann_index = embedding_service.build_ann_index(n_trees=n_trees)

    if ann_index is not None:
        # Save index
        logger.info(f"Saving index to {output_path}")
        ann_index.save(output_path)
        logger.info("ANN index built successfully!")
    else:
        logger.info("Skipping ANN index (will use gensim fallback for similarity search)")

    # Run sanity check
    logger.info("Running sanity check...")
    test_track = embedding_service.item2vec.wv.index_to_key[0]
    neighbors = embedding_service.get_neighbors(test_track, k=10)

    logger.info(f"Test track: {test_track}")
    logger.info(f"Top 10 neighbors: {neighbors[:10]}")
    logger.info(f"\nWith {len(embedding_service.item2vec.wv)} tracks, gensim fallback is fast enough!")


if __name__ == "__main__":
    main()
