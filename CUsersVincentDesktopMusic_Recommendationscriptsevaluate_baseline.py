"""
Evaluate baseline (original contextual engine) for comparison.
"""

import json
import logging
import os
from pathlib import Path
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from services.recommendation.contextual_engine import ContextualRecommendationEngine
from services.recommendation.catalogue import TrackCatalogue

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Evaluate baseline engine."""
    playlist_file = os.getenv("PLAYLIST_FILE", "data/smpd/processed/playlists.json")

    if not Path(playlist_file).exists():
        logger.error(f"Playlist file not found: {playlist_file}")
        logger.info("Run py scripts/download_smpd.py first")
        return

    # Load baseline engine (no embeddings, no ranker)
    logger.info("Loading baseline contextual engine")
    catalogue = TrackCatalogue()
    engine = ContextualRecommendationEngine(catalogue=catalogue)

    # Load playlists
    with open(playlist_file, 'r') as f:
        playlists = json.load(f)

    logger.info("Note: Baseline evaluation requires evaluate.py functionality")
    logger.info("For now, just confirming baseline engine loads successfully")
    logger.info(f"Catalogue has {len(catalogue.all_tracks())} tracks")


if __name__ == "__main__":
    main()
