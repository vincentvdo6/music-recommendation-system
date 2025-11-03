"""
Load trained NCF model for inference.
"""

import logging
from pathlib import Path
from typing import Optional

import torch

from services.recommendation.ncf_model import NeuMF, NCFRecommender

logger = logging.getLogger(__name__)


def load_ncf_model(
    model_path: Path,
    device: str = "cpu",
) -> Optional[NCFRecommender]:
    """
    Load trained NCF model.

    Args:
        model_path: Path to saved model checkpoint
        device: Device to load model on

    Returns:
        NCFRecommender instance or None if loading fails
    """
    try:
        if not model_path.exists():
            logger.warning(f"NCF model not found at {model_path}")
            return None

        logger.info(f"Loading NCF model from {model_path}")
        checkpoint = torch.load(model_path, map_location=device)

        # Create model
        model = NeuMF(
            num_playlists=checkpoint['num_playlists'],
            num_tracks=checkpoint['num_tracks'],
            gmf_dim=64,
            mlp_dim=64,
            mlp_layers=[128, 64, 32],
            dropout=0.2,
        )

        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        # Create recommender
        recommender = NCFRecommender(
            model=model,
            playlist_id_map=checkpoint['playlist_id_map'],
            track_id_map=checkpoint['track_id_map'],
            device=device,
        )

        logger.info(f"✓ NCF model loaded: {checkpoint['num_tracks']:,} tracks")

        return recommender

    except Exception as e:
        logger.error(f"Failed to load NCF model: {e}")
        return None
