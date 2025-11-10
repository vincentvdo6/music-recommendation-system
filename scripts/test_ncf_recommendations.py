#!/usr/bin/env python3
"""
Test the trained BPR model by generating recommendations.

This script loads the trained model and generates recommendations
for sample playlists to verify it's working correctly.

Usage:
    python scripts/test_ncf_recommendations.py
"""

import json
import logging
import random
import torch
from pathlib import Path

from services.recommendation.ncf_loader import load_ncf_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    # Load trained model
    model_path = Path("models/ncf/neumf_bpr_best.pt")

    if not model_path.exists():
        logger.error(f"Model not found at {model_path}")
        logger.error("Please train the model first using:")
        logger.error("  python scripts/train_ncf_bpr.py")
        return

    recommender = load_ncf_model(model_path, device="cpu")

    if recommender is None:
        logger.error("Failed to load model")
        return

    # Load playlist data for testing
    playlists_path = Path("data/smpd/processed/playlists.json")
    with open(playlists_path, 'r') as f:
        playlists = json.load(f)

    logger.info("\n" + "="*80)
    logger.info("Testing NCF Recommendations")
    logger.info("="*80)

    # Test with first 5 playlists
    num_tests = 0
    for i in range(len(playlists)):
        if num_tests >= 5:
            break

        playlist = playlists[i]

        if len(playlist) < 10:
            continue

        # Get playlist ID from the mapping
        playlist_id_str = str(i)
        if playlist_id_str not in recommender.playlist_id_map:
            continue

        playlist_id_int = recommender.playlist_id_map[playlist_id_str]

        # Use first half as context, second half as ground truth
        context_tracks = playlist[:len(playlist)//2]
        ground_truth = playlist[len(playlist)//2:]

        # Get all tracks in model
        all_track_ids = list(recommender.track_id_map.keys())

        # Remove tracks already in the playlist from candidates
        playlist_set = set(playlist)
        candidate_track_ids = [tid for tid in all_track_ids if tid not in playlist_set]

        # Sample 200 candidates for speed
        if len(candidate_track_ids) > 200:
            candidate_track_ids = random.sample(candidate_track_ids, 200)

        # Convert to integer IDs
        candidate_ids = [recommender.track_id_map[tid] for tid in candidate_track_ids]

        logger.info(f"\nPlaylist {i + 1}:")
        logger.info(f"  Context tracks: {len(context_tracks)}")
        logger.info(f"  Ground truth: {len(ground_truth)}")
        logger.info(f"  Candidate tracks: {len(candidate_ids)}")

        # Generate recommendations using the model directly
        with torch.no_grad():
            playlist_ids_tensor = torch.LongTensor([playlist_id_int] * len(candidate_ids))
            track_ids_tensor = torch.LongTensor(candidate_ids)

            scores = recommender.model.forward(playlist_ids_tensor, track_ids_tensor)
            scores = scores.cpu().numpy()

        # Sort by score
        sorted_indices = scores.argsort()[::-1]
        top_k = 20
        top_indices = sorted_indices[:top_k]

        # Convert back to track IDs
        recommendations = [
            (candidate_track_ids[idx], float(scores[idx]))
            for idx in top_indices
        ]

        # Check how many ground truth tracks appear in recommendations
        recommended_ids = {track_id for track_id, _ in recommendations}
        ground_truth_set = set(ground_truth)
        hits = len(recommended_ids & ground_truth_set)

        logger.info(f"\nTop 10 recommendations:")
        for j, (track_id, score) in enumerate(recommendations[:10], 1):
            marker = "✓" if track_id in ground_truth_set else " "
            # Truncate long IDs for display
            display_id = track_id[:40] if len(track_id) > 40 else track_id
            logger.info(f"  {marker} {j:2d}. {display_id:40s} (score: {score:.4f})")

        if hits > 0:
            logger.info(f"\n  ✓ Found {hits}/{len(ground_truth)} ground truth tracks in top-20!")
        else:
            logger.info(f"\n  No ground truth tracks in top-20")

        num_tests += 1

    logger.info("\n" + "="*80)
    logger.info("Testing complete!")
    logger.info(f"Tested {num_tests} playlists")
    logger.info("="*80)


if __name__ == "__main__":
    main()
