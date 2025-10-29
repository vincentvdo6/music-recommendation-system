"""
Evaluation harness for recommendation engine.

Implements standard RecSys 2018 metrics:
- R-Precision: Precision at R (R = number of relevant items)
- NDCG@K: Normalized Discounted Cumulative Gain
- Click@K: Did user click any of top-K?

Based on Spotify Million Playlist Dataset Challenge evaluation.
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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PlaylistContinuationEvaluator:
    """Evaluate recommendation engine on playlist continuation task."""

    def __init__(self, recommendation_engine):
        self.engine = recommendation_engine

    def evaluate_playlists(
        self,
        playlists: List[List[str]],
        n_test_playlists: int = 1000,
        k_values: List[int] = [5, 10, 30, 100],
    ) -> Dict[str, float]:
        """
        Evaluate engine on playlist continuation.

        For each playlist:
        1. Hide last 20-40% of tracks (ground truth)
        2. Generate recommendations from first 60-80%
        3. Measure overlap with ground truth

        Args:
            playlists: List of playlists (each is list of track IDs)
            n_test_playlists: Number of playlists to evaluate
            k_values: Values of K for metrics

        Returns:
            Dictionary of metric values
        """
        logger.info(f"Evaluating on {n_test_playlists} playlists")

        # Sample test playlists
        test_playlists = random.sample(playlists, min(n_test_playlists, len(playlists)))

        # Metrics accumulator
        metrics = defaultdict(list)

        for i, playlist in enumerate(test_playlists):
            if i == 0:
                logger.info(f"First playlist length: {len(playlist)}")

            if len(playlist) < 10:
                if i == 0:
                    logger.warning(f"First playlist too short: {len(playlist)} < 10")
                continue

            # Split playlist
            split_idx = int(len(playlist) * 0.7)
            train_tracks = playlist[:split_idx]
            test_tracks = set(playlist[split_idx:])

            if i == 0:
                logger.info(f"Split: {len(train_tracks)} train, {len(test_tracks)} test")

            if not test_tracks:
                continue

            # Generate recommendations
            try:
                recommendations = self.engine.recommend(
                    playlist_tracks=train_tracks,
                    limit=max(k_values),
                )
            except Exception as e:
                logger.warning(f"Failed to generate recommendations: {e}")
                continue

            # Extract recommended track IDs
            rec_track_ids = [rec['id'] for rec in recommendations if 'id' in rec]

            # Debug first iteration
            if i == 0:
                logger.info(f"Debug - First playlist:")
                logger.info(f"  Train tracks: {len(train_tracks)}")
                logger.info(f"  Test tracks: {len(test_tracks)}")
                logger.info(f"  Recommendations: {len(rec_track_ids)}")
                logger.info(f"  Test track IDs: {list(test_tracks)[:5]}")
                logger.info(f"  Rec track IDs: {rec_track_ids[:5]}")

            # Compute metrics
            r_precision = self._compute_r_precision(rec_track_ids, test_tracks)
            metrics['r_precision'].append(r_precision)

            for k in k_values:
                ndcg = self._compute_ndcg(rec_track_ids[:k], test_tracks)
                click = self._compute_click(rec_track_ids[:k], test_tracks)

                metrics[f'ndcg@{k}'].append(ndcg)
                metrics[f'click@{k}'].append(click)

            if (i + 1) % 100 == 0:
                logger.info(f"Evaluated {i + 1}/{n_test_playlists} playlists")

        # Aggregate metrics
        results = {}
        for metric_name, values in metrics.items():
            results[metric_name] = float(np.mean(values))

        return results

    def _compute_r_precision(
        self,
        recommendations: List[str],
        ground_truth: set,
    ) -> float:
        """
        R-Precision: Precision at R.

        R = number of relevant items (ground truth size)
        Precision@R = (# relevant in top-R) / R
        """
        R = len(ground_truth)
        top_R = recommendations[:R]

        n_relevant = sum(1 for track in top_R if track in ground_truth)

        if R == 0:
            return 0.0

        return n_relevant / R

    def _compute_ndcg(
        self,
        recommendations: List[str],
        ground_truth: set,
    ) -> float:
        """
        Normalized Discounted Cumulative Gain.

        DCG = Σ (rel_i / log2(i + 1))
        NDCG = DCG / IDCG (ideal DCG)

        rel_i = 1 if track is in ground truth, 0 otherwise
        """
        if not recommendations:
            return 0.0

        # Compute DCG
        dcg = 0.0
        for i, track in enumerate(recommendations):
            rel = 1.0 if track in ground_truth else 0.0
            dcg += rel / np.log2(i + 2)  # i+2 because i starts at 0

        # Compute ideal DCG (all relevant items at top)
        idcg = 0.0
        n_relevant = min(len(ground_truth), len(recommendations))
        for i in range(n_relevant):
            idcg += 1.0 / np.log2(i + 2)

        if idcg == 0:
            return 0.0

        return dcg / idcg

    def _compute_click(
        self,
        recommendations: List[str],
        ground_truth: set,
    ) -> float:
        """
        Click@K: Binary metric.

        Returns 1.0 if any recommended track is in ground truth, else 0.0
        """
        for track in recommendations:
            if track in ground_truth:
                return 1.0

        return 0.0


def main():
    """Run evaluation."""
    from services.recommendation.hybrid_engine import HybridRecommendationEngine
    from services.recommendation.embeddings import EmbeddingService
    from services.recommendation.catalogue import TrackCatalogue
    from services.recommendation.ranker import LearnedRanker

    # Configuration
    playlist_file = os.getenv("PLAYLIST_FILE", "data/smpd/processed/playlists.json")
    item2vec_path = os.getenv("ITEM2VEC_PATH", "models/item2vec/item2vec.wordvectors")
    ranker_path = os.getenv("RANKER_PATH", "models/ranker/lightgbm_ranker.txt")

    # Check dependencies
    if not Path(playlist_file).exists():
        logger.error(f"Playlist file not found: {playlist_file}")
        logger.info("Run scripts/download_smpd.py first")
        return

    # Load services
    logger.info("Loading services")
    embedding_service = EmbeddingService(item2vec_path=item2vec_path) if Path(item2vec_path).exists() else None
    catalogue = TrackCatalogue()

    # Load ranker
    ranker = None
    if Path(ranker_path).exists():
        logger.info("Loading learned ranker")
        ranker = LearnedRanker(ranker_path)

    # Initialize hybrid engine with two-stage retrieval
    engine = HybridRecommendationEngine(
        catalogue=catalogue,
        embedding_service=embedding_service,
        ranker=ranker,
        use_mmr=True,
    )

    # Load test playlists
    logger.info("Loading test playlists")
    with open(playlist_file, 'r') as f:
        playlists = json.load(f)

    # Evaluate
    evaluator = PlaylistContinuationEvaluator(engine)
    results = evaluator.evaluate_playlists(
        playlists,
        n_test_playlists=min(100, len(playlists)),  # Use actual playlist count
        k_values=[5, 10, 30, 100],
    )

    # Print results
    logger.info("=" * 50)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 50)

    for metric, value in sorted(results.items()):
        logger.info(f"{metric:20s}: {value:.4f}")

    # Save results
    output_dir = Path("evaluation")
    output_dir.mkdir(exist_ok=True)

    output_path = output_dir / "results.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"Saved results to {output_path}")


if __name__ == "__main__":
    main()
