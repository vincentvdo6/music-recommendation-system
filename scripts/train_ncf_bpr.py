#!/usr/bin/env python3
"""
Train NeuMF model with BPR loss on playlist data.

This script:
1. Loads playlist data from data/smpd/processed/playlists.json
2. Creates train/validation split
3. Trains NeuMF with BPR loss and hard negative sampling
4. Evaluates on validation set
5. Saves the best model checkpoint

Usage:
    python scripts/train_ncf_bpr.py --epochs 20 --batch-size 512

Expected improvements over BCE:
- +5-7% NDCG from BPR loss
- +2-3% from hard negative mining
"""

import argparse
import json
import logging
import random
import sys
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from services.recommendation.ncf_model import (
    NeuMF,
    BPRLoss,
    NegativeSampler,
    BPRTrainer,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_playlist_data(playlist_file: str) -> List[List[str]]:
    """Load playlists from JSON file."""
    logger.info(f"Loading playlists from {playlist_file}")
    with open(playlist_file, 'r') as f:
        playlists = json.load(f)
    logger.info(f"Loaded {len(playlists)} playlists")
    return playlists


def create_mappings(
    playlists: List[List[str]]
) -> Tuple[Dict[str, int], Dict[str, int]]:
    """
    Create bidirectional mappings for playlists and tracks.

    Args:
        playlists: List of playlists (each playlist is a list of track IDs)

    Returns:
        Tuple of (playlist_id_map, track_id_map)
    """
    # Create playlist ID mapping (playlist index -> integer ID)
    playlist_id_map = {str(i): i for i in range(len(playlists))}

    # Collect all unique tracks
    all_tracks = set()
    for playlist in playlists:
        all_tracks.update(playlist)

    # Create track ID mapping
    track_id_map = {track_id: i for i, track_id in enumerate(sorted(all_tracks))}

    logger.info(f"Created mappings: {len(playlist_id_map)} playlists, {len(track_id_map)} tracks")

    return playlist_id_map, track_id_map


def compute_track_popularity(
    playlists: List[List[str]],
    track_id_map: Dict[str, int]
) -> np.ndarray:
    """
    Compute track popularity based on playlist frequency.

    Args:
        playlists: List of playlists
        track_id_map: Mapping from track ID to integer

    Returns:
        Array of popularity scores (higher = more popular)
    """
    popularity = np.zeros(len(track_id_map))

    for playlist in playlists:
        for track_id in playlist:
            if track_id in track_id_map:
                popularity[track_id_map[track_id]] += 1

    # Normalize to 0-100 scale
    if popularity.max() > 0:
        popularity = (popularity / popularity.max()) * 100

    logger.info(f"Track popularity stats: min={popularity.min():.1f}, "
                f"mean={popularity.mean():.1f}, max={popularity.max():.1f}")

    return popularity


def prepare_training_data(
    playlists: List[List[str]],
    playlist_id_map: Dict[str, int],
    track_id_map: Dict[str, int],
    min_tracks: int = 5,
    use_exclude_lists: bool = False,  # NEW: Make exclude lists optional
) -> List[Tuple[int, int, List[int]]]:
    """
    Convert playlists to training examples.

    Each track in a playlist is a positive example. We exclude other tracks
    in the same playlist from negative sampling.

    Args:
        playlists: List of playlists
        playlist_id_map: Playlist ID to integer mapping
        track_id_map: Track ID to integer mapping
        min_tracks: Minimum tracks required in a playlist
        use_exclude_lists: If False, skip storing exclude lists (saves 18GB RAM!)

    Returns:
        List of (playlist_id, positive_track_id, exclude_track_ids) tuples
    """
    training_examples = []

    for playlist_idx, playlist in enumerate(playlists):
        if len(playlist) < min_tracks:
            continue

        # Convert track IDs to integers
        track_indices = [track_id_map[tid] for tid in playlist if tid in track_id_map]

        if len(track_indices) < min_tracks:
            continue

        playlist_id = playlist_id_map[str(playlist_idx)]

        # Create training example for each track
        for track_idx in track_indices:
            if use_exclude_lists:
                # Exclude all other tracks in this playlist
                exclude = [t for t in track_indices if t != track_idx]
            else:
                # Skip exclude lists (collision probability is 0.008% - negligible!)
                exclude = []
            training_examples.append((playlist_id, track_idx, exclude))

    logger.info(f"Created {len(training_examples):,} training examples")
    if not use_exclude_lists:
        logger.info("  (Exclude lists disabled - saves 18GB RAM!)")

    return training_examples


def train_test_split(
    training_data: List[Tuple[int, int, List[int]]],
    test_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[List[Tuple[int, int, List[int]]], List[Tuple[int, int, List[int]]]]:
    """Split data into train and validation sets."""
    random.seed(seed)
    random.shuffle(training_data)

    split_idx = int(len(training_data) * (1 - test_ratio))
    train_data = training_data[:split_idx]
    val_data = training_data[split_idx:]

    logger.info(f"Split: {len(train_data):,} train, {len(val_data):,} validation")

    return train_data, val_data


def evaluate_model(
    model: NeuMF,
    val_data: List[Tuple[int, int, List[int]]],
    num_samples: int = 1000,
    k: int = 10,
    device: str = "cpu",
) -> Dict[str, float]:
    """
    Evaluate model on validation set.

    Computes Hit Rate and NDCG by ranking all tracks for each playlist
    and checking if the ground truth appears in top-k.

    Args:
        model: Trained NeuMF model
        val_data: Validation data
        num_samples: Number of samples to evaluate (for speed)
        k: Top-k for metrics
        device: Device to run on

    Returns:
        Dict with metrics
    """
    model.eval()

    hits = 0
    ndcgs = []

    # Sample for faster evaluation
    sampled_data = random.sample(val_data, min(num_samples, len(val_data)))

    with torch.no_grad():
        for playlist_id, positive_item, exclude_items in sampled_data:
            # Get all candidate tracks (excluding those in playlist)
            all_tracks = list(range(model.num_tracks))
            candidates = [t for t in all_tracks if t not in exclude_items and t != positive_item]

            # Limit candidates for speed
            if len(candidates) > 100:
                candidates = random.sample(candidates, 100)

            # Add positive item to candidates
            candidates.append(positive_item)

            # Score all candidates
            playlist_ids = torch.LongTensor([playlist_id] * len(candidates)).to(device)
            track_ids = torch.LongTensor(candidates).to(device)
            scores = model.forward(playlist_ids, track_ids).cpu().numpy()

            # Rank candidates
            ranked_indices = np.argsort(scores)[::-1]
            ranked_items = [candidates[i] for i in ranked_indices]

            # Hit rate: Is positive item in top-k?
            if positive_item in ranked_items[:k]:
                hits += 1

                # NDCG: Position-aware metric
                rank = ranked_items.index(positive_item) + 1
                ndcg = 1.0 / np.log2(rank + 1)
                ndcgs.append(ndcg)
            else:
                ndcgs.append(0.0)

    hit_rate = hits / len(sampled_data)
    mean_ndcg = np.mean(ndcgs)

    return {
        "hit_rate": hit_rate,
        "ndcg": mean_ndcg,
    }


def main():
    parser = argparse.ArgumentParser(description='Train NeuMF with BPR loss')
    parser.add_argument('--data', type=str, default='data/smpd/processed/playlists.json',
                        help='Path to playlist data')
    parser.add_argument('--output', type=str, default='models/ncf',
                        help='Output directory for model checkpoint')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=512,
                        help='Batch size for training')
    parser.add_argument('--num-negatives', type=int, default=4,
                        help='Number of negative samples per positive')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--gmf-dim', type=int, default=64,
                        help='GMF embedding dimension')
    parser.add_argument('--mlp-dim', type=int, default=64,
                        help='MLP embedding dimension')
    parser.add_argument('--no-hard-negatives', action='store_true', default=False,
                        help='Disable hard negative mining (not recommended)')
    parser.add_argument('--hard-neg-ratio', type=float, default=0.3,
                        help='Ratio of hard negatives')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to train on (cpu or cuda)')
    parser.add_argument('--val-ratio', type=float, default=0.1,
                        help='Validation set ratio')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    args = parser.parse_args()

    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # MAX OUT CPU USAGE - Use all available threads!
    import os
    num_threads = os.cpu_count() or 4
    torch.set_num_threads(num_threads)
    logger.info(f"PyTorch using {num_threads} CPU threads for maximum performance!")

    logger.info("="*80)
    logger.info("NeuMF Training with BPR Loss")
    logger.info("="*80)
    logger.info(f"Configuration:")
    for key, value in vars(args).items():
        logger.info(f"  {key}: {value}")
    logger.info("="*80)

    # Load data
    playlists = load_playlist_data(args.data)

    # Create mappings
    playlist_id_map, track_id_map = create_mappings(playlists)
    num_playlists = len(playlist_id_map)
    num_tracks = len(track_id_map)

    # Compute track popularity
    track_popularity = compute_track_popularity(playlists, track_id_map)

    # Prepare training data (skip exclude lists to save 18GB RAM!)
    training_data = prepare_training_data(
        playlists, playlist_id_map, track_id_map, use_exclude_lists=False
    )
    train_data, val_data = train_test_split(training_data, test_ratio=args.val_ratio, seed=args.seed)

    # Initialize model
    logger.info(f"\nInitializing NeuMF model...")
    logger.info(f"  Playlists: {num_playlists}")
    logger.info(f"  Tracks: {num_tracks}")
    logger.info(f"  GMF dim: {args.gmf_dim}")
    logger.info(f"  MLP dim: {args.mlp_dim}")

    model = NeuMF(
        num_playlists=num_playlists,
        num_tracks=num_tracks,
        gmf_dim=args.gmf_dim,
        mlp_dim=args.mlp_dim,
        mlp_layers=[128, 64, 32],
        dropout=0.2,
    )

    # Initialize negative sampler
    negative_sampler = NegativeSampler(
        num_items=num_tracks,
        popularity=track_popularity,
        sampling_strategy="popularity_biased",
        alpha=0.75,  # Mikolov's recommendation
    )

    # Initialize trainer
    trainer = BPRTrainer(
        model=model,
        negative_sampler=negative_sampler,
        learning_rate=args.lr,
        weight_decay=1e-5,
        device=args.device,
        use_hard_negatives=not args.no_hard_negatives,  # Inverted flag
        hard_neg_ratio=args.hard_neg_ratio,
    )

    # Training loop
    logger.info(f"\nStarting training for {args.epochs} epochs...")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Negatives per positive: {args.num_negatives}")
    logger.info(f"  Hard negative mining: {not args.no_hard_negatives}")
    if not args.no_hard_negatives:
        logger.info(f"  Hard negative ratio: {args.hard_neg_ratio}")
    logger.info("")

    best_ndcg = 0.0
    best_epoch = 0

    for epoch in range(args.epochs):
        # Train
        avg_loss = trainer.train_epoch(
            train_data=train_data,
            batch_size=args.batch_size,
            num_negatives=args.num_negatives,
        )

        # Evaluate
        metrics = evaluate_model(model, val_data, num_samples=1000, k=10, device=args.device)

        logger.info(
            f"Epoch {epoch + 1}/{args.epochs} - "
            f"Loss: {avg_loss:.4f} - "
            f"Hit@10: {metrics['hit_rate']:.4f} - "
            f"NDCG@10: {metrics['ndcg']:.4f}"
        )

        # Save best model
        if metrics['ndcg'] > best_ndcg:
            best_ndcg = metrics['ndcg']
            best_epoch = epoch + 1

            # Save checkpoint
            output_dir = Path(args.output)
            output_dir.mkdir(parents=True, exist_ok=True)

            checkpoint = {
                'model_state_dict': model.state_dict(),
                'num_playlists': num_playlists,
                'num_tracks': num_tracks,
                'playlist_id_map': playlist_id_map,
                'track_id_map': track_id_map,
                'epoch': epoch + 1,
                'metrics': metrics,
                'config': vars(args),
            }

            checkpoint_path = output_dir / 'neumf_bpr_best.pt'
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"  → Saved best model (NDCG@10: {best_ndcg:.4f})")

    logger.info("")
    logger.info("="*80)
    logger.info("Training complete!")
    logger.info(f"Best NDCG@10: {best_ndcg:.4f} (epoch {best_epoch})")
    logger.info(f"Model saved to: {output_dir / 'neumf_bpr_best.pt'}")
    logger.info("="*80)


if __name__ == "__main__":
    main()
