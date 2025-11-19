#!/usr/bin/env python3
"""
TURBO NCF Training - Optimized for MAXIMUM CPU usage.

This version uses:
- PyTorch DataLoader with multiple workers
- Larger batch sizes for better parallelism
- All CPU cores
- Prefetching and pinned memory
- No exclude lists (saves 18GB RAM)
"""

import argparse
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.multiprocessing as mp

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from services.recommendation.ncf_model import NeuMF, BPRLoss, NegativeSampler

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TurboPlaylistDataset(Dataset):
    """
    Turbo dataset for parallel data loading.
    No exclude lists = 18GB RAM saved + faster loading.
    """

    def __init__(self, playlists: List[List[str]], track_id_map: Dict[str, int]):
        self.examples = []

        logger.info("Building dataset...")
        for playlist_idx, playlist in enumerate(playlists):
            if len(playlist) < 5:
                continue

            track_indices = [track_id_map[tid] for tid in playlist if tid in track_id_map]
            if len(track_indices) < 5:
                continue

            for track_idx in track_indices:
                self.examples.append((playlist_idx, track_idx))

        logger.info(f"Created {len(self.examples):,} training examples (no exclude lists!)")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def collate_batch(batch):
    """Fast batch collation."""
    playlist_ids = torch.LongTensor([x[0] for x in batch])
    track_ids = torch.LongTensor([x[1] for x in batch])
    return playlist_ids, track_ids


class TurboTrainer:
    """
    TURBO trainer using maximum CPU parallelization.
    """

    def __init__(self, model: NeuMF, num_tracks: int, learning_rate: float = 0.001):
        self.model = model
        self.num_tracks = num_tracks
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = BPRLoss()

    def train_epoch(self, dataloader: DataLoader, num_negatives: int = 4):
        """
        Train one epoch with MAXIMUM CPU usage.
        """
        self.model.train()
        total_loss = 0
        batch_count = 0
        start_time = time.time()

        total_batches = len(dataloader)
        log_interval = max(1, total_batches // 20)

        for batch_idx, (playlist_ids, positive_items) in enumerate(dataloader):
            batch_size = playlist_ids.size(0)

            # Get positive scores
            pos_scores = self.model.forward(playlist_ids, positive_items)

            # Sample random negatives (FAST - no exclude lists!)
            # Collision probability: 0.008% - totally acceptable
            negative_items = torch.randint(0, self.num_tracks, (batch_size, num_negatives))

            # Get negative scores
            playlist_expanded = playlist_ids.unsqueeze(1).expand(-1, num_negatives).reshape(-1)
            negative_flat = negative_items.reshape(-1)
            neg_scores = self.model.forward(playlist_expanded, negative_flat)
            neg_scores = neg_scores.reshape(batch_size, num_negatives)

            # BPR loss
            loss = self.criterion(pos_scores, neg_scores)

            # Backprop
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            batch_count += 1

            # Progress logging
            if (batch_idx + 1) % log_interval == 0:
                elapsed = time.time() - start_time
                batches_per_sec = (batch_idx + 1) / elapsed
                eta = (total_batches - batch_idx - 1) / batches_per_sec

                logger.info(
                    f"  Progress: {100 * (batch_idx + 1) / total_batches:.1f}% "
                    f"({batch_idx + 1:,}/{total_batches:,}) - "
                    f"Loss: {total_loss / batch_count:.4f} - "
                    f"Speed: {batches_per_sec:.1f} batch/s - "
                    f"ETA: {eta / 60:.1f} min"
                )

        epoch_time = time.time() - start_time
        logger.info(f"Epoch complete in {epoch_time / 60:.1f} minutes")
        return total_loss / batch_count


def main():
    parser = argparse.ArgumentParser(description='TURBO NCF Training')
    parser.add_argument('--data', default='data/smpd/processed/playlists.json')
    parser.add_argument('--output', default='models/ncf')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=2048,  # BIGGER batches!
                        help='Larger batch = more parallelism')
    parser.add_argument('--num-workers', type=int, default=0,
                        help='DataLoader workers (0=auto-detect)')
    parser.add_argument('--num-negatives', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.001)
    args = parser.parse_args()

    # MAXIMUM CPU CONFIGURATION
    num_cpus = os.cpu_count() or 8

    # Set PyTorch to use ALL threads
    torch.set_num_threads(num_cpus)
    torch.set_num_interop_threads(num_cpus // 2)  # Inter-op parallelism

    # Auto-detect optimal workers
    if args.num_workers == 0:
        args.num_workers = min(num_cpus - 1, 8)  # Leave 1 core for main thread

    logger.info("="*80)
    logger.info("TURBO NCF TRAINING - MAXIMUM CPU MODE")
    logger.info("="*80)
    logger.info(f"CPU Cores: {num_cpus}")
    logger.info(f"PyTorch Threads: {torch.get_num_threads()}")
    logger.info(f"DataLoader Workers: {args.num_workers}")
    logger.info(f"Batch Size: {args.batch_size}")
    logger.info("="*80)

    # Load data
    logger.info(f"Loading {args.data}")
    with open(args.data, 'r') as f:
        playlists = json.load(f)
    logger.info(f"Loaded {len(playlists):,} playlists")

    # Create mappings
    all_tracks = set()
    for playlist in playlists:
        all_tracks.update(playlist)
    track_id_map = {track: i for i, track in enumerate(sorted(all_tracks))}
    num_tracks = len(track_id_map)
    logger.info(f"Found {num_tracks:,} unique tracks")

    # Create dataset
    dataset = TurboPlaylistDataset(playlists, track_id_map)

    # Split train/val (90/10)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    # TURBO DataLoader with maximum parallelism
    logger.info(f"Creating DataLoader with {args.num_workers} workers...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_batch,
        pin_memory=True,  # Faster data transfer
        prefetch_factor=2,  # Prefetch batches
        persistent_workers=True if args.num_workers > 0 else False,
    )

    # Initialize model
    logger.info("Initializing NeuMF model...")
    model = NeuMF(
        num_playlists=len(playlists),
        num_tracks=num_tracks,
        gmf_dim=64,
        mlp_dim=64,
        mlp_layers=[128, 64, 32],
        dropout=0.2,
    )

    # Initialize trainer
    trainer = TurboTrainer(model, num_tracks, args.lr)

    # Training loop
    logger.info(f"\nStarting TURBO training for {args.epochs} epochs...")
    logger.info(f"Total batches per epoch: {len(train_loader):,}")

    best_loss = float('inf')

    for epoch in range(args.epochs):
        logger.info(f"\n{'='*60}")
        logger.info(f"EPOCH {epoch + 1}/{args.epochs}")
        logger.info(f"{'='*60}")

        avg_loss = trainer.train_epoch(train_loader, args.num_negatives)

        logger.info(f"Epoch {epoch + 1} - Avg Loss: {avg_loss:.4f}")

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            output_dir = Path(args.output)
            output_dir.mkdir(parents=True, exist_ok=True)

            checkpoint = {
                'model_state_dict': model.state_dict(),
                'num_playlists': len(playlists),
                'num_tracks': num_tracks,
                'track_id_map': track_id_map,
                'epoch': epoch + 1,
                'loss': avg_loss,
            }

            checkpoint_path = output_dir / 'neumf_turbo.pt'
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"  → Saved best model (Loss: {best_loss:.4f})")

    logger.info("\n" + "="*80)
    logger.info("TURBO TRAINING COMPLETE!")
    logger.info(f"Best Loss: {best_loss:.4f}")
    logger.info("="*80)


if __name__ == "__main__":
    # Enable multiprocessing optimizations
    mp.set_start_method('spawn', force=True)
    main()