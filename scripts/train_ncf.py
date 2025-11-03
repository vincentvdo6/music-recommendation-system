"""
Train Neural Collaborative Filtering (NCF) model on playlist data.

Usage:
    python scripts/train_ncf.py
"""

import json
import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Add parent directory to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from services.recommendation.ncf_model import NeuMF

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s:%(name)s:%(message)s'
)
logger = logging.getLogger(__name__)


class PlaylistDataset(Dataset):
    """
    Dataset for playlist-track interactions.

    Generates positive and negative samples for training.
    """

    def __init__(
        self,
        playlist_tracks: Dict[int, List[int]],
        num_tracks: int,
        num_negatives: int = 4,
    ):
        """
        Args:
            playlist_tracks: Dict mapping playlist_id -> list of track_ids
            num_tracks: Total number of tracks in the dataset
            num_negatives: Number of negative samples per positive
        """
        self.playlist_tracks = playlist_tracks
        self.num_tracks = num_tracks
        self.num_negatives = num_negatives

        # Create positive samples
        self.samples = []
        for playlist_id, track_ids in playlist_tracks.items():
            for track_id in track_ids:
                self.samples.append((playlist_id, track_id, 1.0))

        logger.info(f"Created {len(self.samples):,} positive samples")

        # Generate negative samples
        self._generate_negatives()

    def _generate_negatives(self):
        """Generate negative samples (tracks NOT in playlist)."""
        negative_samples = []

        for playlist_id, track_ids in self.playlist_tracks.items():
            track_set = set(track_ids)

            # Sample negatives
            negatives_needed = len(track_ids) * self.num_negatives
            negatives = []

            while len(negatives) < negatives_needed:
                neg_track = random.randint(0, self.num_tracks - 1)
                if neg_track not in track_set:
                    negatives.append((playlist_id, neg_track, 0.0))

            negative_samples.extend(negatives)

        logger.info(f"Generated {len(negative_samples):,} negative samples")
        self.samples.extend(negative_samples)

        # Shuffle
        random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        playlist_id, track_id, label = self.samples[idx]
        return {
            'playlist_id': torch.LongTensor([playlist_id])[0],
            'track_id': torch.LongTensor([track_id])[0],
            'label': torch.FloatTensor([label])[0],
        }


def load_playlists(playlists_path: Path) -> List:
    """Load playlists from JSON."""
    with open(playlists_path, 'r') as f:
        data = json.load(f)

    # Handle different formats
    if isinstance(data, dict):
        if 'playlists' in data:
            return data['playlists']
        else:
            raise ValueError(f"Dict format but no 'playlists' key in {playlists_path}")
    elif isinstance(data, list):
        return data
    else:
        raise ValueError(f"Unexpected playlist format in {playlists_path}")


def create_mappings(playlists: List) -> Tuple[Dict, Dict, Dict, Dict]:
    """
    Create ID mappings for playlists and tracks.

    Returns:
        playlist_id_map: playlist_uri -> int
        track_id_map: track_id -> int
        id_to_playlist: int -> playlist_uri
        id_to_track: int -> track_id
    """
    playlist_id_map = {}
    track_id_map = {}

    for i, playlist in enumerate(playlists):
        # Handle both dict format and list format
        if isinstance(playlist, dict):
            playlist_uri = playlist.get('pid', f"playlist_{i}")
            track_ids = playlist.get('tracks', [])
        else:  # List of track IDs
            playlist_uri = f"playlist_{i}"
            track_ids = playlist

        playlist_id_map[playlist_uri] = i

        for track_id in track_ids:
            if track_id not in track_id_map:
                track_id_map[track_id] = len(track_id_map)

    id_to_playlist = {v: k for k, v in playlist_id_map.items()}
    id_to_track = {v: k for k, v in track_id_map.items()}

    logger.info(f"Created mappings: {len(playlist_id_map)} playlists, {len(track_id_map)} tracks")

    return playlist_id_map, track_id_map, id_to_playlist, id_to_track


def prepare_training_data(
    playlists: List,
    playlist_id_map: Dict,
    track_id_map: Dict,
) -> Dict[int, List[int]]:
    """
    Convert playlists to training format.

    Returns:
        Dict mapping playlist_id (int) -> list of track_ids (int)
    """
    playlist_tracks = defaultdict(list)

    for i, playlist in enumerate(playlists):
        # Handle both dict format and list format
        if isinstance(playlist, dict):
            playlist_uri = playlist.get('pid', f"playlist_{i}")
            track_ids = playlist.get('tracks', [])
        else:  # List of track IDs
            playlist_uri = f"playlist_{i}"
            track_ids = playlist

        playlist_id = playlist_id_map[playlist_uri]

        for track_id in track_ids:
            if track_id in track_id_map:
                track_int_id = track_id_map[track_id]
                playlist_tracks[playlist_id].append(track_int_id)

    return dict(playlist_tracks)


def train_epoch(
    model: NeuMF,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: str,
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0

    for batch in dataloader:
        playlist_ids = batch['playlist_id'].to(device)
        track_ids = batch['track_id'].to(device)
        labels = batch['label'].to(device)

        # Forward pass
        predictions = model(playlist_ids, track_ids)
        loss = criterion(predictions, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate(
    model: NeuMF,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: str,
) -> float:
    """Evaluate on validation set."""
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch in dataloader:
            playlist_ids = batch['playlist_id'].to(device)
            track_ids = batch['track_id'].to(device)
            labels = batch['label'].to(device)

            predictions = model(playlist_ids, track_ids)
            loss = criterion(predictions, labels)

            total_loss += loss.item()

    return total_loss / len(dataloader)


def main():
    # Configuration
    PLAYLISTS_PATH = Path("data/smpd/processed/playlists.json")
    MODEL_DIR = Path("models/ncf")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    BATCH_SIZE = 512
    EPOCHS = 10
    LEARNING_RATE = 0.001
    NUM_NEGATIVES = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info(f"Using device: {DEVICE}")

    # Set random seeds
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # Load playlists
    logger.info(f"Loading playlists from {PLAYLISTS_PATH}")
    playlists = load_playlists(PLAYLISTS_PATH)
    logger.info(f"Loaded {len(playlists):,} playlists")

    # Create mappings
    playlist_id_map, track_id_map, id_to_playlist, id_to_track = create_mappings(playlists)

    # Prepare training data
    playlist_tracks = prepare_training_data(playlists, playlist_id_map, track_id_map)

    # Split into train/validation
    playlist_ids = list(playlist_tracks.keys())
    random.shuffle(playlist_ids)

    split_idx = int(len(playlist_ids) * 0.8)
    train_playlist_ids = playlist_ids[:split_idx]
    val_playlist_ids = playlist_ids[split_idx:]

    train_data = {pid: playlist_tracks[pid] for pid in train_playlist_ids}
    val_data = {pid: playlist_tracks[pid] for pid in val_playlist_ids}

    logger.info(f"Train: {len(train_data)} playlists, Val: {len(val_data)} playlists")

    # Create datasets
    train_dataset = PlaylistDataset(train_data, len(track_id_map), NUM_NEGATIVES)
    val_dataset = PlaylistDataset(val_data, len(track_id_map), NUM_NEGATIVES)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Create model
    model = NeuMF(
        num_playlists=len(playlist_id_map),
        num_tracks=len(track_id_map),
        gmf_dim=64,
        mlp_dim=64,
        mlp_layers=[128, 64, 32],
        dropout=0.2,
    ).to(DEVICE)

    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Training setup
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Train
    best_val_loss = float('inf')

    logger.info("Starting training...")
    for epoch in range(1, EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss = evaluate(model, val_loader, criterion, DEVICE)

        logger.info(f"Epoch {epoch}/{EPOCHS}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'playlist_id_map': playlist_id_map,
                'track_id_map': track_id_map,
                'id_to_playlist': id_to_playlist,
                'id_to_track': id_to_track,
                'num_playlists': len(playlist_id_map),
                'num_tracks': len(track_id_map),
            }, MODEL_DIR / "ncf_model.pt")
            logger.info(f"✓ Saved best model (val_loss={val_loss:.4f})")

    logger.info("Training complete!")
    logger.info(f"Best validation loss: {best_val_loss:.4f}")
    logger.info(f"Model saved to {MODEL_DIR / 'ncf_model.pt'}")


if __name__ == "__main__":
    main()
