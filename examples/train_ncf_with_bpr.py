"""
Example script for training NeuMF with BPR loss and hard negative sampling.

This demonstrates the enhanced training capabilities:
- BPR (Bayesian Personalized Ranking) loss for implicit feedback
- Popularity-biased negative sampling (α=0.75)
- Hard negative mining (30% hard, 70% easy negatives)
- Multiple negatives per positive for stronger gradient signal

Expected improvements over BCE loss:
- Better ranking quality (+5-7% NDCG)
- Faster convergence
- More robust to imbalanced data
"""

import numpy as np
import torch
from pathlib import Path

from services.recommendation.ncf_model import (
    NeuMF,
    BPRLoss,
    NegativeSampler,
    BPRTrainer,
)


def prepare_training_data(playlist_data):
    """
    Convert playlist data to BPR training format.

    Args:
        playlist_data: Dict mapping playlist_id -> list of track_ids

    Returns:
        List of (playlist_id, positive_item, exclude_items) tuples
    """
    training_examples = []

    for playlist_id, track_ids in playlist_data.items():
        # Each track in playlist is a positive example
        for track_id in track_ids:
            # Exclude all other tracks in this playlist from negative sampling
            exclude = [tid for tid in track_ids if tid != track_id]
            training_examples.append((playlist_id, track_id, exclude))

    return training_examples


def train_neumf_with_bpr(
    train_data,
    num_playlists,
    num_tracks,
    track_popularity,
    num_epochs=10,
    batch_size=256,
    num_negatives=4,
    learning_rate=0.001,
    use_hard_negatives=True,
    device="cpu",
):
    """
    Train NeuMF model with BPR loss.

    Args:
        train_data: List of (playlist_id, positive_item, exclude_items) tuples
        num_playlists: Total number of playlists
        num_tracks: Total number of tracks
        track_popularity: (num_tracks,) popularity scores for tracks
        num_epochs: Number of training epochs
        batch_size: Batch size
        num_negatives: Number of negative samples per positive
        learning_rate: Learning rate
        use_hard_negatives: Whether to use hard negative mining
        device: "cpu" or "cuda"

    Returns:
        Trained NeuMF model
    """
    # Initialize model
    model = NeuMF(
        num_playlists=num_playlists,
        num_tracks=num_tracks,
        gmf_dim=64,
        mlp_dim=64,
        mlp_layers=[128, 64, 32],
        dropout=0.2,
    )

    # Initialize negative sampler with popularity bias
    negative_sampler = NegativeSampler(
        num_items=num_tracks,
        popularity=track_popularity,
        sampling_strategy="popularity_biased",
        alpha=0.75,  # Mikolov's recommended value
    )

    # Initialize BPR trainer
    trainer = BPRTrainer(
        model=model,
        negative_sampler=negative_sampler,
        learning_rate=learning_rate,
        weight_decay=1e-5,
        device=device,
        use_hard_negatives=use_hard_negatives,
        hard_neg_ratio=0.3,  # 30% hard negatives, 70% easy
    )

    # Training loop
    print(f"Training NeuMF with BPR loss")
    print(f"  Epochs: {num_epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Negatives per positive: {num_negatives}")
    print(f"  Hard negative mining: {use_hard_negatives}")
    print(f"  Training examples: {len(train_data):,}")
    print()

    for epoch in range(num_epochs):
        avg_loss = trainer.train_epoch(
            train_data=train_data,
            batch_size=batch_size,
            num_negatives=num_negatives,
        )

        print(f"Epoch {epoch + 1}/{num_epochs} - Loss: {avg_loss:.4f}")

        # Optional: Evaluate on validation set here
        # val_ndcg = evaluate(model, val_data)
        # print(f"  Validation NDCG@10: {val_ndcg:.4f}")

    return model


def main():
    """Example usage with synthetic data."""
    # Synthetic data for demonstration
    num_playlists = 1000
    num_tracks = 5000

    # Create synthetic playlist data
    playlist_data = {}
    for playlist_id in range(num_playlists):
        # Each playlist has 20-50 tracks
        num_tracks_in_playlist = np.random.randint(20, 51)
        track_ids = np.random.choice(num_tracks, size=num_tracks_in_playlist, replace=False)
        playlist_data[playlist_id] = track_ids.tolist()

    # Synthetic popularity (power-law distribution)
    track_popularity = np.random.zipf(1.5, size=num_tracks)
    track_popularity = np.clip(track_popularity, 1, 100)

    # Prepare training data
    train_data = prepare_training_data(playlist_data)
    print(f"Prepared {len(train_data):,} training examples from {num_playlists} playlists")

    # Train model
    model = train_neumf_with_bpr(
        train_data=train_data,
        num_playlists=num_playlists,
        num_tracks=num_tracks,
        track_popularity=track_popularity,
        num_epochs=5,
        batch_size=256,
        num_negatives=4,
        learning_rate=0.001,
        use_hard_negatives=True,
        device="cpu",
    )

    # Save model
    output_dir = Path("models/ncf")
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "num_playlists": num_playlists,
        "num_tracks": num_tracks,
    }
    torch.save(checkpoint, output_dir / "neumf_bpr.pt")
    print(f"\nModel saved to {output_dir / 'neumf_bpr.pt'}")

    # Example: Generate recommendations
    model.eval()
    test_playlist_id = 0
    candidate_tracks = list(range(100))  # Test with first 100 tracks

    recommendations = model.recommend(
        playlist_id=test_playlist_id,
        candidate_track_ids=candidate_tracks,
        top_k=10,
    )

    print(f"\nTop 10 recommendations for playlist {test_playlist_id}:")
    for i, (track_id, score) in enumerate(recommendations, 1):
        print(f"  {i}. Track {track_id}: {score:.4f}")


if __name__ == "__main__":
    main()
