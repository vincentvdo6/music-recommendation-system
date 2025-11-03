"""
Neural Collaborative Filtering (NCF) for music recommendations.

Implements NeuMF architecture combining:
- Generalized Matrix Factorization (GMF): Linear interactions
- Multi-Layer Perceptron (MLP): Non-linear interactions
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class GMF(nn.Module):
    """
    Generalized Matrix Factorization.

    Learns linear interactions via element-wise product of embeddings.
    """

    def __init__(self, num_playlists: int, num_tracks: int, embedding_dim: int = 64):
        super().__init__()
        self.playlist_embedding = nn.Embedding(num_playlists, embedding_dim)
        self.track_embedding = nn.Embedding(num_tracks, embedding_dim)

        # Initialize embeddings
        nn.init.normal_(self.playlist_embedding.weight, std=0.01)
        nn.init.normal_(self.track_embedding.weight, std=0.01)

    def forward(self, playlist_ids: torch.Tensor, track_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            playlist_ids: (batch_size,)
            track_ids: (batch_size,)

        Returns:
            (batch_size, embedding_dim) element-wise product
        """
        playlist_emb = self.playlist_embedding(playlist_ids)  # (batch, dim)
        track_emb = self.track_embedding(track_ids)  # (batch, dim)
        return playlist_emb * track_emb  # Element-wise product


class MLP(nn.Module):
    """
    Multi-Layer Perceptron for non-linear interactions.
    """

    def __init__(
        self,
        num_playlists: int,
        num_tracks: int,
        embedding_dim: int = 64,
        hidden_layers: List[int] = [128, 64, 32],
        dropout: float = 0.2,
    ):
        super().__init__()
        self.playlist_embedding = nn.Embedding(num_playlists, embedding_dim)
        self.track_embedding = nn.Embedding(num_tracks, embedding_dim)

        # Initialize embeddings
        nn.init.normal_(self.playlist_embedding.weight, std=0.01)
        nn.init.normal_(self.track_embedding.weight, std=0.01)

        # Build MLP layers
        layers = []
        input_dim = embedding_dim * 2  # Concatenate playlist + track embeddings

        for hidden_dim in hidden_layers:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(dropout),
            ])
            input_dim = hidden_dim

        self.mlp = nn.Sequential(*layers)
        self.output_dim = hidden_layers[-1]

    def forward(self, playlist_ids: torch.Tensor, track_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            playlist_ids: (batch_size,)
            track_ids: (batch_size,)

        Returns:
            (batch_size, output_dim) non-linear features
        """
        playlist_emb = self.playlist_embedding(playlist_ids)  # (batch, dim)
        track_emb = self.track_embedding(track_ids)  # (batch, dim)

        # Concatenate and pass through MLP
        concat = torch.cat([playlist_emb, track_emb], dim=1)  # (batch, 2*dim)
        return self.mlp(concat)


class NeuMF(nn.Module):
    """
    Neural Matrix Factorization: Combines GMF + MLP.

    Architecture from "Neural Collaborative Filtering" (He et al., WWW 2017).
    """

    def __init__(
        self,
        num_playlists: int,
        num_tracks: int,
        gmf_dim: int = 64,
        mlp_dim: int = 64,
        mlp_layers: List[int] = [128, 64, 32],
        dropout: float = 0.2,
    ):
        super().__init__()

        self.gmf = GMF(num_playlists, num_tracks, gmf_dim)
        self.mlp = MLP(num_playlists, num_tracks, mlp_dim, mlp_layers, dropout)

        # Final prediction layer
        final_input_dim = gmf_dim + self.mlp.output_dim
        self.prediction = nn.Sequential(
            nn.Linear(final_input_dim, 1),
            nn.Sigmoid()
        )

        self.num_playlists = num_playlists
        self.num_tracks = num_tracks

    def forward(self, playlist_ids: torch.Tensor, track_ids: torch.Tensor) -> torch.Tensor:
        """
        Predict probability of track in playlist.

        Args:
            playlist_ids: (batch_size,)
            track_ids: (batch_size,)

        Returns:
            (batch_size,) prediction scores in [0, 1]
        """
        gmf_out = self.gmf(playlist_ids, track_ids)  # (batch, gmf_dim)
        mlp_out = self.mlp(playlist_ids, track_ids)  # (batch, mlp_output_dim)

        # Concatenate and predict
        concat = torch.cat([gmf_out, mlp_out], dim=1)
        prediction = self.prediction(concat).squeeze()  # (batch,)

        return prediction

    def recommend(
        self,
        playlist_id: int,
        candidate_track_ids: List[int],
        top_k: int = 100,
        device: str = "cpu",
    ) -> List[Tuple[int, float]]:
        """
        Generate recommendations for a playlist.

        Args:
            playlist_id: ID of the playlist
            candidate_track_ids: List of candidate track IDs
            top_k: Number of recommendations to return
            device: Device to run inference on

        Returns:
            List of (track_id, score) tuples sorted by score descending
        """
        self.eval()

        with torch.no_grad():
            # Create batch
            playlist_ids = torch.LongTensor([playlist_id] * len(candidate_track_ids)).to(device)
            track_ids = torch.LongTensor(candidate_track_ids).to(device)

            # Predict scores
            scores = self.forward(playlist_ids, track_ids)
            scores = scores.cpu().numpy()

        # Sort by score descending
        ranked = sorted(zip(candidate_track_ids, scores), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]


class NCFRecommender:
    """
    Wrapper for NCF model with convenience methods.
    """

    def __init__(
        self,
        model: NeuMF,
        playlist_id_map: Dict[str, int],
        track_id_map: Dict[str, int],
        device: str = "cpu",
    ):
        """
        Args:
            model: Trained NeuMF model
            playlist_id_map: Mapping from playlist URI to integer ID
            track_id_map: Mapping from track Spotify ID to integer ID
            device: Device to run inference on
        """
        self.model = model.to(device)
        self.model.eval()

        self.playlist_id_map = playlist_id_map
        self.track_id_map = track_id_map

        # Reverse mappings
        self.id_to_playlist = {v: k for k, v in playlist_id_map.items()}
        self.id_to_track = {v: k for k, v in track_id_map.items()}

        self.device = device

    def recommend_for_tracks(
        self,
        track_ids: List[str],
        candidate_track_ids: Optional[List[str]] = None,
        top_k: int = 100,
    ) -> List[Tuple[str, float]]:
        """
        Generate recommendations based on input tracks.

        Args:
            track_ids: List of Spotify track IDs (seed tracks)
            candidate_track_ids: List of candidate Spotify track IDs to rank
                                If None, uses all tracks in the model
            top_k: Number of recommendations to return

        Returns:
            List of (track_id, score) tuples
        """
        # Convert track IDs to integers
        seed_ids = [self.track_id_map[tid] for tid in track_ids if tid in self.track_id_map]

        if not seed_ids:
            logger.warning("No seed tracks found in model")
            return []

        # Use all tracks as candidates if not specified
        if candidate_track_ids is None:
            candidate_ids = list(range(self.model.num_tracks))
        else:
            candidate_ids = [
                self.track_id_map[tid] for tid in candidate_track_ids
                if tid in self.track_id_map
            ]

        # Remove seed tracks from candidates
        candidate_ids = [cid for cid in candidate_ids if cid not in seed_ids]

        if not candidate_ids:
            logger.warning("No candidate tracks after filtering")
            return []

        # Create a synthetic "playlist" by averaging seed track embeddings
        # For now, we'll score each candidate against each seed and average
        all_scores = []

        with torch.no_grad():
            for seed_id in seed_ids:
                playlist_ids = torch.LongTensor([seed_id] * len(candidate_ids)).to(self.device)
                track_ids_tensor = torch.LongTensor(candidate_ids).to(self.device)

                scores = self.model.forward(playlist_ids, track_ids_tensor)
                all_scores.append(scores.cpu().numpy())

        # Average scores across seeds
        avg_scores = np.mean(all_scores, axis=0)

        # Convert back to Spotify IDs and sort
        results = [
            (self.id_to_track[cid], float(score))
            for cid, score in zip(candidate_ids, avg_scores)
        ]
        results.sort(key=lambda x: x[1], reverse=True)

        return results[:top_k]
