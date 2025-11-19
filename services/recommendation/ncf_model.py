"""
Neural Collaborative Filtering (NCF) for music recommendations.

Implements NeuMF architecture combining:
- Generalized Matrix Factorization (GMF): Linear interactions
- Multi-Layer Perceptron (MLP): Non-linear interactions

Enhanced with:
- BPR (Bayesian Personalized Ranking) loss for implicit feedback
- Hard negative sampling for better training
- In-batch negatives for efficiency
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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


# ============================================================================
# BPR Loss and Negative Sampling
# ============================================================================


class BPRLoss(nn.Module):
    """
    Bayesian Personalized Ranking loss for implicit feedback.

    BPR optimizes pairwise ranking: positive items should be ranked
    higher than negative items.

    Loss = -log(sigmoid(pos_score - neg_score))

    References:
    - Rendle et al. "BPR: Bayesian Personalized Ranking from Implicit Feedback" (UAI 2009)
    """

    def __init__(self, weight_decay: float = 0.0):
        """
        Args:
            weight_decay: L2 regularization weight (applied to model parameters separately)
        """
        super().__init__()
        self.weight_decay = weight_decay

    def forward(
        self,
        pos_scores: torch.Tensor,
        neg_scores: torch.Tensor,
        model: Optional[nn.Module] = None,
    ) -> torch.Tensor:
        """
        Compute BPR loss.

        Args:
            pos_scores: (batch_size,) scores for positive items
            neg_scores: (batch_size,) or (batch_size, num_negatives) scores for negative items
            model: Optional model for L2 regularization

        Returns:
            Scalar loss
        """
        # BPR loss: maximize difference between positive and negative scores
        # -log(sigmoid(pos - neg))

        if neg_scores.dim() == 2:
            # Multiple negatives per positive: (batch_size, num_negatives)
            pos_scores = pos_scores.unsqueeze(1)  # (batch_size, 1)
            diff = pos_scores - neg_scores  # (batch_size, num_negatives)
            loss = -F.logsigmoid(diff).mean()
        else:
            # Single negative per positive: (batch_size,)
            diff = pos_scores - neg_scores
            loss = -F.logsigmoid(diff).mean()

        # Add L2 regularization if model provided
        if model is not None and self.weight_decay > 0:
            l2_reg = 0
            for param in model.parameters():
                l2_reg += torch.norm(param, p=2)
            loss = loss + self.weight_decay * l2_reg

        return loss


class NegativeSampler:
    """
    Negative sampling strategies for BPR training.

    Supports:
    - Random sampling
    - Popularity-biased sampling (sample popular items more often)
    - Hard negative sampling (items similar to positives but not in playlist)
    """

    def __init__(
        self,
        num_items: int,
        popularity: Optional[np.ndarray] = None,
        sampling_strategy: str = "popularity_biased",
        alpha: float = 0.75,
    ):
        """
        Args:
            num_items: Total number of items in the catalogue
            popularity: (num_items,) popularity scores for items (for biased sampling)
            sampling_strategy: "random", "popularity_biased", or "hard"
            alpha: Exponent for popularity bias (0.75 is Mikolov's recommendation)
        """
        self.num_items = num_items
        self.sampling_strategy = sampling_strategy
        self.alpha = alpha

        # Precompute sampling probabilities for popularity-biased sampling
        if sampling_strategy == "popularity_biased" and popularity is not None:
            # P(item) ∝ popularity^alpha
            # This balances between uniform (alpha=0) and popularity (alpha=1)
            probs = np.power(popularity + 1e-6, alpha)
            self.sampling_probs = probs / probs.sum()
        else:
            self.sampling_probs = None

    def sample(
        self,
        batch_size: int,
        num_negatives: int = 1,
        exclude_items: Optional[List[List[int]]] = None,
    ) -> np.ndarray:
        """
        Sample negative items for a batch.

        Args:
            batch_size: Number of samples in the batch
            num_negatives: Number of negative samples per positive
            exclude_items: List of lists of items to exclude for each sample
                          (e.g., items already in the playlist)

        Returns:
            (batch_size, num_negatives) array of negative item IDs
        """
        if self.sampling_strategy == "random":
            return self._sample_random(batch_size, num_negatives, exclude_items)
        elif self.sampling_strategy == "popularity_biased":
            return self._sample_popularity_biased(batch_size, num_negatives, exclude_items)
        else:
            raise ValueError(f"Unknown sampling strategy: {self.sampling_strategy}")

    def _sample_random(
        self,
        batch_size: int,
        num_negatives: int,
        exclude_items: Optional[List[List[int]]],
    ) -> np.ndarray:
        """Uniform random sampling."""
        negatives = np.zeros((batch_size, num_negatives), dtype=np.int64)

        for i in range(batch_size):
            exclude = set(exclude_items[i]) if exclude_items else set()
            candidates = [j for j in range(self.num_items) if j not in exclude]

            if len(candidates) < num_negatives:
                # Not enough candidates, sample with replacement
                negatives[i] = np.random.choice(candidates, size=num_negatives, replace=True)
            else:
                negatives[i] = np.random.choice(candidates, size=num_negatives, replace=False)

        return negatives

    def _sample_popularity_biased(
        self,
        batch_size: int,
        num_negatives: int,
        exclude_items: Optional[List[List[int]]],
    ) -> np.ndarray:
        """Sample negatives proportional to popularity^alpha."""
        if self.sampling_probs is None:
            logger.warning("No popularity provided, falling back to random sampling")
            return self._sample_random(batch_size, num_negatives, exclude_items)

        negatives = np.zeros((batch_size, num_negatives), dtype=np.int64)

        for i in range(batch_size):
            exclude = set(exclude_items[i]) if exclude_items else set()

            # Adjust probabilities to exclude certain items
            probs = self.sampling_probs.copy()
            if exclude:
                exclude_indices = list(exclude)
                probs[exclude_indices] = 0
                probs = probs / probs.sum()  # Renormalize

            negatives[i] = np.random.choice(
                self.num_items, size=num_negatives, replace=False, p=probs
            )

        return negatives

    @staticmethod
    def sample_hard_negatives(
        model: NeuMF,
        playlist_ids: torch.Tensor,
        positive_items: torch.Tensor,
        num_candidates: int = 20,  # Reduced from 100 for speed (still effective!)
        num_negatives: int = 5,
        exclude_items: Optional[List[List[int]]] = None,
        device: str = "cpu",
    ) -> torch.Tensor:
        """
        Sample hard negatives: items with high scores but not in the playlist.

        These are "hard" because the model currently thinks they're good matches,
        so training on them provides strong learning signal.

        Args:
            model: NeuMF model
            playlist_ids: (batch_size,) playlist IDs
            positive_items: (batch_size,) positive item IDs
            num_candidates: Number of random candidates to score
            num_negatives: Number of hard negatives to return per sample
            exclude_items: Items to exclude (e.g., items already in playlist)
            device: Device to run on

        Returns:
            (batch_size, num_negatives) hard negative item IDs
        """
        model.eval()
        batch_size = playlist_ids.size(0)
        hard_negatives = torch.zeros((batch_size, num_negatives), dtype=torch.long)

        with torch.no_grad():
            for i in range(batch_size):
                playlist_id = playlist_ids[i].item()
                exclude = set(exclude_items[i]) if exclude_items else set()
                exclude.add(positive_items[i].item())  # Also exclude the positive

                # Fast sampling: randomly sample candidates and retry if they're excluded
                # Much faster than creating a list of all 598K tracks!
                sampled_candidates = []
                max_attempts = num_candidates * 10  # Prevent infinite loop
                attempts = 0

                while len(sampled_candidates) < num_candidates and attempts < max_attempts:
                    # Sample random tracks
                    batch = np.random.randint(0, model.num_tracks, size=num_candidates * 2)
                    # Keep only non-excluded ones
                    for track_id in batch:
                        if track_id not in exclude and len(sampled_candidates) < num_candidates:
                            sampled_candidates.append(track_id)
                    attempts += 1

                if len(sampled_candidates) < num_negatives:
                    # Emergency fallback: just sample randomly
                    sampled_candidates = np.random.randint(0, model.num_tracks, size=num_negatives).tolist()

                sampled_candidates = np.array(sampled_candidates[:num_candidates])

                # Score candidates
                playlist_batch = torch.LongTensor([playlist_id] * len(sampled_candidates)).to(device)
                candidate_batch = torch.LongTensor(sampled_candidates).to(device)
                scores = model.forward(playlist_batch, candidate_batch).cpu().numpy()

                # Select top-scoring candidates as hard negatives
                top_indices = np.argsort(scores)[-num_negatives:][::-1]
                hard_negatives[i] = torch.LongTensor(sampled_candidates[top_indices])

        model.train()
        return hard_negatives.to(device)


class BPRTrainer:
    """
    Trainer for NeuMF with BPR loss.

    Handles training loop with negative sampling, hard negative mining,
    and in-batch negatives.
    """

    def __init__(
        self,
        model: NeuMF,
        negative_sampler: NegativeSampler,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-5,
        device: str = "cpu",
        use_hard_negatives: bool = False,
        hard_neg_ratio: float = 0.3,
    ):
        """
        Args:
            model: NeuMF model to train
            negative_sampler: NegativeSampler instance
            learning_rate: Learning rate for optimizer
            weight_decay: L2 regularization weight
            device: Device to train on
            use_hard_negatives: Whether to use hard negative mining
            hard_neg_ratio: Ratio of hard negatives (rest are random/popularity-biased)
        """
        self.model = model.to(device)
        self.negative_sampler = negative_sampler
        self.device = device
        self.use_hard_negatives = use_hard_negatives
        self.hard_neg_ratio = hard_neg_ratio

        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.criterion = BPRLoss(weight_decay=0)  # weight_decay handled by optimizer

    def train_step(
        self,
        playlist_ids: torch.Tensor,
        positive_items: torch.Tensor,
        exclude_items: Optional[List[List[int]]] = None,
        num_negatives: int = 1,
    ) -> float:
        """
        Single training step with BPR loss.

        Args:
            playlist_ids: (batch_size,) playlist IDs
            positive_items: (batch_size,) positive item IDs
            exclude_items: List of lists of items to exclude for negative sampling
            num_negatives: Number of negative samples per positive

        Returns:
            Loss value
        """
        self.model.train()
        batch_size = playlist_ids.size(0)

        # Get positive scores
        pos_scores = self.model.forward(playlist_ids, positive_items)

        # Sample negatives
        if self.use_hard_negatives:
            # Mix hard and easy negatives
            num_hard = max(1, int(num_negatives * self.hard_neg_ratio))
            num_easy = num_negatives - num_hard

            # Hard negatives
            hard_negs = NegativeSampler.sample_hard_negatives(
                self.model,
                playlist_ids,
                positive_items,
                num_negatives=num_hard,
                exclude_items=exclude_items,
                device=self.device,
            )

            # Easy negatives (random or popularity-biased)
            if num_easy > 0:
                easy_negs = self.negative_sampler.sample(
                    batch_size, num_easy, exclude_items
                )
                easy_negs = torch.LongTensor(easy_negs).to(self.device)
                negative_items = torch.cat([hard_negs, easy_negs], dim=1)
            else:
                negative_items = hard_negs
        else:
            # Only easy negatives
            negative_items = self.negative_sampler.sample(
                batch_size, num_negatives, exclude_items
            )
            negative_items = torch.LongTensor(negative_items).to(self.device)

        # Get negative scores
        # Reshape for multiple negatives: (batch_size * num_negatives,)
        playlist_ids_expanded = playlist_ids.unsqueeze(1).expand(-1, num_negatives).reshape(-1)
        negative_items_flat = negative_items.reshape(-1)
        neg_scores = self.model.forward(playlist_ids_expanded, negative_items_flat)
        neg_scores = neg_scores.reshape(batch_size, num_negatives)

        # Compute BPR loss
        loss = self.criterion(pos_scores, neg_scores)

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def train_epoch(
        self,
        train_data: List[Tuple[int, int, List[int]]],
        batch_size: int = 256,
        num_negatives: int = 4,
        shuffle: bool = False,
    ) -> float:
        """
        Train for one epoch.

        Args:
            train_data: List of (playlist_id, positive_item, exclude_items) tuples
            batch_size: Batch size
            num_negatives: Number of negative samples per positive
            shuffle: Whether to shuffle data (WARNING: slow for large datasets!)

        Returns:
            Average loss for the epoch
        """
        import logging
        logger = logging.getLogger(__name__)

        total_loss = 0
        num_batches = 0

        total_batches = len(train_data) // batch_size
        log_interval = max(1, total_batches // 20)  # Log 20 times per epoch

        # Shuffle training data (skip for very large datasets)
        if shuffle:
            logger.info(f"Shuffling {len(train_data):,} training examples...")
            np.random.shuffle(train_data)
            logger.info("Shuffle complete!")

        logger.info(f"Training on {len(train_data):,} examples ({total_batches:,} batches)")

        for i in range(0, len(train_data), batch_size):
            batch = train_data[i:i + batch_size]

            # Prepare batch
            playlist_ids = torch.LongTensor([x[0] for x in batch]).to(self.device)
            positive_items = torch.LongTensor([x[1] for x in batch]).to(self.device)
            exclude_items = [x[2] for x in batch]

            # Training step
            loss = self.train_step(playlist_ids, positive_items, exclude_items, num_negatives)
            total_loss += loss
            num_batches += 1

            # Log progress
            if num_batches % log_interval == 0:
                avg_loss = total_loss / num_batches
                progress = (num_batches / total_batches) * 100
                logger.info(f"  Progress: {progress:.1f}% ({num_batches:,}/{total_batches:,}) - Loss: {avg_loss:.4f}")

        return total_loss / num_batches if num_batches > 0 else 0.0
