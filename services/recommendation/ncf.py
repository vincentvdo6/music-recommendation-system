"""
Item-based NCF (fold-in NeuMF) inference.

The model has item embedding tables only — no playlist tower — so it can
score (context tracks -> candidate) for playlists it has never seen: the
context is represented by mean-pooling its tracks' embeddings ("fold-in").

Checkpoint format (produced by training/kaggle_train_ranker.ipynb):
    {
        "state_dict": ...,
        "track_id_map": {track_id: index},
        "config": {"n_tracks": int, "gmf_dim": int, "mlp_dim": int, "mlp_layers": [int, ...]},
        "format": "item-ncf-v2",
    }

torch is imported lazily inside functions so the application runs fine
without torch installed (the ranker then sees ncf_score=0 / has_ncf=0).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

CHECKPOINT_FORMAT = "item-ncf-v2"
MIN_CONTEXT_TRACKS = 3


def _build_model(torch, config: dict):
    nn = torch.nn

    class ItemNCF(nn.Module):
        def __init__(self):
            super().__init__()
            n = config["n_tracks"]
            self.gmf_emb = nn.Embedding(n, config["gmf_dim"])
            self.mlp_emb = nn.Embedding(n, config["mlp_dim"])

            layers = []
            in_dim = config["mlp_dim"] * 2
            for out_dim in config["mlp_layers"]:
                layers += [nn.Linear(in_dim, out_dim), nn.ReLU()]
                in_dim = out_dim
            self.mlp = nn.Sequential(*layers)
            self.head = nn.Linear(config["gmf_dim"] + in_dim, 1)

        def forward(self, context_idx, candidate_idx):
            # context_idx: (m,) — one shared context; candidate_idx: (n,)
            u_gmf = self.gmf_emb(context_idx).mean(dim=0)
            u_mlp = self.mlp_emb(context_idx).mean(dim=0)

            c_gmf = self.gmf_emb(candidate_idx)
            c_mlp = self.mlp_emb(candidate_idx)

            gmf_out = u_gmf.unsqueeze(0) * c_gmf
            mlp_in = torch.cat([u_mlp.unsqueeze(0).expand(len(candidate_idx), -1), c_mlp], dim=1)
            mlp_out = self.mlp(mlp_in)

            return self.head(torch.cat([gmf_out, mlp_out], dim=1)).squeeze(1)

    return ItemNCF()


class ItemNCFScorer:
    """Batched (context -> candidates) scoring over the trained item-NCF."""

    def __init__(self, model, track_id_map: Dict[str, int], torch_module):
        self._model = model
        self._id_map = track_id_map
        self._torch = torch_module

    @property
    def vocab_size(self) -> int:
        return len(self._id_map)

    def score(
        self,
        context_ids: List[str],
        candidate_ids: List[str],
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Score candidates against the context in one batched forward pass.

        Returns (scores, mask) aligned to candidate_ids — mask is 1.0 where
        the candidate is in the NCF vocabulary — or None when fewer than
        MIN_CONTEXT_TRACKS context tracks are known to the model.
        """
        torch = self._torch
        context_idx = [self._id_map[t] for t in context_ids if t in self._id_map]
        if len(context_idx) < MIN_CONTEXT_TRACKS:
            return None

        cand_positions = [(pos, self._id_map[t]) for pos, t in enumerate(candidate_ids) if t in self._id_map]
        scores = np.zeros(len(candidate_ids), dtype=np.float32)
        mask = np.zeros(len(candidate_ids), dtype=np.float32)
        if not cand_positions:
            return scores, mask

        positions, cand_idx = zip(*cand_positions)
        with torch.inference_mode():
            logits = self._model(
                torch.tensor(context_idx, dtype=torch.long),
                torch.tensor(cand_idx, dtype=torch.long),
            )
            probs = torch.sigmoid(logits).numpy()

        scores[list(positions)] = probs
        mask[list(positions)] = 1.0
        return scores, mask


def load_item_ncf(path: str) -> Optional[ItemNCFScorer]:
    """Load the item-NCF checkpoint; returns None (with a log line) on any failure."""
    if not Path(path).exists():
        logger.info("Item-NCF checkpoint not found at %s", path)
        return None

    try:
        import torch
    except ImportError:
        logger.warning("torch not installed — item-NCF disabled")
        return None

    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if checkpoint.get("format") != CHECKPOINT_FORMAT:
            raise ValueError(
                f"checkpoint format {checkpoint.get('format')!r} != {CHECKPOINT_FORMAT!r} "
                "(old playlist-tower checkpoints are not servable)"
            )
        model = _build_model(torch, checkpoint["config"])
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        scorer = ItemNCFScorer(model, checkpoint["track_id_map"], torch)
        logger.info("Loaded item-NCF (%d-track vocab) from %s", scorer.vocab_size, path)
        return scorer
    except Exception as exc:
        logger.warning("Failed to load item-NCF from %s: %s", path, exc)
        return None
