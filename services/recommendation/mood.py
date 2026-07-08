"""
Embedding-based mood predictor.

Spotify deprecated its audio-features API, so mood attributes (valence,
energy, acousticness, danceability) are predicted from item2vec embeddings
by a model trained while the API was still available. This is the only
"audio" signal in the system and it works entirely offline.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np

from services.recommendation.features import MOOD_DIMS

logger = logging.getLogger(__name__)


class MoodPredictor:
    """Batch mood prediction from item2vec embeddings."""

    def __init__(self, path: Optional[str] = None):
        self._model = None
        self._dim_order: Optional[list] = None

        if path and Path(path).exists():
            self.load(path)

    @property
    def available(self) -> bool:
        return self._model is not None

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            payload = pickle.load(f)
        model = payload.get("model")
        if model is None:
            raise ValueError(f"Mood predictor pickle at {path} has no 'model' key")
        names = payload.get("feature_names", MOOD_DIMS)
        if set(names) != set(MOOD_DIMS):
            raise ValueError(f"Mood predictor outputs {names}, expected {MOOD_DIMS}")
        self._model = model
        # Column order to reshuffle model output into canonical MOOD_DIMS order.
        self._dim_order = [names.index(dim) for dim in MOOD_DIMS]
        logger.info("Loaded mood predictor from %s", path)

    def predict(self, embeddings: np.ndarray) -> Optional[np.ndarray]:
        """(n, d) embeddings -> (n, 4) moods in MOOD_DIMS order, clipped to [0, 1]."""
        if self._model is None or len(embeddings) == 0:
            return None
        X = np.asarray(embeddings)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        preds = np.clip(self._model.predict(X), 0.0, 1.0)
        return preds[:, self._dim_order].astype(np.float32)
