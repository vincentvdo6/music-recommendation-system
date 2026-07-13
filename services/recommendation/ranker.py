"""
Ranking models over the current feature contract (features.FEATURE_NAMES).

LightGBMRanker serves the LambdaRank model trained on Kaggle; its feature
names are asserted against FEATURE_NAMES at load so a stale or mismatched
model file can never silently rank on the wrong columns. LinearFallbackRanker
is a hand-weighted scorer over the same features, used whenever the trained
model is absent or rejected.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from services.recommendation.features import FEATURE_NAMES

logger = logging.getLogger(__name__)


def _top_positive(contribs: np.ndarray, top_k: int) -> List[Tuple[str, float]]:
    """Top features arguing FOR the track — negative contributions would read
    as reasons in the UI ("fits the playlist") even when the signal is absent."""
    order = np.argsort(-contribs)[:top_k]
    out = [(FEATURE_NAMES[i], float(contribs[i])) for i in order if contribs[i] > 0.0]
    if not out:  # degenerate all-zero row: fall back to the least-negative feature
        i = int(np.argmax(contribs))
        out = [(FEATURE_NAMES[i], float(contribs[i]))]
    return out


class LightGBMRanker:
    """LambdaRank model trained by training/kaggle_train_ranker.ipynb."""

    name = "lightgbm-v2"

    def __init__(self, model_path: str):
        import lightgbm as lgb

        if not Path(model_path).exists():
            raise FileNotFoundError(model_path)

        self.model = lgb.Booster(model_file=model_path)
        model_features = self.model.feature_name()
        if model_features != FEATURE_NAMES:
            raise ValueError(
                f"Ranker model features {model_features} do not match the serving "
                f"contract {FEATURE_NAMES} — refusing to load"
            )
        logger.info("Loaded LightGBM ranker (%d trees) from %s", self.model.num_trees(), model_path)

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return self.model.predict(features[FEATURE_NAMES])

    def explain(self, features: pd.DataFrame, top_k: int = 3) -> List[List[Tuple[str, float]]]:
        """Per-row top supporting features via SHAP-style pred_contrib."""
        contribs = self.model.predict(features[FEATURE_NAMES], pred_contrib=True)[:, :-1]  # drop bias column
        return [_top_positive(row, top_k) for row in contribs]


class LinearFallbackRanker:
    """Hand-weighted linear scorer over the current features."""

    name = "linear-fallback"

    # Seed-first philosophy: the searched song dominates (by co-occurrence AND
    # by sound), playlist context and mood coherence follow, popularity is a
    # mild prior.
    DEFAULT_WEIGHTS: Dict[str, float] = {
        "seed_i2v_cos": 0.35,
        "playlist_i2v_cos": 0.12,
        "playlist_i2v_max": 0.07,
        "i2v_seed_rr": 0.05,
        "i2v_playlist_rr": 0.02,
        "ncf_score": 0.08,
        "log_pop": 0.04,
        "artist_in_playlist": 0.02,
        "mood_sim": 0.10,
        "audio_cos_seed": 0.15,
        "audio_seed_rr": 0.03,  # acoustic-channel retrieval strength
    }

    def __init__(self, weights: Dict[str, float] | None = None):
        self.weights = dict(self.DEFAULT_WEIGHTS if weights is None else weights)
        unknown = set(self.weights) - set(FEATURE_NAMES)
        if unknown:
            raise ValueError(f"Fallback weights reference unknown features: {unknown}")
        self._weight_vec = np.array([self.weights.get(name, 0.0) for name in FEATURE_NAMES], dtype=np.float64)

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return features[FEATURE_NAMES].to_numpy(dtype=np.float64) @ self._weight_vec

    def explain(self, features: pd.DataFrame, top_k: int = 3) -> List[List[Tuple[str, float]]]:
        contribs = features[FEATURE_NAMES].to_numpy(dtype=np.float64) * self._weight_vec
        return [_top_positive(row, top_k) for row in contribs]
