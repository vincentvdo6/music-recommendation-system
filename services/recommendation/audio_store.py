"""
Preview-derived audio embedding store.

Loads models/audio/audio_emb.parquet (exported by
training/kaggle_audio_embeddings.ipynb): one row per track with a 256-dim
fp16 Discogs-EffNet embedding (PCA-reduced) serialized as bytes. These are
the "ears" of the recommender — cosine similarity here measures what tracks
actually SOUND like, independent of playlist co-occurrence.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class AudioStore:
    """Read-only lookup over unit-normalized preview embeddings."""

    def __init__(self, path: Optional[str] = None):
        self._matrix: Optional[np.ndarray] = None
        self._row_of: Dict[str, int] = {}
        self._ids: List[str] = []

        if path and Path(path).exists():
            self.load(path)

    @property
    def available(self) -> bool:
        return self._matrix is not None

    @property
    def dim(self) -> int:
        return self._matrix.shape[1] if self._matrix is not None else 0

    @property
    def size(self) -> int:
        return len(self._row_of)

    def load(self, path: str) -> None:
        df = pd.read_parquet(path)
        if not {"track_id", "embedding"}.issubset(df.columns):
            raise ValueError(f"audio embedding parquet at {path} missing track_id/embedding columns")
        matrix = np.stack([
            np.frombuffer(blob, dtype=np.float16).astype(np.float32)
            for blob in df["embedding"]
        ])
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self._matrix = matrix / norms
        self._ids = list(df["track_id"])
        self._row_of = {tid: i for i, tid in enumerate(self._ids)}
        logger.info("Loaded %d audio embeddings (dim=%d) from %s", len(df), self.dim, path)

    def load_extension(self, path: str) -> None:
        """Append extension-catalog embeddings (same PCA space); base rows win
        on id conflict, so re-loading never clobbers the canonical vectors."""
        if self._matrix is None:
            self.load(path)
            return
        df = pd.read_parquet(path)
        if not {"track_id", "embedding"}.issubset(df.columns):
            raise ValueError(f"audio embedding parquet at {path} missing track_id/embedding columns")
        df = df[~df["track_id"].isin(self._row_of)]
        if not len(df):
            return
        matrix = np.stack([
            np.frombuffer(blob, dtype=np.float16).astype(np.float32)
            for blob in df["embedding"]
        ])
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        offset = len(self._ids)
        self._matrix = np.concatenate([self._matrix, matrix / norms])
        new_ids = list(df["track_id"])
        self._ids.extend(new_ids)
        self._row_of.update({tid: offset + i for i, tid in enumerate(new_ids)})
        logger.info("Extended audio store with %d embeddings from %s (total %d)",
                    len(new_ids), path, len(self._ids))

    def nearest(self, vector: np.ndarray, k: int = 100, exclude: Optional[set] = None) -> List[str]:
        """Cosine top-k track ids for a query vector (exact, brute-force —
        a few hundred thousand 256-dim unit rows is a single fast matmul)."""
        if self._matrix is None:
            return []
        q = np.asarray(vector, dtype=np.float32)
        q = q / (np.linalg.norm(q) or 1.0)
        sims = self._matrix @ q
        exclude = exclude or set()
        take = min(k + len(exclude), len(sims))
        idx = np.argpartition(-sims, take - 1)[:take]
        idx = idx[np.argsort(-sims[idx])]
        out = [self._ids[i] for i in idx if self._ids[i] not in exclude]
        return out[:k]

    def vector(self, track_id: str) -> Optional[np.ndarray]:
        row = self._row_of.get(track_id)
        return self._matrix[row] if row is not None else None

    def matrix_for(self, ids: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """(n, dim) unit vectors with zero rows for unknown ids, plus a 0/1 mask."""
        out = np.zeros((len(ids), self.dim), dtype=np.float32)
        mask = np.zeros(len(ids), dtype=np.float32)
        for i, tid in enumerate(ids):
            row = self._row_of.get(tid)
            if row is not None:
                out[i] = self._matrix[row]
                mask[i] = 1.0
        return out, mask

    def mean_vector(self, ids: List[str]) -> Optional[np.ndarray]:
        rows = [self._row_of[t] for t in ids if t in self._row_of]
        if not rows:
            return None
        return self._matrix[rows].mean(axis=0)
