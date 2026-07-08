"""
MPD-derived track metadata store.

Loads models/meta/track_meta.parquet (exported by the Kaggle training
notebook) with real names, artists, durations and playlist counts for every
track in the item2vec vocabulary. This is what lets ranking see real artist
names and a real popularity prior instead of hash-derived stubs.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

COLUMNS = ["name", "artist", "duration_ms", "playlist_count"]


class TrackMetaStore:
    """Read-only lookup over the MPD track metadata table."""

    def __init__(self, path: Optional[str] = None):
        self._df: Optional[pd.DataFrame] = None
        self._log_pop_denom: float = 1.0
        self._artist_groups: Optional[Dict[str, pd.Index]] = None

        if path and Path(path).exists():
            self.load(path)

    @property
    def available(self) -> bool:
        return self._df is not None

    def load(self, path: str) -> None:
        df = pd.read_parquet(path)
        missing = [c for c in COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"track_meta parquet missing columns: {missing}")
        if "track_id" in df.columns:
            df = df.set_index("track_id")
        df["artist_norm"] = df["artist"].fillna("").str.lower().str.strip()
        self._df = df
        self._log_pop_denom = float(np.log1p(max(df["playlist_count"].max(), 1)))
        logger.info("Loaded track metadata for %d tracks from %s", len(df), path)

    def lookup(self, ids: List[str]) -> pd.DataFrame:
        """Rows aligned to ids; NaN rows for unknown tracks."""
        if self._df is None:
            return pd.DataFrame(index=ids, columns=COLUMNS + ["artist_norm"])
        return self._df.reindex(ids)

    def artists(self, ids: List[str]) -> List[str]:
        """Normalized artist per id, "" where unknown."""
        if self._df is None:
            return [""] * len(ids)
        return self.lookup(ids)["artist_norm"].fillna("").tolist()

    def log_pop(self, ids: List[str]) -> np.ndarray:
        """log1p(playlist_count) / log1p(max), 0 where unknown."""
        if self._df is None:
            return np.zeros(len(ids), dtype=np.float32)
        counts = self.lookup(ids)["playlist_count"].fillna(0).to_numpy(dtype=np.float64)
        return (np.log1p(counts) / self._log_pop_denom).astype(np.float32)

    def durations_ms(self, ids: List[str]) -> np.ndarray:
        if self._df is None:
            return np.zeros(len(ids), dtype=np.float32)
        return self.lookup(ids)["duration_ms"].fillna(0).to_numpy(dtype=np.float32)

    def tracks_by_artist(self, artist: str, limit: int = 20) -> List[str]:
        """Track ids by a (normalized) artist, most popular first."""
        if self._df is None or not artist:
            return []
        if self._artist_groups is None:
            self._artist_groups = self._df.groupby("artist_norm").groups
        index = self._artist_groups.get(artist.lower().strip())
        if index is None:
            return []
        rows = self._df.loc[index].nlargest(limit, "playlist_count")
        return list(rows.index)

    def top_popular(self, n: int = 100) -> List[str]:
        """Most-popular track ids by MPD playlist count (deterministic cold-start)."""
        if self._df is None:
            return []
        return list(self._df.nlargest(n, "playlist_count").index)
