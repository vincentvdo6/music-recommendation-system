"""Curated catalogue loader that powers the context-aware recommendation engine."""

from __future__ import annotations

import json
import logging
from copy import deepcopy
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)


def _normalize(text: str) -> str:
    """Case-fold and strip helper used for fuzzy catalogue matching."""
    return " ".join(text.casefold().split()) if text else ""


class TrackCatalogue:
    """Loads a curated track catalogue with rich semantic metadata."""

    def __init__(self, catalogue_path: Optional[Path | str] = None) -> None:
        root = Path(__file__).resolve().parents[2]
        default_path = root / "data" / "catalogue" / "tracks.json"
        self._catalogue_path = Path(catalogue_path or default_path)

        if not self._catalogue_path.exists():
            raise FileNotFoundError(f"Catalogue not found at {self._catalogue_path}")

        self._tracks: List[Dict] = []
        self._id_index: Dict[str, Dict] = {}
        self._name_index: Dict[Tuple[str, str], Dict] = {}

        self._load_catalogue()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def all_tracks(self) -> List[Dict]:
        """Return copies of all catalogue entries."""
        return [deepcopy(track) for track in self._tracks]

    def size(self) -> int:
        """Number of tracks available in the curated catalogue."""
        return len(self._tracks)

    def get_by_id(self, track_id: str) -> Optional[Dict]:
        """Return a single entry looked up by identifier."""
        if not track_id:
            return None
        entry = self._id_index.get(track_id)
        return deepcopy(entry) if entry else None

    def find_best_match(self, name: Optional[str], artist: Optional[str] = None) -> Optional[Dict]:
        """Find the best matching entry for the supplied metadata."""
        if not name:
            return None

        key = (_normalize(name), _normalize(artist or ""))
        direct = self._name_index.get(key)
        if direct:
            return deepcopy(direct)

        # Fallback: search by name only and pick the most popular entry
        name_only_matches = [
            track for (track_name, _), track in self._name_index.items() if track_name == key[0]
        ]

        if name_only_matches:
            best = max(name_only_matches, key=lambda t: t.get("popularity", 0))
            return deepcopy(best)

        # Lightweight fuzzy matching by prefix/contains for resiliance to seed differences
        candidates: List[Dict] = []
        name_norm = key[0]
        for (track_name, artist_name), track in self._name_index.items():
            if name_norm in track_name or track_name in name_norm:
                if artist and artist_name and artist_name == key[1]:
                    return deepcopy(track)
                candidates.append(track)

        if not candidates:
            return None

        # Prefer artist overlap if available, else fall back to popularity
        if artist:
            artist_norm = _normalize(artist)
            artist_matches = [t for t in candidates if _normalize(t.get("artist", "")) == artist_norm]
            if artist_matches:
                return deepcopy(max(artist_matches, key=lambda t: t.get("popularity", 0)))

        return deepcopy(max(candidates, key=lambda t: t.get("popularity", 0)))

    def iterate(self) -> Iterable[Dict]:
        """Yield copies of catalogue entries (useful for streaming large catalogues)."""
        for track in self._tracks:
            yield deepcopy(track)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _load_catalogue(self) -> None:
        try:
            with self._catalogue_path.open("r", encoding="utf-8") as fh:
                payload = json.load(fh)
        except json.JSONDecodeError as exc:  # pragma: no cover - configuration error
            raise ValueError(f"Invalid catalogue JSON: {exc}") from exc

        if not isinstance(payload, list):
            raise ValueError("Catalogue payload must be a list of track objects")

        normalised: List[Dict] = []
        skipped = 0

        for raw in payload:
            entry = self._normalise_entry(raw)
            if not entry:
                skipped += 1
                continue
            normalised.append(entry)

        if skipped:
            logger.warning("Skipped %d malformed catalogue entries", skipped)

        self._tracks = normalised
        self._id_index = {track["id"]: track for track in self._tracks}
        self._name_index = {
            (_normalize(track.get("name", "")), _normalize(track.get("artist", ""))): track
            for track in self._tracks
        }

        logger.info("Loaded %d curated tracks from %s", len(self._tracks), self._catalogue_path)

    @staticmethod
    def _normalise_entry(raw: Dict) -> Optional[Dict]:
        """Ensure an entry has the required shape and sensible defaults."""
        if not isinstance(raw, dict):
            return None

        track_id = raw.get("id")
        name = raw.get("name")
        artist = raw.get("artist")

        if not track_id or not name or not artist:
            return None

        entry = deepcopy(raw)
        entry.setdefault("provider", "catalog")
        entry.setdefault("uri", track_id)
        entry.setdefault("duration_ms", 0)
        entry.setdefault("popularity", 50)
        entry.setdefault("release_year", None)
        entry.setdefault("preview_url", "")
        entry.setdefault("image_url", "")
        entry.setdefault("external_urls", {})

        audio_features = entry.get("audio_features") or {}
        if not isinstance(audio_features, dict):
            audio_features = {}
        entry["audio_features"] = audio_features

        tags = entry.get("tags") or {}
        if not isinstance(tags, dict):
            tags = {}
        tags.setdefault("genres", [])
        tags.setdefault("moods", [])
        tags.setdefault("activities", [])
        tags.setdefault("time_of_day", [])
        tags.setdefault("regions", [])
        entry["tags"] = tags

        return entry
