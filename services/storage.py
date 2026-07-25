"""Small local persistence layer for identities, impressions, and feedback.

The database intentionally stores no IP address, cookie, account id, or other
user identifier.  It captures only the recommendation context and explicit or
behavioral events needed to build a better ranking dataset.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

FEEDBACK_EVENTS = {
    "view",
    "dismiss",
    "preview_start",
    "preview_complete",
    "like",
    "neutral",
    "dislike",
    "open_spotify",
    "open_apple",
    "skip",
    "replay",
    "save",
    "more_like_this",
    "not_similar_enough",
}

SESSION_EVENT_WEIGHTS = {
    "dismiss": -0.20,
    "skip": -0.60,
    "dislike": -1.00,
    "preview_start": 0.10,
    "preview_complete": 0.55,
    "open_spotify": 0.45,
    "open_apple": 0.45,
    "replay": 0.80,
    "like": 1.00,
    "save": 1.20,
    "more_like_this": 1.20,
}


class LocalStore:
    """Thread-safe SQLite store shared by the API and music service."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._db = sqlite3.connect(self.path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        with self._lock:
            self._db.execute("PRAGMA busy_timeout=5000")
            self._db.execute("PRAGMA journal_mode=WAL")
            self._db.execute("PRAGMA foreign_keys=ON")
            self._db.executescript(
                """
                CREATE TABLE IF NOT EXISTS extension_identity (
                    catalog_id TEXT PRIMARY KEY,
                    resolved_json TEXT,
                    updated_at REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS impressions (
                    impression_id TEXT PRIMARY KEY,
                    request_id TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    seed_id TEXT,
                    mode TEXT NOT NULL,
                    playlist_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS impression_items (
                    impression_id TEXT NOT NULL,
                    position INTEGER NOT NULL,
                    track_id TEXT NOT NULL,
                    source TEXT NOT NULL,
                    rank_score REAL,
                    similarity_score REAL,
                    discovery INTEGER NOT NULL DEFAULT 0,
                    components_json TEXT NOT NULL,
                    PRIMARY KEY (impression_id, position),
                    UNIQUE (impression_id, track_id),
                    FOREIGN KEY (impression_id) REFERENCES impressions(impression_id)
                        ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS feedback (
                    impression_id TEXT NOT NULL,
                    track_id TEXT NOT NULL,
                    event TEXT NOT NULL,
                    position INTEGER,
                    dwell_ms INTEGER,
                    created_at REAL NOT NULL,
                    PRIMARY KEY (impression_id, track_id, event),
                    FOREIGN KEY (impression_id, track_id)
                        REFERENCES impression_items(impression_id, track_id)
                        ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS candidate_items (
                    impression_id TEXT NOT NULL,
                    candidate_position INTEGER NOT NULL,
                    track_id TEXT NOT NULL,
                    source TEXT NOT NULL,
                    rank_score REAL,
                    similarity_score REAL,
                    discovery INTEGER NOT NULL DEFAULT 0,
                    components_json TEXT NOT NULL,
                    served_position INTEGER,
                    PRIMARY KEY (impression_id, candidate_position),
                    UNIQUE (impression_id, track_id),
                    FOREIGN KEY (impression_id) REFERENCES impressions(impression_id)
                        ON DELETE CASCADE
                );
                """
            )
            self._ensure_column("impressions", "session_id", "TEXT")
            self._ensure_column("impressions", "profile", "TEXT NOT NULL DEFAULT 'familiar'")
            self._ensure_column("impressions", "provenance_json", "TEXT NOT NULL DEFAULT '{}'")
            self._ensure_column("impression_items", "catalog_track_id", "TEXT")
            self._db.executescript(
                """
                CREATE INDEX IF NOT EXISTS idx_impressions_session_created
                    ON impressions(session_id, created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_feedback_created
                    ON feedback(created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_impression_items_catalog
                    ON impression_items(impression_id, catalog_track_id);
                CREATE INDEX IF NOT EXISTS idx_candidate_items_served
                    ON candidate_items(impression_id, served_position);
                """
            )
            self._db.commit()

    def _ensure_column(self, table: str, column: str, declaration: str) -> None:
        columns = {row["name"] for row in self._db.execute(f"PRAGMA table_info({table})")}
        if column not in columns:
            self._db.execute(f"ALTER TABLE {table} ADD COLUMN {column} {declaration}")

    def close(self) -> None:
        with self._lock:
            self._db.close()

    # ------------------------------------------------------ identity cache

    def get_extension_identity(self, catalog_id: str) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """Return ``(cached, track)``; a cached ``None`` is a negative result."""
        with self._lock:
            row = self._db.execute(
                "SELECT resolved_json FROM extension_identity WHERE catalog_id = ?",
                (catalog_id,),
            ).fetchone()
        if row is None:
            return False, None
        payload = row["resolved_json"]
        return True, json.loads(payload) if payload else None

    def put_extension_identity(self, catalog_id: str, track: Optional[Dict[str, Any]]) -> None:
        payload = json.dumps(track, separators=(",", ":"), sort_keys=True) if track else None
        with self._lock:
            self._db.execute(
                """
                INSERT INTO extension_identity(catalog_id, resolved_json, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(catalog_id) DO UPDATE SET
                    resolved_json=excluded.resolved_json,
                    updated_at=excluded.updated_at
                """,
                (catalog_id, payload, time.time()),
            )
            self._db.commit()

    # --------------------------------------------------------- impressions

    def record_impression(
        self,
        *,
        impression_id: str,
        request_id: str,
        seed_id: Optional[str],
        mode: str,
        playlist_ids: Iterable[str],
        source: str,
        recommendations: Iterable[Dict[str, Any]],
        candidates: Optional[Iterable[Dict[str, Any]]] = None,
        session_id: Optional[str] = None,
        profile: str = "familiar",
        provenance: Optional[Dict[str, Any]] = None,
    ) -> None:
        playlist = [str(track_id) for track_id in playlist_ids if track_id]
        rows = []
        displayed = list(recommendations)
        for position, track in enumerate(displayed):
            recommendation = track.get("recommendation") or {}
            rows.append(
                (
                    impression_id,
                    position,
                    str(track["id"]),
                    str(track.get("catalog_id") or track["id"]),
                    source,
                    _optional_float(track.get("rank_score")),
                    _optional_float(track.get("similarity_score")),
                    int(bool(track.get("discovery"))),
                    json.dumps(recommendation.get("components") or {}, separators=(",", ":"), sort_keys=True),
                )
            )

        served_positions = {
            str(track.get("catalog_id") or track["id"]): position
            for position, track in enumerate(displayed)
        }
        candidate_rows = []
        for position, track in enumerate(candidates or displayed):
            recommendation = track.get("recommendation") or {}
            track_id = str(track.get("catalog_id") or track["id"])
            candidate_rows.append((
                impression_id,
                position,
                track_id,
                source,
                _optional_float(track.get("rank_score")),
                _optional_float(track.get("similarity_score")),
                int(bool(track.get("discovery"))),
                json.dumps(recommendation.get("components") or {}, separators=(",", ":"), sort_keys=True),
                served_positions.get(track_id),
            ))

        with self._lock:
            self._db.execute(
                """
                INSERT INTO impressions(
                    impression_id, request_id, created_at, seed_id, mode, playlist_json,
                    session_id, profile, provenance_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    impression_id,
                    request_id,
                    time.time(),
                    seed_id,
                    mode,
                    json.dumps(playlist, separators=(",", ":")),
                    session_id,
                    profile,
                    json.dumps(provenance or {}, separators=(",", ":"), sort_keys=True),
                ),
            )
            self._db.executemany(
                """
                INSERT INTO impression_items(
                    impression_id, position, track_id, catalog_track_id, source, rank_score,
                    similarity_score, discovery, components_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
            self._db.executemany(
                """
                INSERT INTO candidate_items(
                    impression_id, candidate_position, track_id, source, rank_score,
                    similarity_score, discovery, components_json, served_position
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                candidate_rows,
            )
            self._db.commit()

    def record_feedback(
        self,
        *,
        impression_id: str,
        track_id: str,
        event: str,
        position: Optional[int] = None,
        dwell_ms: Optional[int] = None,
    ) -> bool:
        if event not in FEEDBACK_EVENTS:
            raise ValueError(f"unsupported feedback event: {event}")
        with self._lock:
            known = self._db.execute(
                """
                SELECT 1 FROM impression_items
                WHERE impression_id = ? AND track_id = ?
                """,
                (impression_id, track_id),
            ).fetchone()
            if known is None:
                return False
            self._db.execute(
                """
                INSERT INTO feedback(
                    impression_id, track_id, event, position, dwell_ms, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(impression_id, track_id, event) DO UPDATE SET
                    position=COALESCE(excluded.position, feedback.position),
                    dwell_ms=MAX(COALESCE(excluded.dwell_ms, 0), COALESCE(feedback.dwell_ms, 0)),
                    created_at=excluded.created_at
                """,
                (impression_id, track_id, event, position, dwell_ms, time.time()),
            )
            self._db.commit()
        return True

    def recent_session_signals(
        self,
        session_id: Optional[str],
        *,
        limit: int = 100,
        max_age_seconds: int = 7 * 24 * 60 * 60,
        half_life_seconds: int = 24 * 60 * 60,
    ) -> list[Tuple[str, float]]:
        """Aggregate bounded track-level preference signals for one browser session."""
        if not session_id:
            return []
        with self._lock:
            rows = self._db.execute(
                """
                SELECT COALESCE(x.catalog_track_id, x.track_id) AS track_id,
                       f.event, f.created_at
                FROM feedback f
                JOIN impressions i USING (impression_id)
                JOIN impression_items x
                  ON x.impression_id = f.impression_id AND x.track_id = f.track_id
                WHERE i.session_id = ? AND f.created_at >= ?
                ORDER BY f.created_at DESC
                LIMIT ?
                """,
                (session_id, time.time() - max_age_seconds, max(1, int(limit))),
            ).fetchall()
        now = time.time()
        half_life = max(1, int(half_life_seconds))
        totals: Dict[str, float] = {}
        for row in rows:
            weight = SESSION_EVENT_WEIGHTS.get(row["event"], 0.0)
            if weight:
                age = max(0.0, now - float(row["created_at"]))
                decay = 0.5 ** (age / half_life)
                totals[row["track_id"]] = totals.get(row["track_id"], 0.0) + weight * decay
        return [
            (track_id, max(-1.5, min(1.5, weight)))
            for track_id, weight in totals.items()
            if weight
        ]

    def recent_session_exclusions(
        self,
        session_id: Optional[str],
        *,
        limit: int = 200,
        max_age_seconds: int = 24 * 60 * 60,
    ) -> list[str]:
        """Return recently displayed catalog ids so refreshed slates rotate."""
        if not session_id:
            return []
        with self._lock:
            rows = self._db.execute(
                """
                SELECT COALESCE(x.catalog_track_id, x.track_id) AS track_id,
                       MAX(i.created_at) AS last_seen
                FROM impressions i
                JOIN impression_items x USING (impression_id)
                WHERE i.session_id = ? AND i.created_at >= ?
                GROUP BY COALESCE(x.catalog_track_id, x.track_id)
                ORDER BY last_seen DESC
                LIMIT ?
                """,
                (session_id, time.time() - max_age_seconds, max(1, int(limit))),
            ).fetchall()
        return [str(row["track_id"]) for row in rows]


def _optional_float(value: Any) -> Optional[float]:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None
