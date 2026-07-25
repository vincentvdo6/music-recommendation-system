"""Summarize descriptive live recommendation outcomes by serving profile.

Usage: python scripts/report_feedback.py [data/one_rec.sqlite3]
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = ROOT / "data" / "one_rec.sqlite3"
STRONG_POSITIVE = {"like", "save", "more_like_this", "replay"}
NEGATIVE = {"dislike", "skip"}


def _rate(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def build_profile_report(
    impressions: pd.DataFrame,
    exposures: pd.DataFrame,
    feedback: pd.DataFrame,
) -> pd.DataFrame:
    """Return one descriptive outcome row per user-selected profile."""
    if impressions.empty:
        return pd.DataFrame()
    impressions = impressions.copy()
    impressions["profile"] = impressions["profile"].fillna("balanced")
    exposures = exposures.copy()
    if not exposures.empty:
        exposures["profile"] = exposures["profile"].fillna("balanced")
        exposures["catalog_track_id"] = exposures["catalog_track_id"].fillna(
            exposures["track_id"]
        )
    feedback = feedback.copy()
    if not feedback.empty:
        feedback["profile"] = feedback["profile"].fillna("balanced")

    rows = []
    for profile in sorted(impressions["profile"].dropna().unique()):
        profile_impressions = impressions[impressions["profile"] == profile]
        profile_exposures = exposures[exposures["profile"] == profile]
        profile_feedback = feedback[feedback["profile"] == profile]
        n_impressions = int(profile_impressions["impression_id"].nunique())
        n_exposures = len(profile_exposures)
        viewed = profile_feedback[profile_feedback["event"] == "view"]
        positive = profile_feedback[profile_feedback["event"].isin(STRONG_POSITIVE)]
        negative = profile_feedback[profile_feedback["event"].isin(NEGATIVE)]
        neutral = profile_feedback[profile_feedback["event"] == "neutral"]
        similarity_miss = profile_feedback[
            profile_feedback["event"] == "not_similar_enough"
        ]
        session_exposures = profile_exposures.dropna(subset=["session_id"])
        repeats = int(
            session_exposures.duplicated(
                subset=["session_id", "catalog_track_id"], keep="first"
            ).sum()
        )
        rows.append(
            {
                "profile": profile,
                "impressions": n_impressions,
                "sessions": int(profile_impressions["session_id"].dropna().nunique()),
                "exposures": n_exposures,
                "view_rate": _rate(
                    len(viewed.drop_duplicates(["impression_id", "track_id"])),
                    n_exposures,
                ),
                "strong_positive_rate": _rate(
                    len(positive.drop_duplicates(["impression_id", "track_id"])),
                    n_exposures,
                ),
                "negative_rate": _rate(
                    len(negative.drop_duplicates(["impression_id", "track_id"])),
                    n_exposures,
                ),
                "neutral_rate": _rate(
                    len(neutral.drop_duplicates(["impression_id", "track_id"])),
                    n_exposures,
                ),
                "not_similar_enough_per_impression": _rate(
                    len(similarity_miss.drop_duplicates(["impression_id", "track_id"])),
                    n_impressions,
                ),
                "discovery_share": (
                    float(profile_exposures["discovery"].mean()) if n_exposures else 0.0
                ),
                "repeat_exposure_rate": _rate(repeats, len(session_exposures)),
            }
        )
    return pd.DataFrame(rows).set_index("profile")


def load_frames(path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    with sqlite3.connect(path) as db:
        impressions = pd.read_sql_query(
            """
            SELECT impression_id, created_at, session_id, profile, provenance_json
            FROM impressions
            """,
            db,
        )
        exposures = pd.read_sql_query(
            """
            SELECT x.impression_id, i.session_id, i.profile, x.track_id,
                   x.catalog_track_id, x.discovery
            FROM impression_items x
            JOIN impressions i USING (impression_id)
            """,
            db,
        )
        feedback = pd.read_sql_query(
            """
            SELECT f.impression_id, i.profile, f.track_id, f.event
            FROM feedback f
            JOIN impressions i USING (impression_id)
            """,
            db,
        )
    return impressions, exposures, feedback


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DB
    if not path.exists():
        print(f"not found: {path}")
        return 1
    try:
        report = build_profile_report(*load_frames(path))
    except (sqlite3.DatabaseError, pd.errors.DatabaseError) as exc:
        print(f"could not read feedback database: {exc}")
        return 1
    if report.empty:
        print("no recommendation impressions recorded yet")
        return 0
    print(report.round(4).to_string())
    print("\nProfile comparisons are descriptive because listeners choose their style.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
