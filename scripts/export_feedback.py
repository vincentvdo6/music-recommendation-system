"""Export privacy-minimal recommendation feedback to a training parquet.

The export includes empirical position propensity and clipped
inverse-propensity weights. Items that were returned but never viewed are
retained with ``observed=0`` and must not be treated as negative labels.
Ambiguous dismiss/profile-direction events remain telemetry only.

Usage: python scripts/export_feedback.py [DB_PATH] [OUTPUT_PARQUET]
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = ROOT / "data" / "one_rec.sqlite3"
DEFAULT_OUT = ROOT / "evaluation" / "feedback_training.parquet"


def main() -> int:
    db_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DB
    output = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_OUT
    if not db_path.exists():
        print(f"not found: {db_path}")
        return 1

    with sqlite3.connect(db_path) as db:
        items = pd.read_sql_query(
            """
            SELECT i.impression_id, i.created_at, i.seed_id, i.mode,
                   i.session_id, i.profile,
                   i.provenance_json, i.playlist_json,
                   c.candidate_position, c.served_position,
                   c.served_position AS position, c.track_id,
                   c.track_id AS catalog_track_id, x.track_id AS served_track_id,
                   c.source, c.rank_score, c.similarity_score, c.discovery,
                   c.components_json
            FROM impressions i
            JOIN candidate_items c USING (impression_id)
            LEFT JOIN impression_items x
              ON x.impression_id = c.impression_id
             AND COALESCE(x.catalog_track_id, x.track_id) = c.track_id

            UNION ALL

            SELECT i.impression_id, i.created_at, i.seed_id, i.mode,
                   i.session_id, i.profile,
                   i.provenance_json, i.playlist_json,
                   x.position AS candidate_position, x.position AS served_position,
                   x.position, COALESCE(x.catalog_track_id, x.track_id) AS track_id,
                   COALESCE(x.catalog_track_id, x.track_id) AS catalog_track_id,
                   x.track_id AS served_track_id,
                   x.source, x.rank_score, x.similarity_score, x.discovery,
                   x.components_json
            FROM impressions i
            JOIN impression_items x USING (impression_id)
            WHERE NOT EXISTS (
                SELECT 1 FROM candidate_items c
                WHERE c.impression_id = x.impression_id
                  AND c.track_id = COALESCE(x.catalog_track_id, x.track_id)
            )
            """,
            db,
        )
        events = pd.read_sql_query(
            """
            SELECT f.impression_id,
                   COALESCE(x.catalog_track_id, x.track_id) AS track_id,
                   f.event, f.dwell_ms
            FROM feedback f
            JOIN impression_items x
              ON x.impression_id = f.impression_id AND x.track_id = f.track_id
            """,
            db,
        )

    if items.empty:
        print("no impressions to export")
        return 1

    event_sets = (
        events.groupby(["impression_id", "track_id"])["event"]
        .agg(lambda values: set(values))
        .to_dict()
        if not events.empty
        else {}
    )
    dwell = (
        events.groupby(["impression_id", "track_id"])["dwell_ms"].max().to_dict()
        if not events.empty
        else {}
    )

    labels, observed, eligible, event_json, dwell_ms = [], [], [], [], []
    for row in items.itertuples(index=False):
        key = (row.impression_id, row.track_id)
        seen = event_sets.get(key, set())
        is_observed = bool(seen)
        if seen & {"dislike", "skip"}:
            label = 0
        elif seen & {"like", "save", "more_like_this"}:
            label = 3
        elif seen & {"preview_complete", "replay"}:
            label = 2
        elif seen & {"preview_start", "open_spotify", "open_apple"}:
            label = 1
        else:
            label = 0
        labels.append(label)
        observed.append(int(is_observed))
        eligible.append(int(bool(seen & {
            "preview_start", "preview_complete", "like", "dislike",
            "open_spotify", "open_apple", "skip", "replay", "save",
            "more_like_this",
        })))
        event_json.append(json.dumps(sorted(seen), separators=(",", ":")))
        value = dwell.get(key)
        dwell_ms.append(int(value) if value is not None and not np.isnan(value) else None)

    items["label"] = labels
    items["observed"] = observed
    items["training_eligible"] = eligible
    items["events_json"] = event_json
    items["dwell_ms"] = dwell_ms

    components = items.pop("components_json").map(json.loads).apply(pd.Series)
    components = components.add_prefix("component_")
    items = pd.concat([items, components], axis=1)
    # Estimate P(view | served position) with beta smoothing. This is not a
    # substitute for randomized exploration, but is safer than treating the
    # naturally top-heavy viewed set as unbiased.
    by_position = items[items["served_position"].notna()].groupby("position")["observed"].agg(
        ["sum", "count"]
    )
    propensity = ((by_position["sum"] + 1.0) / (by_position["count"] + 2.0)).to_dict()
    items["position_propensity"] = items["position"].map(propensity).fillna(1.0).astype(float)
    items["ips_weight"] = np.minimum(10.0, 1.0 / np.maximum(items["position_propensity"], 0.05))

    output.parent.mkdir(parents=True, exist_ok=True)
    items.to_parquet(output, index=False)
    print(
        f"{len(items):,} candidate items -> {output} | "
        f"observed={items['observed'].mean():.1%} | "
        f"trainable={items['training_eligible'].sum():,} | positive={(items['label'] > 0).sum():,}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
