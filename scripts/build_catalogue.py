#!/usr/bin/env python3
"""Generate a curated catalogue JSON using live Spotify metadata."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.spotify.client import SpotifyClient  # noqa: E402
from services.recommendation.playlist_builder import build_entries  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build contextual catalogue from Spotify data")
    parser.add_argument(
        "--playlist",
        dest="playlists",
        action="append",
        default=[],
        help="Spotify playlist ID (e.g. 37i9dQZF1DXcBWIGoYBM5M). Can be supplied multiple times.",
    )
    parser.add_argument(
        "--track",
        dest="tracks",
        action="append",
        default=[],
        help="Explicit Spotify track ID to include (e.g. 4uLU6hMCjMI75M1A2tKUQC).",
    )
    parser.add_argument(
        "--query",
        dest="queries",
        action="append",
        default=[],
        help="Free-text search query to pull top tracks for (uses Spotify search).",
    )
    parser.add_argument(
        "--search-limit",
        type=int,
        default=25,
        help="Max tracks to include per query search (default: 25).",
    )
    parser.add_argument(
        "--playlist-limit",
        type=int,
        default=200,
        help="Max tracks to pull per playlist (default: 200).",
    )
    parser.add_argument(
        "--min-popularity",
        type=int,
        default=0,
        help="Drop tracks with Spotify popularity below this threshold (0-100).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data" / "catalogue" / "tracks.json",
        help="Path to write the curated catalogue JSON (default: data/catalogue/tracks.json).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting an existing catalogue file (otherwise abort).",
    )
    return parser.parse_args()
async def gather_tracks(
    client: SpotifyClient,
    playlists: Sequence[str],
    queries: Sequence[str],
    explicit_tracks: Sequence[str],
    *,
    playlist_limit: int,
    search_limit: int,
) -> Dict[str, Dict[str, Any]]:
    collected: Dict[str, Dict[str, Any]] = {}

    for playlist_id in playlists:
        playlist_id = playlist_id.strip()
        if not playlist_id:
            continue
        tracks = await client.get_playlist_tracks(playlist_id, limit=100, max_tracks=playlist_limit)
        for track in tracks:
            collected[track["id"]] = track

    for query in queries:
        query = query.strip()
        if not query:
            continue
        results = await client.search_tracks(query, limit=search_limit)
        for track in results:
            collected[track["id"]] = track

    if explicit_tracks:
        lookup = await client.get_tracks_bulk(list(explicit_tracks))
        for track_id, track in lookup.items():
            collected[track_id] = track

    return collected
async def build_catalogue(args: argparse.Namespace) -> None:
    load_dotenv()

    client = SpotifyClient()
    await client.start()

    try:
        tracks_map = await gather_tracks(
            client,
            args.playlists,
            args.queries,
            args.tracks,
            playlist_limit=args.playlist_limit,
            search_limit=args.search_limit,
        )

        if not tracks_map:
            raise SystemExit("No tracks gathered – provide playlists, queries, or track IDs.")

        track_ids = list(tracks_map.keys())
        features_map = await client.get_tracks_features_bulk(track_ids)

        artist_ids: List[str] = []
        for track in tracks_map.values():
            artist_ids.extend(track.get("metadata", {}).get("spotify_artist_ids", []))

        artist_map = await client.get_artists(artist_ids)

        entries = build_entries(
            tracks_map,
            features_map,
            artist_map,
            min_popularity=args.min_popularity,
        )

        if not entries:
            raise SystemExit("No catalogue entries left after filtering – adjust inputs.")

        entries.sort(key=lambda item: (item["artist"].lower(), item["name"].lower()))

        output_path: Path = args.output
        if output_path.exists() and not args.overwrite:
            raise SystemExit(f"Refusing to overwrite existing file: {output_path}. Use --overwrite to replace it.")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as fh:
            json.dump(entries, fh, indent=2, ensure_ascii=False)

        print(f"✅ Wrote {len(entries)} tracks to {output_path}")
    finally:
        await client.close()


def main() -> None:
    args = parse_args()
    asyncio.run(build_catalogue(args))


if __name__ == "__main__":
    main()
