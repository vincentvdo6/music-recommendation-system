"""
Collect public Spotify playlists to build your own training dataset.

Uses Spotify Web API to fetch featured/category playlists.
"""

import json
import asyncio
import logging
from pathlib import Path
from typing import List, Dict
import random

import sys
import os

# Load environment variables BEFORE importing SpotifyClient
from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from services.spotify.client import SpotifyClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SpotifyPlaylistCollector:
    """Collect playlists from Spotify for training."""

    def __init__(self, spotify_client: SpotifyClient):
        self.spotify = spotify_client
        self.collected_playlists = []

    async def collect_from_categories(self, n_playlists_per_category: int = 50):
        """Collect playlists from Spotify categories."""
        logger.info("Fetching Spotify categories...")

        try:
            response = await self.spotify._request(
                "GET",
                "https://api.spotify.com/v1/browse/categories",
                params={"limit": 50}
            )
            data = response.json()

            categories = data.get("categories", {}).get("items", [])
            logger.info(f"Found {len(categories)} categories")

            for category in categories:
                category_id = category["id"]
                category_name = category["name"]

                logger.info(f"Fetching playlists from category: {category_name}")

                try:
                    cat_response = await self.spotify._request(
                        "GET",
                        f"https://api.spotify.com/v1/browse/categories/{category_id}/playlists",
                        params={"limit": 50}
                    )
                    cat_data = cat_response.json()

                    playlists = cat_data.get("playlists", {}).get("items", [])

                    sampled = random.sample(
                        playlists,
                        min(n_playlists_per_category, len(playlists))
                    )

                    for playlist_item in sampled:
                        if not playlist_item:
                            continue
                        playlist_id = playlist_item.get("id")
                        if playlist_id:
                            await self._fetch_playlist_tracks(playlist_id, category_name)

                except Exception as e:
                    logger.warning(f"Failed to fetch category {category_name}: {e}")
                    continue

                await asyncio.sleep(0.5)

        except Exception as e:
            logger.error(f"Failed to fetch categories: {e}")

    async def collect_from_featured(self, limit: int = 50):
        """Collect from Spotify's featured playlists."""
        logger.info("Fetching featured playlists...")

        try:
            response = await self.spotify._request(
                "GET",
                "https://api.spotify.com/v1/browse/featured-playlists",
                params={"limit": limit}
            )
            data = response.json()

            playlists = data.get("playlists", {}).get("items", [])
            logger.info(f"Found {len(playlists)} featured playlists")

            for playlist_item in playlists:
                if not playlist_item:
                    continue
                playlist_id = playlist_item.get("id")
                if playlist_id:
                    await self._fetch_playlist_tracks(playlist_id, "featured")

                await asyncio.sleep(0.5)

        except Exception as e:
            logger.error(f"Failed to fetch featured playlists: {e}")

    async def collect_from_user_playlists(self, user_ids: List[str]):
        """Collect from specific Spotify users' public playlists."""
        for user_id in user_ids:
            logger.info(f"Fetching playlists for user: {user_id}")

            try:
                response = await self.spotify._request(
                    "GET",
                    f"https://api.spotify.com/v1/users/{user_id}/playlists",
                    params={"limit": 50}
                )
                data = response.json()

                playlists = data.get("items", [])

                for playlist_item in playlists:
                    if not playlist_item:
                        continue

                    if not playlist_item.get("public", False):
                        continue

                    playlist_id = playlist_item.get("id")
                    if playlist_id:
                        await self._fetch_playlist_tracks(playlist_id, f"user:{user_id}")

                    await asyncio.sleep(0.5)

            except Exception as e:
                logger.warning(f"Failed to fetch user {user_id}: {e}")

    async def _fetch_playlist_tracks(self, playlist_id: str, category: str):
        """Fetch tracks for a single playlist using SpotifyClient method."""
        try:
            # Use the existing get_playlist_tracks method
            tracks = await self.spotify.get_playlist_tracks(playlist_id)

            if not tracks or len(tracks) < 5:
                return

            # Convert to SMPD-like format
            playlist_data = {
                "name": f"Playlist {playlist_id}",
                "collaborative": False,
                "pid": len(self.collected_playlists),
                "category": category,
                "num_tracks": len(tracks),
                "tracks": []
            }

            for idx, track in enumerate(tracks):
                track_data = {
                    "pos": idx,
                    "artist_name": track.get("artist", "Unknown"),
                    "track_uri": track.get("uri", ""),
                    "artist_uri": track.get("artist_uri", ""),
                    "track_name": track.get("name", ""),
                    "album_uri": track.get("album_uri", ""),
                    "duration_ms": track.get("duration_ms", 0),
                    "album_name": track.get("album", "Unknown"),
                }
                playlist_data["tracks"].append(track_data)

            self.collected_playlists.append(playlist_data)

            if len(self.collected_playlists) % 50 == 0:
                logger.info(f"Collected {len(self.collected_playlists)} playlists so far...")

        except Exception as e:
            logger.warning(f"Failed to fetch playlist {playlist_id}: {e}")

    def save_playlists(self, output_dir: str = "data/smpd/raw"):
        """Save collected playlists in SMPD format."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        playlists_per_slice = 1000
        n_slices = (len(self.collected_playlists) + playlists_per_slice - 1) // playlists_per_slice

        for slice_idx in range(n_slices):
            start = slice_idx * playlists_per_slice
            end = min(start + playlists_per_slice, len(self.collected_playlists))

            slice_playlists = self.collected_playlists[start:end]

            slice_file = output_path / f"mpd.slice.{slice_idx}.json"

            with open(slice_file, 'w') as f:
                json.dump({
                    "info": {
                        "version": "v1-spotify-collected",
                        "generated_on": "2024-01-01",
                        "slice": slice_idx,
                    },
                    "playlists": slice_playlists
                }, f)

            logger.info(f"Saved {slice_file.name} with {len(slice_playlists)} playlists")

        logger.info(f"\n✓ Saved {len(self.collected_playlists)} playlists to {output_dir}")


async def main():
    """Main collection pipeline."""
    # Initialize Spotify client
    spotify = SpotifyClient()
    await spotify.start()  # Must call start() to initialize HTTP client

    if spotify.demo_mode:
        logger.error("Spotify credentials not configured!")
        logger.info("Set SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET in .env")
        return

    try:
        collector = SpotifyPlaylistCollector(spotify)

        # Strategy 1: Collect from categories
        logger.info("=" * 60)
        logger.info("Strategy 1: Collecting from categories...")
        logger.info("=" * 60)
        await collector.collect_from_categories(n_playlists_per_category=20)

        # Strategy 2: Collect from featured playlists
        logger.info("\n" + "=" * 60)
        logger.info("Strategy 2: Collecting featured playlists...")
        logger.info("=" * 60)
        await collector.collect_from_featured(limit=50)

        # Strategy 3: Collect from known curators
        curators = [
            "spotify",
            "digster",
        ]

        logger.info("\n" + "=" * 60)
        logger.info("Strategy 3: Collecting from curators...")
        logger.info("=" * 60)
        await collector.collect_from_user_playlists(curators)

        # Save results
        logger.info("\n" + "=" * 60)
        logger.info("Saving collected playlists...")
        logger.info("=" * 60)
        collector.save_playlists(output_dir="data/smpd/raw")

        logger.info("\n" + "=" * 60)
        logger.info(f"✓ Collection complete!")
        logger.info(f"Total playlists: {len(collector.collected_playlists)}")
        logger.info("=" * 60)
        logger.info("\nNext steps:")
        logger.info("1. Run: python3 scripts/download_smpd.py")
        logger.info("2. Run: python3 scripts/train_item2vec.py")

    finally:
        await spotify.close()


if __name__ == "__main__":
    asyncio.run(main())
