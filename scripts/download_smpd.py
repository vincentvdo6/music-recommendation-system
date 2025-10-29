"""
Download and preprocess the Spotify Million Playlist Dataset.

The dataset is available at: https://www.aicrowd.com/challenges/spotify-million-playlist-dataset-challenge
Alternatively, use the mirror or torrent from RecSys 2018.

This script:
1. Downloads the SMPD JSON files (if not present)
2. Extracts playlists into a clean format
3. Filters playlists and tracks by quality criteria
4. Saves processed playlists for item2vec training
"""

import json
import logging
import os
from collections import Counter
from pathlib import Path
from typing import List, Dict, Set

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SMPDProcessor:
    """Process Spotify Million Playlist Dataset for item2vec training."""

    def __init__(self, raw_data_dir: str, output_dir: str):
        self.raw_data_dir = Path(raw_data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Quality filters (adjusted for smaller datasets)
        self.min_playlist_length = 5
        self.max_playlist_length = 500
        self.min_track_occurrences = 2  # Filter tracks that appear only once

    def load_raw_playlists(self) -> List[Dict]:
        """Load all JSON slices from SMPD."""
        logger.info(f"Loading playlists from {self.raw_data_dir}")

        all_playlists = []
        json_files = sorted(self.raw_data_dir.glob("mpd.slice.*.json"))

        if not json_files:
            logger.error(f"No SMPD JSON files found in {self.raw_data_dir}")
            logger.info("Please download SMPD from: https://www.aicrowd.com/challenges/spotify-million-playlist-dataset-challenge")
            return []

        for json_file in json_files:
            logger.info(f"Loading {json_file.name}")
            with open(json_file, 'r') as f:
                data = json.load(f)
                all_playlists.extend(data['playlists'])

        logger.info(f"Loaded {len(all_playlists):,} playlists")
        return all_playlists

    def extract_track_sequences(self, playlists: List[Dict]) -> List[List[str]]:
        """Extract track URI sequences from playlists."""
        sequences = []

        for playlist in playlists:
            tracks = playlist.get('tracks', [])

            # Filter by playlist length
            if len(tracks) < self.min_playlist_length:
                continue
            if len(tracks) > self.max_playlist_length:
                continue

            # Extract Spotify track URIs
            track_uris = [track['track_uri'] for track in tracks if 'track_uri' in track]

            if len(track_uris) >= self.min_playlist_length:
                sequences.append(track_uris)

        logger.info(f"Extracted {len(sequences):,} valid playlists")
        return sequences

    def filter_rare_tracks(self, sequences: List[List[str]]) -> List[List[str]]:
        """Remove tracks that appear < min_track_occurrences times."""
        # Count track occurrences
        track_counts = Counter()
        for sequence in sequences:
            track_counts.update(sequence)

        logger.info(f"Total unique tracks: {len(track_counts):,}")

        # Filter tracks
        valid_tracks: Set[str] = {
            track for track, count in track_counts.items()
            if count >= self.min_track_occurrences
        }

        logger.info(f"Tracks after filtering (>={self.min_track_occurrences} occurrences): {len(valid_tracks):,}")

        # Rebuild sequences with only valid tracks
        filtered_sequences = []
        for sequence in sequences:
            filtered = [track for track in sequence if track in valid_tracks]
            if len(filtered) >= self.min_playlist_length:
                filtered_sequences.append(filtered)

        logger.info(f"Playlists after filtering: {len(filtered_sequences):,}")
        return filtered_sequences

    def save_processed_sequences(self, sequences: List[List[str]], filename: str = "playlists.json"):
        """Save processed playlist sequences."""
        output_path = self.output_dir / filename

        logger.info(f"Saving to {output_path}")
        with open(output_path, 'w') as f:
            json.dump(sequences, f)

        logger.info(f"Saved {len(sequences):,} playlists")

        # Save statistics
        stats = {
            "n_playlists": len(sequences),
            "n_tracks": len(set(track for seq in sequences for track in seq)),
            "avg_playlist_length": sum(len(seq) for seq in sequences) / len(sequences),
            "min_playlist_length": min(len(seq) for seq in sequences),
            "max_playlist_length": max(len(seq) for seq in sequences),
        }

        stats_path = self.output_dir / "stats.json"
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)

        logger.info(f"Statistics: {stats}")

    def process(self):
        """Run the full preprocessing pipeline."""
        logger.info("Starting SMPD preprocessing")

        # Load raw data
        playlists = self.load_raw_playlists()
        if not playlists:
            return

        # Extract sequences
        sequences = self.extract_track_sequences(playlists)

        # Filter rare tracks
        filtered_sequences = self.filter_rare_tracks(sequences)

        # Save
        self.save_processed_sequences(filtered_sequences)

        logger.info("Preprocessing complete!")


def main():
    """Main entry point."""
    # Default paths
    raw_data_dir = os.getenv("SMPD_RAW_DIR", "data/smpd/raw")
    output_dir = os.getenv("SMPD_PROCESSED_DIR", "data/smpd/processed")

    processor = SMPDProcessor(raw_data_dir, output_dir)
    processor.process()


if __name__ == "__main__":
    main()
