"""
Download and preprocess the Spotify Million Playlist Dataset.

The dataset is available at: https://www.aicrowd.com/challenges/spotify-million-playlist-dataset-challenge
Alternatively, use the mirror or torrent from RecSys 2018.

This script:
1. Extracts the SMPD zip file (if present)
2. Loads the JSON files from the extracted data
3. Filters playlists and tracks by quality criteria
4. Saves processed playlists for item2vec training
"""

import json
import logging
import os
import zipfile
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

        # Quality filters (optimized for SMPD scale)
        self.min_playlist_length = 5
        self.max_playlist_length = 500
        self.min_track_occurrences = 2  # Lower threshold for niche discovery

        # NICHE DISCOVERY: Include tracks from fewer playlists
        # BUT filter trash by playlist context (exclude low-quality playlists)

    def extract_zip_if_needed(self) -> bool:
        """Extract SMPD zip file if it exists and hasn't been extracted yet."""
        # Check for the zip file in the project root
        zip_path = Path("spotify_million_playlist_dataset.zip")

        if not zip_path.exists():
            logger.warning(f"SMPD zip file not found: {zip_path}")
            return False

        # Check if we already have extracted files
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        json_files = list(self.raw_data_dir.glob("mpd.slice.*.json"))

        if json_files:
            logger.info(f"Found {len(json_files)} extracted JSON files, skipping extraction")
            return True

        # Extract the zip file
        logger.info(f"Extracting {zip_path} ({zip_path.stat().st_size / 1024 / 1024 / 1024:.1f} GB)")
        logger.info("This may take several minutes...")

        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # List contents first
                file_list = zip_ref.namelist()
                json_files_in_zip = [f for f in file_list if f.endswith('.json')]
                logger.info(f"Found {len(json_files_in_zip)} JSON files in zip")

                # Extract only JSON files to raw data directory
                for file_name in json_files_in_zip:
                    logger.info(f"Extracting {file_name}")
                    # Extract to raw_data_dir, stripping any subdirectories
                    base_name = os.path.basename(file_name)
                    if base_name.startswith("mpd.slice."):
                        target_path = self.raw_data_dir / base_name
                        with zip_ref.open(file_name) as source, open(target_path, 'wb') as target:
                            target.write(source.read())

                logger.info(f"Extraction complete! Files saved to {self.raw_data_dir}")
                return True

        except Exception as e:
            logger.error(f"Failed to extract zip: {e}")
            return False

    def load_raw_playlists(self) -> List[Dict]:
        """Load all JSON slices from SMPD."""
        logger.info(f"Loading playlists from {self.raw_data_dir}")

        all_playlists = []
        json_files = sorted(self.raw_data_dir.glob("mpd.slice.*.json"))

        if not json_files:
            logger.error(f"No SMPD JSON files found in {self.raw_data_dir}")
            logger.info("Please ensure the SMPD zip file is in the project root")
            return []

        # Load in batches to track progress
        total_files = len(json_files)
        for idx, json_file in enumerate(json_files):
            logger.info(f"Loading {json_file.name} ({idx + 1}/{total_files})")
            with open(json_file, 'r') as f:
                data = json.load(f)
                all_playlists.extend(data['playlists'])

            # Log progress every 100 files
            if (idx + 1) % 100 == 0:
                logger.info(f"Progress: Loaded {len(all_playlists):,} playlists so far...")

        logger.info(f"Loaded {len(all_playlists):,} total playlists")
        return all_playlists

    @staticmethod
    def _normalise_track_id(track: Dict) -> str:
        """
        Convert any Spotify track identifier to the canonical 22-character ID.

        Handles values in the SMPD format (`spotify:track:...`) as well as raw IDs.
        """
        raw_values = [
            track.get("track_uri"),
            track.get("uri"),
            track.get("track_id"),
        ]

        for value in raw_values:
            if not value:
                continue
            candidate = str(value).strip()
            if not candidate:
                continue
            if candidate.startswith("spotify:track:"):
                candidate = candidate.split(":")[-1]
            candidate = candidate.split("?")[0].split("#")[0]
            if len(candidate) == 22 and candidate.replace("-", "").isalnum():
                return candidate
        return ""

    def extract_track_sequences(self, playlists: List[Dict]) -> List[List[str]]:
        """Extract track URI sequences from playlists."""
        sequences = []

        # Keywords to filter out children's/low-quality playlists
        exclude_keywords = [
            'kids', 'children', 'bambini', 'enfants', 'kinder', 'niños',
            'baby', 'toddler', 'nursery', 'lullaby', 'kinderlieder',
            'canzoni per bambini', 'canciones infantiles',
        ]

        filtered_count = 0
        short_count = 0
        long_count = 0

        for idx, playlist in enumerate(playlists):
            # Log progress every 10000 playlists
            if (idx + 1) % 10000 == 0:
                logger.info(f"Processing playlist {idx + 1:,}/{len(playlists):,}")

            # Filter playlists by name (exclude children's music)
            playlist_name = playlist.get('name', '').lower()
            if any(keyword in playlist_name for keyword in exclude_keywords):
                filtered_count += 1
                continue

            tracks = playlist.get('tracks', [])

            # Filter by playlist length
            if len(tracks) < self.min_playlist_length:
                short_count += 1
                continue
            if len(tracks) > self.max_playlist_length:
                long_count += 1
                continue

            # Extract Spotify track IDs in canonical format
            track_ids = []
            for track in tracks:
                normalised = self._normalise_track_id(track)
                if normalised:
                    track_ids.append(normalised)

            if len(track_ids) >= self.min_playlist_length:
                sequences.append(track_ids)

        logger.info(f"Extracted {len(sequences):,} valid playlists")
        logger.info(f"Filtered out: {filtered_count:,} children's playlists, {short_count:,} too short, {long_count:,} too long")
        return sequences

    def filter_rare_tracks(self, sequences: List[List[str]]) -> List[List[str]]:
        """Remove tracks that appear < min_track_occurrences times."""
        logger.info("Counting track occurrences...")

        # Count track occurrences
        track_counts = Counter()
        for sequence in sequences:
            track_counts.update(sequence)

        logger.info(f"Total unique tracks: {len(track_counts):,}")

        # Show distribution
        occurrence_dist = Counter(track_counts.values())
        logger.info("Track occurrence distribution:")
        for occ in sorted(occurrence_dist.keys())[:10]:
            logger.info(f"  {occ:3d} occurrences: {occurrence_dist[occ]:,} tracks")

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
            "avg_playlist_length": sum(len(seq) for seq in sequences) / len(sequences) if sequences else 0,
            "min_playlist_length": min(len(seq) for seq in sequences) if sequences else 0,
            "max_playlist_length": max(len(seq) for seq in sequences) if sequences else 0,
        }

        stats_path = self.output_dir / "stats.json"
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)

        logger.info(f"Statistics: {stats}")

    def process(self):
        """Run the full preprocessing pipeline."""
        logger.info("Starting SMPD preprocessing")

        # Extract zip if needed
        if not self.extract_zip_if_needed():
            logger.warning("No SMPD data available. Checking for existing JSON files...")

        # Load raw data
        playlists = self.load_raw_playlists()
        if not playlists:
            logger.error("No playlists found. Please ensure spotify_million_playlist_dataset.zip is in the project root.")
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