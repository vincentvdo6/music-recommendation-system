"""
Create a small synthetic dataset for testing the pipeline.

This generates fake playlists with Spotify URIs for testing
without downloading the full SMPD dataset.
"""

import json
import random
from pathlib import Path

# Sample of real Spotify track URIs (for realism)
SAMPLE_TRACKS = [
    "spotify:track:0VjIjW4GlUZAMYd2vXMi3b",
    "spotify:track:7qiZfU4dY1lWllzX7mPBI",
    "spotify:track:3n3Ppam7vgaVa1iaRUc9Lp",
    "spotify:track:0DiWol3AO6WpXZgp0goxAV",
    "spotify:track:6habFhsOp2NvshLv26DqMb",
]

# Expand track pool
for i in range(1000):
    fake_uri = f"spotify:track:{random.randint(10**21, 10**22-1):022x}"[:36]
    SAMPLE_TRACKS.append(fake_uri)


def generate_playlist(pid: int, num_tracks: int):
    """Generate a fake playlist."""
    track_pool = random.sample(SAMPLE_TRACKS, min(num_tracks, len(SAMPLE_TRACKS)))

    return {
        "name": f"Test Playlist {pid}",
        "collaborative": False,
        "pid": pid,
        "modified_at": 1493424000 + pid * 1000,
        "num_tracks": num_tracks,
        "num_albums": int(num_tracks * 0.8),
        "num_followers": random.randint(1, 100),
        "tracks": [
            {
                "pos": i,
                "artist_name": f"Artist {hash(track) % 500}",
                "track_uri": track,
                "artist_uri": f"spotify:artist:{random.randint(10**21, 10**22-1):022x}"[:36],
                "track_name": f"Song {hash(track) % 1000}",
                "album_uri": f"spotify:album:{random.randint(10**21, 10**22-1):022x}"[:36],
                "duration_ms": random.randint(120000, 300000),
                "album_name": f"Album {hash(track) % 300}"
            }
            for i, track in enumerate(track_pool)
        ]
    }


def create_test_dataset(
    output_dir: str = "data/smpd/raw",
    n_slices: int = 10,
    playlists_per_slice: int = 100,
):
    """Create test dataset."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Creating test dataset in {output_path}")
    print(f"  - {n_slices} slices")
    print(f"  - {playlists_per_slice} playlists per slice")
    print(f"  - Total: {n_slices * playlists_per_slice} playlists")

    for slice_id in range(n_slices):
        playlists = []

        for i in range(playlists_per_slice):
            pid = slice_id * playlists_per_slice + i
            num_tracks = random.randint(10, 100)
            playlists.append(generate_playlist(pid, num_tracks))

        slice_file = output_path / f"mpd.slice.{slice_id}.json"

        with open(slice_file, 'w') as f:
            json.dump({
                "info": {
                    "version": "v1-test",
                    "generated_on": "2024-01-01",
                    "slice": slice_id,
                },
                "playlists": playlists
            }, f)

        print(f"  ✓ Created {slice_file.name}")

    print(f"\n✓ Test dataset created!")
    print(f"\nNow run: py scripts/download_smpd.py")


if __name__ == "__main__":
    create_test_dataset()
