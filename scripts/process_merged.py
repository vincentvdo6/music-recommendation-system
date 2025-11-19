"""Process merged playlist data."""
import json
from pathlib import Path
from collections import Counter

def process_merged_playlists():
    """Process merged playlists into training format."""
    input_dir = Path("data/smpd/raw_merged")
    output_dir = Path("data/smpd/processed")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_playlists = []
    
    # Read all slices
    slice_files = sorted(input_dir.glob("mpd.slice.*.json"))
    
    for slice_file in slice_files:
        print(f"Reading {slice_file}...")
        with open(slice_file) as f:
            data = json.load(f)
            all_playlists.extend(data.get("playlists", []))
    
    # Calculate stats
    track_counts = Counter()
    playlist_lengths = []
    
    for playlist in all_playlists:
        tracks = playlist.get("tracks", [])
        playlist_lengths.append(len(tracks))
        
        for track in tracks:
            track_uri = track.get("track_uri", "")
            if track_uri:
                track_counts[track_uri] += 1
    
    stats = {
        "n_playlists": len(all_playlists),
        "n_tracks": len(track_counts),
        "avg_playlist_length": sum(playlist_lengths) / len(playlist_lengths) if playlist_lengths else 0,
        "min_playlist_length": min(playlist_lengths) if playlist_lengths else 0,
        "max_playlist_length": max(playlist_lengths) if playlist_lengths else 0
    }
    
    # Save processed playlists
    with open(output_dir / "playlists.json", 'w') as f:
        json.dump(all_playlists, f)
    
    # Save stats
    with open(output_dir / "stats.json", 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"✓ Processed {stats['n_playlists']} playlists")
    print(f"  Unique tracks: {stats['n_tracks']}")
    print(f"  Avg playlist length: {stats['avg_playlist_length']:.1f}")
    print(f"{'='*60}")
    print(f"\nSaved to: {output_dir}")

if __name__ == "__main__":
    process_merged_playlists()
