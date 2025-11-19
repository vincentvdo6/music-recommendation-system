"""Merge multiple playlist slice files."""
import json
from pathlib import Path
import sys

def merge_playlists(input_dirs, output_dir):
    """Merge playlists from multiple directories."""
    all_playlists = []
    seen_track_hashes = set()
    
    for input_dir in input_dirs:
        input_path = Path(input_dir)
        if not input_path.exists():
            print(f"Warning: {input_dir} does not exist, skipping...")
            continue
            
        slice_files = sorted(input_path.glob("mpd.slice.*.json"))
        
        for slice_file in slice_files:
            print(f"Reading {slice_file}...")
            with open(slice_file) as f:
                data = json.load(f)
                playlists = data.get("playlists", [])
                
                for playlist in playlists:
                    # Create hash of playlist tracks to detect duplicates
                    track_uris = tuple(t.get("track_uri", "") for t in playlist.get("tracks", []))
                    track_hash = hash(track_uris)
                    
                    if track_hash not in seen_track_hashes:
                        seen_track_hashes.add(track_hash)
                        all_playlists.append(playlist)
    
    # Renumber playlist IDs
    for idx, playlist in enumerate(all_playlists):
        playlist["pid"] = idx
    
    print(f"\nTotal unique playlists: {len(all_playlists)}")
    
    # Save merged playlists
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    playlists_per_slice = 1000
    n_slices = (len(all_playlists) + playlists_per_slice - 1) // playlists_per_slice
    
    for slice_idx in range(n_slices):
        start = slice_idx * playlists_per_slice
        end = min(start + playlists_per_slice, len(all_playlists))
        
        slice_playlists = all_playlists[start:end]
        slice_file = output_path / f"mpd.slice.{slice_idx}.json"
        
        with open(slice_file, 'w') as f:
            json.dump({
                "info": {
                    "version": "v1-merged",
                    "slice": slice_idx,
                },
                "playlists": slice_playlists
            }, f)
        
        print(f"Saved {slice_file.name} with {len(slice_playlists)} playlists")
    
    print(f"\n✓ Merged {len(all_playlists)} playlists to {output_dir}")

if __name__ == "__main__":
    # Merge old and new raw data
    merge_playlists(
        input_dirs=["data/smpd/raw", "data/smpd/raw_new"],
        output_dir="data/smpd/raw_merged"
    )
