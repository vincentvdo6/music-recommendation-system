"""
Check how many cached audio features we have to train the mood predictor.
"""

import json
import pickle
from pathlib import Path

def check_cache():
    print("="*80)
    print("Checking Cached Audio Features for Training")
    print("="*80)

    # Check if cache directory exists
    cache_dir = Path(".cache")

    if not cache_dir.exists():
        print("\n❌ No .cache directory found")
        print("The cache might be in memory only or using a different storage.")
        return

    # Look for any pickle/json cache files
    cache_files = list(cache_dir.glob("**/*.pkl")) + list(cache_dir.glob("**/*.json"))

    if not cache_files:
        print("\n❌ No cache files found in .cache directory")
    else:
        print(f"\n✅ Found {len(cache_files)} cache files:")
        for f in cache_files:
            print(f"  - {f}")

    # Check item2vec tracks (we know their IDs)
    item2vec_path = Path("models/item2vec/item2vec.wordvectors")
    if item2vec_path.exists():
        from gensim.models import KeyedVectors
        wv = KeyedVectors.load(str(item2vec_path), mmap='r')
        print(f"\n✅ Item2vec has {len(wv.index_to_key):,} track IDs")
        print(f"   These are our potential training candidates")

    print("\n" + "="*80)
    print("Next Steps:")
    print("="*80)
    print("1. We'll use your item2vec track IDs as training candidates")
    print("2. For tracks with cached features → use as training data")
    print("3. For tracks without features → we'll predict them!")
    print("\n" + "="*80)

if __name__ == "__main__":
    check_cache()
