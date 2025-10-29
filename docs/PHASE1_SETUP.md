# Phase 1: Item2vec + Learned Ranking Setup Guide

This guide walks you through setting up the enhanced recommendation engine with:
- Item2vec collaborative filtering
- Two-stage retrieval (ANN + catalogue)
- Learned LightGBM ranker
- MMR diversity reranking
- Evaluation harness

## Prerequisites

1. **Python 3.9+**
2. **Spotify Million Playlist Dataset (SMPD)**
   - Download from: https://www.aicrowd.com/challenges/spotify-million-playlist-dataset-challenge
   - Alternative: Use the RecSys 2018 dataset mirror
3. **~10GB disk space** for dataset and models
4. **8GB+ RAM** for training

## Step-by-Step Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

New dependencies added:
- `gensim` - Word2Vec for item2vec
- `annoy` - ANN index for fast retrieval
- `lightgbm` - Learned ranking model
- `pandas`, `scikit-learn`, `numpy` - Data processing and ML

### 2. Download and Preprocess SMPD

**Download the dataset:**
```bash
# Create data directory
mkdir -p data/smpd/raw

# Download SMPD (5GB) to data/smpd/raw/
# The dataset contains 1000 JSON files: mpd.slice.0-999.json
```

**Preprocess the dataset:**
```bash
export SMPD_RAW_DIR=data/smpd/raw
export SMPD_PROCESSED_DIR=data/smpd/processed

python scripts/download_smpd.py
```

This will:
- Load 1M+ playlists from JSON files
- Filter playlists (5-500 tracks)
- Filter tracks (must appear ≥50 times)
- Save processed playlists to `data/smpd/processed/playlists.json`

**Expected output:**
- ~900k-1M playlists after filtering
- ~100k-200k unique tracks
- Processing time: ~10-20 minutes

### 3. Train Item2vec Model

Train Word2Vec on playlist co-occurrence:

```bash
export PLAYLIST_FILE=data/smpd/processed/playlists.json
export MODEL_OUTPUT_DIR=models/item2vec

python scripts/train_item2vec.py
```

**Configuration:**
- Vector size: 100 dimensions
- Window size: 50 (playlists are unordered sets)
- Min count: 50 (filter rare tracks)
- Epochs: 10
- Workers: 8 (parallel training)

**Expected output:**
- `models/item2vec/item2vec.model` - Full model
- `models/item2vec/item2vec.wordvectors` - Just the embeddings (faster loading)
- `models/item2vec/metadata.json` - Model info

**Training time:** ~1-2 hours on CPU for 1M playlists

### 4. Build ANN Index

Build Annoy index for fast nearest neighbor search:

```bash
export ITEM2VEC_PATH=models/item2vec/item2vec.wordvectors
export ANN_INDEX_PATH=models/item2vec/ann_index
export ANN_N_TREES=50

python scripts/build_ann_index.py
```

**Expected output:**
- `models/item2vec/ann_index/annoy.index` - Annoy index
- `models/item2vec/ann_index/id_map.json` - Track ID mapping

**Build time:** ~5-10 minutes

**Index parameters:**
- `n_trees=50` - Good balance of speed vs accuracy
- Use `n_trees=100` for better accuracy (slower)
- Use `n_trees=25` for faster queries (lower accuracy)

### 5. Train Learned Ranker

Train LightGBM LambdaMART ranker on playlist continuation:

```bash
export ITEM2VEC_PATH=models/item2vec/item2vec.wordvectors
export PLAYLIST_FILE=data/smpd/processed/playlists.json
export RANKER_OUTPUT=models/ranker/lightgbm_ranker.txt

python scripts/train_ranker.py
```

**What it does:**
- Generates training data from playlist continuation task
- For each playlist: hold out last 30% as ground truth
- Positives: held-out tracks (label=1)
- Negatives: random tracks not in playlist (label=0)
- Trains LambdaMART ranker optimizing NDCG

**Features extracted:**
- `i2v_cosine` - Item2vec similarity
- `audio_similarity` - Audio feature similarity
- `popularity` - Track popularity (0-1)
- `artist_in_playlist` - Artist match
- `genre_jaccard` - Genre overlap
- `era_gap` - Release year difference
- `seed_i2v_cosine`, `seed_audio_sim` - Seed track similarity
- `valence_diff`, `energy_diff` - Mood feature differences

**Expected output:**
- `models/ranker/lightgbm_ranker.txt` - Trained model
- Feature importance printed to console

**Training time:** ~30-60 minutes for 10k playlists

### 6. Evaluate the System

Run evaluation on held-out playlists:

```bash
export PLAYLIST_FILE=data/smpd/processed/playlists.json
export ITEM2VEC_PATH=models/item2vec/item2vec.wordvectors

python scripts/evaluate.py
```

**Metrics computed:**
- **R-Precision**: Precision at R (R = number of ground truth tracks)
- **NDCG@K**: Normalized DCG at K=[5, 10, 30, 100]
- **Click@K**: Did user click any of top-K?

**Expected results (ballpark):**
- R-Precision: 0.08-0.15
- NDCG@10: 0.15-0.25
- Click@10: 0.30-0.50

**Output:**
- Results printed to console
- Saved to `evaluation/results.json`

**Evaluation time:** ~5-10 minutes for 1k playlists

## Using the Enhanced Engine

### Option A: Use Hybrid Engine Directly

```python
from services.recommendation.hybrid_engine import HybridRecommendationEngine
from services.recommendation.embeddings import EmbeddingService
from services.recommendation.ranker import LearnedRanker
from services.recommendation.catalogue import TrackCatalogue

# Load services
catalogue = TrackCatalogue()
embedding_service = EmbeddingService(
    item2vec_path="models/item2vec/item2vec.wordvectors",
    ann_index_path="models/item2vec/ann_index"
)
ranker = LearnedRanker(model_path="models/ranker/lightgbm_ranker.txt")

# Initialize engine
engine = HybridRecommendationEngine(
    catalogue=catalogue,
    embedding_service=embedding_service,
    ranker=ranker,
    use_mmr=True,  # Enable MMR diversity
)

# Build user profile from playlist
playlist_tracks = [...]  # List of track dicts with audio features
user_profile = engine.build_user_profile(playlist_tracks)

# Get recommendations
recommendations = await engine.get_recommendations(
    user_profile=user_profile,
    context={"moods": ["upbeat"], "activities": ["party"]},
    limit=30,
)
```

### Option B: Fallback Without Learned Models

If you don't have trained models yet, the system gracefully falls back:

```python
# Without item2vec or ranker
engine = HybridRecommendationEngine(
    catalogue=catalogue,
    embedding_service=None,  # Will skip item2vec retrieval
    ranker=None,  # Will use SimpleRanker with hand-tuned weights
    use_mmr=False,  # No MMR without embeddings
)

# Still works! Just uses original contextual engine logic
```

### Option C: Integrate into Existing API

Update your API router to use the hybrid engine:

```python
# In api/routers/search.py or your endpoint
from services.recommendation.hybrid_engine import HybridRecommendationEngine
from services.recommendation.embeddings import EmbeddingService
from services.recommendation.ranker import LearnedRanker

# Initialize once at startup
embedding_service = EmbeddingService(
    item2vec_path="models/item2vec/item2vec.wordvectors",
    ann_index_path="models/item2vec/ann_index"
)
ranker = LearnedRanker(model_path="models/ranker/lightgbm_ranker.txt")

engine = HybridRecommendationEngine(
    catalogue=catalogue,
    embedding_service=embedding_service,
    ranker=ranker,
)

# Use in your endpoint
@router.post("/api/v1/playlist/recommendations")
async def get_recommendations(request: RecommendationRequest):
    # Build profile
    user_profile = engine.build_user_profile(request.tracks)

    # Get recommendations
    recommendations = await engine.get_recommendations(
        user_profile=user_profile,
        context=request.context,
        limit=request.limit,
    )

    return recommendations
```

## Architecture Overview

### Two-Stage Retrieval + Ranking

```
┌─────────────────────────────────────────────────┐
│ Stage 1: RETRIEVAL (~1-2k candidates)          │
├─────────────────────────────────────────────────┤
│ • Item2vec neighbors of seed track (500)       │
│ • Item2vec neighbors of playlist (1000)        │
│ • Catalogue filtered by context tags (500)     │
│ → Union & dedupe                                │
└─────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────┐
│ Stage 2: RANKING (score all candidates)        │
├─────────────────────────────────────────────────┤
│ • Extract 10+ features per candidate           │
│ • Score with LightGBM LambdaMART               │
│ • Sort by score descending                     │
└─────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────┐
│ Stage 3: RERANKING (diversity)                 │
├─────────────────────────────────────────────────┤
│ • Apply MMR (λ=0.7)                            │
│ • Enforce max 2 per artist                     │
│ • Return top-K                                 │
└─────────────────────────────────────────────────┘
```

### What You Get

**Compared to original engine:**

| Feature | Original | Phase 1 Enhanced |
|---------|----------|------------------|
| Candidate generation | All catalogue (~1k) | Item2vec ANN (~10k-100k) |
| Collaborative signal | None | Item2vec co-occurrence |
| Ranking | Hand-tuned weights | Learned LambdaMART |
| Diversity | Artist cap only | MMR + artist cap |
| Cold start (single seed) | Weak | Strong (item2vec neighbors) |
| Scalability | O(catalogue size) | O(log(catalogue)) with ANN |

**Expected improvements:**
- **+25-35% NDCG** from item2vec collaborative signal
- **+10-15% NDCG** from learned ranker
- **Better diversity** from MMR without hurting relevance
- **10-100x faster** retrieval with ANN index

## Troubleshooting

### Issue: "Item2vec model not found"

**Solution:** Run `python scripts/train_item2vec.py` first

### Issue: "Playlist file not found"

**Solution:** Run `python scripts/download_smpd.py` to preprocess SMPD

### Issue: "Out of memory during training"

**Solutions:**
- Reduce `max_examples` in `train_ranker.py` (try 5000)
- Reduce `n_negatives_per_positive` (try 5 instead of 10)
- Use a machine with more RAM

### Issue: "ANN search is slow"

**Solutions:**
- Increase `n_trees` when building index (better accuracy, faster search)
- Use FAISS instead of Annoy for GPU acceleration
- Pre-filter candidates by context before ANN search

### Issue: "Recommendations are too similar"

**Solutions:**
- Increase MMR diversity: set `lambda_param=0.5` (50% diversity)
- Increase `max_per_artist=3`
- Reduce item2vec weight in retrieval

### Issue: "Recommendations are too diverse (low relevance)"

**Solutions:**
- Decrease MMR diversity: set `lambda_param=0.85` (85% relevance)
- Increase item2vec retrieval weight
- Reduce random negatives in ranking training data

## Next Steps

After Phase 1 is working:

**Phase 2: Deep Audio Embeddings (Optional)**
- Train GLU-CNN on audio spectrograms
- Replace Spotify features with learned embeddings
- Further +5-10% NDCG improvement

**Phase 3: Advanced Features**
- Attention-pooled playlist embeddings (not just mean)
- Sequence models (GRU4Rec) for order-aware recs
- Contextual bandits for online weight tuning
- CLAP text-to-audio for natural language queries

**Production Hardening:**
- Add caching for ANN search results
- Batch recommendation requests
- Monitor latency and accuracy metrics
- A/B test against original engine

## Questions or Issues?

Check the evaluation results first to understand baseline performance.
If metrics are significantly lower than expected, verify:
1. Item2vec model trained successfully (check vocabulary size)
2. ANN index built correctly (test neighbor queries)
3. Ranker trained without errors (check feature importance)

Good luck! 🚀
