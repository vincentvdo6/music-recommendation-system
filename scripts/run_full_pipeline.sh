#!/bin/bash

# Full Phase 1 training pipeline
# Runs all steps: SMPD preprocessing -> item2vec training -> ANN index -> ranker training -> evaluation

set -e  # Exit on error

echo "=========================================="
echo "Phase 1: Item2vec + Learned Ranking"
echo "Full Training Pipeline"
echo "=========================================="
echo ""

# Configuration
export SMPD_RAW_DIR=${SMPD_RAW_DIR:-"data/smpd/raw"}
export SMPD_PROCESSED_DIR=${SMPD_PROCESSED_DIR:-"data/smpd/processed"}
export PLAYLIST_FILE=${PLAYLIST_FILE:-"data/smpd/processed/playlists.json"}
export MODEL_OUTPUT_DIR=${MODEL_OUTPUT_DIR:-"models/item2vec"}
export ITEM2VEC_PATH=${ITEM2VEC_PATH:-"models/item2vec/item2vec.wordvectors"}
export ANN_INDEX_PATH=${ANN_INDEX_PATH:-"models/item2vec/ann_index"}
export RANKER_OUTPUT=${RANKER_OUTPUT:-"models/ranker/lightgbm_ranker.txt"}

# Step 1: Check SMPD
echo "Step 1: Checking for SMPD dataset..."
if [ ! -d "$SMPD_RAW_DIR" ] || [ -z "$(ls -A $SMPD_RAW_DIR)" ]; then
    echo "ERROR: SMPD dataset not found in $SMPD_RAW_DIR"
    echo ""
    echo "Please download the Spotify Million Playlist Dataset:"
    echo "https://www.aicrowd.com/challenges/spotify-million-playlist-dataset-challenge"
    echo ""
    echo "Extract to: $SMPD_RAW_DIR"
    echo "Expected: mpd.slice.0-999.json files"
    exit 1
fi

echo "✓ SMPD dataset found"
echo ""

# Step 2: Preprocess SMPD
echo "Step 2: Preprocessing SMPD..."
if [ ! -f "$PLAYLIST_FILE" ]; then
    echo "Running preprocessing..."
    python scripts/download_smpd.py
    echo "✓ Preprocessing complete"
else
    echo "✓ Playlist file already exists: $PLAYLIST_FILE"
fi
echo ""

# Step 3: Train item2vec
echo "Step 3: Training item2vec model..."
if [ ! -f "$ITEM2VEC_PATH" ]; then
    echo "This will take ~1-2 hours..."
    python scripts/train_item2vec.py
    echo "✓ Item2vec training complete"
else
    echo "✓ Item2vec model already exists: $ITEM2VEC_PATH"
fi
echo ""

# Step 4: Build ANN index
echo "Step 4: Building ANN index..."
if [ ! -d "$ANN_INDEX_PATH" ]; then
    echo "This will take ~5-10 minutes..."
    python scripts/build_ann_index.py
    echo "✓ ANN index complete"
else
    echo "✓ ANN index already exists: $ANN_INDEX_PATH"
fi
echo ""

# Step 5: Train ranker
echo "Step 5: Training LightGBM ranker..."
if [ ! -f "$RANKER_OUTPUT" ]; then
    echo "This will take ~30-60 minutes..."
    python scripts/train_ranker.py
    echo "✓ Ranker training complete"
else
    echo "✓ Ranker model already exists: $RANKER_OUTPUT"
fi
echo ""

# Step 6: Evaluate
echo "Step 6: Running evaluation..."
python scripts/evaluate.py
echo "✓ Evaluation complete"
echo ""

# Summary
echo "=========================================="
echo "Pipeline Complete! 🚀"
echo "=========================================="
echo ""
echo "Models saved to:"
echo "  - Item2vec: $ITEM2VEC_PATH"
echo "  - ANN index: $ANN_INDEX_PATH"
echo "  - Ranker: $RANKER_OUTPUT"
echo ""
echo "Evaluation results saved to:"
echo "  - evaluation/results.json"
echo ""
echo "Next steps:"
echo "1. Review evaluation metrics"
echo "2. Integrate HybridRecommendationEngine into your API"
echo "3. A/B test against original engine"
echo ""
echo "See docs/PHASE1_SETUP.md for integration guide"
