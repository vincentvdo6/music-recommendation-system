#!/bin/bash

# Quick start script for training and testing BPR model
# This runs the complete pipeline from training to testing

set -e  # Exit on error

echo "========================================"
echo "BPR Model Training Pipeline"
echo "========================================"
echo ""

# Check if data exists
if [ ! -f "data/smpd/processed/playlists.json" ]; then
    echo "Error: Playlist data not found at data/smpd/processed/playlists.json"
    echo "Please ensure your data is in the correct location."
    exit 1
fi

echo "Step 1: Training NeuMF with BPR loss..."
echo "----------------------------------------"
python3 scripts/train_ncf_bpr.py \
    --epochs 10 \
    --batch-size 512 \
    --num-negatives 4 \
    --use-hard-negatives \
    --device cpu

echo ""
echo "Step 2: Testing trained model..."
echo "----------------------------------------"
python3 scripts/test_ncf_recommendations.py

echo ""
echo "========================================"
echo "Pipeline complete!"
echo "========================================"
echo ""
echo "Next steps:"
echo "  1. Check model performance in the output above"
echo "  2. If NDCG@10 > 0.35, the model is working well"
echo "  3. The model is saved at: models/ncf/neumf_bpr_best.pt"
echo "  4. It will be automatically loaded by your API"
echo ""
echo "To train with more epochs for better accuracy:"
echo "  python3 scripts/train_ncf_bpr.py --epochs 20"
echo ""
