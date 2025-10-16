#!/bin/bash
#
# Quick Start Script for Pose Head Retraining Experiment
# 
# This script runs the experiment with proper conda environment setup.
# 
# Usage:
#   bash experiments/run_experiment.sh          # Test run (5 sequences, 2 epochs)
#   bash experiments/run_experiment.sh full     # Full run (100 sequences, 50 epochs)
#

set -e  # Exit on error

echo "=========================================="
echo "Pose Head Retraining Experiment"
echo "=========================================="
echo ""

# Change to project root
cd "$(dirname "$0")/.."
PROJECT_ROOT=$(pwd)
echo "[INFO] Project root: $PROJECT_ROOT"
echo ""

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "[ERROR] conda command not found!"
    echo ""
    echo "Please activate conda manually first, then run:"
    echo "  conda activate anycam"
    echo "  python experiments/train_pose_head_anycalib.py --max_sequences 5 --num_epochs 2"
    exit 1
fi

# Try to activate conda environment
echo "[INFO] Activating conda environment 'anycam'..."
eval "$(conda shell.bash hook)"
conda activate anycam || {
    echo "[ERROR] Failed to activate 'anycam' environment!"
    echo ""
    echo "Available environments:"
    conda env list
    echo ""
    echo "Please create or activate the correct environment manually."
    exit 1
}

echo "[OK] Environment activated"
echo ""

# Check which run mode
if [ "$1" = "full" ]; then
    echo "[MODE] Full training run (100 sequences, 50 epochs)"
    MAX_SEQ=""
    EPOCHS=50
    BATCH_SIZE=2
    SAVE_DIR="experiments/pose_head_experiment_results/full_run"
else
    echo "[MODE] Test run (5 sequences, 2 epochs)"
    MAX_SEQ="--max_sequences 5"
    EPOCHS=2
    BATCH_SIZE=1
    SAVE_DIR="experiments/pose_head_experiment_results/test_run"
fi

echo ""
echo "Parameters:"
echo "  Max sequences: ${MAX_SEQ:-all}"
echo "  Epochs: $EPOCHS"
echo "  Batch size: $BATCH_SIZE"
echo "  Save directory: $SAVE_DIR"
echo ""

# Verify dataset exists
if [ ! -d "/home/kalman/TUM/thesis/Objectron/videos" ]; then
    echo "[ERROR] Objectron dataset not found at /home/kalman/TUM/thesis/Objectron/videos"
    echo "Please check the dataset path in the script."
    exit 1
fi

echo "[OK] Dataset found"
echo "[INFO] Running in UNSUPERVISED mode - no ground truth required!"
echo ""

# Run the experiment
echo "=========================================="
echo "Starting experiment..."
echo "=========================================="
echo ""

python experiments/train_pose_head_anycalib.py \
    $MAX_SEQ \
    --num_epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr 1e-4 \
    --save_dir "$SAVE_DIR" \
    --model_path "pretrained_models/anycam_seq8"

echo ""
echo "=========================================="
echo "Experiment complete!"
echo "=========================================="
echo ""
echo "Results saved to: $SAVE_DIR"
echo ""
echo "To analyze results:"
echo "  cd $SAVE_DIR"
echo "  ls -lh"
echo ""

