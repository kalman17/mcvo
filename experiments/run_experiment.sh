#!/bin/bash
#
# Quick Start Script for Pose Head Retraining Experiment
# 
# This script runs the experiment with proper conda environment setup.
# 
# Usage:
#   bash experiments/run_experiment.sh                  # Test run (5 sequences, 2 epochs, no eval)
#   bash experiments/run_experiment.sh full             # Full run (all sequences, 50 epochs)
#   bash experiments/run_experiment.sh full_with_eval   # Full run + evaluation on test set
#   bash experiments/run_experiment.sh multi_pair       # Multi-pair extraction (more data)
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
RUN_EVAL=""
MULTI_PAIR=""

if [ "$1" = "full" ]; then
    echo "[MODE] Full training run (all sequences, 50 epochs)"
    MAX_SEQ=""
    EPOCHS=50
    BATCH_SIZE=2
    SAVE_DIR="experiments/pose_head_experiment_results/full_run"
elif [ "$1" = "full_with_eval" ]; then
    echo "[MODE] Full training run with evaluation (all sequences, 50 epochs, all pairs)"
    MAX_SEQ=""
    EPOCHS=50
    BATCH_SIZE=2
    SAVE_DIR="experiments/pose_head_experiment_results/full_run_eval"
    RUN_EVAL="--run_evaluation"
    MULTI_PAIR="--extract_all_pairs"
elif [ "$1" = "multi_pair" ]; then
    echo "[MODE] Multi-pair extraction (10 sequences, 10 epochs, all consecutive pairs)"
    MAX_SEQ="--max_sequences 10"
    EPOCHS=10
    BATCH_SIZE=4
    SAVE_DIR="experiments/pose_head_experiment_results/multi_pair_run"
    MULTI_PAIR="--extract_all_pairs"
else
    echo "[MODE] Test run (5 sequences, 2 epochs, single pair per video)"
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
if [ -n "$MULTI_PAIR" ]; then
    echo "  Multi-pair extraction: ENABLED (ALL consecutive pairs)"
else
    echo "  Multi-pair extraction: disabled (single pair per video)"
fi
if [ -n "$RUN_EVAL" ]; then
    echo "  Evaluation: ENABLED (on LightSpeed dataset)"
else
    echo "  Evaluation: disabled"
fi
echo "  Save directory: $SAVE_DIR"
echo ""

# Verify dataset exists
if [ ! -d "/home/kalman/TUM/thesis/Objectron/videos" ]; then
    echo "[ERROR] Objectron dataset not found at /home/kalman/TUM/thesis/Objectron/videos"
    echo "Please check the dataset path in the script."
    exit 1
fi

echo "[OK] Dataset found"
echo "[INFO] Training is UNSUPERVISED - no ground truth required for training!"
if [ -n "$RUN_EVAL" ]; then
    echo "[INFO] Evaluation will use LightSpeed validation dataset to measure accuracy"
    echo "[INFO] LightSpeed: 36 sequences from DynPose-100k with ground truth poses"
fi
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
    --model_path "pretrained_models/anycam_seq8" \
    $MULTI_PAIR \
    $RUN_EVAL \
    --eval_dataset lightspeed

echo ""
echo "=========================================="
echo "Experiment complete!"
echo "=========================================="
echo ""
echo "Results saved to: $SAVE_DIR"
echo ""
echo "Files generated:"
echo "  - checkpoints (every 5 epochs)"
echo "  - loss_curve.png (training loss visualization)"
echo "  - training_log.txt (detailed training log)"
echo "  - training_summary.txt (training statistics)"
if [ -n "$RUN_EVAL" ]; then
    echo "  - evaluation/ (test set performance metrics)"
fi
echo ""
echo "Dataset split saved to: experiments/objectron_split.json"
echo ""
echo "To view results:"
echo "  ls -lh $SAVE_DIR"
echo "  cat $SAVE_DIR/training_summary.txt"
if [ -n "$RUN_EVAL" ]; then
    echo "  cat $SAVE_DIR/evaluation/evaluation_results.json"
fi
echo ""

