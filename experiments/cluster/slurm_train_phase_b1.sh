#!/bin/bash
#SBATCH --job-name=train_B1
#SBATCH --output=/storage/user/maka/logs/train_B1_%j.out
#SBATCH --error=/storage/user/maka/logs/train_B1_%j.err
#SBATCH --partition=NORMAL
#SBATCH --constraint="GPU_GEN:AMPERE|GPU_GEN:ADA|GPU_GEN:HOPPER"
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=32G
#SBATCH --time=12:00:00

set -euo pipefail

REPO="/storage/user/maka/anycam"
PREPROC_DIR="/storage/user/maka/preprocessed"
TRAIN_DIR="/storage/user/maka/train/phase_B1"

export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

eval "$(/storage/user/maka/miniconda3/bin/conda shell.bash hook)"
conda activate anycam
cd /tmp

echo "============================================"
echo "  Phase B1 — FAT Pre-Training (Isolated)"
echo "  Host: $(hostname)"
echo "  GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "  Date: $(date)"
echo "============================================"

mkdir -p "$TRAIN_DIR"

# Precompute vanilla baselines if missing
if [ ! -f "$PREPROC_DIR/val_baselines.pt" ]; then
    echo ""
    echo "=== Precomputing vanilla baselines ==="
    python3 "$REPO/experiments/precompute_vanilla_baselines.py" \
        --data_dir "$PREPROC_DIR" \
        --output_path "$PREPROC_DIR/val_baselines.pt" \
        --anycam_config "$REPO/pretrained_models/anycam_seq8/training_config.yaml" \
        --anycam_checkpoint "$REPO/pretrained_models/anycam_seq8/training_checkpoint_247500.pt" \
        --image_size 336 \
        2>&1
fi

echo ""
echo "=== Phase B1 Training (10 epochs, batch_size=8) ==="
python3 "$REPO/experiments/train_unified.py" \
    --phase B1 \
    --data_dir "$PREPROC_DIR" \
    --save_dir "$TRAIN_DIR" \
    --val_baselines "$PREPROC_DIR/val_baselines.pt" \
    --num_epochs 10 \
    --batch_size 8 \
    --learning_rate 1e-4 \
    --max_ahead 3 \
    --image_size 336 \
    2>&1

echo ""
echo "=== Phase B1 COMPLETE ==="
echo "Date: $(date)"
