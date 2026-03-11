#!/bin/bash
#SBATCH --job-name=train_A4
#SBATCH --output=/storage/user/maka/logs/train_A4_%j.out
#SBATCH --error=/storage/user/maka/logs/train_A4_%j.err
#SBATCH --partition=NORMAL
#SBATCH --comment="Masters thesis deadline 2026-03-31"
#SBATCH --constraint="GPU_GEN:AMPERE|GPU_GEN:ADA|GPU_GEN:HOPPER"
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=32G
#SBATCH --time=14-00:00:00

set -euo pipefail

REPO="/storage/user/maka/anycam"
PREPROC_DIR="/storage/user/maka/preprocessed"
TRAIN_DIR="/storage/user/maka/train/phase_A_v4"

PRETRAINED="$REPO/pretrained_models/anycam_seq8/training_checkpoint_247500.pt"
ANYCAM_CFG="$REPO/pretrained_models/anycam_seq8/training_config.yaml"

export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

eval "$(/storage/user/maka/miniconda3/bin/conda shell.bash hook)"
conda activate anycam
cd /tmp

echo "============================================"
echo "  Phase A v4 — Pose Head + Composed Flow Loss"
echo "  Host: $(hostname)"
echo "  GPU:  $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo unknown)"
echo "  Date: $(date)"
echo "============================================"

mkdir -p "$TRAIN_DIR"

# Precompute vanilla baselines if needed
if [ ! -f "$PREPROC_DIR/val_baselines.pt" ]; then
    echo ""
    echo "=== Precomputing vanilla baselines ==="
    python3 "$REPO/experiments/precompute_vanilla_baselines.py" \
        --data_dir "$PREPROC_DIR" \
        --output_path "$PREPROC_DIR/val_baselines.pt" \
        --anycam_config "$ANYCAM_CFG" \
        --anycam_checkpoint "$PRETRAINED" \
        --image_size 336 \
        2>&1
fi

echo ""
echo "=== Phase A v4 Training (20 epochs, batch_size=32, lr=5e-5 cosine) ==="
python3 "$REPO/experiments/train_unified.py" \
    --phase A \
    --data_dir "$PREPROC_DIR" \
    --save_dir "$TRAIN_DIR" \
    --anycam_config "$ANYCAM_CFG" \
    --pretrained_anycam "$PRETRAINED" \
    --val_baselines "$PREPROC_DIR/val_baselines.pt" \
    --num_epochs 20 \
    --batch_size 32 \
    --learning_rate 5e-5 \
    --max_ahead 3 \
    --image_size 336 \
    $([ -f "$TRAIN_DIR/checkpoints/latest.pt" ] && echo "--resume $TRAIN_DIR/checkpoints/latest.pt") \
    2>&1

echo ""
echo "=== Phase A v4 COMPLETE ==="
echo "Date: $(date)"
