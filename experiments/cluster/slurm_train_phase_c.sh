#!/bin/bash
#SBATCH --job-name=train_C
#SBATCH --output=/storage/user/maka/logs/train_C_%j.out
#SBATCH --error=/storage/user/maka/logs/train_C_%j.err
#SBATCH --partition=NORMAL
#SBATCH --constraint="GPU_GEN:AMPERE|GPU_GEN:ADA|GPU_GEN:HOPPER"
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=32G
#SBATCH --time=24:00:00

set -euo pipefail

REPO="/storage/user/maka/anycam"
PREPROC_DIR="/storage/user/maka/preprocessed"
TRAIN_DIR="/storage/user/maka/train/phase_C"

export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

eval "$(/storage/user/maka/miniconda3/bin/conda shell.bash hook)"
conda activate anycam
cd /tmp

echo "============================================"
echo "  Phase C — Joint End-to-End Training"
echo "  Host: $(hostname)"
echo "  GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "  Date: $(date)"
echo "============================================"

mkdir -p "$TRAIN_DIR"

# Verify dependencies exist (B1 for FAT, A for pose head)
PHASE_B1_CKPT="/storage/user/maka/train/phase_B1/checkpoints/final.pt"
PHASE_A_CKPT="/storage/user/maka/train/phase_A/checkpoints/final.pt"

if [ ! -f "$PHASE_B1_CKPT" ]; then
    echo "ERROR: Phase B1 checkpoint not found: $PHASE_B1_CKPT"
    exit 1
fi
if [ ! -f "$PHASE_A_CKPT" ]; then
    echo "ERROR: Phase A checkpoint not found: $PHASE_A_CKPT"
    exit 1
fi

echo ""
echo "=== Phase C Training (50 epochs, joint — all params unfrozen) ==="
python3 "$REPO/experiments/train_unified.py" \
    --phase C \
    --data_dir "$PREPROC_DIR" \
    --save_dir "$TRAIN_DIR" \
    --phase_b1_checkpoint "$PHASE_B1_CKPT" \
    --phase_a_checkpoint "$PHASE_A_CKPT" \
    --val_baselines "$PREPROC_DIR/val_baselines.pt" \
    --num_epochs 50 \
    --batch_size 2 \
    --learning_rate 1e-5 \
    --lambda_calib 1e-4 \
    --max_ahead 3 \
    --image_size 336 \
    2>&1

echo ""
echo "=== Phase C COMPLETE ==="
echo "Date: $(date)"
