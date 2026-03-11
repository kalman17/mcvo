#!/bin/bash
#SBATCH --job-name=train_C_v3
#SBATCH --output=/storage/user/maka/logs/train_C_v3_%j.out
#SBATCH --error=/storage/user/maka/logs/train_C_v3_%j.err
#SBATCH --partition=NORMAL
#SBATCH --comment="Masters thesis deadline 2026-03-31, Phase C v3 frozen backbones"
#SBATCH --constraint="GPU_GEN:ADA|GPU_GEN:AMPERE"
#SBATCH --gres=gpu:1,VRAM:40G
#SBATCH --cpus-per-task=5
#SBATCH --mem=32G
#SBATCH --time=14-00:00:00

set -euo pipefail

REPO="/storage/user/maka/anycam"
PREPROC_DIR="/storage/user/maka/preprocessed"
TRAIN_DIR="/storage/user/maka/train/phase_C_v3"

export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

eval "$(/storage/user/maka/miniconda3/bin/conda shell.bash hook)"
conda activate anycam
cd /tmp

echo "============================================"
echo "  Phase C v3 — Frozen Backbones (ADA/AMPERE)"
echo "  Only pose_head + FAT trainable (~25M params)"
echo "  Host: $(hostname)"
echo "  GPU:  $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo unknown)"
echo "  Date: $(date)"
echo "============================================"

# Log VRAM every 60 seconds in background
(while true; do
    echo "VRAM $(date +%H:%M:%S): $(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader 2>/dev/null)"
    sleep 60
done) &
VRAM_PID=$!
trap "kill $VRAM_PID 2>/dev/null" EXIT

mkdir -p "$TRAIN_DIR"

PHASE_A_CKPT="/storage/user/maka/train/phase_A_v2/checkpoints/latest.pt"
PHASE_B1_CKPT="/storage/user/maka/train/phase_B1/checkpoints/latest.pt"

if [ ! -f "$PHASE_B1_CKPT" ]; then
    echo "ERROR: Phase B1 checkpoint not found: $PHASE_B1_CKPT"
    exit 1
fi
if [ ! -f "$PHASE_A_CKPT" ]; then
    echo "ERROR: Phase A checkpoint not found: $PHASE_A_CKPT"
    exit 1
fi

# Find best checkpoint to resume from
RESUME_CKPT=""
if [ -f "$TRAIN_DIR/checkpoints/latest.pt" ]; then
    RESUME_CKPT="$TRAIN_DIR/checkpoints/latest.pt"
elif ls "$TRAIN_DIR"/checkpoints/intra_epoch*_save.pt 1>/dev/null 2>&1; then
    RESUME_CKPT=$(ls -t "$TRAIN_DIR"/checkpoints/intra_epoch*_save.pt | head -1)
fi

echo ""
echo "=== Phase C v3 Training (10 epochs, batch_size=12, frozen backbones) ==="
echo "  Phase A checkpoint: $PHASE_A_CKPT"
echo "  Phase B1 checkpoint: $PHASE_B1_CKPT"
echo "  Resume checkpoint: ${RESUME_CKPT:-none}"
python3 "$REPO/experiments/train_unified.py" \
    --phase C \
    --data_dir "$PREPROC_DIR" \
    --save_dir "$TRAIN_DIR" \
    --anycam_config "$REPO/pretrained_models/anycam_seq8/training_config.yaml" \
    --pretrained_anycam "$REPO/pretrained_models/anycam_seq8/training_checkpoint_247500.pt" \
    --phase_b1_checkpoint "$PHASE_B1_CKPT" \
    --phase_a_checkpoint "$PHASE_A_CKPT" \
    --val_baselines "$PREPROC_DIR/val_baselines.pt" \
    --num_epochs 10 \
    --batch_size 12 \
    --learning_rate 1e-4 \
    --lambda_calib 1e-4 \
    --max_ahead 3 \
    --image_size 336 \
    ${RESUME_CKPT:+--resume "$RESUME_CKPT"} \
    2>&1

echo ""
echo "=== Phase C v3 COMPLETE ==="
echo "Date: $(date)"
