#!/bin/bash
#SBATCH --job-name=train_Db_h
#SBATCH --output=/storage/user/maka/logs/train_Db_h100_%j.out
#SBATCH --error=/storage/user/maka/logs/train_Db_h100_%j.err
#SBATCH --partition=NORMAL
#SBATCH --comment="Masters thesis deadline 2026-03-31, Phase Db pose-path layers (H100)"
#SBATCH --constraint="GPU_MODEL:nvidia_h100"
#SBATCH --gres=gpu:1,VRAM:80G
#SBATCH --cpus-per-task=5
#SBATCH --mem=48G
#SBATCH --time=14-00:00:00

set -euo pipefail

REPO="/storage/user/maka/anycam"
PREPROC_DIR="/storage/user/maka/preprocessed"
TRAIN_DIR="/storage/user/maka/train/phase_Db_h100"

export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

eval "$(/storage/user/maka/miniconda3/bin/conda shell.bash hook)"
conda activate anycam
cd /tmp

echo "============================================"
echo "  Phase Db — Pose-Path Fine-Tuning (H100)"
echo "  pose_head + interframe attn + fusion + reassembly (~2.5M params)"
echo "  Calibration (FAT) frozen from Phase C"
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

PHASE_C_CKPT="/storage/user/maka/train/phase_C_v3_h100/checkpoints/latest.pt"

if [ ! -f "$PHASE_C_CKPT" ]; then
    echo "ERROR: Phase C checkpoint not found: $PHASE_C_CKPT"
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
echo "=== Phase Db Training (10 epochs, batch_size=20, pose-path layers, H100) ==="
echo "  Phase C checkpoint: $PHASE_C_CKPT"
echo "  Resume checkpoint: ${RESUME_CKPT:-none}"
python3 "$REPO/experiments/train_unified.py" \
    --phase Db \
    --data_dir "$PREPROC_DIR" \
    --save_dir "$TRAIN_DIR" \
    --anycam_config "$REPO/pretrained_models/anycam_seq8/training_config.yaml" \
    --phase_c_checkpoint "$PHASE_C_CKPT" \
    --val_baselines "$PREPROC_DIR/val_baselines.pt" \
    --num_epochs 10 \
    --batch_size 20 \
    --learning_rate 1e-4 \
    --lambda_calib 1e-4 \
    --max_ahead 3 \
    --image_size 336 \
    ${RESUME_CKPT:+--resume "$RESUME_CKPT"} \
    2>&1

echo ""
echo "=== Phase Db (H100) COMPLETE ==="
echo "Date: $(date)"
