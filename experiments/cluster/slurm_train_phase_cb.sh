#!/bin/bash
#SBATCH --job-name=train_Cb
#SBATCH --output=/storage/user/maka/logs/train_Cb_%j.out
#SBATCH --error=/storage/user/maka/logs/train_Cb_%j.err
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
TRAIN_DIR="/storage/user/maka/train/phase_Cb_v3"

PHASE_A_CKPT="/storage/user/maka/train/phase_A_v3/checkpoints/best.pt"
PHASE_B1_CKPT="/storage/user/maka/train/phase_B1_v2/checkpoints/best.pt"
PRETRAINED="/storage/user/maka/anycam/pretrained_models/anycam_seq8/training_checkpoint_247500.pt"
ANYCAM_CFG="/storage/user/maka/anycam/pretrained_models/anycam_seq8/training_config.yaml"

export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

eval "$(/storage/user/maka/miniconda3/bin/conda shell.bash hook)"
conda activate anycam
cd /tmp

echo "============================================"
echo "  Phase Cb — Joint pose_head + FAT + pose neck (backbones frozen)"
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
echo "=== Phase Cb Training (20 epochs, batch_size=4, lr=2e-5 cosine) ==="
python3 "$REPO/experiments/train_unified.py" \
    --phase Cb \
    --data_dir "$PREPROC_DIR" \
    --save_dir "$TRAIN_DIR" \
    --anycam_config "$ANYCAM_CFG" \
    --pretrained_anycam "$PRETRAINED" \
    --phase_a_checkpoint "$PHASE_A_CKPT" \
    --phase_b1_checkpoint "$PHASE_B1_CKPT" \
    --val_baselines "$PREPROC_DIR/val_baselines.pt" \
    --num_epochs 20 \
    --batch_size 4 \
    --learning_rate 2e-5 \
    --max_ahead 3 \
    --image_size 336 \
    $([ -f "$TRAIN_DIR/checkpoints/latest.pt" ] && echo "--resume $TRAIN_DIR/checkpoints/latest.pt") \
    2>&1

echo ""
echo "=== Phase Cb COMPLETE ==="
echo "Date: $(date)"
