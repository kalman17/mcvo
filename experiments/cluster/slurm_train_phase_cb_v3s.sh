#!/bin/bash
#SBATCH --job-name=Cb3s_h100
#SBATCH --output=/storage/user/maka/logs/train_Cb_v3s_h100_%j.out
#SBATCH --error=/storage/user/maka/logs/train_Cb_v3s_h100_%j.err
#SBATCH --partition=NORMAL
#SBATCH --comment="Masters thesis deadline 2026-03-31"
#SBATCH --constraint="GPU_GEN:HOPPER"
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=64G
#SBATCH --time=14-00:00:00

set -euo pipefail

REPO="/storage/user/maka/anycam"
PREPROC_DIR="/storage/user/maka/preprocessed"
TRAIN_DIR="/storage/user/maka/train/phase_Cb_v3s_h100"

# Auto-resume: own latest.pt on preemption, otherwise v3 ep5 weights
INIT_CKPT="/storage/user/maka/train/phase_Cb_v3_h100/checkpoints/epoch_0005.pt"

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
echo "  Phase Cb v3s H100 — v3 continued with stability fixes"
echo "    - Tikhonov regularization in linear solver"
echo "    - Graceful calibrator failure handling"
echo "    - Weight decay 1e-5, grad_clip 0.3"
echo "    - NO composed flow loss (lambda_comp=0)"
echo "    - Resume from Cb v3 H100 epoch 5 (weights only)"
echo "  Host: $(hostname)"
echo "  GPU:  $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo unknown)"
echo "  Date: $(date)"
echo "============================================"

mkdir -p "$TRAIN_DIR"

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

# Determine resume strategy
if [ -f "$TRAIN_DIR/checkpoints/latest.pt" ]; then
    echo "Resuming from own latest checkpoint (preemption recovery)"
    RESUME_ARGS="--resume $TRAIN_DIR/checkpoints/latest.pt"
else
    echo "Starting from initial checkpoint (weights only)"
    RESUME_ARGS="--resume $INIT_CKPT --resume_weights_only"
fi

echo ""
echo "=== Phase Cb v3s H100 Training (20 epochs, batch_size=10, lr=2e-5 cosine, wd=1e-5, clip=0.3, no comp) ==="
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
    --batch_size 10 \
    --learning_rate 2e-5 \
    --weight_decay 1e-5 \
    --grad_clip 0.3 \
    --lambda_comp 0.0 \
    --max_ahead 3 \
    --image_size 336 \
    $RESUME_ARGS \
    2>&1

echo ""
echo "=== Phase Cb v3s H100 COMPLETE ==="
echo "Date: $(date)"
