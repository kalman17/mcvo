#!/bin/bash
#SBATCH --job-name=vram_C_v3
#SBATCH --output=/storage/user/maka/logs/vram_C_v3_%j.out
#SBATCH --error=/storage/user/maka/logs/vram_C_v3_%j.err
#SBATCH --partition=NORMAL
#SBATCH --constraint="GPU_GEN:ADA|GPU_GEN:AMPERE"
#SBATCH --gres=gpu:1,VRAM:40G
#SBATCH --cpus-per-task=5
#SBATCH --mem=32G
#SBATCH --time=00:30:00

set -euo pipefail

REPO="/storage/user/maka/anycam"
PREPROC_DIR="/storage/user/maka/preprocessed"

export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

eval "$(/storage/user/maka/miniconda3/bin/conda shell.bash hook)"
conda activate anycam
cd /tmp

echo "============================================"
echo "  Phase C v3 VRAM Test — Frozen Backbones"
echo "  Host: $(hostname)"
echo "  GPU:  $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo unknown)"
echo "  Date: $(date)"
echo "============================================"

PHASE_A_CKPT="/storage/user/maka/train/phase_A_v2/checkpoints/latest.pt"
PHASE_B1_CKPT="/storage/user/maka/train/phase_B1/checkpoints/latest.pt"

# Test increasing batch sizes until OOM
for BS in 6 8 10 12 14 16; do
    TRAIN_DIR="/storage/user/maka/train/phase_C_vram_test_bs${BS}"
    mkdir -p "$TRAIN_DIR"

    echo ""
    echo "=== Testing batch_size=$BS ==="
    echo "VRAM before: $(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader 2>/dev/null)"

    # Run 20 batches with timeout
    timeout 180 python3 "$REPO/experiments/train_unified.py" \
        --phase C \
        --data_dir "$PREPROC_DIR" \
        --save_dir "$TRAIN_DIR" \
        --anycam_config "$REPO/pretrained_models/anycam_seq8/training_config.yaml" \
        --pretrained_anycam "$REPO/pretrained_models/anycam_seq8/training_checkpoint_247500.pt" \
        --phase_b1_checkpoint "$PHASE_B1_CKPT" \
        --phase_a_checkpoint "$PHASE_A_CKPT" \
        --num_epochs 1 \
        --batch_size $BS \
        --learning_rate 1e-4 \
        --lambda_calib 1e-4 \
        --max_ahead 3 \
        --image_size 336 \
        --test \
        2>&1
    EXIT_CODE=$?

    echo "VRAM after bs=$BS: $(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader 2>/dev/null)"

    if [ $EXIT_CODE -ne 0 ]; then
        echo "*** batch_size=$BS FAILED (exit code $EXIT_CODE) ***"
        echo "*** Max safe batch size is $((BS - 2)) ***"
        break
    else
        echo "*** batch_size=$BS OK ***"
    fi

    # Clear GPU cache between tests
    python3 -c "import torch; torch.cuda.empty_cache()" 2>/dev/null
    rm -rf "$TRAIN_DIR"
done

echo ""
echo "=== VRAM Test COMPLETE ==="
echo "Date: $(date)"
