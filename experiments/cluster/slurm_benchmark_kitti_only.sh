#!/bin/bash
#SBATCH --job-name=bench_kitti
#SBATCH --output=/storage/user/maka/logs/bench_kitti_training_fps_%j.out
#SBATCH --error=/storage/user/maka/logs/bench_kitti_training_fps_%j.err
#SBATCH --partition=NORMAL
#SBATCH --comment="Masters thesis deadline 2026-03-31, KITTI benchmark at training FPS"
#SBATCH --constraint="GPU_GEN:ADA|GPU_GEN:AMPERE|GPU_GEN:HOPPER"
#SBATCH --gres=gpu:1,VRAM:40G
#SBATCH --cpus-per-task=5
#SBATCH --mem=48G
#SBATCH --time=06:00:00

set -euo pipefail

REPO="/storage/user/maka/anycam"
DATA_ROOT="/storage/user/maka/eval_datasets"
OUTPUT_DIR="/storage/user/maka/train/benchmark_training_fps"

PHASE_C_CKPT="/storage/user/maka/train/phase_C_v3_h100/checkpoints/epoch_0005.pt"
PHASE_DB_CKPT="/storage/user/maka/train/phase_Db_h100/checkpoints/epoch_0002.pt"

export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

eval "$(/storage/user/maka/miniconda3/bin/conda shell.bash hook)"
conda activate anycam
cd /tmp

echo "============================================"
echo "  KITTI-only Benchmark at Training FPS (2fps)"
echo "  dilation_mode=training, kitti dilation=5"
echo "  Host: $(hostname)"
echo "  GPU:  $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo unknown)"
echo "  Date: $(date)"
echo "============================================"

mkdir -p "$OUTPUT_DIR"

# Delete stale sample_indices.json that cached 0 KITTI samples
find "$OUTPUT_DIR" -name "sample_indices.json" -exec rm -v {} \;

# --- 1. Phase C epoch 5 (KITTI only, with baselines) ---
echo ""
echo "=== Phase C epoch 5 — KITTI only (with baselines) ==="
if [ ! -f "$PHASE_C_CKPT" ]; then
    echo "ERROR: Phase C checkpoint not found: $PHASE_C_CKPT"
    exit 1
fi

python3 "$REPO/experiments/benchmark_phase_c_checkpoints.py" \
    --single_checkpoint "$PHASE_C_CKPT" \
    --anycam_config "$REPO/pretrained_models/anycam_seq8/training_config.yaml" \
    --pretrained_anycam "$REPO/pretrained_models/anycam_seq8/training_checkpoint_247500.pt" \
    --data_root "$DATA_ROOT" \
    --datasets "kitti" \
    --num_samples 200 \
    --image_size 336 \
    --dilation_mode training \
    --output_dir "$OUTPUT_DIR" \
    2>&1

echo ""
echo "=== Phase C epoch 5 KITTI DONE ==="
echo "Date: $(date)"

# --- 2. Phase Db epoch 2 (KITTI only, skip baselines) ---
echo ""
echo "=== Phase Db epoch 2 — KITTI only (skip baselines) ==="
if [ ! -f "$PHASE_DB_CKPT" ]; then
    echo "WARNING: Phase Db checkpoint not found: $PHASE_DB_CKPT — skipping"
else
    python3 "$REPO/experiments/benchmark_phase_c_checkpoints.py" \
        --single_checkpoint "$PHASE_DB_CKPT" \
        --anycam_config "$REPO/pretrained_models/anycam_seq8/training_config.yaml" \
        --pretrained_anycam "$REPO/pretrained_models/anycam_seq8/training_checkpoint_247500.pt" \
        --data_root "$DATA_ROOT" \
        --datasets "kitti" \
        --num_samples 200 \
        --image_size 336 \
        --dilation_mode training \
        --skip_baseline \
        --output_dir "$OUTPUT_DIR" \
        2>&1

    echo ""
    echo "=== Phase Db epoch 2 KITTI DONE ==="
fi

echo ""
echo "============================================"
echo "  KITTI BENCHMARK COMPLETE"
echo "  Results: $OUTPUT_DIR"
echo "  Date: $(date)"
echo "============================================"
