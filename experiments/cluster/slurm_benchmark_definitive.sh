#!/bin/bash
#SBATCH --job-name=bench_def
#SBATCH --output=/storage/user/maka/logs/bench_definitive_%j.out
#SBATCH --error=/storage/user/maka/logs/bench_definitive_%j.err
#SBATCH --partition=NORMAL
#SBATCH --comment="Masters thesis deadline 2026-03-31, definitive benchmark best checkpoints"
#SBATCH --constraint="GPU_GEN:ADA|GPU_GEN:AMPERE|GPU_GEN:HOPPER"
#SBATCH --gres=gpu:1,VRAM:40G
#SBATCH --cpus-per-task=5
#SBATCH --mem=48G
#SBATCH --time=18:00:00

set -euo pipefail

REPO="/storage/user/maka/anycam"
DATA_ROOT="/storage/user/maka/eval_datasets"
OUTPUT_DIR="/storage/user/maka/train/benchmark_definitive"

# Best checkpoints by validation loss
PHASE_C_CKPT="/storage/user/maka/train/phase_C_v3_h100/checkpoints/epoch_0003.pt"
PHASE_DB_CKPT="/storage/user/maka/train/phase_Db_h100/checkpoints/epoch_0001.pt"

export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

eval "$(/storage/user/maka/miniconda3/bin/conda shell.bash hook)"
conda activate anycam
cd /tmp

echo "============================================"
echo "  Definitive Benchmark (Best Checkpoints)"
echo "  Phase C e3 (best val), Phase Db e1 (best val)"
echo "  Both dilation modes: training + anycam"
echo "  All datasets: sintel, tumrgbd, kitti"
echo "  Host: $(hostname)"
echo "  GPU:  $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo unknown)"
echo "  Date: $(date)"
echo "============================================"

mkdir -p "$OUTPUT_DIR"

# ================================================================
# PART 1: Training FPS (dilation_mode=training)
# ================================================================
TRAIN_DIR="$OUTPUT_DIR/training_fps"
mkdir -p "$TRAIN_DIR"

echo ""
echo "========================================================"
echo "  PART 1: Training FPS (dilation_mode=training)"
echo "========================================================"

# --- Phase C epoch 3 (with baselines) ---
echo ""
echo "=== Phase C epoch 3 @ training FPS (with baselines) ==="
python3 "$REPO/experiments/benchmark_phase_c_checkpoints.py" \
    --single_checkpoint "$PHASE_C_CKPT" \
    --anycam_config "$REPO/pretrained_models/anycam_seq8/training_config.yaml" \
    --pretrained_anycam "$REPO/pretrained_models/anycam_seq8/training_checkpoint_247500.pt" \
    --data_root "$DATA_ROOT" \
    --datasets "sintel,tumrgbd,kitti" \
    --num_samples 200 \
    --image_size 336 \
    --dilation_mode training \
    --output_dir "$TRAIN_DIR" \
    2>&1

echo ""
echo "=== Phase C epoch 3 @ training FPS DONE ==="
echo "Date: $(date)"

# --- Phase Db epoch 1 (skip baselines, reuse cache) ---
echo ""
echo "=== Phase Db epoch 1 @ training FPS (skip baselines) ==="
python3 "$REPO/experiments/benchmark_phase_c_checkpoints.py" \
    --single_checkpoint "$PHASE_DB_CKPT" \
    --anycam_config "$REPO/pretrained_models/anycam_seq8/training_config.yaml" \
    --pretrained_anycam "$REPO/pretrained_models/anycam_seq8/training_checkpoint_247500.pt" \
    --data_root "$DATA_ROOT" \
    --datasets "sintel,tumrgbd,kitti" \
    --num_samples 200 \
    --image_size 336 \
    --dilation_mode training \
    --skip_baseline \
    --output_dir "$TRAIN_DIR" \
    2>&1

echo ""
echo "=== Phase Db epoch 1 @ training FPS DONE ==="
echo "Date: $(date)"

# ================================================================
# PART 2: AnyCam FPS (dilation_mode=anycam)
# ================================================================
ANYCAM_DIR="$OUTPUT_DIR/anycam_fps"
mkdir -p "$ANYCAM_DIR"

echo ""
echo "========================================================"
echo "  PART 2: AnyCam FPS (dilation_mode=anycam)"
echo "========================================================"

# --- Phase C epoch 3 (with baselines) ---
echo ""
echo "=== Phase C epoch 3 @ anycam FPS (with baselines) ==="
python3 "$REPO/experiments/benchmark_phase_c_checkpoints.py" \
    --single_checkpoint "$PHASE_C_CKPT" \
    --anycam_config "$REPO/pretrained_models/anycam_seq8/training_config.yaml" \
    --pretrained_anycam "$REPO/pretrained_models/anycam_seq8/training_checkpoint_247500.pt" \
    --data_root "$DATA_ROOT" \
    --datasets "sintel,tumrgbd,kitti" \
    --num_samples 200 \
    --image_size 336 \
    --dilation_mode anycam \
    --output_dir "$ANYCAM_DIR" \
    2>&1

echo ""
echo "=== Phase C epoch 3 @ anycam FPS DONE ==="
echo "Date: $(date)"

# --- Phase Db epoch 1 (skip baselines, reuse cache) ---
echo ""
echo "=== Phase Db epoch 1 @ anycam FPS (skip baselines) ==="
python3 "$REPO/experiments/benchmark_phase_c_checkpoints.py" \
    --single_checkpoint "$PHASE_DB_CKPT" \
    --anycam_config "$REPO/pretrained_models/anycam_seq8/training_config.yaml" \
    --pretrained_anycam "$REPO/pretrained_models/anycam_seq8/training_checkpoint_247500.pt" \
    --data_root "$DATA_ROOT" \
    --datasets "sintel,tumrgbd,kitti" \
    --num_samples 200 \
    --image_size 336 \
    --dilation_mode anycam \
    --skip_baseline \
    --output_dir "$ANYCAM_DIR" \
    2>&1

echo ""
echo "=== Phase Db epoch 1 @ anycam FPS DONE ==="
echo "Date: $(date)"

echo ""
echo "============================================"
echo "  ALL DEFINITIVE BENCHMARKS COMPLETE"
echo "  Results: $OUTPUT_DIR"
echo "  Date: $(date)"
echo "============================================"
