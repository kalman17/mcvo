#!/bin/bash
#SBATCH --job-name=bench_v6
#SBATCH --output=/storage/user/maka/logs/bench_v6_%j.out
#SBATCH --error=/storage/user/maka/logs/bench_v6_%j.err
#SBATCH --partition=NORMAL
#SBATCH --comment="Masters thesis deadline 2026-03-31"
#SBATCH --gres=gpu:1,VRAM:24G
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0-04:00:00

set -euo pipefail

REPO="/storage/user/maka/anycam"
DATA_ROOT="/storage/user/maka/eval_datasets"
OUTPUT_DIR="/storage/user/maka/train/benchmark_v6"

# v6 48GB intra-epoch 2 checkpoint
CB_CKPT="/storage/user/maka/train/phase_Cb_v6_48g/checkpoints/intra_epoch2_save.pt"
PRETRAINED="$REPO/pretrained_models/anycam_seq8/training_checkpoint_247500.pt"
ANYCAM_CFG="$REPO/pretrained_models/anycam_seq8/training_config.yaml"

export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

eval "$(/storage/user/maka/miniconda3/bin/conda shell.bash hook)"
conda activate anycam
cd /tmp

echo "============================================"
echo "  Benchmark v6 (intra-epoch 2, flow composition)"
echo "  Host: $(hostname)"
echo "  GPU:  $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo unknown)"
echo "  Date: $(date)"
echo "============================================"

mkdir -p "$OUTPUT_DIR"

# Reuse same sample indices for fair comparison
cp /storage/user/maka/train/benchmark_v3/sample_indices.json "$OUTPUT_DIR/" 2>/dev/null || true

echo ""
echo "=== Benchmarking v6 intra-epoch 2 (quick mode) ==="
python3 "$REPO/experiments/benchmark_phase_c_checkpoints.py" \
    --single_checkpoint "$CB_CKPT" \
    --anycam_config "$ANYCAM_CFG" \
    --pretrained_anycam "$PRETRAINED" \
    --data_root "$DATA_ROOT" \
    --mode quick \
    --image_size 336 \
    --skip_baseline \
    --output_dir "$OUTPUT_DIR" \
    2>&1

echo ""
echo "=== Benchmark v6 COMPLETE ==="
echo "Date: $(date)"
