#!/bin/bash
#SBATCH --job-name=bench_C_q
#SBATCH --output=/storage/user/maka/logs/bench_C_quick_%j.out
#SBATCH --error=/storage/user/maka/logs/bench_C_quick_%j.err
#SBATCH --partition=NORMAL
#SBATCH --comment="Masters thesis deadline 2026-03-31, Phase C quick benchmark"
#SBATCH --gres=gpu:1,VRAM:24G
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0-04:00:00

set -euo pipefail

REPO="/storage/user/maka/anycam"
DATA_ROOT="/storage/user/maka/eval_datasets"
CKPT="/storage/user/maka/train/phase_C_v2_h100/checkpoints/epoch_0003.pt"
OUTPUT_DIR="/storage/user/maka/train/phase_C_v2_h100/benchmark_results/epoch_0003"

export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

eval "$(/storage/user/maka/miniconda3/bin/conda shell.bash hook)"
conda activate anycam
cd /tmp

echo "============================================"
echo "  Phase C — Quick Benchmark (epoch 3)"
echo "  Host: $(hostname)"
echo "  GPU:  $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo unknown)"
echo "  Date: $(date)"
echo "============================================"

mkdir -p "$OUTPUT_DIR"

python3 "$REPO/experiments/benchmark_phase_c_checkpoints.py" \
    --single_checkpoint "$CKPT" \
    --anycam_config "$REPO/pretrained_models/anycam_seq8/training_config.yaml" \
    --pretrained_anycam "$REPO/pretrained_models/anycam_seq8/training_checkpoint_247500.pt" \
    --data_root "$DATA_ROOT" \
    --mode quick \
    --image_size 336 \
    --output_dir "$OUTPUT_DIR" \
    2>&1

echo ""
echo "=== Benchmark COMPLETE ==="
echo "Date: $(date)"
