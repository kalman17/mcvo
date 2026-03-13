#!/bin/bash
#SBATCH --job-name=thesis_viz
#SBATCH --output=/storage/user/maka/logs/thesis_viz_%j.out
#SBATCH --error=/storage/user/maka/logs/thesis_viz_%j.err
#SBATCH --partition=NORMAL
#SBATCH --comment="Masters thesis deadline 2026-03-14, generate trajectory visualizations"
#SBATCH --gres=gpu:1,VRAM:48G
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=4:00:00

set -euo pipefail

REPO="/storage/user/maka/anycam"
CHECKPOINT="/storage/user/maka/train/phase_C_v3_h100/checkpoints/epoch_0005.pt"
DATA_ROOT="/storage/user/maka/eval_datasets"
OUTPUT_DIR="/storage/user/maka/thesis_figures"

export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

eval "$(/storage/user/maka/miniconda3/bin/conda shell.bash hook)"
conda activate anycam
cd /tmp

echo "============================================"
echo "  ATE Scan: Find best sequence for trajectory viz"
echo "  Host: $(hostname)"
echo "  GPU:  $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo unknown)"
echo "  Date: $(date)"
echo "============================================"

mkdir -p "$OUTPUT_DIR"

# Run 1: Sintel — auto cherry-pick by ATE (no --sequence = scan all)
python3 "$REPO/experiments/generate_thesis_visualizations.py" \
    --checkpoint "$CHECKPOINT" \
    --anycam_config "$REPO/pretrained_models/anycam_seq8/training_config.yaml" \
    --pretrained_anycam "$REPO/pretrained_models/anycam_seq8/training_checkpoint_247500.pt" \
    --data_root "$DATA_ROOT" \
    --output_dir "${OUTPUT_DIR}/viz_sintel_best" \
    --dataset sintel \
    --image_size 336 \
    --video_fps 8 \
    --device cuda:0 \
    2>&1

echo ""
echo "=== Sintel scan done ==="
echo ""

# Run 2: TUM-RGBD — auto cherry-pick by ATE
python3 "$REPO/experiments/generate_thesis_visualizations.py" \
    --checkpoint "$CHECKPOINT" \
    --anycam_config "$REPO/pretrained_models/anycam_seq8/training_config.yaml" \
    --pretrained_anycam "$REPO/pretrained_models/anycam_seq8/training_checkpoint_247500.pt" \
    --data_root "$DATA_ROOT" \
    --output_dir "${OUTPUT_DIR}/viz_tumrgbd_best" \
    --dataset tumrgbd \
    --image_size 336 \
    --video_fps 8 \
    --device cuda:0 \
    2>&1

echo ""
echo "=== VISUALIZATIONS COMPLETE ==="
echo "Output: $OUTPUT_DIR"
ls -lhR "$OUTPUT_DIR/viz_sintel_best/" "$OUTPUT_DIR/viz_tumrgbd_best/" 2>/dev/null
echo "Date: $(date)"
