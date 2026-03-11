#!/bin/bash
#SBATCH --job-name=thesis_fig
#SBATCH --output=/storage/user/maka/logs/thesis_figures_%j.out
#SBATCH --error=/storage/user/maka/logs/thesis_figures_%j.err
#SBATCH --partition=NORMAL
#SBATCH --comment="Masters thesis deadline 2026-03-14, generate thesis figures"
#SBATCH --gres=gpu:1,VRAM:48G
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00

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
echo "  Generate Thesis Figures"
echo "  Host: $(hostname)"
echo "  GPU:  $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo unknown)"
echo "  Date: $(date)"
echo "============================================"

mkdir -p "$OUTPUT_DIR"

python3 "$REPO/experiments/generate_thesis_figures.py" \
    --checkpoint "$CHECKPOINT" \
    --anycam_config "$REPO/pretrained_models/anycam_seq8/training_config.yaml" \
    --pretrained_anycam "$REPO/pretrained_models/anycam_seq8/training_checkpoint_247500.pt" \
    --data_root "$DATA_ROOT" \
    --output_dir "$OUTPUT_DIR" \
    --num_sequences 20 \
    --image_size 336 \
    --device cuda:0 \
    2>&1

echo ""
echo "=== FIGURES COMPLETE ==="
echo "Output: $OUTPUT_DIR"
ls -lh "$OUTPUT_DIR/"
echo "Date: $(date)"
