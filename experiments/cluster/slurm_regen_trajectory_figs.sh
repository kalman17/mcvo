#!/bin/bash
#SBATCH --job-name=regen_figs
#SBATCH --output=/storage/user/maka/logs/regen_figs_%j.out
#SBATCH --error=/storage/user/maka/logs/regen_figs_%j.err
#SBATCH --partition=NORMAL
#SBATCH --comment="Masters thesis deadline 2026-03-14"
#SBATCH --gres=gpu:1,VRAM:24G
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0-01:00:00

set -euo pipefail

REPO="/storage/user/maka/anycam"
DATA_ROOT="/storage/user/maka/eval_datasets"
CHECKPOINT="/storage/user/maka/train/phase_C_v3_h100/checkpoints/epoch_0005.pt"
PRETRAINED="$REPO/pretrained_models/anycam_seq8/training_checkpoint_247500.pt"
ANYCAM_CFG="$REPO/pretrained_models/anycam_seq8/training_config.yaml"
OUTPUT_DIR="$REPO/thesis_results/figures"

export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

eval "$(/storage/user/maka/miniconda3/bin/conda shell.bash hook)"
conda activate anycam
cd /tmp

echo "============================================"
echo "  Regenerate trajectory figures (MCT labels)"
echo "  Host: $(hostname)"
echo "  GPU:  $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo unknown)"
echo "  Date: $(date)"
echo "============================================"

# 1. 2D trajectory plot (market_6 from Sintel)
echo ""
echo "=== Generating 2D trajectory plot ==="
python3 "$REPO/experiments/generate_thesis_visualizations.py" \
    --checkpoint "$CHECKPOINT" \
    --anycam_config "$ANYCAM_CFG" \
    --pretrained_anycam "$PRETRAINED" \
    --data_root "$DATA_ROOT" \
    --dataset sintel \
    --sequence market_6 \
    --output_dir "$OUTPUT_DIR" \
    --device cuda:0 \
    2>&1

echo ""
echo "=== Generating 3D trajectory plot ==="
python3 "$REPO/experiments/generate_thesis_3d_trajectory.py" \
    --checkpoint "$CHECKPOINT" \
    --anycam_config "$ANYCAM_CFG" \
    --pretrained_anycam "$PRETRAINED" \
    --data_root "$DATA_ROOT" \
    --dataset sintel \
    --sequence market_6 \
    --output_dir "$OUTPUT_DIR/viz_3d" \
    --save_trajectories "$OUTPUT_DIR/trajectories_market6.npz" \
    --no_video \
    --device cuda:0 \
    2>&1

# Copy results to latex figures dir
echo ""
echo "=== Copying to latex figures directory ==="
LATEX_FIGS="$REPO/kalman-tum-thesis-latex-master/figures"

# 2D trajectory
if [ -f "$OUTPUT_DIR/trajectory_comparison.png" ]; then
    cp "$OUTPUT_DIR/trajectory_comparison.png" "$LATEX_FIGS/trajectory_comparison_2d.png"
    echo "Copied trajectory_comparison_2d.png"
fi

# 3D side view
if [ -f "$OUTPUT_DIR/viz_3d/trajectory_3d_side.png" ]; then
    cp "$OUTPUT_DIR/viz_3d/trajectory_3d_side.png" "$LATEX_FIGS/trajectory_3d_side.png"
    echo "Copied trajectory_3d_side.png"
fi

echo ""
echo "=== DONE ==="
echo "Date: $(date)"
ls -la "$LATEX_FIGS"/trajectory_*.png
