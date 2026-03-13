#!/bin/bash
#SBATCH --job-name=thesis_3d
#SBATCH --output=/storage/user/maka/logs/thesis_3d_%j.out
#SBATCH --error=/storage/user/maka/logs/thesis_3d_%j.err
#SBATCH --partition=NORMAL
#SBATCH --comment="Masters thesis deadline 2026-03-14, 3D trajectory viz"
#SBATCH --gres=gpu:1,VRAM:48G
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00

set -euo pipefail

REPO="/storage/user/maka/anycam"
CHECKPOINT="/storage/user/maka/train/phase_C_v3_h100/checkpoints/epoch_0005.pt"
DATA_ROOT="/storage/user/maka/eval_datasets"
OUTPUT_BASE="/storage/user/maka/thesis_figures"
TRAJ_CACHE="${OUTPUT_BASE}/market6_trajectories.npz"

export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

eval "$(/storage/user/maka/miniconda3/bin/conda shell.bash hook)"
conda activate anycam
cd /tmp

echo "============================================"
echo "  3D Trajectory — market_6 (save + reuse)"
echo "  Host: $(hostname)"
echo "  GPU:  $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo unknown)"
echo "  Date: $(date)"
echo "============================================"

mkdir -p "$OUTPUT_BASE"

# Step 1: Run inference ONCE, save trajectories + generate frustums-only plot
echo ""
echo "=== Step 1: Inference + save trajectories + frustums-only ==="
python3 "$REPO/experiments/generate_thesis_3d_trajectory.py" \
    --checkpoint "$CHECKPOINT" \
    --anycam_config "$REPO/pretrained_models/anycam_seq8/training_config.yaml" \
    --pretrained_anycam "$REPO/pretrained_models/anycam_seq8/training_checkpoint_247500.pt" \
    --data_root "$DATA_ROOT" \
    --output_dir "${OUTPUT_BASE}/viz_3d_frustums" \
    --dataset sintel \
    --sequence market_6 \
    --frustum_every 4 \
    --video_fps 8 \
    --image_size 336 \
    --device cuda:0 \
    --no_pointcloud \
    --save_trajectories "$TRAJ_CACHE" \
    2>&1

echo ""
echo "=== Step 1 done ==="

# Step 2: Reuse saved trajectories, add point cloud + video
echo ""
echo "=== Step 2: Load trajectories + point cloud + video ==="
python3 "$REPO/experiments/generate_thesis_3d_trajectory.py" \
    --checkpoint "$CHECKPOINT" \
    --anycam_config "$REPO/pretrained_models/anycam_seq8/training_config.yaml" \
    --pretrained_anycam "$REPO/pretrained_models/anycam_seq8/training_checkpoint_247500.pt" \
    --data_root "$DATA_ROOT" \
    --output_dir "${OUTPUT_BASE}/viz_3d_pointcloud" \
    --dataset sintel \
    --sequence market_6 \
    --frustum_every 4 \
    --video_fps 8 \
    --image_size 336 \
    --device cuda:0 \
    --load_trajectories "$TRAJ_CACHE" \
    2>&1

echo ""
echo "=== COMPLETE ==="
echo "Output:"
ls -lh "${OUTPUT_BASE}/viz_3d_frustums/" 2>/dev/null
ls -lh "${OUTPUT_BASE}/viz_3d_pointcloud/" 2>/dev/null
echo "Date: $(date)"
