#!/bin/bash
#SBATCH --job-name=dbg_kitti
#SBATCH --output=/storage/user/maka/logs/debug_kitti_%j.out
#SBATCH --error=/storage/user/maka/logs/debug_kitti_%j.err
#SBATCH --partition=NORMAL
#SBATCH --comment="Masters thesis deadline 2026-03-31, debug KITTI intrinsics"
#SBATCH --constraint="GPU_GEN:ADA|GPU_GEN:AMPERE|GPU_GEN:HOPPER"
#SBATCH --gres=gpu:1,VRAM:40G
#SBATCH --cpus-per-task=5
#SBATCH --mem=48G
#SBATCH --time=00:30:00

set -euo pipefail

REPO="/storage/user/maka/anycam"

export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

eval "$(/storage/user/maka/miniconda3/bin/conda shell.bash hook)"
conda activate anycam
cd /tmp

python3 "$REPO/experiments/debug_kitti_intrinsics.py" 2>&1
