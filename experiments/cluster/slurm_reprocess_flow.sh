#!/bin/bash
#SBATCH --job-name=reflow
#SBATCH --output=/storage/user/maka/logs/reflow_%A_%a.out
#SBATCH --error=/storage/user/maka/logs/reflow_%A_%a.err
#SBATCH --partition=NORMAL
#SBATCH --constraint="GPU_GEN:AMPERE|GPU_GEN:ADA|GPU_GEN:HOPPER"
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --array=0-3
#
# Reprocess only flow fields in existing .npz files (fix flow scaling bug).
# Preserves depth and calibration data. Only loads UniMatch.
#
# Datasets:
#   0 = WalkingTours
#   1 = EpicKitchens
#   2 = YouTubeVOS
#   3 = RealEstate10K
#
# Usage:
#   sbatch /storage/user/maka/anycam/experiments/cluster/slurm_reprocess_flow.sh

set -euo pipefail

REPO="/storage/user/maka/anycam"
RESIZED_DIR="/storage/user/maka/resized_336"
PREPROC_DIR="/storage/user/maka/preprocessed"
TARGET_SIZE=336

export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

eval "$(/storage/user/maka/miniconda3/bin/conda shell.bash hook)"
conda activate anycam

mkdir -p /storage/user/maka/logs

echo "============================================"
echo "  Flow Reprocessing — Array Task $SLURM_ARRAY_TASK_ID"
echo "  Host: $(hostname)"
echo "  GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "  Date: $(date)"
echo "============================================"

case $SLURM_ARRAY_TASK_ID in
    0) DS_NAME="WalkingTours" ;;
    1) DS_NAME="EpicKitchens" ;;
    2) DS_NAME="YouTubeVOS" ;;
    3) DS_NAME="RealEstate10K" ;;
    *)
        echo "ERROR: Unknown array task ID: $SLURM_ARRAY_TASK_ID"
        exit 1
        ;;
esac

echo "Dataset: $DS_NAME"
echo "Source:  $RESIZED_DIR/$DS_NAME"
echo "Output:  $PREPROC_DIR"

cd /tmp  # Avoid anycalib namespace conflict

python3 "$REPO/experiments/preprocess_dataset.py" \
    --dataset_path "$RESIZED_DIR/$DS_NAME" \
    --output_dir "$PREPROC_DIR" \
    --dataset_name "$DS_NAME" \
    --image_size "$TARGET_SIZE" \
    --flow_only \
    2>&1

echo ""
echo "============================================"
echo "  $DS_NAME flow reprocessing COMPLETE"
echo "  $(date)"
echo "============================================"
