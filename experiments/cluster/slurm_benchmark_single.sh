#!/bin/bash
#SBATCH --job-name=bench_1
#SBATCH --output=/storage/user/maka/logs/bench_single_%j.out
#SBATCH --error=/storage/user/maka/logs/bench_single_%j.err
#SBATCH --partition=NORMAL
#SBATCH --comment="Masters thesis deadline 2026-03-31, Phase C single checkpoint benchmark"
#SBATCH --gres=gpu:1,VRAM:48G
#SBATCH --cpus-per-task=5
#SBATCH --mem=32G
#SBATCH --time=04:00:00

set -euo pipefail

CKPT="${1:?Usage: sbatch slurm_benchmark_single.sh /path/to/epoch_XXXX.pt [quick|full] [frame_count] [anycam|training]}"
MODE="${2:-quick}"
FRAME_COUNT="${3:-4}"
DILATION_MODE="${4:-anycam}"
REPO="/storage/user/maka/anycam"
DATA_ROOT="/storage/user/maka/eval_datasets"
if [ "$DILATION_MODE" = "training" ]; then
    BENCH_DIR="/storage/user/maka/train/phase_C/benchmark_results/fc${FRAME_COUNT}_train_fps"
else
    BENCH_DIR="/storage/user/maka/train/phase_C/benchmark_results/fc${FRAME_COUNT}"
fi

export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

eval "$(/storage/user/maka/miniconda3/bin/conda shell.bash hook)"
conda activate anycam
cd "$REPO"

# Symlink eval datasets from group storage (complete copies already on cluster)
mkdir -p "$DATA_ROOT"
[ ! -e "$DATA_ROOT/Sintel" ] && ln -s /storage/group/dataset_mirrors/01_incoming/Sintel "$DATA_ROOT/Sintel"
[ ! -e "$DATA_ROOT/TUM_RGBD" ] && ln -s /storage/group/dataset_mirrors/01_incoming/TUM_RGBD_Dataset "$DATA_ROOT/TUM_RGBD"
[ ! -e "$DATA_ROOT/kitti_odom_color" ] && ln -s /storage/group/dataset_mirrors/01_incoming/kitti_odom_color "$DATA_ROOT/kitti_odom_color"

echo "============================================"
echo "  Phase C — Single Checkpoint Benchmark"
echo "  Host: $(hostname)"
echo "  GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "  Date: $(date)"
echo "  Checkpoint: $CKPT"
echo "  Mode: $MODE"
echo "  Frame count: $FRAME_COUNT"
echo "  Dilation mode: $DILATION_MODE"
echo "============================================"

python3 "$REPO/experiments/benchmark_phase_c_checkpoints.py" \
    --single_checkpoint "$CKPT" \
    --anycam_config "$REPO/pretrained_models/anycam_seq8/training_config.yaml" \
    --pretrained_anycam "$REPO/pretrained_models/anycam_seq8/training_checkpoint_247500.pt" \
    --data_root "$DATA_ROOT" \
    --mode "$MODE" \
    --frame_count "$FRAME_COUNT" \
    --dilation_mode "$DILATION_MODE" \
    --image_size 336 \
    --output_dir "$BENCH_DIR" \
    2>&1

echo ""
echo "=== Benchmark COMPLETE ==="
echo "Date: $(date)"
echo "Results: $BENCH_DIR"
