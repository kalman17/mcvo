#!/bin/bash
#SBATCH --job-name=bench_C
#SBATCH --output=/storage/user/maka/logs/bench_C_%j.out
#SBATCH --error=/storage/user/maka/logs/bench_C_%j.err
#SBATCH --partition=NORMAL
#SBATCH --comment="Masters thesis deadline 2026-03-31, Phase C benchmark watcher"
#SBATCH --gres=gpu:1,VRAM:48G
#SBATCH --cpus-per-task=5
#SBATCH --mem=32G
#SBATCH --time=14-00:00:00

set -euo pipefail

REPO="/storage/user/maka/anycam"
TRAIN_DIR="/storage/user/maka/train/phase_C"
CKPT_DIR="$TRAIN_DIR/checkpoints"
BENCH_DIR="$TRAIN_DIR/benchmark_results"
DATA_ROOT="/storage/user/maka/eval_datasets"
TRACKING_FILE="$BENCH_DIR/.benchmarked_epochs"

export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

eval "$(/storage/user/maka/miniconda3/bin/conda shell.bash hook)"
conda activate anycam
cd /tmp

echo "============================================"
echo "  Phase C — Autonomous Benchmark Watcher"
echo "  Host: $(hostname)"
echo "  GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "  Date: $(date)"
echo "============================================"
echo ""
echo "  Checkpoint dir: $CKPT_DIR"
echo "  Benchmark dir:  $BENCH_DIR"
echo "  Data root:      $DATA_ROOT"
echo "  Tracking file:  $TRACKING_FILE"
echo "============================================"

mkdir -p "$BENCH_DIR"
touch "$TRACKING_FILE"

# Determine which datasets are available
DATASETS=""
if [ -d "$DATA_ROOT/Sintel/training/final" ]; then
    DATASETS="${DATASETS:+$DATASETS,}sintel"
    echo "[DATA] Sintel: available"
else
    echo "[DATA] Sintel: NOT FOUND"
fi
if [ -d "$DATA_ROOT/TUM_RGBD" ]; then
    DATASETS="${DATASETS:+$DATASETS,}tumrgbd"
    echo "[DATA] TUM-RGBD: available"
else
    echo "[DATA] TUM-RGBD: NOT FOUND"
fi
if [ -d "$DATA_ROOT/LightSpeed" ] && [ -f "$DATA_ROOT/LightSpeed/poses.pkl" ]; then
    DATASETS="${DATASETS:+$DATASETS,}lightspeed"
    echo "[DATA] LightSpeed: available"
else
    echo "[DATA] LightSpeed: NOT FOUND"
fi

if [ -z "$DATASETS" ]; then
    echo ""
    echo "[ERROR] No evaluation datasets found in $DATA_ROOT"
    echo "        Run slurm_download_eval_data.sh first."
    exit 1
fi

echo ""
echo "[CONFIG] Datasets to benchmark: $DATASETS"
echo ""

# Poll loop
POLL_INTERVAL=1800  # 30 minutes

while true; do
    echo ""
    echo "=== Poll cycle at $(date) ==="

    # Find all epoch checkpoint files
    NEW_CHECKPOINTS=0

    for CKPT in "$CKPT_DIR"/epoch_*.pt; do
        [ -e "$CKPT" ] || continue  # Handle no matches

        CKPT_NAME=$(basename "$CKPT")

        # Skip if already benchmarked
        if grep -qF "$CKPT_NAME" "$TRACKING_FILE" 2>/dev/null; then
            continue
        fi

        echo ""
        echo "[NEW] Found new checkpoint: $CKPT_NAME"
        echo "      Waiting 60s for NFS sync..."
        sleep 60

        # Verify file still exists and is not being written
        if [ ! -f "$CKPT" ]; then
            echo "[WARN] Checkpoint disappeared, skipping"
            continue
        fi

        echo "[RUN] Benchmarking $CKPT_NAME on datasets: $DATASETS"

        python3 "$REPO/experiments/benchmark_phase_c_checkpoints.py" \
            --single_checkpoint "$CKPT" \
            --anycam_config "$REPO/pretrained_models/anycam_seq8/training_config.yaml" \
            --pretrained_anycam "$REPO/pretrained_models/anycam_seq8/training_checkpoint_247500.pt" \
            --data_root "$DATA_ROOT" \
            --datasets "$DATASETS" \
            --num_samples 50 \
            --image_size 336 \
            --output_dir "$BENCH_DIR" \
            2>&1

        BENCH_EXIT=$?

        if [ $BENCH_EXIT -eq 0 ]; then
            echo "$CKPT_NAME" >> "$TRACKING_FILE"
            echo "[OK] $CKPT_NAME benchmarked successfully"
            NEW_CHECKPOINTS=$((NEW_CHECKPOINTS + 1))
        else
            echo "[ERROR] Benchmark failed for $CKPT_NAME (exit code $BENCH_EXIT)"
        fi
    done

    # If we benchmarked anything new, update aggregated plots
    if [ $NEW_CHECKPOINTS -gt 0 ]; then
        echo ""
        echo "[AGGREGATE] Updating evolution plots..."
        python3 "$REPO/experiments/aggregate_benchmark_results.py" \
            --results_dir "$BENCH_DIR" \
            2>&1 || echo "[WARN] Aggregation failed"
    fi

    # Check if training is done (all 10 epochs present and benchmarked)
    TOTAL_EPOCHS=$(ls "$CKPT_DIR"/epoch_*.pt 2>/dev/null | wc -l)
    BENCHMARKED=$(wc -l < "$TRACKING_FILE" 2>/dev/null || echo 0)
    echo ""
    echo "[STATUS] Checkpoints: $TOTAL_EPOCHS found, $BENCHMARKED benchmarked"

    if [ "$TOTAL_EPOCHS" -ge 10 ] && [ "$BENCHMARKED" -ge "$TOTAL_EPOCHS" ]; then
        echo ""
        echo "=== All epochs benchmarked! Exiting. ==="
        echo "Date: $(date)"
        break
    fi

    echo "[SLEEP] Next poll in ${POLL_INTERVAL}s ($(date -d "+${POLL_INTERVAL} seconds" 2>/dev/null || echo 'soon'))..."
    sleep $POLL_INTERVAL
done

echo ""
echo "=== Phase C Benchmark Watcher COMPLETE ==="
echo "Date: $(date)"
