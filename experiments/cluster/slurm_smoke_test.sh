#!/bin/bash
#SBATCH --job-name=smoke_test
#SBATCH --output=/storage/user/maka/logs/smoke_test_%j.out
#SBATCH --error=/storage/user/maka/logs/smoke_test_%j.err
#SBATCH --partition=NORMAL
#SBATCH --constraint="GPU_GEN:AMPERE|GPU_GEN:ADA|GPU_GEN:HOPPER"
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#
# End-to-end smoke test: prepare 10-sec clips, preprocess at 336x336,
# precompute vanilla baselines, and run all training phases (A/B1/B2/C).
#
# Validates that the full pipeline works on a SLURM GPU node before
# launching the large-scale preprocessing job.
#
# Usage:
#   mkdir -p /storage/user/maka/logs
#   sbatch /storage/user/maka/anycam/experiments/cluster/slurm_smoke_test.sh

set -euo pipefail

REPO="/storage/user/maka/anycam"
INCOMING="/storage/group/dataset_mirrors/01_incoming"
CLIPS_DIR="/storage/local/maka/smoke_clips"
PREPROC_DIR="/storage/local/maka/smoke_preprocessed"

export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

eval "$(/storage/user/maka/miniconda3/bin/conda shell.bash hook)"
conda activate anycam

# Work from /tmp to avoid anycalib namespace conflict
cd /tmp

echo "============================================"
echo "  SMOKE TEST — Full Pipeline"
echo "  Host: $(hostname)"
echo "  GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "  Date: $(date)"
echo "============================================"

# Clean up any prior smoke test data
rm -rf "$CLIPS_DIR" "$PREPROC_DIR"
mkdir -p "$CLIPS_DIR"

# ─────────────────────────────────────────────────────────────────────
#  Step 1: Prepare 10-sec test clips from each dataset (CPU only)
# ─────────────────────────────────────────────────────────────────────
echo ""
echo "============================================"
echo "  Step 1: Preparing 10-sec test clips"
echo "============================================"

# 1a. WalkingTours — extract 10sec from middle of Amsterdam video
echo ""
echo "[1/4] WalkingTours"
WTOURS_VIDEO="$INCOMING/WTours/Original_Videos/Amsterdam/Walking in AMSTERDAM ⧸ Netherlands 🇳🇱- 4K 60fps (UHD).mp4"
WTOURS_OUT="$CLIPS_DIR/WalkingTours"
mkdir -p "$WTOURS_OUT"
ffmpeg -y -ss 2450 -t 10 -i "$WTOURS_VIDEO" \
    -c:v libx264 -preset fast -crf 18 -an \
    "$WTOURS_OUT/Amsterdam_10sec.mp4" 2>/dev/null
echo "  Created: Amsterdam_10sec.mp4"

# 1b. RealEstate10K — assemble frames from one sequence
echo ""
echo "[2/4] RealEstate10K"
RE10K_DIR="$INCOMING/realestate10k/frames_720/test"
RE10K_OUT="$CLIPS_DIR/RealEstate10K"
mkdir -p "$RE10K_OUT"

RE10K_BEST=""
RE10K_MAX=0
for d in $(ls "$RE10K_DIR" | head -500); do
    c=$(ls "$RE10K_DIR/$d/" 2>/dev/null | wc -l)
    if [ "$c" -gt "$RE10K_MAX" ]; then
        RE10K_MAX=$c
        RE10K_BEST=$d
    fi
done
echo "  Best sequence: $RE10K_BEST ($RE10K_MAX frames)"

RE10K_TMP=$(mktemp -d)
i=0
for f in $(ls "$RE10K_DIR/$RE10K_BEST/"*.jpg | sort -t/ -k1 -V); do
    ln -s "$f" "$RE10K_TMP/$(printf '%06d.jpg' $i)"
    i=$((i + 1))
done
ffmpeg -y -framerate 30 -i "$RE10K_TMP/%06d.jpg" \
    -c:v libx264 -preset fast -crf 18 -pix_fmt yuv420p \
    "$RE10K_OUT/${RE10K_BEST}_clip.mp4" 2>/dev/null
rm -rf "$RE10K_TMP"
echo "  Created: ${RE10K_BEST}_clip.mp4 ($RE10K_MAX frames)"

# 1c. YouTubeVOS — assemble frames from one sequence
echo ""
echo "[3/4] YouTubeVOS"
YTVOS_DIR="$INCOMING/youtube-vos/train_all_frames/JPEGImages"
YTVOS_OUT="$CLIPS_DIR/YouTubeVOS"
mkdir -p "$YTVOS_OUT"

YTVOS_BEST=""
YTVOS_MAX=0
for d in $(ls "$YTVOS_DIR" | head -500); do
    c=$(ls "$YTVOS_DIR/$d/" 2>/dev/null | wc -l)
    if [ "$c" -gt "$YTVOS_MAX" ]; then
        YTVOS_MAX=$c
        YTVOS_BEST=$d
    fi
done
echo "  Best sequence: $YTVOS_BEST ($YTVOS_MAX frames)"

YTVOS_TMP=$(mktemp -d)
i=0
for f in $(ls "$YTVOS_DIR/$YTVOS_BEST/"*.jpg | sort | head -300); do
    ln -s "$f" "$YTVOS_TMP/$(printf '%06d.jpg' $i)"
    i=$((i + 1))
done
ffmpeg -y -framerate 30 -i "$YTVOS_TMP/%06d.jpg" \
    -c:v libx264 -preset fast -crf 18 -pix_fmt yuv420p \
    "$YTVOS_OUT/${YTVOS_BEST}_clip.mp4" 2>/dev/null
rm -rf "$YTVOS_TMP"
echo "  Created: ${YTVOS_BEST}_clip.mp4 ($i frames)"

# 1d. EpicKitchens — extract 10sec from middle of one video
echo ""
echo "[4/4] EpicKitchens"
EPIC_VIDEOS_DIR="$INCOMING/hd-epickitchens-full/HD-EPIC/Videos/P05"
EPIC_OUT="$CLIPS_DIR/EpicKitchens"
mkdir -p "$EPIC_OUT"

EPIC_VIDEO=$(ls "$EPIC_VIDEOS_DIR"/*.mp4 | head -1)
EPIC_NAME=$(basename "$EPIC_VIDEO" .mp4)
EPIC_DUR=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$EPIC_VIDEO" 2>/dev/null)
EPIC_MID=$(python3 -c "print(int(float('${EPIC_DUR}') / 2))")
ffmpeg -y -ss "$EPIC_MID" -t 10 -i "$EPIC_VIDEO" \
    -c:v libx264 -preset fast -crf 18 -an \
    "$EPIC_OUT/${EPIC_NAME}_10sec.mp4" 2>/dev/null
echo "  Created: ${EPIC_NAME}_10sec.mp4"

# Summary
echo ""
echo "Clips prepared:"
for ds in $(ls "$CLIPS_DIR"); do
    for f in "$CLIPS_DIR/$ds"/*.mp4; do
        frames=$(ffprobe -v error -count_frames -select_streams v:0 \
            -show_entries stream=nb_read_frames -of csv=p=0 "$f" 2>/dev/null)
        echo "  $ds/$(basename $f): $frames frames"
    done
done

# ─────────────────────────────────────────────────────────────────────
#  Step 2: Preprocess all clips at 336x336
# ─────────────────────────────────────────────────────────────────────
echo ""
echo "============================================"
echo "  Step 2: Preprocessing at 336x336"
echo "============================================"

for ds_dir in "$CLIPS_DIR"/*/; do
    ds_name=$(basename "$ds_dir")
    echo ""
    echo ">>> Preprocessing: $ds_name"

    python3 "$REPO/experiments/preprocess_dataset.py" \
        --dataset_path "$ds_dir" \
        --output_dir "$PREPROC_DIR" \
        --dataset_name "$ds_name" \
        --image_size 336 \
        2>&1 | tail -20

    echo "    Done: $ds_name"
done

# ─────────────────────────────────────────────────────────────────────
#  Step 3: Verify preprocessed data
# ─────────────────────────────────────────────────────────────────────
echo ""
echo "============================================"
echo "  Step 3: Verify preprocessed data"
echo "============================================"

python3 -c "
import os, numpy as np
base = '$PREPROC_DIR'
total_frames = 0
for ds in sorted(os.listdir(base)):
    ds_path = os.path.join(base, ds)
    if not os.path.isdir(ds_path) or ds.startswith('_') or ds.startswith('.'):
        continue
    frames = 0
    for vid in sorted(os.listdir(ds_path)):
        vid_path = os.path.join(ds_path, vid)
        if not os.path.isdir(vid_path) or vid.startswith('_'):
            continue
        npz_count = len([f for f in os.listdir(vid_path) if f.endswith('.npz')])
        frames += npz_count
        for f in sorted(os.listdir(vid_path)):
            if f.endswith('.npz'):
                data = np.load(os.path.join(vid_path, f))
                if 'calib' in data:
                    c = data['calib']
                    print(f'  {ds}/{vid}: {npz_count} frames, calib=[{c[0]:.1f}, {c[1]:.1f}, {c[2]:.1f}, {c[3]:.1f}]')
                break
    total_frames += frames
print(f'Total: {total_frames} preprocessed frames')
"

# ─────────────────────────────────────────────────────────────────────
#  Step 4: Precompute vanilla baselines
# ─────────────────────────────────────────────────────────────────────
echo ""
echo "============================================"
echo "  Step 4: Precompute vanilla baselines"
echo "============================================"

python3 "$REPO/experiments/precompute_vanilla_baselines.py" \
    --data_dir "$PREPROC_DIR" \
    --output_path "$PREPROC_DIR/val_baselines.pt" \
    --anycam_config "$REPO/pretrained_models/anycam_seq8/training_config.yaml" \
    --anycam_checkpoint "$REPO/pretrained_models/anycam_seq8/training_checkpoint_247500.pt" \
    --image_size 336 \
    2>&1

# ─────────────────────────────────────────────────────────────────────
#  Step 5: Training smoke tests (all phases)
# ─────────────────────────────────────────────────────────────────────
echo ""
echo "============================================"
echo "  Step 5: Training smoke tests (all phases)"
echo "============================================"

COMMON_ARGS="--data_dir $PREPROC_DIR \
    --anycam_config $REPO/pretrained_models/anycam_seq8/training_config.yaml \
    --val_baselines $PREPROC_DIR/val_baselines.pt \
    --test"

FAILED=0

for phase in A B1 B2 C; do
    echo ""
    echo ">>> Phase $phase"
    SAVE_DIR="/storage/local/maka/smoke_test_phase_$phase"
    rm -rf "$SAVE_DIR"

    if python3 "$REPO/experiments/train_unified.py" \
        --phase "$phase" \
        --save_dir "$SAVE_DIR" \
        $COMMON_ARGS \
        2>&1; then
        echo "    Phase $phase: PASSED"
    else
        echo "    Phase $phase: FAILED"
        FAILED=$((FAILED + 1))
    fi
done

# Phase C with persistent optimizers
echo ""
echo ">>> Phase C (persistent optimizers)"
SAVE_DIR="/storage/local/maka/smoke_test_phase_C_persistent"
rm -rf "$SAVE_DIR"

if python3 "$REPO/experiments/train_unified.py" \
    --phase C \
    --save_dir "$SAVE_DIR" \
    --persistent_optimizers \
    $COMMON_ARGS \
    2>&1; then
    echo "    Phase C persistent: PASSED"
else
    echo "    Phase C persistent: FAILED"
    FAILED=$((FAILED + 1))
fi

# ─────────────────────────────────────────────────────────────────────
#  Cleanup & Summary
# ─────────────────────────────────────────────────────────────────────
rm -rf /storage/local/maka/smoke_test_phase_*
rm -rf "$CLIPS_DIR" "$PREPROC_DIR"

echo ""
echo "============================================"
if [ "$FAILED" -eq 0 ]; then
    echo "  SMOKE TEST PASSED — all phases OK"
    echo "  Safe to launch full preprocessing."
else
    echo "  SMOKE TEST FAILED — $FAILED phase(s) failed"
    echo "  Check logs before launching full preprocessing."
fi
echo "  $(date)"
echo "============================================"

exit $FAILED
