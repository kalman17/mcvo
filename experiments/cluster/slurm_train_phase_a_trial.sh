#!/bin/bash
#SBATCH --job-name=train_A
#SBATCH --output=/storage/user/maka/logs/train_A_%j.out
#SBATCH --error=/storage/user/maka/logs/train_A_%j.err
#SBATCH --partition=NORMAL
#SBATCH --constraint="GPU_GEN:AMPERE|GPU_GEN:ADA|GPU_GEN:HOPPER"
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=32G
#SBATCH --time=01:00:00

set -euo pipefail

REPO="/storage/user/maka/anycam"
INCOMING="/storage/group/dataset_mirrors/01_incoming"
CLIPS_DIR="/storage/local/maka/trial_clips"
PREPROC_DIR="/storage/user/maka/trial_preprocessed"
TRAIN_DIR="/storage/user/maka/train_phase_A_trial"

export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

eval "$(/storage/user/maka/miniconda3/bin/conda shell.bash hook)"
conda activate anycam
cd /tmp

echo "============================================"
echo "  Phase A Trial Training"
echo "  Host: $(hostname)"
echo "  GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "  Date: $(date)"
echo "============================================"

# Check if preprocessed data already exists
mkdir -p "$PREPROC_DIR"
NPZ_COUNT=$(find "$PREPROC_DIR" -name "*.npz" | wc -l)

if [ "$NPZ_COUNT" -gt 0 ]; then
    echo ""
    echo "=== Preprocessed data found at $PREPROC_DIR — skipping Steps 1-2 ==="
    echo "  $NPZ_COUNT .npz files"
else
    echo ""
    echo "=== No preprocessed data found — running Steps 1-2 ==="

    rm -rf "$CLIPS_DIR"
    mkdir -p "$CLIPS_DIR"

    # ── Step 1: Prepare clips ──
    echo ""
    echo "=== Step 1: Preparing test clips ==="

    # WalkingTours
    WTOURS_VIDEO="$INCOMING/WTours/Original_Videos/Amsterdam/Walking in AMSTERDAM ⧸ Netherlands 🇳🇱- 4K 60fps (UHD).mp4"
    WTOURS_OUT="$CLIPS_DIR/WalkingTours"
    mkdir -p "$WTOURS_OUT"
    ffmpeg -y -ss 2450 -t 10 -i "$WTOURS_VIDEO" \
        -c:v libx264 -preset fast -crf 18 -an \
        "$WTOURS_OUT/Amsterdam_10sec.mp4" 2>/dev/null
    echo "  WalkingTours: done"

    # RealEstate10K
    RE10K_DIR="$INCOMING/realestate10k/frames_720/test"
    RE10K_OUT="$CLIPS_DIR/RealEstate10K"
    mkdir -p "$RE10K_OUT"
    RE10K_BEST="" RE10K_MAX=0
    for d in $(ls "$RE10K_DIR" | head -500); do
        c=$(ls "$RE10K_DIR/$d/" 2>/dev/null | wc -l)
        if [ "$c" -gt "$RE10K_MAX" ]; then RE10K_MAX=$c; RE10K_BEST=$d; fi
    done
    RE10K_TMP=$(mktemp -d)
    i=0
    for f in $(ls "$RE10K_DIR/$RE10K_BEST/"*.jpg | sort -t/ -k1 -V); do
        ln -s "$f" "$RE10K_TMP/$(printf '%06d.jpg' $i)"; i=$((i + 1))
    done
    ffmpeg -y -framerate 30 -i "$RE10K_TMP/%06d.jpg" \
        -c:v libx264 -preset fast -crf 18 -pix_fmt yuv420p \
        "$RE10K_OUT/${RE10K_BEST}_clip.mp4" 2>/dev/null
    rm -rf "$RE10K_TMP"
    echo "  RealEstate10K: done ($RE10K_MAX frames)"

    # YouTubeVOS
    YTVOS_DIR="$INCOMING/youtube-vos/train_all_frames/JPEGImages"
    YTVOS_OUT="$CLIPS_DIR/YouTubeVOS"
    mkdir -p "$YTVOS_OUT"
    YTVOS_BEST="" YTVOS_MAX=0
    for d in $(ls "$YTVOS_DIR" | head -500); do
        c=$(ls "$YTVOS_DIR/$d/" 2>/dev/null | wc -l)
        if [ "$c" -gt "$YTVOS_MAX" ]; then YTVOS_MAX=$c; YTVOS_BEST=$d; fi
    done
    YTVOS_TMP=$(mktemp -d)
    i=0
    for f in $(ls "$YTVOS_DIR/$YTVOS_BEST/"*.jpg | sort | head -300); do
        ln -s "$f" "$YTVOS_TMP/$(printf '%06d.jpg' $i)"; i=$((i + 1))
    done
    ffmpeg -y -framerate 30 -i "$YTVOS_TMP/%06d.jpg" \
        -c:v libx264 -preset fast -crf 18 -pix_fmt yuv420p \
        "$YTVOS_OUT/${YTVOS_BEST}_clip.mp4" 2>/dev/null
    rm -rf "$YTVOS_TMP"
    echo "  YouTubeVOS: done ($i frames)"

    # EpicKitchens
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
    echo "  EpicKitchens: done"

    # ── Step 2: Preprocess ──
    echo ""
    echo "=== Step 2: Preprocessing at 336x336 ==="
    for ds_dir in "$CLIPS_DIR"/*/; do
        ds_name=$(basename "$ds_dir")
        echo "  Preprocessing: $ds_name"
        python3 "$REPO/experiments/preprocess_dataset.py" \
            --dataset_path "$ds_dir" \
            --output_dir "$PREPROC_DIR" \
            --dataset_name "$ds_name" \
            --image_size 336 \
            2>&1
        echo "    Done: $ds_name"
    done

    # Clean up clips (preprocessed data persists on NFS)
    rm -rf "$CLIPS_DIR"
fi

# ── Step 3: Precompute baselines (if missing) ──
if [ ! -f "$PREPROC_DIR/val_baselines.pt" ]; then
    echo ""
    echo "=== Step 3: Precompute vanilla baselines ==="
    python3 "$REPO/experiments/precompute_vanilla_baselines.py" \
        --data_dir "$PREPROC_DIR" \
        --output_path "$PREPROC_DIR/val_baselines.pt" \
        --anycam_config "$REPO/pretrained_models/anycam_seq8/training_config.yaml" \
        --anycam_checkpoint "$REPO/pretrained_models/anycam_seq8/training_checkpoint_247500.pt" \
        --image_size 336 \
        2>&1
else
    echo ""
    echo "=== Baselines already exist — skipping Step 3 ==="
fi

mkdir -p "$TRAIN_DIR"

# ── Step 4: Train Phase A (10 epochs) ──
echo ""
echo "=== Step 4: Phase A Training (10 epochs) ==="
python3 "$REPO/experiments/train_unified.py" \
    --phase A \
    --data_dir "$PREPROC_DIR" \
    --save_dir "$TRAIN_DIR" \
    --anycam_config "$REPO/pretrained_models/anycam_seq8/training_config.yaml" \
    --val_baselines "$PREPROC_DIR/val_baselines.pt" \
    --num_epochs 10 \
    --batch_size 2 \
    --learning_rate 1e-4 \
    2>&1

echo ""
echo "=== Phase A Trial COMPLETE ==="
echo "Results saved to: $TRAIN_DIR"
echo "Contents:"
ls -la "$TRAIN_DIR/"
echo ""
echo "Date: $(date)"
