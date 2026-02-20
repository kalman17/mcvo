#!/bin/bash
#SBATCH --job-name=preproc
#SBATCH --output=/storage/user/maka/logs/preproc_%A_%a.out
#SBATCH --error=/storage/user/maka/logs/preproc_%A_%a.err
#SBATCH --partition=NORMAL
#SBATCH --constraint="GPU_GEN:AMPERE|GPU_GEN:ADA|GPU_GEN:HOPPER"
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --array=0-3
#
# Two-step preprocessing: ffmpeg resize+crop → model inference.
#
# Datasets:
#   0 = WalkingTours  (4K .mp4 videos)
#   1 = EpicKitchens  (1080p .mp4 videos)
#   2 = YouTubeVOS    (JPEG sequences)
#   3 = RealEstate10K (JPEG sequences)
#
# Step 1: ffmpeg converts source videos to 360x360 center-crop @ 2fps
# Step 2: Python runs UniMatch/UniDepth/AnyCalib on the small videos
#
# Usage:
#   sbatch /storage/user/maka/anycam/experiments/cluster/slurm_preprocess.sh

set -euo pipefail

# ─── Paths ──────────────────────────────────────────────────────────────
REPO="/storage/user/maka/anycam"
INCOMING="/storage/group/dataset_mirrors/01_incoming"
RESIZED_DIR="/storage/user/maka/resized_360"    # Step 1 output (permanent)
PREPROC_DIR="/storage/user/maka/preprocessed"    # Step 2 output (permanent)
MAX_FRAMES=20000
TARGET_FPS=2
TARGET_SIZE=360

export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

# ─── Activate conda ─────────────────────────────────────────────────────
eval "$(/storage/user/maka/miniconda3/bin/conda shell.bash hook)"
conda activate anycam

mkdir -p "$RESIZED_DIR" "$PREPROC_DIR" /storage/user/maka/logs

echo "============================================"
echo "  SLURM Array Task $SLURM_ARRAY_TASK_ID"
echo "  Host: $(hostname)"
echo "  GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "  CPUs: $SLURM_CPUS_PER_TASK"
echo "  Date: $(date)"
echo "============================================"

# ─── Dataset config per array index ─────────────────────────────────────
case $SLURM_ARRAY_TASK_ID in
    0)
        DS_NAME="WalkingTours"
        SRC_TYPE="video"
        SRC_PATH="$INCOMING/WTours/Original_Videos"
        ;;
    1)
        DS_NAME="EpicKitchens"
        SRC_TYPE="video"
        SRC_PATH="$INCOMING/hd-epickitchens-full/HD-EPIC/Videos"
        ;;
    2)
        DS_NAME="YouTubeVOS"
        SRC_TYPE="jpeg"
        SRC_PATH="$INCOMING/youtube-vos/train_all_frames/JPEGImages"
        NATIVE_FPS=6   # YouTubeVOS is ~6fps
        ;;
    3)
        DS_NAME="RealEstate10K"
        SRC_TYPE="jpeg"
        SRC_PATH="$INCOMING/realestate10k/frames_720/train"
        NATIVE_FPS=2   # RealEstate10K is ~2fps
        ;;
    *)
        echo "ERROR: Unknown array task ID: $SLURM_ARRAY_TASK_ID"
        exit 1
        ;;
esac

DS_RESIZED="$RESIZED_DIR/$DS_NAME"
mkdir -p "$DS_RESIZED"

echo ""
echo "Dataset:     $DS_NAME"
echo "Source type: $SRC_TYPE"
echo "Source:      $SRC_PATH"
echo "Resized dir: $DS_RESIZED"
echo "Target:      ${TARGET_SIZE}x${TARGET_SIZE} @ ${TARGET_FPS}fps"

# ffmpeg filter: center crop to square, then scale to target size, then set fps
VF="fps=${TARGET_FPS},crop=min(iw\,ih):min(iw\,ih),scale=${TARGET_SIZE}:${TARGET_SIZE}"

# ─── Step 1: ffmpeg resize + center crop ─────────────────────────────────
echo ""
echo "============================================"
echo "  Step 1: ffmpeg → ${TARGET_SIZE}x${TARGET_SIZE} @ ${TARGET_FPS}fps"
echo "============================================"

converted=0
skipped=0

if [ "$SRC_TYPE" = "video" ]; then
    # Video datasets: find all .mp4 files, convert each
    while IFS= read -r src_video; do
        rel_path="${src_video#$SRC_PATH/}"
        # Flatten to single directory: replace / with _
        out_name="$(echo "$rel_path" | sed 's|/|_|g')"
        out_mp4="$DS_RESIZED/$out_name"

        if [ -f "$out_mp4" ]; then
            skipped=$((skipped + 1))
            continue
        fi

        echo "  Converting: $rel_path"
        ffmpeg -y -i "$src_video" -vf "$VF" \
            -c:v libx264 -preset fast -crf 18 -pix_fmt yuv420p -an \
            "$out_mp4" 2>/dev/null

        converted=$((converted + 1))
    done < <(find "$SRC_PATH" -name "*.mp4" -type f | sort)

elif [ "$SRC_TYPE" = "jpeg" ]; then
    # JPEG sequence datasets: assemble + resize in one ffmpeg pass
    MAX_SEQS=2000
    total_assembled_frames=0
    seq_count=0

    for seq_dir in $(ls -d "$SRC_PATH"/*/ 2>/dev/null | head -$MAX_SEQS); do
        seq_name=$(basename "$seq_dir")
        out_mp4="$DS_RESIZED/${seq_name}.mp4"

        if [ -f "$out_mp4" ]; then
            skipped=$((skipped + 1))
            seq_count=$((seq_count + 1))
            continue
        fi

        # Count frames
        num_frames=$(ls "$seq_dir"/*.jpg 2>/dev/null | wc -l)
        if [ "$num_frames" -lt 3 ]; then
            continue
        fi

        # Create sequential symlinks
        TMP_LINKS=$(mktemp -d)
        idx=0
        for f in $(ls "$seq_dir"/*.jpg | sort); do
            ln -s "$f" "$TMP_LINKS/$(printf '%06d.jpg' $idx)"
            idx=$((idx + 1))
        done

        # Assemble + resize + crop in one pass
        ffmpeg -y -framerate "${NATIVE_FPS}" -i "$TMP_LINKS/%06d.jpg" \
            -vf "$VF" \
            -c:v libx264 -preset fast -crf 18 -pix_fmt yuv420p \
            "$out_mp4" 2>/dev/null

        rm -rf "$TMP_LINKS"
        converted=$((converted + 1))
        seq_count=$((seq_count + 1))
        total_assembled_frames=$((total_assembled_frames + num_frames))

        if (( seq_count % 200 == 0 )); then
            echo "  Processed $seq_count sequences ($total_assembled_frames raw frames)..."
        fi

        # Stop once we have enough raw frames
        if [ "$total_assembled_frames" -gt $((MAX_FRAMES * 3)) ]; then
            echo "  Enough sequences processed ($total_assembled_frames raw frames). Stopping."
            break
        fi
    done
fi

echo "  Step 1 done: $converted converted, $skipped already existed"
echo "  Output: $(ls "$DS_RESIZED"/*.mp4 2>/dev/null | wc -l) videos in $DS_RESIZED"

# ─── Step 2: Model inference ─────────────────────────────────────────────
echo ""
echo "============================================"
echo "  Step 2: Model inference on resized videos"
echo "============================================"

cd /tmp  # Avoid anycalib namespace conflict

python3 "$REPO/experiments/preprocess_dataset.py" \
    --dataset_path "$DS_RESIZED" \
    --output_dir "$PREPROC_DIR" \
    --dataset_name "$DS_NAME" \
    --image_size "$TARGET_SIZE" \
    --frame_stride 1 \
    --max_total_frames "$MAX_FRAMES" \
    --resume \
    2>&1

echo ""
echo "============================================"
echo "  $DS_NAME COMPLETE"
echo "  $(date)"
echo "============================================"
