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
# Preprocess ~20K frames per dataset (4 datasets in parallel via SLURM array).
#
# Datasets:
#   0 = WalkingTours  (raw .mp4, 60fps → stride 6 → ~10fps)
#   1 = EpicKitchens  (raw .mp4, 60fps → stride 6 → ~10fps)
#   2 = YouTubeVOS    (JPEG sequences → assemble to .mp4, then preprocess)
#   3 = RealEstate10K (JPEG sequences → assemble to .mp4, then preprocess)
#
# Usage:
#   mkdir -p /storage/user/maka/logs
#   sbatch /storage/user/maka/anycam/experiments/cluster/slurm_preprocess.sh
#
# Expected runtime: ~2-3 hours per task on a single GPU.

set -euo pipefail

# ─── Paths ──────────────────────────────────────────────────────────────
REPO="/storage/user/maka/anycam"
INCOMING="/storage/group/dataset_mirrors/01_incoming"
PREPROC_DIR="/storage/user/maka/preprocessed_20k"
CLIPS_DIR="/storage/local/maka/assembled_clips"  # Ephemeral, for frame→mp4 assembly
MAX_FRAMES=20000

export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

# ─── Activate conda ─────────────────────────────────────────────────────
eval "$(/storage/user/maka/miniconda3/bin/conda shell.bash hook)"
conda activate anycam

mkdir -p "$PREPROC_DIR"
mkdir -p /storage/user/maka/logs

echo "============================================"
echo "  SLURM Array Task $SLURM_ARRAY_TASK_ID"
echo "  Host: $(hostname)"
echo "  GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "  Date: $(date)"
echo "============================================"

# ─── Dataset config per array index ─────────────────────────────────────
case $SLURM_ARRAY_TASK_ID in
    0)
        DS_NAME="WalkingTours"
        DS_PATH="$INCOMING/WTours/Original_Videos"
        STRIDE=6        # 60fps → ~10fps
        NEEDS_ASSEMBLY=0
        ;;
    1)
        DS_NAME="EpicKitchens"
        DS_PATH="$INCOMING/hd-epickitchens-full/HD-EPIC/Videos"
        STRIDE=6        # 60fps → ~10fps
        NEEDS_ASSEMBLY=0
        ;;
    2)
        DS_NAME="YouTubeVOS"
        DS_PATH="$CLIPS_DIR/YouTubeVOS"
        STRIDE=1        # Already low fps (~6fps)
        NEEDS_ASSEMBLY=1
        FRAMES_SRC="$INCOMING/youtube-vos/train_all_frames/JPEGImages"
        ;;
    3)
        DS_NAME="RealEstate10K"
        DS_PATH="$CLIPS_DIR/RealEstate10K"
        STRIDE=1        # Already low fps
        NEEDS_ASSEMBLY=1
        FRAMES_SRC="$INCOMING/realestate10k/frames_720/train"
        ;;
    *)
        echo "ERROR: Unknown array task ID: $SLURM_ARRAY_TASK_ID"
        exit 1
        ;;
esac

echo ""
echo "Dataset:    $DS_NAME"
echo "Source:     $DS_PATH"
echo "Stride:     $STRIDE"
echo "Max frames: $MAX_FRAMES"
echo "Assembly:   $NEEDS_ASSEMBLY"

# ─── Step 1: Assemble JPEG sequences to .mp4 (if needed) ───────────────
if [ "$NEEDS_ASSEMBLY" -eq 1 ]; then
    echo ""
    echo "============================================"
    echo "  Assembling JPEG sequences → .mp4 clips"
    echo "============================================"

    mkdir -p "$DS_PATH"

    # Count available sequences
    TOTAL_SEQS=$(ls -d "$FRAMES_SRC"/*/ 2>/dev/null | wc -l)
    echo "Total sequences available: $TOTAL_SEQS"

    # Assemble enough sequences to cover MAX_FRAMES.
    # Conservative estimate: avg ~50 frames/seq for YouTubeVOS, ~15 for RE10K.
    # Assemble up to 1000 sequences (covers 20K+ frames easily).
    MAX_SEQS=1000
    assembled=0
    total_assembled_frames=0

    for seq_dir in $(ls -d "$FRAMES_SRC"/*/ | head -$MAX_SEQS); do
        seq_name=$(basename "$seq_dir")
        out_mp4="$DS_PATH/${seq_name}.mp4"

        if [ -f "$out_mp4" ]; then
            assembled=$((assembled + 1))
            continue
        fi

        # Count frames in this sequence
        num_frames=$(ls "$seq_dir"/*.jpg 2>/dev/null | wc -l)
        if [ "$num_frames" -lt 3 ]; then
            continue  # Skip tiny sequences
        fi

        # Check if frames are sequentially named
        first_frame=$(ls "$seq_dir"/*.jpg | sort | head -1)
        first_name=$(basename "$first_frame" .jpg)

        # Create temp dir with sequential symlinks if needed
        TMP_LINKS=$(mktemp -d)
        idx=0
        for f in $(ls "$seq_dir"/*.jpg | sort); do
            ln -s "$f" "$TMP_LINKS/$(printf '%06d.jpg' $idx)"
            idx=$((idx + 1))
        done

        # Assemble to mp4
        ffmpeg -y -framerate 30 -i "$TMP_LINKS/%06d.jpg" \
            -c:v libx264 -preset fast -crf 18 -pix_fmt yuv420p \
            "$out_mp4" 2>/dev/null

        rm -rf "$TMP_LINKS"
        assembled=$((assembled + 1))
        total_assembled_frames=$((total_assembled_frames + num_frames))

        if (( assembled % 100 == 0 )); then
            echo "  Assembled $assembled sequences ($total_assembled_frames frames so far)..."
        fi

        # Stop once we have enough raw frames for preprocessing
        if [ "$total_assembled_frames" -gt $((MAX_FRAMES * 2)) ]; then
            echo "  Enough frames assembled ($total_assembled_frames). Stopping assembly."
            break
        fi
    done

    echo "  Assembly complete: $assembled clips, ~$total_assembled_frames frames"
fi

# ─── Step 2: Run preprocessing ──────────────────────────────────────────
echo ""
echo "============================================"
echo "  Preprocessing: $DS_NAME"
echo "============================================"

# Change to /tmp to avoid anycalib namespace conflict
cd /tmp

python3 "$REPO/experiments/preprocess_dataset.py" \
    --dataset_path "$DS_PATH" \
    --output_dir "$PREPROC_DIR" \
    --dataset_name "$DS_NAME" \
    --image_size 336 \
    --frame_stride "$STRIDE" \
    --max_total_frames "$MAX_FRAMES" \
    2>&1

echo ""
echo "============================================"
echo "  $DS_NAME COMPLETE"
echo "  $(date)"
echo "============================================"
