#!/bin/bash
# Prepare 10-second test clips from each dataset for preprocessing test.
# Run on devcube1 cluster.
#
# Datasets:
#   1. WalkingTours  — full .mp4, extract 10sec from middle
#   2. RealEstate10K — pre-extracted frames, assemble to .mp4
#   3. YouTubeVOS    — pre-extracted frames, assemble to .mp4
#   4. EpicKitchens  — HD .mp4, extract 10sec from middle
#   (OpenDV — not available on cluster, skipped)

set -euo pipefail

INCOMING="/storage/group/dataset_mirrors/01_incoming"
CLIPS_DIR="/storage/local/maka/video_clips_5ds"
PREPROC_DIR="/storage/local/maka/preprocessed_336"
REPO="/storage/user/maka/anycam"

mkdir -p "$CLIPS_DIR"
rm -rf "$PREPROC_DIR"

echo "============================================"
echo "  Preparing 10sec clips from 4 datasets"
echo "============================================"

# ---------------------------------------------------------------
# 1. WalkingTours — extract 10sec from middle of Amsterdam video
# ---------------------------------------------------------------
echo ""
echo "[1/4] WalkingTours"
WTOURS_VIDEO="$INCOMING/WTours/Original_Videos/Amsterdam/Walking in AMSTERDAM ⧸ Netherlands 🇳🇱- 4K 60fps (UHD).mp4"
WTOURS_OUT="$CLIPS_DIR/WalkingTours"
mkdir -p "$WTOURS_OUT"

# Video is ~4912sec, take from ~2450sec (middle)
ffmpeg -y -ss 2450 -t 10 -i "$WTOURS_VIDEO" \
    -c:v libx264 -preset fast -crf 18 -an \
    "$WTOURS_OUT/Amsterdam_10sec.mp4" 2>/dev/null
echo "  Created: $WTOURS_OUT/Amsterdam_10sec.mp4"
ffprobe -v error -count_frames -select_streams v:0 \
    -show_entries stream=nb_read_frames -of csv=p=0 \
    "$WTOURS_OUT/Amsterdam_10sec.mp4"

# ---------------------------------------------------------------
# 2. RealEstate10K — assemble frames into .mp4
# ---------------------------------------------------------------
echo ""
echo "[2/4] RealEstate10K"
RE10K_DIR="$INCOMING/realestate10k/frames_720/test"
RE10K_OUT="$CLIPS_DIR/RealEstate10K"
mkdir -p "$RE10K_OUT"

# Find the sequence with most frames
RE10K_BEST=""
RE10K_MAX=0
for d in $(ls "$RE10K_DIR" | shuf -n 500); do
    c=$(ls "$RE10K_DIR/$d/" 2>/dev/null | wc -l)
    if [ "$c" -gt "$RE10K_MAX" ]; then
        RE10K_MAX=$c
        RE10K_BEST=$d
    fi
done
echo "  Best sequence: $RE10K_BEST ($RE10K_MAX frames)"

# Frames have non-sequential names (e.g. 45979.jpg, 46312.jpg).
# Create sequential symlinks, then assemble to video.
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
echo "  Created: $RE10K_OUT/${RE10K_BEST}_clip.mp4 ($RE10K_MAX frames)"

# ---------------------------------------------------------------
# 3. YouTubeVOS — assemble frames into .mp4
# ---------------------------------------------------------------
echo ""
echo "[3/4] YouTubeVOS"
YTVOS_DIR="$INCOMING/youtube-vos/train_all_frames/JPEGImages"
YTVOS_OUT="$CLIPS_DIR/YouTubeVOS"
mkdir -p "$YTVOS_OUT"

# Find sequence with most frames
YTVOS_BEST=""
YTVOS_MAX=0
for d in $(ls "$YTVOS_DIR" | shuf -n 500); do
    c=$(ls "$YTVOS_DIR/$d/" 2>/dev/null | wc -l)
    if [ "$c" -gt "$YTVOS_MAX" ]; then
        YTVOS_MAX=$c
        YTVOS_BEST=$d
    fi
done
echo "  Best sequence: $YTVOS_BEST ($YTVOS_MAX frames)"

# YouTubeVOS frames are named 00000.jpg, 00001.jpg, ... (sequential)
# Take up to 300 frames from the middle
YTVOS_TOTAL=$(ls "$YTVOS_DIR/$YTVOS_BEST/"*.jpg | wc -l)
YTVOS_SKIP=$(( (YTVOS_TOTAL - 300) / 2 ))
if [ "$YTVOS_SKIP" -lt 0 ]; then YTVOS_SKIP=0; fi

YTVOS_TMP=$(mktemp -d)
i=0
for f in $(ls "$YTVOS_DIR/$YTVOS_BEST/"*.jpg | sort | tail -n +$((YTVOS_SKIP + 1)) | head -300); do
    ln -s "$f" "$YTVOS_TMP/$(printf '%06d.jpg' $i)"
    i=$((i + 1))
done

ffmpeg -y -framerate 30 -i "$YTVOS_TMP/%06d.jpg" \
    -c:v libx264 -preset fast -crf 18 -pix_fmt yuv420p \
    "$YTVOS_OUT/${YTVOS_BEST}_clip.mp4" 2>/dev/null
rm -rf "$YTVOS_TMP"
echo "  Created: $YTVOS_OUT/${YTVOS_BEST}_clip.mp4 ($i frames)"

# ---------------------------------------------------------------
# 4. EpicKitchens — extract 10sec from middle of one video
# ---------------------------------------------------------------
echo ""
echo "[4/4] EpicKitchens"
EPIC_VIDEOS_DIR="$INCOMING/hd-epickitchens-full/HD-EPIC/Videos/P05"
EPIC_OUT="$CLIPS_DIR/EpicKitchens"
mkdir -p "$EPIC_OUT"

# Pick first video
EPIC_VIDEO=$(ls "$EPIC_VIDEOS_DIR"/*.mp4 | head -1)
EPIC_NAME=$(basename "$EPIC_VIDEO" .mp4)
# Get duration, take from middle
EPIC_DUR=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$EPIC_VIDEO" 2>/dev/null)
EPIC_MID=$(python3 -c "print(int(float('${EPIC_DUR}') / 2))")

ffmpeg -y -ss "$EPIC_MID" -t 10 -i "$EPIC_VIDEO" \
    -c:v libx264 -preset fast -crf 18 -an \
    "$EPIC_OUT/${EPIC_NAME}_10sec.mp4" 2>/dev/null
echo "  Created: $EPIC_OUT/${EPIC_NAME}_10sec.mp4"
ffprobe -v error -count_frames -select_streams v:0 \
    -show_entries stream=nb_read_frames -of csv=p=0 \
    "$EPIC_OUT/${EPIC_NAME}_10sec.mp4"

# ---------------------------------------------------------------
# Summary
# ---------------------------------------------------------------
echo ""
echo "============================================"
echo "  Clips created in: $CLIPS_DIR"
echo "============================================"
for ds in $(ls "$CLIPS_DIR"); do
    echo "  $ds:"
    for f in "$CLIPS_DIR/$ds"/*.mp4; do
        frames=$(ffprobe -v error -count_frames -select_streams v:0 \
            -show_entries stream=nb_read_frames -of csv=p=0 "$f" 2>/dev/null)
        echo "    $(basename $f): $frames frames"
    done
done
