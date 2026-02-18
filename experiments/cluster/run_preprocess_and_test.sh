#!/bin/bash
# Preprocess all dataset clips at 336x336 and run training smoke tests.
# Run on devcube1 cluster after prepare_test_clips.sh.
#
# Usage:
#   conda activate anycam
#   cd /tmp  # avoid anycalib namespace conflict
#   bash /storage/user/maka/anycam/experiments/cluster/run_preprocess_and_test.sh

set -euo pipefail

REPO="/storage/user/maka/anycam"
CLIPS_DIR="/storage/local/maka/video_clips_5ds"
PREPROC_DIR="/storage/local/maka/preprocessed_336"
export PYTHONPATH="$REPO:$PYTHONPATH"

echo "============================================"
echo "  Step 1: Preprocessing all datasets at 336x336"
echo "============================================"

for ds_dir in "$CLIPS_DIR"/*/; do
    ds_name=$(basename "$ds_dir")
    echo ""
    echo ">>> Preprocessing: $ds_name"
    echo "    Input:  $ds_dir"
    echo "    Output: $PREPROC_DIR/$ds_name"

    python3 "$REPO/experiments/preprocess_dataset.py" \
        --dataset_path "$ds_dir" \
        --output_dir "$PREPROC_DIR" \
        --dataset_name "$ds_name" \
        --image_size 336 \
        2>&1 | tail -20

    echo "    Done: $ds_name"
done

echo ""
echo "============================================"
echo "  Step 2: Verify preprocessed data"
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
        # Check first frame's calibration
        for f in sorted(os.listdir(vid_path)):
            if f.endswith('.npz'):
                data = np.load(os.path.join(vid_path, f))
                if 'calib' in data:
                    c = data['calib']
                    print(f'  {ds}/{vid}: {npz_count} frames, calib=[{c[0]:.1f}, {c[1]:.1f}, {c[2]:.1f}, {c[3]:.1f}]')
                break
    total_frames += frames
print(f'Total: {total_frames} preprocessed frames across {len([d for d in os.listdir(base) if os.path.isdir(os.path.join(base,d)) and not d.startswith(\"_\") and not d.startswith(\".\")])} datasets')
"

echo ""
echo "============================================"
echo "  Step 3: Precompute vanilla baselines"
echo "============================================"

python3 "$REPO/experiments/precompute_vanilla_baselines.py" \
    --data_dir "$PREPROC_DIR" \
    --output_path "$PREPROC_DIR/val_baselines.pt" \
    --anycam_config "$REPO/pretrained_models/anycam_seq8/training_config.yaml" \
    --anycam_checkpoint "$REPO/pretrained_models/anycam_seq8/training_checkpoint_247500.pt" \
    --image_size 336 \
    2>&1

echo ""
echo "============================================"
echo "  Step 4: Training smoke tests (all phases)"
echo "============================================"

COMMON_ARGS="--data_dir $PREPROC_DIR \
    --anycam_config $REPO/pretrained_models/anycam_seq8/training_config.yaml \
    --val_baselines $PREPROC_DIR/val_baselines.pt \
    --test"

for phase in A B1 B3 C; do
    echo ""
    echo ">>> Phase $phase"
    SAVE_DIR="/storage/local/maka/test_run_phase_$phase"
    rm -rf "$SAVE_DIR"

    python3 "$REPO/experiments/train_unified.py" \
        --phase "$phase" \
        --save_dir "$SAVE_DIR" \
        $COMMON_ARGS \
        2>&1 | grep -E "INFO|ERROR|PASSED|FAILED|Trainable|Dataset|Validation"

    echo "    Phase $phase done."
done

echo ""
echo "============================================"
echo "  Step 5: Phase C with persistent optimizers"
echo "============================================"

SAVE_DIR="/storage/local/maka/test_run_phase_C_persistent"
rm -rf "$SAVE_DIR"

python3 "$REPO/experiments/train_unified.py" \
    --phase C \
    --save_dir "$SAVE_DIR" \
    --persistent_optimizers \
    $COMMON_ARGS \
    2>&1 | grep -E "INFO|ERROR|PASSED|FAILED|Trainable|Dataset|Validation"

echo ""
echo "============================================"
echo "  ALL DONE"
echo "============================================"

# Cleanup test runs
rm -rf /storage/local/maka/test_run_phase_*
echo "Test run directories cleaned up."
