# Multi-Pair Training & Benchmarking Implementation Summary

**Date:** October 16, 2025  
**Implementation Status:** ✅ Complete

## Overview

Successfully implemented a comprehensive multi-pair training and automatic benchmarking system for the pose head experiment. The system now:
1. Extracts ALL consecutive frame pairs from each video sequence (not just one pair)
2. Automatically splits dataset into train/val/test sets
3. Runs automatic evaluation after training
4. Generates detailed performance metrics and visualizations

## What Was Implemented

### 1. Pose Metrics Module (`experiments/pose_metrics.py`)

**New file** containing core evaluation metrics:

- **Rotation Error**: SO(3) geodesic distance via Lie group
  - Formula: `angle = arccos((trace(R_gt^T @ R_pred) - 1) / 2)`
  - Returns error in degrees
  
- **Translation Direction Error**: Angular error between translation vectors
  - Ignores magnitude, only measures direction
  - Formula: `angle = arccos(dot(t_pred_normalized, t_gt_normalized))`
  
- **Utility Functions**:
  - `pose_error()`: Computes both rotation and translation errors
  - `batch_pose_errors()`: Batch processing
  - `compute_error_statistics()`: Mean, median, std, percentiles
  
- **Unit Tests**: Built-in tests verify correctness (all passing ✓)

### 2. Dataset Modifications (`train_pose_head_anycalib.py`)

#### Multi-Pair Extraction

**Before:**
- Dataset returned 1 sample per video (frames 0-1)
- Total samples = number of videos

**After:**
- Dataset returns ALL consecutive frame pairs per video
- For video with N frames: generates pairs (0-1), (2-3), (4-5), ..., ((N-2)-(N-1))
- Example: Video with 100 frames → 50 pairs
- Total samples = sum of all pairs across all videos

**Implementation Details:**
- New `_build_pair_index()` method precomputes pair mappings
- `pair_info` list stores `(video_idx, start_frame)` tuples
- `__getitem__()` maps dataset index to correct video and frame pair
- `_load_frames_from_video()` accepts `start_frame` parameter

**Benefits:**
- Maximizes data utilization
- More training samples without collecting new data
- Each video contributes multiple training examples

#### Train/Val/Test Splitting

**New Functions:**
- `create_train_val_test_split()`: Deterministic splitting with seed
  - Default: 70% train, 15% val, 15% test
  - Returns video indices for each split
  
- `save_dataset_split()`: Saves split to JSON for reproducibility
- `load_dataset_split()`: Loads existing split

**New Dataset Parameters:**
- `video_indices`: Specify which videos to use (for split)
- `extract_all_pairs`: Enable/disable multi-pair extraction
- `require_gt`: Control whether GT is mandatory

**Split File Format** (`experiments/objectron_split.json`):
```json
{
  "train": [0, 3, 5, 7, ...],
  "val": [1, 8, 15, ...],
  "test": [2, 4, 9, ...],
  "num_videos": 100
}
```

### 3. Evaluation Framework

#### Simplified Evaluation (Integrated)

Added to `train_pose_head_anycalib.py`:
- Automatic evaluation after training (opt-in via `--run_evaluation`)
- Computes rotation and translation errors on test set
- Saves results to JSON
- Prints summary statistics

**New Arguments:**
- `--run_evaluation`: Enable automatic evaluation
- `--extract_all_pairs`: Enable multi-pair extraction
- `--split_file`: Path to split file (default: `experiments/objectron_split.json`)
- `--eval_only`: Skip training, only run evaluation

**Evaluation Output:**
```
evaluation/
├── evaluation_results.json    # Detailed statistics
```

**Metrics Computed:**
- Rotation error: mean, median, std, min, max, P90, P95
- Translation direction error: mean, median, std, min, max, P90, P95
- Number of test samples

#### Full Evaluation Script (`experiments/evaluate_pose_model.py`)

**New standalone script** for comprehensive evaluation:
- Compares trained model vs pretrained AnyCam baseline
- Generates comparison plots:
  - Error histograms (overlaid)
  - CDF curves
  - Side-by-side bar charts
- Creates detailed evaluation report
- Saves all results to JSON

**Features:**
- Per-sequence breakdown
- Distribution visualizations
- Improvement percentage calculations
- Automatic plot generation

**Usage:**
```bash
python experiments/evaluate_pose_model.py \
    --our_model_path experiments/pose_head_experiment_results/full_run/final_model.pt \
    --baseline_model_path pretrained_models/anycam_seq8 \
    --split_file experiments/objectron_split.json \
    --save_dir experiments/evaluation_results
```

### 4. Updated Run Script (`experiments/run_experiment.sh`)

**New Modes:**
```bash
# Test run (5 videos, 2 epochs, single pair per video)
bash experiments/run_experiment.sh

# Full run (all videos, 50 epochs, single pair per video)
bash experiments/run_experiment.sh full

# Full run with automatic evaluation
bash experiments/run_experiment.sh full_with_eval

# Multi-pair extraction (10 videos, 10 epochs, all consecutive pairs)
bash experiments/run_experiment.sh multi_pair
```

**Improvements:**
- Clear mode descriptions
- Automatic flag handling
- Helpful output messages
- Lists all generated files

### 5. Visualization Enhancements

**Training Visualizations** (already implemented):
- `loss_curve.png`: Training loss over epochs
- `training_log.txt`: Detailed batch-by-batch log
- `training_summary.txt`: Statistics and progress
- `loss_history.json`: Raw data for analysis

**Evaluation Visualizations** (new):
- `error_histograms.png`: Distribution comparison
- `error_cdfs.png`: Cumulative distribution functions
- `error_comparison_bars.png`: Bar chart comparison
- `evaluation_report.txt`: Text summary

## File Structure

```
experiments/
├── pose_metrics.py                    # NEW: Evaluation metrics
├── evaluate_pose_model.py             # NEW: Full evaluation script
├── train_pose_head_anycalib.py        # MODIFIED: Multi-pair + eval integration
├── run_experiment.sh                  # MODIFIED: New modes
├── objectron_split.json               # NEW: Generated on first run
├── MULTI_PAIR_IMPLEMENTATION_SUMMARY.md  # This file
└── pose_head_experiment_results/
    ├── test_run/                      # Test experiments
    ├── full_run/                      # Full training runs
    ├── multi_pair_run/                # Multi-pair experiments
    └── full_run_eval/                 # With evaluation
        ├── checkpoints/
        ├── loss_curve.png
        ├── training_log.txt
        ├── training_summary.txt
        ├── loss_history.json
        └── evaluation/                # NEW: Evaluation results
            ├── evaluation_results.json
            └── (plots if using full script)
```

## Usage Examples

### Example 1: Training with Multi-Pair Extraction

```bash
python experiments/train_pose_head_anycalib.py \
    --extract_all_pairs \
    --num_epochs 50 \
    --batch_size 4 \
    --save_dir experiments/pose_head_experiment_results/multi_pair
```

**Expected Behavior:**
- Creates/loads split file
- Extracts all consecutive pairs from training videos
- Trains on significantly more samples
- Saves visualizations and logs

### Example 2: Training with Automatic Evaluation

```bash
python experiments/train_pose_head_anycalib.py \
    --extract_all_pairs \
    --run_evaluation \
    --num_epochs 50 \
    --save_dir experiments/pose_head_experiment_results/with_eval
```

**Expected Behavior:**
- Trains model on training set
- Automatically evaluates on test set (requires GT)
- Computes rotation and translation errors
- Saves metrics to `evaluation/evaluation_results.json`

### Example 3: Evaluation Only

```bash
python experiments/train_pose_head_anycalib.py \
    --eval_only \
    --run_evaluation \
    --save_dir experiments/pose_head_experiment_results/full_run
```

**Expected Behavior:**
- Loads trained model from `save_dir/final_model.pt`
- Runs evaluation on test set
- No training performed

## Key Technical Details

### Multi-Pair Indexing

```python
# In __init__
self.pair_info = []
for video_idx, video_path in enumerate(self.video_files):
    total_frames = get_frame_count(video_path)
    n_pairs = total_frames // 2  # For num_frames=2
    for pair_idx in range(n_pairs):
        start_frame = pair_idx * 2
        self.pair_info.append((video_idx, start_frame))

# In __getitem__
def __getitem__(self, idx):
    video_idx, start_frame = self.pair_info[idx]
    video_path = self.video_files[video_idx]
    frames = load_frames(video_path, start_frame, self.num_frames)
    ...
```

### Relative Pose Computation

For evaluation, we compare predicted relative poses to GT relative poses:

```python
# GT: Absolute poses (camera-to-world)
gt_pose1 = poses[t]    # Camera pose at frame t
gt_pose2 = poses[t+1]  # Camera pose at frame t+1

# Compute relative transformation
gt_rel_pose = inv(gt_pose2) @ gt_pose1

# Compare with predicted relative pose
rot_error = rotation_error_degrees(pred_rel_pose[:3,:3], gt_rel_pose[:3,:3])
trans_error = translation_direction_error_degrees(pred_rel_pose[:3,3], gt_rel_pose[:3,3])
```

### Error Metrics Formulas

**Rotation Error (Lie Group Distance):**
```python
R_diff = R_gt.T @ R_pred
trace = np.trace(R_diff)
angle_rad = arccos(clip((trace - 1) / 2, -1, 1))
angle_deg = degrees(angle_rad)
```

**Translation Direction Error:**
```python
t_pred_norm = t_pred / norm(t_pred)
t_gt_norm = t_gt / norm(t_gt)
cos_angle = dot(t_pred_norm, t_gt_norm)
angle_rad = arccos(clip(cos_angle, -1, 1))
angle_deg = degrees(angle_rad)
```

## Testing

### Unit Tests

```bash
# Test pose metrics
python experiments/pose_metrics.py
# Output: ✓ All tests passed!
```

### Integration Tests

```bash
# Test dataset splitting
python -c "from experiments.train_pose_head_anycalib import create_train_val_test_split; ..."

# Test multi-pair extraction (dry run)
python experiments/train_pose_head_anycalib.py \
    --max_sequences 2 \
    --num_epochs 1 \
    --extract_all_pairs \
    --save_dir /tmp/test
```

## Next Steps

With this implementation complete, you can now:

1. **Run Extended Training**: Use multi-pair extraction for more data
   ```bash
   bash experiments/run_experiment.sh multi_pair
   ```

2. **Evaluate Performance**: Automatically measure accuracy
   ```bash
   bash experiments/run_experiment.sh full_with_eval
   ```

3. **Compare to Baseline**: Use `evaluate_pose_model.py` for detailed comparison
   ```bash
   python experiments/evaluate_pose_model.py --our_model_path ... --baseline_model_path ...
   ```

4. **Proceed to Experiment 2**: Multi-frame reprojection (1->2, 1->3, 1->4, etc.)
   - This will require modifying the pose head to predict multiple frames
   - Flow stacking for long-range flow computation
   - Extended loss function with multiple reprojection terms

## Notes

- **Unsupervised Training**: GT not needed for training (uses flow reprojection loss)
- **GT for Evaluation**: GT required only for evaluation to measure accuracy
- **Reproducibility**: Dataset split is deterministic and saved to JSON
- **Backward Compatible**: Can still use single-pair mode by omitting `--extract_all_pairs`

## Files Created

1. ✅ `experiments/pose_metrics.py` - Evaluation metrics
2. ✅ `experiments/evaluate_pose_model.py` - Full evaluation script
3. ✅ `experiments/MULTI_PAIR_IMPLEMENTATION_SUMMARY.md` - This document

## Files Modified

1. ✅ `experiments/train_pose_head_anycalib.py` - Multi-pair dataset + evaluation
2. ✅ `experiments/run_experiment.sh` - New run modes

## All TODOs Completed

- ✅ Modify ObjectronVideoDataset for multi-pair extraction
- ✅ Implement train/val/test split with deterministic seeding
- ✅ Create pose_metrics.py with rotation and translation error functions
- ✅ Create evaluate_pose_model.py for model comparison
- ✅ Generate error distribution plots
- ✅ Integrate evaluation into train_pose_head_anycalib.py
- ✅ Test complete pipeline

---

**Implementation Complete! Ready for extended training and evaluation experiments.** 🎉

