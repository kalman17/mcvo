# Phase 3 V2 Checkpoint Benchmarking Guide

## Overview

This guide explains how to benchmark all saved Phase 3 checkpoints to evaluate:
1. **Pose accuracy**: Rotation and translation errors vs ground truth
2. **Calibration accuracy (3 comparisons)**:
   - FAT model prediction vs GT mean
   - GT mean variance (intrinsic variability across frames)
   - AnyCalib per-frame average vs GT mean

## Quick Start

From Docker container:
```bash
cd /workspace
./run_phase3_benchmark.sh
```

## What the Benchmark Does

### 1. Checkpoint Discovery
- Scans `experiments/fat_integration/phase3_training_v2/checkpoints/`
- Finds all `checkpoint_epoch_*.pt` files
- Excludes `latest_checkpoint.pt` (duplicate of last epoch)

### 2. For Each Checkpoint
Evaluates on 50 test sequences (adjustable with `--num_samples`):

#### Pose Evaluation
- Loads predicted relative poses between consecutive frames
- Compares against GT relative poses computed from absolute GT poses
- Metrics: Rotation error (degrees), Translation direction error (degrees)

#### Calibration Evaluation
For each sequence (4 frames with max_ahead=3):

**a) FAT Model Prediction**
- Single prediction for entire sequence: [fx, fy, cx, cy]
- Compared against GT mean over 4 frames
- Metrics: MAE, MAPE (%)

**b) GT Variance (Baseline)**
- Computes standard deviation of GT intrinsics across 4 frames
- Shows inherent variability in ground truth
- Useful for understanding lower bound of error

**c) AnyCalib Per-Frame Average**
- Runs AnyCalib on each of 4 frames individually
- Averages predictions: [fx, fy, cx, cy]
- Compared against GT mean
- Metrics: MAE, MAPE (%)

### 3. Output

#### JSON Results
`experiments/fat_integration/phase3_training_v2/benchmark_results/benchmark_results.json`
- Complete results for all epochs
- Raw metrics for programmatic analysis

#### Plots
`experiments/fat_integration/phase3_training_v2/benchmark_results/benchmark_across_epochs.png`
- 4-panel plot showing:
  - Rotation error vs epoch
  - Translation error vs epoch
  - FAT focal length MAPE vs epoch (mean)
  - FAT focal length MAPE vs epoch (median)
- Each includes comparison with AnyCalib baseline

#### Terminal Summary
```
==========================================================================================================
[SUMMARY] Benchmark Results Across Epochs
==========================================================================================================
Epoch    Rot(°)       Trans(°)     FAT MAPE(%)      AnyCalib MAPE(%)
----------------------------------------------------------------------------------------------------------
1        1.2345       2.3456       5.67             3.45
2        1.1234       2.2345       4.56             3.34
3        1.0123       2.1234       3.45             3.23
==========================================================================================================
```

## Ground Truth Format

GT files located in: `/data/thesis/Objectron/processed_gt/` (Docker) or `/home/kalmanm/git/masters/Objectron/processed_gt/` (host)

### File Naming
- Videos: `batch-X_Y_video.MOV`
- GT: `batch-X_Y.json`

### GT JSON Structure
```json
{
  "poses": [
    [16 floats representing 4x4 matrix],  // Frame 0
    [16 floats representing 4x4 matrix],  // Frame 1
    ...
  ],
  "intrinsics_per_frame": [
    [fx, 0, cx, 0, fy, cy, 0, 0, 1],  // Frame 0 (3x3 flattened)
    [fx, 0, cx, 0, fy, cy, 0, 0, 1],  // Frame 1
    ...
  ]
}
```

## Advanced Usage

### Custom Number of Samples
```bash
python experiments/benchmark_phase3_checkpoints.py \
    --checkpoint_dir experiments/fat_integration/phase3_training_v2/checkpoints \
    --num_samples 100 \
    --objectron_videos /data/thesis/Objectron/videos \
    --objectron_gt /data/thesis/Objectron/processed_gt \
    --anycam_config pretrained_models/anycam_seq8/training_config.yaml \
    --output_dir experiments/fat_integration/phase3_training_v2/benchmark_results
```

### Benchmark Specific Checkpoint
```bash
python experiments/benchmark_phase3_checkpoints.py \
    --checkpoint_dir experiments/fat_integration/phase3_training_v2/checkpoints \
    --num_samples 50 \
    --objectron_videos /data/thesis/Objectron/videos \
    --objectron_gt /data/thesis/Objectron/processed_gt \
    --anycam_config pretrained_models/anycam_seq8/training_config.yaml \
    --output_dir experiments/fat_integration/phase3_training_v2/benchmark_epoch1
```
(Then manually select only epoch 1 checkpoint)

## Interpreting Results

### Pose Metrics
- **Rotation error**: Angular error on SO(3). Lower is better. <1° is excellent, <2° is good.
- **Translation error**: Angular error between translation directions. Scale-invariant due to monocular ambiguity.

### Calibration Metrics
- **MAE**: Mean Absolute Error in pixels (focal length) or pixels (principal point)
- **MAPE**: Mean Absolute Percentage Error. More interpretable for focal length (varies across cameras).
  - <5%: Excellent
  - 5-10%: Good
  - >10%: Needs improvement

### Expected Trends
- Rotation error should decrease with training
- FAT MAPE should approach AnyCalib MAPE (AnyCalib is pseudo-GT used in Phase 1/2)
- GT variance shows inherent noise/variability in ground truth (1-5 pixels typical)

## Troubleshooting

### "No checkpoint files found"
- Ensure checkpoints exist in `experiments/fat_integration/phase3_training_v2/checkpoints/`
- Check that files are named `checkpoint_epoch_1.pt`, `checkpoint_epoch_2.pt`, etc.

### "Cannot load GT file"
- Verify GT directory path matches environment (Docker vs host)
- Check that GT files exist: `ls /data/thesis/Objectron/processed_gt/` (Docker)

### CUDA Out of Memory
- Reduce `--num_samples` (e.g., from 50 to 25)
- Run benchmarks sequentially (one epoch at a time)

### Pose errors are very high
- Check that GT poses are in the correct format (4x4 matrices)
- Verify relative pose computation (T_rel = T_i^(-1) @ T_{i+1})

## Files Generated

```
experiments/fat_integration/phase3_training_v2/benchmark_results/
├── benchmark_results.json          # Complete results
├── benchmark_across_epochs.png     # 4-panel plot
└── benchmark_log.txt               # Detailed log (if generated)
```

## Integration with Training

The benchmark script is designed to run **after** training completes, not during training. The training script's built-in benchmarking was disabled due to GT format issues (now fixed).

To run after training:
1. Wait for training to complete (or pause after desired epochs)
2. Run `./run_phase3_benchmark.sh`
3. Analyze results in `benchmark_results/`

## Next Steps

After benchmarking:
1. Analyze trends in `benchmark_across_epochs.png`
2. Compare FAT vs AnyCalib performance
3. Identify best checkpoint (lowest rotation error + lowest calibration MAPE)
4. Use best checkpoint for final thesis results
