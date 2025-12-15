# Multi-Frame Benchmark Evaluation

## Overview

This experiment evaluates the `look_ahead_3.pt` model across different frame counts (2, 3, 4, 5, 6, 7, 8 frames) on the LightSpeed dataset. The goal is to assess how well the model generalizes to different sequence lengths and compare its performance against the AnyCam baseline.

## Experiment Purpose

The `look_ahead_3.pt` model was trained with `max_ahead=3`, meaning it was optimized for sequences of 4 frames (0, 1, 2, 3). This experiment evaluates:

1. **Generalization**: How well does the model perform on sequences longer or shorter than its training configuration?
2. **Frame count sensitivity**: Does performance degrade as the number of frames increases?
3. **Comparison with baseline**: How does the retrained model compare to AnyCam baseline across different frame counts?

## Methodology

### Dataset

- **Dataset**: LightSpeed validation set
- **Sequence extraction**: For each video, extract all non-overlapping sequences of the specified length
  - For `num_frames=5`: Extract sequences [1-5], [6-10], [11-15], etc.
  - Skip leftover frames if insufficient for a complete sequence
- **Evaluation**: Each frame count is evaluated separately

### Models

1. **look_ahead_3.pt**: Experiment 2 model with `max_ahead=3`, trained with AnyCaLib focal length injection
2. **AnyCam Baseline**: Pretrained AnyCam model (training_checkpoint_247500.pt)

### Multi-Frame Processing

Both models process all N frames at once in a single forward pass:
- Input: `images` [B, N, 3, H, W]
- Pose predictor processes all N frames jointly through the transformer backbone
- Output: `pose_result["poses"]` [B, N-1, 4, 4] containing relative poses for consecutive pairs (0→1, 1→2, ..., (N-2)→(N-1))

**Important**: Both models use the same multi-frame processing approach. The AnyCam baseline does NOT use bundle adjustment post-processing (raw neural network output only for fair comparison).

### Metrics

For each predicted relative pose, we compute three error metrics:

1. **Rotation Error** (degrees): SO(3) geodesic distance between predicted and ground truth rotation matrices
2. **Translation Direction Error** (degrees): Angular error between predicted and ground truth translation direction vectors
3. **Translation Magnitude Error**: Euclidean distance between predicted and ground truth translation vectors

Errors are computed for all relative pose pairs in each sequence and aggregated across the dataset.

### Trajectory Visualization

For visualization purposes, relative poses are accumulated to build full trajectories:
- Start with identity pose at frame 0
- For each consecutive pair (i, i+1): `traj[i+1] = traj[i] @ inv(rel_pose[i->i+1])`
- 3D trajectory paths are plotted for qualitative comparison

## Running the Experiment

### Single Frame Count

```bash
python experiments/benchmark_against_anycam.py \
    --exp2_model experiments/pose_head_experiment_results/look_ahead_3.pt \
    --baseline_checkpoint pretrained_models/anycam_seq8/training_checkpoint_247500.pt \
    --dataset lightspeed \
    --num_frames 5 \
    --save_dir experiments/pose_head_experiment_results/test-diff-num-frames/frames_5
```

### All Frame Counts (Automated)

```bash
python experiments/run_multi_frame_benchmark.py \
    --model_path experiments/pose_head_experiment_results/look_ahead_3.pt \
    --baseline_checkpoint pretrained_models/anycam_seq8/training_checkpoint_247500.pt \
    --lightspeed_dir /data/thesis/LightSpeed
```

## Output Structure

```
test-diff-num-frames/
├── frames_2/
│   ├── benchmark_results.json
│   ├── benchmark_report.txt
│   ├── benchmark_comparison.png
│   └── trajectory_comparison.png
├── frames_3/
│   └── ...
├── frames_4/
│   └── ...
├── frames_5/
│   └── ...
├── frames_6/
│   └── ...
├── frames_7/
│   └── ...
└── frames_8/
    └── ...
```

### Output Files

- **benchmark_results.json**: Detailed metrics for all models (rotation, translation direction, translation magnitude)
- **benchmark_report.txt**: Human-readable text report with statistics
- **benchmark_comparison.png**: Comparison plots (histograms, CDFs, box plots) for all error types
- **trajectory_comparison.png**: 3D trajectory visualization plots

## Results Interpretation

### Key Questions

1. **Does the model generalize to different frame counts?**
   - Compare error metrics across frame counts
   - Look for degradation as frame count increases

2. **Is there an optimal frame count?**
   - Check if performance peaks at a specific frame count
   - Note: Model was trained with max_ahead=3 (4 frames)

3. **How does it compare to baseline?**
   - Compare error metrics at each frame count
   - Check if improvements are consistent across frame counts

### Expected Behavior

- **Frame count 2-4**: Should perform well (within training range)
- **Frame count 5-8**: May show degradation due to:
  - Flow accumulation errors (longer sequences)
  - Model not trained for these lengths
  - Increased complexity of pose composition

### Metrics to Focus On

- **Rotation Error**: Most important for camera pose estimation
- **Translation Direction Error**: Important for trajectory direction
- **Translation Magnitude Error**: Important for trajectory scale

## Notes

- Bundle adjustment is **disabled** for AnyCam baseline to ensure fair comparison (raw neural network output only)
- Both models process all frames at once (not pair-by-pair)
- Ground truth contains per-frame poses, allowing extraction of relative poses for any frame count
- Trajectory visualization is for qualitative assessment only (not used in quantitative metrics)

