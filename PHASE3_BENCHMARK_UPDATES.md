# Phase 3 Benchmark Updates - Comprehensive Pose Metrics

## Changes Made

### 1. Enhanced Pose Metrics (4 comprehensive metrics)

**Previously**: Only 2 metrics
- Rotation error (degrees)
- Translation direction error (degrees)

**Now**: 4 comprehensive metrics
1. **SE(3) Distance**: Frobenius norm ||T_pred - T_gt||_F
   - Measures overall pose difference in a single value
   - Accounts for both rotation and translation

2. **Rotation Error**: Geodesic distance on SO(3)
   - Angular error in degrees
   - Standard metric for rotation accuracy

3. **Translation Magnitude**: Euclidean distance ||t_pred - t_gt||
   - Absolute difference in translation vectors
   - Units depend on scene scale

4. **Translation Direction Error**: Angular error between directions
   - Direction-only comparison (scale-invariant)
   - Degrees between unit vectors

### 2. Updated Visualizations

**New Plot Layout**: 3×2 grid (6 subplots)

**Row 1: Pose Metrics (Distances)**
- **Left**: SE(3) distance (purple)
- **Right**: Rotation error (blue)

**Row 2: Pose Metrics (Translation)**
- **Left**: Translation magnitude (green)
- **Right**: Translation direction error (red)

**Row 3: Calibration Metrics**
- **Left**: Focal MAPE mean - FAT vs AnyCalib
- **Right**: Focal MAPE median - FAT vs AnyCalib

**Removed from plots**: GT variance (now only in JSON and terminal output)

### 3. Enhanced Terminal Output

**Per-Epoch Summary**:
```
[RESULTS] Epoch 1 Summary:
  Pose Metrics:
    SE(3) distance:         0.1234 (mean), 0.0987 (median)
    Rotation error:         1.2345° (mean), 0.9876° (median)
    Translation magnitude:  0.0234 (mean), 0.0198 (median)
    Translation direction:  2.3456° (mean), 1.8765° (median)
  Calibration Metrics:
    FAT Focal MAPE:              4.32% (mean), 3.21% (median)
    AnyCalib Focal MAPE:         2.98% (mean), 2.45% (median)
  GT Variance (reference only):
    Focal std: 12.34 (fx), 12.56 (fy)
```

**Final Summary Table**:
```
========================================================================================================================
[SUMMARY] Benchmark Results Across Epochs (Median Values)
========================================================================================================================
Epoch    SE(3)      Rot(°)     TransMag    TransDir(°)  FAT MAPE(%)      AnyCalib MAPE(%)
------------------------------------------------------------------------------------------------------------------------
1        0.1234     0.9876     0.0198      1.8765       3.21             2.45
2        0.1123     0.8765     0.0187      1.7654       2.98             2.34
3        0.1012     0.7654     0.0176      1.6543       2.76             2.23
========================================================================================================================
```

### 4. Updated pose_metrics.py

Added new function:
```python
def se3_distance(pose_pred: np.ndarray, pose_gt: np.ndarray) -> float:
    """
    Compute SE(3) distance using Frobenius norm.

    Returns:
        ||T_pred - T_gt||_F
    """
    return np.linalg.norm(pose_pred - pose_gt, ord='fro')
```

Updated `pose_error()` function signature:
```python
# Old
def pose_error(pose_pred, pose_gt) -> Tuple[float, float]:
    return rot_err, trans_err

# New
def pose_error(pose_pred, pose_gt) -> Tuple[float, float, float, float]:
    return se3_dist, rot_err, trans_mag, trans_dir
```

## How to Use

### Running the Benchmark

Same command as before:
```bash
cd /workspace
./run_phase3_benchmark.sh
```

### Output Files

**benchmark_results.json**: Now includes all 4 pose metrics
```json
{
  "epoch": 1,
  "pose_metrics": {
    "se3_distance_mean": 0.1234,
    "se3_distance_median": 0.0987,
    "rotation_deg_mean": 1.2345,
    "rotation_deg_median": 0.9876,
    "translation_magnitude_mean": 0.0234,
    "translation_magnitude_median": 0.0198,
    "translation_direction_deg_mean": 2.3456,
    "translation_direction_deg_median": 1.8765
  },
  "fat_calibration": { ... },
  "anycalib_calibration": { ... },
  "gt_variance": { ... }
}
```

**benchmark_across_epochs.png**: 3×2 grid with 6 subplots

## Interpreting the New Metrics

### SE(3) Distance
- **What**: Single scalar capturing overall pose error
- **Lower is better**: 0 = perfect match
- **Typical values**: 0.05-0.3 for good pose estimation
- **Use case**: Quick overview of pose accuracy

### Rotation Error
- **What**: Angular difference between rotation matrices
- **Lower is better**: 0° = perfect rotation
- **Typical values**: <1° excellent, 1-2° good, >5° poor
- **Use case**: Primary metric for rotation accuracy

### Translation Magnitude
- **What**: Euclidean distance between translation vectors
- **Lower is better**: 0 = perfect translation match
- **Typical values**: Depends on scene scale (0.01-0.1 common)
- **Use case**: Absolute translation accuracy
- **Note**: Scale-dependent, less interpretable than direction error

### Translation Direction Error
- **What**: Angular error between translation directions
- **Lower is better**: 0° = perfect direction
- **Typical values**: <2° excellent, 2-5° good, >10° poor
- **Use case**: Scale-invariant translation accuracy
- **Note**: Most interpretable translation metric for monocular vision

## GT Variance Location

GT variance is **not plotted** but remains available in:
1. **Terminal output**: Per-epoch summary (reference section)
2. **JSON output**: `results['gt_variance']` field

This declutters the plots while preserving the data for analysis.

## Backward Compatibility

**Breaking changes**: None for end users

The benchmark script signature and command-line interface remain identical. Only the output format has been enhanced.

## Files Modified

1. `experiments/pose_metrics.py`
   - Added `se3_distance()` function
   - Updated `pose_error()` and `batch_pose_errors()` signatures

2. `experiments/benchmark_phase3_checkpoints.py`
   - Updated pose error computation (line ~270)
   - Updated metrics aggregation (line ~305)
   - Updated terminal output (line ~327)
   - Updated plotting (line ~341)
   - Updated summary table (line ~500)

3. This document: `PHASE3_BENCHMARK_UPDATES.md`

## Next Steps

1. Run the updated benchmark:
   ```bash
   cd /workspace
   ./run_phase3_benchmark.sh
   ```

2. Check the new 3×2 plot: `benchmark_across_epochs.png`

3. Analyze all 4 pose metrics to get comprehensive view

4. Use SE(3) distance for quick sanity check

5. Use rotation error + translation direction for detailed analysis

6. Compare translation magnitude across epochs to see if scale consistency improves
