# Cycle Consistency Testing Script

This directory contains a clean Python script for running cycle consistency experiments on videos or image sequences using AnyCam.

## Overview

The script performs the following steps:
1. **Extract frames** from a source video or image folder (configurable selection parameters)
2. **Create pair videos** (forward and backward pairs for true cycle consistency)
3. **Run AnyCam inference** on all pairs using the transformer model
4. **Evaluate cycle consistency** using proper mathematical formulation
5. **Save isolated results** in unique experiment directories

## Adaptive Data Loading

The script now supports **two input modes**:

### 📹 Video Input (MP4, AVI, MOV, MKV)
- Loads frames directly from video files
- Supports frame selection with start position and skipping
- Handles various video formats

### 📁 Image Folder Input (JPEG, PNG, etc.)
- Loads image sequences from directories
- Perfect for datasets like DAVIS 2017, TUM-RGBD, etc.
- Maintains temporal order through filename sorting
- Supports multiple image formats (JPG, JPEG, PNG, BMP, TIFF)

## Usage

### Basic Usage

```bash
# Video input with default settings
python experiments/cycle-consistency/run_cycle_consistency_test.py --input /path/to/video.mp4

# Image folder input
python experiments/cycle-consistency/run_cycle_consistency_test.py --input /path/to/DAVIS/blackswan/

# Custom frame extraction
python experiments/cycle-consistency/run_cycle_consistency_test.py --input /path/to/data --frames 5 --start-frame 10 --skip-frames 2
```

### Advanced Frame Selection

```bash
# Extract 4 frames starting from frame 20, taking every 3rd frame
python experiments/cycle-consistency/run_cycle_consistency_test.py \
    --input /path/to/sequence \
    --frames 4 \
    --start-frame 20 \
    --skip-frames 3

# This extracts frames: [20, 23, 26, 29]
```

### Command Line Options

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--input` | `-i` | Path to input video file or image folder | **Required** |
| `--frames` | `-f` | Number of frames to extract | `3` |
| `--start-frame` | `-s` | Starting frame index (0-based) | `0` |
| `--skip-frames` | `-k` | Frame skip interval (1=consecutive, 2=every other frame) | `1` |
| `--name` | `-n` | Custom experiment name | Auto-generated |
| `--model` | `-m` | Path to AnyCam model | `pretrained_models/anycam_seq8` |
| `--no-recompute` | | Skip recomputation if results exist | `False` |

### Dataset Examples

```bash
# DAVIS 2017 dataset
python experiments/cycle-consistency/run_cycle_consistency_test.py \
    --input /path/to/DAVIS/JPEGImages/480p/blackswan/ \
    --frames 4 --name davis_blackswan

# TUM-RGBD dataset (RGB images)
python experiments/cycle-consistency/run_cycle_consistency_test.py \
    --input /path/to/tum_rgbd/freiburg1_xyz/rgb/ \
    --frames 5 --start-frame 100 --skip-frames 5 \
    --name tum_freiburg1

# Your own video with temporal sampling
python experiments/cycle-consistency/run_cycle_consistency_test.py \
    --input /home/kalman/Videos/anycam-tests/three_frames.mp4 \
    --frames 3 --start-frame 0 --skip-frames 1

# Custom sequence with large temporal gaps
python experiments/cycle-consistency/run_cycle_consistency_test.py \
    --input ./long_video.mp4 \
    --frames 3 --start-frame 50 --skip-frames 10 \
    --name temporal_gaps
```

## Input Validation

The script automatically:
- **Detects input type** (video file vs image folder)
- **Validates frame availability** before processing
- **Calculates frame ranges** with skipping parameters
- **Handles various image formats** in folders
- **Sorts images by filename** to maintain sequence order

### Frame Range Calculation
```
For skip_frames=k, num_frames=n, start_frame=s:
Extracted frames: [s, s+k, s+2k, ..., s+(n-1)k]
Last frame needed: s + (n-1) * k
```

## Output Structure

Each experiment creates an isolated directory structure:

```
experiments/cycle-consistency/
└── {name}_f{frames}_s{start}_k{skip}_{timestamp}_{hash}/
    ├── videos/                      # Generated pair MP4s
    │   ├── pair01.mp4              # Forward pairs: 0→1, 1→2, 0→2, etc.
    │   ├── pair10.mp4              # Backward pairs: 1→0, 2→1, 2→0, etc.
    │   └── ...
    ├── results/                     # AnyCam inference outputs
    │   ├── pair01/
    │   │   ├── trajectory.npy
    │   │   ├── projection.npy
    │   │   └── ...
    │   └── ...
    ├── {experiment_name}_report.md  # Human-readable results
    └── {experiment_name}_data.json  # Raw numerical data
```

### Enhanced Experiment Names
Experiments now include extraction parameters in the name:
- `blackswan_f4_s10_k2_1214_1432_a1b2c3d4` = 4 frames, start=10, skip=2
- `video_f3_s0_k1_1214_1445_e5f6g7h8` = 3 frames, start=0, skip=1 (consecutive)

## Key Features

### 🔄 True Cycle Consistency
- Creates **both forward and backward pairs** for proper loop closure
- Computes **P₁ @ P₂ @ P₃ ≈ I** instead of simple inverse checks
- Handles **variable frame counts** (3, 4, 5+ frames)

### 📊 Flexible Frame Selection
- **Configurable start position** for focusing on specific sequences
- **Frame skipping** for temporal sampling (motion analysis)
- **Range validation** prevents out-of-bounds errors
- **Multiple input formats** (video files + image folders)

### 🧪 Isolated Experiments  
- **Unique experiment names** prevent conflicts
- **Self-contained directories** for each test
- **Timestamped results** for tracking
- **Parameter encoding** in directory names

### 📈 Comprehensive Analysis
- **Composition errors**: P₁₃_direct vs P₁₂ @ P₂₃
- **True cycle errors**: Full loop closure validation
- **Rotation/translation decomposition**
- **Automatic error interpretation**

### 🎯 Research Applications
- **Temporal sampling studies**: Effects of frame skipping
- **Dataset comparison**: Standardized evaluation across datasets
- **Motion analysis**: Higher skip rates for faster motion
- **Sequence optimization**: Finding optimal frame selection

## Expected Behavior by Input Type

### Video Files
- **Automatic frame extraction** from compressed video
- **Precise frame indexing** using OpenCV
- **Frame rate independent** (uses frame indices, not time)
- **Memory efficient** (loads only required frames)

### Image Folders
- **Alphabetical sorting** maintains temporal order
- **Multi-format support** (JPEG, PNG, BMP, TIFF)
- **Direct loading** without video decoding overhead  
- **Perfect for research datasets** (DAVIS, TUM-RGBD, etc.)

### Frame Skip Effects
- **skip_frames=1**: Consecutive frames (default)
- **skip_frames=2**: Every other frame (2x temporal spacing)
- **skip_frames=5**: Every 5th frame (5x temporal spacing)
- **Higher skips**: Better for fast motion, larger baselines

### Error Interpretation
- **Low errors (~0.001-0.01)**: Good consistency, try larger skips
- **Moderate errors (~0.01-0.1)**: Good for analysis  
- **High errors (~0.1+)**: Perfect for optimization experiments
- **Very high errors**: May indicate challenging motion or insufficient frames

## Troubleshooting

### Input Issues
```bash
# Check video properties
ffprobe -v quiet -print_format json -show_streams video.mp4

# Count images in folder
ls -1 /path/to/images/*.jpg | wc -l

# Verify frame range
python -c "
start, frames, skip = 10, 3, 5
end = start + (frames-1)*skip
print(f'Frames needed: {start} to {end}')
"
```

### Frame Range Errors
```bash
# Error: "Not enough frames"
# Solution: Check total frames vs requested range
# Formula: start_frame + (num_frames-1) * skip_frames < total_frames

# Example fix: Reduce frames or skip rate
python run_cycle_consistency_test.py --input data/ --frames 3 --start-frame 10 --skip-frames 2
```

### Dataset-Specific Tips

**DAVIS 2017:**
```bash
# DAVIS has ~70-80 frames per sequence
python run_cycle_consistency_test.py \
    --input /path/to/DAVIS/JPEGImages/480p/sequence_name/ \
    --frames 4 --start-frame 10 --skip-frames 5
```

**TUM-RGBD:**
```bash
# TUM-RGBD has 500-3000+ frames
python run_cycle_consistency_test.py \
    --input /path/to/tum/freiburg1_xyz/rgb/ \
    --frames 5 --start-frame 100 --skip-frames 10
```

**Custom Videos:**
```bash
# Check frame count first
ffprobe -v quiet -select_streams v:0 -count_frames -show_entries stream=nb_frames video.mp4

# Then set appropriate parameters
python run_cycle_consistency_test.py \
    --input video.mp4 \
    --frames 4 --start-frame 0 --skip-frames 3
```

## Performance Considerations

### Memory Usage
- **Image folders**: Higher memory usage (all frames loaded at once)
- **Video files**: Lower memory usage (sequential frame access)
- **Frame skipping**: Reduces memory requirements proportionally

### Processing Speed
- **Sequential frames** (skip=1): Fastest processing
- **Skipped frames** (skip>1): Slightly slower due to seeking
- **Large skips**: Better cycle consistency errors (more baseline)

### Storage Requirements
- **Each experiment**: ~10-100MB depending on frame count and resolution
- **Pair videos**: Minimal storage (2 frames each)
- **AnyCam results**: Trajectory and projection matrices (~1MB per pair)

## Future Extensions

The adaptive data loading enables:

- **Batch processing** multiple sequences
- **Temporal sampling studies** across different skip rates
- **Dataset standardization** with common extraction parameters
- **Motion-adaptive selection** based on optical flow
- **Multi-scale analysis** with different temporal baselines

## Integration with Thesis Work

This enhanced script provides:

1. **Standardized evaluation** across video and image datasets
2. **Flexible temporal sampling** for motion analysis studies
3. **Dataset compatibility** with DAVIS, TUM-RGBD, and custom sequences
4. **Parameter exploration** for optimal frame selection strategies
5. **Research reproducibility** with isolated, timestamped experiments

The adaptive data loading makes it easy to compare cycle consistency across different datasets and temporal sampling strategies, enabling comprehensive analysis for your thesis work. 