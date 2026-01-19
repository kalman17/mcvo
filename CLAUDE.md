# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**AnyCam Extension** - Master's thesis project extending AnyCam (CVPR 2025) with AnyCalib integration and multi-frame consistency improvements for camera pose and intrinsics estimation from casual videos.

**Author**: Kalman Mahlich
**Institution**: Technical University of Munich (TUM)
**Thesis**: "Improving Camera Calibration and Pose Estimation in AnyCam through Integration of AnyCalib and Multi-Frame Consistency"
**Submission**: March 31, 2025

**Core Contribution**: Integration of AnyCalib focal length predictions to replace AnyCam's expensive 32-candidate system, combined with multi-frame consistency training using composed flow losses.

## Repository Structure

```
anycam-extension/
├── anycam/                    # Core AnyCam codebase (CVPR 2025, unchanged)
├── anycalib/                  # AnyCalib submodule for calibration
├── experiments/               # All thesis extension work (primary workspace)
│   ├── train_pose_head_anycalib.py         # Experiment 1: Single pose head
│   ├── train_pose_head_anycalib_exp2.py    # Experiment 2: Multi-frame consistency
│   ├── train_calibration_head_da3_stage*.py # DA3 calibration head training
│   ├── benchmark_*.py                       # Evaluation scripts
│   ├── models/                              # DA3 calibration head components
│   ├── da3_integration/                     # DA3 training results
│   └── pose_head_experiment_results/        # Experiment 1 & 2 results
├── minipytorch3d/            # Minimal PyTorch3D variant (by VGGSfM)
├── unimatch/                  # Customized UniMatch fork for optical flow
└── pretrained_models/         # Downloaded model checkpoints
```

## Key Architecture Concepts

### AnyCam (Base Model)
- **Self-supervised** framework for camera pose and intrinsics recovery
- Uses frozen depth predictor (UniDepth) and optical flow (UniMatch)
- Original focal length system: 32 candidates tested via flow reprojection
- Two prediction heads: `pose_head` (rotation + translation) and `sequence_info_head` (focal length)
- Training loss: Flow reprojection loss (comparing predicted vs observed optical flow)
- **Location**: `anycam/models/anycam.py`, `anycam/models/anycam_blocks.py`

### AnyCalib Integration (Experiments 1 & 2)
- **Replaces** 32-candidate system with direct AnyCalib focal length predictions
- **Wrapper class**: `AnyCamWrapperWithAnyCaLib` injects AnyCalib predictions into calibration matrix
- **Training**: Freezes all components except pose head, trains on flow reprojection loss
- **Experiment 1**: Single pose head with 2-frame pairs
- **Experiment 2**: Multi-frame consistency with composed flow (max_ahead=3 optimal)
- **Results**: Experiment 2 achieves best rotation error (1.23° mean, 0.92° median)

### DA3 Calibration Head (Current Work)
- **Inspired by** Depth Anything 3's camera conditioning architecture
- **Components**: Camera encoder → Visual-camera mixing → Sequence aggregation → Camera decoder
- **Three training stages**:
  - Stage 1: Learn mean calibration without visual tokens (MSE loss vs GT mean)
  - Stage 2: Add visual conditioning with DINOv2-small (vis_dim=384)
  - Stage 3: End-to-end training with flow reprojection loss (self-supervised)
- **Stage 3.1**: Multi-frame variants with max_ahead=3/4 and optional alternating training
- **Location**: `experiments/models/`, training scripts in `experiments/train_calibration_head_da3_stage*.py`

## Common Commands

### Environment Setup

```bash
# Create conda environment with Python 3.11
conda create -n anycam python=3.11 -y
conda activate anycam

# Install PyTorch 2.5.1 with CUDA 12.4
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124

# Install CUDA toolkit for compilation
conda install -c nvidia cuda-toolkit -y

# Install dependencies
pip install -r requirements.txt
```

**Note**: Uses customized forks of UniMatch and UniDepth (in submodules) and minipytorch3d variant by VGGSfM for backward compatibility.

### Download Pretrained Models

```bash
# Download final AnyCam model (seq8, trained on 8 frames)
./download_checkpoints.sh anycam_seq8

# Models saved to: pretrained_models/anycam_seq8/
```

### Running AnyCam Demo

```bash
# Full model with bundle adjustment refinement
python anycam/scripts/anycam_demo.py \
    ++input_path=/path/to/video.mp4 \
    ++model_path=pretrained_models/anycam_seq8 \
    ++visualize=true

# Feed-forward only (no refinement)
python anycam/scripts/anycam_demo.py \
    ++input_path=/path/to/video.mp4 \
    ++model_path=pretrained_models/anycam_seq8 \
    ++ba_refinement=false \
    ++visualize=true

# Export to COLMAP format
python anycam/scripts/anycam_demo.py \
    ++input_path=/path/to/video.mp4 \
    ++model_path=pretrained_models/anycam_seq8 \
    ++export_colmap=true \
    ++output_path=/path/to/output_dir
```

**Visualization on remote server**:
```bash
# Terminal 1: Start rerun viewer
rerun --serve-web

# Terminal 2: Run demo with connect mode
python anycam/scripts/anycam_demo.py \
    ++input_path=/path/to/video.mp4 \
    ++model_path=pretrained_models/anycam_seq8 \
    ++rerun_mode=connect

# Access in browser: http://localhost:9090/?url=ws://localhost:9877
```

### Training Experiments

**Experiment 1 (AnyCalib + Single Pose Head)**:
```bash
python experiments/train_pose_head_anycalib.py \
    --objectron_videos /path/to/Objectron/videos \
    --objectron_gt /path/to/Objectron/processed_gt \
    --num_epochs 50 \
    --batch_size 4 \
    --learning_rate 1e-4 \
    --save_dir experiments/pose_head_experiment_results/full_run
```

**Experiment 2 (Multi-Frame Consistency)**:
```bash
python experiments/train_pose_head_anycalib_exp2.py \
    --objectron_videos /path/to/Objectron/videos \
    --objectron_gt /path/to/Objectron/processed_gt \
    --num_epochs 50 \
    --batch_size 4 \
    --learning_rate 1e-4 \
    --max_ahead 3 \
    --save_dir experiments/pose_head_experiment_results/exp2_optimal_run
```

**DA3 Stage 1 (Mean Calibration)**:
```bash
python experiments/train_calibration_head_da3_stage1.py \
    --objectron_videos /data/thesis/Objectron/videos \
    --objectron_gt /data/thesis/Objectron/processed_gt \
    --num_epochs 50 \
    --batch_size 8 \
    --learning_rate 1e-4 \
    --save_dir experiments/da3_integration/stage1_training
```

**DA3 Stage 2 (Visual Conditioning)**:
```bash
python experiments/train_calibration_head_da3_stage2.py \
    --objectron_videos /data/thesis/Objectron/videos \
    --objectron_gt /data/thesis/Objectron/processed_gt \
    --stage1_checkpoint experiments/da3_integration/stage1_training/checkpoints/final_model.pt \
    --num_epochs 50 \
    --batch_size 4 \
    --learning_rate 5e-5 \
    --save_dir experiments/da3_integration/stage2_training
```

**DA3 Stage 3 (End-to-End Flow Reprojection)**:
```bash
python experiments/train_calibration_head_da3_stage3.py \
    --objectron_videos /data/thesis/Objectron/videos \
    --stage2_checkpoint experiments/da3_integration/stage2_training/checkpoints/final_model.pt \
    --num_epochs 50 \
    --batch_size 2 \
    --learning_rate 1e-5 \
    --save_dir experiments/da3_integration/stage3_training
```

**DA3 Stage 3.1 (Multi-Frame with Alternating Training)**:
```bash
# max_ahead=3, alternating training
python experiments/train_calibration_head_da3_stage3.py \
    --objectron_videos /data/thesis/Objectron/videos \
    --objectron_gt /data/thesis/Objectron/processed_gt \
    --stage2_checkpoint experiments/da3_integration/stage2_training/checkpoints/final_model.pt \
    --num_epochs 50 \
    --batch_size 2 \
    --learning_rate 1e-5 \
    --max_ahead 3 \
    --alternating_training \
    --benchmark_samples 100 \
    --benchmark_no_cycle \
    --save_dir experiments/da3_integration/stage3_1_maxahead3_alternating
```

### Benchmarking

**Pose Estimation Benchmark**:
```bash
# Evaluate on LightSpeed dataset
python experiments/benchmark_against_anycam.py \
    --da3_stage3_model experiments/da3_integration/stage3_training/checkpoints/final_model.pt \
    --save_dir experiments/da3_integration/benchmark_results/stage3_vs_baseline \
    --dataset lightspeed \
    --num_samples 100

# Evaluate on Objectron dataset
python experiments/benchmark_against_anycam.py \
    --da3_stage3_model experiments/da3_integration/stage3_training/checkpoints/final_model.pt \
    --save_dir experiments/da3_integration/benchmark_results/stage3_vs_baseline \
    --dataset objectron \
    --num_samples 100 \
    --split_file experiments/objectron_split.json
```

**DA3 Calibration Accuracy Benchmark**:
```bash
# Stage 1
python experiments/benchmark_da3_calibration_accuracy.py \
    --stage1_checkpoint experiments/da3_integration/stage1_training/checkpoints/final_model.pt \
    --dataset objectron \
    --num_samples 100 \
    --save_dir experiments/da3_integration/benchmark_results/stage1_calibration

# Stage 2
python experiments/benchmark_da3_calibration_accuracy.py \
    --stage2_checkpoint experiments/da3_integration/stage2_training/checkpoints/final_model.pt \
    --dataset objectron \
    --num_samples 100 \
    --save_dir experiments/da3_integration/benchmark_results/stage2_calibration

# Stage 3
python experiments/benchmark_da3_calibration_accuracy.py \
    --stage3_checkpoint experiments/da3_integration/stage3_training/checkpoints/final_model.pt \
    --dataset objectron \
    --num_samples 100 \
    --vis_dim 384 \
    --save_dir experiments/da3_integration/benchmark_results/stage3_calibration
```

**Inter-Stage Comparison** (scientifically valid for validation experiments):
```bash
python experiments/benchmark_da3_stages_comparison.py \
    --stage1_checkpoint experiments/da3_integration/stage1_training/checkpoints/final_model.pt \
    --stage2_checkpoint experiments/da3_integration/stage2_training/checkpoints/final_model.pt \
    --stage3_checkpoint experiments/da3_integration/stage3_training/checkpoints/final_model.pt \
    --dataset objectron \
    --num_samples 100 \
    --vis_dim_stage3 384 \
    --save_dir experiments/da3_integration/benchmark_results/stages_comparison
```

### Evaluation on Standard Datasets

```bash
# Evaluate AnyCam on standard benchmarks
python anycam/scripts/evaluate_trajectories.py \
    -cn evaluate_trajectories \
    ++model_path=pretrained_models/anycam_seq8

# With rerun visualization
python anycam/scripts/evaluate_trajectories.py \
    -cn evaluate_trajectories \
    ++model_path=pretrained_models/anycam_seq8 \
    ++fit_video.ba_refinement.with_rerun=true
```

## Important Implementation Details

### Dataset Locations

**Objectron Dataset** (primary validation dataset):
- Videos: `/home/kalman/TUM/thesis/Objectron/videos/` (100 sequences)
- Ground Truth: `/home/kalman/TUM/thesis/Objectron/processed_gt/` (101 JSON files)
- Split file: `experiments/objectron_split.json` (train/val/test splits)
- **Provides**: GT camera calibration + GT poses → supports both calibration and pose benchmarks

**LightSpeed Dataset** (pose evaluation only):
- Location: Auto-detected by benchmark scripts
- Validation set: 200 samples
- **Provides**: GT poses only (no GT calibration) → supports pose benchmarks only

### Docker Container Usage

When working in Docker container, paths use `/data/thesis/` prefix:
```bash
# Training commands in container use:
--objectron_videos /data/thesis/Objectron/videos
--objectron_gt /data/thesis/Objectron/processed_gt
```

### Data Loading Patterns

**Frame Pairs (Stage 1, Stage 2, Experiments 1-2)**:
- Consecutive frame pairs: (0→1, 1→2, 2→3, ...)
- Dataset parameter: `extract_all_pairs=True`, `max_pairs_per_video=None` (unlimited)
- Each video yields multiple training samples

**Multi-Frame Sequences (Experiment 2, Stage 3.1)**:
- Loads `max_ahead + 1` frames (e.g., 4 frames for max_ahead=3)
- Predicts consecutive pairs: 1→2, 2→3, 3→4
- Composes long-range predictions: 1→3 = 1→2 @ 2→3, 1→4 = 1→3 @ 3→4
- Flow composition: Warps consecutive flows instead of running UniMatch on distant pairs

**All Frames (Stage 1 & 2 only)**:
- Loads ALL available frames from each video (potentially 100-500 frames)
- Computes GT mean calibration from all frames
- Uses chunked processing (chunk_size=16) to avoid OOM

### Flow Composition

Flow composition is a key technique in Experiment 2 and Stage 3.1:
```python
# Compose flow from frame 1→3 using intermediate frame 2
flow_1_to_3_composed = compose_flow(flow_1_to_2, flow_2_to_3)

# Uses bilinear interpolation to warp through intermediate frame
# More efficient than running UniMatch on distant pairs
# Enables multi-frame consistency without excessive computation
```

**Note**: Flow error accumulates with longer sequences. Optimal `max_ahead=3` balances accuracy and error accumulation.

### Visual Token Extraction

**Stage 2 & Stage 3.1 (DINOv2-small)**:
```python
# Uses HuggingFace transformers DINOv2-small (vis_dim=384)
from transformers import AutoModel
visual_backbone = AutoModel.from_pretrained('facebook/dinov2-small')

# Extract CLS token from each frame
outputs = visual_backbone(images)
visual_tokens = outputs.last_hidden_state[:, 0, :]  # [B, N, 384]
```

**Why HuggingFace instead of torch.hub?**
- RTX 5090 has compute capability 12.0
- Original `torch.hub.load('facebookresearch/dinov2', ...)` uses xFormers
- xFormers only supports GPU compute capability ≤ 9.0 → incompatible
- HuggingFace version uses native PyTorch attention (SDPA), compatible with all GPUs

### Memory Management

**GPU Memory Constraints** (24GB VRAM):
- Stage 2 preprocessing: Chunked processing (16 frames at a time)
- Stage 3 training: Batch size 2, periodic cache clearing
- Mixed precision training: `torch.amp.autocast`
- Move intermediate results to CPU immediately after processing

### Checkpoint Loading

**Smart checkpoint loading** handles dimension mismatches:
```python
# Stage 2 loads Stage 1 checkpoint
# Filters out visual_camera_mixing weights (dimension mismatch)
# Safe because visual_camera_mixing was frozen in Stage 1

# Stage 3 loads Stage 2 checkpoint
# Standalone DA3CalibrationHead with own DINOv2-small (vis_dim=384)
# Ensures consistent dimensions across Stage 2 → Stage 3
```

### Loss Functions

**Flow Reprojection Loss** (self-supervised):
- Compares predicted flow (from poses + depth) vs observed flow (UniMatch)
- Used in Experiments 1-2 and DA3 Stage 3
- **Location**: `anycam/loss/pose_loss.py`

**MSE Loss** (supervised):
- Used in DA3 Stage 1 & 2: `F.mse_loss(predicted_calibration, gt_mean_calibration)`
- Requires ground truth calibration (Objectron dataset only)

**Composed Flow Loss** (Experiment 2):
- Consecutive pairs: Full weight (1.0)
- Composed pairs: Reduced weight (0.1) to balance loss magnitude

## Key Findings and Results

### Experiment 1 (Single Pose Head)
- Rotation error: mean 1.39°, median 1.01° (LightSpeed validation)
- Successfully trained fresh pose head with AnyCalib focal lengths
- Comparable to AnyCam baseline

### Experiment 2 (Multi-Frame Consistency)
- **Best rotation error**: mean 1.23°, median 0.92° (LightSpeed validation)
- Optimal `max_ahead=3` (4 frames total)
- Outperforms Experiment 1 and AnyCam baseline
- Flow composition enables multi-frame consistency without excessive computation

### DA3 Integration (Current Status)
- Stage 1 & 2: Complete, trained on Objectron
- Stage 3: Architecture fixed (standalone DINOv2-small for consistent vis_dim=384)
- Stage 3.1: Four variants with max_ahead=3/4 and alternating training
- **Validation experiments only** - dataset-specific overfitting expected
- Inter-stage comparison is scientifically valid benchmark
- Production training requires large-scale diverse datasets

### Benchmark Reliability Notes
- **Inter-stage comparisons** (Stage 1 vs 2 vs 3): Scientifically valid, same training/eval dataset
- **Pose benchmarks** (DA3 vs AnyCam): Somewhat reliable, DINOv2 frozen (visual features fixed)
- **Calibration benchmarks** (DA3 vs AnyCalib): Interpret with caution, different training datasets
- **Objectron dataset**: GT calibration + GT poses → supports all benchmarks
- **LightSpeed dataset**: GT poses only → pose benchmarks only

## Training Datasets (AnyCam Paper)

AnyCam was originally trained on five datasets:
1. RealEstate10K
2. YouTube VOS
3. WalkingTours
4. OpenDV
5. EpicKitchens

**Note**: Thesis experiments use Objectron for validation only. Production training would use larger datasets.

## Important Files for Understanding

**Architecture**:
- `anycam/models/anycam.py` (line 305: visual tokens extraction point)
- `anycam/models/anycam_blocks.py` (pose head and sequence info head)
- `experiments/models/da3_calibration_head.py` (DA3 complete architecture)

**Integration Wrappers**:
- `experiments/train_pose_head_anycalib.py` (lines 506-588: `AnyCamWrapperWithAnyCaLib` and `AnyCamWrapperWithDA3Calibration`)

**Loss Computation**:
- `anycam/loss/pose_loss.py` (flow reprojection loss)
- `anycam/trainer.py` (training loop with loss computation)

**Flow and Depth**:
- `anycam/common/image_processor.py` (optical flow via UniMatch)
- `anycam/models/depth_predictor_wrapper.py` (depth prediction via UniDepth)

**Evaluation**:
- `experiments/pose_metrics.py` (rotation/translation error metrics)
- `experiments/benchmark_dataset_utils.py` (dataset loaders for benchmarks)

## Documentation Files

**Comprehensive thesis documentation**:
- `THESIS_WORK_SUMMARY.md` - Complete timeline, experiments, and findings
- `experiments/ARCHITECTURE_FINDINGS.md` - Detailed AnyCam architecture analysis
- `experiments/da3_integration/README.md` - DA3 training stages overview
- `experiments/da3_integration/IMPLEMENTATION_SUMMARY.md` - Complete DA3 implementation details with bugs and fixes
- `experiments/da3_integration/EXPERIMENT_SUMMARY.md` - DA3 experiment results and analysis
- `experiments/da3_integration/BENCHMARK_USAGE.md` - Comprehensive benchmark guide

**Quick references**:
- `experiments/EXPERIMENT_QUICKSTART.md` - Quick start for Experiments 1-2
- `experiments/COMPOSED_FLOW_IMPLEMENTATION.md` - Flow composition details
- `experiments/MULTI_PAIR_IMPLEMENTATION_SUMMARY.md` - Multi-pair dataset implementation

## Git Workflow

**Main branch**: `main` (unchanged AnyCam codebase)
**Experiment branch**: `repathing-laptop` (current, all thesis work)

**Recent commits focus on**:
- DA3 Stage 3 training implementation
- Multi-frame benchmarking with alternating training strategies
- Documentation updates for BENCHMARK_USAGE.md and EXPERIMENT_SUMMARY.md

## Common Pitfalls and Solutions

### xFormers GPU Compatibility
- **Problem**: RTX 5090 (compute capability 12.0) incompatible with xFormers
- **Solution**: Use HuggingFace DINOv2 (`facebook/dinov2-small`) with native PyTorch attention

### Variable-Length Sequences
- **Problem**: Different videos have different frame counts
- **Solution**: Custom collate function with padding and attention masks

### Out of Memory (OOM)
- **Problem**: Processing 100-500 frames per video
- **Solution**: Chunked processing (16 frames at a time) + periodic cache clearing

### Dimension Mismatches
- **Problem**: Stage 1 trained with vis_dim=768, Stage 2 needs vis_dim=384
- **Solution**: Smart checkpoint loading that filters mismatched keys

### Flow Error Accumulation
- **Problem**: Longer sequences (max_ahead > 3) show degraded performance
- **Solution**: Optimal max_ahead=3 balances accuracy and error accumulation

## Thesis Timeline

- **Submission Deadline**: March 31, 2025
- **Presentation Date**: ~March 10, 2025
- **Current Phase**: DA3 Stage 3 training and evaluation
- **Next Steps**: Inter-stage comparison, thesis writing, defense preparation
