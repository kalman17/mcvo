# DA3 Integration Experiment

This directory contains all experiments, training results, and benchmarks for the Depth Anything 3 (DA3) calibration head integration into AnyCam.

## Directory Structure

```
da3_integration/
├── README.md                          # This file
├── stage1_training/                   # Stage 1 training results
│   ├── checkpoints/                    # Model checkpoints
│   ├── training_summary.txt            # Training logs
│   ├── loss_curve.png                  # Loss visualization
│   └── calibration_accuracy.json       # Phase 1 accuracy metrics
├── stage2_training/                   # Stage 2 training results
│   ├── checkpoints/
│   ├── training_summary.txt
│   └── loss_curve.png
├── stage3_training/                   # Stage 3 training results
│   ├── checkpoints/
│   ├── training_summary.txt
│   └── loss_curve.png
├── stage3_1_maxahead3/                # Stage 3.1: max_ahead=3, no alternating
├── stage3_1_maxahead4/                # Stage 3.1: max_ahead=4, no alternating
├── stage3_1_maxahead3_alternating/   # Stage 3.1: max_ahead=3, alternating
└── stage3_1_maxahead4_alternating/    # Stage 3.1: max_ahead=4, alternating
└── benchmark_results/                 # All benchmark results
    ├── stage1_vs_baseline/            # Stage 1 benchmark
    ├── stage2_vs_baseline/            # Stage 2 benchmark
    └── stage3_vs_baseline/            # Stage 3 benchmark (final)
```

## Training Stages

### Stage 1: Mean Calibration Learning
**Objective**: Learn to aggregate per-frame AnyCalib predictions into sequence-level mean calibration.

**Key Feature**: Loads ALL frames from each video to compute the best possible mean calibration estimate.

**Training Command** (run inside Docker container):
```bash
python experiments/train_calibration_head_da3_stage1.py \
    --objectron_videos /data/thesis/Objectron/videos \
    --objectron_gt /data/thesis/Objectron/processed_gt \
    --num_epochs 50 \
    --batch_size 8 \
    --learning_rate 1e-4 \
    --save_dir experiments/da3_integration/stage1_training
```

**Loss**: MSE(predicted_mean_calibration, gt_mean_calibration)

**Evaluation**: Relative error to target mean calibration (saved in `calibration_accuracy.json`)

### Stage 2: Visual-Conditioned Calibration
**Objective**: Learn to leverage visual features for improved calibration.

**Key Feature**: Loads ALL frames from each video and extracts visual tokens from AnyCam backbone.

**Training Command** (run inside Docker container):
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

**Loss**: MSE(predicted_mean_calibration, gt_mean_calibration) with visual tokens

### Stage 3: End-to-End Flow Reprojection
**Objective**: Integrate into full AnyCam pipeline and train with flow reprojection loss (self-supervised, no GT needed).

**Training Command**:
```bash
python experiments/train_calibration_head_da3_stage3.py \
    --objectron_videos /data/thesis/Objectron/videos \
    --stage2_checkpoint experiments/da3_integration/stage2_training/checkpoints/final_model.pt \
    --num_epochs 50 \
    --batch_size 2 \
    --learning_rate 1e-5 \
    --save_dir experiments/da3_integration/stage3_training
```

**Loss**: Flow reprojection loss (self-supervised)

### Stage 3.1: Multi-Frame Variants with Optional Alternating Training

**Objective**: Extend Stage 3 with multi-frame sequences (max_ahead=3 or 4) and optional alternating training strategy.

**Four Training Variants**:
1. **max_ahead=3, no alternating**: 4-frame sequences, standard training
2. **max_ahead=4, no alternating**: 5-frame sequences, standard training
3. **max_ahead=3, alternating**: 4-frame sequences, alternating training strategy
4. **max_ahead=4, alternating**: 5-frame sequences, alternating training strategy

**Training Commands**:

**Stage 3.1 (max_ahead=3, no alternating)**:
```bash
python experiments/train_calibration_head_da3_stage3.py \
    --objectron_videos /data/thesis/Objectron/videos \
    --objectron_gt /data/thesis/Objectron/processed_gt \
    --stage2_checkpoint experiments/da3_integration/stage2_training/checkpoints/final_model.pt \
    --num_epochs 50 \
    --batch_size 2 \
    --learning_rate 1e-5 \
    --max_ahead 3 \
    --benchmark_samples 100 \
    --benchmark_no_cycle \
    --save_dir experiments/da3_integration/stage3_1_maxahead3
```

**Stage 3.1 (max_ahead=4, no alternating)**:
```bash
python experiments/train_calibration_head_da3_stage3.py \
    --objectron_videos /data/thesis/Objectron/videos \
    --objectron_gt /data/thesis/Objectron/processed_gt \
    --stage2_checkpoint experiments/da3_integration/stage2_training/checkpoints/final_model.pt \
    --num_epochs 50 \
    --batch_size 2 \
    --learning_rate 1e-5 \
    --max_ahead 4 \
    --benchmark_samples 100 \
    --benchmark_no_cycle \
    --save_dir experiments/da3_integration/stage3_1_maxahead4
```

**Stage 3.1 (max_ahead=3, with alternating training)**:
```bash
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

**Stage 3.1 (max_ahead=4, with alternating training)**:
```bash
python experiments/train_calibration_head_da3_stage3.py \
    --objectron_videos /data/thesis/Objectron/videos \
    --objectron_gt /data/thesis/Objectron/processed_gt \
    --stage2_checkpoint experiments/da3_integration/stage2_training/checkpoints/final_model.pt \
    --num_epochs 50 \
    --batch_size 2 \
    --learning_rate 1e-5 \
    --max_ahead 4 \
    --alternating_training \
    --benchmark_samples 100 \
    --benchmark_no_cycle \
    --save_dir experiments/da3_integration/stage3_1_maxahead4_alternating
```

**Key Features**:
- **Multi-frame Input**: Uses `max_ahead+1` frames per sequence (4 for max_ahead=3, 5 for max_ahead=4)
- **Fixed Benchmark**: Uses same 100 samples every epoch (no cycling) for consistent evaluation
- **Alternating Training** (optional): Alternates between training calibration head and pose head each epoch
- **Dataset**: Uses `ObjectronVideoDatasetMultiFrame` for multi-frame sequences

**Alternating Training Strategy**:
- Even epochs (0, 2, 4, ...): Train calibration head, freeze pose head
- Odd epochs (1, 3, 5, ...): Train pose head, freeze calibration head
- Optimizer is recreated each epoch with appropriate trainable parameters

## Benchmarking

### Benchmark DA3 Models Against AnyCam Baseline

**Command**:
```bash
python experiments/benchmark_against_anycam.py \
    --da3_stage3_model experiments/da3_integration/stage3_training/checkpoints/final_model.pt \
    --save_dir experiments/da3_integration/benchmark_results/stage3_vs_baseline \
    --dataset lightspeed
```

**Results Location**: `experiments/da3_integration/benchmark_results/stage3_vs_baseline/`

**Output Files**:
- `benchmark_results.json` - Detailed metrics
- `benchmark_report.txt` - Text report
- `benchmark_comparison.png` - Visualization

## Evaluation

### Evaluate Calibration Accuracy (Stage 1 & 2)

**Command**:
```bash
python experiments/evaluate_da3_calibration.py \
    --checkpoint experiments/da3_integration/stage1_training/checkpoints/final_model.pt \
    --stage 1 \
    --objectron_videos /path/to/videos \
    --objectron_gt /path/to/gt
```

## Model Architecture

The DA3 calibration head consists of:
1. **Camera Encoder**: Encodes camera parameters (fx, fy, cx, cy) to tokens
2. **Visual-Camera Mixing**: Updates camera tokens with visual information (reverse conditioning)
3. **Sequence Aggregation**: Aggregates per-frame tokens to sequence level
4. **Camera Decoder**: Decodes tokens back to camera parameters

### Visual Token Extraction

All stages use **HuggingFace DINOv2-small** for visual token extraction (`vis_dim=384`):
- Uses native PyTorch attention (compatible with RTX 5090 and newer GPUs)
- Extracts CLS token from each frame
- Consistent across Stage 2 and Stage 3 for proper checkpoint loading

**Note**: Stage 3 uses a **standalone** DA3CalibrationHead with its own DINOv2-small, separate from the pose_predictor's backbone. This ensures Stage 2 weights load correctly.

## Key Features

- **Staged Training**: Three stages allow gradual complexity introduction
- **Self-Supervised**: Stage 3 uses flow reprojection loss (no GT calibration needed)
- **Consistent Visual Features**: DINOv2-small used across all stages (vis_dim=384)
- **GPU Compatibility**: HuggingFace DINOv2 uses native attention (no xFormers dependency)
- **Memory Efficient**: Designed for 24GB VRAM with periodic cache clearing
- **Organized Results**: Clear folder structure for easy thesis reference

## Notes

- All training uses Objectron dataset with all available frames (`extract_all_pairs=True`, unlimited)
- Stage 1 & 2 require GT calibration for training (MSE loss against GT mean)
- Stage 3 is fully self-supervised (no GT calibration needed, uses flow reprojection loss)
- Benchmarking uses ground truth poses for evaluation
- **Important**: This is a validation experiment on Objectron only. Comparisons with AnyCam baseline are somewhat reliable because DINOv2 from AnyCam encoder is frozen (visual features fixed, only calibration head trained), making the comparison more fair. Comparisons with general-purpose methods (AnyCalib) should be interpreted with caution due to different training datasets.

