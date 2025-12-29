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
**Objective**: Integrate into full AnyCam pipeline and train with flow reprojection loss.

**Training Command**:
```bash
python experiments/train_calibration_head_da3_stage3.py \
    --objectron_videos /data/thesis/Objectron/videos \
    --objectron_gt /data/thesis/Objectron/processed_gt \
    --stage2_checkpoint experiments/da3_integration/stage2_training/checkpoints/final_model.pt \
    --num_epochs 50 \
    --batch_size 2 \
    --learning_rate 1e-5 \
    --save_dir experiments/da3_integration/stage3_training
```

**Loss**: Flow reprojection loss (self-supervised)

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

## Key Features

- **Staged Training**: Three stages allow gradual complexity introduction
- **Self-Supervised**: Stage 3 uses flow reprojection loss (no GT calibration needed)
- **Comprehensive Evaluation**: Both calibration accuracy (Stage 1/2) and pose accuracy (Stage 3)
- **Organized Results**: Clear folder structure for easy thesis reference

## Notes

- All training uses Objectron dataset with all available frames (`extract_all_pairs=True`, unlimited)
- Stage 1 & 2 require GT calibration for training
- Stage 3 is fully self-supervised (no GT calibration needed)
- Benchmarking uses ground truth poses for evaluation

