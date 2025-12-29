# DA3 Benchmark Scripts - Quick Reference Guide

## Overview

This guide provides all commands needed to run the DA3 calibration head benchmarks. The benchmarks evaluate calibration accuracy and pose estimation performance at different training stages.

**⚠️ IMPORTANT - Validation Experiments Only**:
The current DA3 models are trained exclusively on Objectron for validation purposes. Due to dataset-specific overfitting:
- **Calibration benchmarks**: Comparisons with general-purpose methods (e.g., AnyCalib) are NOT scientifically valid
- **Pose estimation benchmarks**: Comparisons with AnyCam baseline/AnyCalib hybrid (trained on different datasets) are NOT fair comparisons
- **Valid comparisons**: Only inter-stage comparisons (Stage 1 vs 2 vs 3) are scientifically valid at this stage
- **Fair comparisons**: Require retraining all models on the same large-scale, diverse dataset (future work)

**Available Benchmarks:**
1. **Quick Calibration Accuracy** - Individual stage evaluation vs GT mean intrinsics (Objectron only - has GT calibration)
2. **Full Pose Estimation** - Stage 3 vs baselines (for future use after large-scale training)
3. **Inter-Stage Comparison** - Calibration accuracy comparison between Stages 1, 2, 3 (scientifically valid)

## Quick Calibration Accuracy Benchmark

Evaluates individual DA3 stage models on calibration accuracy vs GT mean intrinsics using frame pairs.

### Stage 1 Calibration Accuracy

```bash
python experiments/benchmark_da3_calibration_accuracy.py \
    --stage1_checkpoint experiments/da3_integration/stage1_training/checkpoints/final_model.pt \
    --dataset objectron \
    --num_samples 100 \
    --split_file experiments/objectron_split.json \
    --save_dir experiments/da3_integration/benchmark_results/stage1_calibration \
    --batch_size 8 \
    --device cuda:0
```

### Stage 2 Calibration Accuracy

```bash
python experiments/benchmark_da3_calibration_accuracy.py \
    --stage2_checkpoint experiments/da3_integration/stage2_training/checkpoints/final_model.pt \
    --dataset objectron \
    --num_samples 100 \
    --split_file experiments/objectron_split.json \
    --save_dir experiments/da3_integration/benchmark_results/stage2_calibration \
    --batch_size 8 \
    --device cuda:0
```

### Stage 3 Calibration Accuracy

```bash
python experiments/benchmark_da3_calibration_accuracy.py \
    --stage3_checkpoint experiments/da3_integration/stage3_training/checkpoints/final_model.pt \
    --dataset objectron \
    --num_samples 100 \
    --split_file experiments/objectron_split.json \
    --save_dir experiments/da3_integration/benchmark_results/stage3_calibration \
    --batch_size 8 \
    --device cuda:0 \
    --vis_dim 768
```

**Note**: 
- Stage 3 uses `--vis_dim 768` (AnyCam backbone) while Stages 1/2 use default 384 (DINOv2)
- Stage 3 checkpoint contains the full AnyCam model; the calibration head weights are automatically extracted

## Full Pose Estimation Benchmark

Compares Stage 3 DA3+AnyCam hybrid vs AnyCalib+AnyCam hybrid vs AnyCam baseline on pose estimation accuracy.

```bash
python experiments/benchmark_da3_pose_estimation.py \
    --stage3_checkpoint experiments/da3_integration/stage3_training/checkpoints/final_model.pt \
    --baseline_checkpoint pretrained_models/anycam_seq8/training_checkpoint_247500.pt \
    --dataset lightspeed \
    --num_samples 100 \
    --save_dir experiments/da3_integration/benchmark_results/stage3_pose_estimation \
    --batch_size 2 \
    --device cuda:0 \
    --num_frames 2
```

**For Objectron dataset:**

```bash
python experiments/benchmark_da3_pose_estimation.py \
    --stage3_checkpoint experiments/da3_integration/stage3_training/checkpoints/final_model.pt \
    --baseline_checkpoint pretrained_models/anycam_seq8/training_checkpoint_247500.pt \
    --dataset objectron \
    --num_samples 100 \
    --split_file experiments/objectron_split.json \
    --save_dir experiments/da3_integration/benchmark_results/stage3_pose_estimation \
    --batch_size 2 \
    --device cuda:0 \
    --num_frames 2
```

**⚠️ Note on Fair Comparison**:
This benchmark compares models trained on different datasets and is NOT a fair scientific comparison at this validation stage. Use this benchmark only after retraining all models (DA3, AnyCam baseline, AnyCalib hybrid) on the same large-scale, diverse dataset. The current results demonstrate the technical implementation but do not represent true relative performance.

## Inter-Stage Comparison (✅ Scientifically Valid)

Compares calibration accuracy of DA3 Stages 1, 2, and 3 against each other on the same test set. 

**This is the scientifically valid benchmark** for the validation experiments, as all stages are trained and evaluated on the same dataset (Objectron), allowing fair comparison of the training progression.

```bash
python experiments/benchmark_da3_stages_comparison.py \
    --stage1_checkpoint experiments/da3_integration/stage1_training/checkpoints/final_model.pt \
    --stage2_checkpoint experiments/da3_integration/stage2_training/checkpoints/final_model.pt \
    --stage3_checkpoint experiments/da3_integration/stage3_training/checkpoints/final_model.pt \
    --dataset objectron \
    --num_samples 100 \
    --split_file experiments/objectron_split.json \
    --save_dir experiments/da3_integration/benchmark_results/stages_comparison \
    --batch_size 8 \
    --device cuda:0 \
    --vis_dim_stage3 768
```

## Common Options

All benchmark scripts support the following common options:

- `--dataset`: Dataset to use (`objectron` or `lightspeed`) - paths are auto-detected
  - **Objectron**: Has GT calibration and GT poses → supports both calibration and pose benchmarks
  - **LightSpeed**: Has GT poses only (no GT calibration) → supports pose benchmarks only
- `--num_samples`: Number of frame pairs to evaluate (default: `100`, use `all` for all available)
- `--batch_size`: Batch size for evaluation (default: 2-8 depending on script)
- `--device`: Device to use (default: cuda:0)
- `--split_file`: Dataset split file for Objectron (optional, default: experiments/objectron_split.json)

**Smart Sampling Logic:**
- If `--num_samples` is a number (e.g., `100`), the script samples that many frame pairs without repetition
- If `--num_samples all`, the script uses all available frame pairs
- If requested samples > available, the script automatically uses all available (no repetition)
- Sampling cycles through videos intelligently to avoid using the same pair twice

## Output Locations

All benchmarks save results to organized directories:

- **Quick benchmarks**: 
  - `experiments/da3_integration/benchmark_results/stage1_calibration/`
  - `experiments/da3_integration/benchmark_results/stage2_calibration/`
  - `experiments/da3_integration/benchmark_results/stage3_calibration/`

- **Pose estimation benchmark**: 
  - `experiments/da3_integration/benchmark_results/stage3_pose_estimation/`

- **Inter-stage comparison**: 
  - `experiments/da3_integration/benchmark_results/stages_comparison/`

## Output Files

Each benchmark generates:

- `calibration_accuracy.json` / `pose_estimation_results.json` / `stages_comparison_results.json`: Detailed error metrics
- `*_report.txt`: Human-readable report with professional formatting
- `*_distributions.png`: Error distribution histograms
- `*_statistics.png`: Bar charts comparing error statistics
- `*_comparison.png`: Comparison plots (for multi-model benchmarks)
- `metadata.json`: Benchmark configuration and parameters

## Recommended Workflow (Validation Experiments)

**For current validation experiments (trained on Objectron only):**

Follow this order of operations:

1. **Quick check of Stage 1 calibration accuracy**
   ```bash
   python experiments/benchmark_da3_calibration_accuracy.py \
       --stage1_checkpoint experiments/da3_integration/stage1_training/checkpoints/final_model.pt \
       --dataset objectron \
       --num_samples 100 \
       --save_dir experiments/da3_integration/benchmark_results/stage1_calibration
   ```

2. **Quick check of Stage 2 calibration accuracy**
   ```bash
   python experiments/benchmark_da3_calibration_accuracy.py \
       --stage2_checkpoint experiments/da3_integration/stage2_training/checkpoints/final_model.pt \
       --dataset objectron \
       --num_samples 100 \
       --save_dir experiments/da3_integration/benchmark_results/stage2_calibration
   ```

3. **Train Stage 3** (see training documentation)

4. **Quick check of Stage 3 calibration accuracy**
   ```bash
   python experiments/benchmark_da3_calibration_accuracy.py \
       --stage3_checkpoint experiments/da3_integration/stage3_training/checkpoints/final_model.pt \
       --dataset objectron \
       --num_samples 100 \
       --save_dir experiments/da3_integration/benchmark_results/stage3_calibration \
       --vis_dim 768
   ```

5. **✅ Inter-stage comparison (SCIENTIFICALLY VALID - Primary benchmark for validation)**
   ```bash
   python experiments/benchmark_da3_stages_comparison.py \
       --stage1_checkpoint experiments/da3_integration/stage1_training/checkpoints/final_model.pt \
       --stage2_checkpoint experiments/da3_integration/stage2_training/checkpoints/final_model.pt \
       --stage3_checkpoint experiments/da3_integration/stage3_training/checkpoints/final_model.pt \
       --dataset objectron \
       --num_samples 100 \
       --save_dir experiments/da3_integration/benchmark_results/stages_comparison \
       --vis_dim_stage3 768
   ```

**For future production experiments (after large-scale training):**

6. **Pose estimation benchmark (Stage 3 vs baseline vs hybrid) - ONLY AFTER LARGE-SCALE TRAINING**
   ```bash
   # Use this ONLY after retraining on large, diverse datasets
   python experiments/benchmark_da3_pose_estimation.py \
       --stage3_checkpoint experiments/da3_integration/stage3_training/checkpoints/final_model.pt \
       --baseline_checkpoint pretrained_models/anycam_seq8/training_checkpoint_247500.pt \
       --dataset lightspeed \
       --num_samples 100 \
       --save_dir experiments/da3_integration/benchmark_results/stage3_pose_estimation
   ```

## Quick Testing

For faster testing during development, use `--num_samples` to limit the number of samples:

```bash
python experiments/benchmark_da3_calibration_accuracy.py \
    --stage1_checkpoint experiments/da3_integration/stage1_training/checkpoints/final_model.pt \
    --dataset objectron \
    --num_samples 50 \
    --save_dir experiments/da3_integration/benchmark_results/stage1_calibration
```

**Using all available data:**

```bash
python experiments/benchmark_da3_calibration_accuracy.py \
    --stage1_checkpoint experiments/da3_integration/stage1_training/checkpoints/final_model.pt \
    --dataset objectron \
    --num_samples all \
    --save_dir experiments/da3_integration/benchmark_results/stage1_calibration
```

## Notes

**Validation Experiment Considerations:**
- Current models trained exclusively on Objectron (small-scale validation)
- Dataset-specific overfitting is expected and acceptable at this stage
- Inter-stage comparison is the primary scientifically valid benchmark
- Comparisons with general-purpose methods (AnyCalib, AnyCam baseline) are not valid until after large-scale training

**Technical Details:**
- All benchmarks use **frame pairs** (2 frames) for consistency
- GT mean intrinsics are computed from the pair for fair comparison
- DA3 models output sequence-level mean (appropriate for fixed camera)
- Results are organized clearly for thesis writing with professional labels
- All plots and reports use consistent, academic-style naming conventions

**Dataset Support:**
- **Calibration benchmarks**: Objectron only (has GT calibration)
- **Pose estimation benchmarks**: Both Objectron and LightSpeed (both have GT poses)

