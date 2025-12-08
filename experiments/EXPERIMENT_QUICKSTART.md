# Pose Head Retraining Experiment - Quick Start Guide

## Overview

This experiment tests whether we can successfully train a **fresh pose head** while using **AnyCaLib** for focal length prediction instead of AnyCam's original 32-candidate system.

**Branch:** `experiment/pose-head-retraining-anycalib-focal`

---

## Prerequisites

1. **Conda environment activated:** ` conda activate anycam`
   - If conda command not found, use: `source ~/anaconda3/bin/activate anycam` (or wherever conda is installed)
   
2. **Objectron dataset downloaded:** 
   - Videos: `<DATASETS_ROOT>/Objectron/videos/` (100 sequences ✓)
   - Ground truth: `<DATASETS_ROOT>/Objectron/processed_gt/` (101 JSON files ✓)
   - **Configuration:** Set `DATASETS_ROOT` environment variable or edit `experiments/dataset_paths.py`

3. **Pretrained AnyCam model:**
   - Path: `pretrained_models/anycam_seq8/` ✓

4. **AnyCaLib submodule:**
   - Path: `anycalib/` ✓

---

## Quick Start (Minimal Test)

Test on just **5 sequences** to verify everything works:

```bash
# Set dataset root (optional, defaults to /home/kalmanm/Documents/thesis)
export DATASETS_ROOT=/path/to/your/datasets

# Activate conda (if not already)
conda activate anycam

# Run experiment on 5 sequences, 2 epochs
python experiments/train_pose_head_anycalib.py \
    --max_sequences 5 \
    --num_epochs 2 \
    --batch_size 1 \
    --lr 1e-4 \
    --save_dir experiments/pose_head_experiment_results/test_run
```

**Expected output:**
```
[DATASET] Found 5 video sequences
[DATASET] 5 sequences have valid GT
[ANYCALIB] Loading pretrained model...
[ANYCALIB] Model loaded on cuda
[REINIT] Deleting old pose_head and creating fresh one...
[FREEZE] Freezing all layers except pose_head...
[PARAMS] Trainable: 16,897 / 85,324,271 (0.02%)
[TRAIN] Starting training for 2 epochs...
```

---

## Full Training Run

Train on **all 100 sequences** for 50 epochs:

```bash
python experiments/train_pose_head_anycalib.py \
    --num_epochs 50 \
    --batch_size 2 \
    --lr 1e-4 \
    --save_dir experiments/pose_head_experiment_results/full_run
```

**Estimated time:** ~3-4 hours on a single GPU (depending on your hardware)

---

## Command-Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--videos_dir` | `<DATASETS_ROOT>/Objectron/videos/` | Objectron video directory (from `dataset_paths.py`) |
| `--gt_dir` | `<DATASETS_ROOT>/Objectron/processed_gt/` | Ground truth JSON directory (from `dataset_paths.py`) |
| `--max_sequences` | `None` (all) | Limit number of sequences for debugging |
| `--num_frames` | `2` | Frames per sequence (keep at 2 for now) |
| `--batch_size` | `1` | Batch size (increase if you have GPU memory) |
| `--num_epochs` | `50` | Number of training epochs |
| `--lr` | `1e-4` | Learning rate |
| `--save_dir` | `./experiments/pose_head_experiment_results` | Where to save checkpoints |
| `--model_path` | `pretrained_models/anycam_seq8` | Pretrained AnyCam model path |
| `--anycalib_multi_frame` | `False` | Use multi-frame AnyCaLib averaging (slower) |

---

## AnyCaLib Focal Length Modes

### Mode 1: Single Frame (Default, FASTER)

Runs AnyCaLib only on the **first frame** of each sequence. Assumes constant focal length.

```bash
python experiments/train_pose_head_anycalib.py \
    --max_sequences 5 \
    --num_epochs 2
```

### Mode 2: Multi-Frame Averaging (SLOWER, MORE ROBUST)

Runs AnyCaLib on **all frames** and averages the predictions.

```bash
python experiments/train_pose_head_anycalib.py \
    --max_sequences 5 \
    --num_epochs 2 \
    --anycalib_multi_frame
```

---

## Understanding the Output

### Training Progress

```
[TRAIN] Epoch 1/2 | Batch 0/5 | Loss: 0.123456
[TRAIN] Epoch 1/2 | Batch 1/5 | Loss: 0.098765
...
[EPOCH 1] Average Loss: 0.111111
```

**What to look for:**
- ✅ **Loss decreasing:** The model is learning!
- ❌ **Loss not changing:** Something is frozen that shouldn't be, or learning rate too small
- ❌ **Loss exploding (>1e6):** Learning rate too high, or gradient issues

### Checkpoints

Saved every 5 epochs in `--save_dir`:

```
experiments/pose_head_experiment_results/
├── checkpoint_epoch_5.pt
├── checkpoint_epoch_10.pt
├── ...
└── final_model.pt
```

---

## Troubleshooting

### Issue: "conda: command not found"

**Solution:**
```bash
# Find conda installation
which conda

# If not found, manually activate
source ~/anaconda3/bin/activate
# Or wherever your conda is installed

# Then activate environment
conda activate anycam
```

### Issue: "No module named 'anycalib'"

**Solution:**
```bash
# Check if anycalib submodule is present
ls anycalib/

# If empty, initialize submodule
git submodule update --init --recursive
```

### Issue: "CUDA out of memory"

**Solution:**
```bash
# Reduce batch size
python experiments/train_pose_head_anycalib.py --batch_size 1 --max_sequences 5

# Or reduce number of frames
python experiments/train_pose_head_anycalib.py --num_frames 2
```

### Issue: "Checkpoint not found: pretrained_models/anycam_seq8/training_checkpoint_247500.pt"

**Solution:**
```bash
# Check if model exists
ls pretrained_models/anycam_seq8/

# If missing, download checkpoints
bash download_checkpoints.sh
```

### Issue: Loss not decreasing

**Possible causes:**
1. Learning rate too small → increase with `--lr 1e-3`
2. Pose head not unfrozen → check script output for `[UNFREEZE]` messages
3. Batch size too small → try `--batch_size 2` or `--batch_size 4`

---

## Verifying the Experiment

### 1. Check Parameter Freezing

Look for this in the output:
```
[FREEZE] Freezing all layers except pose_head...
[UNFREEZE] proj0.weight: torch.Size([64, 128])
[UNFREEZE] proj0.bias: torch.Size([64])
[UNFREEZE] proj1.weight: torch.Size([7, 64])
[UNFREEZE] proj1.bias: torch.Size([7])
[PARAMS] Trainable: 16,897 / 85,324,271 (0.02%)
```

✅ **Good:** Only pose_head parameters are unfrozen, trainable is ~0.02% of total

### 2. Check AnyCaLib is Running

Look for:
```
[FORWARD] Running AnyCaLib on batch...
[FORWARD] AnyCaLib focal lengths: tensor([1.2345, 1.3456], device='cuda:0')
```

✅ **Good:** Focal lengths are being predicted per batch

### 3. Check Loss Convergence

Monitor the loss over epochs:
```
[EPOCH 1] Average Loss: 0.500000
[EPOCH 2] Average Loss: 0.450000
[EPOCH 3] Average Loss: 0.420000
...
[EPOCH 50] Average Loss: 0.150000
```

✅ **Good:** Loss steadily decreases (we're overfitting intentionally)

---

## What's Happening Under the Hood?

### 1. Dataset Loading
- Reads video sequences from Objectron
- Extracts 2 consecutive frames per sequence
- Loads ground truth poses and intrinsics from JSON

### 2. Model Setup
- Loads pretrained AnyCam model
- **Deletes** the existing pose_head
- Creates a **fresh randomly initialized** pose_head
- **Freezes** all layers except the new pose_head

### 3. AnyCaLib Integration
- Runs AnyCaLib on input images to predict focal length
- Injects focal length directly (no candidate system)
- Single focal value per sequence

### 4. Training Loop
- Forward pass: Images → Depth → Flow → Pose prediction
- AnyCaLib provides focal length for projection
- Loss: Flow reprojection error
- Backward pass: Gradients only update pose_head
- Optimizer updates only pose_head weights

---

## Next Steps After Training

### 1. Analyze Results

```bash
# Check checkpoint file
python -c "import torch; ckpt = torch.load('experiments/pose_head_experiment_results/final_model.pt'); print(ckpt.keys())"

# Extract loss history (if you logged it)
# Plot loss curve to visualize convergence
```

### 2. Compare Focal Lengths

Run inference with the trained model and compare:
- AnyCaLib predictions
- Original AnyCam candidate system
- Ground truth focal lengths from Objectron

### 3. Visualize Predictions

Use the existing visualization scripts in `experiments/` to:
- Generate point clouds
- Visualize predicted vs. ground truth poses
- Check flow reprojection quality

---

## Files Modified/Created

- ✅ `experiments/ARCHITECTURE_FINDINGS.md` - Detailed architecture analysis
- ✅ `experiments/EXPERIMENT_QUICKSTART.md` - This guide
- ✅ `experiments/train_pose_head_anycalib.py` - Training script

**No core AnyCam files were modified** - all changes are isolated in the experiment script.

---

## Contact / Questions

This experiment was set up to test focal length prediction improvements for Kalman's master's thesis at TUM.

If you encounter issues:
1. Check the troubleshooting section above
2. Verify conda environment is activated
3. Ensure all paths in the script match your system
4. Check GPU memory with `nvidia-smi`

---

**Good luck with your experiment! 🚀**

