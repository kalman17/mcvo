# AnyCam Architecture Analysis & Training Experiment Documentation

**Date:** October 10, 2025  
**Branch:** `experiment/pose-head-retraining-anycalib-focal`  
**Goal:** Inject AnyCaLib focal length predictions into AnyCam's training pipeline to improve focal length estimation

---

## 1. AnyCam Architecture Overview

### 1.1 High-Level Pipeline (from `/anycam/trainer.py`)

The AnyCam training pipeline follows this flow:

```
Input Images → Depth Prediction → Flow Estimation → Pose Prediction → Flow Reprojection Loss
                    ↓                                        ↓
                  (frozen)                          (pose + focal candidates)
```

**Key Components:**

1. **Depth Predictor** (`self.depth_predictor`): Uses a pretrained depth model (frozen, line 272-273)
2. **Image Processor** (`self.image_processor`): Computes optical flow and occlusion maps
3. **Pose Predictor** (`self.pose_predictor`): The main AnyCam model that predicts:
   - Camera poses (rotation + translation)
   - Focal length via **candidate system**
   - Uncertainty maps
4. **Loss Function**: Flow reprojection loss comparing predicted vs. observed optical flow

### 1.2 Focal Length Prediction - Current System

Located in `/anycam/models/anycam.py`:

**Current Method: Focal Length Candidates (lines 444-488)**

The model predicts focal length using a **discrete candidate approach**:

```python
# Line 52-55: Configuration
self.focal_parameterization = "candidates"  # or "log-candidates", "linlog-candidates"
self.focal_min = 0.1
self.focal_max = 4.0
self.focal_num_candidates = 32  # Default: 32 discrete candidates

# Line 475-483: Candidate generation
focal_length_probs = F.softmax(focal_enc, dim=-1)  # Probability distribution over candidates
focal_candidates = torch.linspace(self.focal_min, self.focal_max, self.focal_num_candidates)
focal_length = torch.sum(focal_length_probs * focal_candidates, dim=-1)  # Weighted average
```

**How it works:**
1. The model outputs logits for 32 candidate focal lengths (evenly distributed between 0.1 and 4.0)
2. Softmax converts logits to probabilities
3. Final focal length = weighted average of candidates
4. During training, **all 32 candidates** are tested for flow reprojection (line 405 in trainer.py)
5. The best candidate (with lowest reprojection error) is selected

**Problem:** This is computationally expensive and may not capture the true focal length accurately.

### 1.3 Pose Head Architecture

Located in `/anycam/models/anycam.py` and `/anycam/models/anycam_blocks.py`:

**Pose Processing Pipeline:**

```python
# Lines 327-355: Feature extraction and attention
pose_tokens = self.pose_reassemble_stage(pose_tokens)       # Reassemble features from backbone
pose_tokens = self.pose_feature_fusion_stage(pose_tokens)   # Fuse multi-scale features
pose_token = pose_tokens[-1]

# Self-attention across frames
for i in range(self.self_att_depth):  # Default: 8 attention layers
    pose_token = self.pose_interframe_attention[i](pose_token)

# Sequence-level features
seq_token = self.sequence_token.expand(n, 1, -1)
seq_token = self.sequence_token_attention(seq_token, pose_token)

# Lines 168-176: Two separate heads
self.pose_head = AnyCamPoseTokenHead(...)           # Predicts rotation + translation
self.sequence_info_head = AnyCamPoseTokenHead(...)  # Predicts focal length + scaling features
```

**Key Insight:** There are **TWO separate prediction heads**:
1. `pose_head`: Predicts per-frame poses (rotation + translation) with 32 candidates
2. `sequence_info_head`: Predicts sequence-level information (focal length + scaling features)

**Pose Head Definition** (`anycam_blocks.py` lines 164-177):
```python
class AnyCamPoseTokenHead(nn.Module):
    def __init__(self, in_chn: int, out_chn: int):
        self.proj0 = nn.Linear(in_chn, in_chn // 2)
        self.activation0 = nn.ReLU()
        self.proj1 = nn.Linear(in_chn // 2, out_chn)
```

Simple 2-layer MLP: `[in_dim] → [in_dim/2] → [out_dim]`

---

## 2. AnyCaLib Integration Strategy

### 2.1 What is AnyCaLib?

AnyCaLib is a calibration method that predicts focal length from a **single image** by analyzing visual cues.

**Location:** `/home/kalman/TUM/thesis/anycam/anycalib/anycalib/model/anycalib_pretrained.py`

**Usage Pattern** (from `experiments/focal_pose_consistency.py`):
```python
from anycalib.model.anycalib_pretrained import AnyCalib

model = AnyCalib(model_id="anycalib_pinhole").to(device).eval()
pred = model.predict(img, cam_id="pinhole")
focal_px = pred["intrinsics"][0, 0]  # Extract fx in pixels
```

### 2.2 Integration Points

**Where to inject AnyCaLib predictions:**

The focal length is used in two critical places:

1. **During forward pass** (`trainer.py` line 379):
   ```python
   proj_candidates = make_proj_from_focal_length(focal_length_candidates, w/h)
   ```

2. **During loss computation** (`trainer.py` line 405):
   ```python
   induced_flow, dist = induce_flow_dist(aligned_depths, proj_candidates, poses, ...)
   ```

**Proposed Modification:**

Replace the candidate system with direct AnyCaLib predictions:
```python
# OLD: 32 candidates per sequence
focal_length_candidates = torch.linspace(0.1, 4.0, 32)  # Shape: [32]

# NEW: 1 focal length per frame from AnyCaLib
focal_length_anycalib = run_anycalib_on_frames(images)  # Shape: [batch, n_frames]
focal_length_avg = focal_length_anycalib.mean(dim=1)    # Average across frames (or use first frame only)
```

### 2.3 Two Approaches for Multi-Frame Sequences

**Option A: Single Frame Approach** (RECOMMENDED FOR INITIAL EXPERIMENT)
- Run AnyCaLib only on the **first frame** of each sequence
- Assume constant focal length across all frames (valid for fixed camera)
- **Pros:** Faster, simpler
- **Cons:** Doesn't account for potential focal length changes

**Option B: Multi-Frame Average** (FOR FUTURE)
- Run AnyCaLib on **all frames** in the sequence
- Average the predictions to get a single focal length
- **Pros:** More robust to per-frame noise
- **Cons:** Slower, requires careful batching

---

## 3. Training Experiment Design

### 3.1 Objectives

1. **Remove existing pose heads** and create a fresh `pose_head` with random initialization
2. **Freeze the backbone** (DINOv2 or CroCo) and all other components
3. **Inject AnyCaLib focal length** instead of using the candidate system
4. **Train only the new pose head** to overfit on a small subset of Objectron sequences
5. **Verify learning** by monitoring loss decrease

### 3.2 Dataset

**Objectron Dataset:**
- Location: `/home/kalman/TUM/thesis/Objectron/`
- Videos: `/home/kalman/TUM/thesis/Objectron/videos/` (100 sequences)
- Ground Truth: `/home/kalman/TUM/thesis/Objectron/processed_gt/` (101 JSON files)
- Format: MOV videos + JSON files with camera poses and intrinsics

**Training Configuration:**
- Sequences: 100 (all available)
- Frames per sequence: 2 (for initial experiment)
- Batch size: 1-2 (depending on GPU memory)
- Goal: Overfit to demonstrate learning capability

### 3.3 Architectural Modifications

**Location:** `/anycam/models/anycam.py`

```python
# 1. DISABLE focal length candidate system
self.use_anycalib_focal = True  # NEW FLAG

# 2. In forward() method (line 385):
if self.use_anycalib_focal:
    # ===== ANYCALIB INJECTION POINT =====
    # Run AnyCaLib on all frames
    focal_lengths_per_frame = []
    for i in range(f):  # f = number of frames
        img_np = images[:, i].cpu().numpy().transpose(0, 2, 3, 1)  # [B, H, W, 3]
        focal_pred = self.anycalib_model.infer(img_np)
        focal_lengths_per_frame.append(focal_pred)
    
    focal_length = torch.stack(focal_lengths_per_frame, dim=1).mean(dim=1)  # [B]
    focal_candidates = focal_length.unsqueeze(1)  # [B, 1] - single candidate
    focal_length_probs = torch.ones_like(focal_candidates)  # Dummy probabilities
    # ===== END ANYCALIB INJECTION =====
else:
    # Original candidate system
    focal_length, focal_length_probs, focal_candidates = self.enc_embed_to_focal(focal_enc)
```

**Location:** `/anycam/trainer.py` (AnyCamWrapper)

```python
# 3. FREEZE everything except pose_head
def freeze_for_pose_training(self):
    """Freeze all parameters except the pose head."""
    for param in self.parameters():
        param.requires_grad = False
    
    # Only unfreeze the pose head
    for param in self.pose_predictor.pose_head.parameters():
        param.requires_grad = True
    
    print("[FREEZE] All layers frozen except pose_head")
    trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
    total = sum(p.numel() for p in self.parameters())
    print(f"[PARAMS] Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
```

### 3.4 Training Script Structure

**Location:** `/home/kalman/TUM/thesis/anycam/experiments/train_pose_head_experiment.py`

```python
"""
Pose Head Retraining Experiment with AnyCaLib Focal Length Injection
====================================================================

This script tests whether we can successfully train a fresh pose head
while using AnyCaLib for focal length prediction instead of the candidate system.

Experiment Steps:
1. Load AnyCam model with pretrained weights
2. Replace pose_head with freshly initialized version
3. Freeze all layers except the new pose_head
4. Inject AnyCaLib focal length predictions
5. Train on Objectron dataset (100 sequences, 2 frames each)
6. Monitor loss convergence to verify learning
"""
```

---

## 4. Implementation Checklist

- [x] Understand AnyCam architecture
- [x] Locate focal length prediction mechanism
- [x] Locate pose head architecture
- [x] Create git branch: `experiment/pose-head-retraining-anycalib-focal`
- [x] Verify Objectron dataset (100 sequences ✓)
- [ ] Create modified AnyCam model with AnyCaLib injection
- [ ] Create training script with proper freezing
- [ ] Add comprehensive logging and checkpointing
- [ ] Run training experiment
- [ ] Analyze results and document findings

---

## 5. Key Files to Modify

1. **`/anycam/models/anycam.py`**: Add AnyCaLib integration
2. **`/anycam/trainer.py`**: Modify AnyCamWrapper to support single focal candidate
3. **`/experiments/train_pose_head_experiment.py`**: New training script (TO CREATE)

---

## 6. Expected Challenges

1. **AnyCaLib Integration:** Need to handle batching and GPU/CPU transfers
2. **Flow Reprojection:** Must ensure focal length shape matches expected dimensions
3. **Memory Management:** AnyCaLib + AnyCam in same pipeline may be memory-intensive
4. **Gradient Flow:** Ensure gradients only flow to pose_head
5. **Dataset Loading:** Need to create proper PyTorch dataset for Objectron videos

---

## 7. Success Criteria

✅ **Experiment is successful if:**
1. Model loads without errors
2. Only pose_head parameters have `requires_grad=True`
3. AnyCaLib focal lengths are correctly injected
4. Loss decreases over training iterations
5. Model can overfit on small training set

---

## 8. Next Steps

1. Create training script with all modifications
2. Test on 1-2 sequences first
3. Scale up to full 100 sequences
4. Analyze trained pose head weights
5. Compare focal length predictions: AnyCaLib vs. original candidates

---

**Author:** AI Assistant  
**Human Supervisor:** Kalman (Master's Thesis)  
**Institution:** TUM (Technical University of Munich)

