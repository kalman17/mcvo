# Feature Aggregation Transformer (FAT) Integration

This directory contains experiments for the FAT architecture that aggregates multi-frame DINOv2 features for camera calibration.

## Architecture Overview

FAT operates **between** AnyCalib's Step 1 (DINOv2 feature extraction) and Step 2 (DPT decoder + ray transformation).

**Key difference from DA3**:
- **DA3**: Works on AnyCalib's **final scalar outputs** [fx, fy, cx, cy]
- **FAT**: Works on AnyCalib's **intermediate spatial features** [1024, H/14, W/14]

### Data Flow

**V3 Architecture (Phases 1-2) - Reprojection Loss**:

```
Input: N images [N, 3, H, W]
       ↓
┌─────────────────────────────────────────────────────────────────┐
│ STEP 1: DINOv2 Backbone (FROZEN)                                │
│         Output: [N, 1024, H/14, W/14] × 4 scales                │
└─────────────────────────────────────────────────────────────────┘
       ↓
┌─────────────────────────────────────────────────────────────────┐
│ FAT: Feature Aggregation Transformer (TRAINABLE)                │
│      + Optional visual tokens [N, 384] from DINOv2-small        │
│      Process: Aggregate across N frames per spatial position    │
│      Output: [1, 1024, H/14, W/14] × 4 scales                   │
└─────────────────────────────────────────────────────────────────┘
       ↓
┌─────────────────────────────────────────────────────────────────┐
│ STEP 2: DPT Decoder + Ray Head (FROZEN)                         │
│         Output: rays [1, 3, H_ray, W_ray]                       │
│         Gradients flow through frozen layers                    │
└─────────────────────────────────────────────────────────────────┘
       ↓
┌─────────────────────────────────────────────────────────────────┐
│ TRAINING (V3): Reprojection Loss                                │
│         1. Get per-frame AnyCalib predictions [N, 4]           │
│         2. Average intrinsics: mean(fx, fy, cx, cy) → [4]      │
│         3. Scale intrinsics to ray resolution                   │
│         4. Project rays: u = fx*(rx/rz)+cx, v = fy*(ry/rz)+cy  │
│         5. Loss: MSE(projected_coords, pixel_grid)               │
│         Backpropagates through projection to rays to FAT        │
└─────────────────────────────────────────────────────────────────┘
```

**V2 Architecture (Phase 3) - WLS Loss**:

```
[Same forward pass as above]
       ↓
┌─────────────────────────────────────────────────────────────────┐
│ STEP 3: RANSAC Calibrator (NON-DIFFERENTIABLE)                  │
│         Identifies inliers via RANSAC (~150ms)                  │
│         Output: intrinsics [4], inlier_mask [H*W]               │
└─────────────────────────────────────────────────────────────────┘
       ↓
┌─────────────────────────────────────────────────────────────────┐
│ TRAINING (V2): Differentiable Weighted Least Squares           │
│         Uses RANSAC inliers as CONSTANT hard weights            │
│         Solves for intrinsics via WLS (differentiable)          │
│         Loss: MSE(WLS_intrinsics, RANSAC_intrinsics)            │
│         Backpropagates through WLS solution to rays to FAT      │
└─────────────────────────────────────────────────────────────────┘
```

---

## The Non-Differentiability Problem

### Problem: RANSAC is Non-Differentiable

The AnyCalib calibrator uses **RANSAC (Random Sample Consensus)** + **Gauss-Newton optimization**. These are classical algorithms that cannot provide gradients:

```python
# This FAILS:
intrinsics = calibrator(rays)  # No grad_fn!
loss = mse_loss(intrinsics, gt_intrinsics)
loss.backward()  # ERROR: "element 0 does not require grad"
```

**Why non-differentiable?**
- Random sampling (no gradient for random selection)
- Conditional logic (`if error < threshold` creates discontinuities)
- Argmax operation (choosing best candidate)
- LU factorization (iterative, not differentiable)

### Evolution of Solutions

#### V1 Attempt: Soft Exponential Weights (FAILED)

**Initial approach**: Compute ray consistency loss with soft exponential weights:
```python
# V1 (FAILED): Soft weighting on ray loss
angular_error = acos((predicted_rays * expected_rays).sum(dim=-1))
weights = exp(-(angular_error² / (2*sigma²)))  # Soft weights
loss = (weights * ray_errors).sum() / weights.sum()
```

**Problem observed**: NaN gradients appeared in ALL FAT transformer layers from batch 0:
```
[Batch 0] Loss: 2.574564e-03 (valid)
[Batch 0] Rays: normalized, no NaN
[Batch 0] Soft weights: mean=0.308821, no NaN
[Batch 0] FAT gradients: ALL NaN
```

**Root cause**: The soft weighting approach had gradient flow through the weights themselves. When combined with mixed precision training (FP16), the exponential computation and its gradients caused numerical instability in the transformer attention layers.

**Attempted fixes that failed**:
1. Moving loss computation to FP32
2. Disabling autocast for FAT module entirely
3. Gradient clipping

None resolved the fundamental issue: gradients through the soft weights were numerically unstable.

#### V2 Solution: Implicit Differentiation

**Key insight**: Don't try to differentiate through the weight computation. Instead, treat RANSAC's inlier determination as a **constant** and differentiate only through the least-squares solution.

#### V3 Solution: Reprojection Loss (CURRENT for Phases 1-2)

**Key insight**: Instead of matching intrinsics, directly optimize ray predictions to match expected pixel coordinates. This provides more direct supervision without dependency on RANSAC for loss computation.

**Method**:
1. Get per-frame AnyCalib predictions (fx_i, fy_i, cx_i, cy_i) for N frames using `forward_single_frame()`
2. Compute average intrinsics: `fx_avg = mean(fx_i)`, `fy_avg = mean(fy_i)`, `cx_avg = mean(cx_i)`, `cy_avg = mean(cy_i)`
3. Scale intrinsics to ray resolution (accounting for DINOv2 padding):
   - `scale_x = W_ray / W_orig`, `scale_y = H_ray / H_orig`
   - `fx_down = fx_avg * scale_x`, `fy_down = fy_avg * scale_y`
   - `cx_down = cx_avg * scale_x`, `cy_down = cy_avg * scale_y`
4. Project predicted rays back to image plane:
   - `u_projected = fx_down * (rx/rz) + cx_down`
   - `v_projected = fy_down * (ry/rz) + cy_down`
5. Create pixel grid: `T[y, x] = (x, y)` for all pixels at ray resolution
6. Compute MSE loss: `loss = MSE([u_projected, v_projected], [x_actual, y_actual])`

**Gradient flow**:
```
loss = MSE(projected_coords, pixel_grid)
  ↓ ∂loss/∂projected_coords
projected_coords [u, v] = fx*(rx/rz)+cx, fy*(ry/rz)+cy
  ↓ ∂projected/∂rays (chain rule through division)
predicted_rays [rx, ry, rz]
  ↓ (backprop through frozen DPT, Ray Head)
FAT parameters (UPDATED!)
```

**Advantages**:
- Direct supervision: optimizes rays to match pixel coordinates
- No RANSAC dependency: loss computation doesn't require RANSAC (only for per-frame AnyCalib)
- More stable: avoids numerical issues from WLS matrix inversion
- Faster: no need to run differentiable WLS solver
- Simpler: direct projection vs solving linear system

---

## V3 Training Implementation (Phases 1-2)

### Training Loop

```python
for batch in dataloader:
    optimizer.zero_grad()
    
    # 1. Forward: Get rays (WITH GRADIENTS!)
    result = model.forward_with_differentiable_calibration(images, ...)
    rays = result['rays']  # [1, H*W, 3] - has grad_fn!
    image_size = result['image_size']  # (H_ray, W_ray)
    
    # 2. Get per-frame AnyCalib predictions (no gradients)
    with torch.no_grad():
        per_frame_intrinsics = model.get_per_frame_intrinsics(images)  # [N, 4]
    
    # 3. Compute average intrinsics
    average_intrinsics = per_frame_intrinsics.mean(dim=0)  # [4]
    
    # 4. Get original image size (before padding)
    H_orig, W_orig = images.shape[2], images.shape[3]
    
    # 5. Compute reprojection loss
    loss, loss_info = model.compute_reprojection_loss(
        predicted_rays=rays[0],  # [H*W, 3] - WITH gradients
        average_intrinsics=average_intrinsics,  # [4] - detached
        ray_image_size=image_size,  # (H_ray, W_ray)
        original_image_size=(H_orig, W_orig),  # (H_orig, W_orig)
    )
    
    # 6. Backpropagate (gradients flow through projection to rays to FAT!)
    loss.backward()
    optimizer.step()
```

### Why V3 Works

| Aspect | V2 (WLS vs RANSAC) | V3 (Reprojection) |
|--------|-------------------|-------------------|
| **Loss computation** | MSE(WLS_intrinsics, RANSAC_intrinsics) | MSE(projected_coords, pixel_grid) |
| **Requires RANSAC** | Yes (for inlier mask and reference) | No (only for per-frame AnyCalib) |
| **Matrix operations** | WLS solve (matrix inversion) | Direct projection (multiplication only) |
| **Numerical stability** | Good (with damping) | Excellent (no inversion) |
| **Speed** | Slower (WLS solve) | Faster (direct computation) |
| **Supervision** | Indirect (via intrinsics) | Direct (pixel coordinates) |

**Key difference**: V3 provides more direct supervision by optimizing rays to match pixel coordinates directly, rather than optimizing intrinsics and then comparing them. This eliminates the need for matrix inversion and provides a cleaner gradient path.

---

## V2 Training Implementation (Phase 3)

### Mathematical Formulation

For a pinhole camera, we solve for parameters [p, q, r, s] where:
- p = 1/fx, q = cx/fx
- r = 1/fy, s = cy/fy

The constraint equations for pixel (u, v) with ray (rx, ry, rz):
```
rx/rz = u*p - q   →  [u, -1, 0, 0] @ [p,q,r,s] = rx/rz
ry/rz = v*r - s   →  [0, 0, v, -1] @ [p,q,r,s] = ry/rz
```

This forms a weighted least squares problem:
```
x = argmin_x ||W(Ax - b)||²
```

Where:
- **A** = design matrix from pixel coordinates (CONSTANT, no gradients)
- **b** = tangent coordinates from rays [rx/rz, ry/rz] (DIFFERENTIABLE w.r.t. rays)
- **W** = diagonal weight matrix: 1.0 for RANSAC inliers, 1e-6 for outliers (CONSTANT, no gradients)
- **x** = [p, q, r, s] solution (DIFFERENTIABLE w.r.t. b)

**Closed-form solution**:
```
x = (A^T W² A)^{-1} A^T W² b
```

**This is differentiable w.r.t. b** because:
- A and W are constants (no grad_fn needed)
- Matrix multiplication and inversion are differentiable operations in PyTorch
- `torch.linalg.solve` provides gradients automatically

**Gradient flow**:
```
loss → WLS_intrinsics → params [p,q,r,s] → tangent_coords (b) → rays → FAT
                                            ↑
                                      Differentiable!
```

### Training Loop

```python
for batch in dataloader:
    # 1. Forward: Get rays (WITH GRADIENTS!)
    rays = model(images)  # [B, H*W, 3] - has grad_fn!

    # 2. Run RANSAC calibrator (no gradients, ~150ms)
    with torch.no_grad():
        ransac_result = calibrator(rays.detach())
        ransac_intrinsics = ransac_result['intrinsics']  # [4]
        inlier_mask = ransac_result['inlier_mask']  # [H*W] boolean

    # 3. Solve differentiable weighted least squares
    #    - A: design matrix from pixel coords (constant)
    #    - b: tangent coords from rays (differentiable)
    #    - W: hard weights from RANSAC inliers (constant)
    wls_result = differentiable_calibrator(
        predicted_rays=rays,  # Has gradients!
        image_size=(H, W),
        inlier_mask=inlier_mask.detach(),  # Constant weights
    )
    wls_intrinsics = wls_result['intrinsics']  # [4] - differentiable!

    # 4. Loss: Match WLS solution to RANSAC solution
    loss = mse_loss(wls_intrinsics, ransac_intrinsics.detach())
    loss += regularization_lambda * mse_loss(wls_intrinsics, ransac_intrinsics.detach())

    # 5. Backpropagate (gradients flow through WLS to rays to FAT!)
    loss.backward()
    optimizer.step()
```

### Why V2 Works

| Aspect | V1 (Soft Weights) | V2 (Implicit Differentiation) |
|--------|-------------------|-------------------------------|
| **Weight gradients** | Required (caused NaN) | Not required (constant) |
| **Loss computation** | On rays directly | On intrinsics via WLS |
| **Numerical stability** | Poor (exp in gradient) | Good (linear algebra only) |
| **Mixed precision** | Incompatible | Compatible (WLS in FP32) |

**Key difference**: In V2, gradients never flow through the weight computation. The weights are treated as constants determined by RANSAC, and gradients only flow through the linear algebra of the WLS solution.

---

## Directory Structure

```
final_training_phases/
├── README.md                      # This file
├── ARCHITECTURE_DETAILED.md       # Deep architectural explanation
├── phase1_training/               # Phase 1 v1 (old)
├── phase1_training_v2/            # Phase 1 v2 (WLS loss)
├── phase1_training_v3/            # Phase 1 v3 (Reprojection loss) - CURRENT
├── phase2_training/               # Phase 2 v1 (WLS loss)
├── phase2_training_v2/            # Phase 2 v2 (Reprojection loss) - CURRENT
└── phase3_training/               # Phase 3 (V2 WLS loss) - Future
```

---

## Loss Evolution

| Version | Phase 1 | Phase 2 | Phase 3 | Description |
|---------|---------|---------|---------|-------------|
| **V1** | v1 | - | - | Soft exponential weights - FAILED (NaN gradients from exp/acos gradient path) |
| **V2** | v2 | v1 | WLS (abandoned) | WLS intrinsics vs RANSAC intrinsics (implicit differentiation through WLS) |
| **V3** | v3 | v2 | **V2 Combined** | Reprojection loss (Phases 1-2), Combined loss (Phase 3 V2) |
| **Current** | v3 | v2 (overfits) | **V2 Combined** | Phase 3 V2 uses combined flow + calibration anchor |

**V3 Loss (Phases 1-2)**: Projects predicted rays back to image plane using average per-frame AnyCalib intrinsics, then computes MSE between projected pixel coordinates and actual pixel grid. This provides more direct supervision without dependency on RANSAC for loss computation. No matrix inversion required (faster and more stable than V2 WLS).

**Phase 3 V2 Combined Loss (CURRENT)**:
```
Total = λ_flow * L_flow + λ_calib * L_calib
- L_flow: AnyCam flow reprojection (induced vs observed flow)
- L_calib: Phase 1 reprojection loss (rays → pixel coords)
- λ_flow = 1.0 (fixed), λ_calib = 1e-4 (tunable)
```

Self-supervised end-to-end training with calibration stability anchor.

#### Historical Note: Abandoned Phase 3 Approaches

**V2 WLS Approach (Planned but Not Implemented)**:
- Intended to use differentiable WLS calibrator (implicit differentiation)
- Would train FAT to match WLS solution to RANSAC solution
- **Abandoned because**: Phase 1 V3 reprojection loss worked better (no matrix inversion, faster, more stable)
- **Kept V2 WLS code** in `experiments/models/differentiable_calibrator.py` for reference

**Original Phase 3 with Alternating Training (Abandoned)**:
- Load Phase 2 checkpoint (visual tokens)
- Alternate between FAT and Pose Head training
- Pure flow reprojection loss (no calibration anchor)
- **Abandoned because**: Phase 2 overfitting + no stability for calibration
- **Phase 3 V2 replaces this** with: Skip Phase 2, train together, add calibration anchor

---

## Training Phases

### Phase 1: Feature Aggregation Pre-training (V3)

**Objective**: Learn to aggregate multi-frame features to produce geometrically consistent rays.

| Setting | Value | Reasoning |
|---------|-------|-----------|
| **Trainable** | FAT only (25M params) | Focus on aggregation |
| **Frozen** | DINOv2, DPT, Ray Head | Use pretrained components |
| **Visual tokens** | Disabled | Isolate aggregation learning |
| **Loss** | Reprojection loss (V3) | Direct supervision via pixel reprojection |
| **LR** | 5e-5 | Reduced for stability |
| **Batch size** | 2 | Memory safety (24GB VRAM) |
| **Data** | Step=2 (half frames) | Faster training, maintains video diversity |

**Command**:
```bash
python experiments/train_fat_calibration.py --phase 1 --v2 \
    --objectron_videos /data/thesis/Objectron/videos \
    --objectron_gt /data/thesis/Objectron/processed_gt \
    --max_ahead 3 \
    --num_epochs 50 \
    --batch_size 2 \
    --learning_rate 5e-5 \
    --save_dir experiments/final_training_phases/phase1_training_v3
```

### Phase 2: Visual-Conditioned Aggregation (V2)

**Objective**: Leverage visual tokens (DINOv2-small CLS) for context-aware aggregation.

| Setting | Value | Changes from Phase 1 |
|---------|-------|----------------------|
| **Trainable** | FAT (including visual projection) | Add visual layers |
| **Visual tokens** | Enabled (384-dim from DINOv2-small) | Scene context |
| **Loss** | Reprojection loss (V3) | Same as Phase 1 V3 |
| **LR** | 2.5e-5 | Lower for fine-tuning |
| **Checkpoint** | Load Phase 1 v3 weights | Warm start |
| **Data** | Step=2 (half frames) | Same as Phase 1 |

**Why DINOv2-small via HuggingFace?**
- RTX 5090 has compute capability 12.0
- Original `torch.hub` DINOv2 uses xFormers (only supports ≤9.0)
- HuggingFace version uses native PyTorch attention (compatible with all GPUs)

**Command**:
```bash
python experiments/train_fat_calibration.py --phase 2 --v2 \
    --phase1_checkpoint experiments/final_training_phases/phase1_training_v3/checkpoints/latest_checkpoint.pt \
    --objectron_videos /data/thesis/Objectron/videos \
    --objectron_gt /data/thesis/Objectron/processed_gt \
    --use_visual_conditioning \
    --max_ahead 3 \
    --num_epochs 10 \
    --batch_size 2 \
    --learning_rate 2.5e-5 \
    --save_dir experiments/final_training_phases/phase2_training_v2
```

### Phase 3 V2: Combined Loss Training (Skip Phase 2) - CURRENT IMPLEMENTATION

**Objective**: Train FAT + Pose Head together with combined loss for end-to-end calibration and pose estimation.

**Key Insight**: Phase 2 showed overfitting with visual tokens (val loss exploding: 8.61 → 14.37 → 19.10 while train loss decreased). Phase 3 V2 skips Phase 2 entirely and loads Phase 1 checkpoint directly, avoiding visual conditioning issues.

#### Architecture Overview

```
Input: [B, N, 3, H, W]  (N = max_ahead + 1 frames)
  ↓
┌─────────────────────────────────────────────────────────┐
│ STEP 1: FAT-Enhanced AnyCalib (TRAINABLE)             │
│   DINOv2: Extract features [B*N, 1024, h, w] × 4      │
│   FAT: Aggregate across N frames → [B, 1024, h, w] × 4│
│   DPT + Ray Head: → rays [B, H*W, 3] WITH GRADIENTS   │
│   Calibrator: → intrinsics [B, 4]                      │
│   Extract fx: [B, 4] → [B]                             │
└─────────────────────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────────────────────┐
│ STEP 2: Depth Predictor (FROZEN)                       │
│   [B, N, 3, H, W] → [B, N, 1, H, W]                    │
└─────────────────────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────────────────────┐
│ STEP 3: Flow Processor (FROZEN)                        │
│   [B, N, 3, H, W] → [B, N, 3, H, W] (flow + occlusion) │
└─────────────────────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────────────────────┐
│ STEP 4: Pose Head (TRAINABLE, RANDOM INIT)             │
│   Input: depths, flows, focal_length [B]               │
│   Output: poses [B, N, 4, 4]                            │
└─────────────────────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────────────────────┐
│ STEP 5: Combined Loss Computation                      │
│   L_flow: Flow reprojection (PRIMARY)                   │
│   L_calib: Calibration reprojection (ANCHOR)            │
│   Total: λ_flow * L_flow + λ_calib * L_calib           │
└─────────────────────────────────────────────────────────┘
```

#### Training Configuration

| Setting | Value | Reasoning |
|---------|-------|-----------|
| **Trainable** | FAT + Pose Head | Both trained together (no alternating) |
| **Checkpoint** | Phase 1 only | Visual conditioning not needed (Phase 2 overfits) |
| **Loss** | Flow + Calibration anchor | Prevents calibration explosion during pose learning |
| **GradScaler** | Enabled | Proper mixed precision training (FP16/FP32) |
| **Lambda flow** | 1.0 (fixed) | Primary pose learning signal |
| **Lambda calib** | 1e-4 (tunable) | Start small, adjust empirically based on benchmarks |
| **Weight decay** | 1e-4 | L2 regularization on trainable weights |
| **Learning rate** | 1e-5 | Conservative for joint training stability |
| **Optimizer** | AdamW | Includes weight decay |
| **Pose head init** | Random | Train from scratch alongside FAT (not pretrained) |
| **Batch size** | 2 | Memory safety for 24GB VRAM |
| **Max ahead** | 3 | 4-frame sequences (optimal from Exp 2) |

#### Combined Loss Design

```python
Total Loss = lambda_flow * L_flow + lambda_calib * L_calib

Where:
- L_flow: AnyCam flow reprojection loss (PRIMARY - for pose learning)
  → Compares induced flow (from poses + depths + focal) vs observed flow
  → Trains pose head to predict geometrically consistent poses

- L_calib: Phase 1 reprojection loss (ANCHOR - prevents calibration explosion)
  → Projects FAT rays back to image plane using average per-frame intrinsics
  → Prevents FAT from producing extreme calibrations that minimize flow but diverge
  → NOT meant to constrain to mean (mean is imperfect), only to prevent explosion
```

**Critical Insight**: Without the calibration anchor, FAT could produce intrinsics that minimize flow error but are geometrically unreasonable (e.g., extremely high focal lengths). The small λ_calib (1e-4) provides a soft constraint without dominating the loss.

#### Gradient Flow

**Trainable Path**:
```
loss (scalar)
  ↓ ∂loss/∂L_flow
induced_flow [B, N, 2, H, W]
  ↓ ∂induced/∂poses, ∂induced/∂fx
poses [B, N, 4, 4] ← Pose Head (TRAINABLE - random init)
fx [B] ← FAT intrinsics (TRAINABLE)
  ↓ ∂fx/∂intrinsics → ∂intrinsics/∂rays → ∂rays/∂FAT
FAT parameters (TRAINABLE)

  ↓ ∂loss/∂L_calib
projected_coords (from FAT rays) → FAT parameters (TRAINABLE)
```

**Key Point**: Gradients flow through FAT in two paths:
1. Via focal length → flow reprojection (pose learning signal)
2. Via rays → calibration reprojection (stability anchor)

#### Training Commands

**Standard Training (local, max_ahead=3, batch_size=2)**:
```bash
python experiments/train_fat_calibration.py --phase 3 \
    --phase1_checkpoint_for_phase3 experiments/final_training_phases/phase1_training_v3/checkpoints/latest_checkpoint.pt \
    --objectron_videos /data/thesis/Objectron/videos \
    --objectron_gt /data/thesis/Objectron/processed_gt \
    --anycam_config pretrained_models/anycam_seq8/training_config.yaml \
    --baseline_checkpoint pretrained_models/anycam_seq8/training_checkpoint_247500.pt \
    --max_ahead 3 \
    --num_epochs 10 \
    --batch_size 1 \
    --learning_rate 1e-5 \
    --weight_decay 1e-4 \
    --lambda_calib 1e-4 \
    --benchmark_calibration_samples 50 \
    --benchmark_pose_samples 50 \
    --save_dir experiments/final_training_phases/phase3_training_v2
```

**test pre-cluster Training (max_ahead=2, batch_size=3)**:
```bash
python experiments/train_fat_calibration.py --phase 3 \
    --phase1_checkpoint_for_phase3 experiments/final_training_phases/phase1_training_v3/checkpoints/latest_checkpoint.pt \
    --objectron_videos /data/thesis/Objectron/videos \
    --objectron_gt /data/thesis/Objectron/processed_gt \
    --anycam_config pretrained_models/anycam_seq8/training_config.yaml \
    --baseline_checkpoint pretrained_models/anycam_seq8/training_checkpoint_247500.pt \
    --max_ahead 1 \
    --batch_size 3 \
    --learning_rate 1e-5 \
    --lambda_calib 1e-4 \
    --save_dir experiments/final_training_phases/phase3_test
```

#### Iterative Tuning Strategy

Phase 3 V2 training is **fully self-supervised** but uses **passive monitoring with GT** to tune hyperparameters:

1. **Run 1-epoch experiment** with lambda_calib=1e-4
2. **Monitor benchmarks**:
   - Calibration: Compare FAT intrinsics vs GT intrinsics (relative error)
   - Pose: Compare predicted poses vs GT poses (rotation/translation error)
3. **Diagnose issues**:
   - If calibration is **diverging** (increasing error) → increase lambda_calib (e.g., 1e-3, 0.01, 0.15)
   - If calibration is **too constrained** (not improving) → decrease lambda_calib
   - If flow loss dominates → both components learning (good!)
4. **Re-run experiment** with adjusted weight
5. **Goal**: Stable calibration that improves alongside pose learning (not perfect convergence)

**Important Notes**:
- GT is **never used during training** (fully self-supervised)
- Benchmarks are for **passive monitoring only** (guide hyperparameter tuning)
- The mean calibration (from per-frame AnyCalib) is not perfect, so we don't expect L_calib → 0
- We want calibration **stable and reasonable**, not necessarily matching the mean exactly

#### Benchmarking (Per-Epoch Monitoring)

**Calibration Benchmarking**:
- Compares FAT intrinsics vs GT intrinsics (Objectron dataset)
- Metrics: Relative error for fx, fy, cx, cy
- Uses fixed test samples (no cycling between epochs)
- Saved to: `calibration_benchmark_history.json`, `calibration_benchmark_curve.png`

**Pose Benchmarking**:
- Compares predicted poses vs GT poses (Objectron dataset)
- Compares FAT model vs AnyCam baseline (32 focal candidates)
- Metrics: Rotation error (degrees), translation direction error (degrees)
- Uses fixed test samples (no cycling between epochs)
- Saved to: `pose_benchmark_history.json`, `pose_benchmark_curve.png`

**Loss Tracking**:
- Separate plots for total loss, flow loss, calibration loss
- Shows contribution of each component
- Saved to: `loss_curves_phase3.png`

#### Historical Context: Evolution to Phase 3 V2

**Original Phase 3 Plan (Abandoned)**:
- Load Phase 2 checkpoint (with visual tokens)
- Use alternating training (FAT → Pose Head → FAT → ...)
- Pure flow reprojection loss

**Why Changed**:
- Phase 2 validation loss exploded (8.61 → 14.37 → 19.10) while train loss decreased
- Visual conditioning (DINOv2-small CLS tokens) caused overfitting
- Alternating training added unnecessary complexity
- No calibration anchor risked divergence

**Phase 3 V2 Design Decisions**:
1. **Skip Phase 2**: Avoid visual token overfitting, use Phase 1 checkpoint directly
2. **Train together**: Simpler than alternating, both components learn jointly
3. **Add calibration anchor**: Prevents calibration explosion with small weight (1e-4)
4. **Random pose init**: Train pose head from scratch alongside FAT (not baseline weights)
5. **GradScaler**: Proper mixed precision (FP16 for frozen, FP32 for trainable)
6. **Iterative tuning**: Use benchmarks to guide lambda_calib adjustment

---

## Implementation Details

### Differentiable Calibrator

The core V2 component is `DifferentiableCalibrator` in `experiments/models/differentiable_calibrator.py`:

```python
class DifferentiableCalibrator(nn.Module):
    """
    Differentiable pinhole camera calibrator using implicit differentiation.

    Key insight: The solution of weighted least squares is differentiable
    w.r.t. the target vector (b), even when weights (W) are constant.
    """

    def solve_weighted_lstsq(self, A, b, W):
        """
        Solve: x = argmin ||W(Ax - b)||²
        Solution: x = (A^T W² A)^{-1} A^T W² b

        Differentiable w.r.t. b (target vector from rays).
        """
        W_sq = W * W
        AW = A * W_sq.unsqueeze(1)
        AtWA = A.t() @ AW + self.damping * I  # Regularization for stability
        AtWb = A.t() @ (W_sq * b)
        return torch.linalg.solve(AtWA, AtWb)
```

### Mixed Precision Strategy

FAT runs in **FP32** to avoid numerical issues, while DINOv2/DPT run in FP16:

```python
# DINOv2 backbone (FP16 for speed)
with autocast(device_type='cuda', dtype=torch.float16):
    features = backbone(images)

# FAT aggregation (FP32 for stability)
with torch.amp.autocast(enabled=False):
    features_fp32 = [f.float() for f in features]
    aggregated = fat(features_fp32)

# Calibrator (FP32 required)
with torch.amp.autocast(enabled=False):
    rays_fp32 = rays.float()
    intrinsics = calibrator(rays_fp32)
```

### Gradient Flow Through Frozen Layers

```python
# CORRECT: Freeze parameters, allow gradient flow
frozen_layer.requires_grad_(False)  # Parameters won't update
features = frozen_layer(x)  # But gradients flow through activations

# WRONG: Blocks all gradients
with torch.no_grad():
    features = frozen_layer(x)  # No gradients at all!
```

---

## Key Flags

### Architecture

| Flag | Default | Description |
|------|---------|-------------|
| `--embed_dim` | 1024 | DINOv2 feature dimension |
| `--num_heads` | 8 | Attention heads in FAT |
| `--num_layers` | 2 | Transformer layers |
| `--use_visual_conditioning` | Phase-dependent | Enable visual tokens |

### Training (V2)

| Flag | Description |
|------|-------------|
| `--v2` | **Required**: Use V2 implicit differentiation approach |
| `--regularization_lambda` | Weight for regularization term (default: 0.1) |
| `--max_ahead 3` | 4-frame sequences (N=max_ahead+1) |
| `--batch_size 2` | Memory safety (24GB VRAM) |
| `--learning_rate 5e-5` | Reduced for stability |
| `--resume <path>` | Resume from checkpoint |

---

## Memory Management

**GPU Breakdown** (RTX 5090 Laptop, 24GB):
- Model weights: ~1.4 GB
- Activations (batch=2): ~14 GB
- Optimizer state: ~3 GB
- RANSAC workspace: ~0.5 GB
- **Peak: ~19 GB / 24 GB** (safe margin)

**Safety measures**:
- Batch size 2 (not 4)
- FAT in FP32, backbone in FP16
- Periodic cache clearing every 50 batches

---

## Theoretical Contributions

### 1. Implicit Differentiation Through RANSAC-Guided Least Squares

**Novel approach**: Train neural networks using non-differentiable RANSAC by:
1. Using RANSAC only for inlier identification (no gradients needed)
2. Formulating calibration as weighted least squares with constant weights
3. Differentiating through the WLS closed-form solution

**General pattern**:
```
Non-diff Algorithm → Constant Selection/Weights → Differentiable Optimization → Gradients
```

### 2. Feature-Level vs Output-Level Aggregation

| Level | DA3 | FAT |
|-------|-----|-----|
| **Input** | 4D scalars | 1024D features |
| **Info preserved** | Low | High (spatial) |
| **Calibration-aware** | No | Yes (fine-tuned features) |

**Hypothesis**: Feature-level aggregation captures richer multi-frame relationships.

---

## Computational Cost

**Per Epoch** (batch_size=2, max_ahead=3, step=2):
- ~3,993 train batches (half the frames, step=2)
- ~1.5s per batch (including ~150ms RANSAC)
- **~2 hours per epoch**

**Full Training** (50 epochs):
- **~100 hours (4-5 days)**

---

## Implementation Files

| File | Description | Used In |
|------|-------------|---------|
| `experiments/models/feature_aggregation_transformer.py` | FAT transformer module | All phases |
| `experiments/models/anycalib_with_fat.py` | AnyCalib + FAT integration | All phases |
| `experiments/models/anycam_wrapper_fat.py` | **Phase 3 V2**: Full AnyCam pipeline wrapper | Phase 3 only |
| `experiments/models/differentiable_calibrator.py` | V2 WLS solver (kept for reference, not used) | None (abandoned) |
| `experiments/train_fat_calibration.py` | Unified training script (all 3 phases) | All phases |

**Key methods in `anycalib_with_fat.py`**:
- `forward()`: Standard FAT pipeline (Phase 1-2)
- `get_per_frame_intrinsics()`: Get per-frame AnyCalib predictions [N, 4] (used in V3 reprojection)
- `compute_reprojection_loss()`: **Static method** - Reprojection loss (Phases 1-2, Phase 3 calibration anchor)
- `forward_single_frame()`: Single-frame AnyCalib pipeline (used by `get_per_frame_intrinsics()`)

**Key methods in `anycam_wrapper_fat.py`** (Phase 3 V2):
- `forward()`: Full AnyCam pipeline with FAT calibration
- `forward_with_calibration_info()`: **NEW** - Extends forward() to return calibration loss data:
  - `fat_rays`: [B, H*W, 3] rays WITH gradients (for calibration loss)
  - `per_frame_intrinsics`: [B, N, 4] per-frame AnyCalib predictions
  - `average_intrinsics`: [B, 4] averaged reference (detached)
  - `fat_image_size`: (H_ray, W_ray) ray resolution
  - `original_image_size`: (H, W) original image size
- `freeze_except_fat_and_pose()`: Freeze all except FAT + Pose Head (joint training)
- `freeze_fat_only()`: For alternating training (not used in V2)
- `freeze_pose_only()`: For alternating training (not used in V2)

**Key functions in `train_fat_calibration.py`**:
- `train_phase_1()`: Phase 1 training with V3 reprojection loss
- `train_phase_2()`: Phase 2 training with V3 reprojection loss (overfits with visual tokens)
- `train_phase_3()`: **Phase 3 V2** training with combined loss
- `compute_combined_phase3_loss()`: **NEW** - Computes combined flow + calibration loss
- `plot_phase3_loss_curves()`: **NEW** - Plots total, flow, and calibration losses separately

**Key methods in `differentiable_calibrator.py`** (V2, kept for reference):
- `build_design_matrix()`: Construct A from pixel coords
- `rays_to_tangent()`: Convert rays to b vector [rx/rz, ry/rz]
- `solve_weighted_lstsq()`: Differentiable WLS solver (implicit differentiation)
- `params_to_intrinsics()`: Convert [p,q,r,s] to [fx,fy,cx,cy]
- **Note**: This was designed for V2 WLS approach but abandoned in favor of V3 reprojection

---

## References

- **RANSAC**: Fischler & Bolles (1981)
- **AnyCalib**: Pinhole calibration from casual images
- **DINOv2**: Oquab et al. (2023) - Self-supervised visual features
- **DA3**: Depth Anything v3 - Camera conditioning inspiration
- **Implicit Differentiation**: Differentiating through optimization solutions

---

## Changelog

| Version | Date | Changes |
|---------|------|---------|
| V1 | Jan 2026 | Soft exponential weights - FAILED (NaN gradients) |
| V2 | Jan 2026 | Implicit differentiation through WLS (Phase 1 v2, Phase 2 v1) |
| **V3** | Jan 2026 | Reprojection loss using average per-frame AnyCalib intrinsics (Phase 1 v3, Phase 2 v2) |
| V3 | Jan 2026 | Training data reduction: step=2 (half the frames) for faster training |
