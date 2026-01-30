# Feature Aggregation Transformer (FAT) - Detailed Architecture Explanation

**Author**: Kalman Mahlich
**Purpose**: Master's Thesis Documentation
**Date**: January 2026

---

## Table of Contents

1. [High-Level Overview](#high-level-overview)
2. [Critical Insight: Calibration-Tuned Features](#critical-insight-calibration-tuned-features)
3. [Step-by-Step Architecture](#step-by-step-architecture)
4. [The Training Challenge: Non-Differentiability](#the-training-challenge-non-differentiability)
5. [Comparison with DA3 Integration](#comparison-with-da3-integration)

---

## High-Level Overview

**Goal**: Take multiple frames (e.g., 4 frames) from a video and predict a single camera calibration (fx, fy, cx, cy) that represents the fixed camera for that sequence.

**Key Innovation**: Instead of running AnyCalib on each frame separately and averaging the 4 scalar outputs, we aggregate the **rich spatial features** before they get collapsed into scalars. This preserves spatial relationships and geometric information that would be lost in scalar averaging.

**Architecture Flow**:
```
N Frames → DINOv2 Features → FAT Aggregation → DPT Decoder → Ray Head → Calibrator → Calibration
```

---

## Critical Insight: Calibration-Tuned Features

### Why Sample from DINOv2 Features?

**The DINOv2 backbone in AnyCalib is NOT the off-the-shelf pretrained model.**

During AnyCalib's training, the DINOv2 ViT-L/14 backbone was **fine-tuned end-to-end** on the calibration task. This means:

1. **Standard DINOv2** (pretrained on ImageNet/self-supervised):
   - Learns general visual features (objects, textures, semantics)
   - Not specifically optimized for geometric/calibration tasks

2. **AnyCalib's DINOv2** (fine-tuned for calibration):
   - Features are **specifically tuned** to encode calibration-relevant information
   - Learns to detect: vanishing points, line orientations, geometric structure, perspective cues
   - Optimized through the calibration loss (ray prediction + camera fitting)

### Implications for FAT

This fine-tuning is **critical** for why the FAT architecture makes sense:

✅ **The spatial features at [N, 1024, h, w] already encode rich calibration information**
- These features have been trained to capture geometric cues relevant for calibration
- Aggregating these calibration-tuned features leverages the full power of AnyCalib's learned representations

❌ **If we only use AnyCalib's final outputs [N, 4]**:
- We throw away all the rich intermediate representations
- We collapse spatial geometry into 4 scalars per frame
- Aggregation happens on impoverished representations

**Analogy**:
- Using final outputs: Like having 4 people each measure a room and averaging their measurements
- Using FAT: Like having 4 detailed 3D scans of a room and intelligently combining them before taking measurements

---

## Step-by-Step Architecture

### STEP 1: DINOv2 Backbone - Multi-Scale Feature Extraction

#### High-Level
**Input**: N frames (e.g., 4 frames) from a video sequence
**Output**: Rich spatial features for each frame at 4 different scales
**What it does**: Extracts visual features from each image using a Vision Transformer (DINOv2 ViT-L/14)
**Status**: **FROZEN** (pretrained by AnyCalib, not retrained)

#### The "× 4" Multi-Scale Architecture

DINOv2 is a **multi-scale feature extractor**. It outputs **4 different feature maps** extracted from different depths of the Vision Transformer:

| Scale | Transformer Block | Level | Information Captured |
|-------|------------------|-------|---------------------|
| Scale 1 | Block 4 | Early | Low-level patterns (edges, textures, corners) |
| Scale 2 | Block 11 | Mid-early | Medium-level patterns (object parts, junctions) |
| Scale 3 | Block 17 | Mid-late | High-level patterns (object shapes, structures) |
| Scale 4 | Block 23 | Final | Very high-level (scene layout, context) |

#### Concrete Dimensions

For a single frame (H=480, W=640):
```python
Input: [1, 3, 480, 640]  # RGB image

DINOv2 forward pass:
  Tokenization: 14×14 patches → (480/14, 640/14) = (34, 46) tokens

  Output (4 scales):
    Scale 1 (Block 4):  [1, 1024, 34, 46]
    Scale 2 (Block 11): [1, 1024, 34, 46]
    Scale 3 (Block 17): [1, 1024, 34, 46]
    Scale 4 (Block 23): [1, 1024, 34, 46]
```

For **N=4 frames**, we stack features from all frames:
```python
Scale 1: [4, 1024, 34, 46]  # 4 frames' worth of scale 1 features
Scale 2: [4, 1024, 34, 46]  # 4 frames' worth of scale 2 features
Scale 3: [4, 1024, 34, 46]  # 4 frames' worth of scale 3 features
Scale 4: [4, 1024, 34, 46]  # 4 frames' worth of scale 4 features
```

---

### STEP 1.5: Feature Aggregation Transformer (FAT) - NEW CONTRIBUTION

#### High-Level
**Input**: Multi-frame features at 4 scales from DINOv2
**Output**: Single aggregated feature map at each of the 4 scales
**What it does**: For each spatial position in the feature maps, aggregate information across all N frames using self-attention
**Status**: **TRAINABLE** (this is what we're training)

#### Why Aggregate at Spatial Level Instead of Scalars?

**Option A (Naive - What We DON'T Do)**:
```python
for frame in frames:
    calibration = AnyCalib(frame)  # [4] scalars
mean_calibration = mean(calibrations)  # Average 4 scalars
```

**Option B (FAT - What We DO)**:
```python
features = [DINOv2(frame) for frame in frames]  # Rich spatial features
aggregated_features = FAT(features)  # Aggregate before collapse
calibration = AnyCalib_continue(aggregated_features)  # One calibration
```

**Key advantage**: Spatial features preserve:
- Geometric relationships between points
- Consistency of geometric structure across frames
- Sub-pixel accuracy in feature correspondences

Once collapsed to scalars, this information is **irretrievably lost**.

#### Detailed Operation (Per Scale)

For **one scale** (e.g., Scale 1: [4, 1024, 34, 46]):

**Step 1: Reshape to treat each spatial position as a sequence**
```python
Input: [4, 1024, 34, 46]
       ↓ permute(2, 3, 0, 1)
       [34, 46, 4, 1024]
       ↓ reshape
       [1564, 4, 1024]  # S=1564 spatial positions, N=4 frames, C=1024 channels
```

**Step 2: For EACH spatial position (in parallel via batching)**
```python
# At position (h=10, w=20), we have:
position_tokens = [
    frame0_feature[10, 20, :],  # [1024]
    frame1_feature[10, 20, :],  # [1024]
    frame2_feature[10, 20, :],  # [1024]
    frame3_feature[10, 20, :],  # [1024]
]  # Shape: [4, 1024]

# Optional: Add visual context (DINOv2-small CLS tokens)
if use_visual_conditioning:
    visual_tokens = dinov2_small_cls(frames)  # [4, 384]
    visual_projected = linear(visual_tokens)  # [4, 1024]
    all_tokens = concat([position_tokens, visual_projected])  # [8, 1024]
else:
    all_tokens = position_tokens  # [4, 1024]

# Apply self-attention layers
for layer in transformer_layers:
    all_tokens = layer(all_tokens)  # Multi-head self-attention + FFN

# Aggregate to single token (mean pooling or learnable)
aggregated = mean(all_tokens[:4], dim=0)  # [1024]
```

**Step 3: Reshape back to spatial format**
```python
Aggregated: [1564, 1024]
            ↓ reshape
            [34, 46, 1024]
            ↓ permute
            [1, 1024, 34, 46]
```

---

### STEP 2: DPT Decoder + Ray Head

#### High-Level
**Input**: Aggregated features at 4 scales (from FAT)
**Output**: 3D rays for each pixel in the image
**What it does**: Converts multi-scale features into geometric predictions (rays)
**Status**: **FROZEN** (pretrained by AnyCalib)

**DPT Decoder Process**:
```python
Input: 4 scales of [1, 1024, 34, 46]

Step 1: Reassemble blocks (upsample each scale)
Step 2: Fusion blocks (bottom-up fusion)
Output: [1, 256, 69, 92]  # H/7, W/7
```

**Ray Head**:
```python
Input: [1, 256, 69, 92]
→ Tangent Head → [1, 2, 69, 92]
→ Convex Upsampling → [1, 2, 480, 640]
→ Exponential Map → [1, 3, 480, 640]  # 3D unit rays
```

---

### STEP 3: Calibrator

#### High-Level
**Input**: 3D rays (one per pixel)
**Output**: Camera calibration (fx, fy, cx, cy)
**What it does**: Fits a camera model to the predicted rays using RANSAC + Gauss-Newton
**Status**: **NON-DIFFERENTIABLE** (classical algorithm)

---

## The Training Challenge: Non-Differentiability

### The Core Problem

The calibrator uses **RANSAC (Random Sample Consensus)** which is fundamentally non-differentiable:

1. **Random sampling**: Cannot differentiate through random selection
2. **Conditional logic**: `if error < threshold` creates discontinuities
3. **Argmax operation**: Choosing best candidate is non-differentiable
4. **LU factorization**: Iterative, not PyTorch-differentiable

```python
# This FAILS:
rays = model(images)  # Has grad_fn
intrinsics = calibrator(rays)  # NO grad_fn!
loss = mse_loss(intrinsics, gt_intrinsics)
loss.backward()  # ERROR: no gradients
```

### Evolution of Solutions

#### V1 Attempt: Soft Exponential Weights (FAILED)

**Initial approach**: Bypass the non-differentiable calibrator by computing a differentiable loss directly on rays using soft exponential weights:

```python
# V1 (FAILED)
# 1. Run calibrator to get intrinsics (detached)
with torch.no_grad():
    intrinsics = calibrator(rays.detach())

# 2. Compute expected rays from intrinsics
expected_rays = compute_expected_rays(intrinsics, image_size)

# 3. Compute angular error
cos_theta = (predicted_rays * expected_rays).sum(dim=-1)
angular_error = torch.acos(cos_theta.clamp(-1, 1))

# 4. Soft exponential weighting (KEY IDEA)
# Higher error → lower weight (smooth, differentiable)
sigma = threshold_degrees / 2.0
weights = torch.exp(-(angular_error ** 2) / (2 * sigma ** 2))

# 5. Weighted ray consistency loss
ray_errors = (predicted_rays - expected_rays).pow(2).sum(dim=-1)
loss = (weights * ray_errors).sum() / (weights.sum() + 1e-8)
```

**Intended gradient flow**:
```
loss → weights → angular_error → rays → FAT
loss → ray_errors → rays → FAT
```

**What happened**: NaN gradients appeared in ALL FAT transformer layers from batch 0:
```
[Batch 0] Loss: 2.574564e-03 (valid)
[Batch 0] Rays: normalized, no NaN
[Batch 0] Soft weights: mean=0.308821, no NaN
[Batch 0] FAT layer 0 gradients: NaN
[Batch 0] FAT layer 1 gradients: NaN
```

**Root cause analysis**:
1. The soft weights depend on `angular_error` which depends on rays
2. Gradients flow through: `loss → weights → exp() → angular_error → acos() → rays`
3. The composition of `exp()` and `acos()` creates numerical instability
4. Combined with FP16 mixed precision, gradients exploded to NaN

**Attempted fixes (all failed)**:
1. Moving loss computation to FP32 - still NaN
2. Disabling autocast for FAT entirely - still NaN
3. Gradient clipping - didn't prevent initial NaN
4. Smaller sigma values - made it worse

**Fundamental issue**: Differentiating through the soft weights introduced a complex gradient path that was numerically unstable regardless of precision settings.

---

#### V2 Solution: Implicit Differentiation (CURRENT)

**Key insight from supervisor**: The problem is not making RANSAC differentiable. The problem is finding a differentiable formulation where:
- RANSAC provides **constant** selection/weights (no gradients)
- A differentiable optimization uses those constants
- Gradients flow only through the optimization, not the selection

**Mathematical formulation**:

For a pinhole camera, we parameterize intrinsics as [p, q, r, s] where:
```
p = 1/fx    (inverse focal length x)
q = cx/fx   (normalized principal point x)
r = 1/fy    (inverse focal length y)
s = cy/fy   (normalized principal point y)
```

For each pixel (u, v) with predicted ray (rx, ry, rz), the constraint equations are:
```
rx/rz = u*p - q   →  [u, -1, 0, 0] @ [p,q,r,s]^T = rx/rz
ry/rz = v*r - s   →  [0, 0, v, -1] @ [p,q,r,s]^T = ry/rz
```

This is a **weighted least squares problem**:
```
x = argmin_x ||W(Ax - b)||²
```

Where:
- **A** ∈ ℝ^{2N×4}: Design matrix from pixel coordinates
  - Rows for x: [u, -1, 0, 0]
  - Rows for y: [0, 0, v, -1]
  - **CONSTANT** - depends only on pixel grid

- **b** ∈ ℝ^{2N}: Target vector from rays
  - First N elements: rx/rz for each pixel
  - Last N elements: ry/rz for each pixel
  - **DIFFERENTIABLE** - depends on predicted rays

- **W** ∈ ℝ^{2N}: Diagonal weight matrix
  - 1.0 for RANSAC inliers
  - 1e-6 for outliers
  - **CONSTANT** - determined by RANSAC, treated as fixed

- **x** ∈ ℝ^4: Solution [p, q, r, s]
  - **DIFFERENTIABLE** w.r.t. b

**Closed-form solution**:
```
x = (A^T W² A)^{-1} A^T W² b
```

**Why this is differentiable w.r.t. b**:
- A and W are constants (no grad_fn)
- Matrix multiplication is differentiable
- `torch.linalg.solve` provides automatic differentiation
- Only b carries gradients → only the solution depends on rays

**Conversion to intrinsics**:
```python
p, q, r, s = x
fx = 1 / p
fy = 1 / r
cx = q / p  # = q * fx
cy = s / r  # = s * fy
```

**Gradient flow in V2**:
```
loss = MSE(wls_intrinsics, ransac_intrinsics)
  ↓ ∂loss/∂intrinsics
wls_intrinsics [fx, fy, cx, cy]
  ↓ ∂intrinsics/∂params (chain rule through 1/p, q/p, etc.)
params [p, q, r, s]
  ↓ ∂params/∂b (implicit diff through WLS solution)
tangent_coords b = [rx/rz, ry/rz]
  ↓ ∂tangent/∂rays (chain rule through division)
predicted_rays [rx, ry, rz]
  ↓ (backprop through frozen DPT, Ray Head)
FAT parameters (UPDATED!)
```

**Critical difference from V1**:
- V1: Gradients flow through weight computation (exp, acos) → NaN
- V2: Weights are **constant** (no gradients) → stable linear algebra

---

### V2 Implementation

```python
class DifferentiableCalibrator(nn.Module):
    def __init__(self, inlier_weight=1.0, outlier_weight=1e-6, damping=1e-6):
        super().__init__()
        self.inlier_weight = inlier_weight
        self.outlier_weight = outlier_weight
        self.damping = damping

    def build_design_matrix(self, H, W, device):
        """Build A from pixel coordinates (CONSTANT)."""
        u_coords, v_coords = torch.meshgrid(...)
        A = torch.zeros(2*H*W, 4)
        A[:N, 0] = u_flat      # x equations: [u, -1, 0, 0]
        A[:N, 1] = -1.0
        A[N:, 2] = v_flat      # y equations: [0, 0, v, -1]
        A[N:, 3] = -1.0
        return A

    def rays_to_tangent(self, rays):
        """Convert rays to tangent coords (DIFFERENTIABLE)."""
        rz = rays[:, 2:3].clamp(min=1e-6)
        return rays[:, :2] / rz  # [rx/rz, ry/rz]

    def solve_weighted_lstsq(self, A, b, W):
        """
        Solve x = argmin ||W(Ax - b)||²

        DIFFERENTIABLE w.r.t. b because:
        - A and W are treated as constants
        - torch.linalg.solve provides gradients
        """
        W_sq = W * W
        AW = A * W_sq.unsqueeze(1)
        AtWA = A.t() @ AW + self.damping * torch.eye(4)
        AtWb = A.t() @ (W_sq * b)
        return torch.linalg.solve(AtWA, AtWb)

    def params_to_intrinsics(self, params):
        """Convert [p,q,r,s] to [fx,fy,cx,cy] (DIFFERENTIABLE)."""
        p, q, r, s = params
        fx = 1.0 / p.clamp(min=1e-8)
        fy = 1.0 / r.clamp(min=1e-8)
        cx = q / p.clamp(min=1e-8)
        cy = s / r.clamp(min=1e-8)
        return torch.stack([fx, fy, cx, cy])

    def forward(self, predicted_rays, image_size, inlier_mask):
        """
        Args:
            predicted_rays: [H*W, 3] - WITH GRADIENTS
            inlier_mask: [H*W] boolean - CONSTANT (from RANSAC)
        Returns:
            intrinsics: [4] - DIFFERENTIABLE w.r.t. rays
        """
        H, W = image_size
        A = self.build_design_matrix(H, W, predicted_rays.device)

        tangent = self.rays_to_tangent(predicted_rays)
        b = torch.cat([tangent[:, 0], tangent[:, 1]])  # [2*H*W]

        # Hard weights from RANSAC (CONSTANT)
        W = torch.where(inlier_mask, self.inlier_weight, self.outlier_weight)
        W = torch.cat([W, W])  # Expand for x and y equations

        params = self.solve_weighted_lstsq(A, b, W)
        intrinsics = self.params_to_intrinsics(params)

        return {'intrinsics': intrinsics}
```

---

### Training Loop (V2)

```python
for batch in dataloader:
    optimizer.zero_grad()

    # 1. Forward pass - rays WITH GRADIENTS
    rays = model(images)  # [B, H*W, 3]

    # 2. Run RANSAC (no gradients needed)
    with torch.no_grad():
        ransac_result = calibrator(rays.detach())
        ransac_intrinsics = ransac_result['intrinsics']  # [4]
        inlier_mask = ransac_result['inlier_mask']  # [H*W]

    # 3. Differentiable WLS with constant weights
    wls_result = diff_calibrator(
        predicted_rays=rays,  # Has gradients!
        image_size=(H, W),
        inlier_mask=inlier_mask.detach(),  # Constant!
    )
    wls_intrinsics = wls_result['intrinsics']  # Has gradients!

    # 4. Loss
    loss = F.mse_loss(wls_intrinsics, ransac_intrinsics.detach())

    # 5. Backward - gradients flow through WLS to rays to FAT
    loss.backward()
    optimizer.step()
```

---

### Why V2 Works

| Aspect | V1 (Soft Weights) | V2 (Implicit Diff) |
|--------|-------------------|-------------------|
| **Weight computation** | Differentiable (caused NaN) | Constant (no gradients) |
| **Gradient path** | loss→exp→acos→rays | loss→WLS→tangent→rays |
| **Numerical operations** | exp(), acos() (unstable) | Linear algebra (stable) |
| **Mixed precision** | Failed even in FP32 | Works with FP32 WLS |

**Key insight**: By treating weights as constants, we eliminate the problematic gradient path through exponential and inverse trigonometric functions. The remaining gradient path through weighted least squares involves only numerically stable linear algebra operations.

---

#### V3 Solution: Reprojection Loss (CURRENT for Phases 1-2)

**Key insight**: Instead of matching intrinsics (which requires solving a least-squares problem), directly optimize ray predictions to match expected pixel coordinates. This provides more direct supervision and eliminates the need for matrix inversion.

**Mathematical formulation**:

1. **Get per-frame AnyCalib predictions**: For N frames, get (fx_i, fy_i, cx_i, cy_i) for each frame
2. **Compute average intrinsics**: 
   ```
   fx_avg = mean(fx_i), fy_avg = mean(fy_i)
   cx_avg = mean(cx_i), cy_avg = mean(cy_i)
   ```
3. **Scale intrinsics to ray resolution**: Account for DINOv2 padding
   ```
   scale_x = W_ray / W_orig
   scale_y = H_ray / H_orig
   fx_down = fx_avg * scale_x
   fy_down = fy_avg * scale_y
   cx_down = cx_avg * scale_x
   cy_down = cy_avg * scale_y
   ```
4. **Project rays to image plane**:
   ```
   u_projected = fx_down * (rx/rz) + cx_down
   v_projected = fy_down * (ry/rz) + cy_down
   ```
5. **Compute loss**: MSE between projected coordinates and actual pixel grid
   ```
   loss = MSE([u_projected, v_projected], [u_actual, v_actual])
   ```

**Gradient flow in V3**:
```
loss = MSE(projected_coords, actual_pixel_grid)
  ↓ ∂loss/∂projected_coords
projected_coords [u, v] = fx*(rx/rz)+cx, fy*(ry/rz)+cy
  ↓ ∂projected/∂rays (chain rule through division and multiplication)
predicted_rays [rx, ry, rz]
  ↓ (backprop through frozen DPT, Ray Head)
FAT parameters (UPDATED!)
```

**Critical advantages over V2**:
- **No matrix inversion**: Direct computation, no numerical stability issues
- **Faster**: No need to solve WLS system
- **More direct**: Optimizes rays directly to match pixel coordinates
- **No RANSAC dependency**: Loss computation doesn't require RANSAC

**Implementation**:
```python
@staticmethod
def compute_reprojection_loss(
    predicted_rays: Tensor,  # [H*W, 3] normalized rays
    average_intrinsics: Tensor,  # [4] (fx, fy, cx, cy) for original image
    ray_image_size: Tuple[int, int],  # (H_ray, W_ray)
    original_image_size: Tuple[int, int],  # (H_orig, W_orig)
) -> Tuple[Tensor, Dict]:
    H_ray, W_ray = ray_image_size
    H_orig, W_orig = original_image_size
    
    # Scale intrinsics to ray resolution
    scale_x = W_ray / W_orig
    scale_y = H_ray / H_orig
    fx_down = average_intrinsics[0] * scale_x
    fy_down = average_intrinsics[1] * scale_y
    cx_down = average_intrinsics[2] * scale_x
    cy_down = average_intrinsics[3] * scale_y
    
    # Project rays to image plane
    rz = predicted_rays[:, 2:3].clamp(min=1e-6)
    projected_x = fx_down * (predicted_rays[:, 0] / rz.squeeze()) + cx_down
    projected_y = fy_down * (predicted_rays[:, 1] / rz.squeeze()) + cy_down
    projected = torch.stack([projected_x, projected_y], dim=-1)  # [H*W, 2]
    
    # Create actual pixel grid
    y_coords, x_coords = torch.meshgrid(
        torch.arange(H_ray, device=predicted_rays.device, dtype=predicted_rays.dtype),
        torch.arange(W_ray, device=predicted_rays.device, dtype=predicted_rays.dtype),
        indexing='ij'
    )
    actual_coords = torch.stack([x_coords.flatten(), y_coords.flatten()], dim=-1)  # [H*W, 2]
    
    # Compute MSE loss
    loss = F.mse_loss(projected, actual_coords.float())
    return loss, {'reprojection_loss': loss.item()}
```

**Training loop (V3)**:
```python
for batch in dataloader:
    optimizer.zero_grad()
    
    # 1. Forward pass - rays WITH GRADIENTS
    result = model.forward_with_differentiable_calibration(images, ...)
    rays = result['rays']  # [1, H*W, 3] - has gradients
    
    # 2. Get per-frame AnyCalib predictions (no gradients)
    with torch.no_grad():
        per_frame_intrinsics = model.get_per_frame_intrinsics(images)  # [N, 4]
    
    # 3. Compute average intrinsics
    average_intrinsics = per_frame_intrinsics.mean(dim=0)  # [4]
    
    # 4. Compute reprojection loss
    loss, loss_info = model.compute_reprojection_loss(
        predicted_rays=rays[0],  # [H*W, 3] - WITH gradients
        average_intrinsics=average_intrinsics,  # [4] - detached
        ray_image_size=result['image_size'],
        original_image_size=(H_orig, W_orig),
    )
    
    # 5. Backward - gradients flow through projection to rays to FAT
    loss.backward()
    optimizer.step()
```

### Comparison: V2 vs V3

| Aspect | V2 (WLS vs RANSAC) | V3 (Reprojection) |
|--------|-------------------|-------------------|
| **Loss computation** | MSE(WLS_intrinsics, RANSAC_intrinsics) | MSE(projected_coords, pixel_grid) |
| **Requires RANSAC** | Yes (for inlier mask and reference) | No (only for per-frame AnyCalib) |
| **Matrix operations** | WLS solve (matrix inversion) | Direct projection (multiplication only) |
| **Numerical stability** | Good (with damping) | Excellent (no inversion) |
| **Speed** | Slower (WLS solve) | Faster (direct computation) |
| **Gradient path** | loss→WLS→tangent→rays | loss→projection→rays |
| **Supervision** | Indirect (via intrinsics) | Direct (pixel coordinates) |
| **Used in** | Phase 3 | Phases 1-2 |

**Key insight**: V3 provides more direct supervision by optimizing rays to match pixel coordinates directly, rather than optimizing intrinsics and then comparing them. This eliminates the need for matrix inversion and provides a cleaner gradient path.

---

## Training Data Reduction (V3)

To reduce training time while maintaining video diversity, the dataset now samples sequences with **step=2** (every other frame) instead of step=1.

**Implementation**:
```python
# OLD (V2): Create sequences with step size 1
for start in range(0, safe_total - self.num_frames + 1, 1):
    self.sequences.append({...})

# NEW (V3): Create sequences with step size 2 (use half the frames)
for start in range(0, safe_total - self.num_frames + 1, 2):
    self.sequences.append({...})
```

**Effect**:
- **Same videos**: All videos in train/val/test splits are still used
- **Half the sequences**: Samples every other starting frame position
- **Faster training**: ~3,993 batches/epoch (down from ~7,987)
- **Video diversity maintained**: Still covers diverse temporal samples from each video
- **Test set unchanged**: Benchmarking consistency maintained

---

## Gradient Flow Through Frozen Layers

A critical implementation detail: **how to freeze parameters while allowing gradient flow**.

### The Wrong Approach (Blocks All Gradients)

```python
# WRONG - Blocks gradient flow entirely
with torch.no_grad():
    features = frozen_layer(x)
    # features has no grad_fn - gradient flow is BLOCKED
```

### The Correct Approach (Freeze Parameters, Allow Gradients)

```python
# Freeze parameters (one-time setup)
for param in frozen_layer.parameters():
    param.requires_grad = False

# During forward pass (no torch.no_grad()!)
features = frozen_layer(x)
# features HAS grad_fn - gradients flow through activations
# but frozen_layer.parameters() won't be updated
```

**Why this works**:
- `param.requires_grad = False`: Parameters won't be updated
- But intermediate activations still track gradients
- Gradients backpropagate through to trainable FAT

---

## Comparison with DA3 Integration

### DA3 Architecture (Previous Experiment)

```
N Frames
    ↓
Per-frame AnyCalib → [N, 4] calibrations
    ↓
Camera Encoder → [N, 256] camera tokens
    ↓
Visual-Camera Mixing (self-attention)
    ↓
Sequence Aggregation → [1, 256] sequence token
    ↓
Camera Decoder → [1, 4] final calibration
```

**What it aggregates**: 4 scalar values per frame (fx, fy, cx, cy)

### FAT Architecture (This Experiment)

```
N Frames
    ↓
DINOv2 (per-frame) → [N, 1024, h, w] × 4 features
    ↓
FAT (spatial aggregation) → [1, 1024, h, w] × 4 features
    ↓
DPT Decoder + Ray Head → rays
    ↓
Calibrator → [1, 4] final calibration
```

**What it aggregates**: 1024-dimensional spatial feature maps

### Key Differences

| Aspect | DA3 | FAT |
|--------|-----|-----|
| **Aggregation input** | 4 scalars per frame | 1024-dim spatial features |
| **Information preserved** | Low (scalars) | High (rich spatial features) |
| **Aggregation location** | After AnyCalib | Inside AnyCalib (between steps 1-2) |
| **Uses AnyCalib training** | Only final outputs | Leverages intermediate features |
| **Spatial structure** | Lost before aggregation | Preserved during aggregation |

### Why FAT Should Outperform DA3

1. **Richer representations**: 1024-dim features >> 4 scalars
2. **Spatial preservation**: Geometric relationships maintained
3. **Calibration-tuned**: Features already optimized for calibration
4. **More direct**: Aggregates at the "right level" in the pipeline

---

## Summary for Thesis

### Key Points

1. **Motivation**: AnyCalib's DINOv2 features are calibration-aware (fine-tuned for calibration). Aggregating these rich spatial features preserves geometric information that would be lost when aggregating scalar outputs.

2. **Challenge**: RANSAC calibrator is non-differentiable, preventing standard end-to-end training.

3. **Failed Approach (V1)**: Soft exponential weights on ray consistency loss caused NaN gradients from batch 0 due to numerically unstable gradient paths through exp() and acos(). Even FP32 couldn't stabilize.

4. **Solution (V2 - Abandoned for Phases 1-2)**: Implicit differentiation through weighted least squares:
   - RANSAC provides constant weights (no gradients needed)
   - WLS solution is differentiable w.r.t. ray-derived target vector (b)
   - Gradients flow through stable linear algebra operations only
   - **Not used**: V3 reprojection loss proved simpler and more stable
   - **Kept for reference**: Code in `differentiable_calibrator.py`

5. **Solution (V3 - Phases 1-2)**: Reprojection loss using per-frame AnyCalib as reference:
   - Get per-frame AnyCalib predictions (detached): [N, 4]
   - Average intrinsics: [4] (detached reference)
   - Project FAT rays back to image plane using average intrinsics
   - Compute MSE between projected coordinates and pixel grid
   - Direct supervision without matrix inversion or RANSAC dependency
   - Faster and more numerically stable than V2 WLS
   - Successfully trained Phases 1 and 2

6. **Solution (Phase 3 V2 - Combined Loss)**: End-to-end with stability anchor:
   - **Primary loss**: Flow reprojection (AnyCam standard) for pose learning
   - **Anchor loss**: Calibration reprojection (Phase 1 V3 style) for stability
   - **Combined**: L_total = 1.0 * L_flow + 1e-4 * L_calib (tunable)
   - **Prevents**: Calibration explosion (extreme values that minimize flow but diverge)
   - **Joint training**: FAT + Pose Head trained together (not alternating)
   - **Skip Phase 2**: Load Phase 1 directly (Phase 2 overfits with visual tokens)

7. **Theoretical Contributions**:

   **Pattern for Non-Differentiable Algorithms**:
   ```
   Non-diff Algorithm → Constant Selection → Differentiable Optimization → Gradients
   ```
   Examples:
   - V2: RANSAC → inlier mask (constant) → WLS solution (differentiable)
   - V3: AnyCalib → average intrinsics (constant) → ray reprojection (differentiable)

   **Combined Loss for Stability**:
   - Primary self-supervised loss (flow reprojection) for main task (pose)
   - Anchor loss (calibration reprojection) prevents divergence with small weight
   - Both losses reach FAT through different gradient paths (focal vs rays)
   - Enables joint training without instability

### Training Evolution

| Phase | Loss | Checkpoint | Visual Tokens | Result |
|-------|------|-----------|---------------|--------|
| **Phase 1 V3** | Reprojection (V3) | None | No | Stable (converged) |
| **Phase 2 V2** | Reprojection (V3) | Phase 1 | Yes (DINOv2-small) | **Overfitting** (val loss exploded) |
| **Phase 3 V2** | **Combined** (Flow + Calib) | Phase 1 (skip 2) | No | **CURRENT** |

**Key Lesson**: Visual conditioning (Phase 2) caused overfitting. Phase 3 V2 avoids this by loading Phase 1 directly and using combined loss for stability.

---

## References

- **AnyCalib Paper**: Details on DINOv2 fine-tuning and calibration pipeline
- **DINOv2**: Vision transformer architecture, multi-scale features
- **DPT Decoder**: Dense Prediction Transformer for feature fusion
- **DA3 (Depth Anything 3)**: Camera conditioning inspiration
- **Implicit Differentiation**: Differentiating through optimization solutions

---

---

## Phase 3 V2: End-to-End Training with Combined Loss

### Overview

Phase 3 V2 integrates FAT-enhanced AnyCalib into the full AnyCam pipeline for end-to-end training with **combined self-supervised loss**. Unlike Phases 1-2 which train only FAT, Phase 3 trains both FAT and Pose Head **together** (not alternating) with a dual-objective loss function.

**Key Design Decisions**:
1. **Skip Phase 2**: Load Phase 1 checkpoint directly (Phase 2 overfits with visual tokens)
2. **Combined Loss**: Flow reprojection (PRIMARY) + Calibration reprojection (ANCHOR)
3. **Joint Training**: Train FAT + Pose Head together (no alternating epochs)
4. **Random Pose Init**: Train pose head from scratch alongside FAT (not pretrained)
5. **Calibration Stability**: Small λ_calib prevents extreme calibrations without over-constraining

### Architecture Flow

```
Input: [B, N, 3, H, W] (N = max_ahead + 1 frames)
  ↓
┌─────────────────────────────────────────────────────────────────┐
│ STEP 1: FAT-Enhanced AnyCalib (TRAINABLE)                      │
│         DINOv2: [B*N, 3, H, W] → [B*N, 1024, h, w] × 4        │
│         FAT: Aggregate [B*N, ...] → [B, 1024, h, w] × 4       │
│         DPT+Ray: [B, 1024, h, w] → rays [B, H*W, 3] WITH GRADS│
│         Calibrator: rays → intrinsics [B, 4]                   │
│         Extract fx: [B, 4] → [B]                                │
│                                                                  │
│  ALSO CAPTURES for calibration loss:                            │
│    - FAT rays [B, H*W, 3] WITH GRADIENTS                       │
│    - Per-frame intrinsics [B, N, 4] (from individual frames)   │
│    - Average intrinsics [B, 4] (detached reference)            │
└─────────────────────────────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────────────────────────────┐
│ STEP 2: Depth Predictor (FROZEN)                               │
│         [B, N, 3, H, W] → [B, N, 1, H, W]                      │
└─────────────────────────────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────────────────────────────┐
│ STEP 3: Flow Processor (FROZEN)                                │
│         [B, N, 3, H, W] → [B, N, 3, H, W] (flow + occlusion)  │
└─────────────────────────────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────────────────────────────┐
│ STEP 4: Pose Head (TRAINABLE, RANDOM INIT)                     │
│         Input: depths, flows, focal_length [B]                  │
│         Output: poses [B, N, 4, 4], uncertainties               │
└─────────────────────────────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────────────────────────────┐
│ STEP 5: Combined Loss Computation                               │
│                                                                  │
│  L_flow (PRIMARY - Pose Learning):                              │
│    induce_flow(depths, poses, fx) → induced_flow                │
│    MSE(induced_flow, observed_flow) → L_flow                    │
│                                                                  │
│  L_calib (ANCHOR - Calibration Stability):                      │
│    project_rays(FAT_rays, average_intrinsics) → coords          │
│    MSE(coords, pixel_grid) → L_calib                            │
│                                                                  │
│  Total Loss:                                                     │
│    L_total = λ_flow * L_flow + λ_calib * L_calib               │
│              (1.0)            (1e-4, tunable)                   │
└─────────────────────────────────────────────────────────────────┘
```

### Combined Loss: Mathematical Formulation

#### Flow Reprojection Loss (PRIMARY)

**Purpose**: Train pose head to predict geometrically consistent poses

**Pipeline**:
```
1. Induce flow from geometry:
   induced_flow = warp(frame_i, depth_i, pose_i→j, focal_length)

2. Compare with observed flow (from UniMatch):
   L_flow = MSE(induced_flow, observed_flow) * occlusion_mask
```

**Gradient Path**:
```
L_flow
  ↓ ∂L/∂induced_flow
induced_flow [B, N, 2, H, W]
  ↓ ∂induced/∂poses  ∂induced/∂focal
poses [B, N, 4, 4]   focal [B]
  ↑                    ↑
Pose Head          FAT intrinsics
(TRAINABLE)        (TRAINABLE)
```

#### Calibration Reprojection Loss (ANCHOR)

**Purpose**: Prevent FAT from producing extreme calibrations that minimize flow but are geometrically unreasonable

**Pipeline**:
```
1. Get per-frame AnyCalib predictions (detached):
   per_frame_intrinsics = [AnyCalib(frame_i) for i in 1..N]  # [N, 4]
   average_intrinsics = mean(per_frame_intrinsics)  # [4] - detached

2. FAT produces aggregated rays (WITH gradients):
   FAT_rays = FAT(DINOv2_features) → DPT → Ray_Head  # [H*W, 3]

3. Project FAT rays back to image plane using average intrinsics:
   u_proj = fx_avg * (ray_x / ray_z) + cx_avg
   v_proj = fy_avg * (ray_y / ray_z) + cy_avg

4. Compare with actual pixel coordinates:
   L_calib = MSE([u_proj, v_proj], [x_actual, y_actual])
```

**Key Insight**: The average intrinsics are NOT perfect (AnyCalib per-frame predictions have variance). We're not trying to match them exactly. We're only using them as a **soft anchor** to prevent explosion.

**Gradient Path**:
```
L_calib
  ↓ ∂L/∂projected_coords
projected_coords [H*W, 2]
  ↓ ∂proj/∂FAT_rays (through division: rx/rz, ry/rz)
FAT_rays [H*W, 3] WITH GRADIENTS
  ↓ (backprop through frozen DPT, Ray Head)
FAT parameters (TRAINABLE)
```

**Critical**: average_intrinsics is **detached** (no gradients). Only FAT_rays carry gradients.

#### Combined Loss

```python
Total = λ_flow * L_flow + λ_calib * L_calib

Default values:
- λ_flow = 1.0 (fixed)
- λ_calib = 1e-4 (tunable: 1e-4 to 0.15)
```

**Why This Works**:
- **L_flow dominates** (λ_flow >> λ_calib): Primary signal for pose learning
- **L_calib provides gentle constraint**: Prevents FAT from diverging to extreme values
- **Both gradients reach FAT**: Through different paths (focal vs rays)

**Tuning Strategy**:
1. Start with λ_calib = 1e-4 (small anchor)
2. Monitor calibration benchmarks (FAT intrinsics vs GT)
3. If calibration **diverges** (increasing error) → increase λ_calib
4. If calibration **over-constrained** (not improving) → decrease λ_calib
5. Goal: **Stable and reasonable** calibration, not perfect convergence

### Forward Pass with Dimensions

**Example**: B=2, N=4, H=480, W=640

```
Input: [2, 4, 3, 480, 640]
  ↓
STEP 1: FAT-Enhanced AnyCalib (per batch)
  For each b in [0, 1]:
    # FAT aggregation
    DINOv2: [4, 3, 490, 644] → [4, 1024, 35, 46] × 4 scales
    FAT: [4, 1024, 35, 46] → [1, 1024, 35, 46] × 4 (aggregate across 4 frames)
    DPT+Ray: [1, 1024, 35, 46] → rays [1, H*W, 3] WITH GRADIENTS
    Calibrator: rays → intrinsics [1, 4]

    # Per-frame intrinsics (detached)
    For each frame i in [0, 1, 2, 3]:
      AnyCalib(frame_i) → intrinsics_i [4]
    per_frame_intrinsics: [4, 4]
    average_intrinsics: mean([4, 4], dim=0) → [4] (detached)

  Stack:
    batch_intrinsics: [2, 4] (FAT intrinsics)
    batch_rays: [2, H*W, 3] (WITH GRADIENTS)
    batch_per_frame: [2, 4, 4]
    batch_average: [2, 4] (detached)
    focal_length: [2] (extract fx from batch_intrinsics)
  ↓
STEP 2: Depth Predictor (frozen)
  [2, 4, 3, 480, 640] → depths [2, 4, 1, 480, 640]
  ↓
STEP 3: Flow Processor (frozen)
  [2, 4, 3, 480, 640] → flow_occs [2, 4, 3, 480, 640]
  ↓
STEP 4: Pose Head (trainable)
  Input: depths, flows, focal_length [2]
  Output: poses [2, 4, 4, 4], uncertainties
  ↓
STEP 5: Combined Loss
  # Flow loss
  induced_flow = induce_flow(depths, poses, focal) → [2, 4, 2, 480, 640]
  L_flow = MSE(induced_flow, flow_occs[:,:,:2]) → scalar

  # Calibration loss (per batch)
  For each b in [0, 1]:
    project(batch_rays[b], batch_average[b]) → coords [H*W, 2]
    L_calib_b = MSE(coords, pixel_grid)
  L_calib = mean([L_calib_0, L_calib_1]) → scalar

  # Combined
  Total = 1.0 * L_flow + 1e-4 * L_calib → scalar
```

### Backward Pass (Gradient Flow)

**Dual Gradient Paths to FAT**:

```
                    ┌─────────────────────────────────────┐
                    │        Total Loss (scalar)          │
                    └────────────┬────────────────────────┘
                                 │
                 ┌───────────────┴───────────────┐
                 │                               │
           ∂L/∂L_flow                      ∂L/∂L_calib
                 │                               │
                 ↓                               ↓
        ┌────────────────────┐         ┌────────────────────┐
        │ L_flow (flow loss) │         │ L_calib (calib loss)│
        └────────┬───────────┘         └────────┬───────────┘
                 │                               │
                 ↓                               ↓
        ∂L_flow/∂induced_flow         ∂L_calib/∂projected_coords
                 │                               │
                 ↓                               ↓
         induced_flow                    projected_coords
         [B, N, 2, H, W]                 [B, H*W, 2]
                 │                               │
         ┌───────┴───────┐                       │
         │               │             ∂proj/∂FAT_rays
  ∂induced/∂poses  ∂induced/∂focal             │
         │               │                       ↓
         ↓               ↓                 FAT_rays [B, H*W, 3]
    poses [B, N, 4, 4]  focal [B]               │
         │               │              (backprop through
         ↑               ↑               frozen DPT, Ray Head)
    Pose Head       FAT intrinsics              │
    (TRAINABLE)     (TRAINABLE)                 ↓
         │               │             ┌────────────────────┐
         │               └─────────────┤  FAT Parameters    │
         │                             │   (TRAINABLE)      │
         └─────────────────────────────┤                    │
                                       └────────────────────┘
```

**Path 1 (via Flow Loss)**:
```
L_total → L_flow → induced_flow → focal_length → FAT_intrinsics → FAT
```

**Path 2 (via Calibration Loss)**:
```
L_total → L_calib → projected_coords → FAT_rays → FAT
```

**Path 3 (via Pose Loss)**:
```
L_total → L_flow → induced_flow → poses → Pose_Head
```

**Key Points**:
- FAT receives gradients from **both losses** (different paths)
- Pose Head receives gradients from **flow loss only**
- Depth predictor, flow processor, DINOv2, DPT, Ray Head are **frozen**
- Frozen components allow gradient flow but don't update parameters

### Training Strategy

**NOT Alternating (Unlike Original Phase 3 Plan)**:
- Train FAT + Pose Head **together** every epoch
- Simpler than alternating (both components co-adapt)
- Single optimizer with all trainable parameters

**Optimizer**:
```python
trainable_params = [p for p in model.parameters() if p.requires_grad]
optimizer = AdamW(trainable_params, lr=1e-5, weight_decay=1e-4)
```

**Mixed Precision**:
```python
scaler = GradScaler()

with autocast(device_type='cuda', dtype=torch.float16):
    output_data = model.forward_with_calibration_info(data)
    loss, loss_info = compute_combined_phase3_loss(...)

scaler.scale(loss).backward()
scaler.unscale_(optimizer)
torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
scaler.step(optimizer)
scaler.update()
```

**Why No Alternating**:
- Original plan had alternating to prevent one component dominating
- **Combined loss makes this unnecessary**: λ_calib is small (1e-4), so flow naturally dominates
- Simpler implementation and training dynamics
- Both components learn together from the start

### Benchmarking (Passive Monitoring with GT)

**Per-Epoch Evaluation**:

1. **Calibration Benchmarking**:
   - Compare FAT intrinsics vs GT intrinsics (Objectron dataset)
   - Metrics: Relative error for fx, fy, cx, cy
   - Uses fixed test samples (no cycling between epochs)
   - **Purpose**: Monitor if calibration is stable or diverging
   - Saved to: `calibration_benchmark_history.json`, `calibration_benchmark_curve.png`

2. **Pose Benchmarking**:
   - Compare FAT model vs AnyCam baseline (32 focal candidates)
   - Metrics: Rotation error (degrees), translation error (degrees)
   - Uses fixed test samples (no cycling between epochs)
   - **Purpose**: Check if FAT improves pose accuracy over baseline
   - Saved to: `pose_benchmark_history.json`, `pose_benchmark_curve.png`

3. **Loss Tracking**:
   - Separate plots for total loss, flow loss, calibration loss
   - Shows contribution percentage of each component
   - Saved to: `loss_curves_phase3.png`

**Important**: GT is **never used during training** (fully self-supervised). Benchmarks are for **passive monitoring only** to guide hyperparameter tuning.

### Comparison with DA3 Stage 3

| Aspect | DA3 Stage 3 | FAT Phase 3 V2 |
|--------|------------|----------------|
| **Calibration** | DA3 head (scalar aggregation) | FAT (feature aggregation) |
| **Input frames** | Pairs (2 frames) | Multi-frame (N = max_ahead+1) |
| **Trainable** | Calibration head only | FAT + Pose Head (both) |
| **Loss** | Flow reprojection only | Flow + Calibration anchor |
| **Pose head** | Pretrained (from baseline) | Random init (trained from scratch) |
| **Training strategy** | Single component | Joint training |
| **Stability mechanism** | None | Calibration anchor (λ_calib) |

**Key Difference**: FAT aggregates rich spatial features before calibration, while DA3 aggregates scalar calibration outputs. FAT Phase 3 V2 also adds calibration stability through combined loss.

### Historical Context: Abandoned Approaches

**Original Phase 3 Plan (Not Implemented)**:
- Load Phase 2 checkpoint (visual tokens)
- Alternating training (FAT → Pose → FAT → ...)
- Pure flow reprojection loss (no calibration anchor)

**Why Abandoned**:
1. **Phase 2 overfitting**: Val loss exploded (8.61 → 14.37 → 19.10)
2. **Visual tokens problematic**: DINOv2-small CLS tokens caused divergence
3. **No stability mechanism**: Pure flow loss could cause calibration to diverge
4. **Unnecessary complexity**: Alternating training not needed with proper loss weighting

**Phase 3 V2 Solutions**:
1. **Skip Phase 2**: Load Phase 1 checkpoint directly (no visual tokens)
2. **Joint training**: Train both together (simpler, both co-adapt)
3. **Calibration anchor**: Small λ_calib prevents divergence without over-constraining
4. **Random pose init**: Train pose head from scratch alongside FAT

---

**Document Version**: 4.0 (Phase 3 V2 Combined Loss)
**Last Updated**: January 2026

**Version History**:
- **V1** (Jan 2026): Soft exponential weights - FAILED (NaN gradients from exp/acos gradient path)
- **V2** (Jan 2026): Implicit differentiation through WLS (Phase 1 v2, Phase 2 v1) - works but complex
- **V3** (Jan 2026): Reprojection loss using average per-frame AnyCalib intrinsics (Phase 1 v3, Phase 2 v2) - simpler and stable
- **Phase 3 V2** (Jan 2026): Combined loss (flow + calibration anchor) with joint training, skips Phase 2 - **CURRENT IMPLEMENTATION**

**Implementation Status**:
- ✅ Phase 1 V3: Complete, trained successfully
- ✅ Phase 2 V2: Complete, but overfits with visual tokens (abandoned for Phase 3)
- ✅ Phase 3 V2: Implementation complete, ready for training
  - Combined loss function: `compute_combined_phase3_loss()`
  - Wrapper with calibration info: `forward_with_calibration_info()`
  - Joint training: `train_phase_3()` with GradScaler
  - Checkpoint loading: Phase 1 → Phase 3 (skips Phase 2)
  - Benchmarking: Calibration + Pose monitoring per epoch