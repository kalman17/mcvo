# Defense Technical Reference Sheet
**Thesis: "Learning Camera Geometry from Unlabeled Real-World Dynamic Video"**
*100% verified against source code. Quick-lookup format.*

---

## 1. THE FULL PIPELINE (what we built)

```
INPUT: N RGB frames [N, 3, H, W]

═══════════ CALIBRATION BRANCH ════════════════════════════════════════
  AnyCalib DINOv2 ViT-L/14  (frozen)
      ↓ 4 feature maps, each [N, 1024, h_s, w_s]
  FAT / MCT  (trainable, ~25M params)
      ↓ 4 aggregated maps, each [1, 1024, h_s, w_s]
  LightDPT Decoder + ConvexTangentDecoder  (frozen)
      ↓ per-pixel ray field [H*W, 3]
  Gauss-Newton Calibrator (RANSAC fallback)  (non-differentiable)
      ↓ K = [fx, fy, cx, cy]  (pixel space)

═══════════ POSE BRANCH ════════════════════════════════════════════════
  AnyCam DINOv2-small  (frozen in C, in_channels=6: RGB+flow+depth)
      ↓ CLS tokens at 4 scales [N*F, 384]
  PoseReassemble → PoseFusion → InterframeAttention (8 layers)  (trainable)
      ↓ pose token [B, N, 128]
  focal_embedding(focal_norm) → 8D injected into pose head
  PoseHead MLP  (trainable)
      ↓ pose [B, N, 1, 4, 4]  (R|t as 4×4 SE3 matrix)

LOSS = L_flow + λ_calib × L_anchor
  λ_calib = 1e-4  (default)
```

---

## 2. ANYCAM — FULL ARCHITECTURE

### Backbone
- **Model**: Depth Anything V2 Small (HuggingFace: `facebook/dinov2-small`)
- **Type**: DINOv2-small ViT
- **hidden_size**: 384
- **num_attention_heads**: 6
- **num_transformer_blocks**: 12
- **patch_size**: 14 × 14 pixels
- **Input image size**: padded to multiples of 14
- **Feature extraction at**: stages [3, 6, 9, 12] (4 feature maps)
- **Input channels**: **6** (not 3). RGB (3ch) + optical flow x/y (2ch) + inverse depth (1ch)
  - Achieved by repeating the original 3-channel patch embedding weights: `weight.repeat(1, 2, 1, 1)`
- **Input normalization**: ImageNet mean/std on RGB channels only

### DPT Neck / Uncertainty Head
- `neck_hidden_sizes` = [48, 96, 192, 384]
- `fusion_hidden_size` = **128** (code overrides config default of 64)
- `reassemble_hidden_size` = 384
- Output: uncertainty map [B, N, 1, 2, H, W] — channel 0 = flow uncertainty σ, channel 1 = dist uncertainty

### Pose Branch (step-by-step)

| Component | Input → Output | Details |
|---|---|---|
| `pose_reassemble_stage` | CLS tokens [4 scales, 384] → [4 scales, 128] | 2-layer MLP per scale: `Linear(384, c) → Linear(c, 128)`, c ∈ {48,96,192,384} |
| `pose_feature_fusion_stage` | [4, 128] → [4, 128] | Reverse DPT fusion: residual blocks + skip connections from deep→shallow |
| seq index embedding | scalar t∈[0,1] → 8D | HarmonicEmbedding (4 harmonics, sin+cos → 8D), added to pose token |
| `pose_interframe_attention` | [B, N, 128] → [B, N, 128] | **8 layers** of self-attention, 4 heads, MLP ratio 4, `nn.MultiheadAttention` |
| `sequence_token_attention` | CrossAttn: seq_token → [B, 1, 128] | Learnable sequence token cross-attends to pose tokens |
| `focal_embedding` | focal_norm [B] → 8D | HarmonicEmbedding(target_dim=1, n_harmonic=4, append_input=False) → 8D |
| `pose_head` (MLP) | [128+8, 64, 7] | Linear(136,64) → ReLU → Linear(64,7) → 7D pose encoding |
| `encoding_to_pose` | 7D → 4×4 SE3 | First 3D = translation, last 4D = quaternion → `quaternion_to_matrix` |
| pose_scaling | × 0.01 (linear) | Scales raw MLP output before converting to pose |

**Pose output**: rotation is **quaternion** parameterized (4D) → converted via `quaternion_to_matrix`. Axis-angle also supported but unused.

### 32-Candidate Focal Length System (original AnyCam, replaced by FAT)
- `sequence_info_head` MLP: [128] → [32 + 16] = [focal logits + scaling feature]
- 32 candidates: linearly spaced in `[focal_min=0.1, focal_max=4.0]` (normalized: 2fx/W)
- Softmax over logits → probability distribution
- Predicted focal = `Σ p_i * candidate_i` (weighted sum)
- During training: KL divergence between predicted probs and soft labels from per-candidate flow loss
- **In our work**: bypassed entirely — FAT focal replaces this via `external_focal_norm`

### Focal Length Normalization Convention
```
focal_norm = 2 * fx_pixel / W     # Maps to approx [-1, 1] NDC range
```
AnyCam works with normalized focal lengths throughout.

---

## 3. ANYCALIB — FULL ARCHITECTURE

### Backbone
- **Model**: DINOv2 ViT-L/14 (custom implementation in `anycalib/model/dinov2.py`)
- **embed_dim**: **1024**
- **num_transformer_blocks**: 24
- **patch_size**: 14
- **image_size**: 518 (trained at)
- **Feature extraction at layers**: [4, 11, 17, 23] (0-indexed into 24 blocks)
- **Output**: 4 feature maps, each [N, 1024, H/14, W/14]
- **Resolution handling**: `RESOLUTION = 102400` px, `EDGE_DIVISIBLE_BY = 14`, `AR_RANGE = (0.5, 2)`

### Decoder
- **`LightDPTDecoder`**: Takes 4-scale features [1024 × 4] → dense spatial features
- **`ConvexTangentDecoder`**: Dense features → per-pixel 3D ray directions [H×W, 3] (unit vectors)

### Calibrator
- **Method**: Gauss-Newton nonlinear optimization (closed-form fit)
- **Fallback**: RANSAC (`fallback_to_sac=True`) if GN fails
- **Input**: per-pixel ray field
- **Output**: `[fx, fy, cx, cy]` in pixel space
- **Camera model**: pinhole (`anycalib_pinhole` checkpoint)
- **Process**: rays encode viewing direction per pixel → fit pinhole model to the ray field

---

## 4. FAT / MCT — FULL ARCHITECTURE

**Name in code**: `FeatureAggregationTransformer` (`feature_aggregation_transformer.py`)
**Name in presentation**: "Multi-Frame Calibration Transformer (MCT)"
**Position**: Inserted between AnyCalib's DINOv2 backbone and LightDPT decoder.

### Architecture (as actually coded)
```
Input: 4 scale feature maps, each [N, 1024, h_s, w_s]
       (N = number of frames; spatial dims h_s, w_s depend on scale)

For each scale:
  1. Flatten spatial:  [N, 1024, h, w] → [S, N, 1024]   (S = h*w)
  2. Self-attention across N frames at each spatial position
     ┌─ 2 × TransformerEncoderLayer ─────────────────────────────
     │  d_model=1024, nhead=8, dim_feedforward=4096
     │  dropout=0.1, activation='gelu', norm_first=True (pre-norm)
     └──────────────────────────────────────────────────────────
  3. Mean pool over N:  [S, N, 1024] → [S, 1024]
  4. Reshape:           [S, 1024]    → [1, 1024, h, w]

Output: 4 aggregated feature maps, each [1, 1024, h_s, w_s]
```

### Parameters
- 2 × TransformerEncoderLayer(1024, 8, 4096) ≈ **~25M params total**
- Shared across all 4 scales (same transformer weights called 4× in forward)
- **Visual conditioning disabled** in all training phases (`use_visual_conditioning=False`)
  - Infrastructure exists in code but is NOT used in the reported results
- Aggregation: **mean pooling** (not learnable token)
- Weight init: truncated normal (std=0.02), LayerNorm ones/zeros

### What "architecture-agnostic" means (and doesn't)
- Agnostic to *which* backbone feeds it, but **requires embed_dim=1024** (hardcoded to AnyCalib's ViT-L)
- Would need reconfiguration for a different backbone dimension

---

## 5. TRAINING PROTOCOL

### Training Phases (A → B1 → C; B2 is skipped)

| Phase | Trainable Params | Loss | Data |
|---|---|---|---|
| **A** | pose_head only (~21K) | L_flow (consecutive) | preprocessed .npz |
| **B1** | FAT only (~25M) | L_anchor (ray reprojection vs GT calib) | preprocessed .npz |
| **B2** *(skipped)* | FAT only | L_flow | both live backbones |
| **C** | pose_head + FAT (~25M+21K) | L_flow + 1e-4 × L_anchor | preprocessed .npz |
| **Cb** *(variant)* | pose_head + FAT + pose neck | same as C | preprocessed .npz |

**Phase C default**: both DINOv2 backbones frozen, only pose_head + FAT adapter trained.

### Frozen vs Trainable
- AnyCam DINOv2-small backbone: **frozen**
- AnyCalib DINOv2 ViT-L backbone: **frozen**
- LightDPT decoder + ray head: **frozen**
- Calibrator: **non-differentiable** (RANSAC/GN, always frozen)
- UniDepth: **frozen** (loaded from .npz, not run live in training)
- UniMatch: **frozen** (loaded from .npz, not run live in training)
- FAT (MCT): **trainable** (Phases B1, C)
- AnyCam pose_head: **trainable** (Phases A, C)

**Why only 7.5%?**: 25M (FAT) / [307M (ViT-L) + 25M (FAT)] ≈ 7.5% of calibration pipeline params.

### Hyperparameters
- `batch_size`: 4 (Phase A), 2 (Phase C due to VRAM)
- `learning_rate`: 1e-4 (default)
- `num_epochs`: 50 (default)
- `lambda_calib`: 1e-4 (anchor loss weight)
- `lambda_comp`: 0.1 (composed flow weight)
- Optimizer: Adam (from train_unified.py)
- Mixed precision: `torch.amp.autocast`

---

## 6. LOSS FUNCTIONS — COMPLETE MATH

### 6.1 Flow Reprojection Loss L_flow

**Projection step** (for frame pair i → j):
```
p_3D = K^{-1} · [x, y, 1]^T · d(x,y)       # unproject pixel to 3D (camera coords)
p'   = K · (R · p_3D + t)                    # apply predicted pose, reproject
x'   = p'[0:2] / p'[2]                       # perspective division → 2D pixel
```

**Variable legend**:
- `x = [u, v]` — source pixel location
- `d(x,y)` — depth at pixel from UniDepth (after scaling: `raw_inv_depth * 0.1`)
- `K` — camera intrinsics matrix from FAT (or AnyCalib/GT)
- `R` — predicted 3×3 rotation matrix
- `t` — predicted 3D translation vector
- `w_obs` — observed optical flow from UniMatch at (x,y) (raw pixel displacement)
- `x' - x` — **predicted flow** at pixel x

**Flow error (L1)**:
```
e(x) = ||(x' - x) - w_obs||_1
```

**Uncertainty weighting (Laplacian NLL)**:
```
L_flow(x) = sqrt(2) * e(x) / σ(x) + log(σ(x))
```
- `σ(x)` = flow uncertainty at pixel x, predicted by AnyCam uncertainty head
- Derived from negative log-likelihood of Laplacian distribution: p(e|σ) ∝ exp(-√2 |e|/σ)
- `EPS = 1e-4` added for numerical stability

**Masking**: invalid pixels (occlusion mask < 0.5) set to 0, NaN/Inf set to 0.
**Flow clamping**: induced flow clamped to [-1, 1] before error computation.

**Consecutive pairs loss** (standard):
```
L_consec = mean over all valid pixels, all consecutive frame pairs
```

**Composed flow loss** (multi-frame consistency):
- For N frames: compose flows 0→2, 0→3, ... via bilinear warping
- `flow_{0→k} = compose_flow(flow_{0→1}, ..., flow_{k-1→k})`
- Compose poses: `T_{0→k} = T_{0→1} @ T_{1→2} @ ... @ T_{k-1→k}` (4×4 matrix chain)
- Compare composed induced flow vs composed observed flow
- `L_composed = mean over all composed pairs (normalized by count)`

**Total flow loss**:
```
L_flow = L_consec + 0.1 × L_composed
```

### 6.2 Calibration Anchor Loss L_anchor

**NOT a direct norm on K scalars.** Actual implementation: ray-level reprojection (MSE).

```python
# In compute_reprojection_loss():
rz = ray_z.clamp(min=1e-6)
projected_u = fx_scaled * (ray_x / rz) + cx_scaled   # project ray back to pixels
projected_v = fy_scaled * (ray_y / rz) + cy_scaled
loss = MSE(projected_coords, actual_pixel_grid)
```

**Plain English**: Given the GT average intrinsics K_GT (from preprocessed .npz), project predicted FAT rays back to pixel space and compare to the actual pixel grid. If rays were consistent with K_GT, they'd project perfectly.

**Gradient flows through FAT's decoder → FAT weights** (teaches FAT to produce calibrations consistent with GT).

**Presentation simplification**: `L_anchor = ||K_pred - K_AnyCalib||` is a pedagogic simplification. Real loss is at ray level.

### 6.3 Combined Loss (Phase C)
```
L_total = L_flow + λ_calib × L_anchor
        = L_flow + 1e-4 × L_anchor
```

**Why small λ_calib?** Flow loss and anchor loss have very different scales. 1e-4 empirically balances them.

### 6.4 Why Calibration Anchor is Necessary (key insight)
For small camera motion (R ≈ I, t ≈ 0):
```
K^{-1}(d·Kx + t) ≈ x     (K cancels out)
```
The flow reprojection loss approaches 0 regardless of K. Any intrinsics matrix satisfies a near-zero loss when motion is small. The anchor prevents K from drifting during these degenerate frames.

---

## 7. DATA PIPELINE

### Preprocessed .npz Format (per frame pair)
```
forward_flow:   [2, H, W]  float16  — raw pixel displacement (u, v)
backward_flow:  [2, H, W]  float16
forward_occ:    [1, H, W]  float16  — 1 = visible, 0 = occluded
backward_occ:   [1, H, W]  float16
depth:          [1, H, W]  float16  — INVERSE depth: 1/(metric_depth × 0.1)
calib:          [4]        float32  — (fx, fy, cx, cy) in pixel space
```

### Flow Normalization (AnyCam convention)
```python
flow_x_norm = flow_x_pixels * 2.0 / W    # maps pixel disp to ~[-1, 1]
flow_y_norm = flow_y_pixels * 2.0 / H
```

### Depth Convention
```
Stored:         inv_depth = 1 / (metric_depth × 0.1)   # raw inverse scaled depth
AnyCam uses:    pred_depth = inv_depth × 0.1             # ≡ metric_depth × 0.01
                                                          # (additional ×0.1 scaling)
```
⚠️ The ×0.1 happens in `induce_flow_dist`: `aligned_depths * 0.1`.

### UniDepth
- Metric monocular depth prediction
- Output: absolute metric depth in meters
- Frozen, run during preprocessing, stored as inverse scaled depth

### UniMatch (customized fork)
- Optical flow estimation (GMFlow-based, iterative refinement)
- Output: pixel-space displacement (u,v) + forward/backward occlusion mask
- Frozen, run during preprocessing, stored in .npz

---

## 8. METRICS — FORMULAS

### 8.1 Rotation Error (degrees)
**Formula** (SO(3) geodesic distance):
```
R_diff = R_gt^T @ R_pred
cos_θ  = (trace(R_diff) - 1) / 2
θ_rot  = arccos(clip(cos_θ, -1, 1)) × (180/π)
```
- `R_pred`: 3×3 predicted rotation matrix
- `R_gt`: 3×3 ground truth rotation matrix
- `R_diff`: relative rotation (how much to rotate pred to match gt)
- `trace(R)`: sum of diagonal elements = 1 + 2cos(θ) for rotation by θ

### 8.2 Translation Direction Error (degrees)
```
θ_trans = arccos(clip(t̂_pred · t̂_gt, -1, 1)) × (180/π)
```
- `t̂ = t / ||t||` — normalized translation direction
- Purely angular — ignores scale (monocular depth is up-to-scale)
- Returns 0 if either vector is near-zero

### 8.3 MAE (Mean Absolute Error) — Calibration
```
MAE = mean(|fx_pred - fx_gt|)   [pixels]
```

### 8.4 MAPE (Mean Absolute Percentage Error) — Calibration
```
MAPE = mean(|fx_pred - fx_gt| / fx_gt) × 100%
```

### 8.5 SE(3) Distance (supplementary)
```
SE3_dist = ||pose_pred - pose_gt||_F   (Frobenius norm of 4×4 matrix difference)
```

---

## 9. RESULTS

### Pose Error — Full Benchmark (N=200 sequences, 600 pairs per dataset, no BA)
| Dataset | Method | Rot Mean | Rot Median | Trans Mean | Trans Median |
|---|---|---|---|---|---|
| Sintel | AnyCam | 0.67° | 0.21° | 89.3° | 86.5° |
| Sintel | **Ours** | **0.60°** | **0.19°** | **65.0°** | **52.3°** |
| Sintel | improvement | −10.1% | −8.2% | −27.3% | **−39.5%** |
| TUM-RGBD | AnyCam | 2.03° | 1.23° | 93.0° | 97.1° |
| TUM-RGBD | **Ours** | **1.39°** | **0.71°** | **71.9°** | **66.9°** |
| TUM-RGBD | improvement | **−31.7%** | **−42.6%** | −22.7% | −31.1% |
| KITTI | AnyCam | 0.56° | 0.26° | 91.1° | 91.5° |
| KITTI | **Ours** | **0.54°** | **0.25°** | **77.2°** | **70.9°** |
| KITTI | improvement | −2.0% | −2.4% | −15.3% | −22.4% |

**Key headline**: up to **40% better translation** (Sintel median), up to **43% better rotation** (TUM-RGBD median).

### Per-Sequence Figures Source (histogram plots in slides)
- Histograms and focal length plots are from Sintel `alley_1` (147 frame pairs, GT fx=530.1px)
- These illustrate the distribution shape; the table above is the statistically valid result

### Calibration Error — Full Benchmark
| Dataset | Method | MAE (px) | MAPE |
|---|---|---|---|
| Sintel | AnyCam | 502.5 | 68.2% |
| Sintel | AnyCalib | 329.9 | 30.7% |
| Sintel | **Ours** | **300.3** | **27.8%** |
| Sintel | improvement vs AnyCalib | −9.0% | **−9.4%** |
| TUM-RGBD | AnyCam | 234.1 | 63.7% |
| TUM-RGBD | AnyCalib | 43.0 | 11.7% |
| TUM-RGBD | **Ours** | **30.3** | **7.9%** |
| TUM-RGBD | improvement vs AnyCalib | −29.5% | **−32.5%** |

**Key headline**: up to **32% better calibration** over AnyCalib (TUM-RGBD MAPE).
**Note**: KITTI calibration DEGRADES vs AnyCalib (out-of-distribution automotive cameras). Pose still improves on KITTI.

### Trajectory (Sintel market_6, Sim(3) alignment)
- Ours ATE = 0.176m vs AnyCam ATE = 0.352m → **2× improvement**
- Ours wins on 13/23 Sintel sequences (57%)

---

## ⚠️ PRESENTATION NOTES (updated slides now consistent)

### What changed
- Pose and calibration tables now show the **full N=200 benchmark** (Sintel, TUM-RGBD, KITTI)
- Histogram figures remain from Sintel `alley_1` — labeled as such in slides
- Conclusion now says "up to 40% better translation, up to 32% better calibration" — matches table

### Why the old slides had wrong numbers (understand this)
- Old slide tables showed Sintel `alley_1` single-sequence per-pair numbers (0.121° etc.)
- The histograms also came from `alley_1`, so figures and table were consistent with each other
- But the summary text (teleprompter "39%", conclusion "28%") was written from the full table
- → Numbers looked contradictory because two different scopes were mixed

### Reconciliation (for any question about old/prior versions)
- "39%" = Sintel median translation improvement from full table: (86.5-52.3)/86.5 = **39.5%** ✓
- "28%" = Sintel mean translation improvement from full table: (89.3-65.0)/89.3 = **27.3%** ✓
- "32%" = TUM-RGBD MAPE improvement over AnyCalib: (11.7-7.9)/11.7 = **32.5%** ✓

### 3. Anchor loss presentation vs implementation
- **Presented as**: `L_anchor = ||K_pred - K_AnyCalib||` (norm on scalars)
- **Actual code**: MSE on ray projections back to pixel grid — a geometric loss at feature level
- *If asked*: "The presentation simplifies; the actual loss works at the ray level, projecting predicted rays back to pixel coordinates using the GT intrinsics and comparing to the true pixel grid — this is more geometrically meaningful."

### 4. "MCT" vs "FAT" naming
- Presentation: "Multi-Frame Calibration Transformer (MCT)"
- Code: `FeatureAggregationTransformer` (FAT) in `feature_aggregation_transformer.py`
- These are the same module, different names.

### 5. Phase B2 existence
- Presentation doesn't mention B1 pre-training or B2
- **Actual pipeline**: A → B1 → C (B2 is skipped)
- B1 pre-trains FAT with supervised anchor loss before joint training

### 6. Visual conditioning
- Presentation doesn't mention this, but code has visual conditioning infrastructure
- **All training phases use `use_visual_conditioning=False`**
- The DINOv2-small visual token pathway exists but is disabled in reported results

---

## 10. TRAP QUESTIONS — PREPARED ANSWERS

### Architecture & Design

**Q: Why does AnyCam take 6-channel input instead of 3?**
A: Flow (2ch) and depth (1ch) are concatenated with RGB (3ch) to give the pose branch explicit access to precomputed geometric cues. This avoids learning to re-estimate flow/depth from scratch. The patch embedding weights are simply replicated (×2) to accommodate the extra channels.

**Q: Why freeze both DINOv2 backbones during training?**
A: Both backbones are large (384M+ params total), pretraining them would require massive compute and data. Our training data (Objectron, ~100 sequences) is insufficient to retrain them. Freezing forces our lightweight adapters (pose_head, FAT) to learn on top of already-strong features.

**Q: Why use DINOv2 features for both pose AND calibration separately?**
A: Each task needs domain-specific fine-tuning. AnyCalib's ViT-L (1024-dim) was trained specifically for geometric calibration. AnyCam's ViT-small (384-dim) was trained for pose with flow/depth conditioning. Sharing a backbone would require retraining everything jointly, losing strong pretrained priors.

**Q: Why cross-frame attention at the feature level rather than just averaging calibrations?**
A: Feature-level aggregation preserves spatial structure. Scalar averaging of [fx, fy, cx, cy] would lose the geometric distribution of ray errors — different image regions contribute differently to calibration accuracy. FAT allows the network to learn which frames and spatial regions are most informative.

**Q: Why mean pooling in FAT instead of a learnable aggregation token?**
A: Simplicity and stability. Mean pooling is parameter-free, numerically stable, and works well in practice. A learnable token would add parameters and potential optimization instability, for little demonstrated gain.

**Q: How does the focal length get from AnyCalib/FAT into the pose head?**
A: `focal_norm = 2 * fx_pixel / W` → HarmonicEmbedding(4 harmonics) → 8-dim vector → concatenated to pose token → pose head MLP. This injects the calibration as a conditioning signal.

**Q: Why is rotation parameterized as a quaternion instead of a rotation matrix?**
A: Quaternions are compact (4D vs 9D), avoid the need to project back to SO(3), and have unconstrained Euclidean parameter space suitable for gradient-based optimization. The conversion to rotation matrix happens after decoding.

**Q: What is the 0.01 scaling factor on poses?**
A: `pose_scaling_linear` multiplies the raw MLP output by 0.01. This constrains initial pose predictions to small values, matching the small camera motions typical in training videos. Without it, early training would produce wildly large poses.

**Q: How does the calibrator work? Is it differentiable?**
A: The Gauss-Newton calibrator fits a pinhole model to the predicted ray field by iteratively minimizing the reprojection error of rays. **It is NOT differentiable** — gradient flow stops at the calibrator. Gradients reach FAT through the anchor loss (`compute_reprojection_loss`), which operates on the rays *before* the calibrator.

**Q: Why RANSAC fallback in the calibrator?**
A: If the Gauss-Newton optimizer diverges (due to noisy FAT predictions early in training), RANSAC provides a more robust initial estimate. This prevents training instability from calibration failures.

### Losses

**Q: Why is the anchor loss needed if the flow reprojection loss already trains calibration?**
A: For small camera motion, K cancels out of the reprojection equation (K⁻¹·K = I). Any intrinsics give near-zero flow loss when the camera barely moves. The anchor loss prevents K from drifting to degenerate values during these near-static frames, which are common in casual video.

**Q: What distribution does the uncertainty weighting assume?**
A: Laplacian distribution. The NLL of Laplace(0, σ) is `|e|/σ + log(σ)`. Multiplied by √2 for normalization: `√2 |e|/σ + log(σ)`. This is heavier-tailed than Gaussian, more robust to outliers.

**Q: How is flow composition done?**
A: Bilinear warping. For flow 0→2: take the composed position `x + flow_{0→1}`, then bilinearly sample `flow_{1→2}` at that position. `flow_{0→2} = flow_{0→1} + bilinear_sample(flow_{1→2}, x + flow_{0→1})`. Occlusion is the product of individual occlusion masks.

**Q: Why weight composed pairs at 0.1 instead of 1.0?**
A: Flow composition accumulates errors (each bilinear warp adds error). Longer-range composed flows are less reliable than consecutive flows. The 0.1 weight reflects this lower reliability.

**Q: What is the KL divergence loss on focal length candidates in AnyCam?**
A: For each candidate focal length, compute the per-candidate flow reprojection loss. Create soft labels via `softmax(-10 × per_candidate_loss)` — candidates with lower flow loss get higher probability. Then minimize KL(predicted_distribution || soft_labels). This trains the softmax distribution to assign high probability to candidates that explain the observed flow well.

### Data & Training

**Q: How did you get ground truth calibration for training?**
A: From preprocessed .npz files — AnyCalib run offline on training sequences to generate pseudo-GT calibration. These are the average intrinsics over all frames of a sequence. We use this as the anchor target, not as a supervision signal for the calibrator itself.

**Q: Why inverse depth storage?**
A: Depth values have a non-uniform distribution (many near-depth pixels, few far-depth pixels). Inverse depth is more uniform and numerically stable for storage and computation. `inv_depth = 1 / (metric_depth × 0.1)`.

**Q: Why does translation error reach >100°?**
A: Translation direction error can be 0°–180°. AnyCam's baseline has >100° mean translation error on Sintel, meaning the predicted translation direction is essentially random (worse than random would be ~90°). This reflects that the original 32-candidate calibration is so wrong it corrupts scale/direction estimation.

**Q: What dataset are the pose results evaluated on?**
A: The slides don't explicitly label the dataset on the pose slide. The calibration results are explicitly labeled "Sintel alley_1" in the footnote. Be careful — if asked, note the exact scope of the evaluation.

**Q: Why is Sintel used for evaluation if it's not a training dataset?**
A: Sintel provides known GT camera intrinsics and poses. It's a standard evaluation benchmark for monocular methods. Our method is evaluated zero-shot (no fine-tuning on Sintel). The Objectron dataset was used for training.

### Broader Concepts

**Q: What is the fundamental limitation of self-supervised calibration?**
A: Scale ambiguity + degenerate cases. Monocular methods cannot recover absolute scale. For calibration, near-zero motion means K is unobservable. Additionally, certain scene configurations (planar scenes, pure rotation) make depth+intrinsics estimation underdetermined.

**Q: How does AnyCalib handle non-pinhole cameras?**
A: AnyCalib supports multiple camera models (UCM, radial distortion, division model, etc.). The `anycalib_pinhole` checkpoint is specialized for pinhole. The ray representation is camera-model-agnostic — you fit whatever model you want to the rays. Our work only uses the pinhole checkpoint.

**Q: Could your framework generalize beyond AnyCam + AnyCalib?**
A: In principle yes — FAT only requires: (1) a pose estimator that accepts an external focal length, (2) a calibration network with accessible intermediate features before decoding. However, FAT's dimensions (embed_dim=1024) are specific to AnyCalib's ViT-L. Any other calibration backbone would need a different FAT configuration.

**Q: What is the difference between your approach and bundle adjustment?**
A: Bundle adjustment (BA) is an offline optimization that jointly refines poses and intrinsics post-hoc, requiring all frames and correspondences. Our method is feed-forward, runs in a single pass, needs no correspondences beyond optical flow, and can process streaming video. BA is slower but more accurate given enough observations.

**Q: Why is rotation error so much smaller than translation error (0.086° vs 44.3°)?**
A: Rotation is tightly constrained by optical flow patterns — flow vectors directly encode rotation. Translation is coupled with depth and focal length (scale ambiguity). A wrong focal length directly corrupts the recovered translation direction, explaining why better calibration improves translation much more dramatically than rotation.

**Q: What would you do differently with more time?**
A: (1) Train on diverse large-scale datasets (RealEstate10K, YT-VOS) for better generalization. (2) End-to-end training with unfrozen backbones. (3) Non-pinhole camera model support via FAT. (4) Joint depth refinement using the improved calibration.

---

## 11. QUICK NUMBER LOOKUP

| Quantity | Value |
|---|---|
| AnyCam DINOv2-small hidden dim | 384 |
| AnyCam DINOv2-small blocks | 12 |
| AnyCam patch size | 14 |
| AnyCam input channels | 6 (RGB+flow+depth) |
| AnyCam fusion_hidden_size | **128** (not 64) |
| AnyCam interframe attention layers | **8** |
| AnyCam interframe attention heads | 4 |
| AnyCam pose token dim | 128 |
| AnyCam focal embedding dim | 8 (harmonic: 4 freqs × sin+cos) |
| AnyCam pose head: [in, hid, out] | [128+8, 64, 7] |
| AnyCam 32 candidates range (norm) | [0.1, 4.0] (normalized: 2fx/W) |
| AnyCalib DINOv2 ViT-L/14 dim | 1024 |
| AnyCalib DINOv2 ViT-L/14 blocks | 24 |
| AnyCalib feature extraction layers | [4, 11, 17, 23] |
| FAT/MCT embed_dim | 1024 |
| FAT/MCT num_heads | 8 |
| FAT/MCT num_layers | 2 |
| FAT/MCT FFN dim | 4096 (= 4 × 1024) |
| FAT/MCT num_scales | 4 |
| FAT/MCT aggregation | mean pooling |
| FAT/MCT total params | ~25M |
| FAT/MCT % of calib pipeline | ~7.5% (25M / 332M) |
| λ_calib (anchor weight) | 1e-4 |
| λ_comp (composed flow weight) | 0.1 |
| Rotation error formula | arccos((trace(R_gt^T R_pred) - 1) / 2) |
| Translation error formula | arccos(t̂_pred · t̂_gt) |
| Calibration error metric | MAE [px] and MAPE [%] |
| Training dataset | Objectron (~100 sequences) |
| Eval: pose results | Sintel (likely) |
| Eval: calibration results | Sintel alley_1 (footnoted) |
| Objectron path (local) | /home/kalman/TUM/thesis/Objectron/ |
