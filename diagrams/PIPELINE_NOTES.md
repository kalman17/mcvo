# Pipeline Block-by-Block Reference

Quick reference for understanding and defending each block in the diagrams during the presentation.

---

## AnyCam Pipeline (Diagram 1)

### Inputs

**Images I^i** — Raw video frames (336×336, cropped/resized). N consecutive frames from a video (N=2-8 during training).

**UniDepth** — Off-the-shelf monocular depth estimator. Produces per-pixel depth maps D^i for each frame. These are concatenated as extra input channels to the backbone (not used standalone). Frozen, never trained.

**UniMatch** — Off-the-shelf optical flow network. Produces dense flow fields F^{i→j} between consecutive frame pairs. Also concatenated as extra input channels. Frozen.

**Why are depth+flow inputs?** The backbone receives [RGB + depth + flow] as a 6-8 channel input. This gives the network geometric cues beyond just appearance — depth provides scale information, flow provides motion information. The backbone was trained from scratch to expect these channels.

### DINOv2-small (ViT-S/14)

- **What**: Vision Transformer (small variant), pretrained by Meta on large-scale data using self-supervised learning (DINOv2)
- **Architecture**: ViT-S/14 = 12 transformer layers, patch size 14×14, embedding dim 384
- **In AnyCam**: Modified to accept extra input channels (depth+flow). Extracts CLS tokens from 4 intermediate stages (layers 3, 6, 9, 12)
- **Output**: 4 CLS tokens [384-dim each] per frame — compact per-frame representations
- **Also outputs**: Spatial feature maps used by the Uncertainty Head (DPT decoder → per-pixel σ)
- **Frozen?** In our method: frozen. In original AnyCam training: trained end-to-end.

### Pose Neck

- **What**: Custom transformer-based processing module that takes per-frame CLS tokens and produces pose tokens
- **Steps**:
  1. **Reassemble**: Projects 4 CLS tokens from [384] to [128] via learned linear layers
  2. **Fusion**: Merges the 4 multi-scale features into a single representation per frame
  3. **8-layer Self-Attention**: All N frames attend to each other simultaneously. This is where inter-frame information exchange happens — frame 1's features can influence frame 3's pose prediction
  4. **Sequence Token**: A learnable token that attends to all pose tokens via cross-attention, producing a single summary of the whole sequence
- **Output**:
  - N−1 **pose tokens** [128-dim] — one per consecutive frame pair (1→2, 2→3, etc.)
  - 1 **sequence token** [128-dim] — used by Sequence Head for focal length selection
- **Key insight**: The self-attention across frames is what makes AnyCam work on sequences, not just pairs. Each pose prediction is informed by the full sequence context.

### 32-Candidate System

**The problem**: Camera intrinsics (focal length f) are entangled with camera motion in optical flow. The same flow pattern could mean "large rotation with telephoto lens" or "small rotation with wide-angle lens." Predicting f directly as a free variable makes training unstable.

**AnyCam's solution**: Don't predict f — instead, try all possibilities.

- **32 predefined focal lengths** {f₁, ..., f₃₂}: Fixed set spanning narrow to wide FoV, log-spaced
- **Pose Head ×32**: A single MLP ([128] → [64] → [7×32]) that outputs 32 pose predictions simultaneously — one for each "what if the focal length were f_k?" scenario. Shared weights, wider output layer.
  - Output [7] per candidate = quaternion [4] + translation [3]
- **Sequence Head**: Separate MLP on the sequence token. Outputs 32 likelihood scores. Trained via KL divergence against softmax of inverted flow losses (the candidate whose pose produces the best flow match gets the highest score)
- **Selection**: Softmax over scores → weighted sum of predefined f_k values → final focal length. Best-scoring candidate's pose → final pose.

**Why it's expensive**: 32× redundant computation, and only 32 discrete choices for f.

---

## AnyCalib Pipeline (Diagram 2)

### DINOv2 ViT-L/14

- **What**: Large Vision Transformer, same DINOv2 family but much bigger
- **Architecture**: ViT-L/14 = 24 transformer layers, patch size 14×14, embedding dim 1024
- **Output**: 4 scales of spatial feature maps [1024, h, w] from intermediate blocks (4, 11, 17, 23)
- **Key difference from AnyCam's backbone**: Extracts dense spatial features (not just CLS tokens), and uses the large model (1024-dim vs 384-dim)
- **Frozen**: Always frozen, pretrained DINOv2 weights

### Light-DPT (CNN decoder + ray head)

Two sub-components merged in the diagram:

**Light-DPT decoder**:
- Modified DPT (Dense Prediction Transformer) decoder — but it's actually CNN-based, not a transformer
- Takes the 4 multi-scale feature maps from ViT-L and fuses them
- Uses transposed convolutions (modified to avoid expensive ones) + upsampling
- Output: dense feature map [1, 256, H/7, W/7]

**Ray head (ConvexTangentDecoder)**:
- Takes the decoded features and predicts per-pixel FoV field
- Uses **convex upsampling** (learned upsampling weights) to go from H/7 to full resolution
- Predicts **tangent coordinates θ = (θ_x, θ_y)** in the tangent plane at the optical axis z₁
- These are the **FoV field** — per-pixel angular coordinates in T_{z₁}S²

### FoV Field

- **What**: Per-pixel Field-of-View coordinates θ ∈ ℝ²
- **Representation**: Each pixel gets (θ_x, θ_y) in the tangent plane of the unit sphere at the optical axis
- **Why this representation?**
  - Unit rays on S² aren't closed under addition (can't do standard convolutions)
  - FoV fields in the tangent plane ARE in ℝ² — standard operations work
  - Minimal and unconstrained — no normalization needed during training
- **Conversion to rays**: Exponential map: p = Exp_{z₁}(θ) maps tangent vectors to unit rays on S²

### Closed-form K

- **Input**: Per-pixel rays p ∈ S² and their image coordinates x
- **Method**:
  1. The projection equation π(p) = x gives linear constraints once principal point (c) and aspect ratio (a) are estimated
  2. First estimate c and a via linear system (Eq. 11 in paper: uYa − Yac_x + Xc_y = vX)
  3. Then focal length f and distortion coefficients from Ax = b (overconstrained linear system)
  4. Final refinement: 5 iterations of Gauss-Newton on angular distance between predicted and fitted rays
- **Model-agnostic**: Works for pinhole, Brown-Conrady, Kannala-Brandt, fisheye, etc.
- **Non-differentiable**: Uses RANSAC for outlier rejection → can't backprop through this
- **Output**: K = [f_x, f_y, c_x, c_y]

---

## Our Pipeline (Diagram 3)

### What changed from AnyCam?

1. **Removed**: 32-Candidate System (Pose Head ×32 + Sequence Head)
2. **Added**: MCT between ViT-L and Light-DPT
3. **Added**: Focal Embedding — extracts f from K, harmonic encoding → [8-dim]
4. **Modified**: Pose Head now takes [128+8=136] input (pose token + focal embedding)

### MCT (Multi-Frame Calibration Transformer)

- **What**: Our main architectural contribution. A transformer that aggregates multi-frame features from AnyCalib's backbone into a single consistent representation.
- **Where it sits**: Between DINOv2 ViT-L (backbone) and Light-DPT (decoder) — inserted into AnyCalib's pipeline
- **Why**: AnyCalib processes each frame independently → noisy, inconsistent calibration across frames. MCT enables cross-frame communication.

**Architecture** (~25M params):
- Processes each of the 4 scales independently (shared weights across scales)
- For each scale:
  1. Flatten spatial dims: [N, 1024, h, w] → [h×w, N, 1024]
  2. Each spatial position becomes an independent batch item
  3. **2-layer Transformer Encoder** (pre-norm):
     - Multi-Head Attention: 8 heads, d_head=128, Q/K/V projections 1024→1024
     - FFN: 1024 → 4096 → 1024 (GELU)
     - Residual connections + LayerNorm
  4. Attention is across the **frame dimension** — at each pixel, frames attend to each other
  5. No spatial attention — pixels are independent
  6. Mean pool across N frames → single output per pixel
  7. Reshape: [h×w, 1024] → [1, 1024, h, w]

**Key insight**: The same camera produced all frames → geometric patterns (focal length, distortion) are consistent. The MCT learns to exploit this by letting frames share calibration information at every spatial position.

**Optional visual conditioning**: DINOv2-small CLS tokens [N, 384] can be projected to [N, 1024] and concatenated with the spatial tokens before attention, providing global image context. Discarded after attention.

### Focal Embedding

- **What**: Harmonic positional encoding that converts scalar focal length to [8-dim] vector
- **Code**: `self.focal_embedding = PoseEmbedding(target_dim=1, n_harmonic_functions=4, append_input=False)`
- **Process**: f (from K) → normalize by image size → 4 harmonics (sin/cos at different frequencies) → [8-dim]
- **Why**: The pose head needs to know the focal length to correctly predict rotation vs translation. A raw scalar doesn't provide enough signal — harmonic encoding gives the MLP richer features to work with.
- **This is NEW**: Original AnyCam's pose head had focal_embed_dim=0 (no focal input). We added the [8-dim] slot.

### Pose Head (modified)

- **Original**: [128] → [64] → [7×32] (outputs all 32 candidates, no focal input)
- **Ours**: [128+8=136] → [64] → [7] (single output, conditioned on actual focal length)
- **Output [7]**: quaternion [4] + translation [3] for relative pose between consecutive frames

---

## Potential Questions & Answers

**Q: Why not just predict focal length directly instead of 32 candidates?**
A: AnyCam tried this — training is unstable because f is entangled with camera motion in the flow. The 32-candidate approach sidesteps this by trying all possibilities. Our approach sidesteps it differently: we get f from a separate dedicated calibration network (AnyCalib+MCT).

**Q: Why do you need MCT? Why not just use AnyCalib per-frame?**
A: Per-frame AnyCalib gives noisy, inconsistent calibration. A single camera has one fixed focal length — MCT enforces this by letting frames share information through cross-frame attention. The result is a single consistent calibration.

**Q: Why 4 scales?**
A: DINOv2 ViT-L has 24 transformer blocks. AnyCalib extracts features at blocks 4, 11, 17, 23 — early to late, capturing different levels of abstraction. The DPT decoder fuses these multi-scale features for dense prediction.

**Q: Is the MCT differentiable?**
A: Yes, the MCT itself is fully differentiable. The non-differentiable part is the closed-form K fitting (RANSAC). During training, we use a calibration anchor loss (K vs AnyCalib pseudo-GT) to provide gradient signal to MCT despite the non-differentiable calibrator.

**Q: Why harmonic embedding for focal length?**
A: Same principle as positional encoding in transformers — a scalar has limited expressiveness for an MLP. Harmonic encoding (sin/cos at multiple frequencies) projects it into a richer space where the network can learn non-linear relationships.

**Q: How many parameters are trainable?**
A: ~27.5M out of ~370M total (7.5%). Trainable: MCT (~25M), Pose Neck (~2.5M), Pose Head (~21K). Everything else is frozen.

**Q: What's the training loss?**
A: Self-supervised, no GT labels needed:
1. Flow reprojection loss (main): induced_flow(pose, depth, K) vs observed_flow(UniMatch), weighted by uncertainty σ (Laplacian NLL)
2. Pose consistency (forward-backward): P^{i→j} · P^{j→i} ≈ I
3. Composed flow loss: long-range consistency via flow composition (weight 0.1×)
4. Calibration anchor: K_predicted vs K_anycalib (keeps calibration near AnyCalib's estimates)
