# Teleprompter — Defense Presentation

Read naturally, not word-for-word. Each slide ~1.5 min. Total ~15 min.
**SPOKEN** = say this. **REFERENCE** = silent technical notes, for Q&A backup only.

---

## Slide 1: Title

**SPOKEN**: *[No speaking — wait for introduction by committee]*

---

## Slide 2: Outline

**SPOKEN**: "I'll start with the motivation, then introduce the pretrained models we build on — AnyCam for pose and AnyCalib for calibration. Then I'll present our framework for fusing them together, the contributions, the training setup and losses, results, and conclude with future directions."

---

## Slide 3: Motivation

**SPOKEN**: "Recovering camera poses and intrinsics from video is fundamental to 3D vision — autonomous driving, augmented reality, 3D reconstruction all need this.

But casual video makes this really hard. The scenes are dynamic — people walking, cars moving. The cameras are unknown — no calibration information. And there are no 3D labels at scale. Classical methods like COLMAP assume static scenes and fail. Supervised methods need expensive ground truth that simply doesn't exist for most video.

Now, we do have strong pretrained models for individual sub-tasks — depth estimation, optical flow, single-image calibration, pose prediction. But they each work in isolation. There's no system that combines them with joint reasoning. Let me show you two such models and why combining them is the opportunity."

**REFERENCE**:
- "Unlabeled" means no 3D GT, no calibration data, no pose GT — only raw video frames
- SfM (COLMAP) fails on dynamic scenes because it assumes scene points are static across views
- Supervised alternatives: require GT poses (MoCap, GPS/IMU, LiDAR) which is expensive and doesn't generalize
- Key insight: individual sub-task models are already very strong; the gap is in *joint* reasoning

---

## Slide 4: The Calibration Problem in AnyCam

**SPOKEN**: "This is AnyCam — a self-supervised pose estimator published at CVPR 2025, currently state of the art. It predicts camera poses from unlabeled video using flow reprojection losses. The architecture uses multi-frame self-attention for pose reasoning, which is quite effective.

But look at how it handles calibration — the 32-candidate system on the right. It can only choose from 32 predefined focal lengths. The numbers show this is often very wrong: 45% error on Sintel, 64% on TUM-RGBD. And since calibration directly enters the projection equation, bad calibration means bad poses. This is the bottleneck we want to address."

**REFERENCE**:
- **AnyCam backbone**: DINOv2-small (384-dim), patch_size=14, 12 blocks, 6-channel input (3 RGB + 2 flow + 1 depth)
- **Pose head**: `AnyCamPoseTokenHead(in=136, hid=64, out=7)` — 7D output: quaternion (4D) + translation (3D)
- **Focal embedding**: `HarmonicEmbedding(n_harmonic=4)` → 8D; concatenated into pose head input
- **Multi-frame attention**: 8 interframe attention layers, 4 heads, `fusion_hidden_size=128`
- **32-candidate system**: tests 32 predefined f values via flow reprojection KL divergence; sequence head selects best
- **Why calibration affects translation**: focal length sets the scale of the projection cone; wrong f → wrong translation direction and magnitude
- MAPE numbers: Sintel 45.1%, TUM-RGBD 63.7% (from thesis Table 6.2)

---

## Slide 5: AnyCalib — A Dedicated Calibration Network

**SPOKEN**: "Now here's a dedicated calibration network — AnyCalib. Completely different approach: a large DINOv2 backbone, a decoder that predicts per-pixel ray directions, and a closed-form fit to recover intrinsics. Much more accurate than 32 candidates.

But notice: it processes each frame independently. No temporal consistency at all. A single camera has fixed intrinsics throughout a video, yet the per-frame predictions are noisy and inconsistent.

So we have a pose model that's great at poses but bad at calibration, and a calibration model that's accurate but per-frame only. The natural question: what if we fuse them?"

**REFERENCE**:
- **AnyCalib backbone**: DINOv2 ViT-L/14, embed_dim=1024, 24 transformer blocks; feature taps at layers [4, 11, 17, 23] → ~307M params (frozen in our work)
- **AnyCalib decoder**: DPT-style decoder, outputs per-pixel 3D unit ray directions
- **Calibration solver**: algebraic closed-form fit (RANSAC-free) over ray directions → intrinsics (f_x, f_y, c_x, c_y)
- **Why per-frame is noisy**: each frame sees only partial scene; calibration is underdetermined from a single view with little structure
- AnyCalib MAPE: Sintel 30.7%, TUM-RGBD 11.7% — already much better than 32-candidate, but still per-frame

---

## Slide 6: Our Framework — Fusing Calibration & Pose

**SPOKEN**: "This is our framework. We fuse a self-supervised pose estimator with a dedicated calibration network.

The top row is the pose branch — in our case, AnyCam. The bottom row is the calibration branch — AnyCalib's backbone, then our MCT for multi-frame aggregation, then AnyCalib's decoder.

The key: the MCT is inserted between the calibration backbone and decoder. It aggregates features from N frames into one consistent representation. The resulting calibration is injected into the pose head via a focal embedding.

This framework is not specific to AnyCam or AnyCalib — any self-supervised pose estimator could use this calibration branch. We demonstrate it on these two because they're state of the art."

**REFERENCE**:
- **MCT insertion point**: between DINOv2 ViT-L/14 and the DPT decoder, at 4 feature pyramid scales
- **Multi-scale features**: DINOv2 ViT-L/14 outputs at layers [4,11,17,23]; each scale has embed_dim=1024; MCT processes all 4 independently with shared transformer weights
- **Focal embedding injection**: predicted f_x is passed through HarmonicEmbedding → 8D vector, concatenated to pose neck input alongside DINOv2-small features
- **Architecture-agnostic claim**: MCT only touches the calibration branch; pose branch can be any model that accepts an intrinsics embedding
- **N frames**: during training, N=4 (max_ahead=3); during inference can be variable

---

## Slide 7: Contributions

**SPOKEN**: "Two contributions.

First, the Multi-Frame Calibration Transformer — the MCT. It's a lightweight module that aggregates calibration backbone features across N video frames via cross-frame attention. It's architecture-agnostic: you insert it between any feature extractor and decoder. Only 25 million trainable parameters while everything else stays frozen.

Second, we demonstrate that this framework works in practice. Fusing AnyCam with AnyCalib plus the MCT improves both pose accuracy and calibration accuracy over the individual models. And the whole thing is self-supervised — no ground truth labels needed for training."

**REFERENCE**:
- **MCT details**: 2 `TransformerEncoderLayer` blocks per scale (shared weights); each layer: d_model=1024, nhead=8, dim_feedforward=4096, dropout=0.1
- **Aggregation**: concatenate all N frame tokens → transformer → mean pooling over frame dimension → single fused feature per scale
- **~25M params**: only the MCT transformer weights; DINOv2-small (AnyCam), DINOv2 ViT-L/14 (AnyCalib), DPT decoder, UniDepth, UniMatch — ALL frozen
- **7.5% trainable**: 25M / ~330M total
- **Self-supervised**: no GT poses, no GT intrinsics at training time; only raw video + pseudo-GT from frozen AnyCalib (for anchor loss)

---

## Slide 8: Training Setup

**SPOKEN**: "The training view. Blue blocks are frozen pretrained models — both DINOv2 backbones, the decoder, UniDepth, UniMatch. Orange blocks are what we train — the Pose Neck, Pose Head, and MCT.

Only 7.5% of parameters are trained. Everything pretrained stays frozen."

**REFERENCE**:
- **Phase A** (~10 epochs, batch=16): Train Pose Neck + Pose Head only; use AnyCalib per-frame predictions as fixed calibration input; loss = flow reprojection only
- **Phase B1** (~10 epochs, batch=8): Train MCT only; loss = reprojection of MCT-predicted rays vs AnyCalib pseudo-GT rays; Pose Head frozen
- **Phase C** (~50 epochs, batch=2, checkpoint at epoch 5): Train Pose Neck + Pose Head + MCT jointly; loss = L_flow + 1e-4 × L_anchor
- **Phase B2 was skipped**: originally planned end-to-end MCT through frozen pose pipeline; abandoned in favor of joint Phase C
- **Training data**: ~82k frames, 77k sequences from 4 datasets (not the same as AnyCam's original training sets)
- **UniDepth/UniMatch**: provide depth and optical flow; fully frozen throughout all phases
- **Why freeze backbones**: unfreezing DINOv2 ViT-L/14 caused training divergence

---

## Slide 9: Training Losses

**SPOKEN**: "The main signal is flow reprojection: project a pixel to 3D, apply the predicted pose, reproject, and compare against observed optical flow. Fully self-supervised.

Now here's the key insight about calibration. For small camera displacements, the intrinsic matrix K cancels out of the reprojection equation — K-inverse times K is the identity. So the reprojection loss can be near zero regardless of what K is. Calibration has too much freedom.

That's why we need the calibration anchor. It keeps the calibration near the pseudo ground truth from the calibration network, preventing drift.

This is especially important on real-world videos where dynamic objects create large flow even with small camera motion — you can't reliably filter out near-static frames, so the anchor is essential."

**REFERENCE**:
- **Flow reprojection equation**: `x' = π(K(R·K⁻¹·[x,y,1]ᵀ·d + t))`, observed flow `w = x' - x`; loss = `‖x' - x - w_obs‖`
- **Why K cancels (near-static frames)**: when R≈I, t≈0: `K⁻¹·(d·K·x + t) ≈ x` for any K — any focal length satisfies the loss
- **Composed flow**: `L_flow = L_consecutive + 0.1 × L_composed`; composed = flow for non-adjacent pairs via bilinear warping; weight 0.1 to balance magnitude
- **Anchor loss (actual implementation)**: NOT `‖K_pred - K_AnyCalib‖`; instead, reprojects MCT's predicted rays back to pixel coordinates using GT intrinsics (from AnyCalib pseudo-GT), computes MSE vs actual pixel grid; gradient flows through MCT ray predictions → encourages MCT to match AnyCalib's per-frame output
- **Total loss**: `L_total = L_flow + 1e-4 × L_anchor`; λ=1e-4 keeps anchor as soft regularizer
- **Flow source**: UniMatch (frozen) on raw frame pairs; normalized as `flow_x_norm = flow_x_pixels * 2.0 / W`
- **Depth convention**: stored as `1/(metric_depth × 0.1)`; code multiplies by 0.1 in `induce_flow_dist` → recovers metric scale
- **Laplacian NLL** (in AnyCam's original loss, not ours): `√2·|e|/σ + log(σ)` with learned per-pixel uncertainty σ

---

## Slide 10: Results — Calibration Error

**SPOKEN**: "The plots show calibration stability over time — our method in blue tracks the ground truth much more closely than the noisy per-frame AnyCalib predictions, and without the large outliers from AnyCam's 32-candidate system.

The table compares on two datasets. We exclude KITTI because its automotive cameras are out of distribution for our training data — the MCT doesn't generalize its calibration there.

On TUM-RGBD, which has well-characterized indoor cameras, we reduce focal length error from 43 to 30 pixels — a 32 percent improvement over AnyCalib. On Sintel we see a more modest 9 percent improvement. Both come from the same mechanism: feature-level aggregation across frames preserves geometric structure that per-frame prediction discards."

**REFERENCE**:
- **MAE metric**: `(1/N) Σ |f_x_pred - f_x_gt|` in pixels; absolute focal length error
- **MAPE metric** (not shown, but know it): `(100/N) Σ |f_pred - f_gt| / |f_gt|` in %; Sintel MAPE: AnyCam 68.2% → AnyCalib 30.7% → Ours 27.8%; TUM: 63.7% → 11.7% → 7.9%
- **Full MAE numbers**: Sintel: AnyCam 502.5 → AnyCalib 329.9 → Ours 300.3; TUM-RGBD: AnyCam 234.1 → AnyCalib 43.0 → Ours 30.3
- **KITTI calibration limitation**: long focal-length automotive cameras far outside training distribution; MCT calibration degrades — but pose still improves on KITTI (next slide), showing multi-frame consistency contributes independently
- **Why feature-level > scalar averaging**: scalar mean of per-frame AnyCalib predictions loses spatial/geometric correlations the DPT decoder needs; MCT aggregates at the 1024-dim feature level before decoding
- **Plots**: from Sintel `alley_1` (GT f_x = 530.1 px); illustrative only

---

## Slide 11: Results — Pose Error

**SPOKEN**: "Now with better calibration feeding into the pose head — does pose improve? Yes.

The histograms illustrate the shape of the improvement on one sequence. The table is the full benchmark: 200 sequences, 600 pose pairs per dataset, no bundle adjustment.

On rotation both methods are already quite accurate — under a degree on Sintel and KITTI. The bigger story is translation. The baseline was struggling here, often predicting entirely the wrong direction — errors in the 90-degree range. Better calibration directly reduces this: focal length sets the projection scale, so fixing calibration fixes translation. We see 15 to 40 percent improvements across all three datasets, with the median improvement on Sintel reaching 40 percent."

**REFERENCE**:
- **Rotation metric**: geodesic distance on SO(3): `arccos((trace(R_gtᵀ · R_pred) − 1) / 2)` in degrees
- **Translation metric**: angular direction error (scale-invariant): `arccos(t̂_pred · t̂_gt)` in degrees; direction only — monocular scale is undefined
- **Full numbers**:
  - Sintel: rot 0.67/0.21 → 0.60/0.19 (−10.1%/−8.2%); trans 89.3/86.5 → 65.0/52.3 (−27.3%/−39.5%)
  - TUM-RGBD: rot 2.03/1.23 → 1.39/0.71 (−31.7%/−42.6%); trans 93.0/97.1 → 71.9/66.9 (−22.7%/−31.1%)
  - KITTI: rot 0.56/0.26 → 0.54/0.25 (−2.0%/−2.4%); trans 91.1/91.5 → 77.2/70.9 (−15.3%/−22.4%)
- **Why TUM benefits most on rotation**: handheld indoor, frequent direction changes; multi-frame context stabilizes
- **Why KITTI rotation barely changes**: forward-facing driving, low rotation variance
- **KITTI pose still improves despite calibration degrading**: multi-frame temporal consistency in the pose branch contributes independently
- **Histograms**: from Sintel `alley_1` (147 pairs); illustrative only
- **No BA**: raw feed-forward; bundle adjustment would improve both methods further

---

## Slide 12: Trajectory Visualization — Sintel `market_6`

**SPOKEN**: "A qualitative example. Our trajectory in blue closely follows the ground truth in green. The baseline in orange drifts significantly. This is a direct consequence of improved calibration and translation accuracy combining over the full sequence.

The numbers: our ATE is 0.18 metres, versus 0.35 for AnyCam — a factor of two improvement on this sequence. Across all 23 Sintel sequences, we achieve lower ATE on 57 percent of them."

**REFERENCE**:
- **ATE (Average Trajectory Error)**: Sim(3) alignment first (SE(3) + scale, via Procrustes / evo library), then Euclidean distance between aligned camera positions per frame
- **Sintel market_6 numbers**: Ours ATE = 0.176 m, AnyCam ATE = 0.352 m → 2.0× improvement
- **57% of Sintel sequences**: our method wins on 13/23; largest gains on sequences with substantial camera translation through dynamic scenes
- **Why not 100%**: trained on 4-frame windows; AnyCam used progressive training at 2 then 8 frames — longer windows would likely close this gap
- **Sim(3) alignment**: monocular methods are scale-ambiguous; alignment removes global scale/rotation/translation before comparing local accuracy

---

## Slide 13: Conclusion & Future Work

**SPOKEN**: "To summarise: we proposed a framework for fusing pretrained models for camera geometry. The core contribution is the MCT — a multi-frame calibration consistency module that aggregates features before decoding. We demonstrated it by fusing AnyCam with AnyCalib, achieving up to 40 percent better translation and up to 32 percent better calibration. The framework is self-supervised and architecture-agnostic.

Future work: our model was trained on only 4 frames at a time — extending to longer sequences is a natural next step. And the most promising direction is end-to-end training with everything unfrozen, on a larger and more diverse dataset, which should let the whole system adapt jointly rather than relying on the frozen pretrained priors.

Thank you — happy to take questions."

**REFERENCE**:
- **"Up to 40%"**: median translation improvement on Sintel (39.5% = 86.5° → 52.3°)
- **"Up to 32%"**: MAE improvement on TUM-RGBD calibration (43.0 → 30.3 px, which is 29.5%; MAPE is 32.5%)
- **"Architecture-agnostic"**: MCT touches only calibration branch; pose branch is a black box that accepts an intrinsics embedding
- **Non-pinhole**: AnyCalib already supports fisheye/generic camera models via ray directions; extending MCT to variable distortion coefficients is natural future work
- **Longer sequences**: current max_ahead=3 (4 frames); longer windows would help trajectory but require managing flow composition error accumulation
- **Other model combinations**: e.g., RayDiffusion + MonST3R, or any depth/flow estimator that exposes a calibration interface
