# Teleprompter — Defense Presentation

Read naturally, not word-for-word. Each slide ~1.5 min. Total ~15 min.

---

## Slide 1: Title

*[No speaking — wait for introduction by committee]*

---

## Slide 2: Outline

"I'll start with the motivation, then introduce the pretrained models we build on — AnyCam for pose and AnyCalib for calibration. Then I'll present our framework for fusing them together, the contributions, the training setup and losses, results, and conclude with future directions."

---

## Slide 3: Motivation

"Recovering camera poses and intrinsics from video is fundamental to 3D vision — autonomous driving, augmented reality, 3D reconstruction all need this.

But casual video makes this really hard. The scenes are dynamic — people walking, cars moving. The cameras are unknown — no calibration information. And there are no 3D labels at scale. Classical methods like COLMAP assume static scenes and fail. Supervised methods need expensive ground truth that simply doesn't exist for most video.

Now, we do have strong pretrained models for individual sub-tasks — depth estimation, optical flow, single-image calibration, pose prediction. But they each work in isolation. There's no system that combines them with joint reasoning. Let me show you two such models and why combining them is the opportunity."

---

## Slide 4: The Calibration Problem in AnyCam

"This is AnyCam — a self-supervised pose estimator published at CVPR 2025, currently state of the art. It predicts camera poses from unlabeled video using flow reprojection losses. The architecture uses multi-frame self-attention for pose reasoning, which is quite effective.

But look at how it handles calibration — the 32-candidate system on the right. It can only choose from 32 predefined focal lengths. The numbers show this is often very wrong: 45% error on Sintel, 64% on TUM-RGBD. And since calibration directly enters the projection equation, bad calibration means bad poses. This is the bottleneck we want to address."

---

## Slide 5: AnyCalib — A Dedicated Calibration Network

"Now here's a dedicated calibration network — AnyCalib. Completely different approach: a large DINOv2 backbone, a decoder that predicts per-pixel ray directions, and a closed-form fit to recover intrinsics. Much more accurate than 32 candidates.

But notice: it processes each frame independently. No temporal consistency at all. A single camera has fixed intrinsics throughout a video, yet the per-frame predictions are noisy and inconsistent.

So we have a pose model that's great at poses but bad at calibration, and a calibration model that's accurate but per-frame only. The natural question: what if we fuse them?"

---

## Slide 6: Our Framework — Fusing Calibration & Pose

"This is our framework. We fuse a self-supervised pose estimator with a dedicated calibration network.

The top row is the pose branch — in our case, AnyCam. The bottom row is the calibration branch — AnyCalib's backbone, then our MCT for multi-frame aggregation, then AnyCalib's decoder.

The key: the MCT is inserted between the calibration backbone and decoder. It aggregates features from N frames into one consistent representation. The resulting calibration is injected into the pose head via a focal embedding.

This framework is not specific to AnyCam or AnyCalib — any self-supervised pose estimator could use this calibration branch. We demonstrate it on these two because they're state of the art."

---

## Slide 7: Contributions

"Two contributions.

First, the Multi-Frame Calibration Transformer — the MCT. It's a lightweight module that aggregates calibration backbone features across N video frames via cross-frame attention. It's architecture-agnostic: you insert it between any feature extractor and decoder. Only 25M trainable parameters while everything else stays frozen.

Second, we demonstrate that this framework works in practice. Fusing AnyCam with AnyCalib plus the MCT improves both pose accuracy and calibration accuracy over the individual models. And the whole thing is self-supervised — no ground truth labels needed for training."

---

## Slide 8: Training Setup

"The training view. Blue blocks are frozen pretrained models — both DINOv2 backbones, the decoder, UniDepth, UniMatch. Orange blocks are what we train — the Pose Neck, Pose Head, and MCT.

Only 7.5% of parameters are trained. Everything pretrained stays frozen."

---

## Slide 9: Training Losses

"The main signal is flow reprojection: project a pixel to 3D, apply the predicted pose, reproject, and compare against observed optical flow. Fully self-supervised.

Now here's the key insight about calibration. For small camera displacements, the intrinsic matrix K cancels out of the reprojection equation — K-inverse times K is the identity. So the reprojection loss can be near zero regardless of what K is. Calibration has too much freedom.

That's why we need the calibration anchor. It keeps K near the pseudo ground truth from the calibration network, preventing drift.

This is especially important on real-world videos where dynamic objects create large flow even with small camera motion — you can't reliably filter out near-static frames, so the anchor is essential."

---

## Slide 10: Results — Pose Accuracy

"Rotation error on the left — both methods are already accurate here, under one degree. We see marginal improvements.

Translation on the right is the real story. This was the weakness of the baseline pose estimator. We reduce translation error by up to 39% on Sintel. Better calibration directly improves translation because the focal length determines projection scale."

---

## Slide 11: Results — Calibration Accuracy

"For calibration: the baseline pose estimator's 32-candidate system has 68% error. The standalone calibration network per-frame gets 7.6%. Our MCT brings it to 5.4%.

The key insight: feature-level aggregation in the MCT preserves geometric structure that scalar averaging would discard."

---

## Slide 12: Trajectory Visualization

"A qualitative example. Our trajectory in blue closely follows ground truth. The baseline in orange drifts. Direct consequence of improved calibration and translation accuracy."

---

## Slide 13: Conclusion & Future Work

"To summarise: we proposed a framework for fusing pretrained models for camera geometry. The core contribution is the MCT — a multi-frame calibration consistency module that aggregates features before decoding. We demonstrated it by fusing AnyCam with AnyCalib, achieving up to 39% better translation and 32% better calibration. The framework is self-supervised and architecture-agnostic.

Future work: longer sequences, non-pinhole cameras, and applying the framework to other model combinations.

Thank you — happy to take questions."
