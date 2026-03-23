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

"Estimating camera motion and intrinsics from video is fundamental to 3D computer vision — autonomous driving, augmented reality, 3D reconstruction all depend on knowing the camera geometry.

There are billions of casual videos online — YouTube, dashcams, phone footage — all unlabeled. Self-supervised learning can potentially unlock this data without ground truth labels.

Today we have strong pretrained models for individual sub-tasks: depth estimation, optical flow, single-image calibration, pose prediction. But they operate in isolation. The question driving this thesis is: can we fuse multiple pretrained models into a unified system that jointly reasons about calibration and pose, with multi-frame consistency, while keeping training self-supervised?"

---

## Slide 4: The Calibration Problem in AnyCam

"Here's AnyCam — a self-supervised pose estimator from CVPR 2025. It's the state of the art for this task. Frames go through a DINOv2-small backbone, then a Pose Neck with 8 layers of self-attention across frames.

But look at how it handles calibration: the 32-candidate system. It tries 32 predefined focal length guesses and picks the best one. This is inherently limited — 32 discrete values, expensive to evaluate, and the numbers show it's often quite wrong: 45% error on Sintel, 64% on TUM-RGBD. Since calibration feeds directly into the projection equation, bad focal length means bad poses."

---

## Slide 5: AnyCalib — A Dedicated Calibration Network

"AnyCalib is a dedicated calibration network. It uses a large DINOv2 ViT-L backbone, a Light-DPT decoder, and predicts per-pixel ray directions. From those rays, intrinsics are recovered in closed form. Much more accurate than AnyCam's 32-candidate system.

But it processes each frame independently — no temporal consistency. A single camera has one fixed focal length, yet per-frame predictions vary. What if we could aggregate these predictions while preserving the spatial structure in the features?"

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
