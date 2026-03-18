# Teleprompter — Defense Presentation

Read naturally, not word-for-word. Each slide ~1.5 min. Total ~15 min.

---

## Slide 1: Title

*[No speaking — wait for introduction by committee]*

---

## Slide 2: Outline

"I'll start with the motivation behind this work, then introduce the AnyCam baseline and its limitations. I'll show how we address those limitations by integrating AnyCalib, walk through our method and contributions, explain the training setup, present results, and conclude with future directions."

---

## Slide 3: Motivation

"Estimating camera motion and intrinsics from video is fundamental to 3D computer vision. Applications like autonomous driving, augmented reality, and 3D reconstruction all depend on knowing where the camera was and what lens it used.

The exciting opportunity is that there are billions of casual videos online — YouTube, dashcams, phone footage — all unlabeled, all potentially useful for training. But unlocking this data requires methods that work without calibration information or 3D ground truth.

Classical approaches like Structure-from-Motion and SLAM struggle with dynamic scenes — moving people, cars, changing lighting. Supervised methods need expensive 3D labels that don't exist at scale.

AnyCam, published at CVPR 2025, was the first self-supervised method to directly predict both camera poses and intrinsics from casual video, trained entirely on unlabeled YouTube data. It's the current state of the art — but it has a calibration bottleneck that we set out to fix."

---

## Slide 4: The Calibration Problem in AnyCam

"Here's the AnyCam pipeline. Frames go through a DINOv2-small backbone, then a Pose Neck with 8 layers of self-attention across frames — this is where inter-frame reasoning happens.

The problem is on the right: the 32-candidate system. AnyCam doesn't predict the focal length directly — instead it tries 32 predefined guesses, runs a pose head for each one, and uses a sequence head to pick the best.

This is expensive — 32 forward passes per prediction — and coarse, since the true focal length may fall between candidates. And the numbers show it's often wrong: 45% mean error on Sintel, 64% on TUM-RGBD. That's massive. And since calibration feeds directly into the projection equation, bad focal length means bad poses."

---

## Slide 5: AnyCalib — A Path to Better Calibration

"AnyCalib takes a completely different approach. It uses a large DINOv2 ViT-L backbone, a Light-DPT decoder, and predicts per-pixel ray directions — a Field-of-View field. From those rays, camera intrinsics are recovered in closed form.

It's much more accurate than AnyCam's 32-candidate system. Even naively averaging AnyCalib's per-frame predictions already helps.

But AnyCalib processes each frame independently — there's no cross-frame communication. So you get noisy, inconsistent calibration estimates from frame to frame. A single camera should have one fixed focal length. What if we could enforce that consistency?"

---

## Slide 6: Our Method — AnyCalib × AnyCam

"This is our complete pipeline — it's essentially AnyCam and AnyCalib merged.

The top row is AnyCam's pose branch: frames plus depth and flow go through DINOv2-small, through the Pose Neck, into the Pose Head, which now outputs the final pose.

The bottom row is AnyCalib's calibration branch: the same frames go through DINOv2 ViT-L — one per frame — then our new Multi-Frame Calibration Transformer aggregates them, and the rest of AnyCalib's decoder produces the intrinsics K.

The key connection: K from the calibration branch is converted to a focal embedding and injected into the Pose Head. This replaces the entire 32-candidate system with a single accurate calibration."

---

## Slide 7: Contributions

"Two main contributions.

First, the integration itself — AnyCalib times AnyCam. We inject AnyCalib's calibration directly into AnyCam's pose head, replacing the 32-candidate system entirely. This alone improves both pose and calibration.

Second, the Multi-Frame Calibration Transformer — the MCT. It operates on AnyCalib's backbone features at 4 scales, applying cross-frame attention so that frames share calibration information. The result is a single consistent calibration instead of N noisy per-frame estimates.

The plot on the right shows this clearly: AnyCam's 32-candidate system in orange has large outliers. AnyCalib per-frame in green is better but noisy. Our MCT in blue is smooth and accurate."

---

## Slide 8: Training Setup

"This diagram shows the training view. Blue blocks are frozen — both DINOv2 backbones, UniDepth, UniMatch, the Light-DPT decoder. Orange blocks are trainable — the Pose Neck, Pose Head, and MCT.

Only 27.5 million parameters out of 370 million total are trained — that's 7.5%. The training is fully self-supervised using flow reprojection loss, pose consistency, composed flow for multi-frame consistency, and a calibration anchor that keeps the MCT near AnyCalib's predictions."

---

## Slide 9: Training Losses

"Let me walk through the loss functions that drive our training.

The main signal is flow reprojection: we take a pixel, unproject it to 3D using predicted depth and calibration K, apply the predicted pose rotation and translation, reproject back to the image, and compare against the observed optical flow from UniMatch. This is fully self-supervised — no ground truth labels.

Now, there's a subtle but important problem with calibration. Look at this equation: when the camera barely moves — small rotation, small translation — the reprojection simplifies to K-inverse times K times x, which is just x. The K cancels out. This means for small displacements, the reprojection loss is near zero regardless of what K is. Calibration has too much freedom — any K would work.

That's why we need the calibration anchor loss. It keeps our predicted K near AnyCalib's pseudo ground truth, preventing the calibration from drifting to arbitrary values during training. It's a regulariser that provides gradient signal even when the reprojection loss is uninformative about K.

The remaining two losses are the composed flow loss for multi-frame consistency, and a forward-backward pose consistency term.

And this calibration anchor is especially important because we're training on real-world videos. With casual footage, some frames will have very small camera displacement — the camera barely moved between frames. Simple optical flow-based frame sampling can't reliably filter these out, because dynamic objects in the scene create large flow even when the camera is nearly static. So the anchor is an essential part of making this work on real data — without it, those near-static frames would let calibration drift unchecked."

---

## Slide 10: Results — Pose Accuracy

"On the left, rotation error histograms — both methods are already very accurate here, AnyCam's rotation error is already under one degree. We see marginal improvements.

But the real story is translation, on the right. This was AnyCam's weakness. We reduce translation error by up to 39% on Sintel and TUM-RGBD. The table shows the numbers — consistent improvements across all three evaluation datasets. Better calibration directly translates to better translation estimation, because the focal length determines the scale of the translation."

---

## Slide 11: Results — Calibration Accuracy

"For calibration, the improvement is clear. AnyCam's 32-candidate system has over 54% mean error. AnyCalib per-frame brings that down to 21%. Our MCT brings it further down to under 18%.

The histogram on the right shows this visually — our errors are concentrated near zero while AnyCam's are spread across the range.

The key insight is that feature-level aggregation in the MCT preserves geometric structure that's lost when you simply average scalar predictions."

---

## Slide 12: Trajectory Visualization

"This is a qualitative example on a Sintel sequence. Our trajectory in blue closely follows the ground truth in black. AnyCam in orange accumulates drift over the sequence. This is a direct consequence of the improved calibration and translation accuracy."

---

## Slide 13: Conclusion & Future Work

"To summarize: we integrated AnyCalib's calibration pipeline into AnyCam, replacing the expensive 32-candidate system. We introduced the Multi-Frame Calibration Transformer for calibration consistency. The whole system is self-supervised, training only 7.5% of parameters.

The results show up to 39% better translation accuracy and 32% better calibration compared to the baselines.

For future work, three directions: supporting longer sequences through hierarchical composition, extending to non-pinhole camera models like fisheye, and jointly refining depth alongside calibration and pose.

Thank you — I'm happy to take questions."
