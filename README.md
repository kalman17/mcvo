<div align="center">

# MCVO — Multi-frame, Camera-only Visual Odometry

**Camera pose from images alone, trained self-supervised on raw video — no labels, no depth or flow network at test time.**

[Kalman Mahlich](https://github.com/kalman17) · TU Munich, Chair of Computer Vision (Prof. Cremers) · 2026 · weights on [Hugging Face](https://huggingface.co/thekman17/mcvo)

</div>

**MCVO** (*Multi-frame, Camera-only Visual Odometry*) is a transformer that reads a short window of video frames and predicts the relative camera pose between consecutive frames, plus a per-pixel uncertainty map. It sees **images only**. Training needs no ground truth: pretrained depth (UniDepth) and optical-flow (UniMatch) networks and a single-image calibrator (AnyCalib) act as *training-time teachers* through AnyCam's flow-reprojection loss, and are absent at inference. It grew out of a TU Munich master's thesis whose calibration model, MCT, is kept in this repository as the calibration branch (second half of this page).

---

## Where it sits: accuracy · cost · supervision, one table

Rows are metrics, columns are methods. The first six columns are **measured here** under one protocol: one process per model on the same NVIDIA A40, identical 4-frame windows (30 per dataset for cost, 16 per sequence for accuracy), CUDA-synchronised, warm-up excluded, end-to-end per call (images in → poses/intrinsics out, including each model's own preprocessing and, for AnyCam, its depth and flow networks). The last four columns are **as reported by their authors** on other hardware and protocols — listed so the picture is complete, and queued to be measured on this protocol. Raw: [`honest_benchmarks/latency_summary.json`](honest_benchmarks/latency_summary.json), `experiments/bench_latency.py`, `honest_benchmarks/{e3_final_square336,thesis_final_e4_square336,e0_*,kfix_*}`.

| | **MCVO (ours)** | π³ | VGGT-1B | Depth Anything 3 | AnyCam (CVPR'25) | MCT + AnyCam (thesis) | DPVO* | FVO / VoT* | Monodepth2-style pose net* | ORB-SLAM3* |
|---|---|---|---|---|---|---|---|---|---|---|
| Measured here | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | reported | reported | reported | reported |
| Labels for training | **none** | GT poses/depth | GT | GT | none | none | GT poses (TartanAir) | GT poses | none (photometric) | none (classical) |
| Inputs at test time | images | images | images | images | images (+ runs depth & flow nets) | images (+ depth & flow nets) | images + **intrinsics** | images | image pairs | images + **intrinsics** |
| Parameters | **154 M** (67 M trained) | 959 M | 1257 M | 1690 M | 115 M + teachers | 460 M (25 M trained) | small | ~500 M | ~15 M | — |
| Weights on disk | **0.57 GiB** | 3.57 GiB | 4.68 GiB | 6.30 GiB | 0.43 GiB | 1.71 GiB | — | — | ~0.06 GiB | — |
| Peak GPU memory, 4-frame window | **0.69 GiB** | 5.5 GiB | 7.0 GiB | 9.7 GiB | 3.7 GiB | 5.0 GiB | ~4.9 GB (3090, streaming) | not published | small | CPU |
| Latency, 4-frame window (A40) | **75 ms** | 171 ms | 203 ms | 600 ms | 413 ms | 820 ms | ~60 fps @512×384 (3090); 120 fps variant | "~2× DPVO", "10× 3D foundation models" (3090) | milliseconds / pair | real-time, CPU |
| Latency, 8-frame window | **132 ms** | 300 ms | 376 ms | 1180 ms | 877 ms | 1645 ms | — | — | — | — |
| Rotation error, median — Sintel / TUM / KITTI | 0.46° / 0.89° / 0.19° | 0.22° / 0.26° / 0.11° | 0.28° / 0.32° / 0.12° | 0.19° / 0.27° / 0.09° | 0.50° / 0.74° / 0.20° | 0.40° / 0.67° / 0.23° | — | — | — | — |
| Heading error, KITTI (zero-shot for ours) | 7.0° | 2.2° | 4.6° | 1.3° | 28.6° | 28.2° | — | — | — | — |
| Heading error — Sintel / TUM | 81° / 90° | 27° / 34° | 38° / 37° | 19° / 32° | 49° / 50° | 47° / 65° | — | — | — | — |
| Focal error — Sintel / TUM / KITTI | — (pose only) | 25.2 % / 7.6 % / 28.9 % | 34.0 % / 25.8 % / 37.1 % | 24.4 % / 4.6 % / 15.8 % | 70.3 % / 14.6 % / 66.9 % | 20.7 % / 12.9 % / 20.4 % | needs intrinsics | — | — | needs intrinsics |
| Trajectory (Sintel ATE, Sim3) | 0.18 | — | — | — | 0.10 | 0.18 | strong (BA inside) | strong | weak | strong |
| Source | this repo | [paper](https://arxiv.org/abs/2507.13347) | [paper](https://arxiv.org/abs/2503.11651) | [paper](https://arxiv.org/abs/2511.10647) | [paper](https://arxiv.org/abs/2503.23282) | this repo | [Teed 2023](https://proceedings.neurips.cc/paper_files/paper/2023/file/7ac484b0f1a1719ad5be9aa8c8455fbb-Paper-Conference.pdf) | [Yugay 2025](https://arxiv.org/abs/2510.03348) | [Godard 2019](https://arxiv.org/abs/1806.01260) | [Campos 2021](https://arxiv.org/abs/2007.11898) |

*\* as reported by the authors, not measured here; different GPUs, resolutions and protocols.*

How to read it: MCVO runs at 5–14× lower peak memory and 2–8× lower latency than the billion-parameter supervised models and 5–11× below the two self-supervised pipelines, with no labels at any stage — at roughly twice their rotation error, competitive heading on driving video, and near-chance heading on small-baseline indoor video (a limitation that did not respond to longer context, teacher distillation, an epipolar loss, or motion-rich extra data). It is not the cheapest VO in existence: patch-based SLAM systems with bundle adjustment (DPVO) and tiny photometric pose nets are cheaper per frame, and classical SLAM runs on a CPU — those need known intrinsics, ground-truth poses for training, or give up much accuracy. Among image-only, feed-forward multi-frame transformers that reach VGGT-class rotation accuracy, MCVO is the smallest, the cheapest measured, and the only one trained without labels. If intrinsics are needed, pair it with the calibration branch below (MCT: 161 ms / 1.4 GiB, specialist-level focal accuracy) — a calibration head inside MCVO is in progress.

## How MCVO works

<p align="center"><img src="assets/mcvo_pipeline.svg" alt="MCVO pipeline: frozen DINOv2 backbone, 10 temporal/spatial attention blocks with per-frame camera tokens, pose and uncertainty heads; training-only teachers (UniDepth, UniMatch, AnyCalib) feed a flow-reprojection loss" width="100%"></p>

- **Backbone:** frozen DINOv2-base (86 M) → patch tokens per frame.
- **Decoder:** 10 blocks, each = temporal attention (every patch position attends across the frames of the window) → spatial attention within each frame including a learned per-frame **camera token** → MLP. 67 M trained parameters.
- **Heads:** the camera tokens of adjacent frames are concatenated and regressed to a relative pose (quaternion + translation); patch tokens give a per-pixel uncertainty map that lets the loss discount moving objects and unreliable regions.
- **Loss (self-supervised):** unproject the teacher depth of frame *i*, move it by the predicted pose, re-project into frame *i+1*, compare with the teacher optical flow under a Laplacian likelihood weighted by the predicted uncertainty. Intrinsics for the unprojection come from the cached AnyCalib per-frame estimates. All three teachers are consulted only inside the loss.
- **Data:** ~80 k frames of unlabeled video (RealEstate10K, YouTube-VOS, EpicKitchens, WalkingTours) preprocessed once with the AnyCam pipeline; 8-frame clips; 6 epochs on one GPU.
- **Code:** [`mcvo/`](mcvo/) (`model.py`, `loss.py`, `train.py`), evaluated with [`experiments/honest_benchmark.py`](experiments/honest_benchmark.py) (`--models mcvo:<ckpt>`); cost benchmark [`experiments/bench_latency.py`](experiments/bench_latency.py).

```bash
# train
PYTHONPATH=. python mcvo/train.py --data_dir /path/to/preprocessed --save_dir runs/mcvo \
    --backbone facebook/dinov2-base --d_model 640 --depth 10 --heads 8 --max_ahead 7 \
    --batch_size 4 --lr 1.5e-4 --epochs 6
# evaluate on the honest window protocol
PYTHONPATH=. python experiments/honest_benchmark.py --run_name mcvo_eval \
    --datasets sintel,tumrgbd,kitti --models mcvo:runs/mcvo/checkpoints/epoch_0006.pt
```

**Known limitations, stated plainly.** Translation direction on small-baseline indoor video is near chance; four interventions (longer context, teacher distillation, an epipolar/Sampson loss, motion-rich extra data) did not move it and it is treated as structural to flow-reprojection self-supervision when parallax is tiny. Trajectory-level accuracy trails AnyCam's long-context inference (0.18 vs 0.10 Sintel ATE) because short-window errors accumulate. No intrinsics head yet. History of every number on this page, including corrections: [CHANGELOG.md](CHANGELOG.md).

---

## The calibration branch: MCT (from the master's thesis)

*Everything below this line documents the thesis pipeline (MCT + AnyCam), which is the calibration branch of this repository. It is kept in full for reproducibility; the model above is the current work.*

### TL;DR (thesis)

The thesis introduced the **Multi-Frame Calibration Transformer (MCT)** — a lightweight, architecture-agnostic module that fuses a pretrained single-image calibration network ([AnyCalib](https://arxiv.org/abs/2503.12701)) with a self-supervised pose estimator ([AnyCam, CVPR 2025](https://arxiv.org/abs/2503.23282)), enforcing multi-frame calibration consistency through cross-frame attention on **intermediate features** rather than on scalar outputs.

Trained fully self-supervised on ~82 k frames of in-the-wild video, the fused system delivers calibration on par with the single-image specialist inside one self-supervised pipeline:

|  | Result | vs. | Dataset |
|---|---|---|---|
| **Focal error, native wide frames** (each method with its own preprocessing) | 15.7 % | AnyCalib 14.2 % · VGGT 11.6 % · Pi3 20.4 % · DA3 38.0 % | KITTI (full frames) |
| **Focal error, driving windows** | 20.4 % | AnyCalib 18.4 % · AnyCam 66.9 % | KITTI |
| **Rotation error** (median) | **0.40° vs 0.50°** | AnyCam | MPI Sintel |

Only **~25 M of ~370 M parameters (7.5 %)** are trainable; all pretrained backbones (DINOv2 ViT-L/14, DINOv2 ViT-S/14, UniDepth, UniMatch) remain frozen.

> **Note (August 2026; tables updated 17 August 2026 — see [CHANGELOG.md](CHANGELOG.md)).** The results on this page supersede the numbers in the original thesis document. An audit of the evaluation pipeline uncovered three measurement bugs (a silently broken baseline among them); after fixing them and retraining the calibration module with corrected input handling, all benchmarks were rerun from scratch — including against VGGT, Pi3 and Depth Anything 3. The full chronology is in [CHANGELOG.md](CHANGELOG.md). Raw per-window results for every table below are in [`honest_benchmarks/`](honest_benchmarks/) and can be recomputed with `experiments/honest_report.py`.

---

### What the thesis part of this repository contains

This is a fork of the [AnyCam (CVPR 2025)](https://github.com/Brummi/anycam) codebase. The upstream AnyCam pipeline is preserved on the [`upstream-anycam`](../../tree/upstream-anycam) branch. Everything on `main` is the thesis contribution.

| Component | What it is | Where |
|---|---|---|
| **Multi-Frame Calibration Transformer (MCT)** | The core contribution — 25 M-param cross-frame attention module operating at the feature level | `experiments/models/` |
| **Calibration ↔ pose coupling** | Wrapper that injects MCT-aggregated focal length into AnyCam's pose head via an 8-dim harmonic focal embedding | `experiments/train_pose_head_anycalib*.py` |
| **Three-phase staged training** | Pose-head warm-start → MCT pre-training → joint self-supervised fine-tuning, unified entry point | `experiments/train_unified.py` |
| **Evaluation suite** | MPI Sintel · TUM-RGBD · KITTI · Objectron benchmarks | `experiments/benchmark_*.py` |
| **Final training artefacts** | Loss histories, training logs, benchmark outputs, figures from the thesis runs | `experiments/final_training_phases/` &nbsp;·&nbsp; `thesis_results/` |
| **Thesis source** | LaTeX project, figures, bibliography | `kalman-tum-thesis-latex-master/` |
| **Defense presentation** | Beamer slides (TUM theme) | `presentation/` |

> **Naming note.** The calibration head went through two working names during development ("DA3", then "FAT"); code and docs now consistently use the thesis name **MCT (Multi-Frame Calibration Transformer)**. Old class names remain as aliases.

---

### The problem

Recovering camera intrinsics and pose from casual, uncalibrated video is a prerequisite for 3D reconstruction, novel-view synthesis, AR/VR, and autonomous navigation. Strong pretrained models exist for the individual sub-tasks — self-supervised pose, single-image calibration, monocular depth, optical flow — **but they operate in isolation, with no mechanism for joint reasoning or temporal consistency.** AnyCam in particular approximates focal length by selecting among 32 hard-coded candidates at inference, which is both computationally expensive and fundamentally limited in accuracy.

The question this thesis addresses:

> *Can we fuse multiple pretrained models into a unified system that jointly reasons about calibration and pose, while preserving fully self-supervised training?*

---

### How MCT works

The Multi-Frame Calibration Transformer sits **between** AnyCalib's frozen DINOv2 ViT-L/14 backbone and its frozen Light-DPT decoder — a slot that did not exist in either pretrained pipeline before. Pictorially:

```text
   ┌─────────────────────  CALIBRATION  BRANCH  (Ours: MCT inserted)  ─────────────────────┐
                                                                                            
   N frames  ──►  DINOv2 ViT-L/14  ──► [F₁,…,F_N]  ──►   ╔═══════════╗ ──►  Light-DPT  ──►  K
   (h × w × 3)        (frozen)         multi-scale       ║   MCT     ║       (frozen)
                                       per-frame         ║ (~25M, ✱) ║
                                       features          ╚═══════════╝
                                                              ▲
                                                              │
                              cross-frame self-attention + mean-pool over N


   
   ┌──────────────────────  POSE  BRANCH  (Ours: focal embedding added)  ──────────────────┐
                                                                                            
   N frames + depth(UniDepth) + flow(UniMatch) ──► DINOv2 ViT-S/14 (frozen) ──► Pose Neck
                                                                                  (✱)
                                                                                   │
                                                  focal f from K ──► harmonic ────►│
                                                                     embedding     ▼
                                                                                Pose Head ──► (R, t)
                                                                                  (✱)
```

For each spatial position, MCT treats the N frames as a sequence of tokens and applies two layers of multi-head self-attention **across the frame dimension only** (no spatial attention — positions are independent batch items), then mean-pools. The result has the same shape as a single-frame feature map, so the frozen decoder consumes it without modification.

**The key design choice: aggregate at the feature level, not the output level.** Per-frame calibration networks produce noisy intrinsics that *should* be averaged across a video — intrinsics are physically constant within a sequence. But naive scalar averaging of final predictions discards the rich spatial structure in the backbone's intermediate representations. Feature-level aggregation preserves geometric information that scalar averaging would throw away. The numbers bear this out:

> On TUM-RGBD, feature-level aggregation (MCT) cuts calibration MAPE from **11.7 % → 7.9 %** vs. per-frame averaging of the *same* AnyCalib backbone — a 32.5 % relative reduction from a one-line architectural change.

The MCT's output is then fed back into AnyCam's pose head via a learned **focal embedding** (8-dim harmonic encoding), replacing AnyCam's expensive 32-candidate focal-length selection system. The two branches are trained jointly: improved calibration directly benefits pose, and the pose-side flow reprojection loss propagates gradient signal back into the MCT.

---

### How MCT training works

Training is split into **three phases** (thesis §5.3.4). Each phase isolates a subset of trainable components to prevent degenerate solutions during joint optimisation.

| Phase | What trains | Loss | Why |
|---|---|---|---|
| **A** | Pose head only | Flow reprojection + pose consistency | Warm-start the pose head against static (averaged) AnyCalib calibration. Validates that AnyCalib can replace AnyCam's 32-candidate system. |
| **B** | MCT only | Ray reprojection vs. AnyCalib pseudo-GT | Bring MCT's output up to at least per-frame averaging quality before exposing it to joint training. Prevents collapse. |
| **C** | MCT + pose neck + pose head | Flow reprojection + pose consistency + tiny calibration anchor (λ<sub>calib</sub> = 10⁻⁴) | Joint fine-tuning. The small anchor lets MCT improve calibration via the pose signal without drifting from the reasonable AnyCalib prior. |

**Training data:** RealEstate10K · YouTube VOS · WalkingTours · OpenDV (~82 k frames, ~77 k 4-frame training windows).

**Multi-frame consistency at O(N), not O(N²).** Naively, training on N frames requires O(N²) pairwise flow computations. The thesis composes long-range flows from consecutive flows via bilinear warping:

$$\mathbf{w}_{1 \to 3}(u,v) = \mathbf{w}_{2 \to 3}\big((u,v) + \mathbf{w}_{1 \to 2}(u,v)\big)$$

This yields 5 training pairs per 4-frame window (3 consecutive + 2 composed) from O(N) flow computations. Composed pairs are down-weighted (λ<sub>comp</sub> = 0.1) to account for compounding interpolation error. The ablation in Appendix A confirms `max_ahead = 3` is optimal — shorter windows underutilise multi-frame information; longer windows compound flow errors.

---

### Thesis results

**Protocol.** Fixed public test splits (Sintel `particlesfm`, 14 sequences · TUM-RGBD `monst3r` dynamic, 8 sequences · KITTI odometry 00–10), 16 evenly-spaced 4-frame windows per sequence, identical inputs for every method, no filtering, failures logged rather than skipped. The checkpoint was selected by validation loss before any test data was touched. As a sanity anchor, this harness reproduces the published AnyCam Sintel trajectory numbers to the third decimal. Raw rows: [`honest_benchmarks/`](honest_benchmarks/).

#### Calibration

Focal-length error (median absolute percentage error) against ground-truth intrinsics:

| Dataset | AnyCam (32-cand.) | AnyCalib (per-frame avg) | **Ours (MCT)** |
|---|---:|---:|---:|
| **KITTI** | 66.9 % | **18.4 %** | 20.4 % |
| **Sintel** | 70.3 % | 20.1 % | 20.7 % (parity) |
| **TUM-RGBD** | 14.6 % (mean 65.6 %) | **11.2 %** | 12.9 % |

**Native-resolution, single forward pass** — full uncropped frames, 16 four-frame windows per sequence, every method with its own official preprocessing:

| Input | **Ours (MCT)** | VGGT-1B | DA3-Giant | AnyCalib | Pi3 |
|---|---:|---:|---:|---:|---:|
| KITTI wide frames (AR 3.3:1) | 15.7 % | **11.6 %** | 38.0 % | 14.2 % | 20.4 % |
| Sintel frames | 20.1 % | 19.9 % | 21.5 % | 20.3 % | **14.1 %** |

<sub>VGGT and Pi3 handle wide frames well when given the full frame; DA3's fixed-size preprocessing and the periphery-discarding resize of the single-image specialists (which MCT inherits from AnyCalib) cost accuracy at wide aspect ratios. An earlier version of this table reported 3.99 % for MCT from whole-sequence 8-frame multi-crop inference (`experiments/adaptive_multicrop_calib.py`); that number is reproducible with that script but is not comparable to the per-window protocol used for every method here.</sub>

**Multi-frame aggregation** (the core thesis claim): calibration accuracy vs number of frames aggregated across a sequence, against per-frame averaging of the same backbone —

| KITTI, frames aggregated | 1 | 2 | 4 | 8 | 16 |
|---|---:|---:|---:|---:|---:|
| Per-frame averaging | 16.6 % | 18.8 % | 19.2 % | 19.8 % | 17.1 % |
| MCT | 18.6 % | 21.0 % | 21.4 % | 21.9 % | 19.7 % |

(For this checkpoint, aggregating features across frames does not beat averaging per-frame estimates on KITTI, Sintel or TUM-RGBD; the earlier reported gain was an evaluation artefact, see CHANGELOG. Both result sets are in `honest_benchmarks/`.)

#### Pose estimation

| Dataset | Method | Rotation (°) ↓ mean / median | Translation direction (°) ↓ mean / median |
|---|---|---:|---:|
| **Sintel** | AnyCam | 0.98 / 0.50 | 63.5 / 49.9 |
| | **Ours (MCT)** | 0.98 / **0.40** | 62.3 / **47.3** |
| **TUM-RGBD** | AnyCam | 1.40 / 0.74 | **59.2 / 51.1** |
| | **Ours (MCT)** | **1.31 / 0.67** | 69.7 / 65.0 |
| **KITTI** | AnyCam | 0.41 / 0.20 | 63.0 / 28.6 |
| | **Ours (MCT)** | 0.41 / 0.23 | 61.9 / 28.2 |

Honest summary: rotation improves consistently (median −20 % on Sintel, −9 % on TUM), translation direction matches AnyCam on KITTI but is *worse* than AnyCam on TUM-RGBD. On full trajectories AnyCam's long-context inference remains ahead (Sintel ATE 0.100 vs 0.176 for chained 4-frame windows); large supervised models (Depth Anything 3 in particular) lead absolute pose accuracy on all datasets. What this system offers is specialist-level calibration and pose inside one self-supervised pipeline, with no labels at any stage.

### Quick start (thesis pipeline)

#### Environment

```bash
conda create -n anycam python=3.11 -y && conda activate anycam
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu124
conda install -c nvidia cuda-toolkit -y
pip install -r requirements.txt
```

#### Inference (upstream AnyCam baseline, for comparison)

```bash
./download_checkpoints.sh anycam_seq8

python anycam/scripts/anycam_demo.py \
    ++input_path=/path/to/video.mp4 \
    ++model_path=pretrained_models/anycam_seq8 \
    ++visualize=true
```

#### Training the MCT pipeline (Phases A → B → C)

The unified entry point is `experiments/train_unified.py`. Each phase loads the previous phase's checkpoint and trains the components specified in the table above.

```bash
# Phase A — pose-head warm-start (against static AnyCalib calibration)
python experiments/train_unified.py --phase A \
    --data_dir /path/to/training_data \
    --num_epochs 50 --batch_size 4 --learning_rate 1e-4 \
    --save_dir experiments/final_training_phases/phase_a

# Phase B1 — MCT pre-training against AnyCalib pseudo ground truth
python experiments/train_unified.py --phase B1 \
    --data_dir /path/to/training_data \
    --phase_a_checkpoint experiments/final_training_phases/phase_a/checkpoints/final.pt \
    --num_epochs 50 --batch_size 4 --learning_rate 5e-5 \
    --save_dir experiments/final_training_phases/phase_b1

# Phase C — joint self-supervised fine-tuning (multi-frame, max_ahead=3)
python experiments/train_unified.py --phase C \
    --data_dir /path/to/training_data \
    --phase_b1_checkpoint experiments/final_training_phases/phase_b1/checkpoints/final.pt \
    --num_epochs 50 --batch_size 2 --learning_rate 1e-5 \
    --max_ahead 3 --lambda_calib 1e-4 --lambda_comp 0.1 \
    --save_dir experiments/final_training_phases/phase_c
```

#### Benchmarking

```bash
# Pose vs. AnyCam on Sintel / TUM-RGBD / KITTI
python experiments/benchmark_against_anycam.py \
    --da3_stage3_model experiments/final_training_phases/phase_c/checkpoints/final.pt \
    --dataset sintel --num_samples 200 \
    --save_dir experiments/final_training_phases/benchmark_results

# Sweep Phase C checkpoints to pick the best epoch
python experiments/benchmark_phase_c_checkpoints.py \
    --checkpoint_dir experiments/final_training_phases/phase_c/checkpoints \
    --output_dir   experiments/final_training_phases/phase_c/benchmark_results
```

---

## Repository layout

```
mcvo/                    # (repository, formerly anycam-extension)
├── anycam/                          # Upstream AnyCam (CVPR 2025) — unchanged
├── anycalib/                        # AnyCalib submodule
├── unimatch/                        # UniMatch optical-flow fork
├── minipytorch3d/                   # Minimal PyTorch3D variant
├── experiments/                     # ★ Thesis contribution
│   ├── models/                      #   MCT architecture
│   ├── train_unified.py             #   Unified Phase A / B / C entry point
│   ├── train_pose_head_*.py         #   Pose-head experiments + AnyCalib wrapper
│   ├── benchmark_*.py               #   Sintel / TUM-RGBD / KITTI / Objectron
│   └── final_training_phases/      #   Checkpoints, loss histories, benchmark outputs
├── thesis_results/                  # Final reported figures + benchmark tables
├── presentation/                    # Defense slides + figures
├── diagrams/                        # Architecture diagrams (TikZ + PDF)
└── kalman-tum-thesis-latex-master/  # Thesis LaTeX source
```

---

### Reproducing the corrected thesis benchmarks

With the evaluation datasets prepared under `data/eval/` (Sintel `training/`, the eight
TUM-RGBD freiburg3 dynamic sequences, KITTI odometry 00–10) and a trained checkpoint:

```bash
# window-level pose + calibration vs AnyCam and AnyCalib (all tables above)
python experiments/honest_benchmark.py --run_name repro_square336 \
    --datasets sintel,tumrgbd,kitti --models ours,anycam,anycalib \
    --windows_per_seq 16 --ours_ckpt /path/to/mct_checkpoint.pt
python experiments/honest_report.py honest_benchmarks/repro_square336

# calibration vs number of aggregated frames
python experiments/sequence_calib_benchmark.py --run_name repro_seqcalib \
    --datasets sintel,tumrgbd,kitti --n_frames 1,2,4,8,16 \
    --ours_ckpt /path/to/mct_checkpoint.pt

# native-resolution calibration (wide KITTI frames + Sintel)
python experiments/adaptive_multicrop_calib.py \
    --ckpt /path/to/mct_checkpoint.pt --out honest_benchmarks/repro_native.json
```

To retrain the calibration module with corrected input handling (what produced the
numbers above — warm-started from a phase-C checkpoint, checkpoint then picked by
validation loss):

```bash
python experiments/train_unified.py --phase B1 --input_normalization \
    --data_dir /path/to/preprocessed --save_dir out/b1_normfix \
    --phase_b1_checkpoint /path/to/phase_c_checkpoint.pt \
    --num_epochs 8 --batch_size 8 --learning_rate 5e-5
python experiments/merge_finetuned_fat.py out/b1_normfix/checkpoints/<val_best>.pt \
    out/mct_final.pt /path/to/phase_c_checkpoint.pt
```

**Checkpoint:** the corrected final checkpoint is on Hugging Face —
[`thekman17/anycam-mct`](https://huggingface.co/thekman17/anycam-mct) (`mct_final.pt`;
load via `create_inference_model(..., input_normalization=True)` +
`load_phase_c_checkpoint`, see `experiments/benchmark_phase_c_checkpoints.py`).

---

## Citing this work

For MCVO (the model above) there is no paper yet; cite the repository:

```bibtex
@misc{mahlich2026mcvo,
  title  = {MCVO: Multi-frame, Camera-only Visual Odometry, self-supervised from raw video},
  author = {Mahlich, Kalman},
  year   = {2026},
  howpublished = {\url{https://github.com/kalman17/mcvo}},
}
```

For the calibration branch (MCT), the thesis:

```bibtex
@mastersthesis{mahlich2026learning,
  title  = {Learning Camera Geometry from Unlabeled Real-World Dynamic Video},
  author = {Mahlich, Kalman Eddi},
  school = {Technical University of Munich},
  year   = {2026},
  type   = {Master's Thesis},
}
```

If you use this code, please also cite the upstream **AnyCam** paper:

```bibtex
@inproceedings{wimbauer2025anycam,
  title     = {AnyCam: Learning to Recover Camera Poses and Intrinsics from Casual Videos},
  author    = {Wimbauer, Felix and Chen, Weirong and Muhle, Dominik and Rupprecht, Christian and Cremers, Daniel},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year      = {2025}
}
```

---

## Acknowledgements

This work builds directly on **[AnyCam](https://github.com/Brummi/anycam)** (Wimbauer et al., CVPR 2025) and **[AnyCalib](https://arxiv.org/abs/2503.12701)** (Tirado-Garín et al., 2025), and relies on **UniDepth** (Piccinelli et al.) and **UniMatch** (Xu et al.) as frozen helper networks. Supervision by **Daniil Sinitsyn**; thesis examined by **Prof. Dr. Daniel Cremers** at the Technical University of Munich, Chair of Computer Vision & Artificial Intelligence.
