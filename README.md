<div align="center">

# Learning Camera Geometry from Unlabeled Real-World Dynamic Video

**Master's Thesis · Technical University of Munich · 2026**

[Kalman Eddi Mahlich](https://github.com/kalman17) &nbsp;·&nbsp; Supervised by Daniil Sinitsyn &nbsp;·&nbsp; Examined by Prof. Dr. Daniel Cremers
*School of Computation, Information and Technology — Informatics*

</div>

<p align="center">
  <img src="https://github.com/Brummi/anycam/raw/main/assets/teaser_v2.gif" alt="Recovering camera geometry from casual video" width="90%">
  <br>
  <em>The problem: recover camera intrinsics <strong>K</strong> and per-frame poses <strong>(R, t)</strong> from a single casual, uncalibrated video — no calibration target, no ground truth, no offline bundle adjustment. Teaser by <a href="https://github.com/Brummi/anycam">AnyCam (CVPR 2025)</a>, the upstream model this thesis extends.</em>
</p>

---

## TL;DR

This thesis introduces the **Multi-Frame Calibration Transformer (MCT)** — a lightweight, architecture-agnostic module that fuses a pretrained single-image calibration network ([AnyCalib](https://arxiv.org/abs/2503.12701)) with a self-supervised pose estimator ([AnyCam, CVPR 2025](https://arxiv.org/abs/2503.23282)), enforcing multi-frame calibration consistency through cross-frame attention on **intermediate features** rather than on scalar outputs.

Trained fully self-supervised on ~82 k frames of in-the-wild video, the fused system delivers calibration at (and beyond) specialist level inside a single pipeline:

|  | Result | vs. | Dataset |
|---|---|---|---|
| **Focal error, native wide frames** | **3.99 %** (single forward pass) | VGGT 15.6 % · DA3 40.1 % · AnyCalib 73.6 % · Pi3 98.6 % | KITTI (full frames) |
| **Focal error, driving windows** | **7.2 %**, better on 95 % of windows | AnyCalib 10.4 % · AnyCam 94.8 % | KITTI |
| **Multi-frame gain** (8 frames vs per-frame averaging) | **6.0 % vs 9.1 %**, 100 % win rate | AnyCalib per-frame avg | KITTI |
| **Rotation error** (median) | **0.40° vs 0.50°** | AnyCam | MPI Sintel |

Only **~25 M of ~370 M parameters (7.5 %)** are trainable; all pretrained backbones (DINOv2 ViT-L/14, DINOv2 ViT-S/14, UniDepth, UniMatch) remain frozen.

> **Note (August 2026).** The results on this page supersede the numbers in the original thesis document. An audit of the evaluation pipeline uncovered three measurement bugs (a silently broken baseline among them); after fixing them and retraining the calibration module with corrected input handling, all benchmarks were rerun from scratch — including against VGGT, Pi3 and Depth Anything 3. The full chronology is in [CHANGELOG.md](CHANGELOG.md). Raw per-window results for every table below are in [`honest_benchmarks/`](honest_benchmarks/) and can be recomputed with `experiments/honest_report.py`.

---

## What this repository contains

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

## The problem

Recovering camera intrinsics and pose from casual, uncalibrated video is a prerequisite for 3D reconstruction, novel-view synthesis, AR/VR, and autonomous navigation. Strong pretrained models exist for the individual sub-tasks — self-supervised pose, single-image calibration, monocular depth, optical flow — **but they operate in isolation, with no mechanism for joint reasoning or temporal consistency.** AnyCam in particular approximates focal length by selecting among 32 hard-coded candidates at inference, which is both computationally expensive and fundamentally limited in accuracy.

The question this thesis addresses:

> *Can we fuse multiple pretrained models into a unified system that jointly reasons about calibration and pose, while preserving fully self-supervised training?*

---

## How MCT works

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

## How training works

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

## Results

**Protocol.** Fixed public test splits (Sintel `particlesfm`, 14 sequences · TUM-RGBD `monst3r` dynamic, 8 sequences · KITTI odometry 00–10), 16 evenly-spaced 4-frame windows per sequence, identical inputs for every method, no filtering, failures logged rather than skipped. The checkpoint was selected by validation loss before any test data was touched. As a sanity anchor, this harness reproduces the published AnyCam Sintel trajectory numbers to the third decimal. Raw rows: [`honest_benchmarks/`](honest_benchmarks/).

### Calibration

Focal-length error (median absolute percentage error) against ground-truth intrinsics:

| Dataset | AnyCam (32-cand.) | AnyCalib (per-frame avg) | **Ours (MCT)** |
|---|---:|---:|---:|
| **KITTI** | 94.8 % | 10.4 % | **7.2 %** (better on 95 % of windows) |
| **Sintel** | 70.3 % | 20.1 % | 20.7 % (parity) |
| **TUM-RGBD** | 14.6 % (mean 65.6 %) | **11.2 %** | 12.9 % |

**Native-resolution, single forward pass** — full uncropped frames, each competitor with its own preferred preprocessing where applicable:

| Input | **Ours (MCT)** | VGGT-1B | DA3-Giant | AnyCalib | Pi3 |
|---|---:|---:|---:|---:|---:|
| KITTI wide frames (AR 3.3:1) | **3.99 %** | 15.6 % | 40.1 % | 73.6 % | 98.6 % |
| Sintel frames | **17.3 %** | 28.9 % | 24.3 %* | 21.7 % | 13.4 % |

<sub>*DA3 measured on square windows; Pi3 leads Sintel. On the wide-format case every existing method degrades sharply because standard preprocessing discards the frame periphery, where field-of-view evidence is strongest — MCT trained with corrected input handling does not.</sub>

**Multi-frame aggregation** (the core thesis claim): calibration accuracy vs number of frames aggregated across a sequence, against per-frame averaging of the same backbone —

| KITTI, frames aggregated | 1 | 2 | 4 | 8 | 16 |
|---|---:|---:|---:|---:|---:|
| Per-frame averaging | 12.4 % | 11.8 % | 12.2 % | 9.1 % | 10.8 % |
| **MCT** | **10.1 %** | **9.5 %** | **9.3 %** | **6.0 %** | **8.1 %** |

(100 % per-sequence win rate at N ≥ 2. On Sintel/TUM this retrained checkpoint ties per-frame averaging rather than beating it — the earlier checkpoint wins there at N ≥ 8; both result sets are in `honest_benchmarks/`.)

### Pose estimation

| Dataset | Method | Rotation (°) ↓ mean / median | Translation direction (°) ↓ mean / median |
|---|---|---:|---:|
| **Sintel** | AnyCam | 0.98 / 0.50 | 63.5 / 49.9 |
| | **Ours (MCT)** | 0.98 / **0.40** | 62.3 / **47.3** |
| **TUM-RGBD** | AnyCam | 1.40 / 0.74 | **59.2 / 51.1** |
| | **Ours (MCT)** | **1.31 / 0.67** | 69.7 / 65.0 |
| **KITTI** | AnyCam | 0.48 / 0.24 | 89.4 / 89.8 |
| | **Ours (MCT)** | 0.48 / 0.24 | **73.5 / 68.8** |

Honest summary: rotation improves consistently (median −20 % on Sintel, −9 % on TUM), translation direction improves markedly on KITTI (where the baseline is at chance level) but is *worse* than AnyCam on TUM-RGBD. On full trajectories AnyCam's long-context inference remains ahead (Sintel ATE 0.100 vs 0.176 for chained 4-frame windows); large supervised models (Depth Anything 3 in particular) lead absolute pose accuracy on all datasets. What this system uniquely offers is specialist-grade, multi-frame calibration inside one self-supervised pipeline.

## Quick start

### Environment

```bash
conda create -n anycam python=3.11 -y && conda activate anycam
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu124
conda install -c nvidia cuda-toolkit -y
pip install -r requirements.txt
```

### Inference (upstream AnyCam baseline, for comparison)

```bash
./download_checkpoints.sh anycam_seq8

python anycam/scripts/anycam_demo.py \
    ++input_path=/path/to/video.mp4 \
    ++model_path=pretrained_models/anycam_seq8 \
    ++visualize=true
```

### Training the MCT pipeline (Phases A → B → C)

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

### Benchmarking

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
anycam-extension/
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

## Reproducing the corrected benchmarks

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
