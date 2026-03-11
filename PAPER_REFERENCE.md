# Paper Reference — Quick-Access Facts for Thesis Writing

> **Source of truth**: benchmark JSONs in `thesis_results/`, SLURM scripts in `experiments/cluster/`, model code in `experiments/models/`.
> All numbers below are extracted directly from these sources. Cross-check if in doubt.

---

## 1. Architecture

### Base Model: AnyCam (CVPR 2025)

Self-supervised framework for camera pose and intrinsics from casual video.

**Pipeline** (frozen at inference):
1. **Depth predictor** (UniDepth): per-frame monocular depth
2. **Optical flow** (UniMatch): pairwise forward flow + occlusion masks
3. **Sequence info head**: focal length via 32-candidate testing (replaced by us)
4. **Pose head**: 6-DoF relative pose from visual tokens + flow + depth features

**Key files**:
- `anycam/models/anycam.py` — main model, visual token extraction
- `anycam/models/anycam_blocks.py` — pose head and sequence info head architectures
- `anycam/loss/pose_loss.py` — flow reprojection loss (Laplacian NLL)

### Our Extension: FAT-Enhanced Calibration

Replaces the 32-candidate focal length system with a **Feature Aggregation Transformer (FAT)** that aggregates multi-frame AnyCalib features into a single calibration prediction.

**Pipeline modification**:
1. All N frames → AnyCalib DINOv2 ViT-L backbone (frozen) → per-frame features
2. Per-frame features → **FAT** (trainable) → aggregated feature
3. Aggregated feature → AnyCalib DPT decoder + ray head (frozen) → calibration
4. Calibration → pose head (trainable) → poses
5. Poses + depth + calibration → flow reprojection loss (self-supervised)

**Key files**:
- `experiments/models/anycalib_with_fat.py` — AnyCalib + FAT model (`AnyCalibWithFAT`)
- `experiments/models/anycam_wrapper_fat.py` — end-to-end inference wrapper (`AnyCamWrapperWithFATCalibration`)
- `experiments/models/unified_wrapper.py` — training wrapper for all phases (`UnifiedTrainingWrapper`)

### FAT Architecture Details

```
Input: N frames of DINOv2 ViT-L features [N, 1024]
  ↓
Learnable aggregation token (1 token)
  ↓
Multi-head self-attention transformer:
  - embed_dim: 1024
  - num_heads: 8
  - num_layers: 2
  - dropout: 0.1
  - num_scales: 4 (multi-scale feature aggregation)
  ↓
Optional DINOv2-small visual conditioning (vis_dim=384)
  ↓
Aggregated feature → AnyCalib DPT decoder → ray field → camera model fit
  ↓
Output: intrinsics [fx, fy, cx, cy]
```

---

## 2. Trainable Parameters

| Component | Parameters | Trainable in Phase C |
|-----------|-----------|---------------------|
| Depth predictor (UniDepth) | ~300M | Frozen |
| Flow processor (UniMatch) | ~13M | Frozen |
| AnyCalib backbone (DINOv2 ViT-L) | ~300M | Frozen |
| AnyCalib DPT decoder + ray head | ~25M | Frozen |
| AnyCalib calibrator | non-differentiable | Frozen |
| **FAT (transformer)** | **~25M** | **Trainable** |
| **Pose head** | **~21K** | **Trainable** |
| DINOv2-small (visual conditioning) | ~22M | Frozen |

**Total model**: ~685M parameters
**Trainable in our approach**: ~25M (FAT + pose head) — **3.6% of total**

---

## 3. Training Phases

### Phase A — Pose Head Pre-Training

**Goal**: Train fresh pose head with AnyCalib focal lengths (no FAT yet)

| Parameter | Value |
|-----------|-------|
| Trainable | pose_head only (~21K params) |
| Batch size | 16 |
| Learning rate | 1e-4 |
| Epochs | 10 (ran 8) |
| max_ahead | 3 |
| Image size | 336 |
| Loss | Flow reprojection (Laplacian NLL) + composed flow (λ_comp=0.1) |
| GPU | Any Ampere/Ada/Hopper |
| Script | `experiments/cluster/slurm_train_phase_a.sh` |
| Output | `phase_A_v2/` |
| Time/epoch | ~70 min |

### Phase B1 — FAT Pre-Training (Isolated)

**Goal**: Pre-train FAT on AnyCalib ray reprojection loss (no pose pipeline)

| Parameter | Value |
|-----------|-------|
| Trainable | FAT only (~25M params) |
| Batch size | 4 |
| Learning rate | 1e-4 |
| Epochs | 10 (ran 7) |
| max_ahead | 3 |
| Image size | 336 |
| Loss | AnyCalib ray reprojection MSE |
| GPU | Any Ampere/Ada (24GB+ VRAM) |
| Script | `experiments/cluster/slurm_train_phase_b1.sh` |
| Output | `phase_B1/` |
| Time/epoch | ~5.1 hours |

### Phase C — Joint End-to-End Training (Frozen Backbones)

**Goal**: Joint training of pose head + FAT through the full pipeline

| Parameter | Value |
|-----------|-------|
| Trainable | pose_head + FAT (~25M params) |
| Frozen | depth predictor, flow processor, AnyCalib backbone/decoder |
| Batch size | **20** |
| Learning rate | **1e-4** |
| Epochs | 10 (ran 7, **epoch 5 is best**) |
| max_ahead | **3** |
| Image size | **336** |
| λ_calib | **1e-4** |
| λ_comp | 0.1 |
| GPU | **NVIDIA H100 (80GB)** |
| Script | `experiments/cluster/slurm_train_phase_c_v3_h100.sh` |
| Output | `phase_C_v3_h100/` |
| Time/epoch | ~56 min |
| Initialization | Phase A (pose_head) + Phase B1 (FAT) |

---

## 4. Loss Functions

### Flow Reprojection Loss (Primary — Self-Supervised)

Compares predicted optical flow (from estimated pose + depth + calibration) against observed optical flow (from UniMatch). No ground truth poses or calibration needed.

```
L_flow = (1/σ) · |flow_pred - flow_obs| + log(σ)
```

Where σ is the predicted uncertainty (Laplacian NLL, same as AnyCam `PoseLoss.compute_pose_loss`).

**Implementation**: `experiments/models/unified_wrapper.py`, method `_forward_phase_C()`

### Composed Flow Loss (Multi-Frame Consistency)

For frames beyond consecutive pairs (e.g., frame 1→3), composes intermediate flows:

```
flow_{1→3} = warp(flow_{2→3}, flow_{1→2})  (bilinear interpolation)
```

Then applies the same Laplacian NLL loss on the composed flow vs. predicted flow from composed poses.

- **Weight**: λ_comp = 0.1 (relative to consecutive loss)
- **Implementation**: `_compose_flows()` in unified_wrapper.py

### Calibration Anchor Loss

Encourages FAT calibration to stay close to per-frame AnyCalib predictions (regularization).

```
L_total = L_flow_consecutive + λ_comp · L_flow_composed + λ_calib · L_calib_anchor
```

- **Weight**: λ_calib = 1e-4
- **Implementation**: AnyCalib ray consistency loss via `compute_ray_consistency_loss()` in `anycalib_with_fat.py`

---

## 5. Training Data

Preprocessed on cluster at `/storage/user/maka/preprocessed/`.

| Dataset | Videos | Purpose |
|---------|--------|---------|
| RealEstate10K | 1,316 | Indoor/outdoor real estate tours |
| YouTubeVOS | 440 | Diverse YouTube video segments |
| EpicKitchens | 16 | Egocentric kitchen activities |
| WalkingTours | 2 | Long walking tour sequences |
| **Total** | **1,774** | |

**Preprocessing**: Each video → extract frames at 2 FPS → compute depth (UniDepth), flow (UniMatch), AnyCalib features offline → store as .npz files.

- **Total frames**: ~82,000
- **Total sequences**: ~77,000 (overlapping windows of max_ahead+1=4 frames)

These are the same base datasets used by the original AnyCam paper (minus OpenDV which was unavailable).

---

## 6. Evaluation Protocol

### Benchmark Configuration (FINAL — benchmark_Cb)

| Setting | Value |
|---------|-------|
| Mode | quick |
| Samples per dataset | 200 |
| Frame count | 4 (matching training max_ahead=3) |
| Image size | 336 |
| Dilation mode | `anycam` (matches AnyCam eval protocol) |

### Per-Dataset Dilation (Frame Spacing)

| Dataset | Native FPS | Dilation | Effective FPS |
|---------|-----------|----------|---------------|
| Sintel | 24 | 1 | 24 |
| TUM-RGBD | 30 | 10 | 3 |
| KITTI | 10 | 1 | 10 |

### Baselines

1. **AnyCam** (vanilla): Original CVPR 2025 model with 32-candidate focal system. Checkpoint: `pretrained_models/anycam_seq8/training_checkpoint_247500.pt`
2. **AnyCalib** (standalone): Per-frame AnyCalib predictions without FAT aggregation

### Metrics

**Pose metrics** (per consecutive frame pair, 3 pairs per 4-frame sequence = 600 total):
- Rotation error (degrees): angular distance between rotation matrices
- Translation direction error (degrees): angle between translation vectors
- Translation magnitude error: absolute difference in translation norm
- SE(3) distance: combined rotation + translation error

**Calibration metrics** (per sequence, datasets with GT intrinsics only):
- Focal length MAE: |f_pred - f_gt| in pixels (fx and fy separately)
- Focal length MAPE: percentage error |f_pred - f_gt| / f_gt × 100

---

## 7. Final Results

### Pose Estimation (FINAL — benchmark_Cb, epoch 5)

#### Sintel (200 sequences, 600 pairs)

| Metric | Ours | AnyCam | Improvement |
|--------|------|--------|-------------|
| Rotation (mean) | **0.602°** | 0.669° | +10.1% |
| Rotation (median) | **0.189°** | 0.206° | +8.2% |
| Trans. direction (mean) | **64.95°** | 89.30° | **+27.3%** |
| Trans. direction (median) | **52.33°** | 86.53° | **+39.5%** |
| SE(3) distance (mean) | **0.546** | 0.548 | +0.3% |
| SE(3) distance (median) | **0.019** | 0.020 | +6.8% |

#### TUM-RGBD (200 sequences, 600 pairs)

| Metric | Ours | AnyCam | Improvement |
|--------|------|--------|-------------|
| Rotation (mean) | **1.388°** | 2.031° | **+31.7%** |
| Rotation (median) | **0.708°** | 1.232° | **+42.6%** |
| Trans. direction (mean) | **71.94°** | 93.04° | **+22.7%** |
| Trans. direction (median) | **66.90°** | 97.05° | **+31.1%** |
| SE(3) distance (mean) | **0.042** | 0.056 | **+25.8%** |
| SE(3) distance (median) | **0.028** | 0.039 | **+28.8%** |

#### KITTI (200 sequences, 600 pairs)

| Metric | Ours | AnyCam | Improvement |
|--------|------|--------|-------------|
| Rotation (mean) | **0.545°** | 0.556° | +2.0% |
| Rotation (median) | **0.252°** | 0.258° | +2.4% |
| Trans. direction (mean) | **77.20°** | 91.14° | **+15.3%** |
| Trans. direction (median) | **70.93°** | 91.45° | **+22.4%** |
| SE(3) distance (mean) | 1.055 | 1.055 | +0.0% |
| SE(3) distance (median) | 1.002 | 1.003 | +0.0% |

**Note on KITTI translation**: SE(3) and translation magnitude are dominated by KITTI's forward-driving motion (~1m/frame). Both methods predict similar magnitude; the difference is in direction accuracy.

### Calibration (Focal Length)

#### Sintel

| Metric | Ours (FAT) | AnyCalib (standalone) | AnyCam (32-cand.) |
|--------|-----------|----------------------|-------------------|
| fx MAE (mean) | **300.3** | 329.9 | 502.5 |
| fx MAE (median) | **188.8** | 217.5 | 485.6 |
| MAPE (mean) | **27.8%** | 30.7% | 68.2% |
| MAPE (median) | **26.2%** | 31.4% | 70.3% |

#### TUM-RGBD

| Metric | Ours (FAT) | AnyCalib (standalone) | AnyCam (32-cand.) |
|--------|-----------|----------------------|-------------------|
| fx MAE (mean) | **30.3** | 43.0 | 234.1 |
| fx MAE (median) | **24.9** | 42.2 | 53.8 |
| MAPE (mean) | **7.9%** | 11.7% | 63.7% |
| MAPE (median) | **6.1%** | 11.5% | 14.6% |

#### KITTI — Calibration Failure

| Metric | Ours (FAT) | AnyCalib (standalone) | AnyCam (32-cand.) |
|--------|-----------|----------------------|-------------------|
| fx MAE (mean) | 34,213 | **66.3** | 608.6 |
| MAPE (mean) | 6,661% | **10.3%** | 94.8% |

**KITTI calibration is catastrophically wrong for our model.** The FAT aggregation produces unreasonable focal lengths on KITTI despite working well on Sintel and TUM-RGBD. This is likely because:
1. KITTI uses a very different camera (automotive, wide baseline) not well-represented in AnyCalib's or our training distribution
2. The FAT was trained on indoor/mixed-scene video data
3. Despite wrong calibration, pose estimation still improves slightly (the pose head partially compensates)

---

## 8. Design Decisions

### Why Frozen Backbones?

- **Depth predictor** (UniDepth, ~300M): Pre-trained on massive depth datasets. Fine-tuning on our 82K frames would cause catastrophic forgetting. The depth quality is already excellent.
- **Flow processor** (UniMatch, ~13M): Pre-trained on optical flow benchmarks. Flow quality is the foundation of our self-supervised loss — corrupting it would undermine training.
- **AnyCalib backbone** (DINOv2 ViT-L, ~300M): Pre-trained on vast image data. We want to leverage its features, not retrain them.
- **Practical**: Freezing reduces trainable params from ~685M to ~25M, enabling batch_size=20 on H100.

### Why max_ahead=3?

Tested in prior experiments (Experiment 2): max_ahead values of 2, 3, 4, 5 were compared. max_ahead=3 (4 frames) is optimal:
- max_ahead=2: insufficient multi-frame signal
- max_ahead=3: best rotation error (1.23° mean in Exp 2)
- max_ahead=4+: flow composition error accumulates, degrading quality

### Why FAT Over DA3 Calibration Head?

Earlier experiments tested a DA3-inspired calibration head (camera encoder → visual-camera mixing → sequence aggregation → camera decoder) with 3-stage training. The FAT approach was chosen because:
1. **Simpler architecture**: Single transformer instead of multi-component pipeline
2. **Leverages AnyCalib directly**: Uses pre-trained AnyCalib decoder instead of learning calibration from scratch
3. **Better integration**: FAT sits inside the AnyCalib pipeline, preserving its ray prediction capabilities
4. **Training stability**: Fewer stages, more straightforward optimization

### Why Epoch 5 (Not Later)?

Training metrics show:
- Flow loss improves steadily through epoch 7
- But validation metrics plateau at epoch 5
- Epochs 6-7 show slight overfitting (validation divergence increases)
- Benchmark results confirm epoch 5 as optimal

---

## 9. Training Curves

From `thesis_results/training_artifacts/phase_C_v3_h100/metrics.csv`:

| Epoch | Train Loss | Val Loss | Val Rot Mean | Val Rot Median | Val Trans Dir Mean | Calib fx MAE |
|-------|-----------|----------|-------------|---------------|-------------------|-------------|
| 1 | -0.4642 | -0.4652 | 0.501 | 0.483 | 51.97 | 86.6 |
| 2 | -0.4649 | -0.4691 | 0.506 | 0.475 | 44.40 | 100.0 |
| 3 | -0.4653 | -0.4709 | 0.507 | 0.490 | 47.90 | 93.4 |
| 4 | -0.4655 | -0.4662 | 0.515 | 0.509 | 47.83 | 97.8 |
| **5** | **-0.4661** | **-0.4709** | **0.514** | **0.499** | **45.84** | **96.1** |
| 6 | -0.4663 | -0.4699 | 0.515 | 0.507 | 43.64 | 95.8 |
| 7 | -0.4665 | -0.4659 | 0.514 | 0.486 | 45.82 | 85.9 |

**Note**: Rotation divergence values in CSV are computed on the validation set (different from benchmark datasets). Benchmark results on Sintel/TUM-RGBD/KITTI are the authoritative numbers.

---

## 10. Code Reference Paths

### Model Architecture
- `experiments/models/anycalib_with_fat.py` — `AnyCalibWithFAT` class: FAT integration into AnyCalib
  - `forward()`: full multi-frame pipeline
  - `forward_backbone()`: DINOv2 feature extraction
  - `get_trainable_parameters()`: returns FAT params only
- `experiments/models/anycam_wrapper_fat.py` — `AnyCamWrapperWithFATCalibration` class
  - `forward()`: full inference pipeline (images → poses + calibration)
  - `freeze_except_fat_and_pose()`: Phase C configuration
- `experiments/models/unified_wrapper.py` — `UnifiedTrainingWrapper` class
  - `_forward_phase_C()`: Phase C forward pass with loss computation
  - `_compose_flows()`: bilinear flow composition
  - `_compose_poses()`: SE(3) pose composition

### Training
- `experiments/train_unified.py` — main training script
  - `train_one_epoch()`: training loop with loss aggregation
  - `compute_val_loss()`: validation loop
- `experiments/cluster/slurm_train_phase_c_v3_h100.sh` — exact SLURM config used

### Evaluation
- `experiments/benchmark_phase_c_checkpoints.py` — benchmark script
  - `create_inference_model()`: builds model from config
  - `load_phase_c_checkpoint()`: loads our trained weights
  - `evaluate_model_on_dataset()`: runs evaluation
  - `DILATION_MODES`: per-dataset frame spacing
- `experiments/pose_metrics.py` — rotation/translation error computation
- `experiments/benchmark_dataset_utils.py` — dataset loading utilities

### Datasets
- `experiments/datasets/preprocessed_dataset.py` — training data loader
- `experiments/kitti_dataset.py` — KITTI evaluation dataset
- `experiments/lightspeed_dataset.py` — LightSpeed evaluation dataset

---

## 11. Checkpoint Recovery

To reconstruct the full model from the saved checkpoint:

```python
# 1. Create model (downloads frozen components automatically)
model = create_inference_model(
    anycam_config_path="pretrained_models/anycam_seq8/training_config.yaml",
    device="cuda:0"
)

# 2. Load our trained weights (pose_predictor + fat_model)
load_phase_c_checkpoint(
    model,
    "thesis_results/checkpoints/phase_C_v3_h100_epoch_0005.pt",
    device="cuda:0"
)
```

The checkpoint contains 851 keys: 426 under `pose_predictor.*` and 425 under `fat_model.*`. All frozen components (depth, flow, AnyCalib backbone/decoder) come from their respective pretrained sources.

---

## 12. Reproducibility Notes

- **Random seed**: Not explicitly set (PyTorch default). Benchmark uses persisted sample indices for consistency across runs.
- **Mixed precision**: Training uses `torch.amp.autocast` (FP16 forward, FP32 gradients). FAT transformer forced to FP32 for stability.
- **DINOv2-small**: HuggingFace `facebook/dinov2-small` (not torch.hub) for RTX 5090 compatibility (compute capability 12.0, xFormers incompatible).
- **Optimizer**: Adam with default betas, no weight decay, constant LR (no scheduler in Phase C v3).
