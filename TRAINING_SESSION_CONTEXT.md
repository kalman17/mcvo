# Training Session Context Dump (March 11, 2026)

Dumped from active conversation context before compaction.

---

## V3 Model — Our Best (Phase C, Frozen Backbones)

### Configuration (from SLURM script slurm_train_phase_c_v3_h100.sh)
- **Phase**: C (only pose_head + FAT trainable, ~25M params)
- **Commit**: `8159f6b` (March 2, 2026)
- **batch_size**: 20 (H100 80GB)
- **learning_rate**: 1e-4
- **lambda_calib**: 1e-4
- **lambda_comp**: 0.1 (hardcoded in unified_wrapper.py, raw L1, effectively zero contribution)
- **max_ahead**: 3 (4-frame sequences)
- **image_size**: 336
- **Phase A checkpoint**: phase_A_v2/checkpoints/latest.pt
- **Phase B1 checkpoint**: phase_B1/checkpoints/latest.pt
- **Pretrained AnyCam**: pretrained_models/anycam_seq8/training_checkpoint_247500.pt
- **Epochs completed**: 5 (NaN collapse at epoch 6 due to FAT linear solver singularity)
- **Training loss at ep5**: flow ≈ -0.469 (from log train_C_v3_h100_1502859.out)
- **No weight_decay, no grad_clip** (these were added in later versions)
- **Optimizer**: AdamW
- **Scheduler**: Cosine annealing

### V3 Benchmark Results (Quick Mode — FINAL, from benchmark_Cb/all_results.json)

**Sintel (200 sequences)**:
| Metric | Ours | AnyCam Baseline | AnyCalib Baseline | Improvement |
|--------|------|-----------------|-------------------|-------------|
| Rotation mean | 0.602° | 0.669° | — | 10% better |
| Rotation median | 0.189° | 0.206° | — | 8% better |
| Trans direction mean | 65.0° | 89.3° | — | 27% better |
| Trans direction median | 52.3° | 86.5° | — | 40% better |
| f_MAPE mean | 27.8% | 68.2% | 30.7% | 59% vs AnyCam, 9% vs AnyCalib |
| f_MAPE median | 26.2% | 70.3% | 31.4% | 63% vs AnyCam, 16% vs AnyCalib |

**TUM-RGBD (200 sequences)**:
| Metric | Ours | AnyCam Baseline | AnyCalib Baseline | Improvement |
|--------|------|-----------------|-------------------|-------------|
| Rotation mean | 1.39° | 2.03° | — | 32% better |
| Rotation median | 0.708° | 1.23° | — | 43% better |
| Trans direction mean | 71.9° | 93.0° | — | 23% better |
| Trans direction median | 66.9° | 97.1° | — | 31% better |
| f_MAPE mean | 7.9% | 63.7% | 11.7% | 88% vs AnyCam, 32% vs AnyCalib |
| f_MAPE median | 6.1% | 14.6% | 11.5% | 58% vs AnyCam, 47% vs AnyCalib |

**KITTI (200 sequences)**:
| Metric | Ours | AnyCam Baseline | Improvement |
|--------|------|-----------------|-------------|
| Rotation mean | 0.545° | 0.556° | 2% better |
| Rotation median | 0.252° | 0.258° | 2% better |
| Trans direction mean | 77.2° | 91.1° | 15% better |
| Trans direction median | 70.9° | 91.5° | 22% better |
| f_MAPE | BROKEN (6660%) | BROKEN (94.8%) | Known KITTI normalization issue |

---

## Post-V3 Training Variants

### V3s (V3 continued with stability fixes, lambda_comp=0.0 intended but hardcoded 0.1)
- Resume from v3 ep5 weights, fresh optimizer
- Added: Tikhonov regularization, FAT fallback, weight_decay=1e-5, grad_clip=0.3
- v3s H100 (1506758→1507251): reached epoch 4, consec=-0.473, composed=-0.041
- v3s 48GB (1506759→1507252): reached epoch 3, PENDING in queue
- Both CANCELLED for restart with proper resume

### V4 (V3 + composed flow loss, lambda_comp=0.1 raw L1)
- Same as v3 but flow composition was supposed to help
- Result: WORSE than v3 on benchmarks (translation direction -11% to -12% vs v3)
- Composed loss was raw L1 (~0.008), negligible vs NLL consecutive (~-0.45)

### V5 (V4 weights + stability fixes)
- Trained from v4 epoch 5 weights with stability fixes
- H100 (1506325): reached ~epoch 8, consec=-0.446, composed=0.008 (still raw L1)
- 48GB (1506326): reached ~epoch 3, consec=-0.465
- CANCELLED — composed flow contribution still negligible

### V6 (V3 weights + PROPER flow composition with uncertainty weighting)
- **Key fix**: Added uncertainty weighting to composed loss in both `_forward_phase_a()` and `_forward_combined()`
- Composed loss now uses Laplacian NLL matching consecutive loss scale
- lambda_comp=0.5 (meaningful now that both losses on same NLL scale)
- Resume from v3 ep5 weights_only
- V6 H100 (1507253): reached epoch 2, consec=-0.478, composed=-0.044, CANCELLED (H100 contested)
- **V6 48GB (1507254): RUNNING on node14**, epoch ~2.5, consec=-0.478, composed=-0.047
  - Training loss BETTER than v3 (consec -0.478 vs -0.469)
  - But benchmark at intra_epoch2 WORSE than v3 (Sintel rot 0.671° vs 0.602°, trans_dir 98.4° vs 65.0°)
  - Likely needs more epochs to converge (fresh optimizer, only ~1.5 effective new epochs)

---

## Key Code Changes Made During This Session

### 1. Tikhonov Regularization (anycalib/anycalib/utils.py)
Added `AtWA = AtWA + 1e-6 * I` in `solve_2dweighted_lstsq()`, `cxcy_and_pix_ar_from_rays()`, `cxcy_from_rays()`. Prevents singular matrix errors that caused NaN at epoch 6-7.

### 2. FAT Calibrator Failure Handling (unified_wrapper.py `_forward_combined()`)
- Track `fat_success` per batch item
- On failure: fallback to GT avg calibration (detached)
- Skip calib_loss for failed samples

### 3. lambda_comp Fix (unified_wrapper.py)
- Was hardcoded to 0.1 inside `_forward_phase_a()` and `_forward_combined()`
- Fixed: now uses `self.lambda_comp` attribute, set from `args.lambda_comp`

### 4. Uncertainty-Weighted Composed Loss (unified_wrapper.py)
In both `_forward_phase_a()` and `_forward_combined()`:
```python
src_uncert = uncert[:, 0, 0, :1, :, :].to(torch.float32).clamp(min=0.01, max=10.0)
# Inside ahead loop after comp_err L1:
comp_err = comp_err * (2 ** 0.5) / (src_uncert + EPS) + (src_uncert + EPS).log()
comp_err = comp_err.clamp(max=10.0)
```

### 5. Training Stability Args (train_unified.py)
- Added `--weight_decay` (default 1e-5), `--grad_clip` (default 0.5) arguments
- Added gradient norm logging every 500 batches

---

## Cluster Job Status (as of March 11, 2026 ~11:00)

| Job | ID | Status | Notes |
|-----|-----|--------|-------|
| v3s H100 | 1507251 | CANCELLED | Was PENDING |
| v3s 48GB | 1507252 | PENDING | Waiting for GPU |
| v6 H100 | 1507253 | CANCELLED | H100 fully occupied |
| v6 48GB | 1507254 | RUNNING node14 | Epoch ~2.5, consec=-0.478 |
| bench v6 | 1507305 | RUNNING node1 (P6000) | Quick mode, KITTI in progress |

## Cluster Paths

| Item | Path |
|------|------|
| Repo | /storage/user/maka/anycam/ |
| Preprocessed data | /storage/user/maka/preprocessed/ (82K frames, 77K seqs) |
| V3 best checkpoint | /storage/user/maka/train/phase_C_v3_h100/checkpoints/epoch_0005.pt |
| Phase A v2 | /storage/user/maka/train/phase_A_v2/checkpoints/best.pt |
| Phase B1 | /storage/user/maka/train/phase_B1/checkpoints/best.pt |
| Benchmark FINAL | /storage/user/maka/train/benchmark_Cb/all_results.json |
| V3 training log | /storage/user/maka/logs/train_C_v3_h100_1502859.out |
| Eval datasets | /storage/user/maka/eval_datasets/{Sintel,TUM_RGBD,kitti_odom_color} |

## Cluster SSH
- Primary: `ssh -p 58022 maka@atcremers99.vision.in.tum.de`
- Backup: `ssh -p 58022 maka@devcube1.cvai.cit.tum.de`
- Account: `stud` QOS, cannot use DEADLINE partitions

## Architecture Quick Reference

### Phase C (V3) — What's Frozen vs Trainable
**Frozen**: DINOv2-small backbone, DINOv2 ViT-L backbone, DPT decoder, ray head, uncertainty head (`self.head`), depth predictor, flow estimator
**Trainable**: `pose_predictor.pose_head` (~21K), `fat_model.fat` (~25M FAT adapter)

### Phase Cb (V3s/V6) — Additional Unfreezes
Also unfreezes: `pose_reassemble_stage`, `pose_feature_fusion_stage`, `pose_interframe_attention`, `sequence_token_attention`, `sequence_token`, `sequence_info_head`, `focal_embedding`

### Uncertainty Head
`self.head` in anycam.py produces uncertainty. It's the DPT head repurposed for 2-channel uncertainty (flow + distance). **Always frozen** in Phase C and Cb. Cannot be "gamed" during training.

### Flow Composition
- `_compose_flows()`: bilinear warping of consecutive flows
- `_compose_poses()`: 4x4 matrix multiplication T_{0→N} = T_{0→1} @ T_{1→2} @ ... @ T_{N-1→N}
- For N=4 (max_ahead=3): predicts 0→1, 1→2, 2→3 consecutive + composes 0→2, 0→3

### Inference Model
`AnyCamWrapperWithFATCalibration` in anycam_wrapper_fat.py:
- `forward_with_calibration_info()` returns: poses, FAT intrinsics, per_frame_intrinsics, fat_rays
- Benchmark uses: `create_inference_model()` + `load_phase_c_checkpoint()` from benchmark_phase_c_checkpoints.py

---

## Why V6 Benchmark Was Worse Despite Better Loss

1. Training loss comparison not apples-to-apples: v6 includes composed loss (lambda_comp=0.5) in total
2. Fresh optimizer reset — model in transient state at only ~1.5 effective new epochs
3. v3 trained for 5 full epochs with consistent optimizer momentum
4. Uncertainty head IS frozen (verified: `self.head` not in unfreeze list), so not gaming NLL
5. V3 ep5 consecutive loss was flow=-0.469 (from training log), v6 consec=-0.478 is genuinely better
6. Need more epochs for v6 to generalize — training distribution ≠ eval distribution
