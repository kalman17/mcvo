"""
Benchmark Phase C checkpoints against Ground Truth evaluation datasets.

Evaluates per-epoch checkpoints comparing our model vs vanilla AnyCam baseline:
1. Pose accuracy vs GT (Sintel, TUM-RGBD, LightSpeed)
2. Calibration accuracy vs GT (Sintel, TUM-RGBD — datasets with known intrinsics)

Based on experiments/benchmark_phase3_checkpoints.py, adapted for Phase C
UnifiedTrainingWrapper checkpoints + multi-dataset GT evaluation.

Usage:
    python experiments/benchmark_phase_c_checkpoints.py \
        --checkpoint_dir /storage/user/maka/train/phase_C/checkpoints \
        --anycam_config pretrained_models/anycam_seq8/training_config.yaml \
        --pretrained_anycam pretrained_models/anycam_seq8/training_checkpoint_247500.pt \
        --data_root /storage/user/maka/eval_datasets \
        --datasets sintel,tumrgbd,lightspeed \
        --output_dir /storage/user/maka/train/phase_C/benchmark_results
"""

import argparse
import json
import logging
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import yaml

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.pose_metrics import (
    rotation_error_degrees,
    se3_distance,
    translation_direction_error_degrees,
    translation_magnitude_error,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Intrinsics error (reused from benchmark_phase3_checkpoints.py)
# ============================================================================

def compute_intrinsics_error(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    """Compute intrinsics errors (MAE and MAPE) between pred and gt [fx, fy, cx, cy]."""
    mae = np.abs(pred - gt)
    mape = np.abs(pred - gt) / (np.abs(gt) + 1e-8) * 100
    return {
        'fx_mae': float(mae[0]),
        'fy_mae': float(mae[1]),
        'cx_mae': float(mae[2]),
        'cy_mae': float(mae[3]),
        'fx_mape': float(mape[0]),
        'fy_mape': float(mape[1]),
        'f_mape': float((mape[0] + mape[1]) / 2),
    }


def aggregate_metrics(errors: List[Dict], keys: List[str]) -> Dict:
    """Compute mean and median for each key in a list of error dicts."""
    if not errors:
        logger.warning("No valid errors to aggregate — all samples may have failed")
        return {f'{k}_mean': float('nan') for k in keys} | {f'{k}_median': float('nan') for k in keys}
    result = {}
    for key in keys:
        values = [e[key] for e in errors if key in e and not np.isnan(e[key])]
        if values:
            result[f'{key}_mean'] = float(np.mean(values))
            result[f'{key}_median'] = float(np.median(values))
        else:
            result[f'{key}_mean'] = float('nan')
            result[f'{key}_median'] = float('nan')
    return result


# ============================================================================
# Dataset loading helpers
# ============================================================================

def load_sintel_dataset(data_root: str, num_samples: int, image_size: int, frame_count: int = 4, dilation: int = 1):
    """Load Sintel dataset for benchmarking (24 FPS native).

    Instantiates SintelDataset directly (bypassing make_datasets) to avoid:
    - make_datasets hardcodes return_depth=True for test_dataset
    - make_datasets sets index_selector=index_selector_pair which returns
      only 1 frame instead of N (keyframe excluded)
    """
    from anycam.datasets.sintel.sintel_dataset import SintelDataset
    from anycam.datasets.common import flow_selector_seq
    dataset = SintelDataset(
        data_path=str(Path(data_root) / 'Sintel' / 'training'),
        split_path=None,
        image_size=image_size,
        frame_count=frame_count,
        dilation=dilation,
        return_depth=False,
        return_flow=False,
        flow_selector=flow_selector_seq,
    )
    return dataset


def load_tumrgbd_dataset(data_root: str, num_samples: int, image_size: int, frame_count: int = 4, dilation: int = 10):
    """Load TUM-RGBD dataset for benchmarking (30 FPS native).

    Instantiates TUMRGBDDataset directly with absolute split path and no
    index_selector (so keyframe + targets are all returned).
    """
    from anycam.datasets.tum_rgbd.tumrgbd_dataset import TUMRGBDDataset
    from anycam.datasets.common import flow_selector_seq
    repo_root = Path(__file__).resolve().parent.parent
    split_path = str(repo_root / 'anycam' / 'datasets' / 'tum_rgbd' / 'splits' / 'dynamic_seqs' / 'train_files.txt')
    dataset = TUMRGBDDataset(
        data_path=str(Path(data_root) / 'TUM_RGBD'),
        split_path=split_path,
        image_size=image_size,
        frame_count=frame_count,
        dilation=dilation,
        return_depth=False,
        return_flow=False,
        flow_selector=flow_selector_seq,
    )
    return dataset


def load_kitti_dataset(data_root: str, num_samples: int, image_size: int, frame_count: int = 4, dilation: int = 1):
    """Load KITTI Odometry dataset for benchmarking (10 FPS native, sequences 00-10 with GT)."""
    from experiments.kitti_dataset import KITTIOdometryDataset
    dataset = KITTIOdometryDataset(
        data_path=str(Path(data_root) / 'kitti_odom_color'),
        image_size=image_size,
        frame_count=frame_count,
        dilation=dilation,
    )
    return dataset


def load_lightspeed_dataset(data_root: str, num_samples: int, image_size: int, frame_count: int = 4, dilation: int = 1):
    """Load LightSpeed dataset (24 FPS native)."""
    from experiments.lightspeed_dataset import LightSpeedDataset
    lightspeed_dir = Path(data_root) / 'LightSpeed'
    dataset = LightSpeedDataset(
        lightspeed_dir=str(lightspeed_dir),
        num_frames=frame_count,
        image_size=(image_size, image_size),
    )
    return dataset


DATASET_LOADERS = {
    'sintel': load_sintel_dataset,
    'tumrgbd': load_tumrgbd_dataset,
    'kitti': load_kitti_dataset,
    'lightspeed': load_lightspeed_dataset,
}

# Datasets that have GT intrinsics (calibration)
DATASETS_WITH_GT_INTRINSICS = {'sintel', 'tumrgbd', 'kitti'}

# Dilation modes: controls frame spacing per dataset
# 'anycam' matches AnyCam's official evaluation protocol for fair baseline comparison
# 'training' matches our 2fps training data (native_fps / 2)
DILATION_MODES = {
    'anycam':   {'sintel': 1,  'tumrgbd': 10, 'kitti': 1,  'lightspeed': 1},
    'training': {'sintel': 12, 'tumrgbd': 15, 'kitti': 5,  'lightspeed': 12},
}


# ============================================================================
# Model creation
# ============================================================================

def create_inference_model(anycam_config_path: str, device: str):
    """
    Create an AnyCamWrapperWithFATCalibration model for inference benchmarking.

    This wrapper computes depth, flow, and calibration live from raw images
    (unlike UnifiedTrainingWrapper which expects preprocessed data).
    """
    from experiments.models.anycalib_with_fat import AnyCalibWithFAT
    from experiments.models.anycam_wrapper_fat import AnyCamWrapperWithFATCalibration

    with open(anycam_config_path, 'r') as f:
        full_config = yaml.safe_load(f)

    pose_predictor_config = full_config['model']['pose_predictor']
    depth_predictor_config = full_config['model']['depth_predictor']

    fat_config = {
        "embed_dim": 1024,
        "num_heads": 8,
        "num_layers": 2,
        "dropout": 0.1,
        "use_learnable_agg_token": False,
        "use_visual_conditioning": True,
        "visual_token_dim": 384,
        "num_scales": 4,
    }

    fat_model = AnyCalibWithFAT(
        model_id="anycalib_pinhole",
        use_fat=True,
        fat_config=fat_config,
        use_dinov2_small=True,
        use_dinov2_full=False,
        freeze_backbone=True,
        freeze_decoder=True,
        freeze_calibrator=True,
    )
    fat_model = fat_model.to(device)

    model = AnyCamWrapperWithFATCalibration(
        fat_model=fat_model,
        pose_predictor_config=pose_predictor_config,
        depth_predictor_config=depth_predictor_config,
        use_provided_depth=False,
        use_provided_flow=False,
    )

    return model


def create_baseline_model(anycam_config_path: str, pretrained_path: str, device: str):
    """
    Create the vanilla AnyCam baseline model from the original CVPR 2025 paper.

    This is the UNMODIFIED AnyCam pipeline with the 32-candidate focal length
    system. No AnyCalib, no FAT — just the original model.
    """
    from omegaconf import OmegaConf
    from anycam.scripts.common import load_model

    config = OmegaConf.load(anycam_config_path)
    config["model"]["use_provided_flow"] = False
    config["model"]["train_directions"] = "forward"

    model = load_model(config, pretrained_path)
    model = model.to(device)
    model.eval()
    return model


def load_phase_c_checkpoint(model, checkpoint_path: str, device: str):
    """
    Load Phase C checkpoint weights into the inference model.

    Phase C checkpoint contains both pose_predictor and fat_model state dicts
    under the UnifiedTrainingWrapper naming. We remap to match
    AnyCamWrapperWithFATCalibration.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    epoch = checkpoint.get('epoch', -1)
    state_dict = checkpoint.get('model_state_dict', checkpoint)

    # Filter out shape-mismatched keys (e.g. pose_head.proj0 changed by focal_embed_dim)
    current_state = model.state_dict()
    filtered = {
        k: v for k, v in state_dict.items()
        if k not in current_state or current_state[k].shape == v.shape
    }
    n_skipped = len(state_dict) - len(filtered)

    missing, unexpected = model.load_state_dict(filtered, strict=False)

    # Log loading summary
    n_loaded = len(filtered) - len(unexpected)
    print(f"[CHECKPOINT] Loaded epoch {epoch}: {n_loaded} keys, "
          f"{len(missing)} missing, {len(unexpected)} unexpected, "
          f"{n_skipped} shape-mismatched (skipped)")

    return epoch


# ============================================================================
# Pose + calibration evaluation on a single dataset
# ============================================================================

def extract_gt_from_sample(sample: Dict, dataset_name: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Extract GT poses and intrinsics from a dataset sample.

    Returns:
        gt_poses: [N, 4, 4] numpy array of absolute poses, or None
        gt_intrinsics: [N, 4] numpy array of [fx, fy, cx, cy], or None if not available
    """
    gt_poses = None
    gt_intrinsics = None

    # GT poses — all datasets provide 'poses' as [N, 4, 4]
    if 'poses' in sample:
        poses = sample['poses']
        if isinstance(poses, torch.Tensor):
            gt_poses = poses.numpy()
        else:
            gt_poses = np.array(poses, dtype=np.float32)
        if gt_poses.ndim == 2:
            gt_poses = gt_poses.reshape(-1, 4, 4)

    # GT intrinsics — from 'projs' key which is [N, 3, 3] K matrices
    if dataset_name in DATASETS_WITH_GT_INTRINSICS and 'projs' in sample:
        projs = sample['projs']
        if isinstance(projs, torch.Tensor):
            projs = projs.numpy()
        else:
            projs = np.array(projs, dtype=np.float32)
        # Extract [fx, fy, cx, cy] from 3x3 K matrices
        if projs.ndim == 3 and projs.shape[-1] == 3:
            n_frames = projs.shape[0]
            intrinsics = np.zeros((n_frames, 4), dtype=np.float32)
            for i in range(n_frames):
                intrinsics[i] = [projs[i, 0, 0], projs[i, 1, 1], projs[i, 0, 2], projs[i, 1, 2]]
            # Check if intrinsics are identity (LightSpeed placeholder) — skip if so
            if np.allclose(intrinsics[:, 0], 1.0) and np.allclose(intrinsics[:, 1], 1.0):
                gt_intrinsics = None
            else:
                gt_intrinsics = intrinsics

    return gt_poses, gt_intrinsics


def _run_model_forward(model: nn.Module, data: Dict, is_fat_model: bool) -> Dict:
    """
    Run forward pass on a model, handling the two different model interfaces:
    - AnyCamWrapperWithFATCalibration (our model): has forward_with_calibration_info()
    - AnyCamWrapper (vanilla baseline): has forward() returning data dict with proc_poses

    Returns a normalized output dict with:
        pred_poses: [N-1, 4, 4] numpy array of predicted relative poses
        model_intrinsics: [4] numpy array of model's focal prediction, or None
        baseline_intrinsics: [4] numpy from vanilla AnyCam's selected projection, or None
    """
    if is_fat_model:
        output = model.forward_with_calibration_info(data)
        poses = output['pose_result']['poses']  # [1, N-1, 1, 4, 4] or [1, N-1, 4, 4]
        if poses.dim() == 5:
            poses = poses[:, :, 0]
        pred_poses = poses[0].cpu().numpy()

        model_intr = None
        batch_intr = output.get('intrinsics')
        if batch_intr is not None:
            intr = batch_intr[0].cpu().numpy()  # [4] in ray resolution space
            # Scale from ray resolution to input image resolution
            fat_image_size = output.get('fat_image_size')
            if fat_image_size is not None:
                H_ray, W_ray = fat_image_size
                H_img, W_img = data['imgs'].shape[-2], data['imgs'].shape[-1]
                sx, sy = W_img / W_ray, H_img / H_ray
                model_intr = np.array([intr[0]*sx, intr[1]*sy, intr[2]*sx, intr[3]*sy], dtype=np.float32)
            else:
                model_intr = intr

        return {
            'pred_poses': pred_poses,
            'model_intrinsics': model_intr,
            'baseline_intrinsics': None,
        }
    else:
        # Vanilla AnyCamWrapper — forward() returns mutated data dict
        output = model(data)
        pred_poses = output['proc_poses']  # [B, N-1, 4, 4]
        pred_poses = pred_poses[0].cpu().numpy()

        # Extract the model's selected projection (from 32-candidate system)
        baseline_intr = None
        proc_projs = output.get('proc_projs')  # [B, 1, 3, 3]
        if proc_projs is not None:
            K = proc_projs[0, 0].cpu().numpy()  # [3, 3]
            h, w = data['imgs'].shape[-2], data['imgs'].shape[-1]
            # proc_projs are normalized by normalize_proj():
            #   fx_n = 2*fx/w, fy_n = 2*fy/h, cx_n = 2*cx/w - 1, cy_n = 2*cy/h - 1
            # Denormalize back to pixel space:
            fx = K[0, 0] * w / 2
            fy = K[1, 1] * h / 2
            cx = (K[0, 2] + 1) * w / 2
            cy = (K[1, 2] + 1) * h / 2
            baseline_intr = np.array([fx, fy, cx, cy], dtype=np.float32)

        return {
            'pred_poses': pred_poses,
            'model_intrinsics': baseline_intr,
            'baseline_intrinsics': baseline_intr,
        }


def evaluate_model_on_dataset(
    model: nn.Module,
    dataset,
    dataset_name: str,
    indices: np.ndarray,
    device: torch.device,
    model_label: str = "Model",
    is_fat_model: bool = True,
) -> Dict:
    """
    Evaluate a model on a dataset with GT poses (and optionally GT intrinsics).

    Args:
        is_fat_model: True for AnyCamWrapperWithFATCalibration (our model),
            False for vanilla AnyCamWrapper (baseline with 32-candidate system).

    Returns dict with:
        pose_errors: list of per-pair error dicts
        intrinsics_errors: list of per-sequence intrinsics error dicts
        num_sequences: number of valid sequences evaluated
    """
    model.eval()
    pose_errors = []
    intrinsics_errors = []
    num_valid = 0

    with torch.no_grad():
        for idx in indices:
            try:
                sample = dataset[int(idx)]

                gt_poses, gt_intrinsics = extract_gt_from_sample(sample, dataset_name)
                if gt_poses is None or gt_poses.shape[0] < 2:
                    continue

                # Prepare model input
                if 'imgs' in sample:
                    imgs = sample['imgs']  # [N, 3, H, W]
                elif 'images' in sample:
                    imgs = sample['images']
                else:
                    continue

                if isinstance(imgs, np.ndarray):
                    imgs = torch.from_numpy(imgs).float()
                imgs = imgs.unsqueeze(0).to(device)  # [1, N, 3, H, W]
                data = {'imgs': imgs}

                # Vanilla AnyCamWrapper requires data["projs"] (GT intrinsics as 3x3 K)
                if not is_fat_model:
                    if 'projs' in sample:
                        projs = sample['projs']
                        if isinstance(projs, np.ndarray):
                            projs = torch.from_numpy(projs).float()
                        data['projs'] = projs.unsqueeze(0).to(device)  # [1, N, 3, 3]
                    else:
                        # Construct identity-ish projs if not available
                        h, w = imgs.shape[-2], imgs.shape[-1]
                        K = torch.eye(3, device=device).unsqueeze(0).unsqueeze(0)
                        K = K.expand(1, imgs.shape[1], -1, -1).clone()
                        K[:, :, 0, 0] = 1.0  # normalized focal
                        K[:, :, 1, 1] = 1.0
                        K[:, :, 0, 2] = 0.5
                        K[:, :, 1, 2] = 0.5
                        data['projs'] = K

                # Forward pass (handles both model interfaces)
                fwd = _run_model_forward(model, data, is_fat_model)
                pred_poses = fwd['pred_poses']  # [N-1, 4, 4]

                # Compute pose errors (relative pose comparison)
                # pred_poses[i] = T_{i→last} (relative to last frame, which is identity)
                # Convert to frame-to-frame: T_{i→i+1} = inv(T_{i→last}) @ T_{i+1→last}
                n_pairs = min(len(pred_poses) - 1, gt_poses.shape[0] - 1)
                for i in range(n_pairs):
                    pred_rel = np.linalg.inv(pred_poses[i]) @ pred_poses[i + 1]
                    gt_rel = np.linalg.inv(gt_poses[i]) @ gt_poses[i + 1]

                    rot_err = rotation_error_degrees(pred_rel[:3, :3], gt_rel[:3, :3])
                    trans_dir = translation_direction_error_degrees(pred_rel[:3, 3], gt_rel[:3, 3])
                    trans_mag = translation_magnitude_error(pred_rel[:3, 3], gt_rel[:3, 3])
                    se3_dist = se3_distance(pred_rel, gt_rel)

                    if not any(np.isnan(v) for v in [rot_err, trans_dir, trans_mag, se3_dist]):
                        pose_errors.append({
                            'rotation_deg': float(rot_err),
                            'translation_direction_deg': float(trans_dir),
                            'translation_magnitude': float(trans_mag),
                            'se3_distance': float(se3_dist),
                        })

                # Compute intrinsics errors (if GT K available and model provides intrinsics)
                if gt_intrinsics is not None and fwd['model_intrinsics'] is not None:
                    gt_mean_intr = gt_intrinsics.mean(axis=0)  # [4]
                    intr_err = compute_intrinsics_error(fwd['model_intrinsics'], gt_mean_intr)
                    intrinsics_errors.append(intr_err)

                num_valid += 1
                torch.cuda.empty_cache()

            except Exception as e:
                print(f"[WARN] {model_label} failed on {dataset_name} idx={idx}: {e}")
                traceback.print_exc()
                continue

    return {
        'pose_errors': pose_errors,
        'intrinsics_errors': intrinsics_errors,
        'num_sequences': num_valid,
    }


# ============================================================================
# Benchmark a single checkpoint across all datasets
# ============================================================================

def benchmark_single_checkpoint(
    checkpoint_path: Path,
    datasets: Dict[str, Tuple],  # {name: (dataset, indices)}
    anycam_config_path: str,
    baseline_cache: Optional[Dict],
    device: torch.device,
) -> Dict:
    """
    Benchmark one Phase C checkpoint on all GT datasets.

    Returns per-dataset results including comparison with baseline.
    """
    epoch_start = time.time()
    print(f"\n{'=' * 70}")
    print(f"[BENCHMARK] Evaluating: {checkpoint_path.name}")
    print(f"{'=' * 70}")

    # Create model and load checkpoint
    model = create_inference_model(anycam_config_path, str(device))
    model = model.to(device)
    epoch = load_phase_c_checkpoint(model, str(checkpoint_path), str(device))
    model.eval()

    results = {
        'epoch': epoch,
        'checkpoint': checkpoint_path.name,
        'timestamp': datetime.now().isoformat(),
        'datasets': {},
    }

    for ds_name, (dataset, indices) in datasets.items():
        print(f"\n  --- {ds_name} ({len(indices)} sequences) ---")

        # Evaluate our model
        our_results = evaluate_model_on_dataset(
            model, dataset, ds_name, indices, device, model_label=f"Phase C (epoch {epoch})"
        )

        # Aggregate pose metrics
        pose_keys = ['se3_distance', 'rotation_deg', 'translation_magnitude', 'translation_direction_deg']
        our_pose = aggregate_metrics(our_results['pose_errors'], pose_keys)

        # Aggregate intrinsics metrics (if available)
        intr_keys = ['fx_mae', 'fy_mae', 'f_mape']
        our_intr = aggregate_metrics(our_results['intrinsics_errors'], intr_keys)

        ds_result = {
            'num_sequences': our_results['num_sequences'],
            'num_pose_errors': len(our_results['pose_errors']),
            'ours': {**our_pose, **our_intr},
        }

        # Add baseline comparison if cached
        if baseline_cache and ds_name in baseline_cache:
            baseline = baseline_cache[ds_name]
            ds_result['baseline'] = baseline

            # Compute improvement percentages — pose vs AnyCam
            improvement = {}
            for key in our_pose:
                if key in baseline and baseline[key] != 0:
                    pct = (baseline[key] - our_pose[key]) / abs(baseline[key]) * 100
                    improvement[key + '_improvement_pct'] = round(pct, 2)

            # Compute improvement percentages — calibration vs AnyCalib
            for key in our_intr:
                anycalib_key = f'anycalib_{key}'
                if anycalib_key in baseline and baseline[anycalib_key] != 0:
                    pct = (baseline[anycalib_key] - our_intr[key]) / abs(baseline[anycalib_key]) * 100
                    improvement[f'{key}_vs_anycalib_improvement_pct'] = round(pct, 2)

            # Compute improvement percentages — calibration vs AnyCam 32-candidate
            for key in our_intr:
                anycam_key = f'anycam_{key}'
                if anycam_key in baseline and baseline[anycam_key] != 0:
                    pct = (baseline[anycam_key] - our_intr[key]) / abs(baseline[anycam_key]) * 100
                    improvement[f'{key}_vs_anycam_improvement_pct'] = round(pct, 2)

            ds_result['improvement'] = improvement

        results['datasets'][ds_name] = ds_result

        # Print summary
        print(f"    Ours:     rot={our_pose.get('rotation_deg_mean', 0):.3f}° (mean), "
              f"trans_dir={our_pose.get('translation_direction_deg_mean', 0):.3f}°")
        if our_intr.get('f_mape_mean', 0) > 0:
            print(f"    Ours cal: f_MAPE={our_intr.get('f_mape_mean', 0):.2f}% (mean)")
        if baseline_cache and ds_name in baseline_cache:
            bl = baseline_cache[ds_name]
            print(f"    AnyCam:   rot={bl.get('rotation_deg_mean', 0):.3f}° (mean), "
                  f"trans_dir={bl.get('translation_direction_deg_mean', 0):.3f}°")
            if bl.get('anycam_f_mape_mean', 0) > 0:
                print(f"    AnyCam cal: f_MAPE={bl.get('anycam_f_mape_mean', 0):.2f}% (mean)")
            if bl.get('anycalib_f_mape_mean', 0) > 0:
                print(f"    AnyCalib: f_MAPE={bl.get('anycalib_f_mape_mean', 0):.2f}% (mean)")

    # Cleanup
    del model
    torch.cuda.empty_cache()

    elapsed = time.time() - epoch_start
    print(f"\n[BENCHMARK] Epoch {epoch} done in {elapsed:.0f}s")
    return results


# ============================================================================
# Baseline evaluation (run once, cache)
# ============================================================================

def evaluate_anycalib_calibration(
    dataset,
    dataset_name: str,
    indices: np.ndarray,
    device: torch.device,
) -> List[Dict]:
    """
    Run standalone AnyCalib on dataset images and compare against GT intrinsics.

    This is the calibration baseline — raw AnyCalib predictions (no FAT, no
    candidate selection, just per-frame focal length from AnyCalib).
    """
    from experiments.train_pose_head_anycalib import AnyCaLibBatchInference

    if dataset_name not in DATASETS_WITH_GT_INTRINSICS:
        return []

    anycalib = AnyCaLibBatchInference(device=str(device), use_multi_frame=True)
    intrinsics_errors = []

    with torch.no_grad():
        for idx in indices:
            try:
                sample = dataset[int(idx)]
                _, gt_intrinsics = extract_gt_from_sample(sample, dataset_name)
                if gt_intrinsics is None:
                    continue

                if 'imgs' in sample:
                    imgs = sample['imgs']
                elif 'images' in sample:
                    imgs = sample['images']
                else:
                    continue

                if isinstance(imgs, np.ndarray):
                    imgs = torch.from_numpy(imgs).float()
                imgs = imgs.unsqueeze(0).to(device)  # [1, N, 3, H, W]

                # AnyCalib predicts focal length in pixels
                focal = anycalib.predict_focal_length(imgs)  # [1]
                f_px = float(focal[0].cpu())
                h, w = imgs.shape[-2], imgs.shape[-1]
                pred_intr = np.array([f_px, f_px, w / 2.0, h / 2.0], dtype=np.float32)

                gt_mean_intr = gt_intrinsics.mean(axis=0)
                intr_err = compute_intrinsics_error(pred_intr, gt_mean_intr)
                intrinsics_errors.append(intr_err)

            except Exception as e:
                print(f"[WARN] AnyCalib failed on {dataset_name} idx={idx}: {e}")
                continue

    del anycalib
    torch.cuda.empty_cache()
    return intrinsics_errors


def evaluate_baseline(
    anycam_config_path: str,
    pretrained_path: str,
    datasets: Dict[str, Tuple],
    device: torch.device,
    cache_path: Optional[Path] = None,
) -> Dict:
    """
    Evaluate both baselines on all datasets. Cache results.

    Baselines:
    1. Vanilla AnyCam (original CVPR 2025 model, 32-candidate focal length)
       → Pose baseline: rotation, translation, SE(3) metrics
    2. Standalone AnyCalib (raw per-frame focal length predictions)
       → Calibration baseline: focal length MAPE vs GT

    Both are computed once and cached together.
    """
    # Check cache
    if cache_path and cache_path.exists():
        print(f"[BASELINE] Loading cached baseline from {cache_path}")
        with open(cache_path, 'r') as f:
            return json.load(f)

    print(f"\n{'=' * 70}")
    print(f"[BASELINE] Evaluating vanilla AnyCam (pose) + AnyCalib (calibration)")
    print(f"{'=' * 70}")

    # --- Part 1: Vanilla AnyCam for pose baseline ---
    print(f"\n[BASELINE] Part 1: Vanilla AnyCam (pose)")
    model = create_baseline_model(anycam_config_path, pretrained_path, str(device))

    baseline_results = {}

    for ds_name, (dataset, indices) in datasets.items():
        print(f"\n  --- {ds_name} ({len(indices)} sequences) ---")

        bl_results = evaluate_model_on_dataset(
            model, dataset, ds_name, indices, device,
            model_label="Vanilla AnyCam",
            is_fat_model=False,
        )

        pose_keys = ['se3_distance', 'rotation_deg', 'translation_magnitude', 'translation_direction_deg']
        bl_pose = aggregate_metrics(bl_results['pose_errors'], pose_keys)

        # AnyCam's 32-candidate selected intrinsics (from proc_projs)
        intr_keys = ['fx_mae', 'fy_mae', 'f_mape']
        anycam_intr = aggregate_metrics(bl_results['intrinsics_errors'], intr_keys)

        baseline_results[ds_name] = {
            **bl_pose,
            'num_sequences': bl_results['num_sequences'],
            'num_pose_errors': len(bl_results['pose_errors']),
            # AnyCam's own calibration (32-candidate selection)
            **{f'anycam_{k}': v for k, v in anycam_intr.items()},
        }

        print(f"    Pose: rot={bl_pose.get('rotation_deg_mean', 0):.3f}° (mean), "
              f"trans_dir={bl_pose.get('translation_direction_deg_mean', 0):.3f}°")

    del model
    torch.cuda.empty_cache()

    # --- Part 2: Standalone AnyCalib for calibration baseline ---
    print(f"\n[BASELINE] Part 2: AnyCalib (calibration)")
    for ds_name, (dataset, indices) in datasets.items():
        if ds_name not in DATASETS_WITH_GT_INTRINSICS:
            print(f"  --- {ds_name}: no GT intrinsics, skipping ---")
            continue

        print(f"  --- {ds_name} ({len(indices)} sequences) ---")
        anycalib_errors = evaluate_anycalib_calibration(dataset, ds_name, indices, device)

        intr_keys = ['fx_mae', 'fy_mae', 'f_mape']
        anycalib_intr = aggregate_metrics(anycalib_errors, intr_keys)

        if ds_name in baseline_results:
            baseline_results[ds_name].update(
                {f'anycalib_{k}': v for k, v in anycalib_intr.items()}
            )

        if anycalib_intr.get('f_mape_mean', 0) > 0:
            print(f"    AnyCalib: f_MAPE={anycalib_intr.get('f_mape_mean', 0):.2f}% (mean)")

    # Cache baseline results
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, 'w') as f:
            json.dump(baseline_results, f, indent=2)
        print(f"[BASELINE] Cached to {cache_path}")

    return baseline_results


# ============================================================================
# Plotting
# ============================================================================

def plot_benchmark_results(all_results: List[Dict], baseline_cache: Dict, output_dir: Path):
    """Plot benchmark results across epochs for all datasets."""
    if len(all_results) < 1:
        return

    epochs = [r['epoch'] for r in all_results]
    ds_names = list(all_results[0]['datasets'].keys())

    # Determine which datasets have calibration data
    ds_with_calib = [ds for ds in ds_names if ds in DATASETS_WITH_GT_INTRINSICS]
    n_cols = 3 if ds_with_calib else 2  # Add calibration column if any dataset has GT K

    n_datasets = len(ds_names)
    fig, axes = plt.subplots(n_datasets, n_cols, figsize=(7 * n_cols, 5 * n_datasets), squeeze=False)

    for row, ds_name in enumerate(ds_names):
        # Rotation error
        rot_mean = [r['datasets'].get(ds_name, {}).get('ours', {}).get('rotation_deg_mean', 0) for r in all_results]
        rot_median = [r['datasets'].get(ds_name, {}).get('ours', {}).get('rotation_deg_median', 0) for r in all_results]

        axes[row, 0].plot(epochs, rot_mean, 'o-', label='Ours (mean)', linewidth=2, color='blue')
        axes[row, 0].plot(epochs, rot_median, 's--', label='Ours (median)', linewidth=2, color='blue', alpha=0.6)

        if ds_name in baseline_cache:
            bl_rot = baseline_cache[ds_name].get('rotation_deg_mean', 0)
            axes[row, 0].axhline(y=bl_rot, color='red', linestyle=':', linewidth=2, label=f'AnyCam ({bl_rot:.2f}°)')

        axes[row, 0].set_xlabel('Epoch')
        axes[row, 0].set_ylabel('Rotation Error (degrees)')
        axes[row, 0].set_title(f'{ds_name}: Rotation Error')
        axes[row, 0].legend(fontsize=8)
        axes[row, 0].grid(True, alpha=0.3)

        # Translation direction error
        trans_mean = [r['datasets'].get(ds_name, {}).get('ours', {}).get('translation_direction_deg_mean', 0) for r in all_results]
        trans_median = [r['datasets'].get(ds_name, {}).get('ours', {}).get('translation_direction_deg_median', 0) for r in all_results]

        axes[row, 1].plot(epochs, trans_mean, 'o-', label='Ours (mean)', linewidth=2, color='green')
        axes[row, 1].plot(epochs, trans_median, 's--', label='Ours (median)', linewidth=2, color='green', alpha=0.6)

        if ds_name in baseline_cache:
            bl_trans = baseline_cache[ds_name].get('translation_direction_deg_mean', 0)
            axes[row, 1].axhline(y=bl_trans, color='red', linestyle=':', linewidth=2, label=f'AnyCam ({bl_trans:.2f}°)')

        axes[row, 1].set_xlabel('Epoch')
        axes[row, 1].set_ylabel('Translation Direction Error (degrees)')
        axes[row, 1].set_title(f'{ds_name}: Translation Direction Error')
        axes[row, 1].legend(fontsize=8)
        axes[row, 1].grid(True, alpha=0.3)

        # Calibration (focal length MAPE) — only for datasets with GT intrinsics
        if n_cols == 3:
            if ds_name in DATASETS_WITH_GT_INTRINSICS:
                fmape_mean = [r['datasets'].get(ds_name, {}).get('ours', {}).get('f_mape_mean', 0) for r in all_results]
                fmape_median = [r['datasets'].get(ds_name, {}).get('ours', {}).get('f_mape_median', 0) for r in all_results]

                axes[row, 2].plot(epochs, fmape_mean, 'o-', label='Ours FAT (mean)', linewidth=2, color='darkblue')
                axes[row, 2].plot(epochs, fmape_median, 's--', label='Ours FAT (median)', linewidth=2, color='darkblue', alpha=0.6)

                if ds_name in baseline_cache:
                    bl_ac = baseline_cache[ds_name].get('anycalib_f_mape_mean', 0)
                    if bl_ac > 0:
                        axes[row, 2].axhline(y=bl_ac, color='darkorange', linestyle=':', linewidth=2,
                                             label=f'AnyCalib ({bl_ac:.1f}%)')

                axes[row, 2].set_xlabel('Epoch')
                axes[row, 2].set_ylabel('Focal Length MAPE (%)')
                axes[row, 2].set_title(f'{ds_name}: Calibration (f MAPE)')
                axes[row, 2].legend(fontsize=8)
                axes[row, 2].grid(True, alpha=0.3)
            else:
                axes[row, 2].text(0.5, 0.5, f'{ds_name}\nNo GT intrinsics',
                                  ha='center', va='center', fontsize=12, color='gray',
                                  transform=axes[row, 2].transAxes)
                axes[row, 2].set_axis_off()

    plt.tight_layout()
    plt.savefig(output_dir / 'benchmark_across_epochs.png', dpi=150)
    plt.close()
    print(f"[PLOT] Saved: {output_dir / 'benchmark_across_epochs.png'}")


# ============================================================================
# Main
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='Benchmark Phase C checkpoints on GT datasets',
        epilog="""
Examples:
  # Benchmark a single checkpoint:
  python experiments/benchmark_phase_c_checkpoints.py \\
      --single_checkpoint /storage/user/maka/train/phase_C/checkpoints/epoch_0001.pt \\
      --data_root /storage/user/maka/eval_datasets \\
      --output_dir /storage/user/maka/train/phase_C/benchmark_results

  # Benchmark all checkpoints in a directory:
  python experiments/benchmark_phase_c_checkpoints.py \\
      --checkpoint_dir /storage/user/maka/train/phase_C/checkpoints \\
      --data_root /storage/user/maka/eval_datasets \\
      --output_dir /storage/user/maka/train/phase_C/benchmark_results
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--checkpoint_dir', type=str, default=None,
                        help='Directory containing epoch_*.pt checkpoint files (scans all)')
    parser.add_argument('--single_checkpoint', type=str, default=None,
                        help='Evaluate a single checkpoint file')
    parser.add_argument('--anycam_config', type=str,
                        default='pretrained_models/anycam_seq8/training_config.yaml',
                        help='Path to AnyCam config YAML')
    parser.add_argument('--pretrained_anycam', type=str,
                        default='pretrained_models/anycam_seq8/training_checkpoint_247500.pt',
                        help='Path to vanilla AnyCam pretrained checkpoint (baseline)')
    parser.add_argument('--data_root', type=str, required=True,
                        help='Root directory containing evaluation datasets (Sintel/, TUM_RGBD/, LightSpeed/)')
    parser.add_argument('--mode', type=str, default='quick', choices=['quick', 'full'],
                        help='Benchmark mode: quick (sintel+tumrgbd, 1000 samples) or full (+kitti, 5000 samples)')
    parser.add_argument('--datasets', type=str, default=None,
                        help='Comma-separated list of datasets (overrides --mode default)')
    parser.add_argument('--num_samples', type=int, default=None,
                        help='Max sequences per dataset (overrides --mode default)')
    parser.add_argument('--frame_count', type=int, default=4,
                        help='Number of frames per sequence (default 4, matching training max_ahead=3)')
    parser.add_argument('--image_size', type=int, default=336,
                        help='Image size (must match training)')
    parser.add_argument('--dilation_mode', type=str, default='anycam',
                        choices=['anycam', 'training'],
                        help='anycam: match AnyCam eval protocol (dilation=1/10); '
                             'training: match 2fps training data (dilation=12/15/5)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for benchmark results')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to run on')
    parser.add_argument('--skip_baseline', action='store_true',
                        help='Skip baseline evaluation (use cached if available)')
    args = parser.parse_args()

    # Apply mode defaults (user overrides take precedence)
    mode_defaults = {
        'quick': {'datasets': 'sintel,tumrgbd,kitti', 'num_samples': 200},
        'full':  {'datasets': 'sintel,tumrgbd,kitti', 'num_samples': 1000},
    }
    defaults = mode_defaults[args.mode]
    if args.datasets is None:
        args.datasets = defaults['datasets']
    if args.num_samples is None:
        args.num_samples = defaults['num_samples']

    if not args.single_checkpoint and not args.checkpoint_dir:
        parser.error('Either --single_checkpoint or --checkpoint_dir is required')

    return args


def main():
    args = parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(levelname)s: %(message)s')

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.single_checkpoint:
        print(f"[CONFIG] Checkpoint: {args.single_checkpoint}")
    else:
        print(f"[CONFIG] Checkpoint dir: {args.checkpoint_dir}")
    print(f"[CONFIG] Mode: {args.mode}")
    print(f"[CONFIG] Datasets: {args.datasets}")
    print(f"[CONFIG] Num samples per dataset: {args.num_samples}")
    print(f"[CONFIG] Frame count: {args.frame_count}")
    print(f"[CONFIG] Image size: {args.image_size}")
    print(f"[CONFIG] Dilation mode: {args.dilation_mode}")
    print(f"[CONFIG] Per-dataset dilations: {DILATION_MODES[args.dilation_mode]}")
    print(f"[CONFIG] Device: {device}")
    print(f"[CONFIG] Output: {output_dir}")

    # ---- Discover checkpoints ----
    if args.single_checkpoint:
        checkpoint_files = [Path(args.single_checkpoint)]
    else:
        checkpoint_dir = Path(args.checkpoint_dir)
        checkpoint_files = sorted([
            f for f in checkpoint_dir.glob('epoch_*.pt')
            if f.stem.startswith('epoch_') and f.stem != 'latest'
        ], key=lambda x: int(x.stem.split('_')[1]))

    if not checkpoint_files:
        print(f"[ERROR] No checkpoint files found")
        return

    print(f"\n[INFO] Found {len(checkpoint_files)} checkpoints:")
    for ckpt in checkpoint_files:
        print(f"  - {ckpt.name}")

    # ---- Load datasets + persisted sample indices ----
    # Sample indices are saved to a file so every checkpoint run uses the EXACT
    # same data, even across separate script invocations (e.g. from the watcher).
    indices_path = output_dir / 'sample_indices.json'
    saved_indices = {}
    if indices_path.exists():
        with open(indices_path, 'r') as f:
            saved_indices = json.load(f)
        print(f"[INDICES] Loaded persisted sample indices from {indices_path}")

    requested_datasets = [d.strip() for d in args.datasets.split(',')]
    loaded_datasets = {}
    dilation_map = DILATION_MODES[args.dilation_mode]

    for ds_name in requested_datasets:
        if ds_name not in DATASET_LOADERS:
            print(f"[WARN] Unknown dataset '{ds_name}', skipping")
            continue

        ds_dir_map = {
            'sintel': Path(args.data_root) / 'Sintel',
            'tumrgbd': Path(args.data_root) / 'TUM_RGBD',
            'kitti': Path(args.data_root) / 'kitti_odom_color',
            'lightspeed': Path(args.data_root) / 'LightSpeed',
        }
        ds_dir = ds_dir_map.get(ds_name)
        if ds_dir and not ds_dir.exists():
            print(f"[WARN] Dataset dir not found: {ds_dir}, skipping {ds_name}")
            continue

        ds_dilation = dilation_map.get(ds_name, 1)
        try:
            dataset = DATASET_LOADERS[ds_name](args.data_root, args.num_samples, args.image_size, args.frame_count, dilation=ds_dilation)

            # Use persisted indices if available, otherwise generate and save
            if ds_name in saved_indices:
                indices = np.array(saved_indices[ds_name], dtype=np.int64)
                # Validate indices are still in range
                indices = indices[indices < len(dataset)]
                print(f"[DATASET] {ds_name}: {len(dataset)} total, reusing {len(indices)} persisted samples")
            else:
                num_samples = min(args.num_samples, len(dataset))
                np.random.seed(42)
                indices = np.random.choice(len(dataset), num_samples, replace=False)
                saved_indices[ds_name] = indices.tolist()
                print(f"[DATASET] {ds_name}: {len(dataset)} total, selected {num_samples} NEW samples (seed=42)")

            loaded_datasets[ds_name] = (dataset, indices)
        except Exception as e:
            print(f"[ERROR] Failed to load {ds_name}: {e}")
            traceback.print_exc()
            continue

    if not loaded_datasets:
        print("[ERROR] No datasets loaded successfully")
        return

    # Persist sample indices for future runs
    with open(indices_path, 'w') as f:
        json.dump({k: v if isinstance(v, list) else v.tolist()
                   for k, v in saved_indices.items()}, f, indent=2)
    print(f"[INDICES] Saved sample indices to {indices_path}")

    # ---- Evaluate baseline (once, cached) ----
    baseline_cache_path = output_dir / 'baseline_cache.json'
    baseline_cache = evaluate_baseline(
        args.anycam_config, args.pretrained_anycam,
        loaded_datasets, device, baseline_cache_path
    )

    # ---- Benchmark each checkpoint ----
    all_results = []

    for checkpoint_path in checkpoint_files:
        # Check if already benchmarked
        epoch_num = checkpoint_path.stem.split('_')[1]
        epoch_dir = output_dir / f'epoch_{epoch_num}'

        if (epoch_dir / 'results.json').exists():
            print(f"[SKIP] {checkpoint_path.name} already benchmarked, loading cached results")
            with open(epoch_dir / 'results.json', 'r') as f:
                cached = json.load(f)
            all_results.append(cached)
            continue

        results = benchmark_single_checkpoint(
            checkpoint_path, loaded_datasets,
            args.anycam_config, baseline_cache, device,
        )

        # Attach benchmark config metadata
        results['config'] = {
            'mode': args.mode,
            'frame_count': args.frame_count,
            'image_size': args.image_size,
            'num_samples': args.num_samples,
            'datasets': args.datasets,
            'dilation_mode': args.dilation_mode,
            'per_dataset_dilations': DILATION_MODES[args.dilation_mode],
            'per_dataset_samples': {ds_name: len(indices) for ds_name, (_, indices) in loaded_datasets.items()},
        }

        # Save per-epoch results
        epoch_dir.mkdir(parents=True, exist_ok=True)
        with open(epoch_dir / 'results.json', 'w') as f:
            json.dump(results, f, indent=2)

        all_results.append(results)

    # ---- Save aggregated results ----
    with open(output_dir / 'all_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    # ---- Plot ----
    plot_benchmark_results(all_results, baseline_cache, output_dir)

    # ---- Print summary table ----
    print(f"\n{'=' * 120}")
    print(f"[SUMMARY] Phase C Benchmark Results (Median Values)")
    print(f"{'=' * 120}")

    for ds_name in loaded_datasets.keys():
        print(f"\n  Dataset: {ds_name}")
        print(f"  {'Epoch':<8} {'Rot(°)':<12} {'TransDir(°)':<14} {'SE3':<10}", end='')
        if ds_name in DATASETS_WITH_GT_INTRINSICS:
            print(f" {'f_MAPE(%)':<12}", end='')
        print()
        print(f"  {'-' * 60}")

        for r in all_results:
            ds = r['datasets'].get(ds_name, {})
            ours = ds.get('ours', {})
            epoch = r['epoch']
            rot = ours.get('rotation_deg_median', 0)
            trans = ours.get('translation_direction_deg_median', 0)
            se3 = ours.get('se3_distance_median', 0)
            line = f"  {epoch:<8} {rot:<12.4f} {trans:<14.4f} {se3:<10.4f}"
            if ds_name in DATASETS_WITH_GT_INTRINSICS:
                fmape = ours.get('f_mape_median', 0)
                line += f" {fmape:<12.2f}"
            print(line)

        if ds_name in baseline_cache:
            bl = baseline_cache[ds_name]
            rot = bl.get('rotation_deg_median', 0)
            trans = bl.get('translation_direction_deg_median', 0)
            se3 = bl.get('se3_distance_median', 0)
            # AnyCam row with AnyCalib calibration baseline
            line = f"  {'AnyCam':<8} {rot:<12.4f} {trans:<14.4f} {se3:<10.4f}"
            if ds_name in DATASETS_WITH_GT_INTRINSICS:
                fmape = bl.get('anycalib_f_mape_median', 0)
                line += f" {fmape:<12.2f} (AnyCalib)"
            print(line)
            # Also show AnyCam's own 32-candidate calibration
            if ds_name in DATASETS_WITH_GT_INTRINSICS:
                anycam_fmape = bl.get('anycam_f_mape_median', 0)
                if anycam_fmape > 0:
                    line2 = f"  {'AnyCam32':<8} {'':>12} {'':>14} {'':>10}"
                    line2 += f" {anycam_fmape:<12.2f} (AnyCam)"
                    print(line2)

    print(f"\n{'=' * 120}")
    print(f"\nResults saved to: {output_dir}")


if __name__ == '__main__':
    main()
