#!/usr/bin/env python3
"""
=============================================================================
DA3 Calibration Head Training - Stage 3: End-to-End Flow Reprojection
=============================================================================

EXPERIMENT GOAL:
----------------
Integrate DA3 calibration head into full AnyCam pipeline and train with
flow reprojection loss (self-supervised).

STAGE 3 OBJECTIVE:
------------------
Train the DA3 calibration head end-to-end with the full AnyCam pipeline.
- Load Stage 2 checkpoint
- Freeze all except calibration head (or use LoRA)
- Train with flow reprojection loss (self-supervised)

LOSS FUNCTION:
--------------
L_stage3 = Flow_Reprojection_Loss(poses, focal_length, flows, depths)

TRAINING DATA:
--------------
- Objectron dataset with all available frames
- extract_all_pairs=True, max_pairs_per_video=None (unlimited)
- Self-supervised (no GT calibration or poses needed)
- GT directory argument is optional and not used in loss computation

PER-EPOCH POSE BENCHMARKING:
----------------------------
- Every epoch, evaluate pose accuracy on 20 test samples with GT
- Compare DA3 model vs AnyCam baseline (with 32 focal length candidates)
- Log and plot pose metrics separately from loss curves
- Cycling through test set samples across epochs

Author: AI Assistant for Kalman's Master's Thesis
Date: December 2025
"""

import sys
import os
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from tqdm import tqdm

# Add project paths
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "anycalib"))

# DA3 imports
from experiments.train_pose_head_anycalib import (
    ObjectronVideoDataset,
    AnyCaLibBatchInference,
    AnyCamWrapperWithDA3Calibration,
    AnyCamWrapperWithAnyCaLib,
    load_dataset_split,
)
from experiments.train_pose_head_anycalib_exp2 import (
    ObjectronVideoDatasetMultiFrame as BaseObjectronVideoDatasetMultiFrame,
)

# Override ObjectronVideoDatasetMultiFrame to handle __getitem__ properly
class ObjectronVideoDatasetMultiFrame(BaseObjectronVideoDatasetMultiFrame):
    """
    Extended ObjectronVideoDatasetMultiFrame that properly handles __getitem__
    for dictionary-based pair_info format.
    """
    
    def __init__(self, *args, **kwargs):
        """Override __init__ to ensure _build_pair_index is called."""
        # Extract max_ahead before calling super
        max_ahead = kwargs.pop('max_ahead', 3)
        self.max_ahead = max_ahead
        
        # Call parent __init__ (this may create tuple-based pair_info)
        # Parent class filters video_files based on video_indices, so self.video_files
        # is already the filtered list
        super().__init__(*args, **kwargs)
        
        # Override with dictionary-based pair_info by calling our _build_pair_index
        self._build_pair_index()
        
        print(f"[DATASET] Multi-frame mode: max_ahead={max_ahead}, loading {max_ahead + 1} frames per sequence")
    
    def _build_pair_index(self):
        """
        Build index of frame sequences for multi-frame training.
        For max_ahead=3, creates sequences like [0,1,2,3], [4,5,6,7], etc.
        """
        import cv2
        
        self.pair_info = []
        
        # Iterate over video_files (already filtered by parent class based on video_indices)
        # Use local index since video_files is already the filtered list
        for local_idx, video_path in enumerate(self.video_files):
            # Use local_idx as video_idx since video_files is already filtered
            video_idx = local_idx
            
            # Get total frames safely
            cap = cv2.VideoCapture(str(video_path))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            
            # Safety margin to avoid corrupted frames at the end
            safe_total_frames = max(0, total_frames - 2)
            
            if safe_total_frames < self.max_ahead + 1:
                print(f"[WARN] Video {video_path.name} has only {safe_total_frames} frames, skipping")
                continue
            
            # Create overlapping sequences to use all available frames
            # For max_ahead=3: sequences start at 0, 1, 2, 3, 4, 5, ... (step_size=1)
            step_size = 1  # Use all possible sequences (changed from 10 to maximize data usage)
            for start_frame in range(0, safe_total_frames - self.max_ahead, step_size):
                frame_indices = list(range(start_frame, start_frame + self.max_ahead + 1))
                
                self.pair_info.append({
                    'video_idx': video_idx,
                    'start_frame': start_frame,
                    'frame_indices': frame_indices,
                    'video_path': video_path,
                })
        
        print(f"[DATASET] Built multi-frame index: {len(self.pair_info)} sequences from {len(self.video_files)} videos")
        print(f"[DATASET] Each sequence contains {self.max_ahead + 1} frames (max_ahead={self.max_ahead})")
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Load a sequence of frames and optionally ground truth data.
        
        Handles dictionary-based pair_info format from _build_pair_index.
        """
        import cv2
        import json
        import numpy as np
        
        # Get pair info (dict format)
        pair_info = self.pair_info[idx]
        video_idx = pair_info['video_idx']
        start_frame = pair_info['start_frame']
        frame_indices = pair_info.get('frame_indices', list(range(start_frame, start_frame + self.num_frames)))
        video_path = pair_info.get('video_path', self.video_files[video_idx])
        
        # Load frames from video
        frames, actual_frame_indices = self._load_frames_from_video(video_path, start_frame=start_frame)
        
        # Convert frames to tensors
        imgs = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).float() / 255.0
        
        # Try to load ground truth (optional)
        projs = None
        poses = None
        
        if self.gt_dir:
            # Try multiple naming patterns
            gt_path1 = self.gt_dir / f"{video_path.stem}.json"
            stem_without_video = video_path.stem.replace("_video", "")
            gt_path2 = self.gt_dir / f"{stem_without_video}.json"
            
            gt_path = None
            if gt_path1.exists():
                gt_path = gt_path1
            elif gt_path2.exists():
                gt_path = gt_path2
            
            if gt_path:
                try:
                    with open(gt_path, 'r') as f:
                        gt_data = json.load(f)
                    # Extract poses first (critical for benchmarking)
                    try:
                        poses_np = self._extract_camera_poses(gt_data, actual_frame_indices)
                        poses = torch.from_numpy(poses_np).float()
                    except Exception as e:
                        print(f"[WARN] Failed to load GT poses for {video_path.name}: {e}")
                        poses = None
                    # Extract intrinsics if available; otherwise will fall back to identity K
                    try:
                        projs_np = self._extract_projection_matrices(gt_data, actual_frame_indices)
                        projs = torch.from_numpy(projs_np).float()
                    except Exception as e:
                        print(f"[WARN] Failed to load GT intrinsics for {video_path.name}: {e}")
                        projs = None
                except Exception as e:
                    print(f"[WARN] Failed to load GT for {video_path.name}: {e}")
        
        # If no GT available, create dummy placeholders (not used in unsupervised training)
        if projs is None:
            # Create dummy identity projection matrices
            projs = torch.eye(3).unsqueeze(0).repeat(self.num_frames, 1, 1)
        
        if poses is None:
            # Create dummy identity pose matrices
            poses = torch.eye(4).unsqueeze(0).repeat(self.num_frames, 1, 1)
        
        return {
            'imgs': imgs,  # [num_frames, 3, H, W]
            'projs': projs,  # [num_frames, 3, 3]
            'poses': poses,  # [num_frames, 4, 4]
            'video_name': video_path.stem,
            'frame_indices': actual_frame_indices,
        }

# Additional imports needed for training
from experiments.train_calibration_head_da3_stage1 import (
    plot_loss_curve,
    save_training_summary,
)
from experiments.dataset_paths import (
    get_objectron_videos, get_objectron_gt, get_lightspeed_root
)
from experiments.pose_metrics import (
    rotation_error_degrees,
    translation_direction_error_degrees,
    translation_magnitude_error,
    compute_error_statistics,
)

# AnyCam imports
from anycam.models import make_pose_predictor, make_depth_predictor
from anycam.loss import make_loss

print("[INIT] Imports successful")


# =============================================================================
# POSE BENCHMARKING FUNCTIONS
# =============================================================================

class PoseBenchmarkIterator:
    """
    Iterator that provides test set samples for per-epoch benchmarking.
    
    Can operate in two modes:
    - Cycling mode: Cycles through test set samples across epochs
    - Fixed mode: Always returns the same fixed set of samples (no cycling)
    """
    
    def __init__(self, dataset: ObjectronVideoDataset, num_samples: int = 100, no_cycle: bool = True):
        """
        Initialize with a dataset that has extract_all_pairs=True.
        
        Args:
            dataset: Dataset to sample from
            num_samples: Number of samples to use (or max available if less)
            no_cycle: If True, use fixed samples (no cycling). If False, cycle through dataset.
        
        The dataset's pair_info list has format: [(video_idx, start_frame), ...] or
        [{'video_idx': ..., 'start_frame': ..., ...}, ...] for multi-frame datasets.
        """
        self.dataset = dataset
        self.no_cycle = no_cycle
        
        # Handle different pair_info formats
        if len(dataset.pair_info) > 0:
            if isinstance(dataset.pair_info[0], dict):
                # Multi-frame format: list of dicts
                self.ordered_indices = list(range(len(dataset.pair_info)))
            else:
                # Standard format: list of tuples
                # Reorganize pairs: group by pair_idx first, then video_idx
                pair_by_pair_idx = {}
                for i, pair_info in enumerate(dataset.pair_info):
                    if isinstance(pair_info, tuple):
                        video_idx, start_frame = pair_info
                        pair_idx = start_frame // dataset.num_frames
                    else:
                        # Handle dict format
                        video_idx = pair_info.get('video_idx', 0)
                        start_frame = pair_info.get('start_frame', 0)
                        pair_idx = start_frame // dataset.num_frames
                    
                    if pair_idx not in pair_by_pair_idx:
                        pair_by_pair_idx[pair_idx] = []
                    pair_by_pair_idx[pair_idx].append(i)
                
                # Flatten: first all pair_idx=0, then all pair_idx=1, etc.
                self.ordered_indices = []
                for pair_idx in sorted(pair_by_pair_idx.keys()):
                    self.ordered_indices.extend(pair_by_pair_idx[pair_idx])
        else:
            self.ordered_indices = []
        
        # Store fixed indices (first num_samples, or all if less)
        max_available = len(self.ordered_indices)
        num_to_use = min(num_samples, max_available)
        self.fixed_indices = self.ordered_indices[:num_to_use]
        
        if self.no_cycle:
            print(f"[BENCHMARK] Fixed mode: Using {len(self.fixed_indices)} fixed samples (no cycling)")
        else:
            self.current_idx = 0
            print(f"[BENCHMARK] Cycling mode: {len(self.ordered_indices)} samples available, using {num_to_use} per epoch")
    
    def get_next_samples(self, num_samples: int) -> List[Dict]:
        """
        Get samples from the test set.
        
        In fixed mode (no_cycle=True): Always returns the same fixed samples.
        In cycling mode (no_cycle=False): Cycles through dataset.
        
        Returns:
            List of sample dictionaries from the dataset
        """
        samples = []
        
        if self.no_cycle:
            # Fixed mode: always return same samples
            indices_to_use = self.fixed_indices[:num_samples]
        else:
            # Cycling mode: advance through dataset
            indices_to_use = []
            for _ in range(num_samples):
                dataset_idx = self.ordered_indices[self.current_idx]
                indices_to_use.append(dataset_idx)
                self.current_idx = (self.current_idx + 1) % len(self.ordered_indices)
        
        # Load samples
        for dataset_idx in indices_to_use:
            try:
                sample = self.dataset[dataset_idx]
                samples.append(sample)
            except Exception as e:
                print(f"[WARN] Failed to load sample {dataset_idx}: {e}")
        
        return samples


def evaluate_pose_single_sample(
    model: nn.Module,
    batch_data: Dict,
    device: torch.device,
    model_name: str = "Model"
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Evaluate pose prediction for a single sample.
    
    Returns:
        Tuple of (rotation_error, trans_dir_error, trans_mag_error) or (None, None, None) if failed
    """
    model.eval()
    
    try:
        # Move data to device
        batch_data_gpu = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                         for k, v in batch_data.items()}
        
        # Check for GT poses
        if 'poses' not in batch_data_gpu:
            return None, None, None
        
        gt_poses = batch_data_gpu['poses']  # [batch, num_frames, 4, 4]
        
        # Compute ground truth relative poses
        batch_size, num_frames = gt_poses.shape[0], gt_poses.shape[1]
        if num_frames < 2:
            return None, None, None
        
        pose1 = gt_poses[:, 0]  # [batch, 4, 4]
        pose2 = gt_poses[:, 1]  # [batch, 4, 4]
        gt_rel_pose = torch.linalg.inv(pose1) @ pose2  # T_1->2
        
        # Get model predictions
        with torch.no_grad():
            output = model(batch_data_gpu)
            pred_poses = output['proc_poses']  # [batch, num_frames-1, 4, 4] or similar
            
            # Handle different output shapes
            if pred_poses.dim() == 4 and pred_poses.shape[1] == num_frames:
                # Extract relative pose from absolute poses
                pred_pose1 = pred_poses[:, 0]
                pred_pose2 = pred_poses[:, 1]
                pred_rel_pose = torch.linalg.inv(pred_pose1) @ pred_pose2
            elif pred_poses.dim() == 4 and pred_poses.shape[1] == num_frames - 1:
                # Already relative pose
                pred_rel_pose = pred_poses[:, 0]
            else:
                # Try to use first pose directly
                pred_rel_pose = pred_poses[:, 0] if pred_poses.dim() >= 2 else pred_poses
        
        # Compute errors (batch_size should be 1 typically)
        pred_np = pred_rel_pose[0].cpu().numpy()  # [4, 4]
        gt_np = gt_rel_pose[0].cpu().numpy()  # [4, 4]
        
        rot_err = rotation_error_degrees(pred_np[:3, :3], gt_np[:3, :3])
        trans_dir_err = translation_direction_error_degrees(pred_np[:3, 3], gt_np[:3, 3])
        trans_mag_err = translation_magnitude_error(pred_np[:3, 3], gt_np[:3, 3])
        
        if np.isnan(rot_err) or np.isnan(trans_dir_err) or np.isnan(trans_mag_err):
            return None, None, None
        
        return float(rot_err), float(trans_dir_err), float(trans_mag_err)
        
    except Exception as e:
        print(f"[WARN] Pose evaluation failed for {model_name}: {e}")
        return None, None, None


def benchmark_pose_accuracy(
    da3_model: nn.Module,
    baseline_model: nn.Module,
    benchmark_iterator: PoseBenchmarkIterator,
    num_samples: int,
    device: torch.device,
    epoch: int,
) -> Dict:
    """
    Run pose benchmark comparing DA3 model vs AnyCam baseline.
    
    Args:
        da3_model: DA3 calibration model
        baseline_model: AnyCam baseline with 32 candidates
        benchmark_iterator: Iterator for test samples
        num_samples: Number of samples to evaluate
        device: Torch device
        epoch: Current epoch number
    
    Returns:
        Dictionary with benchmark results
    """
    print(f"\n[BENCHMARK] Epoch {epoch}: Evaluating pose accuracy on {num_samples} samples...")
    
    # Get samples
    samples = benchmark_iterator.get_next_samples(num_samples)
    
    da3_rot_errors = []
    da3_trans_dir_errors = []
    da3_trans_mag_errors = []
    
    baseline_rot_errors = []
    baseline_trans_dir_errors = []
    baseline_trans_mag_errors = []
    
    for i, sample in enumerate(samples):
        # Create batch from single sample
        batch_data = {k: v.unsqueeze(0) if isinstance(v, torch.Tensor) else v 
                      for k, v in sample.items()}
        
        # Evaluate DA3 model
        da3_rot, da3_trans_dir, da3_trans_mag = evaluate_pose_single_sample(
            da3_model, batch_data, device, "DA3 Stage 3"
        )
        if da3_rot is not None:
            da3_rot_errors.append(da3_rot)
            da3_trans_dir_errors.append(da3_trans_dir)
            da3_trans_mag_errors.append(da3_trans_mag)
        
        # Evaluate baseline model
        baseline_rot, baseline_trans_dir, baseline_trans_mag = evaluate_pose_single_sample(
            baseline_model, batch_data, device, "AnyCam Baseline"
        )
        if baseline_rot is not None:
            baseline_rot_errors.append(baseline_rot)
            baseline_trans_dir_errors.append(baseline_trans_dir)
            baseline_trans_mag_errors.append(baseline_trans_mag)
        
        # Clear cache periodically
        if i % 5 == 0:
            torch.cuda.empty_cache()
    
    # Compute statistics
    results = {
        'epoch': epoch,
        'num_samples': num_samples,
        'num_valid_da3': len(da3_rot_errors),
        'num_valid_baseline': len(baseline_rot_errors),
    }
    
    if da3_rot_errors:
        results['da3'] = {
            'rot_mean': float(np.mean(da3_rot_errors)),
            'rot_median': float(np.median(da3_rot_errors)),
            'rot_std': float(np.std(da3_rot_errors)),
            'trans_dir_mean': float(np.mean(da3_trans_dir_errors)),
            'trans_dir_median': float(np.median(da3_trans_dir_errors)),
            'trans_mag_mean': float(np.mean(da3_trans_mag_errors)),
            'rot_errors': da3_rot_errors,
            'trans_dir_errors': da3_trans_dir_errors,
            'trans_mag_errors': da3_trans_mag_errors,
        }
    else:
        results['da3'] = None
    
    if baseline_rot_errors:
        results['baseline'] = {
            'rot_mean': float(np.mean(baseline_rot_errors)),
            'rot_median': float(np.median(baseline_rot_errors)),
            'rot_std': float(np.std(baseline_rot_errors)),
            'trans_dir_mean': float(np.mean(baseline_trans_dir_errors)),
            'trans_dir_median': float(np.median(baseline_trans_dir_errors)),
            'trans_mag_mean': float(np.mean(baseline_trans_mag_errors)),
            'rot_errors': baseline_rot_errors,
            'trans_dir_errors': baseline_trans_dir_errors,
            'trans_mag_errors': baseline_trans_mag_errors,
        }
    else:
        results['baseline'] = None
    
    # Print summary
    print(f"[BENCHMARK] Epoch {epoch} Results:")
    if results['da3']:
        print(f"  DA3 Stage 3: Rot={results['da3']['rot_mean']:.2f}° | Trans={results['da3']['trans_dir_mean']:.2f}°")
    if results['baseline']:
        print(f"  AnyCam Baseline: Rot={results['baseline']['rot_mean']:.2f}° | Trans={results['baseline']['trans_dir_mean']:.2f}°")
    
    return results


def plot_pose_benchmark_history(pose_history: List[Dict], save_dir: Path):
    """Plot pose benchmark results over epochs."""
    if not pose_history or len(pose_history) == 0:
        return
    
    # Extract data
    epochs = [h['epoch'] for h in pose_history]
    
    da3_rot_means = []
    da3_trans_means = []
    baseline_rot_means = []
    baseline_trans_means = []
    
    for h in pose_history:
        if h.get('da3'):
            da3_rot_means.append(h['da3']['rot_mean'])
            da3_trans_means.append(h['da3']['trans_dir_mean'])
        else:
            da3_rot_means.append(np.nan)
            da3_trans_means.append(np.nan)
        
        if h.get('baseline'):
            baseline_rot_means.append(h['baseline']['rot_mean'])
            baseline_trans_means.append(h['baseline']['trans_dir_mean'])
        else:
            baseline_rot_means.append(np.nan)
            baseline_trans_means.append(np.nan)
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Rotation error plot
    axes[0].plot(epochs, da3_rot_means, 'b-o', linewidth=2, markersize=6, label='DA3 Stage 3')
    axes[0].plot(epochs, baseline_rot_means, 'r--s', linewidth=2, markersize=6, label='AnyCam Baseline')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Rotation Error (degrees)', fontsize=12)
    axes[0].set_title('Rotation Error vs Epoch', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # Translation direction error plot
    axes[1].plot(epochs, da3_trans_means, 'b-o', linewidth=2, markersize=6, label='DA3 Stage 3')
    axes[1].plot(epochs, baseline_trans_means, 'r--s', linewidth=2, markersize=6, label='AnyCam Baseline')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Translation Direction Error (degrees)', fontsize=12)
    axes[1].set_title('Translation Direction Error vs Epoch', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = save_dir / "pose_benchmark_curve.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[SAVE] Pose benchmark curve saved: {plot_path}")


def save_pose_benchmark_log(pose_history: List[Dict], save_dir: Path):
    """Save pose benchmark results to log file."""
    log_path = save_dir / "pose_benchmark_log.txt"
    json_path = save_dir / "pose_benchmark_history.json"
    
    # Save JSON
    # Remove raw error arrays for cleaner JSON
    clean_history = []
    for h in pose_history:
        clean_h = {k: v for k, v in h.items() if k not in ['da3', 'baseline']}
        if h.get('da3'):
            clean_h['da3'] = {k: v for k, v in h['da3'].items() 
                            if k not in ['rot_errors', 'trans_dir_errors', 'trans_mag_errors']}
        if h.get('baseline'):
            clean_h['baseline'] = {k: v for k, v in h['baseline'].items()
                                  if k not in ['rot_errors', 'trans_dir_errors', 'trans_mag_errors']}
        clean_history.append(clean_h)
    
    with open(json_path, 'w') as f:
        json.dump(clean_history, f, indent=2)
    
    # Save text log
    with open(log_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("DA3 Stage 3 Training - Pose Benchmark History\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"{'Epoch':<8} {'DA3 Rot':<12} {'DA3 Trans':<12} {'Base Rot':<12} {'Base Trans':<12}\n")
        f.write("-"*80 + "\n")
        
        for h in pose_history:
            epoch = h['epoch']
            da3_rot = h['da3']['rot_mean'] if h.get('da3') else float('nan')
            da3_trans = h['da3']['trans_dir_mean'] if h.get('da3') else float('nan')
            base_rot = h['baseline']['rot_mean'] if h.get('baseline') else float('nan')
            base_trans = h['baseline']['trans_dir_mean'] if h.get('baseline') else float('nan')
            
            f.write(f"{epoch:<8} {da3_rot:<12.4f} {da3_trans:<12.4f} {base_rot:<12.4f} {base_trans:<12.4f}\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("Legend: Rot = Rotation Error (degrees), Trans = Translation Direction Error (degrees)\n")
        f.write("="*80 + "\n")
    
    print(f"[SAVE] Pose benchmark log saved: {log_path}")


# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================

def plot_loss_curve_stage3(loss_history: List[Dict], val_loss_history: Optional[List[Dict]], save_dir: Path):
    """Plot and save training and validation loss curves for Stage 3."""
    if not loss_history:
        return
    
    epochs = [item['epoch'] for item in loss_history]
    train_losses = [item['loss'] for item in loss_history]
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, 'b-', linewidth=2, label='Training Loss')
    
    # Plot validation if available
    val_epochs = []
    val_losses = []
    if val_loss_history:
        val_epochs = [item['epoch'] for item in val_loss_history]
        val_losses = [item['loss'] for item in val_loss_history]
        plt.plot(val_epochs, val_losses, 'r-', linewidth=2, label='Validation Loss')
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss (Flow Reprojection)', fontsize=12)
    plt.title('DA3 Stage 3 Training Loss', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    
    if len(train_losses) > 0:
        plt.annotate(f'Train Start: {train_losses[0]:.4f}', 
                    xy=(epochs[0], train_losses[0]), 
                    xytext=(10, 10), 
                    textcoords='offset points',
                    fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.5))
        plt.annotate(f'Train Final: {train_losses[-1]:.4f}', 
                    xy=(epochs[-1], train_losses[-1]), 
                    xytext=(-60, -20), 
                    textcoords='offset points',
                    fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.5))
    
    if val_loss_history and len(val_losses) > 0:
        plt.annotate(f'Val Final: {val_losses[-1]:.4f}', 
                    xy=(val_epochs[-1], val_losses[-1]), 
                    xytext=(-60, 20), 
                    textcoords='offset points',
                    fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightcoral', alpha=0.5))
    
    plot_path = save_dir / "loss_curve.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[SAVE] Loss curve saved: {plot_path}")

def validate_epoch_stage3(
    model: AnyCamWrapperWithDA3Calibration,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Run validation for one epoch and return average loss."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(dataloader):
            batch_data = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                         for k, v in batch_data.items()}
            try:
                with autocast(device_type='cuda', dtype=torch.float16):
                    output_data = model(batch_data)
                    loss_dict = criterion(output_data)
                    loss = loss_dict.get('loss', loss_dict.get('total_loss', sum(loss_dict.values())))
                
                # Check for NaN or Inf
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"[WARN] Validation batch {batch_idx} produced NaN/Inf loss, skipping")
                    continue
                
                total_loss += loss.item()
                num_batches += 1
            except Exception as e:
                print(f"[WARN] Validation batch {batch_idx} failed: {e}")
                continue
    
    model.train()
    if num_batches == 0:
        print("[WARN] No valid validation batches processed, returning 0.0")
        return 0.0
    return total_loss / num_batches


def train_stage3(
    model: AnyCamWrapperWithDA3Calibration,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    num_epochs: int,
    device: torch.device,
    save_dir: Path,
    val_dataloader: Optional[DataLoader] = None,
    benchmark_iterator: Optional[PoseBenchmarkIterator] = None,
    baseline_model: Optional[nn.Module] = None,
    benchmark_samples_per_epoch: int = 20,
    log_interval: int = 10,
    alternating_training: bool = False,
    learning_rate: float = 1e-5,
    start_epoch: int = 0,
    existing_loss_history: Optional[List] = None,
    existing_val_loss_history: Optional[List] = None,
    existing_pose_benchmark_history: Optional[List] = None,
):
    """
    Stage 3 training loop: End-to-end training with flow reprojection loss.
    
    Uses standalone DA3CalibrationHead with DINOv2-small visual tokens (vis_dim=384).
    This matches Stage 2's setup for consistent staged training.
    
    Includes per-epoch pose benchmarking if benchmark_iterator and baseline_model are provided.
    
    Args:
        alternating_training: If True, alternate between training calibration head and pose head each epoch.
        learning_rate: Learning rate for optimizer (used when recreating optimizer in alternating mode).
    """
    model.train()
    
    # Initial freeze/unfreeze setup
    if alternating_training:
        print(f"[TRAIN] Alternating training mode: Will alternate between calibration head and pose head each epoch")
        # Start with calibration head (epoch 0 will train calibration head)
        # Freeze everything first
        for param in model.parameters():
            param.requires_grad = False
        
        # Unfreeze calibration head for first epoch
        if hasattr(model, 'da3_calibration_head') and model.da3_calibration_head is not None:
            for param in model.da3_calibration_head.parameters():
                param.requires_grad = True
            print(f"[TRAIN] Epoch 0: Training calibration head (pose head frozen)")
        else:
            raise ValueError("Standalone DA3 calibration head not found in model!")
    else:
        # Standard mode: freeze everything except calibration head
        for param in model.parameters():
            param.requires_grad = False
        
        # Unfreeze only the standalone calibration head (not pose_predictor's internal one)
        if hasattr(model, 'da3_calibration_head') and model.da3_calibration_head is not None:
            for param in model.da3_calibration_head.parameters():
                param.requires_grad = True
            print(f"[TRAIN] Standalone DA3 calibration head: UNFROZEN")
        else:
            raise ValueError("Standalone DA3 calibration head not found in model!")
    
    # Count trainable parameters
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[MODEL] Trainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    
    # Initialize history from existing data if provided (for resume)
    loss_history = existing_loss_history if existing_loss_history is not None else []
    val_loss_history = existing_val_loss_history if existing_val_loss_history is not None else []
    pose_benchmark_history = existing_pose_benchmark_history if existing_pose_benchmark_history is not None else []
    batch_losses = []
    
    log_file = save_dir / "training_log.txt"
    loss_json_path = save_dir / "loss_history.json"
    loss_plot_path = save_dir / "loss_curve.png"
    
    # Initialize or append to log file
    if start_epoch > 0:
        # Append mode for resume
        with open(log_file, 'a') as f:
            f.write(f"\n{'='*70}\n")
            f.write(f"RESUMED TRAINING from epoch {start_epoch}\n")
            f.write(f"Previous training: {len(loss_history)} epochs completed\n")
            f.write(f"Continuing for {num_epochs - start_epoch} more epochs (total: {num_epochs})\n")
            f.write(f"{'='*70}\n\n")
    else:
        # Write mode for new training
        with open(log_file, 'w') as f:
            f.write(f"DA3 Stage 3 Training Log\n")
            f.write(f"{'='*70}\n")
            f.write(f"Num epochs: {num_epochs}\n")
            f.write(f"Batch size: {dataloader.batch_size}\n")
            f.write(f"Validation: {'Enabled' if val_dataloader else 'Disabled'}\n")
            f.write(f"Pose Benchmark: {'Enabled (' + str(benchmark_samples_per_epoch) + ' samples/epoch)' if benchmark_iterator else 'Disabled'}\n")
            f.write(f"Device: {device}\n")
            f.write(f"{'='*70}\n\n")
    
    print(f"\n{'='*70}")
    if start_epoch > 0:
        print(f"[TRAIN] RESUMING Stage 3 training from epoch {start_epoch}")
        print(f"[TRAIN] Previous training: {len(loss_history)} epochs completed")
        print(f"[TRAIN] Continuing for {num_epochs - start_epoch} more epochs (total: {num_epochs})")
    else:
        print(f"[TRAIN] Starting Stage 3 training for {num_epochs} epochs")
    print(f"[TRAIN] Loss: Flow reprojection (self-supervised)")
    if val_dataloader:
        print(f"[TRAIN] Validation: Enabled ({len(val_dataloader)} batches)")
    if benchmark_iterator and baseline_model:
        print(f"[TRAIN] Pose Benchmark: Enabled ({benchmark_samples_per_epoch} samples/epoch)")
    print(f"{'='*70}\n")
    
    scaler = GradScaler('cuda')
    
    for epoch in range(start_epoch, num_epochs):
        # Handle alternating training: switch which component is trainable each epoch
        if alternating_training:
            # Freeze everything first
            for param in model.parameters():
                param.requires_grad = False
            
            if epoch % 2 == 0:
                # Even epochs: Train calibration head, freeze pose head
                if hasattr(model, 'da3_calibration_head') and model.da3_calibration_head is not None:
                    for param in model.da3_calibration_head.parameters():
                        param.requires_grad = True
                print(f"[TRAIN] Epoch {epoch+1}: Training CALIBRATION HEAD (pose head frozen)")
            else:
                # Odd epochs: Train pose head, freeze calibration head
                # Find pose head parameters
                if hasattr(model, 'pose_predictor') and hasattr(model.pose_predictor, 'pose_head'):
                    for param in model.pose_predictor.pose_head.parameters():
                        param.requires_grad = True
                    print(f"[TRAIN] Epoch {epoch+1}: Training POSE HEAD (calibration head frozen)")
                else:
                    print(f"[WARN] Pose head not found, skipping pose head training")
                    # Fall back to calibration head
                    if hasattr(model, 'da3_calibration_head') and model.da3_calibration_head is not None:
                        for param in model.da3_calibration_head.parameters():
                            param.requires_grad = True
            
            # Recreate optimizer with new trainable parameters
            trainable_params = [p for p in model.parameters() if p.requires_grad]
            optimizer = optim.Adam(trainable_params, lr=learning_rate)
            
            # Count trainable parameters
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"[MODEL] Trainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
        
        epoch_loss = 0.0
        
        for batch_idx, batch_data in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            # Move data to device
            batch_data = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                         for k, v in batch_data.items()}
            
            # Forward pass
            optimizer.zero_grad()
            
            try:
                with autocast(device_type='cuda', dtype=torch.float16):
                    output_data = model(batch_data)
                    
                    # Compute loss (flow reprojection loss)
                    loss_dict = criterion(output_data)
                    loss = loss_dict.get('loss', loss_dict.get('total_loss', sum(loss_dict.values())))
                
                # Check for NaN or Inf
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"[WARN] Batch {batch_idx} produced NaN/Inf loss, skipping")
                    torch.cuda.empty_cache()
                    continue
                
                # Backward pass
                scaler.scale(loss).backward()
                
                # Check gradients
                if batch_idx % 100 == 0:
                    total_grad_norm = 0.0
                    for name, param in model.named_parameters():
                        if param.requires_grad and param.grad is not None:
                            param_grad_norm = param.grad.data.norm(2)
                            total_grad_norm += param_grad_norm.item() ** 2
                    total_grad_norm = total_grad_norm ** (1. / 2)
                    if total_grad_norm > 0:
                        print(f"[GRAD] Gradient norm: {total_grad_norm:.6f}")
                
                scaler.step(optimizer)
                scaler.update()
                
                epoch_loss += loss.item()
                batch_losses.append(loss.item())
                
                if batch_idx % log_interval == 0:
                    log_msg = (f"[TRAIN] Epoch {epoch+1}/{num_epochs} | "
                              f"Batch {batch_idx}/{len(dataloader)} | "
                              f"Loss: {loss.item():.6f}")
                    print(log_msg)
                    with open(log_file, 'a') as f:
                        f.write(f"{log_msg}\n")
                        
            except Exception as e:
                print(f"[ERROR] Batch {batch_idx} failed: {e}")
                import traceback
                traceback.print_exc()
                # Clear GPU cache on error to recover memory
                torch.cuda.empty_cache()
                continue
            
            # Periodic GPU cache clear to manage memory (24GB constraint)
            if batch_idx % 50 == 0 and batch_idx > 0:
                torch.cuda.empty_cache()
        
        # Clear cache at end of epoch
        torch.cuda.empty_cache()
        
        # Epoch summary
        avg_loss = epoch_loss / max(len(dataloader), 1)
        loss_history.append({'epoch': epoch + 1, 'loss': avg_loss})
        
        # Validation
        val_loss = None
        if val_dataloader:
            val_loss = validate_epoch_stage3(model, val_dataloader, criterion, device)
            if not (torch.isnan(torch.tensor(val_loss)) or torch.isinf(torch.tensor(val_loss))):
                val_loss_history.append({'epoch': epoch + 1, 'loss': val_loss})
            else:
                print(f"[WARN] Validation loss is NaN/Inf, not adding to history")
                val_loss = None
            log_msg = f"\n[EPOCH {epoch+1}] Train Loss: {avg_loss:.6f} | Val Loss: {val_loss if val_loss is not None else 'nan':.6f}\n"
        else:
            log_msg = f"\n[EPOCH {epoch+1}] Average Loss: {avg_loss:.6f}\n"
        
        print(log_msg)
        with open(log_file, 'a') as f:
            f.write(f"{log_msg}\n")
        
        # ===== PER-EPOCH POSE BENCHMARK =====
        if benchmark_iterator and baseline_model:
            benchmark_result = benchmark_pose_accuracy(
                da3_model=model,
                baseline_model=baseline_model,
                benchmark_iterator=benchmark_iterator,
                num_samples=benchmark_samples_per_epoch,
                device=device,
                epoch=epoch + 1,
            )
            pose_benchmark_history.append(benchmark_result)
            
            # Save and plot benchmark results after each epoch
            save_pose_benchmark_log(pose_benchmark_history, save_dir)
            plot_pose_benchmark_history(pose_benchmark_history, save_dir)
            
            # Put model back in training mode
            model.train()
        
        # Save loss history and plot after each epoch (overwrite previous)
        with open(loss_json_path, 'w') as f:
            json.dump({
                'epoch_losses': loss_history,
                'val_epoch_losses': val_loss_history,
                'batch_losses': batch_losses,
            }, f, indent=2)
        
        # Plot loss curve after each epoch (overwrite previous)
        plot_loss_curve_stage3(loss_history, val_loss_history if val_loss_history else None, save_dir)
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0 or epoch == num_epochs - 1:
            checkpoint_path = save_dir / "checkpoints" / f"checkpoint_epoch_{epoch+1}.pt"
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'val_loss': val_loss,
                'loss_history': loss_history,
                'val_loss_history': val_loss_history,
            }, checkpoint_path)
            print(f"[SAVE] Checkpoint saved to {checkpoint_path}")
        
        # Also save latest checkpoint after each epoch
        latest_checkpoint_path = save_dir / "checkpoints" / "latest_checkpoint.pt"
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
            'val_loss': val_loss,
            'loss_history': loss_history,
            'val_loss_history': val_loss_history,
        }, latest_checkpoint_path)
    
    # Save final model
    final_model_path = save_dir / "checkpoints" / "final_model.pt"
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss_history[-1]['loss'],
        'val_loss': val_loss_history[-1]['loss'] if val_loss_history else None,
        'loss_history': loss_history,
        'val_loss_history': val_loss_history,
    }, final_model_path)
    print(f"[SAVE] Final model saved to {final_model_path}")
    
    # Final loss history and plot (already saved each epoch, but save one more time)
    with open(loss_json_path, 'w') as f:
        json.dump({
            'epoch_losses': loss_history,
            'val_epoch_losses': val_loss_history,
            'batch_losses': batch_losses,
        }, f, indent=2)
    
    # Final plot (already saved each epoch, but save one more time)
    plot_loss_curve_stage3(loss_history, val_loss_history if val_loss_history else None, save_dir)
    save_training_summary(loss_history, batch_losses, save_dir, val_loss_history if val_loss_history else None)
    
    return loss_history, val_loss_history


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="DA3 Stage 3 Training: End-to-End Flow Reprojection")
    
    # Dataset arguments
    parser.add_argument("--objectron_videos", type=str, default=get_objectron_videos(),
                       help="Objectron videos directory")
    parser.add_argument("--objectron_gt", type=str, default=None,
                       help="Objectron GT directory (optional, not used in self-supervised training)")
    parser.add_argument("--split_file", type=str, default="experiments/objectron_split.json",
                       help="Dataset split file")
    parser.add_argument("--num_frames", type=int, default=None,
                       help="Number of frames per sequence (auto-set to max_ahead+1 if max_ahead > 2)")
    
    # Training arguments
    parser.add_argument("--num_epochs", type=int, default=50,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=2,
                       help="Batch size (memory intensive)")
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                       help="Learning rate (very low for fine-tuning)")
    parser.add_argument("--device", type=str, default="cuda:0",
                       help="Device to use")
    
    # Stage 2 checkpoint
    parser.add_argument("--stage2_checkpoint", type=str,
                       default="experiments/da3_integration/stage2_training/checkpoints/final_model.pt",
                       help="Path to Stage 2 checkpoint")
    
    # Resume training
    parser.add_argument("--resume", type=str, default=None,
                       help="Path to checkpoint to resume from (e.g., checkpoints/latest_checkpoint.pt). If provided, will resume from this checkpoint instead of loading Stage 2 checkpoint.")
    
    # Config file
    parser.add_argument("--config_file", type=str,
                       default="pretrained_models/anycam_seq8/training_config.yaml",
                       help="Path to training config file")
    
    # Output arguments
    parser.add_argument("--save_dir", type=str,
                       default="experiments/da3_integration/stage3_training_pose",
                       help="Directory to save results")
    
    # Pose benchmark arguments
    parser.add_argument("--enable_pose_benchmark", action="store_true", default=True,
                       help="Enable per-epoch pose benchmarking against AnyCam baseline")
    parser.add_argument("--disable_pose_benchmark", action="store_true",
                       help="Disable pose benchmarking (faster training)")
    parser.add_argument("--benchmark_samples", type=int, default=100,
                       help="Number of samples to evaluate per epoch for pose benchmark")
    parser.add_argument("--benchmark_no_cycle", action="store_true", default=True,
                       help="Use fixed benchmark samples (no cycling across epochs)")
    parser.add_argument("--baseline_checkpoint", type=str,
                       default="pretrained_models/anycam_seq8/training_checkpoint_247500.pt",
                       help="Path to AnyCam baseline checkpoint for comparison")
    
    # Stage 3.1 arguments: max_ahead and alternating training
    parser.add_argument("--max_ahead", type=int, default=2,
                       help="Number of frames ahead to predict (2=pairs, 3=4 frames, 4=5 frames)")
    parser.add_argument("--alternating_training", action="store_true",
                       help="Enable alternating training: freeze calibration/pose head each epoch")
    
    args = parser.parse_args()
    
    # Handle benchmark enable/disable flags
    if args.disable_pose_benchmark:
        args.enable_pose_benchmark = False
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[INIT] Using device: {device}")
    
    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / "checkpoints").mkdir(exist_ok=True)
    
    # Load config
    import yaml
    config_path = Path(args.config_file)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        full_config = yaml.safe_load(f)
    
    pose_predictor_config = full_config['model']['pose_predictor']
    depth_predictor_config = full_config['model']['depth_predictor']
    
    # Add use_da3_calibration to pose predictor config
    pose_predictor_config['use_da3_calibration'] = True
    
    # Load dataset split
    if Path(args.split_file).exists():
        split_data = load_dataset_split(args.split_file)
        train_indices = split_data.get('train', split_data.get('train_indices', []))
        val_indices = split_data.get('val', split_data.get('val_indices', []))
        test_indices = split_data.get('test', split_data.get('test_indices', []))
    else:
        raise FileNotFoundError(f"Split file not found: {args.split_file}")
    
    # Initialize AnyCalib
    anycalib_inference = AnyCaLibBatchInference(device=device)
    
    # Determine num_frames based on max_ahead
    if args.max_ahead > 2:
        # Multi-frame mode: load max_ahead + 1 frames
        num_frames = args.max_ahead + 1
        if args.num_frames is not None and args.num_frames != num_frames:
            print(f"[WARN] Overriding num_frames={args.num_frames} with max_ahead+1={num_frames}")
    else:
        # Pair mode: use 2 frames (or user-specified)
        num_frames = args.num_frames if args.num_frames is not None else 2
    
    # Create dataset (with all available frames, unlimited pairs)
    print(f"\n[STEP 1] Creating dataset...")
    print(f"[DATASET] max_ahead={args.max_ahead}, num_frames={num_frames}")
    
    if args.max_ahead > 2:
        # Use multi-frame dataset
        train_dataset = ObjectronVideoDatasetMultiFrame(
            videos_dir=args.objectron_videos,
            gt_dir=args.objectron_gt,
            num_frames=num_frames,
            video_indices=train_indices,
            require_gt=False,  # Self-supervised, no GT needed
            extract_all_pairs=False,  # Multi-frame dataset handles sequences differently
            max_pairs_per_video=999999,  # Effectively unlimited
            max_ahead=args.max_ahead,
        )
    else:
        # Use standard pair dataset
        train_dataset = ObjectronVideoDataset(
            videos_dir=args.objectron_videos,
            gt_dir=args.objectron_gt,
            num_frames=num_frames,
            video_indices=train_indices,
            require_gt=False,  # Self-supervised, no GT needed
            extract_all_pairs=True,  # Extract all consecutive pairs
            max_pairs_per_video=999999,  # Effectively unlimited (extract all pairs)
        )
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )
    
    # Create validation dataset if val_indices are available
    val_dataloader = None
    if val_indices:
        if args.max_ahead > 2:
            val_dataset = ObjectronVideoDatasetMultiFrame(
                videos_dir=args.objectron_videos,
                gt_dir=args.objectron_gt,
                num_frames=num_frames,
                video_indices=val_indices,
                require_gt=False,
                extract_all_pairs=False,
                max_pairs_per_video=999999,
                max_ahead=args.max_ahead,
            )
        else:
            val_dataset = ObjectronVideoDataset(
                videos_dir=args.objectron_videos,
                gt_dir=args.objectron_gt,
                num_frames=num_frames,
                video_indices=val_indices,
                require_gt=False,
                extract_all_pairs=True,
                max_pairs_per_video=999999,  # Effectively unlimited (extract all pairs)
            )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
        )
        print(f"[DATASET] Loaded {len(train_dataset)} training, {len(val_dataset)} validation sequences")
    else:
        print(f"[DATASET] Loaded {len(train_dataset)} training sequences (no validation set)")
    
    # Create test dataset for pose benchmarking (requires GT)
    benchmark_iterator = None
    baseline_model = None
    
    if args.enable_pose_benchmark and test_indices:
        # Need GT for pose benchmarking
        objectron_gt_dir = args.objectron_gt or get_objectron_gt()
        if objectron_gt_dir and Path(objectron_gt_dir).exists():
            print(f"\n[BENCHMARK] Creating test dataset for pose benchmarking...")
            if args.max_ahead > 2:
                test_dataset = ObjectronVideoDatasetMultiFrame(
                    videos_dir=args.objectron_videos,
                    gt_dir=objectron_gt_dir,
                    num_frames=num_frames,
                    video_indices=test_indices,
                    require_gt=True,  # Require GT for pose benchmarking
                    extract_all_pairs=False,
                    max_pairs_per_video=5,  # Limit sequences per video for benchmark
                    max_ahead=args.max_ahead,
                )
            else:
                test_dataset = ObjectronVideoDataset(
                    videos_dir=args.objectron_videos,
                    gt_dir=objectron_gt_dir,
                    num_frames=num_frames,
                    video_indices=test_indices,
                    require_gt=True,  # Require GT for pose benchmarking
                    extract_all_pairs=True,
                    max_pairs_per_video=5,  # Limit pairs per video for benchmark
                )
            
            if len(test_dataset) > 0:
                benchmark_iterator = PoseBenchmarkIterator(
                    test_dataset, 
                    num_samples=args.benchmark_samples,
                    no_cycle=args.benchmark_no_cycle
                )
                print(f"[BENCHMARK] Test dataset: {len(test_dataset)} sequences with GT")
            else:
                print(f"[WARN] Test dataset is empty, disabling pose benchmarking")
        else:
            print(f"[WARN] GT directory not found ({objectron_gt_dir}), disabling pose benchmarking")
    
    # Create baseline model for comparison (AnyCam with 32 candidates)
    if benchmark_iterator and args.enable_pose_benchmark:
        print(f"\n[BENCHMARK] Loading AnyCam baseline for comparison...")
        baseline_anycalib = AnyCaLibBatchInference(device=device)
        baseline_model = AnyCamWrapperWithAnyCaLib(
            pose_predictor_config=pose_predictor_config,
            depth_predictor_config=depth_predictor_config,
            anycalib_model=baseline_anycalib,
        ).to(device)
        
        # Load baseline checkpoint
        if Path(args.baseline_checkpoint).exists():
            print(f"[BENCHMARK] Loading baseline from {args.baseline_checkpoint}")
            baseline_checkpoint = torch.load(args.baseline_checkpoint, map_location=device, weights_only=False)
            if 'model' in baseline_checkpoint:
                state_dict = baseline_checkpoint['model']
            elif 'model_state_dict' in baseline_checkpoint:
                state_dict = baseline_checkpoint['model_state_dict']
            else:
                state_dict = baseline_checkpoint
            
            # Load with non-strict (baseline model structure may differ slightly)
            missing, unexpected = baseline_model.load_state_dict(state_dict, strict=False)
            if missing:
                print(f"[BENCHMARK] Missing keys (expected): {len(missing)} keys")
            print(f"[BENCHMARK] Baseline model loaded successfully")
            baseline_model.eval()
        else:
            print(f"[WARN] Baseline checkpoint not found: {args.baseline_checkpoint}")
            print(f"[WARN] Disabling pose benchmarking")
            benchmark_iterator = None
            baseline_model = None
    
    # Create model with DA3 calibration head
    # Uses standalone DINOv2-small (vis_dim=384) to match Stage 2's training setup
    print(f"\n[STEP 2] Creating model with DA3 calibration head...")
    print(f"[MODEL] Using standalone DA3CalibrationHead with vis_dim=384 (DINOv2-small)")
    print(f"[MODEL] This matches Stage 2's setup for consistent staged training")
    
    model = AnyCamWrapperWithDA3Calibration(
        pose_predictor_config=pose_predictor_config,
        depth_predictor_config=depth_predictor_config,
        anycalib_model=anycalib_inference,
        use_da3_calibration=True,
        vis_dim=384,  # Match Stage 2's DINOv2-small output dimension
    ).to(device)
    
    # Initialize resume state variables
    start_epoch = 0
    existing_loss_history = None
    existing_val_loss_history = None
    existing_pose_benchmark_history = None
    resume_checkpoint_data = None  # Store checkpoint data for optimizer loading later
    
    # Check if resuming from a checkpoint
    if args.resume and Path(args.resume).exists():
        print(f"[RESUME] Resuming from checkpoint: {args.resume}")
        resume_checkpoint_data = torch.load(args.resume, map_location=device, weights_only=False)
        
        # Load model state
        model.load_state_dict(resume_checkpoint_data['model_state_dict'])
        print(f"[RESUME] Model state loaded")
        
        # Load training state
        start_epoch = resume_checkpoint_data.get('epoch', 0)
        existing_loss_history = resume_checkpoint_data.get('loss_history', [])
        existing_val_loss_history = resume_checkpoint_data.get('val_loss_history', [])
        
        # Load pose benchmark history from JSON if it exists
        pose_benchmark_json = save_dir / "pose_benchmark_history.json"
        if pose_benchmark_json.exists():
            try:
                with open(pose_benchmark_json, 'r') as f:
                    existing_pose_benchmark_history = json.load(f)
                print(f"[RESUME] Loaded {len(existing_pose_benchmark_history)} pose benchmark entries")
            except Exception as e:
                print(f"[WARN] Failed to load pose benchmark history: {e}")
                existing_pose_benchmark_history = []
        else:
            existing_pose_benchmark_history = []
        
        print(f"[RESUME] Resuming from epoch {start_epoch}")
        print(f"[RESUME] Previous training: {len(existing_loss_history)} epochs completed")
        if existing_val_loss_history:
            print(f"[RESUME] Previous validation: {len(existing_val_loss_history)} epochs completed")
    else:
        # Load Stage 2 checkpoint (calibration head weights) - only if not resuming
        if args.resume:
            print(f"[WARN] Resume checkpoint not found: {args.resume}")
            print(f"[WARN] Falling back to Stage 2 checkpoint loading")
        
        if Path(args.stage2_checkpoint).exists():
            print(f"[LOAD] Loading Stage 2 checkpoint from {args.stage2_checkpoint}")
            checkpoint = torch.load(args.stage2_checkpoint, map_location=device, weights_only=False)
            
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            
            # Load into our standalone da3_calibration_head (not pose_predictor's internal one)
            try:
                model.da3_calibration_head.load_state_dict(state_dict, strict=True)
                print(f"[LOAD] Stage 2 calibration head weights loaded successfully (strict=True)")
            except RuntimeError as e:
                # Try non-strict loading if there are minor mismatches
                print(f"[WARN] Strict loading failed: {e}")
                print(f"[LOAD] Attempting non-strict loading...")
                model.da3_calibration_head.load_state_dict(state_dict, strict=False)
                print(f"[LOAD] Stage 2 calibration head weights loaded (strict=False)")
        else:
            print(f"[WARN] Stage 2 checkpoint not found: {args.stage2_checkpoint}")
            print(f"[WARN] Starting from random initialization")
    
    # Clear GPU cache after loading
    torch.cuda.empty_cache()
    
    # Create loss function (flow reprojection loss)
    # The config may have 'loss' as a list of dicts or a single dict
    loss_config_raw = full_config.get('loss', None)
    
    if isinstance(loss_config_raw, list) and len(loss_config_raw) > 0:
        # Take the first loss config from the list
        loss_config = loss_config_raw[0]
    elif isinstance(loss_config_raw, dict):
        loss_config = loss_config_raw
    else:
        # Use default loss config
        loss_config = {
            "type": "pose_loss",
            "flow_criterion": "l1",
            "dist_criterion": "l1",
            "lambda_flow": 1.0,
            "lambda_dist": 0.0,  # Match the config
            "use_flow_uncertainty": True,
            "use_dist_uncertainty": True,
        }
    
    print(f"[LOSS] Using loss config: {loss_config.get('type', 'pose_loss')}")
    criterion = make_loss(loss_config)
    print(f"[LOSS] Flow reprojection loss configured")
    
    # Create optimizer (only for calibration head)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(trainable_params, lr=args.learning_rate)
    
    # Load optimizer state if resuming
    if resume_checkpoint_data and 'optimizer_state_dict' in resume_checkpoint_data:
        try:
            optimizer.load_state_dict(resume_checkpoint_data['optimizer_state_dict'])
            print(f"[RESUME] Optimizer state loaded")
        except Exception as e:
            print(f"[WARN] Failed to load optimizer state: {e}")
            print(f"[WARN] Optimizer will start fresh (may affect learning rate scheduling)")
    
    # Train
    print(f"\n[STEP 3] Starting training...")
    loss_history, val_loss_history = train_stage3(
        model=model,
        dataloader=train_dataloader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=args.num_epochs,
        device=device,
        save_dir=save_dir,
        val_dataloader=val_dataloader,
        benchmark_iterator=benchmark_iterator,
        baseline_model=baseline_model,
        benchmark_samples_per_epoch=args.benchmark_samples,
        alternating_training=args.alternating_training,
        learning_rate=args.learning_rate,
        start_epoch=start_epoch,
        existing_loss_history=existing_loss_history,
        existing_val_loss_history=existing_val_loss_history,
        existing_pose_benchmark_history=existing_pose_benchmark_history,
    )
    
    print(f"\n{'='*70}")
    print(f"[COMPLETE] Stage 3 training complete!")
    print(f"[COMPLETE] Results saved to: {save_dir}")
    if val_loss_history:
        print(f"[COMPLETE] Final validation loss: {val_loss_history[-1]['loss']:.6f}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()

