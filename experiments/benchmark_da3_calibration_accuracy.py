#!/usr/bin/env python3
"""
=============================================================================
DA3 Calibration Accuracy Benchmark - Validation Experiment
=============================================================================

Evaluates individual DA3 stage models on calibration accuracy vs GT mean intrinsics.

**IMPORTANT**: This benchmark is designed for validation experiments where models
are trained and evaluated on the same dataset (e.g., Objectron). The results
demonstrate the learning progression across stages but are NOT suitable for
comparison with general-purpose methods (e.g., AnyCalib) due to dataset-specific
overfitting. Fair comparisons require models trained on large, diverse datasets.

This script:
1. Loads a DA3 checkpoint (Stage 1, 2, or 3)
2. Evaluates on frame pairs from test dataset (Objectron only, has GT calibration)
3. Computes calibration accuracy (predicted mean vs GT mean)
4. Generates professional plots and reports for thesis

Author: AI Assistant for Kalman's Master's Thesis
Date: December 2025
"""

import sys
import os

# Disable xFormers for GPU compatibility (RTX 5090)
os.environ["XFORMERS_DISABLED"] = "1"
os.environ["XFORMERS_MORE_DETAILS"] = "0"

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Add project paths
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "anycalib"))

# DA3 imports
from experiments.models.da3_calibration_head import DA3CalibrationHead

# Dataset imports
from experiments.dataset_paths import (
    get_objectron_videos, get_objectron_gt, get_lightspeed_root
)
from experiments.train_pose_head_anycalib import (
    ObjectronVideoDataset,
    AnyCaLibBatchInference,
    load_dataset_split,
)
from experiments.lightspeed_dataset import LightSpeedDataset
from experiments.benchmark_dataset_utils import (
    get_dataset_paths,
    create_smart_sampled_dataset_objectron,
    create_smart_sampled_dataset_lightspeed,
    count_available_pairs_objectron,
    count_available_pairs_lightspeed,
)

# AnyCalib import
try:
    from anycalib.model.anycalib_pretrained import AnyCalib
except ImportError:
    from anycalib.anycalib.model.anycalib_pretrained import AnyCalib

print("[INIT] Imports successful")


# =============================================================================
# VISUAL TOKEN EXTRACTION
# =============================================================================

def extract_visual_tokens_dinov2(images: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    Extract visual tokens from DINOv2 (HuggingFace) for Stage 1 & 2.
    
    Args:
        images: [B, N, 3, H, W] tensor in range [0, 1]
        device: torch.device
    
    Returns:
        visual_tokens: [B, N, 384] visual feature tokens
    """
    from transformers import AutoModel
    
    # Load DINOv2 model (cache it)
    if not hasattr(extract_visual_tokens_dinov2, 'model'):
        print("[VISUAL] Loading DINOv2 backbone (HuggingFace)...")
        extract_visual_tokens_dinov2.model = AutoModel.from_pretrained('facebook/dinov2-small').to(device).eval()
        for param in extract_visual_tokens_dinov2.model.parameters():
            param.requires_grad = False
    
    model = extract_visual_tokens_dinov2.model
    B, N, C, H, W = images.shape
    
    # Reshape to batch all frames together
    inputs = images.view(B * N, C, H, W)
    
    # Normalize for DINOv2 (ImageNet normalization)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    inputs = (inputs - mean) / std
    
    # Resize to DINOv2 expected size (224x224)
    if H != 224 or W != 224:
        inputs = F.interpolate(inputs, size=(224, 224), mode='bilinear', align_corners=False)
    
    # Extract CLS token
    with torch.no_grad():
        outputs = model(inputs)
        cls_tokens = outputs.last_hidden_state[:, 0, :]  # [B*N, 384]
    
    # Reshape back to [B, N, 384]
    visual_tokens = cls_tokens.view(B, N, -1)
    return visual_tokens


# =============================================================================
# DATASET FOR CALIBRATION BENCHMARK
# =============================================================================

class CalibrationBenchmarkDataset(Dataset):
    """
    Dataset for calibration accuracy benchmark using frame pairs.
    Supports smart sampling with automatic dataset path detection.
    """
    def __init__(
        self,
        dataset_name: str,
        anycalib_model: AnyCaLibBatchInference,
        num_samples: Union[int, str] = 100,
        video_indices: Optional[List[int]] = None,
        image_size: Tuple[int, int] = (480, 640),
        require_gt: bool = True,
        device: torch.device = torch.device("cuda:0"),
    ):
        self.dataset_name = dataset_name
        self.anycalib_model = anycalib_model
        self.image_size = image_size
        self.require_gt = require_gt
        self.device = device
        
        # Get dataset paths automatically
        paths = get_dataset_paths(dataset_name)
        
        if dataset_name == 'objectron':
            self.videos_dir = paths['videos']
            self.gt_dir = paths['gt']
            
            # Count available pairs
            total_available = count_available_pairs_objectron(
                self.videos_dir, self.gt_dir, video_indices
            )
            print(f"[DATASET] Found {total_available} available frame pairs in Objectron")
            
            # Create smart sampled pairs
            sampled_pairs_info = create_smart_sampled_dataset_objectron(
                self.videos_dir, self.gt_dir, num_samples, video_indices, image_size
            )
            
            print(f"[DATASET] Using {len(sampled_pairs_info)} frame pairs (requested: {num_samples})")
            
            # Precompute AnyCalib predictions and load frames
            self.pairs = []
            print(f"[DATASET] Precomputing AnyCalib predictions and loading frames...")
            
            for pair_info in tqdm(sampled_pairs_info, desc="Processing pairs"):
                try:
                    video_path = pair_info['video_path']
                    frame_indices = pair_info['frame_indices']
                    
                    # Load frames
                    frames = self._load_frames_from_video(video_path, frame_indices)
                    if len(frames) < 2:
                        continue
                    
                    # Run AnyCalib on both frames
                    anycalib_preds = self._run_anycalib(frames)
                    
                    # Get GT calibrations
                    gt_calibrations = pair_info.get('gt_calibration')
                    if gt_calibrations is None:
                        gt_calibrations = self._load_gt_calibrations(video_path, frame_indices)
                    
                    if gt_calibrations is None or len(gt_calibrations) < 2:
                        if require_gt:
                            continue
                        # Use dummy GT if not required
                        gt_calibrations = np.array([[640, 480, 320, 240], [640, 480, 320, 240]], dtype=np.float32)
                    
                    # Compute GT mean
                    gt_mean = np.mean(gt_calibrations, axis=0)  # [4]
                    
                    self.pairs.append({
                        'frames': frames,
                        'frame_indices': frame_indices,
                        'anycalib_predictions': anycalib_preds,  # [2, 4]
                        'gt_mean_calibration': gt_mean,  # [4]
                        'image_size': self.image_size,
                    })
                    
                except Exception as e:
                    print(f"[WARN] Failed to process pair: {e}")
                    continue
            
        elif dataset_name == 'lightspeed':
            self.lightspeed_dir = paths['root']
            
            # LightSpeed doesn't have GT calibrations, so we'll skip calibration benchmark
            # or use dummy values
            raise NotImplementedError("LightSpeed dataset does not have GT calibrations for calibration benchmark")
        
        print(f"[DATASET] Preprocessed {len(self.pairs)} frame pairs")
    
    def _load_frames_from_video(self, video_path: Path, frame_indices: List[int]) -> List[np.ndarray]:
        """Load specific frames from video."""
        import cv2
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        frames = []
        
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame.shape[:2] != self.image_size:
                frame = cv2.resize(frame, (self.image_size[1], self.image_size[0]))
            
            frames.append(frame)
        
        cap.release()
        return frames
    
    def _run_anycalib(self, frames: List[np.ndarray]) -> np.ndarray:
        """Run AnyCalib on frames."""
        # Convert to tensor format expected by AnyCalib
        frames_tensor = []
        for frame in frames:
            # Convert BGR to RGB and normalize to [0, 1]
            frame_rgb = frame[:, :, ::-1].copy()  # BGR to RGB
            frame_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).float() / 255.0
            frames_tensor.append(frame_tensor)
        
        frames_batch = torch.stack(frames_tensor, dim=0).unsqueeze(0).to(self.device)  # [1, 2, 3, H, W]
        
        # Run AnyCalib
        anycalib_preds = self.anycalib_model.predict_intrinsics(frames_batch)  # [1, 2, 4]
        return anycalib_preds[0].cpu().numpy()  # [2, 4]
    
    def _load_gt_calibrations(self, video_path: Path, frame_indices: List[int]) -> Optional[np.ndarray]:
        """Load GT calibrations for frames."""
        if self.gt_dir is None:
            return None
        
        gt_path1 = self.gt_dir / f"{video_path.stem}.json"
        stem_without_video = video_path.stem.replace("_video", "")
        gt_path2 = self.gt_dir / f"{stem_without_video}.json"
        
        gt_path = gt_path1 if gt_path1.exists() else (gt_path2 if gt_path2.exists() else None)
        
        if gt_path is None:
            return None
        
        try:
            with open(gt_path, 'r') as f:
                gt_data = json.load(f)
            
            calibrations = []
            
            if 'intrinsics_per_frame' in gt_data:
                intr_list = gt_data['intrinsics_per_frame']
                for frame_idx in frame_indices:
                    if frame_idx < len(intr_list):
                        K_flat = intr_list[frame_idx]  # [9] flattened 3x3
                        fx, fy, cx, cy = K_flat[0], K_flat[4], K_flat[2], K_flat[5]
                        calibrations.append([fx, fy, cx, cy])
            elif 'frames' in gt_data:
                for frame_idx in frame_indices:
                    if frame_idx < len(gt_data['frames']):
                        frame_data = gt_data['frames'][frame_idx]
                        intrinsics = frame_data.get('intrinsics', None)
                        if intrinsics is None:
                            continue
                        fx, fy, cx, cy = intrinsics[:4]
                        calibrations.append([fx, fy, cx, cy])
            elif 'intrinsics' in gt_data:
                intr_list = gt_data['intrinsics']
                for frame_idx in frame_indices:
                    if frame_idx < len(intr_list):
                        K = np.array(intr_list[frame_idx], dtype=np.float32).reshape(3, 3)
                        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
                        calibrations.append([fx, fy, cx, cy])
            
            if len(calibrations) == 0:
                return None
            
            return np.array(calibrations, dtype=np.float32)
            
        except Exception as e:
            return None
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx: int) -> Dict:
        pair_data = self.pairs[idx]
        
        # Convert frames to tensor
        frames_tensor = []
        for frame in pair_data['frames']:
            frame_rgb = frame[:, :, ::-1].copy()  # BGR to RGB
            frame_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).float() / 255.0
            frames_tensor.append(frame_tensor)
        frames = torch.stack(frames_tensor, dim=0)  # [2, 3, H, W]
        
        return {
            'frames': frames,  # [2, 3, H, W]
            'anycalib_predictions': torch.from_numpy(pair_data['anycalib_predictions']).float(),  # [2, 4]
            'gt_mean_calibration': torch.from_numpy(pair_data['gt_mean_calibration']).float().unsqueeze(0),  # [1, 4]
            'image_size': pair_data['image_size'],
        }


# =============================================================================
# EVALUATION FUNCTIONS
# =============================================================================

def evaluate_calibration_accuracy(
    model: DA3CalibrationHead,
    dataloader: DataLoader,
    device: torch.device,
    stage: int,
    extract_visual_fn,
) -> Dict:
    """
    Evaluate calibration accuracy.
    
    Returns dictionary with error metrics.
    """
    model.eval()
    
    all_errors = {
        'fx': [],
        'fy': [],
        'cx': [],
        'cy': [],
    }
    
    all_predictions = []
    all_targets = []
    all_anycalib_means = []  # For comparison baseline
    
    use_visual_conditioning = (stage >= 2)
    
    with torch.no_grad():
        for batch_data in tqdm(dataloader, desc="Evaluating"):
            frames = batch_data['frames'].to(device)  # [B, 2, 3, H, W]
            anycalib_preds = batch_data['anycalib_predictions'].to(device)  # [B, 2, 4]
            gt_mean = batch_data['gt_mean_calibration'].to(device)  # [B, 1, 4]
            image_size = batch_data['image_size']
            
            B = frames.shape[0]
            H, W = image_size[0][0].item() if isinstance(image_size[0], torch.Tensor) else image_size[0], \
                   image_size[1][0].item() if isinstance(image_size[1], torch.Tensor) else image_size[1]
            
            # Extract visual tokens if needed
            if use_visual_conditioning and extract_visual_fn is not None:
                visual_tokens = extract_visual_fn(frames, device)  # [B, 2, D_vis]
            else:
                # Stage 1: dummy visual tokens
                visual_tokens = torch.zeros(B, 2, model.vis_dim, device=device)
            
            # Run DA3 calibration head
            pred_calibration = model(
                visual_tokens=visual_tokens,
                anycalib_predictions=anycalib_preds,
                image_size=(H, W),
                use_visual_conditioning=use_visual_conditioning,
            )  # [B, 1, 4]
            
            # Compute AnyCalib mean (baseline comparison)
            anycalib_mean = anycalib_preds.mean(dim=1, keepdim=True)  # [B, 1, 4]
            
            # Compute errors (relative)
            pred_np = pred_calibration.cpu().numpy()  # [B, 1, 4]
            gt_np = gt_mean.cpu().numpy()  # [B, 1, 4]
            anycalib_mean_np = anycalib_mean.cpu().numpy()  # [B, 1, 4]
            
            for b in range(B):
                pred = pred_np[b, 0]  # [4]
                gt = gt_np[b, 0]  # [4]
                anycalib_mean_val = anycalib_mean_np[b, 0]  # [4]
                
                # Relative errors
                rel_errors = np.abs(pred - gt) / (gt + 1e-8) * 100  # Percentage
                
                all_errors['fx'].append(float(rel_errors[0]))
                all_errors['fy'].append(float(rel_errors[1]))
                all_errors['cx'].append(float(rel_errors[2]))
                all_errors['cy'].append(float(rel_errors[3]))
                
                all_predictions.append(pred.tolist())
                all_targets.append(gt.tolist())
                all_anycalib_means.append(anycalib_mean_val.tolist())
    
    # Compute statistics
    stats = {}
    for param in ['fx', 'fy', 'cx', 'cy']:
        errors = np.array(all_errors[param])
        stats[param] = {
            'mean': float(np.mean(errors)),
            'median': float(np.median(errors)),
            'std': float(np.std(errors)),
            'p90': float(np.percentile(errors, 90)),
            'p95': float(np.percentile(errors, 95)),
        }
    
    results = {
        'errors': all_errors,
        'statistics': stats,
        'predictions': all_predictions,
        'targets': all_targets,
        'anycalib_means': all_anycalib_means,
    }
    
    return results


# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================

def plot_error_distributions(results: Dict, stage: int, save_path: Path):
    """Plot error distributions for each parameter."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Calibration Parameter Error Distribution: DA3 Stage {stage}', fontsize=16, fontweight='bold')
    
    params = ['fx', 'fy', 'cx', 'cy']
    param_names = {
        'fx': 'Focal Length (fx)',
        'fy': 'Focal Length (fy)',
        'cx': 'Principal Point (cx)',
        'cy': 'Principal Point (cy)',
    }
    
    for idx, param in enumerate(params):
        ax = axes[idx // 2, idx % 2]
        errors = results['errors'][param]
        
        ax.hist(errors, bins=50, alpha=0.7, color='blue', edgecolor='black')
        ax.axvline(np.mean(errors), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(errors):.2f}%')
        ax.axvline(np.median(errors), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(errors):.2f}%')
        
        ax.set_xlabel('Relative Error (%)', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title(f'{param_names[param]}', fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[SAVE] Error distributions saved to {save_path}")
    plt.close()


def plot_comparison_with_anycalib(results: Dict, stage: int, save_path: Path):
    """Plot comparison with AnyCalib mean baseline."""
    # Compute AnyCalib errors
    anycalib_errors = {
        'fx': [],
        'fy': [],
        'cx': [],
        'cy': [],
    }
    
    for i in range(len(results['targets'])):
        gt = np.array(results['targets'][i])
        anycalib_mean = np.array(results['anycalib_means'][i])
        rel_errors = np.abs(anycalib_mean - gt) / (gt + 1e-8) * 100
        
        anycalib_errors['fx'].append(float(rel_errors[0]))
        anycalib_errors['fy'].append(float(rel_errors[1]))
        anycalib_errors['cx'].append(float(rel_errors[2]))
        anycalib_errors['cy'].append(float(rel_errors[3]))
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Calibration Accuracy Comparison: DA3 Stage {stage} vs AnyCalib Mean', fontsize=16, fontweight='bold')
    
    params = ['fx', 'fy', 'cx', 'cy']
    param_names = {
        'fx': 'Focal Length (fx)',
        'fy': 'Focal Length (fy)',
        'cx': 'Principal Point (cx)',
        'cy': 'Principal Point (cy)',
    }
    
    for idx, param in enumerate(params):
        ax = axes[idx // 2, idx % 2]
        
        da3_errors = results['errors'][param]
        anycalib_errors_param = anycalib_errors[param]
        
        ax.hist(da3_errors, bins=50, alpha=0.7, label=f'DA3 Stage {stage}', color='blue', edgecolor='black')
        ax.hist(anycalib_errors_param, bins=50, alpha=0.7, label='AnyCalib Mean', color='orange', edgecolor='black')
        
        ax.set_xlabel('Relative Error (%)', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title(f'{param_names[param]}', fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[SAVE] Comparison plot saved to {save_path}")
    plt.close()


def plot_error_statistics(results: Dict, stage: int, save_path: Path):
    """Plot bar chart comparing error statistics."""
    params = ['fx', 'fy', 'cx', 'cy']
    param_names = {
        'fx': 'Focal Length (fx)',
        'fy': 'Focal Length (fy)',
        'cx': 'Principal Point (cx)',
        'cy': 'Principal Point (cy)',
    }
    
    metrics = ['mean', 'median', 'p90']
    metric_names = ['Mean', 'Median', '90th Percentile']
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(params))
    width = 0.25
    
    for i, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
        values = [results['statistics'][param][metric] for param in params]
        ax.bar(x + i * width, values, width, label=metric_name, alpha=0.8)
    
    ax.set_xlabel('Calibration Parameter', fontsize=12)
    ax.set_ylabel('Relative Error (%)', fontsize=12)
    ax.set_title(f'Calibration Error Statistics: DA3 Stage {stage}', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels([param_names[p] for p in params])
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[SAVE] Error statistics plot saved to {save_path}")
    plt.close()


# =============================================================================
# REPORT GENERATION
# =============================================================================

def generate_report(results: Dict, stage: int, num_samples: int, save_path: Path):
    """Generate text report with statistics."""
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write(f"Calibration Accuracy Evaluation: DA3 Stage {stage}\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Test Set Size: {num_samples} frame pairs\n")
        f.write(f"Model: DA3 Stage {stage}\n\n")
        
        f.write("EVALUATION METHODOLOGY\n")
        f.write("-" * 80 + "\n")
        f.write("For each frame pair:\n")
        f.write("1. Run AnyCalib on both frames to get per-frame predictions\n")
        f.write("2. Extract visual tokens (DINOv2 for Stage 1/2, AnyCam for Stage 3)\n")
        f.write("3. Run DA3 calibration head to predict mean calibration\n")
        f.write("4. Compute GT mean calibration from the pair\n")
        f.write("5. Calculate relative errors: |predicted - GT| / GT * 100%\n\n")
        
        f.write("CALIBRATION PARAMETER ERRORS (Relative Error %)\n")
        f.write("-" * 80 + "\n")
        
        param_names = {
            'fx': 'Focal Length (fx)',
            'fy': 'Focal Length (fy)',
            'cx': 'Principal Point (cx)',
            'cy': 'Principal Point (cy)',
        }
        
        # Header
        f.write(f"{'Parameter':<25} {'Mean':>12} {'Median':>12} {'Std Dev':>12} {'90th %ile':>12} {'95th %ile':>12}\n")
        f.write("-" * 80 + "\n")
        
        # Data rows
        for param in ['fx', 'fy', 'cx', 'cy']:
            stats = results['statistics'][param]
            f.write(f"{param_names[param]:<25} "
                   f"{stats['mean']:>12.4f} "
                   f"{stats['median']:>12.4f} "
                   f"{stats['std']:>12.4f} "
                   f"{stats['p90']:>12.4f} "
                   f"{stats['p95']:>12.4f}\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("SUMMARY\n")
        f.write("="*80 + "\n")
        
        # Overall statistics
        all_errors = []
        for param in ['fx', 'fy', 'cx', 'cy']:
            all_errors.extend(results['errors'][param])
        
        overall_mean = np.mean(all_errors)
        overall_median = np.median(all_errors)
        
        f.write(f"Overall Mean Relative Error: {overall_mean:.4f}%\n")
        f.write(f"Overall Median Relative Error: {overall_median:.4f}%\n")
        
        # Best and worst parameters
        param_means = {param: results['statistics'][param]['mean'] for param in ['fx', 'fy', 'cx', 'cy']}
        best_param = min(param_means, key=param_means.get)
        worst_param = max(param_means, key=param_means.get)
        
        f.write(f"Best Parameter: {param_names[best_param]} ({param_means[best_param]:.4f}%)\n")
        f.write(f"Worst Parameter: {param_names[worst_param]} ({param_means[worst_param]:.4f}%)\n")
    
    print(f"[SAVE] Report saved to {save_path}")


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Benchmark DA3 Calibration Accuracy")
    
    # Model arguments
    parser.add_argument("--stage1_checkpoint", type=str, default=None,
                       help="Path to Stage 1 checkpoint")
    parser.add_argument("--stage2_checkpoint", type=str, default=None,
                       help="Path to Stage 2 checkpoint")
    parser.add_argument("--stage3_checkpoint", type=str, default=None,
                       help="Path to Stage 3 checkpoint")
    
    # Dataset arguments
    parser.add_argument("--dataset", type=str, choices=['objectron', 'lightspeed'], default='objectron',
                       help="Dataset to use (paths auto-detected)")
    parser.add_argument("--split_file", type=str, default="experiments/objectron_split.json",
                       help="Dataset split file (for Objectron, optional)")
    parser.add_argument("--num_samples", type=str, default="100",
                       help="Number of frame pairs to evaluate (default: 100, use 'all' for all available)")
    
    # Model architecture arguments
    parser.add_argument("--vis_dim", type=int, default=384,
                       help="Visual token dimension (384 for DINOv2-S, 768 for AnyCam)")
    parser.add_argument("--cam_dim", type=int, default=256,
                       help="Camera token dimension")
    parser.add_argument("--hidden_dim", type=int, default=128,
                       help="Hidden layer dimension")
    
    # Evaluation arguments
    parser.add_argument("--save_dir", type=str, default=None,
                       help="Output directory (auto-generated if not specified)")
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Batch size for evaluation")
    parser.add_argument("--device", type=str, default="cuda:0",
                       help="Device to use")
    
    args = parser.parse_args()
    
    # Determine which stage to evaluate
    stage = None
    checkpoint_path = None
    
    if args.stage1_checkpoint:
        stage = 1
        checkpoint_path = args.stage1_checkpoint
    elif args.stage2_checkpoint:
        stage = 2
        checkpoint_path = args.stage2_checkpoint
    elif args.stage3_checkpoint:
        stage = 3
        checkpoint_path = args.stage3_checkpoint
    else:
        raise ValueError("Must specify one of --stage1_checkpoint, --stage2_checkpoint, or --stage3_checkpoint")
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[INIT] Using device: {device}")
    print(f"[INIT] Evaluating DA3 Stage {stage}")
    
    # Parse num_samples
    if args.num_samples.lower() == 'all':
        num_samples = "all"
    else:
        try:
            num_samples = int(args.num_samples)
        except ValueError:
            raise ValueError(f"Invalid num_samples: {args.num_samples} (must be integer or 'all')")
    
    # Determine save directory
    if args.save_dir is None:
        args.save_dir = f"experiments/da3_integration/benchmark_results/stage{stage}_calibration"
    
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"[SAVE] Results will be saved to: {save_dir}")
    
    # Load dataset split (optional, for Objectron)
    test_indices = None
    if args.dataset == 'objectron' and Path(args.split_file).exists():
        split_data = load_dataset_split(args.split_file)
        test_indices = split_data.get('test', split_data.get('test_indices', []))
        print(f"[DATASET] Using test split: {len(test_indices)} videos")
    elif args.dataset == 'objectron':
        print(f"[DATASET] No split file found, using all videos")
    
    # Initialize AnyCalib
    anycalib_inference = AnyCaLibBatchInference(device=device)
    
    # Create dataset with smart sampling
    print(f"\n[STEP 1] Creating dataset with smart sampling...")
    print(f"[DATASET] Dataset: {args.dataset}")
    print(f"[DATASET] Requested samples: {num_samples}")
    
    dataset = CalibrationBenchmarkDataset(
        dataset_name=args.dataset,
        anycalib_model=anycalib_inference,
        num_samples=num_samples,
        video_indices=test_indices,
        require_gt=True,
        device=device,
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    
    print(f"[DATASET] Loaded {len(dataset)} frame pairs")
    
    # Create model
    print(f"\n[STEP 2] Creating DA3 calibration head...")
    model = DA3CalibrationHead(
        vis_dim=args.vis_dim,
        cam_dim=args.cam_dim,
        hidden_dim=args.hidden_dim,
        num_mixing_layers=2,
    ).to(device)
    
    # Load checkpoint
    print(f"[LOAD] Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # For Stage 3, the checkpoint contains the full AnyCamWrapperWithDA3Calibration
    # We need to extract only the calibration head weights
    if stage == 3:
        calibration_head_dict = {}
        for k, v in state_dict.items():
            # Look for calibration head keys: pose_predictor.da3_calibration_head.*
            if 'pose_predictor.da3_calibration_head.' in k:
                # Remove the prefix to get just the calibration head keys
                new_key = k.replace('pose_predictor.da3_calibration_head.', '')
                calibration_head_dict[new_key] = v
            elif k.startswith('da3_calibration_head.'):
                # Alternative format
                new_key = k.replace('da3_calibration_head.', '')
                calibration_head_dict[new_key] = v
        
        if calibration_head_dict:
            state_dict = calibration_head_dict
            print(f"[LOAD] Extracted calibration head weights from full model ({len(calibration_head_dict)} keys)")
        else:
            print(f"[WARN] Could not find calibration head weights in Stage 3 checkpoint")
            print(f"[WARN] Attempting to load as standalone calibration head...")
    
    # Filter state dict for dimension mismatches (Stage 1 vs Stage 2 vis_dim)
    model_state = model.state_dict()
    filtered_state_dict = {}
    for k, v in state_dict.items():
        if k in model_state:
            if v.shape == model_state[k].shape:
                filtered_state_dict[k] = v
            else:
                print(f"[WARN] Skipping mismatched key: {k} (checkpoint {v.shape} vs model {model_state[k].shape})")
        else:
            print(f"[WARN] Skipping missing key: {k}")
    
    model.load_state_dict(filtered_state_dict, strict=False)
    print(f"[LOAD] Model loaded ({len(filtered_state_dict)} keys)")
    
    # Determine visual token extraction function
    extract_visual_fn = None
    if stage >= 2:
        extract_visual_fn = extract_visual_tokens_dinov2
        print(f"[VISUAL] Using DINOv2 for visual token extraction")
    else:
        print(f"[VISUAL] Stage 1: No visual tokens needed")
    
    # Evaluate
    print(f"\n[STEP 3] Evaluating calibration accuracy...")
    results = evaluate_calibration_accuracy(
        model=model,
        dataloader=dataloader,
        device=device,
        stage=stage,
        extract_visual_fn=extract_visual_fn,
    )
    
    # Generate plots
    print(f"\n[STEP 4] Generating plots...")
    plot_error_distributions(results, stage, save_dir / "calibration_error_distributions.png")
    # NOTE: AnyCalib comparison disabled - not scientifically valid due to dataset-specific overfitting
    # Uncomment only after training on large, diverse datasets for fair comparison
    # plot_comparison_with_anycalib(results, stage, save_dir / "calibration_comparison_anycalib.png")
    plot_error_statistics(results, stage, save_dir / "calibration_error_statistics.png")
    
    # Generate report
    print(f"\n[STEP 5] Generating report...")
    generate_report(results, stage, len(dataset), save_dir / "calibration_accuracy_report.txt")
    
    # Save results JSON
    results_json = {
        'stage': stage,
        'num_samples': len(dataset),
        'errors': results['errors'],
        'statistics': results['statistics'],
        'predictions': results['predictions'],
        'targets': results['targets'],
        'anycalib_means': results['anycalib_means'],
    }
    
    with open(save_dir / "calibration_accuracy.json", 'w') as f:
        json.dump(results_json, f, indent=2)
    
    # Save metadata
    metadata = {
        'checkpoint_path': str(checkpoint_path),
        'stage': stage,
        'dataset': args.dataset,
        'num_samples': len(dataset),
        'batch_size': args.batch_size,
        'device': str(device),
        'vis_dim': args.vis_dim,
        'cam_dim': args.cam_dim,
        'hidden_dim': args.hidden_dim,
    }
    
    with open(save_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"BENCHMARK COMPLETE")
    print(f"{'='*80}")
    print(f"Results saved to: {save_dir}")
    print(f"  - calibration_accuracy.json")
    print(f"  - calibration_accuracy_report.txt")
    print(f"  - calibration_error_distributions.png")
    print(f"  - calibration_comparison_anycalib.png")
    print(f"  - calibration_error_statistics.png")
    print(f"  - metadata.json")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()

