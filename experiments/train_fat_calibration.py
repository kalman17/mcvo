#!/usr/bin/env python3
"""
=============================================================================
Feature Aggregation Transformer (FAT) Unified Training Script
=============================================================================

This script trains the FAT architecture that inserts a transformer between
AnyCalib's Step 1 (DINOv2 features) and Step 2 (DPT decoder).

TRAINING PHASES:
----------------
Phase 1: Feature aggregation pre-training (no visual tokens)
Phase 2: Visual-conditioned aggregation (with DINOv2-small CLS tokens)
Phase 3: End-to-end with flow reprojection loss (self-supervised)

USAGE:
------
# Phase 1
python experiments/train_fat_calibration.py --phase 1 --objectron_videos /path/to/videos \\
    --objectron_gt /path/to/gt --max_ahead 3 --num_epochs 50 \\
    --save_dir experiments/final_training_phases/phase1_training

# Phase 2
python experiments/train_fat_calibration.py --phase 2 --phase1_checkpoint <path> \\
    --objectron_videos /path/to/videos --objectron_gt /path/to/gt \\
    --use_visual_conditioning --max_ahead 3 --num_epochs 50 \\
    --save_dir experiments/final_training_phases/phase2_training

# Phase 3
python experiments/train_fat_calibration.py --phase 3 --phase2_checkpoint <path> \\
    --objectron_videos /path/to/videos --max_ahead 3 \\
    --benchmark_samples 100 --benchmark_no_cycle --num_epochs 50 \\
    --save_dir experiments/final_training_phases/phase3_training

Author: AI Assistant for Kalman's Master's Thesis
Date: January 2026
"""

import sys
import os
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import numpy as np
import cv2

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast
from tqdm import tqdm
import yaml
from pathlib import Path

# Add project paths
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "anycalib"))

# Disable xFormers for GPU compatibility
os.environ["XFORMERS_DISABLED"] = "1"


# =============================================================================
# ARGUMENT PARSING
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Feature Aggregation Transformer (FAT) Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Phase selection
    parser.add_argument("--phase", type=int, required=True, choices=[1, 2, 3],
                        help="Training phase (1, 2, or 3)")
    parser.add_argument("--v2", action="store_true",
                        help="Use V2 training with differentiable calibrator (implicit differentiation)")
    parser.add_argument("--regularization_lambda", type=float, default=0.1,
                        help="Regularization weight for V2 training (anchor to RANSAC)")

    # Dataset
    parser.add_argument("--objectron_videos", type=str, required=True,
                        help="Path to Objectron videos directory")
    parser.add_argument("--objectron_gt", type=str, default=None,
                        help="Path to Objectron GT directory (optional for phase 3)")
    parser.add_argument("--split_file", type=str,
                        default="experiments/objectron_split.json",
                        help="Path to dataset split file")

    # Training
    parser.add_argument("--num_epochs", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Batch size (auto-selected if not specified)")
    parser.add_argument("--learning_rate", type=float, default=None,
                        help="Learning rate (auto-selected if not specified)")
    parser.add_argument("--num_workers", type=int, default=0,
                        help="Number of DataLoader workers (default: 0 for Docker safety)")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device to use")

    # FAT architecture
    parser.add_argument("--embed_dim", type=int, default=1024,
                        help="DINOv2 feature dimension")
    parser.add_argument("--num_heads", type=int, default=8,
                        help="Number of attention heads")
    parser.add_argument("--num_layers", type=int, default=2,
                        help="Number of transformer layers")
    parser.add_argument("--use_learnable_agg_token", action="store_true",
                        help="Use learnable token for aggregation (default: mean pooling)")

    # Visual conditioning
    parser.add_argument("--use_visual_conditioning", action="store_true",
                        help="Enable visual token conditioning")
    parser.add_argument("--visual_token_dim", type=int, default=384,
                        help="Visual token dimension (DINOv2-small)")

    # DINOv2 variant
    parser.add_argument("--use_dinov2_small", action="store_true", default=True,
                        help="Use DINOv2-small for visual tokens (local dev)")
    parser.add_argument("--use_dinov2_full", action="store_true",
                        help="Use torch.hub DINOv2 for visual tokens (cluster)")

    # Multi-frame settings
    parser.add_argument("--max_ahead", type=int, default=3,
                        help="Number of frames ahead (total frames = max_ahead + 1)")

    # Benchmark settings (phase 3)
    parser.add_argument("--benchmark_samples", type=int, default=100,
                        help="Number of samples for pose benchmark per epoch")
    parser.add_argument("--benchmark_calibration_samples", type=int, default=50,
                        help="Number of samples for calibration benchmark per epoch (phase 3)")
    parser.add_argument("--benchmark_pose_samples", type=int, default=50,
                        help="Number of samples for pose benchmarking per epoch")
    parser.add_argument("--benchmark_no_cycle", action="store_true",
                        help="Use fixed benchmark samples (no cycling)")
    parser.add_argument("--disable_benchmark", action="store_true",
                        help="Disable pose benchmarking")
    
    # Phase 3 specific
    parser.add_argument("--enable_alternating_training", action="store_true",
                        help="Alternate between training FAT and Pose Head each epoch")
    parser.add_argument("--anycam_config", type=str,
                        default="pretrained_models/anycam_seq8/training_config.yaml",
                        help="Path to AnyCam config file (for Phase 3)")

    # Checkpoint handling
    parser.add_argument("--phase1_checkpoint", type=str, default=None,
                        help="Path to phase 1 checkpoint (for phase 2)")
    parser.add_argument("--phase2_checkpoint", type=str, default=None,
                        help="Path to phase 2 checkpoint (for phase 3)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--baseline_checkpoint", type=str,
                        default="pretrained_models/anycam_seq8/training_checkpoint_247500.pt",
                        help="Path to AnyCam baseline for comparison")

    # Phase 3 V2: Combined loss training (skip Phase 2, use Phase 1 directly)
    parser.add_argument("--phase1_checkpoint_for_phase3", type=str, default=None,
                        help="Path to phase 1 checkpoint for phase 3 (skip phase 2, no visual tokens)")
    parser.add_argument("--lambda_calib", type=float, default=1e-4,
                        help="Weight for calibration reprojection loss (1e-4 to 0.15, tune empirically)")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help="Weight decay for AdamW optimizer (L2 regularization)")

    # Output
    parser.add_argument("--save_dir", type=str, required=True,
                        help="Directory to save results")

    return parser.parse_args()


# =============================================================================
# CONFIGURATION
# =============================================================================

PHASE_CONFIG = {
    1: {
        'batch_size': 4,
        'learning_rate': 5e-5,  # Reduced from 1e-4 for stability
        'use_visual_conditioning': False,
        'freeze_backbone': True,
        'freeze_decoder': True,
    },
    2: {
        'batch_size': 4,
        'learning_rate': 5e-5,
        'use_visual_conditioning': True,
        'freeze_backbone': True,
        'freeze_decoder': True,
    },
    3: {
        'batch_size': 2,
        'learning_rate': 1e-5,
        'use_visual_conditioning': False,  # Phase 3 V2: Skip visual tokens, use Phase 1 directly
        'freeze_backbone': True,
        'freeze_decoder': True,
    },
}


# =============================================================================
# DATASET
# =============================================================================

class ObjectronFATDataset(Dataset):
    """
    Dataset for FAT training that loads multi-frame sequences from Objectron.

    For phases 1 and 2: Uses MSE loss against GT mean calibration
    For phase 3: Uses flow reprojection loss (self-supervised)
    """

    def __init__(
        self,
        video_dir: Path,
        gt_dir: Optional[Path],
        video_indices: List[int],
        max_ahead: int = 3,
        phase: int = 1,
        image_size: Tuple[int, int] = (480, 640),  # (H, W)
    ):
        self.video_dir = Path(video_dir)
        self.gt_dir = Path(gt_dir) if gt_dir else None
        self.max_ahead = max_ahead
        self.num_frames = max_ahead + 1
        self.phase = phase
        self.image_size = image_size

        # Get video files (Objectron uses .MOV/.mov)
        all_videos = sorted(list(self.video_dir.glob("*.MOV")) +
                           list(self.video_dir.glob("*.mov")))
        self.video_files = [all_videos[i] for i in video_indices if i < len(all_videos)]

        print(f"[DATASET] Found {len(self.video_files)} videos for indices {len(video_indices)}")

        # Build sequence index
        self._build_sequence_index()

    def _build_sequence_index(self):
        """Build index of frame sequences."""
        self.sequences = []

        for video_idx, video_path in enumerate(self.video_files):
            cap = cv2.VideoCapture(str(video_path))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

            # Safety margin
            safe_total = max(0, total_frames - 2)

            if safe_total < self.num_frames:
                print(f"[WARN] Video {video_path.name} has only {safe_total} frames, skipping")
                continue

            # Create sequences with step size 2 (use half the frames)
            for start in range(0, safe_total - self.num_frames + 1, 2):
                self.sequences.append({
                    'video_idx': video_idx,
                    'video_path': video_path,
                    'start_frame': start,
                    'frame_indices': list(range(start, start + self.num_frames)),
                })

        print(f"[DATASET] Built {len(self.sequences)} sequences from {len(self.video_files)} videos")

    def __len__(self):
        return len(self.sequences)

    def _load_frames(self, video_path: Path, frame_indices: List[int]) -> np.ndarray:
        """Load specific frames from video."""
        cap = cv2.VideoCapture(str(video_path))
        frames = []

        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Resize to consistent size
                if frame.shape[:2] != self.image_size:
                    frame = cv2.resize(frame, (self.image_size[1], self.image_size[0]))

                frames.append(frame)
            else:
                # Duplicate last frame if read fails
                if frames:
                    frames.append(frames[-1].copy())
                else:
                    raise RuntimeError(f"Failed to read frame {idx} from {video_path}")

        cap.release()
        return np.stack(frames)

    def _load_gt_intrinsics(self, video_path: Path) -> Optional[np.ndarray]:
        """Load GT intrinsics from JSON file."""
        if self.gt_dir is None:
            return None

        # Try different naming patterns
        stem = video_path.stem
        patterns = [
            self.gt_dir / f"{stem}.json",
            self.gt_dir / f"{stem.replace('_video', '')}.json",
        ]

        for gt_path in patterns:
            if gt_path.exists():
                try:
                    with open(gt_path, 'r') as f:
                        data = json.load(f)

                    # Format 1: 'frames' with 'intrinsics' dict
                    if 'frames' in data and len(data['frames']) > 0:
                        frame = data['frames'][0]
                        if 'intrinsics' in frame:
                            intr = frame['intrinsics']
                            if isinstance(intr, dict):
                                return np.array([
                                    intr['fx'], intr['fy'],
                                    intr['cx'], intr['cy']
                                ], dtype=np.float32)
                            elif isinstance(intr, (list, tuple)) and len(intr) >= 4:
                                return np.array(intr[:4], dtype=np.float32)

                    # Format 2: 'intrinsics_per_frame' (Objectron processed format)
                    # Flattened 3x3 matrices [fx, 0, cx, 0, fy, cy, 0, 0, 1]
                    if 'intrinsics_per_frame' in data and len(data['intrinsics_per_frame']) > 0:
                        K_flat = data['intrinsics_per_frame'][0]  # First frame
                        # Extract from flattened 3x3: [fx, 0, cx, 0, fy, cy, 0, 0, 1]
                        fx = K_flat[0]
                        fy = K_flat[4]
                        cx = K_flat[2]
                        cy = K_flat[5]
                        return np.array([fx, fy, cx, cy], dtype=np.float32)

                except Exception as e:
                    print(f"[WARN] Failed to load GT from {gt_path}: {e}")

        return None

    def __getitem__(self, idx: int) -> Dict:
        seq = self.sequences[idx]
        video_path = seq['video_path']
        frame_indices = seq['frame_indices']

        # Load frames
        frames = self._load_frames(video_path, frame_indices)

        # Convert to tensor [N, 3, H, W] in [0, 1]
        imgs = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0

        result = {
            'imgs': imgs,
            'video_idx': seq['video_idx'],
            'frame_indices': torch.tensor(frame_indices),
        }

        # Load GT intrinsics for phases 1 and 2
        if self.phase in [1, 2]:
            gt_intr = self._load_gt_intrinsics(video_path)
            if gt_intr is not None:
                result['gt_intrinsics'] = torch.from_numpy(gt_intr)
        
        # Load GT poses for phase 3 (for benchmarking)
        if self.phase == 3 and self.gt_dir:
            gt_poses = self._load_gt_poses(video_path, frame_indices)
            if gt_poses is not None:
                result['gt_poses'] = gt_poses
                result['poses'] = gt_poses  # Also add as 'poses' for compatibility

        return result
    
    def _load_gt_poses(self, video_path: Path, frame_indices: List[int]) -> Optional[torch.Tensor]:
        """Load GT poses from JSON file for specified frame indices."""
        if self.gt_dir is None:
            return None
        
        # Try different naming patterns
        stem = video_path.stem
        patterns = [
            self.gt_dir / f"{stem}.json",
            self.gt_dir / f"{stem.replace('_video', '')}.json",
        ]
        
        for gt_path in patterns:
            if gt_path.exists():
                try:
                    with open(gt_path, 'r') as f:
                        data = json.load(f)
                    
                    # Extract poses from GT data
                    # Objectron format: 'frames' with 'camera_pose' or 'pose'
                    if 'frames' in data:
                        poses_list = []
                        for idx in frame_indices:
                            if idx < len(data['frames']):
                                frame = data['frames'][idx]
                                # Try different pose field names
                                pose_data = frame.get('camera_pose') or frame.get('pose') or frame.get('transform')
                                if pose_data:
                                    # Convert to 4x4 matrix
                                    if isinstance(pose_data, list):
                                        pose_matrix = np.array(pose_data).reshape(4, 4)
                                    elif isinstance(pose_data, dict):
                                        # Extract rotation and translation
                                        rot = pose_data.get('rotation', np.eye(3))
                                        trans = pose_data.get('translation', [0, 0, 0])
                                        pose_matrix = np.eye(4)
                                        pose_matrix[:3, :3] = np.array(rot).reshape(3, 3)
                                        pose_matrix[:3, 3] = np.array(trans)[:3]
                                    else:
                                        continue
                                    poses_list.append(pose_matrix)
                                else:
                                    # Use identity if pose not found
                                    poses_list.append(np.eye(4))
                            else:
                                # Use identity if frame index out of range
                                poses_list.append(np.eye(4))
                        
                        if poses_list:
                            return torch.from_numpy(np.stack(poses_list)).float()
                
                except Exception as e:
                    print(f"[WARN] Failed to load GT poses from {gt_path}: {e}")
        
        return None


def collate_fn(batch: List[Dict]) -> Dict:
    """Collate function for variable-size batches."""
    result = {}

    # Stack images: [B, N, 3, H, W]
    imgs = torch.stack([b['imgs'] for b in batch])
    result['imgs'] = imgs

    # Optional GT intrinsics
    if 'gt_intrinsics' in batch[0]:
        gt_intr = torch.stack([b['gt_intrinsics'] for b in batch])
        result['gt_intrinsics'] = gt_intr

    # Other fields
    result['video_idx'] = [b['video_idx'] for b in batch]
    result['frame_indices'] = torch.stack([b['frame_indices'] for b in batch])

    return result


def load_dataset_split(split_file: str) -> Dict:
    """Load or create dataset split."""
    split_path = Path(split_file)

    if split_path.exists():
        with open(split_path, 'r') as f:
            return json.load(f)

    # Create default split (70 train, 15 val, 15 test)
    np.random.seed(42)
    indices = np.random.permutation(100).tolist()

    split = {
        'train': indices[:70],
        'val': indices[70:85],
        'test': indices[85:],
    }

    with open(split_path, 'w') as f:
        json.dump(split, f, indent=2)

    return split


# =============================================================================
# MODEL UTILITIES
# =============================================================================

def create_fat_model(args, phase: int, device: str):
    """Create AnyCalibWithFAT model with appropriate configuration."""
    from experiments.models.anycalib_with_fat import AnyCalibWithFAT

    phase_cfg = PHASE_CONFIG[phase]

    fat_config = {
        'embed_dim': args.embed_dim,
        'num_heads': args.num_heads,
        'num_layers': args.num_layers,
        'use_learnable_agg_token': args.use_learnable_agg_token,
        'use_visual_conditioning': args.use_visual_conditioning or phase_cfg['use_visual_conditioning'],
        'visual_token_dim': args.visual_token_dim,
    }

    model = AnyCalibWithFAT(
        model_id="anycalib_pinhole",
        use_fat=True,
        fat_config=fat_config,
        use_dinov2_small=args.use_dinov2_small and not args.use_dinov2_full,
        use_dinov2_full=args.use_dinov2_full,
        freeze_backbone=phase_cfg['freeze_backbone'],
        freeze_decoder=phase_cfg['freeze_decoder'],
    )

    return model.to(device)


def load_checkpoint(model, checkpoint_path: str, strict: bool = False):
    """Load checkpoint into model."""
    print(f"[CHECKPOINT] Loading from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    # Filter for FAT weights only
    fat_state = {k: v for k, v in state_dict.items() if 'fat' in k.lower()}

    if fat_state:
        missing, unexpected = model.load_state_dict(fat_state, strict=False)
        print(f"[CHECKPOINT] Loaded FAT weights: {len(fat_state)} keys")
        if missing:
            print(f"[CHECKPOINT] Missing keys: {len(missing)}")
    else:
        print("[CHECKPOINT] No FAT weights found, using random initialization")

    return checkpoint.get('epoch', 0), checkpoint.get('loss_history', [])


# =============================================================================
# TRAINING UTILITIES
# =============================================================================

def normalize_intrinsics(intrinsics: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """Normalize intrinsics by max(H, W) for consistent scale."""
    max_dim = max(H, W)
    return intrinsics / max_dim


def denormalize_intrinsics(intrinsics: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """Denormalize intrinsics back to pixel coordinates."""
    max_dim = max(H, W)
    return intrinsics * max_dim


def compute_calibration_loss(pred_intr: torch.Tensor, gt_intr: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """
    Compute MSE loss on normalized intrinsics.

    NOTE: This function is kept for compatibility but is NOT used in Phase 1 & 2 training.
    Instead, we use ray consistency loss (see compute_ray_consistency_loss).

    Args:
        pred_intr: [B, 4] predicted (fx, fy, cx, cy)
        gt_intr: [B, 4] ground truth (fx, fy, cx, cy)
        H, W: image dimensions

    Returns:
        Scalar loss
    """
    pred_norm = normalize_intrinsics(pred_intr, H, W)
    gt_norm = normalize_intrinsics(gt_intr, H, W)
    return F.mse_loss(pred_norm, gt_norm)


def plot_loss_curve(loss_history: List, val_loss_history: List, save_dir: Path):
    """Plot training and validation loss curves."""
    plt.figure(figsize=(10, 6))

    epochs = [h['epoch'] for h in loss_history]
    losses = [h['loss'] for h in loss_history]
    plt.plot(epochs, losses, 'b-', label='Training Loss', linewidth=2)

    if val_loss_history:
        val_epochs = [h['epoch'] for h in val_loss_history]
        val_losses = [h['loss'] for h in val_loss_history]
        plt.plot(val_epochs, val_losses, 'r--', label='Validation Loss', linewidth=2)

    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('FAT Training Loss', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / 'loss_curve.png', dpi=150)
    plt.close()


def save_training_summary(args, loss_history: List, val_loss_history: List, save_dir: Path):
    """Save training summary to text file."""
    summary_path = save_dir / 'training_summary.txt'

    with open(summary_path, 'w') as f:
        f.write(f"FAT Training Summary - Phase {args.phase}\n")
        f.write(f"{'='*60}\n\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("Configuration:\n")
        f.write(f"  Phase: {args.phase}\n")
        f.write(f"  Epochs: {args.num_epochs}\n")
        f.write(f"  Batch size: {args.batch_size}\n")
        f.write(f"  Learning rate: {args.learning_rate}\n")
        f.write(f"  Max ahead: {args.max_ahead}\n")
        f.write(f"  Visual conditioning: {args.use_visual_conditioning}\n")
        f.write(f"  Learnable agg token: {args.use_learnable_agg_token}\n\n")

        if loss_history:
            f.write("Training Results:\n")
            f.write(f"  Initial loss: {loss_history[0]['loss']:.6f}\n")
            f.write(f"  Final loss: {loss_history[-1]['loss']:.6f}\n")
            improvement = (1 - loss_history[-1]['loss'] / loss_history[0]['loss']) * 100
            f.write(f"  Improvement: {improvement:.2f}%\n")

        if val_loss_history:
            f.write(f"\n  Final val loss: {val_loss_history[-1]['loss']:.6f}\n")


# =============================================================================
# CALIBRATION BENCHMARKING
# =============================================================================

class CalibrationBenchmarkIterator:
    """
    Iterator that provides test set samples for per-epoch calibration benchmarking.

    Provides fixed or cycling samples from the test set for consistent evaluation.
    """

    def __init__(self, dataset, num_samples: int = 50, no_cycle: bool = True):
        """
        Initialize with a dataset that has GT intrinsics.

        Args:
            dataset: ObjectronFATDataset with GT intrinsics
            num_samples: Number of samples to use (or max available if less)
            no_cycle: If True, use fixed samples (no cycling)
        """
        self.dataset = dataset
        self.no_cycle = no_cycle

        # Get indices of samples with GT
        self.valid_indices = []
        for i, seq in enumerate(dataset.sequences):
            if 'gt_intrinsics' in seq or dataset._load_gt_intrinsics(seq['video_path']) is not None:
                self.valid_indices.append(i)

        # Limit to num_samples
        self.num_samples = min(num_samples, len(self.valid_indices))
        self.current_idx = 0

        print(f"[BENCHMARK] Initialized with {len(self.valid_indices)} samples with GT, using {self.num_samples}")

    def get_next_samples(self, n: int) -> List[Dict]:
        """Get next n samples for benchmarking."""
        samples = []
        indices_to_use = self.valid_indices[:self.num_samples]

        if self.no_cycle:
            # Fixed samples
            for i in indices_to_use[:n]:
                try:
                    samples.append(self.dataset[i])
                except Exception:
                    continue
        else:
            # Cycling through samples
            for _ in range(n):
                idx = indices_to_use[self.current_idx % len(indices_to_use)]
                try:
                    samples.append(self.dataset[idx])
                except Exception:
                    pass
                self.current_idx += 1

        return samples


def benchmark_calibration_accuracy(
    model,
    benchmark_iterator: CalibrationBenchmarkIterator,
    num_samples: int,
    device: str,
    epoch: int,
) -> Dict:
    """
    Run calibration benchmark comparing FAT model vs AnyCalib baseline.

    Args:
        model: FAT model
        benchmark_iterator: Iterator for test samples
        num_samples: Number of samples to evaluate
        device: Torch device
        epoch: Current epoch number

    Returns:
        Dictionary with benchmark results
    """
    print(f"\n[BENCHMARK] Epoch {epoch}: Evaluating calibration accuracy on {num_samples} samples...")

    model.eval()

    # Get samples
    samples = benchmark_iterator.get_next_samples(num_samples)

    fat_errors = {'fx': [], 'fy': [], 'cx': [], 'cy': []}
    baseline_errors = {'fx': [], 'fy': [], 'cx': [], 'cy': []}

    with torch.no_grad():
        for i, sample in enumerate(samples):
            try:
                imgs = sample['imgs'].unsqueeze(0).to(device)  # [1, N, 3, H, W]

                # Check for GT intrinsics
                if 'gt_intrinsics' not in sample:
                    continue
                gt_intr = sample['gt_intrinsics'].to(device)  # [4]

                B, N, C, H, W = imgs.shape

                # Evaluate FAT model
                result = model(imgs[0], cam_id="pinhole", return_calibration_info=True)
                pred = result['intrinsics'][0]  # [4]

                if not isinstance(pred, torch.Tensor):
                    pred = torch.tensor(pred, device=device)

                pred_np = pred.cpu().numpy()
                gt_np = gt_intr.cpu().numpy()

                # Compute relative errors (percentage)
                rel_err = np.abs(pred_np - gt_np) / (gt_np + 1e-8) * 100

                fat_errors['fx'].append(float(rel_err[0]))
                fat_errors['fy'].append(float(rel_err[1]))
                fat_errors['cx'].append(float(rel_err[2]))
                fat_errors['cy'].append(float(rel_err[3]))

                # Evaluate AnyCalib baseline (mean of per-frame predictions)
                anycalib_preds = []
                for n in range(N):
                    frame_result = model.forward_single_frame(imgs[0, n].unsqueeze(0), cam_id="pinhole")
                    frame_intr = frame_result['intrinsics'][0]
                    if isinstance(frame_intr, list):
                        frame_intr = frame_intr[0]
                    if not isinstance(frame_intr, torch.Tensor):
                        frame_intr = torch.tensor(frame_intr, device=device)
                    anycalib_preds.append(frame_intr)

                baseline_pred = torch.stack(anycalib_preds).mean(dim=0)
                baseline_np = baseline_pred.cpu().numpy()
                baseline_rel_err = np.abs(baseline_np - gt_np) / (gt_np + 1e-8) * 100

                baseline_errors['fx'].append(float(baseline_rel_err[0]))
                baseline_errors['fy'].append(float(baseline_rel_err[1]))
                baseline_errors['cx'].append(float(baseline_rel_err[2]))
                baseline_errors['cy'].append(float(baseline_rel_err[3]))

            except Exception as e:
                print(f"[WARN] Benchmark sample {i} failed: {e}")
                continue

            # Clear cache periodically
            if i % 10 == 0:
                torch.cuda.empty_cache()

    model.train()

    # Compute statistics
    results = {
        'epoch': epoch,
        'num_samples': num_samples,
        'num_valid_fat': len(fat_errors['fx']),
        'num_valid_baseline': len(baseline_errors['fx']),
    }

    # FAT results
    if fat_errors['fx']:
        all_fat_errors = fat_errors['fx'] + fat_errors['fy'] + fat_errors['cx'] + fat_errors['cy']
        results['fat'] = {
            'fx_mean': float(np.mean(fat_errors['fx'])),
            'fy_mean': float(np.mean(fat_errors['fy'])),
            'cx_mean': float(np.mean(fat_errors['cx'])),
            'cy_mean': float(np.mean(fat_errors['cy'])),
            'overall_mean': float(np.mean(all_fat_errors)),
            'overall_median': float(np.median(all_fat_errors)),
        }
    else:
        results['fat'] = None

    # Baseline results
    if baseline_errors['fx']:
        all_baseline_errors = baseline_errors['fx'] + baseline_errors['fy'] + baseline_errors['cx'] + baseline_errors['cy']
        results['baseline'] = {
            'fx_mean': float(np.mean(baseline_errors['fx'])),
            'fy_mean': float(np.mean(baseline_errors['fy'])),
            'cx_mean': float(np.mean(baseline_errors['cx'])),
            'cy_mean': float(np.mean(baseline_errors['cy'])),
            'overall_mean': float(np.mean(all_baseline_errors)),
            'overall_median': float(np.median(all_baseline_errors)),
        }
    else:
        results['baseline'] = None

    # Print summary
    print(f"[BENCHMARK] Epoch {epoch} Results:")
    if results['fat']:
        print(f"  FAT: Overall Mean={results['fat']['overall_mean']:.2f}%, Median={results['fat']['overall_median']:.2f}%")
    if results['baseline']:
        print(f"  AnyCalib Baseline: Overall Mean={results['baseline']['overall_mean']:.2f}%, Median={results['baseline']['overall_median']:.2f}%")

    return results


# =============================================================================
# POSE BENCHMARKING (Phase 3)
# =============================================================================

class PoseBenchmarkIterator:
    """
    Iterator that provides test set samples for per-epoch pose benchmarking.
    
    Provides fixed or cycling samples from the test set for consistent evaluation.
    """
    
    def __init__(self, dataset, num_samples: int = 50, no_cycle: bool = True):
        """
        Initialize with a dataset that has GT poses.
        
        Args:
            dataset: ObjectronFATDataset with GT poses
            num_samples: Number of samples to use (or max available if less)
            no_cycle: If True, use fixed samples (no cycling)
        """
        self.dataset = dataset
        self.no_cycle = no_cycle
        
        # Get indices of samples with GT poses
        self.valid_indices = []
        for i in range(len(dataset)):
            try:
                sample = dataset[i]
                if 'gt_poses' in sample or 'poses' in sample:
                    self.valid_indices.append(i)
            except Exception:
                continue
        
        # Limit to num_samples
        self.num_samples = min(num_samples, len(self.valid_indices))
        self.current_idx = 0
        
        print(f"[BENCHMARK] Pose iterator initialized with {len(self.valid_indices)} samples with GT, using {self.num_samples}")
    
    def get_next_samples(self, n: int) -> List[Dict]:
        """Get next n samples for benchmarking."""
        samples = []
        indices_to_use = self.valid_indices[:self.num_samples]
        
        if self.no_cycle:
            # Fixed samples
            for i in indices_to_use[:n]:
                try:
                    samples.append(self.dataset[i])
                except Exception:
                    continue
        else:
            # Cycling through samples
            for _ in range(n):
                idx = indices_to_use[self.current_idx % len(indices_to_use)]
                try:
                    samples.append(self.dataset[idx])
                except Exception:
                    pass
                self.current_idx += 1
        
        return samples


def rotation_error_degrees(R_pred: np.ndarray, R_gt: np.ndarray) -> float:
    """Compute rotation error in degrees."""
    R_diff = R_pred @ R_gt.T
    trace = np.trace(R_diff)
    trace = np.clip(trace, -1, 3)
    angle_rad = np.arccos((trace - 1) / 2)
    return np.degrees(angle_rad)


def translation_direction_error_degrees(t_pred: np.ndarray, t_gt: np.ndarray) -> float:
    """Compute translation direction error in degrees."""
    t_pred_norm = t_pred / (np.linalg.norm(t_pred) + 1e-8)
    t_gt_norm = t_gt / (np.linalg.norm(t_gt) + 1e-8)
    dot = np.clip(np.dot(t_pred_norm, t_gt_norm), -1, 1)
    angle_rad = np.arccos(dot)
    return np.degrees(angle_rad)


def translation_magnitude_error(t_pred: np.ndarray, t_gt: np.ndarray) -> float:
    """Compute translation magnitude error (relative)."""
    mag_pred = np.linalg.norm(t_pred)
    mag_gt = np.linalg.norm(t_gt)
    if mag_gt < 1e-8:
        return 0.0
    return abs(mag_pred - mag_gt) / mag_gt


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
        if 'poses' not in batch_data_gpu and 'gt_poses' not in batch_data_gpu:
            return None, None, None
        
        gt_poses = batch_data_gpu.get('poses') or batch_data_gpu.get('gt_poses')  # [batch, num_frames, 4, 4]
        
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
            pred_poses = output.get('proc_poses') or output.get('pose_result', {}).get('poses')  # [batch, num_frames, 4, 4] or similar
            
            if pred_poses is None:
                return None, None, None
            
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
    fat_model: nn.Module,
    baseline_model: nn.Module,
    benchmark_iterator: PoseBenchmarkIterator,
    num_samples: int,
    device: torch.device,
    epoch: int,
) -> Dict:
    """
    Run pose benchmark comparing FAT model vs AnyCam baseline.
    
    Args:
        fat_model: FAT Phase 3 model
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
    
    fat_rot_errors = []
    fat_trans_dir_errors = []
    fat_trans_mag_errors = []
    
    baseline_rot_errors = []
    baseline_trans_dir_errors = []
    baseline_trans_mag_errors = []
    
    for i, sample in enumerate(samples):
        # Create batch from single sample
        batch_data = {k: v.unsqueeze(0) if isinstance(v, torch.Tensor) else v 
                      for k, v in sample.items()}
        
        # Evaluate FAT model
        fat_rot, fat_trans_dir, fat_trans_mag = evaluate_pose_single_sample(
            fat_model, batch_data, device, "FAT Phase 3"
        )
        if fat_rot is not None:
            fat_rot_errors.append(fat_rot)
            fat_trans_dir_errors.append(fat_trans_dir)
            fat_trans_mag_errors.append(fat_trans_mag)
        
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
        'num_valid_fat': len(fat_rot_errors),
        'num_valid_baseline': len(baseline_rot_errors),
    }
    
    if fat_rot_errors:
        results['fat'] = {
            'rot_mean': float(np.mean(fat_rot_errors)),
            'rot_median': float(np.median(fat_rot_errors)),
            'rot_std': float(np.std(fat_rot_errors)),
            'trans_dir_mean': float(np.mean(fat_trans_dir_errors)),
            'trans_dir_median': float(np.median(fat_trans_dir_errors)),
            'trans_mag_mean': float(np.mean(fat_trans_mag_errors)),
            'rot_errors': fat_rot_errors,
            'trans_dir_errors': fat_trans_dir_errors,
            'trans_mag_errors': fat_trans_mag_errors,
        }
    else:
        results['fat'] = None
    
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
    if results['fat']:
        print(f"  FAT Phase 3: Rot={results['fat']['rot_mean']:.2f}° | Trans={results['fat']['trans_dir_mean']:.2f}°")
    if results['baseline']:
        print(f"  AnyCam Baseline: Rot={results['baseline']['rot_mean']:.2f}° | Trans={results['baseline']['trans_dir_mean']:.2f}°")
    
    return results


def plot_pose_benchmark_history(pose_history: List[Dict], save_dir: Path):
    """Plot pose benchmark results over epochs."""
    if not pose_history or len(pose_history) == 0:
        return
    
    # Extract data
    epochs = [h['epoch'] for h in pose_history]
    
    fat_rot_means = []
    fat_trans_means = []
    baseline_rot_means = []
    baseline_trans_means = []
    
    for h in pose_history:
        if h.get('fat'):
            fat_rot_means.append(h['fat']['rot_mean'])
            fat_trans_means.append(h['fat']['trans_dir_mean'])
        else:
            fat_rot_means.append(np.nan)
            fat_trans_means.append(np.nan)
        
        if h.get('baseline'):
            baseline_rot_means.append(h['baseline']['rot_mean'])
            baseline_trans_means.append(h['baseline']['trans_dir_mean'])
        else:
            baseline_rot_means.append(np.nan)
            baseline_trans_means.append(np.nan)
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Rotation error plot
    axes[0].plot(epochs, fat_rot_means, 'b-o', linewidth=2, markersize=6, label='FAT Phase 3')
    axes[0].plot(epochs, baseline_rot_means, 'r--s', linewidth=2, markersize=6, label='AnyCam Baseline')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Rotation Error (degrees)', fontsize=12)
    axes[0].set_title('Rotation Error vs Epoch', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # Translation direction error plot
    axes[1].plot(epochs, fat_trans_means, 'b-o', linewidth=2, markersize=6, label='FAT Phase 3')
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
    
    print(f"[VIZ] Pose benchmark curve saved: {plot_path}")


def save_pose_benchmark_log(pose_history: List[Dict], save_dir: Path):
    """Save pose benchmark history to JSON."""
    log_path = save_dir / "pose_benchmark_history.json"
    with open(log_path, 'w') as f:
        json.dump(pose_history, f, indent=2)
    print(f"[BENCHMARK] Pose benchmark history saved: {log_path}")


def plot_benchmark_history(benchmark_history: List[Dict], save_dir: Path):
    """Plot calibration benchmark results over epochs."""
    if not benchmark_history or len(benchmark_history) == 0:
        return

    epochs = [h['epoch'] for h in benchmark_history]

    fat_means = []
    baseline_means = []

    for h in benchmark_history:
        if h.get('fat'):
            fat_means.append(h['fat']['overall_mean'])
        else:
            fat_means.append(None)

        if h.get('baseline'):
            baseline_means.append(h['baseline']['overall_mean'])
        else:
            baseline_means.append(None)

    plt.figure(figsize=(10, 6))

    # Filter out None values for plotting
    valid_fat = [(e, m) for e, m in zip(epochs, fat_means) if m is not None]
    valid_baseline = [(e, m) for e, m in zip(epochs, baseline_means) if m is not None]

    if valid_fat:
        plt.plot([e for e, _ in valid_fat], [m for _, m in valid_fat],
                 'b-o', label='FAT', linewidth=2, markersize=4)

    if valid_baseline:
        plt.plot([e for e, _ in valid_baseline], [m for _, m in valid_baseline],
                 'r--s', label='AnyCalib Baseline', linewidth=2, markersize=4)

    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Mean Relative Error (%)', fontsize=12)
    plt.title('Calibration Accuracy: FAT vs AnyCalib Baseline', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(save_dir / 'benchmark_calibration_accuracy.png', dpi=150)
    plt.close()


# =============================================================================
# TRAINING PHASES
# =============================================================================

def train_phase_1_2(
    model,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    args,
    device: str,
    save_dir: Path,
    start_epoch: int = 0,
    existing_history: Optional[List] = None,
):
    """
    Phase 1 and 2 training: MSE loss on calibration.

    Phase 1: No visual conditioning
    Phase 2: With visual conditioning
    """
    model.train()

    # Get trainable parameters
    trainable_params = model.get_trainable_parameters()
    if not trainable_params:
        print("[ERROR] No trainable parameters!")
        return

    optimizer = optim.Adam(trainable_params, lr=args.learning_rate)

    # Count parameters
    trainable_count = sum(p.numel() for p in trainable_params)
    total_count = sum(p.numel() for p in model.parameters())
    print(f"[MODEL] Trainable: {trainable_count:,} / {total_count:,} ({100*trainable_count/total_count:.2f}%)")

    # Initialize history
    loss_history = existing_history if existing_history else []
    val_loss_history = []

    log_file = save_dir / "training_log.txt"

    # Initialize log file
    mode = 'a' if start_epoch > 0 else 'w'
    with open(log_file, mode) as f:
        if start_epoch == 0:
            f.write(f"FAT Phase {args.phase} Training Log\n")
            f.write(f"{'='*60}\n")
            f.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        else:
            f.write(f"\nResumed from epoch {start_epoch}\n\n")

    for epoch in range(start_epoch, args.num_epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Phase {args.phase} Epoch {epoch+1}/{args.num_epochs}")

        for batch_idx, batch in enumerate(pbar):
            batch_start = torch.cuda.Event(enable_timing=True)
            batch_end = torch.cuda.Event(enable_timing=True)
            batch_start.record()

            imgs = batch['imgs'].to(device)  # [B, N, 3, H, W]
            B, N, C, H, W = imgs.shape

            optimizer.zero_grad()

            try:
                # Track RANSAC time across batch
                total_ransac_time = 0.0
                total_inliers = 0
                total_outliers = 0

                # Forward pass with mixed precision (FP16)
                with autocast(device_type='cuda', dtype=torch.float16):
                    # Process each sequence in batch
                    batch_rays = []
                    batch_intrinsics = []
                    batch_image_sizes = []
                    batch_inlier_masks = []
                    batch_soft_weights = []

                    for b in range(B):
                        seq_imgs = imgs[b]  # [N, 3, H, W]

                        # Forward pass: get rays and fitted calibration
                        result = model(seq_imgs, cam_id="pinhole", return_calibration_info=True)

                        # Extract results
                        rays = result['rays']  # [1, H*W, 3] - has gradients!
                        intrinsics = result['intrinsics'][0]  # [4] - detached from graph
                        image_size = result['image_size']  # (H, W)
                        ransac_time = result['ransac_time_ms']

                        # Check for NaN/Inf in rays BEFORE passing to calibrator
                        if torch.isnan(rays).any() or torch.isinf(rays).any():
                            print(f"[WARN] Batch {batch_idx}, seq {b} produced NaN/Inf rays, skipping batch")
                            raise ValueError("NaN/Inf in rays")

                        # Convert intrinsics to tensor if needed
                        if not isinstance(intrinsics, torch.Tensor):
                            intrinsics = torch.tensor(intrinsics, device=device, dtype=torch.float32)

                        # Check if calibrator failed (returned invalid intrinsics)
                        if torch.isnan(intrinsics).any() or torch.isinf(intrinsics).any():
                            print(f"[WARN] Batch {batch_idx}, seq {b}: calibrator produced NaN/Inf intrinsics, skipping batch")
                            raise ValueError("NaN/Inf in intrinsics")

                        # Compute inlier mask and soft weights from fitted calibration
                        # This replicates RANSAC's inlier identification but with soft weighting
                        inlier_mask, residuals, soft_weights = model.compute_inlier_mask_from_residuals(
                            rays[0],  # [H*W, 3]
                            intrinsics,
                            image_size,
                            threshold_degrees=5.0,  # More lenient: 5° instead of 1°
                            use_soft_weights=True,  # Use soft exponential weighting
                        )

                        # Accumulate
                        batch_rays.append(rays)
                        batch_intrinsics.append(intrinsics)
                        batch_image_sizes.append(image_size)
                        batch_inlier_masks.append(inlier_mask)
                        batch_soft_weights.append(soft_weights)

                        total_ransac_time += ransac_time
                        total_inliers += inlier_mask.sum().item()
                        total_outliers += (~inlier_mask).sum().item()

                    # Stack batch
                    batch_rays = torch.cat(batch_rays, dim=0)  # [B, H*W, 3]
                    batch_inlier_masks = torch.stack(batch_inlier_masks)  # [B, H*W]
                    batch_soft_weights = torch.stack(batch_soft_weights)  # [B, H*W]

                # CRITICAL: Compute loss in FP32 (outside autocast) for numerical stability
                # Convert to FP32 for loss computation
                batch_rays_fp32 = batch_rays.float()
                batch_soft_weights_fp32 = batch_soft_weights.float()

                # Compute ray consistency loss in FP32
                loss, loss_info = model.compute_ray_consistency_loss(
                    batch_rays_fp32,
                    batch_intrinsics,
                    batch_image_sizes[0],  # Assume same size
                    soft_weights=batch_soft_weights_fp32,  # Use soft weights in FP32
                    inlier_mask=batch_inlier_masks,  # For logging only
                )

                # Check for NaN
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"[WARN] Batch {batch_idx} produced NaN/Inf loss, skipping")
                    optimizer.zero_grad()
                    continue

                # Backward
                loss.backward()

                # Gradient clipping (CRITICAL: prevents gradient explosion)
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)

                # Check for NaN in gradients
                has_nan_grad = False
                for p in trainable_params:
                    if p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any()):
                        has_nan_grad = True
                        break

                if has_nan_grad:
                    print(f"[WARN] Batch {batch_idx} produced NaN/Inf gradients, skipping")
                    if batch_idx == 0:
                        # Extra diagnostics for first batch
                        print(f"[DEBUG] Loss value: {loss.item():.6e}")
                        print(f"[DEBUG] Ray norms: min={batch_rays.norm(dim=-1).min():.6f}, max={batch_rays.norm(dim=-1).max():.6f}")
                        print(f"[DEBUG] Soft weights: min={batch_soft_weights.min():.6f}, max={batch_soft_weights.max():.6f}, mean={batch_soft_weights.mean():.6f}")
                        # Check which parameters have NaN gradients
                        for name, param in model.named_parameters():
                            if param.requires_grad and param.grad is not None:
                                if torch.isnan(param.grad).any():
                                    print(f"[DEBUG] NaN gradient in: {name}")
                    optimizer.zero_grad()
                    continue

                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

                batch_end.record()
                torch.cuda.synchronize()
                batch_time = batch_start.elapsed_time(batch_end) / 1000.0  # Convert to seconds
                avg_ransac_time = total_ransac_time / B

                # Calculate inlier ratio
                total_rays = total_inliers + total_outliers
                inlier_ratio = total_inliers / total_rays if total_rays > 0 else 0.0

                pbar.set_postfix({
                    'loss': f'{loss.item():.6f}',
                    'ransac': f'{avg_ransac_time:.1f}ms',
                    'inliers': f'{inlier_ratio*100:.1f}%',
                    'time': f'{batch_time:.2f}s'
                })

            except Exception as e:
                print(f"[ERROR] Batch {batch_idx} failed: {e}")
                torch.cuda.empty_cache()
                continue

            # Periodic cache clear and model health check
            if batch_idx % 50 == 0 and batch_idx > 0:
                torch.cuda.empty_cache()

                # Check for NaN in model weights (early corruption detection)
                has_nan_weights = False
                for name, param in model.named_parameters():
                    if param.requires_grad and (torch.isnan(param).any() or torch.isinf(param).any()):
                        print(f"[ERROR] NaN/Inf detected in parameter: {name}")
                        has_nan_weights = True

                if has_nan_weights:
                    print("[ERROR] Model weights corrupted! Training cannot continue.")
                    print("[ERROR] Please restart from last valid checkpoint.")
                    return

        # Epoch summary
        avg_loss = epoch_loss / max(num_batches, 1)
        loss_history.append({'epoch': epoch + 1, 'loss': avg_loss})

        # Validation
        val_loss = None
        if val_loader:
            val_loss = validate_phase_1_2(model, val_loader, device)
            val_loss_history.append({'epoch': epoch + 1, 'loss': val_loss})

        # Log
        log_msg = f"Epoch {epoch+1}: Train Loss = {avg_loss:.6f}"
        if val_loss is not None:
            log_msg += f", Val Loss = {val_loss:.6f}"
        print(log_msg)

        with open(log_file, 'a') as f:
            f.write(f"{log_msg}\n")

        # Save history and plot (append mode - growing)
        with open(save_dir / 'loss_history.json', 'w') as f:
            json.dump({
                'epoch_losses': loss_history,
                'val_epoch_losses': val_loss_history,
            }, f, indent=2)

        plot_loss_curve(loss_history, val_loss_history, save_dir)

        # Save checkpoint
        checkpoint_dir = save_dir / 'checkpoints'
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Always save latest
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
            'loss_history': loss_history,
            'val_loss_history': val_loss_history,
            'args': vars(args),
        }, checkpoint_dir / 'latest_checkpoint.pt')

        # Periodic and final checkpoints
        if (epoch + 1) % 10 == 0 or epoch == args.num_epochs - 1:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'loss_history': loss_history,
                'val_loss_history': val_loss_history,
                'args': vars(args),
            }, checkpoint_dir / f'checkpoint_epoch_{epoch+1}.pt')

        torch.cuda.empty_cache()

    # Final model
    torch.save({
        'epoch': args.num_epochs,
        'model_state_dict': model.state_dict(),
        'loss_history': loss_history,
        'val_loss_history': val_loss_history,
        'args': vars(args),
    }, save_dir / 'checkpoints' / 'final_model.pt')

    save_training_summary(args, loss_history, val_loss_history, save_dir)
    print(f"\n[COMPLETE] Phase {args.phase} training finished!")


def validate_phase_1_2(model, val_loader: DataLoader, device: str) -> float:
    """Validation for phases 1 and 2 using ray consistency loss."""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            imgs = batch['imgs'].to(device)
            B, N, C, H, W = imgs.shape

            try:
                # Process each sequence
                batch_rays = []
                batch_intrinsics = []
                batch_image_sizes = []
                batch_inlier_masks = []
                batch_soft_weights = []

                for b in range(B):
                    # Forward pass
                    result = model(imgs[b], cam_id="pinhole", return_calibration_info=True)

                    rays = result['rays']
                    intrinsics = result['intrinsics'][0]
                    image_size = result['image_size']

                    if not isinstance(intrinsics, torch.Tensor):
                        intrinsics = torch.tensor(intrinsics, device=device, dtype=torch.float32)

                    # Compute inlier mask and soft weights
                    inlier_mask, _, soft_weights = model.compute_inlier_mask_from_residuals(
                        rays[0], intrinsics, image_size, threshold_degrees=5.0, use_soft_weights=True
                    )

                    batch_rays.append(rays)
                    batch_intrinsics.append(intrinsics)
                    batch_image_sizes.append(image_size)
                    batch_inlier_masks.append(inlier_mask)
                    batch_soft_weights.append(soft_weights)

                # Stack batch
                batch_rays = torch.cat(batch_rays, dim=0)
                batch_inlier_masks = torch.stack(batch_inlier_masks)
                batch_soft_weights = torch.stack(batch_soft_weights)

                # Compute loss
                loss, _ = model.compute_ray_consistency_loss(
                    batch_rays,
                    batch_intrinsics,
                    batch_image_sizes[0],
                    soft_weights=batch_soft_weights,
                    inlier_mask=batch_inlier_masks,
                )

                if not (torch.isnan(loss) or torch.isinf(loss)):
                    total_loss += loss.item()
                    num_batches += 1
            except Exception:
                continue

    model.train()
    return total_loss / max(num_batches, 1)


# =============================================================================
# V2 TRAINING: Differentiable Calibrator with Implicit Differentiation
# =============================================================================

def train_phase_1_2_v2(
    model,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    args,
    device: str,
    save_dir: Path,
    start_epoch: int = 0,
    existing_history: Optional[List] = None,
    benchmark_iterator: Optional['CalibrationBenchmarkIterator'] = None,
    benchmark_samples_per_epoch: int = 50,
):
    """
    V2 Phase 1/2 training using differentiable calibrator with implicit differentiation.

    Key differences from V1:
    - Uses weighted least squares instead of ray consistency loss
    - Hard weights (1.0 for inliers, 1e-6 for outliers) instead of soft weights
    - Differentiates through the argmin solution, not RANSAC
    - Gradient flows: loss → intrinsics → least_squares_solution → rays → FAT

    This is the correct approach for training with non-differentiable RANSAC.

    Args:
        benchmark_iterator: Optional iterator for per-epoch calibration benchmarking
        benchmark_samples_per_epoch: Number of samples to benchmark each epoch
    """
    from experiments.models.differentiable_calibrator import (
        DifferentiableCalibrator,
        compute_differentiable_calibration_loss,
    )

    model.train()

    # Get trainable parameters
    trainable_params = model.get_trainable_parameters()
    if not trainable_params:
        print("[ERROR] No trainable parameters!")
        return

    optimizer = optim.Adam(trainable_params, lr=args.learning_rate)

    # Count parameters
    trainable_count = sum(p.numel() for p in trainable_params)
    total_count = sum(p.numel() for p in model.parameters())
    print(f"[MODEL] Trainable: {trainable_count:,} / {total_count:,} ({100*trainable_count/total_count:.2f}%)")

    # Initialize history
    loss_history = existing_history if existing_history else []
    val_loss_history = []
    benchmark_history = []

    log_file = save_dir / "training_log.txt"

    # Initialize log file
    mode = 'a' if start_epoch > 0 else 'w'
    with open(log_file, mode) as f:
        if start_epoch == 0:
            f.write(f"FAT Phase {args.phase} V2 Training Log (Differentiable Calibrator)\n")
            f.write(f"{'='*60}\n")
            f.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            if benchmark_iterator:
                f.write(f"Calibration Benchmark: Enabled ({benchmark_samples_per_epoch} samples/epoch)\n")
            f.write("\n")
        else:
            f.write(f"\nResumed from epoch {start_epoch}\n\n")

    # Regularization weight for anchoring to RANSAC solution
    regularization_lambda = getattr(args, 'regularization_lambda', 0.1)

    for epoch in range(start_epoch, args.num_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_gt_loss = 0.0
        epoch_reg_loss = 0.0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Phase {args.phase} V2 Epoch {epoch+1}/{args.num_epochs}")

        for batch_idx, batch in enumerate(pbar):
            batch_start = torch.cuda.Event(enable_timing=True)
            batch_end = torch.cuda.Event(enable_timing=True)
            batch_start.record()

            imgs = batch['imgs'].to(device)  # [B, N, 3, H, W]
            B, N, C, H, W = imgs.shape

            # Get GT intrinsics if available
            gt_intrinsics = batch.get('gt_intrinsics', None)
            if gt_intrinsics is not None:
                gt_intrinsics = gt_intrinsics.to(device)  # [B, 4]

            optimizer.zero_grad()

            try:
                total_ransac_time = 0.0
                total_inliers = 0
                total_rays = 0
                batch_losses = []

                for b in range(B):
                    seq_imgs = imgs[b]  # [N, 3, H, W]

                    # Forward with differentiable calibration
                    # This runs FAT in FP32 (already set in forward())
                    result = model.forward_with_differentiable_calibration(
                        seq_imgs,
                        cam_id="pinhole",
                        inlier_threshold_degrees=1.0,  # Hard threshold for RANSAC
                        inlier_weight=1.0,
                        outlier_weight=1e-6,
                    )

                    intrinsics_diff = result['intrinsics_diff']  # [4] - differentiable!
                    intrinsics_ransac = result['intrinsics_ransac']  # [4] - detached
                    inlier_mask = result['inlier_mask']
                    ransac_time = result['ransac_time_ms']

                    total_ransac_time += ransac_time
                    total_inliers += inlier_mask.sum().item()
                    total_rays += inlier_mask.numel()

                    # NEW LOSS: Reprojection loss using average per-frame AnyCalib intrinsics (phases 1-2 only)
                    # Get per-frame AnyCalib predictions
                    with torch.no_grad():
                        per_frame_intrinsics = model.get_per_frame_intrinsics(seq_imgs, cam_id="pinhole")  # [N, 4]

                    # Compute average intrinsics
                    average_intrinsics = per_frame_intrinsics.mean(dim=0)  # [4]

                    # Get original image size (before padding)
                    H_orig, W_orig = seq_imgs.shape[2], seq_imgs.shape[3]

                    # Get rays and image size from result
                    rays = result['rays']  # [1, H*W, 3] - WITH gradients
                    image_size = result['image_size']  # (H_ray, W_ray)

                    # Compute reprojection loss
                    loss, loss_info = model.compute_reprojection_loss(
                        predicted_rays=rays[0],  # [H*W, 3] - WITH gradients
                        average_intrinsics=average_intrinsics,  # [4] - detached (from no_grad)
                        ray_image_size=image_size,  # (H_ray, W_ray)
                        original_image_size=(H_orig, W_orig),  # (H_orig, W_orig)
                    )

                    batch_losses.append(loss)

                # Average loss over batch
                total_loss = sum(batch_losses) / len(batch_losses)

                # Check for NaN
                if torch.isnan(total_loss) or torch.isinf(total_loss):
                    print(f"[WARN] Batch {batch_idx} produced NaN/Inf loss, skipping")
                    optimizer.zero_grad()
                    continue

                # Backward (no GradScaler needed - we're in FP32 for trainable parts)
                total_loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)

                # Check for NaN gradients
                has_nan_grad = False
                for p in trainable_params:
                    if p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any()):
                        has_nan_grad = True
                        break

                if has_nan_grad:
                    print(f"[WARN] Batch {batch_idx} produced NaN/Inf gradients, skipping")
                    if batch_idx == 0:
                        # Debug first batch
                        print(f"[DEBUG] Loss value: {total_loss.item():.6e}")
                        for name, param in model.named_parameters():
                            if param.requires_grad and param.grad is not None:
                                if torch.isnan(param.grad).any():
                                    print(f"[DEBUG] NaN gradient in: {name}")
                    optimizer.zero_grad()
                    continue

                optimizer.step()

                epoch_loss += total_loss.item()
                num_batches += 1

                batch_end.record()
                torch.cuda.synchronize()
                batch_time = batch_start.elapsed_time(batch_end) / 1000.0

                avg_ransac_time = total_ransac_time / B
                inlier_ratio = total_inliers / total_rays if total_rays > 0 else 0.0

                pbar.set_postfix({
                    'loss': f'{total_loss.item():.6f}',
                    'ransac': f'{avg_ransac_time:.1f}ms',
                    'inliers': f'{inlier_ratio*100:.1f}%',
                    'time': f'{batch_time:.2f}s'
                })

            except Exception as e:
                print(f"[ERROR] Batch {batch_idx} failed: {e}")
                import traceback
                traceback.print_exc()
                torch.cuda.empty_cache()
                optimizer.zero_grad()
                continue

            # Periodic cache clear
            if batch_idx % 50 == 0 and batch_idx > 0:
                torch.cuda.empty_cache()

        # Epoch summary
        avg_loss = epoch_loss / max(num_batches, 1)
        loss_history.append({'epoch': epoch + 1, 'loss': avg_loss})

        # Validation
        val_loss = None
        if val_loader:
            val_loss = validate_phase_1_2_v2(model, val_loader, device, regularization_lambda)
            val_loss_history.append({'epoch': epoch + 1, 'loss': val_loss})

        # Calibration Benchmarking
        if benchmark_iterator:
            benchmark_result = benchmark_calibration_accuracy(
                model,
                benchmark_iterator,
                benchmark_samples_per_epoch,
                device,
                epoch + 1,
            )
            benchmark_history.append(benchmark_result)

            # Save benchmark history
            with open(save_dir / 'benchmark_history.json', 'w') as f:
                json.dump(benchmark_history, f, indent=2)

            # Plot benchmark history
            plot_benchmark_history(benchmark_history, save_dir)

        # Log
        log_msg = f"Epoch {epoch+1}: Train Loss = {avg_loss:.6f}"
        if val_loss is not None:
            log_msg += f", Val Loss = {val_loss:.6f}"
        print(log_msg)

        with open(log_file, 'a') as f:
            f.write(f"{log_msg}\n")

        # Save history and plot
        with open(save_dir / 'loss_history.json', 'w') as f:
            json.dump({
                'epoch_losses': loss_history,
                'val_epoch_losses': val_loss_history,
            }, f, indent=2)

        plot_loss_curve(loss_history, val_loss_history, save_dir)

        # Save checkpoint
        checkpoint_dir = save_dir / 'checkpoints'
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Always save latest
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
            'loss_history': loss_history,
            'val_loss_history': val_loss_history,
            'args': vars(args),
        }, checkpoint_dir / 'latest_checkpoint.pt')

        # Periodic checkpoints
        if (epoch + 1) % 10 == 0 or epoch == args.num_epochs - 1:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'loss_history': loss_history,
                'val_loss_history': val_loss_history,
                'args': vars(args),
            }, checkpoint_dir / f'checkpoint_epoch_{epoch+1}.pt')

        torch.cuda.empty_cache()

    # Final model
    torch.save({
        'epoch': args.num_epochs,
        'model_state_dict': model.state_dict(),
        'loss_history': loss_history,
        'val_loss_history': val_loss_history,
        'args': vars(args),
    }, save_dir / 'checkpoints' / 'final_model.pt')

    save_training_summary(args, loss_history, val_loss_history, save_dir)
    print(f"\n[COMPLETE] Phase {args.phase} V2 training finished!")


def validate_phase_1_2_v2(model, val_loader: DataLoader, device: str, regularization_lambda: float = 0.1) -> float:
    """Validation for V3 reprojection loss (same as training)."""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            imgs = batch['imgs'].to(device)
            B, N, C, H, W = imgs.shape

            try:
                batch_losses = []
                for b in range(B):
                    seq_imgs = imgs[b]  # [N, 3, H, W]
                    
                    # Forward pass with FAT aggregation
                    result = model(seq_imgs, cam_id="pinhole")
                    
                    # Get per-frame AnyCalib predictions (for average intrinsics)
                    per_frame_intrinsics = model.get_per_frame_intrinsics(seq_imgs, cam_id="pinhole")  # [N, 4]
                    
                    # Compute average intrinsics
                    average_intrinsics = per_frame_intrinsics.mean(dim=0)  # [4]
                    
                    # Get original image size (before padding)
                    H_orig, W_orig = seq_imgs.shape[2], seq_imgs.shape[3]
                    
                    # Get rays and image size from result
                    rays = result['rays']  # [1, H*W, 3]
                    image_size = result['image_size']  # (H_ray, W_ray)
                    
                    # Compute reprojection loss (same as training)
                    loss, _ = model.compute_reprojection_loss(
                        predicted_rays=rays[0],  # [H*W, 3]
                        average_intrinsics=average_intrinsics,  # [4]
                        ray_image_size=image_size,  # (H_ray, W_ray)
                        original_image_size=(H_orig, W_orig),  # (H_orig, W_orig)
                    )
                    
                    if not (torch.isnan(loss) or torch.isinf(loss)):
                        batch_losses.append(loss.item())

                if batch_losses:
                    total_loss += sum(batch_losses) / len(batch_losses)
                    num_batches += 1

            except Exception as e:
                print(f"[WARN] Validation batch failed: {e}")
                continue

    model.train()
    return total_loss / max(num_batches, 1)


# =============================================================================
# PHASE 3 COMBINED LOSS FUNCTION
# =============================================================================

def compute_combined_phase3_loss(
    model,
    output_data: Dict,
    flow_criterion,
    lambda_flow: float = 1.0,
    lambda_calib: float = 1e-4,
) -> Tuple[torch.Tensor, Dict]:
    """
    Compute combined loss for Phase 3 V2:
    1. AnyCam flow reprojection loss (PRIMARY - for pose learning)
    2. Phase 1 calibration reprojection loss (ANCHOR - prevents calibration explosion)

    The calibration loss is NOT meant to constrain FAT to the mean calibration
    (the mean is not perfect either). It's only to prevent explosion of calibration
    in favor of pose. The weight (lambda_calib) should be tuned empirically through
    small-scale experiments.

    Args:
        model: AnyCamWrapperWithFATCalibration model
        output_data: Output from model.forward_with_calibration_info()
        flow_criterion: AnyCam flow loss function (PoseLoss)
        lambda_flow: Weight for flow loss (default 1.0)
        lambda_calib: Weight for calibration loss (1e-4 to 0.15, tune empirically)

    Returns:
        total_loss: Combined loss tensor
        loss_info: Dict with individual loss components
    """
    from experiments.models.anycalib_with_fat import AnyCalibWithFAT

    # 1. Flow reprojection loss (AnyCam) - PRIMARY
    flow_loss_dict = flow_criterion(output_data)
    flow_loss = flow_loss_dict.get('loss', flow_loss_dict.get('total_loss', sum(flow_loss_dict.values())))

    # 2. Calibration reprojection loss (Phase 1 style) - ANCHOR
    # Extract data needed for calibration loss
    fat_rays = output_data["fat_rays"]  # [B, H*W, 3] - WITH GRADIENTS
    fat_image_size = output_data["fat_image_size"]  # (H_ray, W_ray)
    average_intrinsics = output_data["average_intrinsics"]  # [B, 4] - detached
    original_size = output_data["original_image_size"]  # (H_orig, W_orig)

    B = fat_rays.shape[0]
    calib_losses = []

    for b in range(B):
        calib_loss, _ = AnyCalibWithFAT.compute_reprojection_loss(
            predicted_rays=fat_rays[b],  # [H*W, 3] - WITH gradients
            average_intrinsics=average_intrinsics[b],  # [4] - detached
            ray_image_size=fat_image_size,
            original_image_size=original_size,
        )
        calib_losses.append(calib_loss)

    calib_loss = sum(calib_losses) / len(calib_losses)

    # 3. Combined loss
    total_loss = lambda_flow * flow_loss + lambda_calib * calib_loss

    # Calculate contribution as fraction of total optimization magnitude
    flow_magnitude = abs(lambda_flow * flow_loss.item())
    calib_magnitude = abs(lambda_calib * calib_loss.item())
    total_magnitude = flow_magnitude + calib_magnitude

    loss_info = {
        'total_loss': total_loss.item(),
        'flow_loss': flow_loss.item(),
        'calib_loss': calib_loss.item(),
        'lambda_flow': lambda_flow,
        'lambda_calib': lambda_calib,
        'calib_contribution': calib_magnitude / (total_magnitude + 1e-8) * 100,
    }

    return total_loss, loss_info


def plot_phase3_loss_curves(loss_history: List[Dict], val_loss_history: List[Dict], save_dir: Path):
    """Plot Phase 3 loss curves showing flow and calibration losses separately."""
    if not loss_history:
        return

    epochs = [h['epoch'] for h in loss_history]
    total_losses = [h['loss'] for h in loss_history]
    flow_losses = [h.get('flow_loss', 0) for h in loss_history]
    calib_losses = [h.get('calib_loss', 0) for h in loss_history]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Total loss
    axes[0].plot(epochs, total_losses, 'b-o', linewidth=2, markersize=4, label='Train')
    if val_loss_history:
        val_epochs = [h['epoch'] for h in val_loss_history]
        val_losses = [h['loss'] for h in val_loss_history]
        axes[0].plot(val_epochs, val_losses, 'r--s', linewidth=2, markersize=4, label='Val')
        axes[0].legend()
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Total Loss')
    axes[0].set_title('Total Loss (Flow + Calib)')
    axes[0].grid(True, alpha=0.3)

    # Flow loss
    axes[1].plot(epochs, flow_losses, 'g-s', linewidth=2, markersize=4)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Flow Loss')
    axes[1].set_title('Flow Reprojection Loss')
    axes[1].grid(True, alpha=0.3)

    # Calibration loss
    axes[2].plot(epochs, calib_losses, 'r-^', linewidth=2, markersize=4)
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Calibration Loss')
    axes[2].set_title('Calibration Reprojection Loss')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_dir / 'loss_curves_phase3.png', dpi=150, bbox_inches='tight')
    plt.close()


# =============================================================================
# PHASE 3 TRAINING
# =============================================================================

def train_phase_3(
    model,  # This will be AnyCamWrapperWithFATCalibration
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    args,
    device: str,
    save_dir: Path,
    start_epoch: int = 0,
    existing_history: Optional[List] = None,
    benchmark_iterator: Optional = None,
    benchmark_samples_per_epoch: int = 50,
):
    """
    Phase 3 V2 training: Combined flow + calibration loss (self-supervised).

    This integrates FAT into the full AnyCam pipeline with combined loss:
    1. Flow reprojection loss (PRIMARY - for pose learning)
    2. Calibration reprojection loss (ANCHOR - prevents calibration explosion)

    Key differences from original Phase 3:
    - Loads Phase 1 checkpoint directly (skips Phase 2, no visual tokens)
    - Combined loss: flow + calibration anchor
    - No alternating training (FAT + Pose Head trained together)
    - Proper GradScaler for mixed precision
    - Separate logging of flow and calibration losses

    Training is FULLY SELF-SUPERVISED:
    - GT only used for passive benchmark monitoring
    - Benchmark results help decide if lambda_calib needs adjustment
    """
    from anycam.loss import make_loss
    from torch.cuda.amp import GradScaler
    from experiments.models.anycam_wrapper_fat import AnyCamWrapperWithFATCalibration

    print("[INFO] Phase 3 V2 training - Combined flow + calibration loss")
    print("[INFO] Training FAT + Pose Head together (no alternating)")

    # Ensure model is AnyCamWrapperWithFATCalibration
    if not isinstance(model, AnyCamWrapperWithFATCalibration):
        raise ValueError(f"Phase 3 requires AnyCamWrapperWithFATCalibration, got {type(model)}")

    model.train()

    # Setup freezing: Always train both FAT and Pose Head together
    model.freeze_except_fat_and_pose()

    # Count trainable parameters
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[MODEL] Trainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    # Create flow loss function (AnyCam)
    loss_config = {
        "type": "pose_loss",
        "flow_criterion": "l1",
        "dist_criterion": "l1",
        "lambda_flow": 1.0,
        "lambda_dist": 0.0,
        "use_flow_uncertainty": True,
        "use_dist_uncertainty": True,
    }
    flow_criterion = make_loss(loss_config)

    # Get loss weights from args
    lambda_flow = 1.0
    lambda_calib = getattr(args, 'lambda_calib', 1e-4)
    weight_decay = getattr(args, 'weight_decay', 1e-4)

    print(f"[LOSS] Combined loss: lambda_flow={lambda_flow}, lambda_calib={lambda_calib}")
    print(f"[LOSS] Calibration loss serves as ANCHOR to prevent explosion (not for convergence)")

    # Create optimizer with weight decay
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=args.learning_rate, weight_decay=weight_decay)
    print(f"[OPTIM] AdamW with lr={args.learning_rate}, weight_decay={weight_decay}")

    # GradScaler for mixed precision
    scaler = GradScaler()
    print(f"[TRAIN] GradScaler enabled for mixed precision training")

    # Initialize history
    loss_history = existing_history if existing_history is not None else []
    val_loss_history = []
    batch_losses = []

    log_file = save_dir / "training_log.txt"
    loss_json_path = save_dir / "loss_history.json"

    # Initialize log file
    if start_epoch > 0:
        with open(log_file, 'a') as f:
            f.write(f"\n{'='*70}\n")
            f.write(f"RESUMED TRAINING from epoch {start_epoch}\n")
            f.write(f"{'='*70}\n\n")
    else:
        with open(log_file, 'w') as f:
            f.write(f"FAT Phase 3 V2 Training Log (Combined Loss)\n")
            f.write(f"{'='*70}\n")
            f.write(f"Num epochs: {args.num_epochs}\n")
            f.write(f"Batch size: {train_loader.batch_size}\n")
            f.write(f"Max ahead: {args.max_ahead}\n")
            f.write(f"Learning rate: {args.learning_rate}\n")
            f.write(f"Weight decay: {weight_decay}\n")
            f.write(f"Lambda flow: {lambda_flow}\n")
            f.write(f"Lambda calib: {lambda_calib}\n")
            f.write(f"GradScaler: Enabled\n")
            f.write(f"Alternating training: Disabled (train both)\n")
            f.write(f"Validation: {'Enabled' if val_loader else 'Disabled'}\n")
            f.write(f"Benchmark: {'Enabled' if benchmark_iterator else 'Disabled'}\n")
            f.write(f"Device: {device}\n")
            f.write(f"{'='*70}\n\n")

    print(f"\n{'='*70}")
    print(f"[TRAIN] Starting Phase 3 V2 training for {args.num_epochs} epochs")
    print(f"[TRAIN] Loss: Flow reprojection + Calibration anchor (self-supervised)")
    if val_loader:
        print(f"[TRAIN] Validation: Enabled ({len(val_loader)} batches)")
    if benchmark_iterator:
        print(f"[TRAIN] Benchmark: Enabled ({benchmark_samples_per_epoch} samples/epoch)")
    print(f"{'='*70}\n")

    for epoch in range(start_epoch, args.num_epochs):
        epoch_loss = 0.0
        epoch_flow_loss = 0.0
        epoch_calib_loss = 0.0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Phase 3 V2 Epoch {epoch+1}/{args.num_epochs}")

        for batch_idx, batch in enumerate(pbar):
            # Move to device
            imgs = batch['imgs'].to(device)  # [B, N, 3, H, W]

            # Prepare data dict for AnyCam wrapper
            data = {'imgs': imgs}

            optimizer.zero_grad()

            try:
                # Forward pass with mixed precision (use forward_with_calibration_info)
                with autocast(device_type='cuda', dtype=torch.float16):
                    output_data = model.forward_with_calibration_info(data)

                    # Compute combined loss
                    loss, loss_info = compute_combined_phase3_loss(
                        model, output_data, flow_criterion,
                        lambda_flow=lambda_flow,
                        lambda_calib=lambda_calib,
                    )

                # Check for NaN/Inf
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"[WARN] Batch {batch_idx} produced NaN/Inf loss, skipping")
                    torch.cuda.empty_cache()
                    continue

                # Backward with GradScaler
                scaler.scale(loss).backward()

                # Unscale for gradient clipping
                scaler.unscale_(optimizer)

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)

                # Check for NaN gradients
                has_nan_grad = any(
                    p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any())
                    for p in trainable_params
                )

                if has_nan_grad:
                    print(f"[WARN] Batch {batch_idx} produced NaN/Inf gradients, skipping")
                    optimizer.zero_grad()
                    torch.cuda.empty_cache()
                    continue

                # Optimizer step with scaler
                scaler.step(optimizer)
                scaler.update()

                # Accumulate losses
                epoch_loss += loss_info['total_loss']
                epoch_flow_loss += loss_info['flow_loss']
                epoch_calib_loss += loss_info['calib_loss']
                num_batches += 1
                batch_losses.append(loss_info)

                pbar.set_postfix({
                    'loss': f"{loss_info['total_loss']:.4f}",
                    'flow': f"{loss_info['flow_loss']:.4f}",
                    'calib': f"{loss_info['calib_loss']:.2f}",
                })

                # Log every 10 batches
                if batch_idx % 10 == 0:
                    log_msg = (f"[TRAIN] Epoch {epoch+1} | Batch {batch_idx}/{len(train_loader)} | "
                               f"Loss: {loss_info['total_loss']:.6f} | Flow: {loss_info['flow_loss']:.6f} | "
                               f"Calib: {loss_info['calib_loss']:.4f} | CalibContrib: {loss_info['calib_contribution']:.1f}%")
                    with open(log_file, 'a') as f:
                        f.write(f"{log_msg}\n")

                # Debug first few batches in first epoch
                if epoch == 0 and batch_idx < 5:
                    print(f"[DEBUG] Batch {batch_idx}: flow={loss_info['flow_loss']:.6f}, "
                          f"calib={loss_info['calib_loss']:.4f}, "
                          f"calib_contrib={loss_info['calib_contribution']:.1f}%")

            except Exception as e:
                print(f"[ERROR] Batch {batch_idx} failed: {e}")
                import traceback
                traceback.print_exc()
                torch.cuda.empty_cache()
                continue

            # Periodic GPU cache clear
            if batch_idx % 50 == 0 and batch_idx > 0:
                torch.cuda.empty_cache()

        # Clear cache at end of epoch
        torch.cuda.empty_cache()

        # Epoch summary
        avg_loss = epoch_loss / max(num_batches, 1)
        avg_flow_loss = epoch_flow_loss / max(num_batches, 1)
        avg_calib_loss = epoch_calib_loss / max(num_batches, 1)

        loss_history.append({
            'epoch': epoch + 1,
            'loss': avg_loss,
            'flow_loss': avg_flow_loss,
            'calib_loss': avg_calib_loss,
        })

        # Validation
        val_loss = None
        if val_loader:
            model.eval()
            val_loss_sum = 0.0
            val_flow_sum = 0.0
            val_calib_sum = 0.0
            val_batches = 0
            with torch.no_grad():
                for val_batch in val_loader:
                    val_imgs = val_batch['imgs'].to(device)
                    val_data = {'imgs': val_imgs}

                    try:
                        with autocast(device_type='cuda', dtype=torch.float16):
                            val_output = model.forward_with_calibration_info(val_data)
                            val_loss_tensor, val_loss_info = compute_combined_phase3_loss(
                                model, val_output, flow_criterion,
                                lambda_flow=lambda_flow,
                                lambda_calib=lambda_calib,
                            )

                        if not (torch.isnan(val_loss_tensor) or torch.isinf(val_loss_tensor)):
                            val_loss_sum += val_loss_info['total_loss']
                            val_flow_sum += val_loss_info['flow_loss']
                            val_calib_sum += val_loss_info['calib_loss']
                            val_batches += 1
                    except Exception as e:
                        print(f"[WARN] Validation batch failed: {e}")
                        continue

            if val_batches > 0:
                val_loss = val_loss_sum / val_batches
                val_flow_loss = val_flow_sum / val_batches
                val_calib_loss = val_calib_sum / val_batches
                val_loss_history.append({
                    'epoch': epoch + 1,
                    'loss': val_loss,
                    'flow_loss': val_flow_loss,
                    'calib_loss': val_calib_loss,
                })
            model.train()

        log_msg = f"\n[EPOCH {epoch+1}] Train: loss={avg_loss:.6f}, flow={avg_flow_loss:.6f}, calib={avg_calib_loss:.4f}"
        if val_loss is not None:
            log_msg += f"\n[EPOCH {epoch+1}] Val:   loss={val_loss:.6f}, flow={val_flow_loss:.6f}, calib={val_calib_loss:.4f}"
        log_msg += "\n"

        print(log_msg)
        with open(log_file, 'a') as f:
            f.write(log_msg)

        # Per-epoch benchmarking (if enabled)
        # These benchmarks use GT for passive monitoring only (not used in training)
        if benchmark_iterator:
            # Calibration benchmarking
            if hasattr(benchmark_iterator, 'get_next_samples'):
                # This is a pose benchmark iterator, skip calibration
                pass
            else:
                # Calibration benchmarking
                try:
                    device_torch = torch.device(device)
                    calibration_result = benchmark_calibration_accuracy(
                        model.fat_model,  # Use FAT model for calibration
                        benchmark_iterator,
                        benchmark_samples_per_epoch,
                        device_torch,
                        epoch + 1,
                    )
                    # Log calibration benchmark results
                    print(f"[BENCHMARK] Calibration: {calibration_result}")
                except Exception as e:
                    print(f"[WARN] Calibration benchmark failed: {e}")

        # Pose benchmarking (if enabled and baseline model available)
        if hasattr(args, 'baseline_model') and args.baseline_model is not None:
            if hasattr(benchmark_iterator, 'get_next_samples'):
                try:
                    # This is a pose benchmark iterator
                    pose_result = benchmark_pose_accuracy(
                        fat_model=model,
                        baseline_model=args.baseline_model,
                        benchmark_iterator=benchmark_iterator,
                        num_samples=benchmark_samples_per_epoch,
                        device=torch.device(device),
                        epoch=epoch + 1,
                    )
                    # Save pose benchmark results
                    if not hasattr(model, '_pose_benchmark_history'):
                        model._pose_benchmark_history = []
                    model._pose_benchmark_history.append(pose_result)
                    save_pose_benchmark_log(model._pose_benchmark_history, save_dir)
                    plot_pose_benchmark_history(model._pose_benchmark_history, save_dir)
                except Exception as e:
                    print(f"[WARN] Pose benchmark failed: {e}")

        # Save checkpoint (includes scaler state)
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'loss_history': loss_history,
            'val_loss_history': val_loss_history,
            'lambda_calib': lambda_calib,
            'lambda_flow': lambda_flow,
            'weight_decay': weight_decay,
        }
        torch.save(checkpoint, save_dir / 'checkpoints' / f'checkpoint_epoch_{epoch+1}.pt')
        torch.save(checkpoint, save_dir / 'checkpoints' / 'latest_checkpoint.pt')

        # Save loss history
        with open(loss_json_path, 'w') as f:
            json.dump({
                'epoch_losses': loss_history,
                'val_epoch_losses': val_loss_history,
                'batch_losses': batch_losses[-100:],  # Keep last 100 batches
            }, f, indent=2)

        # Plot loss curves (shows flow and calib separately)
        plot_phase3_loss_curves(loss_history, val_loss_history, save_dir)

    print(f"\n{'='*70}")
    print(f"[COMPLETE] Phase 3 V2 training complete!")
    print(f"[INFO] Final train loss: {loss_history[-1]['loss']:.6f}")
    print(f"[INFO] Final flow loss: {loss_history[-1]['flow_loss']:.6f}")
    print(f"[INFO] Final calib loss: {loss_history[-1]['calib_loss']:.4f}")
    print(f"{'='*70}\n")


# =============================================================================
# MAIN
# =============================================================================

def main():
    args = parse_args()

    # Set device (keep as string like phases 1/2)
    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")

    # Apply phase defaults
    phase_cfg = PHASE_CONFIG[args.phase]
    if args.batch_size is None:
        args.batch_size = phase_cfg['batch_size']
    if args.learning_rate is None:
        args.learning_rate = phase_cfg['learning_rate']
    if args.phase >= 2:
        args.use_visual_conditioning = True

    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / 'checkpoints').mkdir(exist_ok=True)

    # Load dataset split
    split = load_dataset_split(args.split_file)

    # Create datasets
    video_dir = Path(args.objectron_videos)
    gt_dir = Path(args.objectron_gt) if args.objectron_gt else None

    train_dataset = ObjectronFATDataset(
        video_dir=video_dir,
        gt_dir=gt_dir,
        video_indices=split['train'],
        max_ahead=args.max_ahead,
        phase=args.phase,
    )

    val_dataset = ObjectronFATDataset(
        video_dir=video_dir,
        gt_dir=gt_dir,
        video_indices=split['val'],
        max_ahead=args.max_ahead,
        phase=args.phase,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # Create test dataset for benchmarking (only for phase 3)
    benchmark_iterator = None
    baseline_model = None
    if args.phase == 3:
        # Create test dataset with GT poses for benchmarking
        test_dataset = ObjectronFATDataset(
            video_dir=video_dir,
            gt_dir=gt_dir,
            video_indices=split.get('test', split['val']),  # Use test split if available, else val
            max_ahead=args.max_ahead,
            phase=args.phase,
        )
        
        # Create pose benchmark iterator
        benchmark_iterator = PoseBenchmarkIterator(
            dataset=test_dataset,
            num_samples=getattr(args, 'benchmark_pose_samples', 50),
            no_cycle=getattr(args, 'benchmark_no_cycle', True),
        )
        print(f"[BENCHMARK] Initialized pose benchmark with {len(test_dataset)} test sequences")
        
        # Create baseline model for comparison (AnyCam with 32 candidates)
        if not args.disable_benchmark:
            print(f"[BENCHMARK] Loading AnyCam baseline model...")
            try:
                from experiments.train_pose_head_anycalib import AnyCamWrapperWithAnyCaLib, AnyCaLibBatchInference
                from anycam.models import make_pose_predictor, make_depth_predictor
                
                # Load config
                config_path = Path(args.anycam_config)
                with open(config_path, 'r') as f:
                    full_config = yaml.safe_load(f)
                
                pose_predictor_config = full_config['model']['pose_predictor']
                depth_predictor_config = full_config['model']['depth_predictor']
                
                # Create baseline (AnyCam with 32 candidates, no DA3/FAT)
                anycalib_inference = AnyCaLibBatchInference(device=torch.device(device))  # Convert to torch.device for AnyCaLibBatchInference
                baseline_model = AnyCamWrapperWithAnyCaLib(
                    pose_predictor_config=pose_predictor_config,
                    depth_predictor_config=depth_predictor_config,
                    anycalib_model=anycalib_inference,
                )
                baseline_model = baseline_model.to(device)  # device is string, .to() accepts strings
                baseline_model.eval()
                
                # Load baseline checkpoint if available
                baseline_checkpoint_path = Path(args.baseline_checkpoint)
                if baseline_checkpoint_path.exists():
                    checkpoint = torch.load(baseline_checkpoint_path, map_location=device, weights_only=False)  # map_location accepts strings
                    if 'model_state_dict' in checkpoint:
                        baseline_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                    print(f"[BENCHMARK] Baseline model loaded from {baseline_checkpoint_path}")
                
                # Store baseline model in args for train_phase_3
                args.baseline_model = baseline_model
                print(f"[BENCHMARK] Baseline model ready for comparison")
            except Exception as e:
                print(f"[WARN] Failed to load baseline model: {e}")
                print(f"[WARN] Pose benchmarking will be disabled")
                args.baseline_model = None

    print(f"[DATA] Train: {len(train_dataset)} sequences, Val: {len(val_dataset)} sequences")

    # Create model
    if args.phase == 3:
        # Phase 3: Create AnyCamWrapperWithFATCalibration
        # Follow same pattern as phases 1/2: create model, move to device once
        from experiments.models.anycam_wrapper_fat import AnyCamWrapperWithFATCalibration
        
        # Load AnyCam config
        config_path = Path(args.anycam_config)
        if not config_path.exists():
            raise FileNotFoundError(f"AnyCam config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            full_config = yaml.safe_load(f)
        
        pose_predictor_config = full_config['model']['pose_predictor']
        depth_predictor_config = full_config['model']['depth_predictor']
        
        # Create FAT model (AnyCalibWithFAT) - already on device from create_fat_model
        fat_model = create_fat_model(args, args.phase, device)
        
        # Create wrapper
        model = AnyCamWrapperWithFATCalibration(
            fat_model=fat_model,
            pose_predictor_config=pose_predictor_config,
            depth_predictor_config=depth_predictor_config,
            use_provided_depth=False,
            use_provided_flow=False,
        )
        
        print(f"[MODEL] Created AnyCamWrapperWithFATCalibration for Phase 3")
    else:
        # Phase 1/2: Use standard AnyCalibWithFAT
        model = create_fat_model(args, args.phase, device)

    # Load checkpoint if specified (same pattern as phases 1/2)
    start_epoch = 0
    existing_history = None

    if args.resume:
        if args.phase == 3:
            # For Phase 3, load full wrapper checkpoint
            print(f"[RESUME] Loading Phase 3 checkpoint...")
            checkpoint = torch.load(args.resume, map_location='cpu')
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            start_epoch = checkpoint.get('epoch', 0)
            existing_history = checkpoint.get('loss_history', [])
            print(f"[RESUME] Resuming from epoch {start_epoch}")
        else:
            start_epoch, existing_history = load_checkpoint(model, args.resume)
            print(f"[RESUME] Resuming from epoch {start_epoch}")
    elif args.phase == 2 and args.phase1_checkpoint:
        load_checkpoint(model, args.phase1_checkpoint)
    elif args.phase == 3 and args.phase1_checkpoint_for_phase3:
        # Phase 3 V2: Load Phase 1 checkpoint directly (skip Phase 2, no visual tokens)
        print(f"[LOAD] Loading Phase 1 checkpoint for Phase 3 V2 (skipping Phase 2)...")
        checkpoint = torch.load(args.phase1_checkpoint_for_phase3, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint

        # Filter for FAT weights (no visual conditioning weights in Phase 1)
        fat_state = {k: v for k, v in state_dict.items() if 'fat' in k.lower()}
        if fat_state:
            # Load into FAT model (which is inside the wrapper)
            missing, unexpected = model.fat_model.load_state_dict(fat_state, strict=False)
            print(f"[LOAD] Loaded {len(fat_state)} FAT weights from Phase 1 checkpoint")
            print(f"[LOAD] Note: Pose head initialized randomly (will be trained from scratch)")
            if missing:
                print(f"[LOAD] Missing keys (expected for pose head): {len(missing)}")
        else:
            print("[LOAD] No FAT weights found in Phase 1 checkpoint, starting fresh")
    elif args.phase == 3 and args.phase2_checkpoint:
        # Load Phase 2 checkpoint into FAT model (same pattern as phase 2 loading phase 1)
        print(f"[LOAD] Loading Phase 2 checkpoint into FAT model...")
        checkpoint = torch.load(args.phase2_checkpoint, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # Filter for FAT weights
        fat_state = {k: v for k, v in state_dict.items() if 'fat' in k.lower()}
        if fat_state:
            # Load into FAT model (which is inside the wrapper)
            missing, unexpected = model.fat_model.load_state_dict(fat_state, strict=False)
            print(f"[LOAD] Loaded {len(fat_state)} FAT weights from Phase 2 checkpoint")
        else:
            print("[LOAD] No FAT weights found in Phase 2 checkpoint")

    # Move to device (same as phases 1/2 - simple and works)
    model = model.to(device)

    # Verify GPU usage (same pattern as phases 1/2)
    if torch.cuda.is_available():
        print(f"[GPU] Device: {torch.cuda.get_device_name(0)}")
        print(f"[GPU] Memory allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
        print(f"[GPU] Model is on: {next(model.parameters()).device}")

    print(f"\n{'='*60}")
    print(f"[TRAIN] Starting Phase {args.phase} training")
    print(f"[TRAIN] Epochs: {args.num_epochs}, Batch size: {args.batch_size}, LR: {args.learning_rate}")
    print(f"[TRAIN] Max ahead: {args.max_ahead} ({args.max_ahead + 1} frames)")
    print(f"[TRAIN] Visual conditioning: {args.use_visual_conditioning}")
    print(f"[TRAIN] Learnable agg token: {args.use_learnable_agg_token}")
    print(f"{'='*60}\n")

    # Train
    benchmark_samples_per_epoch = getattr(args, 'benchmark_calibration_samples', 50)
    
    if args.phase in [1, 2]:
        if args.v2:
            print("[INFO] Using V2 training with differentiable calibrator")
            train_phase_1_2_v2(
                model, train_loader, val_loader, args, device, save_dir,
                start_epoch, existing_history,
                benchmark_iterator=None,  # No benchmarking for phase 1/2
                benchmark_samples_per_epoch=benchmark_samples_per_epoch,
            )
        else:
            train_phase_1_2(
                model, train_loader, val_loader, args, device, save_dir,
                start_epoch, existing_history,
            )
    else:
        train_phase_3(
            model, train_loader, val_loader, args, device, save_dir,
            start_epoch, existing_history,
            benchmark_iterator=benchmark_iterator,
            benchmark_samples_per_epoch=benchmark_samples_per_epoch,
        )


if __name__ == "__main__":
    main()
