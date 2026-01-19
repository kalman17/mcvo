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
    --save_dir experiments/fat_integration/phase1_training

# Phase 2
python experiments/train_fat_calibration.py --phase 2 --phase1_checkpoint <path> \\
    --objectron_videos /path/to/videos --objectron_gt /path/to/gt \\
    --use_visual_conditioning --max_ahead 3 --num_epochs 50 \\
    --save_dir experiments/fat_integration/phase2_training

# Phase 3
python experiments/train_fat_calibration.py --phase 3 --phase2_checkpoint <path> \\
    --objectron_videos /path/to/videos --max_ahead 3 \\
    --benchmark_samples 100 --benchmark_no_cycle --num_epochs 50 \\
    --save_dir experiments/fat_integration/phase3_training

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
    parser.add_argument("--benchmark_no_cycle", action="store_true",
                        help="Use fixed benchmark samples (no cycling)")
    parser.add_argument("--disable_benchmark", action="store_true",
                        help="Disable pose benchmarking")

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
        'use_visual_conditioning': True,
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

        return result


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


def train_phase_3(
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
    Phase 3 training: End-to-end with flow reprojection loss.

    This integrates FAT into the full AnyCam pipeline.
    For now, uses V2 training with calibration benchmarking.
    """
    print("[INFO] Phase 3 training - using V2 calibration loss with benchmarking")
    print("[INFO] For full phase 3, AnyCam pipeline integration is required")

    # Use V2 training with benchmarking enabled
    train_phase_1_2_v2(
        model, train_loader, val_loader, args, device, save_dir,
        start_epoch, existing_history,
        benchmark_iterator=benchmark_iterator,
        benchmark_samples_per_epoch=benchmark_samples_per_epoch,
    )


# =============================================================================
# MAIN
# =============================================================================

def main():
    args = parse_args()

    # Set device
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
    if args.phase == 3:
        test_dataset = ObjectronFATDataset(
            video_dir=video_dir,
            gt_dir=gt_dir,
            video_indices=split.get('test', split['val']),  # Use test split if available, else val
            max_ahead=args.max_ahead,
            phase=args.phase,
        )
        benchmark_iterator = CalibrationBenchmarkIterator(
            dataset=test_dataset,
            num_samples=getattr(args, 'benchmark_samples_per_epoch', 50),
            no_cycle=True,
        )
        print(f"[BENCHMARK] Initialized with {len(test_dataset)} test sequences")

    print(f"[DATA] Train: {len(train_dataset)} sequences, Val: {len(val_dataset)} sequences")

    # Create model
    model = create_fat_model(args, args.phase, device)

    # Load checkpoint if specified
    start_epoch = 0
    existing_history = None

    if args.resume:
        start_epoch, existing_history = load_checkpoint(model, args.resume)
        print(f"[RESUME] Resuming from epoch {start_epoch}")
    elif args.phase == 2 and args.phase1_checkpoint:
        load_checkpoint(model, args.phase1_checkpoint)
    elif args.phase == 3 and args.phase2_checkpoint:
        load_checkpoint(model, args.phase2_checkpoint)

    # Move to device
    model = model.to(device)

    # Verify GPU usage
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
