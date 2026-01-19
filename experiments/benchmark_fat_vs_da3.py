#!/usr/bin/env python3
"""
=============================================================================
Benchmark: FAT vs DA3 vs AnyCalib Calibration Accuracy
=============================================================================

Compares calibration accuracy across three approaches:
1. FAT Phase 2: Multi-frame feature aggregation via transformer
2. DA3 Stage 2: Multi-frame calibration via learned head
3. AnyCalib: Single-frame baseline (mean of per-frame predictions)

All evaluated on Objectron dataset with GT intrinsics.

Author: AI Assistant for Kalman's Master's Thesis
Date: January 2026
"""

import sys
import os

# Disable xFormers for GPU compatibility (RTX 5090)
os.environ["XFORMERS_DISABLED"] = "1"
os.environ["XFORMERS_MORE_DETAILS"] = "0"

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import cv2

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

print("[INIT] Imports starting...")


# =============================================================================
# DATASET
# =============================================================================

class CalibrationBenchmarkDataset(Dataset):
    """Dataset for calibration benchmark using multi-frame sequences."""

    def __init__(
        self,
        video_dir: Path,
        gt_dir: Path,
        video_indices: List[int],
        max_ahead: int = 3,
        num_samples: int = 50,
        image_size: Tuple[int, int] = (480, 640),
    ):
        self.video_dir = Path(video_dir)
        self.gt_dir = Path(gt_dir)
        self.max_ahead = max_ahead
        self.num_frames = max_ahead + 1
        self.image_size = image_size

        # Get video files
        all_videos = sorted(list(self.video_dir.glob("*.MOV")) +
                           list(self.video_dir.glob("*.mov")))
        self.video_files = [all_videos[i] for i in video_indices if i < len(all_videos)]

        # Build sequence index with GT
        self._build_sequence_index(num_samples)

    def _build_sequence_index(self, num_samples: int):
        """Build index of sequences that have GT intrinsics."""
        self.sequences = []

        for video_idx, video_path in enumerate(self.video_files):
            # Check for GT
            gt_intr = self._load_gt_intrinsics(video_path)
            if gt_intr is None:
                continue

            cap = cv2.VideoCapture(str(video_path))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

            safe_total = max(0, total_frames - 2)
            if safe_total < self.num_frames:
                continue

            # Sample sequences
            for start in range(0, safe_total - self.num_frames + 1, 10):  # Step by 10
                self.sequences.append({
                    'video_idx': video_idx,
                    'video_path': video_path,
                    'start_frame': start,
                    'frame_indices': list(range(start, start + self.num_frames)),
                    'gt_intrinsics': gt_intr,
                })

                if len(self.sequences) >= num_samples:
                    break

            if len(self.sequences) >= num_samples:
                break

        print(f"[DATASET] Built {len(self.sequences)} sequences with GT")

    def _load_gt_intrinsics(self, video_path: Path) -> Optional[np.ndarray]:
        """Load GT intrinsics from JSON file."""
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

                    if 'intrinsics_per_frame' in data and len(data['intrinsics_per_frame']) > 0:
                        K_flat = data['intrinsics_per_frame'][0]
                        fx, fy = K_flat[0], K_flat[4]
                        cx, cy = K_flat[2], K_flat[5]
                        return np.array([fx, fy, cx, cy], dtype=np.float32)

                    if 'frames' in data and len(data['frames']) > 0:
                        frame = data['frames'][0]
                        if 'intrinsics' in frame:
                            intr = frame['intrinsics']
                            if isinstance(intr, dict):
                                return np.array([intr['fx'], intr['fy'], intr['cx'], intr['cy']], dtype=np.float32)
                except Exception:
                    pass
        return None

    def _load_frames(self, video_path: Path, frame_indices: List[int]) -> np.ndarray:
        """Load frames from video."""
        cap = cv2.VideoCapture(str(video_path))
        frames = []

        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if frame.shape[:2] != self.image_size:
                    frame = cv2.resize(frame, (self.image_size[1], self.image_size[0]))
                frames.append(frame)
            else:
                if frames:
                    frames.append(frames[-1].copy())

        cap.release()
        return np.stack(frames)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict:
        seq = self.sequences[idx]
        frames = self._load_frames(seq['video_path'], seq['frame_indices'])
        imgs = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0

        return {
            'imgs': imgs,
            'gt_intrinsics': torch.from_numpy(seq['gt_intrinsics']),
            'video_idx': seq['video_idx'],
        }


def collate_fn(batch: List[Dict]) -> Dict:
    """Collate function."""
    return {
        'imgs': torch.stack([b['imgs'] for b in batch]),
        'gt_intrinsics': torch.stack([b['gt_intrinsics'] for b in batch]),
        'video_idx': [b['video_idx'] for b in batch],
    }


# =============================================================================
# MODEL LOADING
# =============================================================================

def load_fat_model(checkpoint_path: str, device: torch.device):
    """Load FAT model from checkpoint."""
    from experiments.models.anycalib_with_fat import AnyCalibWithFAT

    print(f"[FAT] Loading from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # Get config from checkpoint
    args = checkpoint.get('args', {})

    fat_config = {
        'embed_dim': args.get('embed_dim', 1024),
        'num_heads': args.get('num_heads', 8),
        'num_layers': args.get('num_layers', 2),
        'use_learnable_agg_token': args.get('use_learnable_agg_token', False),
        'use_visual_conditioning': args.get('use_visual_conditioning', True),
        'visual_token_dim': args.get('visual_token_dim', 384),
    }

    model = AnyCalibWithFAT(
        model_id="anycalib_pinhole",
        use_fat=True,
        fat_config=fat_config,
        use_dinov2_small=True,
        freeze_backbone=True,
        freeze_decoder=True,
    )

    # Load state dict
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()

    print(f"[FAT] Loaded successfully")
    return model


def load_da3_model(checkpoint_path: str, device: torch.device):
    """Load DA3 calibration head."""
    from experiments.models.da3_calibration_head import DA3CalibrationHead

    print(f"[DA3] Loading from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    model = DA3CalibrationHead(
        vis_dim=384,
        cam_dim=256,
        hidden_dim=128,
        num_mixing_layers=2,
    )

    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    # Filter for matching keys
    model_state = model.state_dict()
    filtered = {k: v for k, v in state_dict.items() if k in model_state and v.shape == model_state[k].shape}
    model.load_state_dict(filtered, strict=False)

    model = model.to(device)
    model.eval()

    print(f"[DA3] Loaded successfully ({len(filtered)} keys)")
    return model


def load_anycalib_model(device: torch.device):
    """Load AnyCalib for baseline predictions."""
    try:
        from anycalib.model.anycalib_pretrained import AnyCalib
    except ImportError:
        from anycalib.anycalib.model.anycalib_pretrained import AnyCalib

    model = AnyCalib(model_id="anycalib_pinhole")
    model = model.to(device)
    model.eval()
    print("[AnyCalib] Loaded baseline model")
    return model


# =============================================================================
# EVALUATION
# =============================================================================

def extract_visual_tokens(images: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Extract visual tokens using DINOv2-small."""
    from transformers import AutoModel

    if not hasattr(extract_visual_tokens, 'model'):
        extract_visual_tokens.model = AutoModel.from_pretrained('facebook/dinov2-small').to(device).eval()
        for p in extract_visual_tokens.model.parameters():
            p.requires_grad = False

    B, N, C, H, W = images.shape
    inputs = images.view(B * N, C, H, W)

    # Normalize
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    inputs = (inputs - mean) / std

    if H != 224 or W != 224:
        inputs = F.interpolate(inputs, size=(224, 224), mode='bilinear', align_corners=False)

    with torch.no_grad():
        outputs = extract_visual_tokens.model(inputs)
        cls_tokens = outputs.last_hidden_state[:, 0, :]

    return cls_tokens.view(B, N, -1)


def evaluate_fat(model, dataloader: DataLoader, device: torch.device) -> Dict:
    """Evaluate FAT model on calibration accuracy."""
    model.eval()

    errors = {'fx': [], 'fy': [], 'cx': [], 'cy': []}
    predictions = []
    targets = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating FAT"):
            imgs = batch['imgs'].to(device)  # [B, N, 3, H, W]
            gt = batch['gt_intrinsics'].to(device)  # [B, 4]
            B, N, C, H, W = imgs.shape

            for b in range(B):
                try:
                    # Forward through FAT model
                    result = model(imgs[b], cam_id="pinhole", return_calibration_info=True)
                    pred = result['intrinsics'][0]  # [4]

                    if not isinstance(pred, torch.Tensor):
                        pred = torch.tensor(pred, device=device)

                    pred_np = pred.cpu().numpy()
                    gt_np = gt[b].cpu().numpy()

                    # Compute relative errors
                    rel_err = np.abs(pred_np - gt_np) / (gt_np + 1e-8) * 100

                    errors['fx'].append(rel_err[0])
                    errors['fy'].append(rel_err[1])
                    errors['cx'].append(rel_err[2])
                    errors['cy'].append(rel_err[3])

                    predictions.append(pred_np.tolist())
                    targets.append(gt_np.tolist())

                except Exception as e:
                    print(f"[WARN] FAT eval failed: {e}")
                    continue

    return compute_stats(errors, predictions, targets)


def evaluate_da3(model, anycalib, dataloader: DataLoader, device: torch.device) -> Dict:
    """Evaluate DA3 model on calibration accuracy."""
    model.eval()

    errors = {'fx': [], 'fy': [], 'cx': [], 'cy': []}
    predictions = []
    targets = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating DA3"):
            imgs = batch['imgs'].to(device)
            gt = batch['gt_intrinsics'].to(device)
            B, N, C, H, W = imgs.shape

            for b in range(B):
                try:
                    # Get AnyCalib per-frame predictions
                    anycalib_preds = []
                    for n in range(N):
                        frame = imgs[b, n].unsqueeze(0)  # [1, 3, H, W]
                        calib = anycalib.predict(frame, cam_id="pinhole")
                        # AnyCalib returns intrinsics as a tensor or list
                        intrinsics = calib['intrinsics']
                        if isinstance(intrinsics, list):
                            intrinsics = intrinsics[0]  # Get first (only) batch item
                        if not isinstance(intrinsics, torch.Tensor):
                            intrinsics = torch.tensor(intrinsics, device=device)
                        anycalib_preds.append(intrinsics)

                    anycalib_preds = torch.stack(anycalib_preds, dim=0)  # [N, 4]
                    anycalib_preds = anycalib_preds.unsqueeze(0)  # [1, N, 4]

                    # Get visual tokens
                    visual_tokens = extract_visual_tokens(imgs[b:b+1], device)  # [1, N, 384]

                    # Run DA3
                    pred = model(
                        visual_tokens=visual_tokens,
                        anycalib_predictions=anycalib_preds,
                        image_size=(H, W),
                        use_visual_conditioning=True,
                    )  # [1, 1, 4]

                    pred_np = pred[0, 0].cpu().numpy()
                    gt_np = gt[b].cpu().numpy()

                    rel_err = np.abs(pred_np - gt_np) / (gt_np + 1e-8) * 100

                    errors['fx'].append(rel_err[0])
                    errors['fy'].append(rel_err[1])
                    errors['cx'].append(rel_err[2])
                    errors['cy'].append(rel_err[3])

                    predictions.append(pred_np.tolist())
                    targets.append(gt_np.tolist())

                except Exception as e:
                    print(f"[WARN] DA3 eval failed: {e}")
                    import traceback
                    traceback.print_exc()
                    continue

    return compute_stats(errors, predictions, targets)


def evaluate_anycalib_baseline(anycalib, dataloader: DataLoader, device: torch.device) -> Dict:
    """Evaluate AnyCalib baseline (mean of per-frame predictions)."""
    errors = {'fx': [], 'fy': [], 'cx': [], 'cy': []}
    predictions = []
    targets = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating AnyCalib"):
            imgs = batch['imgs'].to(device)
            gt = batch['gt_intrinsics'].to(device)
            B, N, C, H, W = imgs.shape

            for b in range(B):
                try:
                    # Get per-frame predictions and average
                    anycalib_preds = []
                    for n in range(N):
                        frame = imgs[b, n].unsqueeze(0)
                        calib = anycalib.predict(frame, cam_id="pinhole")
                        # AnyCalib returns intrinsics as a tensor or list
                        intrinsics = calib['intrinsics']
                        if isinstance(intrinsics, list):
                            intrinsics = intrinsics[0]  # Get first (only) batch item
                        if not isinstance(intrinsics, torch.Tensor):
                            intrinsics = torch.tensor(intrinsics, device=device)
                        anycalib_preds.append(intrinsics)

                    anycalib_preds = torch.stack(anycalib_preds, dim=0)  # [N, 4]
                    pred = anycalib_preds.mean(dim=0)  # [4] - mean pooling

                    pred_np = pred.cpu().numpy()
                    gt_np = gt[b].cpu().numpy()

                    rel_err = np.abs(pred_np - gt_np) / (gt_np + 1e-8) * 100

                    errors['fx'].append(rel_err[0])
                    errors['fy'].append(rel_err[1])
                    errors['cx'].append(rel_err[2])
                    errors['cy'].append(rel_err[3])

                    predictions.append(pred_np.tolist())
                    targets.append(gt_np.tolist())

                except Exception as e:
                    print(f"[WARN] AnyCalib eval failed: {e}")
                    import traceback
                    traceback.print_exc()
                    continue

    return compute_stats(errors, predictions, targets)


def compute_stats(errors: Dict, predictions: List, targets: List) -> Dict:
    """Compute error statistics."""
    stats = {}
    for param in ['fx', 'fy', 'cx', 'cy']:
        if errors[param]:
            arr = np.array(errors[param])
            stats[param] = {
                'mean': float(np.mean(arr)),
                'median': float(np.median(arr)),
                'std': float(np.std(arr)),
                'p90': float(np.percentile(arr, 90)),
            }
        else:
            stats[param] = {'mean': float('inf'), 'median': float('inf'), 'std': 0, 'p90': float('inf')}

    # Overall
    all_errors = []
    for param in ['fx', 'fy', 'cx', 'cy']:
        all_errors.extend(errors[param])

    if all_errors:
        stats['overall'] = {
            'mean': float(np.mean(all_errors)),
            'median': float(np.median(all_errors)),
        }
    else:
        stats['overall'] = {'mean': float('inf'), 'median': float('inf')}

    return {
        'errors': errors,
        'statistics': stats,
        'predictions': predictions,
        'targets': targets,
        'num_samples': len(predictions),
    }


# =============================================================================
# PLOTTING
# =============================================================================

def plot_comparison(results: Dict, save_dir: Path):
    """Plot comparison of all methods."""
    methods = list(results.keys())
    params = ['fx', 'fy', 'cx', 'cy']
    param_names = {'fx': 'fx', 'fy': 'fy', 'cx': 'cx', 'cy': 'cy'}

    # Bar chart comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Calibration Error Comparison: FAT vs DA3 vs AnyCalib', fontsize=16, fontweight='bold')

    x = np.arange(len(methods))
    width = 0.35

    for idx, param in enumerate(params):
        ax = axes[idx // 2, idx % 2]

        means = [results[m]['statistics'][param]['mean'] for m in methods]
        medians = [results[m]['statistics'][param]['median'] for m in methods]

        ax.bar(x - width/2, means, width, label='Mean', alpha=0.8, color='blue')
        ax.bar(x + width/2, medians, width, label='Median', alpha=0.8, color='orange')

        ax.set_xlabel('Method', fontsize=11)
        ax.set_ylabel('Relative Error (%)', fontsize=11)
        ax.set_title(f'{param_names[param]}', fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(methods)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(save_dir / 'calibration_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Overall comparison table
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('off')

    table_data = []
    for method in methods:
        stats = results[method]['statistics']
        row = [
            method,
            f"{stats['fx']['mean']:.2f}",
            f"{stats['fy']['mean']:.2f}",
            f"{stats['cx']['mean']:.2f}",
            f"{stats['cy']['mean']:.2f}",
            f"{stats['overall']['mean']:.2f}",
        ]
        table_data.append(row)

    table = ax.table(
        cellText=table_data,
        colLabels=['Method', 'fx (%)', 'fy (%)', 'cx (%)', 'cy (%)', 'Overall (%)'],
        loc='center',
        cellLoc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.5)

    ax.set_title('Mean Relative Error by Parameter', fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(save_dir / 'calibration_summary_table.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"[PLOT] Saved comparison plots to {save_dir}")


def generate_report(results: Dict, save_dir: Path):
    """Generate text report."""
    report_path = save_dir / 'benchmark_report.txt'

    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("FAT vs DA3 vs AnyCalib Calibration Benchmark\n")
        f.write("="*80 + "\n\n")

        for method, data in results.items():
            f.write(f"\n{method.upper()}\n")
            f.write("-"*40 + "\n")
            f.write(f"Samples evaluated: {data['num_samples']}\n\n")

            stats = data['statistics']
            f.write(f"{'Parameter':<12} {'Mean':>10} {'Median':>10} {'Std':>10} {'P90':>10}\n")
            f.write("-"*52 + "\n")

            for param in ['fx', 'fy', 'cx', 'cy']:
                s = stats[param]
                f.write(f"{param:<12} {s['mean']:>10.2f} {s['median']:>10.2f} {s['std']:>10.2f} {s['p90']:>10.2f}\n")

            f.write(f"\nOverall Mean: {stats['overall']['mean']:.2f}%\n")
            f.write(f"Overall Median: {stats['overall']['median']:.2f}%\n")

        f.write("\n" + "="*80 + "\n")
        f.write("SUMMARY\n")
        f.write("="*80 + "\n\n")

        # Find best method
        overall_means = {m: results[m]['statistics']['overall']['mean'] for m in results}
        best = min(overall_means, key=overall_means.get)
        f.write(f"Best performing method: {best} ({overall_means[best]:.2f}% mean error)\n")

    print(f"[REPORT] Saved to {report_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Benchmark FAT vs DA3 vs AnyCalib")

    parser.add_argument("--fat_checkpoint", type=str, required=True,
                       help="Path to FAT phase 2 checkpoint")
    parser.add_argument("--da3_checkpoint", type=str, required=True,
                       help="Path to DA3 stage 2 checkpoint")
    parser.add_argument("--objectron_videos", type=str, required=True,
                       help="Path to Objectron videos")
    parser.add_argument("--objectron_gt", type=str, required=True,
                       help="Path to Objectron GT")
    parser.add_argument("--num_samples", type=int, default=50,
                       help="Number of samples to evaluate")
    parser.add_argument("--max_ahead", type=int, default=3,
                       help="Frames ahead for multi-frame methods")
    parser.add_argument("--save_dir", type=str, required=True,
                       help="Directory to save results")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--split_file", type=str, default="experiments/objectron_split.json")

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Load split
    split_path = Path(args.split_file)
    if split_path.exists():
        with open(split_path, 'r') as f:
            split = json.load(f)
        test_indices = split.get('test', list(range(85, 100)))
    else:
        test_indices = list(range(85, 100))

    print(f"[DATA] Using {len(test_indices)} test videos")

    # Create dataset
    dataset = CalibrationBenchmarkDataset(
        video_dir=args.objectron_videos,
        gt_dir=args.objectron_gt,
        video_indices=test_indices,
        max_ahead=args.max_ahead,
        num_samples=args.num_samples,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    print(f"[DATA] Loaded {len(dataset)} samples")

    # Load models
    print("\n[LOAD] Loading models...")

    fat_model = load_fat_model(args.fat_checkpoint, device)
    da3_model = load_da3_model(args.da3_checkpoint, device)
    anycalib = load_anycalib_model(device)

    # Evaluate
    print("\n[EVAL] Starting evaluation...")

    results = {}

    print("\n--- Evaluating FAT Phase 2 ---")
    results['FAT'] = evaluate_fat(fat_model, dataloader, device)
    torch.cuda.empty_cache()

    print("\n--- Evaluating DA3 Stage 2 ---")
    results['DA3'] = evaluate_da3(da3_model, anycalib, dataloader, device)
    torch.cuda.empty_cache()

    print("\n--- Evaluating AnyCalib Baseline ---")
    results['AnyCalib'] = evaluate_anycalib_baseline(anycalib, dataloader, device)
    torch.cuda.empty_cache()

    # Save results
    with open(save_dir / 'benchmark_results.json', 'w') as f:
        # Remove raw errors for JSON (too large)
        results_json = {}
        for method, data in results.items():
            results_json[method] = {
                'statistics': data['statistics'],
                'num_samples': data['num_samples'],
            }
        json.dump(results_json, f, indent=2)

    # Generate plots and report
    plot_comparison(results, save_dir)
    generate_report(results, save_dir)

    # Print summary
    print("\n" + "="*60)
    print("BENCHMARK COMPLETE")
    print("="*60)

    for method in ['FAT', 'DA3', 'AnyCalib']:
        stats = results[method]['statistics']
        print(f"{method:10}: Overall Mean={stats['overall']['mean']:.2f}%, Median={stats['overall']['median']:.2f}%")

    print(f"\nResults saved to: {save_dir}")


if __name__ == "__main__":
    main()
