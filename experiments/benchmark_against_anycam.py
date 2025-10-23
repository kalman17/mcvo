#!/usr/bin/env python3
"""
Automatic Benchmarking Script: Compare Trained Model vs AnyCam Baseline

This script takes a trained model checkpoint and automatically:
1. Loads your trained model
2. Loads the pretrained AnyCam baseline
3. Evaluates both on the same test set with ground truth poses
4. Generates clear comparison visualizations and metrics

Usage:
    python experiments/benchmark_against_anycam.py \
        --trained_model experiments/pose_head_experiment_results/full_run_eval/final_model.pt \
        --dataset lightspeed

Author: AI Assistant for Kalman's Master's Thesis
Date: October 17, 2025
"""

import sys
import os
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add project paths
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "anycalib"))

# Import from training script
from experiments.train_pose_head_anycalib import (
    ObjectronVideoDataset,
    AnyCamWrapperWithAnyCaLib,
    load_dataset_split,
)
from experiments.lightspeed_dataset import LightSpeedDataset
from experiments.pose_metrics import (
    rotation_error_degrees,
    translation_direction_error_degrees,
    compute_error_statistics,
)

# AnyCam imports
from anycam.models import make_pose_predictor, make_depth_predictor, make_depth_aligner
from anycam.common.image_processor import make_image_processor

# AnyCaLib import
try:
    from anycalib.model.anycalib_pretrained import AnyCalib
except ImportError:
    from anycalib.anycalib.model.anycalib_pretrained import AnyCalib

print("[BENCHMARK] Imports successful")


# =============================================================================
# POSE EVALUATION FUNCTIONS
# =============================================================================

def evaluate_model_on_dataset(model, dataloader, device, model_name="Model"):
    """
    Evaluate a model on a dataset with ground truth poses.
    
    Returns:
        rot_errors: List of rotation errors in degrees
        trans_errors: List of translation direction errors in degrees
    """
    model.eval()
    rot_errors = []
    trans_errors = []
    
    print(f"\n[EVAL] Evaluating {model_name}...")
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(tqdm(dataloader, desc=f"Evaluating {model_name}")):
            # Move data to device
            batch_data = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                         for k, v in batch_data.items()}
            
            # Get ground truth poses
            if 'poses' not in batch_data:
                print(f"[WARN] No GT poses in batch {batch_idx}, skipping")
                continue
                
            gt_poses = batch_data['poses']  # [batch, num_frames, 4, 4]
            
            # Compute ground truth relative poses
            batch_size, num_frames = gt_poses.shape[0], gt_poses.shape[1]
            gt_rel_poses = []
            for i in range(num_frames - 1):
                pose1 = gt_poses[:, i]  # [batch, 4, 4]
                pose2 = gt_poses[:, i + 1]  # [batch, 4, 4]
                rel_pose = torch.linalg.inv(pose1) @ pose2  # T_1->2
                gt_rel_poses.append(rel_pose)
            gt_rel_poses = torch.stack(gt_rel_poses, dim=1)  # [batch, num_pairs, 4, 4]
            
            # Get model predictions
            try:
                output = model(batch_data)
                # Use proc_poses - this contains the selected poses after candidate filtering
                pred_poses = output['proc_poses']  # [batch, num_pairs, 4, 4]
                
                # Compute errors
                for b in range(batch_size):
                    num_pairs = min(pred_poses.shape[1] if len(pred_poses.shape) > 1 else 1, 
                                   gt_rel_poses.shape[1])
                    for p in range(num_pairs):
                        # Extract poses as numpy arrays
                        pred_pose_np = pred_poses[b, p].cpu().numpy() if len(pred_poses.shape) > 2 else pred_poses[b].cpu().numpy()
                        gt_pose_np = gt_rel_poses[b, p].cpu().numpy()
                        
                        # Calculate errors (pass full 4x4 poses)
                        rot_err = rotation_error_degrees(
                            pred_pose_np[:3, :3],  # Extract 3x3 rotation
                            gt_pose_np[:3, :3]
                        )
                        trans_err = translation_direction_error_degrees(
                            pred_pose_np[:3, 3],  # Extract 3D translation
                            gt_pose_np[:3, 3]
                        )
                        
                        # Only append valid errors (scalar values)
                        if not np.isnan(rot_err) and not np.isnan(trans_err):
                            rot_errors.append(float(rot_err))
                            trans_errors.append(float(trans_err))
                            
            except Exception as e:
                print(f"[ERROR] Failed on batch {batch_idx}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    return np.array(rot_errors), np.array(trans_errors)


# =============================================================================
# VISUALIZATION AND REPORTING
# =============================================================================

def plot_comparison(our_rot, our_trans, baseline_rot, baseline_trans, save_dir):
    """Generate comparison plots."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Rotation error histogram
    axes[0, 0].hist(our_rot, bins=50, alpha=0.7, label='Our Model', color='blue')
    axes[0, 0].hist(baseline_rot, bins=50, alpha=0.7, label='AnyCam Baseline', color='orange')
    axes[0, 0].set_xlabel('Rotation Error (degrees)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Rotation Error Distribution')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Translation error histogram
    axes[0, 1].hist(our_trans, bins=50, alpha=0.7, label='Our Model', color='blue')
    axes[0, 1].hist(baseline_trans, bins=50, alpha=0.7, label='AnyCam Baseline', color='orange')
    axes[0, 1].set_xlabel('Translation Direction Error (degrees)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Translation Error Distribution')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # CDF for rotation
    axes[0, 2].hist(our_rot, bins=100, cumulative=True, density=True, 
                    histtype='step', linewidth=2, label='Our Model', color='blue')
    axes[0, 2].hist(baseline_rot, bins=100, cumulative=True, density=True,
                    histtype='step', linewidth=2, label='AnyCam Baseline', color='orange')
    axes[0, 2].set_xlabel('Rotation Error (degrees)')
    axes[0, 2].set_ylabel('Cumulative Probability')
    axes[0, 2].set_title('Rotation Error CDF')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # CDF for translation
    axes[1, 0].hist(our_trans, bins=100, cumulative=True, density=True,
                    histtype='step', linewidth=2, label='Our Model', color='blue')
    axes[1, 0].hist(baseline_trans, bins=100, cumulative=True, density=True,
                    histtype='step', linewidth=2, label='AnyCam Baseline', color='orange')
    axes[1, 0].set_xlabel('Translation Direction Error (degrees)')
    axes[1, 0].set_ylabel('Cumulative Probability')
    axes[1, 0].set_title('Translation Error CDF')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Bar chart comparison - Rotation
    metrics = ['Mean', 'Median', 'P90']
    our_rot_metrics = [np.mean(our_rot), np.median(our_rot), np.percentile(our_rot, 90)]
    baseline_rot_metrics = [np.mean(baseline_rot), np.median(baseline_rot), np.percentile(baseline_rot, 90)]
    
    x = np.arange(len(metrics))
    width = 0.35
    axes[1, 1].bar(x - width/2, our_rot_metrics, width, label='Our Model', color='blue', alpha=0.7)
    axes[1, 1].bar(x + width/2, baseline_rot_metrics, width, label='AnyCam Baseline', color='orange', alpha=0.7)
    axes[1, 1].set_ylabel('Rotation Error (degrees)')
    axes[1, 1].set_title('Rotation Error Comparison')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(metrics)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    # Bar chart comparison - Translation
    our_trans_metrics = [np.mean(our_trans), np.median(our_trans), np.percentile(our_trans, 90)]
    baseline_trans_metrics = [np.mean(baseline_trans), np.median(baseline_trans), np.percentile(baseline_trans, 90)]
    
    axes[1, 2].bar(x - width/2, our_trans_metrics, width, label='Our Model', color='blue', alpha=0.7)
    axes[1, 2].bar(x + width/2, baseline_trans_metrics, width, label='AnyCam Baseline', color='orange', alpha=0.7)
    axes[1, 2].set_ylabel('Translation Direction Error (degrees)')
    axes[1, 2].set_title('Translation Error Comparison')
    axes[1, 2].set_xticks(x)
    axes[1, 2].set_xticklabels(metrics)
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_dir / 'benchmark_comparison.png', dpi=150, bbox_inches='tight')
    print(f"[SAVE] Comparison plots saved to {save_dir / 'benchmark_comparison.png'}")
    plt.close()


def generate_report(our_rot_stats, our_trans_stats, baseline_rot_stats, baseline_trans_stats, 
                   num_samples, save_path):
    """Generate text report comparing models."""
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("BENCHMARK: Trained Model vs AnyCam Baseline\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Test Set Size: {num_samples} frame pairs\n\n")
        
        f.write("ROTATION ERROR (degrees)\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Metric':<20} {'Our Model':>15} {'AnyCam Baseline':>20} {'Improvement':>15}\n")
        f.write("-" * 80 + "\n")
        
        for metric in ['mean', 'median', 'std', 'p90']:
            our_val = our_rot_stats[metric]
            baseline_val = baseline_rot_stats[metric]
            improvement = ((baseline_val - our_val) / baseline_val * 100) if baseline_val != 0 else 0
            f.write(f"{metric.upper():<20} {our_val:>15.4f} {baseline_val:>20.4f} {improvement:>14.2f}%\n")
        
        f.write("\n")
        f.write("TRANSLATION DIRECTION ERROR (degrees)\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Metric':<20} {'Our Model':>15} {'AnyCam Baseline':>20} {'Improvement':>15}\n")
        f.write("-" * 80 + "\n")
        
        for metric in ['mean', 'median', 'std', 'p90']:
            our_val = our_trans_stats[metric]
            baseline_val = baseline_trans_stats[metric]
            improvement = ((baseline_val - our_val) / baseline_val * 100) if baseline_val != 0 else 0
            f.write(f"{metric.upper():<20} {our_val:>15.4f} {baseline_val:>20.4f} {improvement:>14.2f}%\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("SUMMARY\n")
        f.write("="*80 + "\n")
        
        rot_improvement = ((baseline_rot_stats['mean'] - our_rot_stats['mean']) / 
                          baseline_rot_stats['mean'] * 100)
        trans_improvement = ((baseline_trans_stats['mean'] - our_trans_stats['mean']) / 
                            baseline_trans_stats['mean'] * 100)
        
        if rot_improvement > 0:
            f.write(f"✓ Rotation error improved by {rot_improvement:.2f}%\n")
        else:
            f.write(f"✗ Rotation error degraded by {abs(rot_improvement):.2f}%\n")
            
        if trans_improvement > 0:
            f.write(f"✓ Translation error improved by {trans_improvement:.2f}%\n")
        else:
            f.write(f"✗ Translation error degraded by {abs(trans_improvement):.2f}%\n")
    
    print(f"[SAVE] Report saved to {save_path}")


# =============================================================================
# MAIN BENCHMARKING FUNCTION
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Benchmark Trained Model Against AnyCam Baseline")
    parser.add_argument("--trained_model", type=str, required=True,
                       help="Path to trained model checkpoint (.pt file)")
    parser.add_argument("--baseline_checkpoint", type=str,
                       default="pretrained_models/anycam_seq8/training_checkpoint_247500.pt",
                       help="Path to baseline AnyCam checkpoint")
    parser.add_argument("--dataset", type=str, choices=['objectron', 'lightspeed'], default='lightspeed',
                       help="Which dataset to use for evaluation")
    parser.add_argument("--objectron_videos", type=str,
                       default="/home/kalman/TUM/thesis/Objectron/videos/",
                       help="Objectron videos directory")
    parser.add_argument("--objectron_gt", type=str,
                       default="/home/kalman/TUM/thesis/Objectron/annotations/",
                       help="Objectron GT directory")
    parser.add_argument("--lightspeed_dir", type=str,
                       default="/home/kalman/TUM/thesis/dynpose-100k/lightspeed/",
                       help="LightSpeed dataset directory")
    parser.add_argument("--split_file", type=str,
                       default="experiments/objectron_split.json",
                       help="Dataset split file (for Objectron)")
    parser.add_argument("--save_dir", type=str,
                       default=None,
                       help="Directory to save results (default: same as trained model)")
    parser.add_argument("--batch_size", type=int, default=2,
                       help="Batch size for evaluation")
    parser.add_argument("--device", type=str, default="cuda:0",
                       help="Device to use")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="Maximum number of samples to evaluate (for faster testing)")
    
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[BENCHMARK] Using device: {device}")
    
    # Determine save directory
    if args.save_dir is None:
        trained_model_path = Path(args.trained_model)
        args.save_dir = trained_model_path.parent / "benchmark_results"
    
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"[BENCHMARK] Results will be saved to: {save_dir}")
    
    # =============================================================================
    # Load Test Dataset
    # =============================================================================
    print(f"\n[STEP 1] Loading {args.dataset} test dataset...")
    
    if args.dataset == 'objectron':
        # Load dataset split
        split_data = load_dataset_split(args.split_file)
        test_indices = split_data['test_indices']
        
        test_dataset = ObjectronVideoDataset(
            videos_dir=args.objectron_videos,
            gt_dir=args.objectron_gt,
            num_frames=2,
            video_indices=test_indices,
            require_gt=True,
            extract_all_pairs=False,  # One pair per video for evaluation
        )
    else:  # lightspeed
        test_dataset = LightSpeedDataset(
            lightspeed_dir=args.lightspeed_dir,
            num_frames=2,
            image_size=(480, 640),
        )
    
    # Limit dataset size if requested
    if args.max_samples is not None and args.max_samples < len(test_dataset):
        from torch.utils.data import Subset
        import numpy as np
        indices = np.arange(min(args.max_samples, len(test_dataset)))
        test_dataset = Subset(test_dataset, indices)
        print(f"[DATASET] Limited to {len(test_dataset)} samples (--max_samples={args.max_samples})")
    
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,  # Avoid multiprocessing issues
        pin_memory=False,
    )
    
    print(f"[DATASET] Loaded {len(test_dataset)} test samples")
    
    # =============================================================================
    # Load Trained Model
    # =============================================================================
    print(f"\n[STEP 2] Loading trained model from {args.trained_model}...")
    
    # Load config from baseline checkpoint
    import yaml
    baseline_config_path = Path(args.baseline_checkpoint).parent / "training_config.yaml"
    if not baseline_config_path.exists():
        raise FileNotFoundError(f"Config file not found: {baseline_config_path}")
    
    with open(baseline_config_path, 'r') as f:
        full_config = yaml.safe_load(f)
    
    # Extract model configs
    pose_predictor_config = full_config['model']['pose_predictor']
    depth_predictor_config = full_config['model']['depth_predictor']
    
    # Create AnyCaLibBatchInference wrapper
    from experiments.train_pose_head_anycalib import AnyCaLibBatchInference
    anycalib_inference = AnyCaLibBatchInference(device=device)
    
    # The trained model was saved with the original architecture (32 candidates),
    # even though it uses AnyCaLib. We need to match that architecture.
    trained_model = AnyCamWrapperWithAnyCaLib(
        pose_predictor_config=pose_predictor_config,  # Use original config
        depth_predictor_config=depth_predictor_config,
        anycalib_model=anycalib_inference,
    )
    
    # Move model to device
    trained_model = trained_model.to(device)
    
    # Load checkpoint
    checkpoint = torch.load(args.trained_model, map_location=device)
    if 'model_state_dict' in checkpoint:
        trained_model.load_state_dict(checkpoint['model_state_dict'])
        print(f"[MODEL] Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    else:
        trained_model.load_state_dict(checkpoint)
    
    trained_model.eval()
    print("[MODEL] Trained model loaded successfully")
    
    # =============================================================================
    # Load Baseline AnyCam
    # =============================================================================
    print(f"\n[STEP 3] Loading baseline AnyCam from {args.baseline_checkpoint}...")
    
    # Create baseline model using the SAME wrapper as trained model
    # This ensures consistent behavior - we just load different weights
    baseline_anycalib = AnyCaLibBatchInference(device=device)
    
    baseline_model = AnyCamWrapperWithAnyCaLib(
        pose_predictor_config=pose_predictor_config,
        depth_predictor_config=depth_predictor_config,
        anycalib_model=baseline_anycalib,
    )
    baseline_model = baseline_model.to(device)
    
    # Load PRETRAINED checkpoint (not our trained one)
    baseline_checkpoint = torch.load(args.baseline_checkpoint, map_location=device)
    if 'model' in baseline_checkpoint:
        baseline_checkpoint_data = baseline_checkpoint['model']
    else:
        baseline_checkpoint_data = baseline_checkpoint
    
    # Load the full model state dict from pretrained checkpoint
    baseline_model.pose_predictor.load_state_dict(baseline_checkpoint_data, strict=False)
    
    baseline_model.eval()
    print("[MODEL] Baseline AnyCam loaded successfully")
    
    # =============================================================================
    # Run Evaluation on Both Models
    # =============================================================================
    print("\n[STEP 4] Evaluating models...")
    
    our_rot_errors, our_trans_errors = evaluate_model_on_dataset(
        trained_model, test_dataloader, device, "Trained Model (AnyCaLib)"
    )
    
    baseline_rot_errors, baseline_trans_errors = evaluate_model_on_dataset(
        baseline_model, test_dataloader, device, "AnyCam Baseline"
    )
    
    # =============================================================================
    # Compute Statistics and Generate Report
    # =============================================================================
    print("\n[STEP 5] Computing statistics and generating report...")
    
    # Check if we have any valid results
    if len(our_rot_errors) == 0 or len(baseline_rot_errors) == 0:
        print(f"\n[ERROR] No valid pose comparisons were made!")
        print(f"  Trained model errors collected: {len(our_rot_errors)}")
        print(f"  Baseline model errors collected: {len(baseline_rot_errors)}")
        print(f"\nThis likely means the pose prediction or GT loading failed.")
        return
    
    print(f"[STATS] Collected {len(our_rot_errors)} errors from trained model")
    print(f"[STATS] Collected {len(baseline_rot_errors)} errors from baseline model")
    
    our_rot_stats = compute_error_statistics(our_rot_errors)
    our_trans_stats = compute_error_statistics(our_trans_errors)
    baseline_rot_stats = compute_error_statistics(baseline_rot_errors)
    baseline_trans_stats = compute_error_statistics(baseline_trans_errors)
    
    # Save results to JSON
    results = {
        'trained_model': {
            'rotation': our_rot_stats,
            'translation': our_trans_stats,
            'num_samples': len(our_rot_errors),
        },
        'baseline_model': {
            'rotation': baseline_rot_stats,
            'translation': baseline_trans_stats,
            'num_samples': len(baseline_rot_errors),
        },
    }
    
    with open(save_dir / 'benchmark_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Generate plots
    plot_comparison(
        our_rot_errors, our_trans_errors,
        baseline_rot_errors, baseline_trans_errors,
        save_dir
    )
    
    # Generate text report
    generate_report(
        our_rot_stats, our_trans_stats,
        baseline_rot_stats, baseline_trans_stats,
        len(our_rot_errors),
        save_dir / 'benchmark_report.txt'
    )
    
    # Print summary to console
    print("\n" + "="*80)
    print("BENCHMARK RESULTS SUMMARY")
    print("="*80)
    print(f"\nTest samples: {len(our_rot_errors)}")
    print(f"\nRotation Error (degrees):")
    print(f"  Our Model:        Mean={our_rot_stats['mean']:.4f}, Median={our_rot_stats['median']:.4f}")
    print(f"  AnyCam Baseline:  Mean={baseline_rot_stats['mean']:.4f}, Median={baseline_rot_stats['median']:.4f}")
    print(f"\nTranslation Error (degrees):")
    print(f"  Our Model:        Mean={our_trans_stats['mean']:.4f}, Median={our_trans_stats['median']:.4f}")
    print(f"  AnyCam Baseline:  Mean={baseline_trans_stats['mean']:.4f}, Median={baseline_trans_stats['median']:.4f}")
    print("\n" + "="*80)
    print(f"\nResults saved to: {save_dir}")
    print("  - benchmark_results.json (detailed metrics)")
    print("  - benchmark_comparison.png (visualization)")
    print("  - benchmark_report.txt (full text report)")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()

