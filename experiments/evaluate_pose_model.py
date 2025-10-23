#!/usr/bin/env python3
"""
Pose Model Evaluation Script

Compares trained pose estimation model against pretrained AnyCam baseline
on Objectron test set. Computes rotation and translation direction errors.

Author: AI Assistant for Kalman's Master's Thesis
Date: October 16, 2025
"""

import sys
import os
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project paths
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "anycalib"))

# Import pose metrics
from experiments.pose_metrics import (
    rotation_error_degrees,
    translation_direction_error_degrees,
    pose_error,
    batch_pose_errors,
    compute_error_statistics,
    relative_pose_from_absolute,
)

print("[EVAL] Imports successful")


# =============================================================================
# POSE EXTRACTION UTILITIES
# =============================================================================

def extract_relative_poses_from_output(output_data: Dict, use_gt: bool = False) -> torch.Tensor:
    """
    Extract relative poses from model output or ground truth.
    
    Args:
        output_data: Dictionary containing model predictions or GT
        use_gt: If True, extract from 'poses' (GT), else from 'pose_result'
        
    Returns:
        Tensor of relative poses [batch, num_pairs, 4, 4]
    """
    if use_gt:
        # Extract from ground truth
        poses_abs = output_data['poses']  # [batch, num_frames, 4, 4]
    else:
        # Extract from predictions
        pose_result = output_data['pose_result']
        poses_rel = pose_result['poses']  # [batch, num_pairs, num_candidates, 4, 4]
        
        # Take first candidate (or average across candidates)
        if poses_rel.dim() == 5:
            poses_rel = poses_rel[:, :, 0]  # [batch, num_pairs, 4, 4]
        
        return poses_rel
    
    # Compute relative poses from absolute poses
    batch_size, num_frames, _, _ = poses_abs.shape
    num_pairs = num_frames - 1
    
    poses_rel = torch.zeros(batch_size, num_pairs, 4, 4, device=poses_abs.device)
    
    for i in range(num_pairs):
        pose1 = poses_abs[:, i]  # [batch, 4, 4]
        pose2 = poses_abs[:, i + 1]  # [batch, 4, 4]
        
        # Compute relative pose: T_rel = inv(pose2) @ pose1
        poses_rel[:, i] = torch.inverse(pose2) @ pose1
    
    return poses_rel


def compute_pose_errors_batch(poses_pred: torch.Tensor, poses_gt: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute rotation and translation errors for a batch of poses.
    
    Args:
        poses_pred: Predicted poses [batch, num_pairs, 4, 4]
        poses_gt: Ground truth poses [batch, num_pairs, 4, 4]
        
    Returns:
        Tuple of (rot_errors, trans_errors) as numpy arrays
    """
    poses_pred_np = poses_pred.detach().cpu().numpy()
    poses_gt_np = poses_gt.detach().cpu().numpy()
    
    batch_size, num_pairs = poses_pred_np.shape[:2]
    
    rot_errors = []
    trans_errors = []
    
    for b in range(batch_size):
        for p in range(num_pairs):
            rot_err, trans_err = pose_error(poses_pred_np[b, p], poses_gt_np[b, p])
            rot_errors.append(rot_err)
            trans_errors.append(trans_err)
    
    return np.array(rot_errors), np.array(trans_errors)


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_error_distributions(
    our_rot_errors: np.ndarray,
    our_trans_errors: np.ndarray,
    baseline_rot_errors: np.ndarray,
    baseline_trans_errors: np.ndarray,
    save_dir: Path,
):
    """Generate comparison plots for error distributions."""
    
    # Plot 1: Rotation error histogram
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(our_rot_errors, bins=50, alpha=0.6, label='Our Model', color='blue', edgecolor='black')
    plt.hist(baseline_rot_errors, bins=50, alpha=0.6, label='AnyCam Baseline', color='red', edgecolor='black')
    plt.xlabel('Rotation Error (degrees)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Rotation Error Distribution', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Translation error histogram
    plt.subplot(1, 2, 2)
    plt.hist(our_trans_errors, bins=50, alpha=0.6, label='Our Model', color='blue', edgecolor='black')
    plt.hist(baseline_trans_errors, bins=50, alpha=0.6, label='AnyCam Baseline', color='red', edgecolor='black')
    plt.xlabel('Translation Direction Error (degrees)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Translation Direction Error Distribution', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'error_histograms.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Plot 3: CDF curves
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    our_rot_sorted = np.sort(our_rot_errors)
    baseline_rot_sorted = np.sort(baseline_rot_errors)
    our_rot_cdf = np.arange(1, len(our_rot_sorted) + 1) / len(our_rot_sorted)
    baseline_rot_cdf = np.arange(1, len(baseline_rot_sorted) + 1) / len(baseline_rot_sorted)
    plt.plot(our_rot_sorted, our_rot_cdf, label='Our Model', linewidth=2, color='blue')
    plt.plot(baseline_rot_sorted, baseline_rot_cdf, label='AnyCam Baseline', linewidth=2, color='red')
    plt.xlabel('Rotation Error (degrees)', fontsize=12)
    plt.ylabel('Cumulative Probability', fontsize=12)
    plt.title('Rotation Error CDF', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    our_trans_sorted = np.sort(our_trans_errors)
    baseline_trans_sorted = np.sort(baseline_trans_errors)
    our_trans_cdf = np.arange(1, len(our_trans_sorted) + 1) / len(our_trans_sorted)
    baseline_trans_cdf = np.arange(1, len(baseline_trans_sorted) + 1) / len(baseline_trans_sorted)
    plt.plot(our_trans_sorted, our_trans_cdf, label='Our Model', linewidth=2, color='blue')
    plt.plot(baseline_trans_sorted, baseline_trans_cdf, label='AnyCam Baseline', linewidth=2, color='red')
    plt.xlabel('Translation Direction Error (degrees)', fontsize=12)
    plt.ylabel('Cumulative Probability', fontsize=12)
    plt.title('Translation Direction Error CDF', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'error_cdfs.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Plot 4: Bar chart comparison
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    
    metrics = ['Mean', 'Median', 'Std']
    our_rot_stats = [np.mean(our_rot_errors), np.median(our_rot_errors), np.std(our_rot_errors)]
    baseline_rot_stats = [np.mean(baseline_rot_errors), np.median(baseline_rot_errors), np.std(baseline_rot_errors)]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    ax[0].bar(x - width/2, our_rot_stats, width, label='Our Model', color='blue', alpha=0.7)
    ax[0].bar(x + width/2, baseline_rot_stats, width, label='AnyCam Baseline', color='red', alpha=0.7)
    ax[0].set_ylabel('Rotation Error (degrees)', fontsize=12)
    ax[0].set_title('Rotation Error Statistics', fontsize=14, fontweight='bold')
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(metrics)
    ax[0].legend(fontsize=11)
    ax[0].grid(True, alpha=0.3, axis='y')
    
    our_trans_stats = [np.mean(our_trans_errors), np.median(our_trans_errors), np.std(our_trans_errors)]
    baseline_trans_stats = [np.mean(baseline_trans_errors), np.median(baseline_trans_errors), np.std(baseline_trans_errors)]
    
    ax[1].bar(x - width/2, our_trans_stats, width, label='Our Model', color='blue', alpha=0.7)
    ax[1].bar(x + width/2, baseline_trans_stats, width, label='AnyCam Baseline', color='red', alpha=0.7)
    ax[1].set_ylabel('Translation Direction Error (degrees)', fontsize=12)
    ax[1].set_title('Translation Error Statistics', fontsize=14, fontweight='bold')
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(metrics)
    ax[1].legend(fontsize=11)
    ax[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_dir / 'error_comparison_bars.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"[VIZ] Saved error distribution plots to {save_dir}")


def generate_evaluation_report(
    our_rot_stats: Dict,
    our_trans_stats: Dict,
    baseline_rot_stats: Dict,
    baseline_trans_stats: Dict,
    num_samples: int,
    save_path: Path,
):
    """Generate text report summarizing evaluation results."""
    
    # Compute improvements
    rot_improvement = ((baseline_rot_stats['mean'] - our_rot_stats['mean']) / 
                      baseline_rot_stats['mean'] * 100)
    trans_improvement = ((baseline_trans_stats['mean'] - our_trans_stats['mean']) / 
                        baseline_trans_stats['mean'] * 100)
    
    report = []
    report.append("=" * 70)
    report.append("POSE ESTIMATION EVALUATION REPORT")
    report.append("=" * 70)
    report.append(f"\nTest Set Size: {num_samples} frame pairs\n")
    
    report.append("\nROTATION ERROR (degrees):")
    report.append("-" * 70)
    report.append(f"  Our Model:")
    report.append(f"    Mean:   {our_rot_stats['mean']:8.4f}")
    report.append(f"    Median: {our_rot_stats['median']:8.4f}")
    report.append(f"    Std:    {our_rot_stats['std']:8.4f}")
    report.append(f"    Min:    {our_rot_stats['min']:8.4f}")
    report.append(f"    Max:    {our_rot_stats['max']:8.4f}")
    report.append(f"    P90:    {our_rot_stats['p90']:8.4f}")
    report.append(f"    P95:    {our_rot_stats['p95']:8.4f}")
    report.append(f"\n  AnyCam Baseline:")
    report.append(f"    Mean:   {baseline_rot_stats['mean']:8.4f}")
    report.append(f"    Median: {baseline_rot_stats['median']:8.4f}")
    report.append(f"    Std:    {baseline_rot_stats['std']:8.4f}")
    report.append(f"    Min:    {baseline_rot_stats['min']:8.4f}")
    report.append(f"    Max:    {baseline_rot_stats['max']:8.4f}")
    report.append(f"    P90:    {baseline_rot_stats['p90']:8.4f}")
    report.append(f"    P95:    {baseline_rot_stats['p95']:8.4f}")
    report.append(f"\n  Improvement: {rot_improvement:+.2f}% (positive = better)")
    
    report.append("\n\nTRANSLATION DIRECTION ERROR (degrees):")
    report.append("-" * 70)
    report.append(f"  Our Model:")
    report.append(f"    Mean:   {our_trans_stats['mean']:8.4f}")
    report.append(f"    Median: {our_trans_stats['median']:8.4f}")
    report.append(f"    Std:    {our_trans_stats['std']:8.4f}")
    report.append(f"    Min:    {our_trans_stats['min']:8.4f}")
    report.append(f"    Max:    {our_trans_stats['max']:8.4f}")
    report.append(f"    P90:    {our_trans_stats['p90']:8.4f}")
    report.append(f"    P95:    {our_trans_stats['p95']:8.4f}")
    report.append(f"\n  AnyCam Baseline:")
    report.append(f"    Mean:   {baseline_trans_stats['mean']:8.4f}")
    report.append(f"    Median: {baseline_trans_stats['median']:8.4f}")
    report.append(f"    Std:    {baseline_trans_stats['std']:8.4f}")
    report.append(f"    Min:    {baseline_trans_stats['min']:8.4f}")
    report.append(f"    Max:    {baseline_trans_stats['max']:8.4f}")
    report.append(f"    P90:    {baseline_trans_stats['p90']:8.4f}")
    report.append(f"    P95:    {baseline_trans_stats['p95']:8.4f}")
    report.append(f"\n  Improvement: {trans_improvement:+.2f}% (positive = better)")
    
    report.append("\n" + "=" * 70)
    report.append(f"\nSUMMARY:")
    report.append(f"  Rotation Error:    {our_rot_stats['mean']:.4f}° vs {baseline_rot_stats['mean']:.4f}° ({rot_improvement:+.2f}%)")
    report.append(f"  Translation Error: {our_trans_stats['mean']:.4f}° vs {baseline_trans_stats['mean']:.4f}° ({trans_improvement:+.2f}%)")
    
    if rot_improvement > 0 and trans_improvement > 0:
        report.append(f"\n  ✓ Our model outperforms baseline on both metrics!")
    elif rot_improvement > 0 or trans_improvement > 0:
        report.append(f"\n  ~ Mixed results: improvement on some metrics")
    else:
        report.append(f"\n  ✗ Baseline outperforms our model")
    
    report.append("\n" + "=" * 70)
    
    report_text = "\n".join(report)
    
    # Save to file
    with open(save_path, 'w') as f:
        f.write(report_text)
    
    # Print to console
    print(f"\n{report_text}\n")
    print(f"[REPORT] Saved evaluation report to: {save_path}")


# =============================================================================
# MAIN EVALUATION FUNCTION
# =============================================================================

def evaluate_models(
    our_model,
    baseline_model,
    test_loader: DataLoader,
    device: torch.device,
    save_dir: Path,
):
    """
    Run evaluation comparing trained model vs baseline on test set.
    """
    our_model.eval()
    baseline_model.eval()
    
    our_rot_errors_all = []
    our_trans_errors_all = []
    baseline_rot_errors_all = []
    baseline_trans_errors_all = []
    
    print(f"\n[EVAL] Running evaluation on {len(test_loader)} batches...")
    
    with torch.no_grad():
        for batch_data in tqdm(test_loader, desc="Evaluating"):
            # Move to device
            batch_data = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                         for k, v in batch_data.items()}
            
            # Get predictions from our model
            our_output = our_model(batch_data)
            our_poses = extract_relative_poses_from_output(our_output, use_gt=False)
            
            # Get predictions from baseline
            baseline_output = baseline_model(batch_data)
            baseline_poses = extract_relative_poses_from_output(baseline_output, use_gt=False)
            
            # Get ground truth
            gt_poses = extract_relative_poses_from_output(batch_data, use_gt=True)
            
            # Compute errors
            our_rot_err, our_trans_err = compute_pose_errors_batch(our_poses, gt_poses)
            baseline_rot_err, baseline_trans_err = compute_pose_errors_batch(baseline_poses, gt_poses)
            
            our_rot_errors_all.extend(our_rot_err)
            our_trans_errors_all.extend(our_trans_err)
            baseline_rot_errors_all.extend(baseline_rot_err)
            baseline_trans_errors_all.extend(baseline_trans_err)
    
    # Convert to numpy
    our_rot_errors = np.array(our_rot_errors_all)
    our_trans_errors = np.array(our_trans_errors_all)
    baseline_rot_errors = np.array(baseline_rot_errors_all)
    baseline_trans_errors = np.array(baseline_trans_errors_all)
    
    # Compute statistics
    our_rot_stats = compute_error_statistics(our_rot_errors)
    our_trans_stats = compute_error_statistics(our_trans_errors)
    baseline_rot_stats = compute_error_statistics(baseline_rot_errors)
    baseline_trans_stats = compute_error_statistics(baseline_trans_errors)
    
    # Save results to JSON
    results = {
        'our_model': {
            'rotation': our_rot_stats,
            'translation': our_trans_stats,
        },
        'baseline': {
            'rotation': baseline_rot_stats,
            'translation': baseline_trans_stats,
        },
        'num_samples': len(our_rot_errors),
    }
    
    with open(save_dir / 'evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Generate plots
    plot_error_distributions(
        our_rot_errors, our_trans_errors,
        baseline_rot_errors, baseline_trans_errors,
        save_dir
    )
    
    # Generate report
    generate_evaluation_report(
        our_rot_stats, our_trans_stats,
        baseline_rot_stats, baseline_trans_stats,
        len(our_rot_errors),
        save_dir / 'evaluation_report.txt'
    )
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Pose Estimation Model")
    parser.add_argument("--our_model_path", type=str, required=True,
                       help="Path to trained model checkpoint")
    parser.add_argument("--baseline_model_path", type=str,
                       default="pretrained_models/anycam_seq8",
                       help="Path to baseline AnyCam model")
    parser.add_argument("--test_data_dir", type=str,
                       default="/home/kalman/TUM/thesis/Objectron/videos/",
                       help="Directory with test videos")
    parser.add_argument("--test_gt_dir", type=str,
                       default="/home/kalman/TUM/thesis/Objectron/annotations/",
                       help="Directory with test ground truth")
    parser.add_argument("--split_file", type=str,
                       default="experiments/objectron_split.json",
                       help="Dataset split file")
    parser.add_argument("--save_dir", type=str,
                       default="experiments/evaluation_results",
                       help="Directory to save evaluation results")
    parser.add_argument("--batch_size", type=int, default=1,
                       help="Batch size for evaluation")
    
    args = parser.parse_args()
    
    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[EVAL] Evaluation script initialized")
    print(f"[EVAL] Results will be saved to: {save_dir}")
    print(f"\n[NOTE] Full implementation requires model loading - see train_pose_head_anycalib.py for integration")

