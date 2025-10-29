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


def plot_comparison_multi_model(model_results: Dict, save_dir: Path):
    """Generate comparison plots for multiple models."""
    model_names = list(model_results.keys())
    colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Rotation error histogram
    for i, (model_name, results) in enumerate(model_results.items()):
        axes[0, 0].hist(results['rot_errors'], bins=50, alpha=0.7, 
                       label=model_name, color=colors[i % len(colors)])
    axes[0, 0].set_xlabel('Rotation Error (degrees)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Rotation Error Distribution')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Translation error histogram
    for i, (model_name, results) in enumerate(model_results.items()):
        axes[0, 1].hist(results['trans_errors'], bins=50, alpha=0.7, 
                       label=model_name, color=colors[i % len(colors)])
    axes[0, 1].set_xlabel('Translation Direction Error (degrees)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Translation Error Distribution')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # CDF for rotation
    for i, (model_name, results) in enumerate(model_results.items()):
        axes[0, 2].hist(results['rot_errors'], bins=100, cumulative=True, density=True, 
                        histtype='step', linewidth=2, label=model_name, color=colors[i % len(colors)])
    axes[0, 2].set_xlabel('Rotation Error (degrees)')
    axes[0, 2].set_ylabel('Cumulative Probability')
    axes[0, 2].set_title('Rotation Error CDF')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # CDF for translation
    for i, (model_name, results) in enumerate(model_results.items()):
        axes[1, 0].hist(results['trans_errors'], bins=100, cumulative=True, density=True,
                        histtype='step', linewidth=2, label=model_name, color=colors[i % len(colors)])
    axes[1, 0].set_xlabel('Translation Direction Error (degrees)')
    axes[1, 0].set_ylabel('Cumulative Probability')
    axes[1, 0].set_title('Translation Error CDF')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Box plots for rotation
    rot_data = [results['rot_errors'] for results in model_results.values()]
    axes[1, 1].boxplot(rot_data, labels=list(model_results.keys()))
    axes[1, 1].set_ylabel('Rotation Error (degrees)')
    axes[1, 1].set_title('Rotation Error Box Plot')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Box plots for translation
    trans_data = [results['trans_errors'] for results in model_results.values()]
    axes[1, 2].boxplot(trans_data, labels=list(model_results.keys()))
    axes[1, 2].set_ylabel('Translation Direction Error (degrees)')
    axes[1, 2].set_title('Translation Error Box Plot')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'benchmark_comparison.png', dpi=150, bbox_inches='tight')
    print(f"[SAVE] Multi-model comparison plots saved to {save_dir / 'benchmark_comparison.png'}")
    plt.close()


def generate_report_multi_model(model_stats: Dict, num_samples: int, save_path: Path):
    """Generate text report comparing multiple models."""
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("BENCHMARK: Multi-Model Comparison\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Test Set Size: {num_samples} frame pairs\n")
        f.write(f"Models Evaluated: {', '.join(model_stats.keys())}\n\n")
        
        f.write("ROTATION ERROR (degrees)\n")
        f.write("-" * 80 + "\n")
        
        # Header
        header = f"{'Metric':<20}"
        for model_name in model_stats.keys():
            header += f"{model_name:>15}"
        f.write(header + "\n")
        f.write("-" * 80 + "\n")
        
        # Data rows
        for metric in ['mean', 'median', 'std', 'p90']:
            row = f"{metric.upper():<20}"
            for model_name, stats in model_stats.items():
                row += f"{stats['rot_stats'][metric]:>15.4f}"
            f.write(row + "\n")
        
        f.write("\n")
        f.write("TRANSLATION DIRECTION ERROR (degrees)\n")
        f.write("-" * 80 + "\n")
        
        # Header
        header = f"{'Metric':<20}"
        for model_name in model_stats.keys():
            header += f"{model_name:>15}"
        f.write(header + "\n")
        f.write("-" * 80 + "\n")
        
        # Data rows
        for metric in ['mean', 'median', 'std', 'p90']:
            row = f"{metric.upper():<20}"
            for model_name, stats in model_stats.items():
                row += f"{stats['trans_stats'][metric]:>15.4f}"
            f.write(row + "\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("SUMMARY\n")
        f.write("="*80 + "\n")
        
        # Find best models
        best_rot_model = min(model_stats.keys(), key=lambda x: model_stats[x]['rot_stats']['mean'])
        best_trans_model = min(model_stats.keys(), key=lambda x: model_stats[x]['trans_stats']['mean'])
        
        f.write(f"Best Rotation Accuracy: {best_rot_model} ({model_stats[best_rot_model]['rot_stats']['mean']:.4f}°)\n")
        f.write(f"Best Translation Accuracy: {best_trans_model} ({model_stats[best_trans_model]['trans_stats']['mean']:.4f}°)\n")
    
    print(f"[SAVE] Multi-model report saved to {save_path}")


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
    parser = argparse.ArgumentParser(description="Benchmark Models Against Each Other")
    
    # Model arguments - now supporting multiple models
    parser.add_argument("--exp1_model", type=str, default=None,
                       help="Path to Experiment 1 model checkpoint (.pt file)")
    parser.add_argument("--exp2_model", type=str, default=None,
                       help="Path to Experiment 2 model checkpoint (.pt file)")
    parser.add_argument("--baseline_checkpoint", type=str,
                       default="pretrained_models/anycam_seq8/training_checkpoint_247500.pt",
                       help="Path to baseline AnyCam checkpoint")
    
    # Dataset arguments
    parser.add_argument("--dataset", type=str, choices=['objectron', 'lightspeed'], default='lightspeed',
                       help="Which dataset to use for evaluation")
    parser.add_argument("--objectron_videos", type=str,
                       default="/home/kalman/TUM/thesis/Objectron/videos/",
                       help="Objectron videos directory")
    parser.add_argument("--objectron_gt", type=str,
                       default="/home/kalman/TUM/thesis/Objectron/processed_gt/",
                       help="Objectron GT directory")
    parser.add_argument("--lightspeed_dir", type=str,
                       default="/home/kalman/TUM/thesis/dynpose-100k/lightspeed/",
                       help="LightSpeed dataset directory")
    parser.add_argument("--split_file", type=str,
                       default="experiments/objectron_split.json",
                       help="Dataset split file (for Objectron)")
    
    # Evaluation arguments
    parser.add_argument("--save_dir", type=str,
                       default=None,
                       help="Directory to save results")
    parser.add_argument("--batch_size", type=int, default=2,
                       help="Batch size for evaluation")
    parser.add_argument("--device", type=str, default="cuda:0",
                       help="Device to use")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="Maximum number of samples to evaluate (for faster testing)")
    
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[BENCHMARK] Using device: {device}")
    
    # Determine which models to evaluate
    models_to_eval = []
    if args.exp1_model:
        models_to_eval.append(("Experiment 1", args.exp1_model))
    if args.exp2_model:
        models_to_eval.append(("Experiment 2", args.exp2_model))
    if args.baseline_checkpoint:
        models_to_eval.append(("AnyCam Baseline", args.baseline_checkpoint))
    
    if not models_to_eval:
        print("[ERROR] No models specified for evaluation!")
        print("Use --exp1_model, --exp2_model, and/or --baseline_checkpoint")
        return
    
    print(f"[BENCHMARK] Will evaluate {len(models_to_eval)} models:")
    for name, path in models_to_eval:
        print(f"  - {name}: {path}")
    
    # Determine save directory
    if args.save_dir is None:
        # Use the first model's directory as base
        first_model_path = Path(models_to_eval[0][1])
        args.save_dir = first_model_path.parent / "benchmark_results"
    
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
    # Load Models
    # =============================================================================
    print(f"\n[STEP 2] Loading models...")
    
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
    
    # Load each model
    loaded_models = {}
    
    for model_name, model_path in models_to_eval:
        print(f"\n[LOAD] Loading {model_name} from {model_path}...")
        
        if model_name == "AnyCam Baseline":
            # Load baseline model
            baseline_anycalib = AnyCaLibBatchInference(device=device)
            model = AnyCamWrapperWithAnyCaLib(
                pose_predictor_config=pose_predictor_config,
                depth_predictor_config=depth_predictor_config,
                anycalib_model=baseline_anycalib,
            )
            model = model.to(device)
            
            # Load PRETRAINED checkpoint
            baseline_checkpoint = torch.load(model_path, map_location=device)
            if 'model' in baseline_checkpoint:
                baseline_checkpoint_data = baseline_checkpoint['model']
            else:
                baseline_checkpoint_data = baseline_checkpoint
            
            # Load the full model state dict from pretrained checkpoint
            model.pose_predictor.load_state_dict(baseline_checkpoint_data, strict=False)
            
        else:
            # Load experiment model (Exp1 or Exp2)
            if model_name == "Experiment 2":
                # Import Experiment 2 wrapper
                from experiments.train_pose_head_anycalib_exp2 import AnyCamWrapperMultiFrame
                model = AnyCamWrapperMultiFrame(
                    pose_predictor_config=pose_predictor_config,
                    depth_predictor_config=depth_predictor_config,
                    anycalib_model=anycalib_inference,
                    max_ahead=3,  # Default max_ahead for evaluation
                )
            else:
                # Experiment 1 uses regular wrapper
                model = AnyCamWrapperWithAnyCaLib(
                    pose_predictor_config=pose_predictor_config,
                    depth_predictor_config=depth_predictor_config,
                    anycalib_model=anycalib_inference,
                )
            
            model = model.to(device)
            
            # Load checkpoint
            checkpoint = torch.load(model_path, map_location=device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                print(f"[MODEL] Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
            else:
                model.load_state_dict(checkpoint)
        
        model.eval()
        loaded_models[model_name] = model
        print(f"[MODEL] {model_name} loaded successfully")
    
    # =============================================================================
    # Run Evaluation on All Models
    # =============================================================================
    print(f"\n[STEP 3] Evaluating {len(loaded_models)} models...")
    
    # Evaluate each model
    model_results = {}
    
    for model_name, model in loaded_models.items():
        print(f"\n[EVAL] Evaluating {model_name}...")
        
        rot_errors, trans_errors = evaluate_model_on_dataset(
            model, test_dataloader, device, model_name
        )
        
        model_results[model_name] = {
            'rot_errors': rot_errors,
            'trans_errors': trans_errors,
        }
        
        print(f"[EVAL] {model_name}: {len(rot_errors)} samples evaluated")
    
    # =============================================================================
    # Compute Statistics and Generate Report
    # =============================================================================
    print(f"\n[STEP 4] Computing statistics and generating report...")
    
    # Check if we have any valid results
    valid_models = {name: results for name, results in model_results.items() 
                   if len(results['rot_errors']) > 0}
    
    if not valid_models:
        print(f"\n[ERROR] No valid pose comparisons were made!")
        print(f"This likely means the pose prediction or GT loading failed.")
        return
    
    print(f"[STATS] Computing statistics for {len(valid_models)} models...")
    
    # Compute statistics for each model
    model_stats = {}
    for model_name, results in valid_models.items():
        rot_stats = compute_error_statistics(results['rot_errors'])
        trans_stats = compute_error_statistics(results['trans_errors'])
        
        model_stats[model_name] = {
            'rot_stats': rot_stats,
            'trans_stats': trans_stats,
        }
        
        print(f"[STATS] {model_name}: {len(results['rot_errors'])} errors collected")
    
    # Save results to JSON
    results = {}
    for model_name, stats in model_stats.items():
        results[model_name] = {
            'rotation': stats['rot_stats'],
            'translation': stats['trans_stats'],
        }
    
    # Add metadata
    results['metadata'] = {
        'num_samples': len(list(valid_models.values())[0]['rot_errors']),
        'dataset': args.dataset,
        'models_evaluated': list(valid_models.keys()),
    }
    
    results_path = save_dir / "benchmark_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Generate plots for all models
    plot_comparison_multi_model(model_results, save_dir)
    
    # Generate report
    report_path = save_dir / "benchmark_report.txt"
    generate_report_multi_model(model_stats, results['metadata']['num_samples'], report_path)
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"BENCHMARK RESULTS SUMMARY")
    print(f"{'='*80}")
    print(f"\nTest samples: {results['metadata']['num_samples']}")
    print(f"Models evaluated: {', '.join(valid_models.keys())}")
    
    print(f"\nRotation Error (degrees):")
    for model_name, stats in model_stats.items():
        print(f"  {model_name:<20} Mean={stats['rot_stats']['mean']:.4f}, Median={stats['rot_stats']['median']:.4f}")
    
    print(f"\nTranslation Error (degrees):")
    for model_name, stats in model_stats.items():
        print(f"  {model_name:<20} Mean={stats['trans_stats']['mean']:.4f}, Median={stats['trans_stats']['median']:.4f}")
    
    print(f"\n{'='*80}")
    
    print(f"\nResults saved to: {save_dir}")
    print(f"  - benchmark_results.json (detailed metrics)")
    print(f"  - benchmark_comparison.png (visualization)")
    print(f"  - benchmark_report.txt (full text report)")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()

