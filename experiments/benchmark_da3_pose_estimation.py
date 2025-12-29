#!/usr/bin/env python3
"""
=============================================================================
DA3 Pose Estimation Benchmark - Stage 3 vs Baseline vs Hybrid
=============================================================================

Compares Stage 3 DA3+AnyCam hybrid vs AnyCalib+AnyCam hybrid vs AnyCam baseline
on pose estimation accuracy.

**IMPORTANT - Fair Comparison Requires Large-Scale Training**:
This benchmark is designed for use AFTER training on large, diverse datasets.
Comparing models trained on different datasets (e.g., DA3 Stage 3 on Objectron
vs AnyCam Baseline/AnyCalib Hybrid on other datasets) is NOT scientifically
valid and does not represent true relative performance.

**Current Status**: DA3 models trained on Objectron only (validation experiment).
Fair comparison with baseline methods requires retraining DA3 on the same
large-scale diverse dataset used for AnyCam and AnyCalib.

This script:
1. Loads three models: DA3 Stage 3, AnyCalib Hybrid, AnyCam Baseline
2. Evaluates on frame pairs with GT poses (Objectron or LightSpeed)
3. Computes rotation and translation errors
4. Generates professional comparison plots and reports

Author: AI Assistant for Kalman's Master's Thesis
Date: December 2025
"""

import sys
import os

# Disable xFormers for GPU compatibility
os.environ["XFORMERS_DISABLED"] = "1"
os.environ["XFORMERS_MORE_DETAILS"] = "0"

import argparse
import json
from pathlib import Path
from typing import Dict, List, Union
import numpy as np
from pathlib import Path
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

# Import evaluation functions
from experiments.benchmark_against_anycam import (
    evaluate_model_on_dataset,
    plot_comparison_multi_model,
    plot_trajectories,
    generate_report_multi_model,
)
from experiments.pose_metrics import (
    compute_error_statistics,
)

# Dataset imports
from experiments.dataset_paths import (
    get_objectron_videos, get_objectron_gt, get_lightspeed_root
)
from experiments.train_pose_head_anycalib import (
    ObjectronVideoDataset,
    AnyCamWrapperWithAnyCaLib,
    AnyCamWrapperWithDA3Calibration,
    AnyCaLibBatchInference,
    load_dataset_split,
)
from experiments.lightspeed_dataset import LightSpeedDataset
from experiments.benchmark_dataset_utils import (
    get_dataset_paths,
    create_smart_sampled_dataset_lightspeed,
    count_available_pairs_lightspeed,
)

# AnyCam imports
from anycam.models import make_pose_predictor, make_depth_predictor

print("[INIT] Imports successful")


# =============================================================================
# MODEL LOADING
# =============================================================================

def load_da3_stage3_model(
    checkpoint_path: str,
    pose_predictor_config: Dict,
    depth_predictor_config: Dict,
    device: torch.device,
) -> nn.Module:
    """Load DA3 Stage 3 model with calibration head."""
    from experiments.train_pose_head_anycalib import AnyCaLibBatchInference
    
    anycalib_inference = AnyCaLibBatchInference(device=device)
    
    # Enable DA3 calibration in pose predictor config
    pose_predictor_config_da3 = pose_predictor_config.copy()
    pose_predictor_config_da3['use_da3_calibration'] = True
    
    model = AnyCamWrapperWithDA3Calibration(
        pose_predictor_config=pose_predictor_config_da3,
        depth_predictor_config=depth_predictor_config,
        anycalib_model=anycalib_inference,
        use_da3_calibration=True,
    ).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    if 'model_state_dict' in checkpoint:
        checkpoint_data = checkpoint['model_state_dict']
    else:
        checkpoint_data = checkpoint
    
    model.load_state_dict(checkpoint_data, strict=False)
    print(f"[LOAD] DA3 Stage 3 model loaded from {checkpoint_path}")
    
    return model


def load_anycalib_hybrid_model(
    pose_predictor_config: Dict,
    depth_predictor_config: Dict,
    device: torch.device,
) -> nn.Module:
    """Load AnyCalib-AnyCam hybrid model with multi-frame averaging."""
    from experiments.train_pose_head_anycalib import AnyCaLibBatchInference
    
    anycalib_inference = AnyCaLibBatchInference(device=device)
    anycalib_inference.use_multi_frame = True  # Enable multi-frame averaging
    
    model = AnyCamWrapperWithAnyCaLib(
        pose_predictor_config=pose_predictor_config,
        depth_predictor_config=depth_predictor_config,
        anycalib_model=anycalib_inference,
    ).to(device)
    
    print(f"[LOAD] AnyCalib-AnyCam Hybrid model loaded (multi-frame averaging enabled)")
    
    return model


def load_anycam_baseline(
    checkpoint_path: str,
    pose_predictor_config: Dict,
    depth_predictor_config: Dict,
    device: torch.device,
) -> nn.Module:
    """Load AnyCam baseline model (32-candidate system)."""
    from experiments.train_pose_head_anycalib import AnyCaLibBatchInference
    
    anycalib_inference = AnyCaLibBatchInference(device=device)
    
    model = AnyCamWrapperWithAnyCaLib(
        pose_predictor_config=pose_predictor_config,
        depth_predictor_config=depth_predictor_config,
        anycalib_model=anycalib_inference,
    ).to(device)
    
    # Load pretrained checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    if 'model' in checkpoint:
        baseline_checkpoint_data = checkpoint['model']
    else:
        baseline_checkpoint_data = checkpoint
    
    model.pose_predictor.load_state_dict(baseline_checkpoint_data, strict=False)
    print(f"[LOAD] AnyCam Baseline model loaded from {checkpoint_path}")
    
    return model


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Benchmark DA3 Stage 3 vs Baseline vs Hybrid")
    
    # Model arguments
    parser.add_argument("--stage3_checkpoint", type=str, required=True,
                       help="Path to DA3 Stage 3 checkpoint")
    parser.add_argument("--baseline_checkpoint", type=str,
                       default="pretrained_models/anycam_seq8/training_checkpoint_247500.pt",
                       help="Path to AnyCam baseline checkpoint")
    
    # Dataset arguments
    parser.add_argument("--dataset", type=str, choices=['objectron', 'lightspeed'], default='lightspeed',
                       help="Dataset to use (paths auto-detected)")
    parser.add_argument("--split_file", type=str, default="experiments/objectron_split.json",
                       help="Dataset split file (for Objectron, optional)")
    parser.add_argument("--num_samples", type=str, default="100",
                       help="Number of frame pairs to evaluate (default: 100, use 'all' for all available)")
    
    # Evaluation arguments
    parser.add_argument("--save_dir", type=str,
                       default="experiments/da3_integration/benchmark_results/stage3_pose_estimation",
                       help="Output directory")
    parser.add_argument("--batch_size", type=int, default=2,
                       help="Batch size for evaluation")
    parser.add_argument("--device", type=str, default="cuda:0",
                       help="Device to use")
    parser.add_argument("--num_frames", type=int, default=2,
                       help="Number of frames per sequence (default: 2 for pairs)")
    
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[INIT] Using device: {device}")
    
    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"[SAVE] Results will be saved to: {save_dir}")
    
    # Load config from baseline checkpoint
    import yaml
    baseline_config_path = Path(args.baseline_checkpoint).parent / "training_config.yaml"
    if not baseline_config_path.exists():
        raise FileNotFoundError(f"Config file not found: {baseline_config_path}")
    
    with open(baseline_config_path, 'r') as f:
        full_config = yaml.safe_load(f)
    
    pose_predictor_config = full_config['model']['pose_predictor']
    depth_predictor_config = full_config['model']['depth_predictor']
    
    # Parse num_samples
    if args.num_samples.lower() == 'all':
        num_samples = "all"
    else:
        try:
            num_samples = int(args.num_samples)
        except ValueError:
            raise ValueError(f"Invalid num_samples: {args.num_samples} (must be integer or 'all')")
    
    # Get dataset paths automatically
    paths = get_dataset_paths(args.dataset)
    
    # Load test dataset with smart sampling
    print(f"\n[STEP 1] Loading {args.dataset} test dataset...")
    print(f"[DATASET] Requested samples: {num_samples}")
    
    if args.dataset == 'objectron':
        # Load split if available
        test_indices = None
        if Path(args.split_file).exists():
            split_data = load_dataset_split(args.split_file)
            test_indices = split_data.get('test', split_data.get('test_indices', []))
            print(f"[DATASET] Using test split: {len(test_indices)} videos")
        else:
            print(f"[DATASET] No split file found, using all videos")
        
        # Count available pairs
        total_available = count_available_pairs_objectron(
            paths['videos'], paths['gt'], test_indices
        )
        print(f"[DATASET] Found {total_available} available frame pairs")
        
        # Create dataset with smart sampling
        # For pose estimation, we need to use ObjectronVideoDataset but with smart sampling
        # We'll create a custom dataset that samples pairs intelligently
        test_dataset = ObjectronVideoDataset(
            videos_dir=str(paths['videos']),
            gt_dir=str(paths['gt']),
            num_frames=args.num_frames,
            video_indices=test_indices,
            require_gt=True,
            extract_all_pairs=True,  # Extract all pairs, then sample
        )
        
        # Apply smart sampling
        if num_samples != "all" and num_samples < len(test_dataset):
            from torch.utils.data import Subset
            indices = np.random.choice(len(test_dataset), size=num_samples, replace=False)
            test_dataset = Subset(test_dataset, indices.tolist())
            print(f"[DATASET] Sampled {len(test_dataset)} pairs from {total_available} available")
        else:
            print(f"[DATASET] Using all {len(test_dataset)} available pairs")
            
    else:  # lightspeed
        # Count available pairs
        total_available = count_available_pairs_lightspeed(paths['root'])
        print(f"[DATASET] Found {total_available} available frame pairs")
        
        # Create dataset with smart sampling
        test_dataset = create_smart_sampled_dataset_lightspeed(
            paths['root'], num_samples
        )
        print(f"[DATASET] Using {len(test_dataset)} frame pairs (requested: {num_samples})")
    
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    
    print(f"[DATASET] Loaded {len(test_dataset)} test samples")
    
    # Load models
    print(f"\n[STEP 2] Loading models...")
    
    models_to_eval = {}
    
    # DA3 Stage 3
    print(f"\n[LOAD] Loading DA3 Stage 3 model...")
    da3_model = load_da3_stage3_model(
        checkpoint_path=args.stage3_checkpoint,
        pose_predictor_config=pose_predictor_config,
        depth_predictor_config=depth_predictor_config,
        device=device,
    )
    da3_model.eval()
    models_to_eval["DA3 Stage 3"] = da3_model
    
    # AnyCalib Hybrid
    print(f"\n[LOAD] Loading AnyCalib-AnyCam Hybrid model...")
    anycalib_hybrid = load_anycalib_hybrid_model(
        pose_predictor_config=pose_predictor_config,
        depth_predictor_config=depth_predictor_config,
        device=device,
    )
    anycalib_hybrid.eval()
    models_to_eval["AnyCalib-AnyCam Hybrid"] = anycalib_hybrid
    
    # AnyCam Baseline
    print(f"\n[LOAD] Loading AnyCam Baseline model...")
    baseline = load_anycam_baseline(
        checkpoint_path=args.baseline_checkpoint,
        pose_predictor_config=pose_predictor_config,
        depth_predictor_config=depth_predictor_config,
        device=device,
    )
    baseline.eval()
    models_to_eval["AnyCam Baseline"] = baseline
    
    # Evaluate all models
    print(f"\n[STEP 3] Evaluating {len(models_to_eval)} models...")
    
    model_results = {}
    
    for model_name, model in models_to_eval.items():
        print(f"\n[EVAL] Evaluating {model_name}...")
        
        rot_errors, trans_dir_errors, trans_mag_errors, trajectories_pred, trajectories_gt = evaluate_model_on_dataset(
            model, test_dataloader, device, model_name
        )
        
        model_results[model_name] = {
            'rot_errors': rot_errors,
            'trans_dir_errors': trans_dir_errors,
            'trans_mag_errors': trans_mag_errors,
            'trajectories_pred': trajectories_pred,
            'trajectories_gt': trajectories_gt,
        }
        
        print(f"[EVAL] {model_name}: {len(rot_errors)} samples evaluated")
    
    # Compute statistics
    print(f"\n[STEP 4] Computing statistics...")
    
    model_stats = {}
    for model_name, results in model_results.items():
        if len(results['rot_errors']) > 0:
            rot_stats = compute_error_statistics(results['rot_errors'])
            trans_dir_stats = compute_error_statistics(results['trans_dir_errors'])
            trans_mag_stats = compute_error_statistics(results['trans_mag_errors'])
            
            model_stats[model_name] = {
                'rot_stats': rot_stats,
                'trans_dir_stats': trans_dir_stats,
                'trans_mag_stats': trans_mag_stats,
            }
    
    # Save results to JSON
    results_json = {}
    for model_name, stats in model_stats.items():
        results_json[model_name] = {
            'rotation': stats['rot_stats'],
            'translation_direction': stats['trans_dir_stats'],
            'translation_magnitude': stats['trans_mag_stats'],
        }
    
    results_json['metadata'] = {
        'num_samples': len(list(model_results.values())[0]['rot_errors']),
        'num_frames': args.num_frames,
        'dataset': args.dataset,
        'models_evaluated': list(model_stats.keys()),
    }
    
    results_path = save_dir / "pose_estimation_results.json"
    with open(results_path, 'w') as f:
        json.dump(results_json, f, indent=2)
    
    # Generate plots
    print(f"\n[STEP 5] Generating plots...")
    plot_comparison_multi_model(model_results, save_dir)
    plot_trajectories(model_results, save_dir)
    
    # Generate report
    print(f"\n[STEP 6] Generating report...")
    report_path = save_dir / "pose_estimation_benchmark_report.txt"
    generate_report_multi_model(model_stats, results_json['metadata']['num_samples'], report_path)
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"POSE ESTIMATION BENCHMARK RESULTS")
    print(f"{'='*80}")
    print(f"\nTest samples: {results_json['metadata']['num_samples']}")
    print(f"Number of frames: {args.num_frames}")
    print(f"Models evaluated: {', '.join(model_stats.keys())}")
    
    print(f"\nRotation Error (degrees):")
    for model_name, stats in model_stats.items():
        print(f"  {model_name:<30} Mean={stats['rot_stats']['mean']:.4f}, Median={stats['rot_stats']['median']:.4f}")
    
    print(f"\nTranslation Direction Error (degrees):")
    for model_name, stats in model_stats.items():
        print(f"  {model_name:<30} Mean={stats['trans_dir_stats']['mean']:.4f}, Median={stats['trans_dir_stats']['median']:.4f}")
    
    print(f"\nTranslation Magnitude Error:")
    for model_name, stats in model_stats.items():
        print(f"  {model_name:<30} Mean={stats['trans_mag_stats']['mean']:.4f}, Median={stats['trans_mag_stats']['median']:.4f}")
    
    print(f"\n{'='*80}")
    print(f"\nResults saved to: {save_dir}")
    print(f"  - pose_estimation_results.json")
    print(f"  - pose_estimation_benchmark_report.txt")
    print(f"  - benchmark_comparison.png")
    print(f"  - trajectory_comparison.png")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()

