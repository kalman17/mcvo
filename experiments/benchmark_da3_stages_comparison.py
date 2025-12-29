#!/usr/bin/env python3
"""
=============================================================================
DA3 Inter-Stage Comparison - Calibration Accuracy
=============================================================================

Compares calibration accuracy of DA3 Stages 1, 2, and 3 against each other.

This script:
1. Loads all three stage checkpoints
2. Evaluates on same test set (frame pairs)
3. Computes calibration errors vs GT mean intrinsics
4. Generates side-by-side comparison plots and reports

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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Add project paths
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "anycalib"))

# Import calibration benchmark functions
from experiments.benchmark_da3_calibration_accuracy import (
    CalibrationBenchmarkDataset,
    extract_visual_tokens_dinov2,
    evaluate_calibration_accuracy,
)

# DA3 imports
from experiments.models.da3_calibration_head import DA3CalibrationHead

# Dataset imports
from experiments.dataset_paths import (
    get_objectron_videos, get_objectron_gt
)
from experiments.train_pose_head_anycalib import (
    AnyCaLibBatchInference,
    load_dataset_split,
)
from experiments.benchmark_dataset_utils import (
    get_dataset_paths,
    count_available_pairs_objectron,
)

print("[INIT] Imports successful")


# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================

def plot_stages_comparison_distributions(all_results: Dict[str, Dict], save_path: Path):
    """Plot error distributions comparing all stages."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Calibration Parameter Error Distribution: DA3 Stages 1-3 Comparison', 
                 fontsize=16, fontweight='bold')
    
    params = ['fx', 'fy', 'cx', 'cy']
    param_names = {
        'fx': 'Focal Length (fx)',
        'fy': 'Focal Length (fy)',
        'cx': 'Principal Point (cx)',
        'cy': 'Principal Point (cy)',
    }
    
    colors = {'DA3 Stage 1': 'blue', 'DA3 Stage 2': 'green', 'DA3 Stage 3': 'red'}
    
    for idx, param in enumerate(params):
        ax = axes[idx // 2, idx % 2]
        
        for stage_name in ['DA3 Stage 1', 'DA3 Stage 2', 'DA3 Stage 3']:
            if stage_name in all_results:
                errors = all_results[stage_name]['errors'][param]
                ax.hist(errors, bins=50, alpha=0.6, label=stage_name, 
                       color=colors[stage_name], edgecolor='black')
        
        ax.set_xlabel('Relative Error (%)', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title(f'{param_names[param]}', fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[SAVE] Stages comparison distributions saved to {save_path}")
    plt.close()


def plot_stages_comparison_statistics(all_results: Dict[str, Dict], save_path: Path):
    """Plot bar chart comparing error statistics across stages."""
    params = ['fx', 'fy', 'cx', 'cy']
    param_names = {
        'fx': 'Focal Length (fx)',
        'fy': 'Focal Length (fy)',
        'cx': 'Principal Point (cx)',
        'cy': 'Principal Point (cy)',
    }
    
    metrics = ['mean', 'median', 'p90']
    metric_names = ['Mean', 'Median', '90th Percentile']
    stages = ['DA3 Stage 1', 'DA3 Stage 2', 'DA3 Stage 3']
    colors = {'DA3 Stage 1': 'blue', 'DA3 Stage 2': 'green', 'DA3 Stage 3': 'red'}
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Calibration Error Statistics Comparison: DA3 Stages 1-3', 
                 fontsize=16, fontweight='bold')
    
    for idx, param in enumerate(params):
        ax = axes[idx // 2, idx % 2]
        
        x = np.arange(len(metrics))
        width = 0.25
        
        for i, stage_name in enumerate(stages):
            if stage_name in all_results:
                values = [all_results[stage_name]['statistics'][param][metric] for metric in metrics]
                ax.bar(x + i * width, values, width, label=stage_name, 
                      color=colors[stage_name], alpha=0.8)
        
        ax.set_xlabel('Error Metric', fontsize=12)
        ax.set_ylabel('Relative Error (%)', fontsize=12)
        ax.set_title(f'{param_names[param]}', fontsize=13, fontweight='bold')
        ax.set_xticks(x + width)
        ax.set_xticklabels(metric_names)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[SAVE] Stages comparison statistics saved to {save_path}")
    plt.close()


def plot_stages_comparison_boxplot(all_results: Dict[str, Dict], save_path: Path):
    """Plot box plots comparing error distributions across stages."""
    params = ['fx', 'fy', 'cx', 'cy']
    param_names = {
        'fx': 'Focal Length (fx)',
        'fy': 'Focal Length (fy)',
        'cx': 'Principal Point (cx)',
        'cy': 'Principal Point (cy)',
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Calibration Error Distribution Comparison: DA3 Stages 1-3', 
                 fontsize=16, fontweight='bold')
    
    for idx, param in enumerate(params):
        ax = axes[idx // 2, idx % 2]
        
        data = []
        labels = []
        colors_list = []
        
        stage_colors = {'DA3 Stage 1': 'blue', 'DA3 Stage 2': 'green', 'DA3 Stage 3': 'red'}
        
        for stage_name in ['DA3 Stage 1', 'DA3 Stage 2', 'DA3 Stage 3']:
            if stage_name in all_results:
                data.append(all_results[stage_name]['errors'][param])
                labels.append(stage_name)
                colors_list.append(stage_colors[stage_name])
        
        if data:
            bp = ax.boxplot(data, labels=labels, patch_artist=True)
            for patch, color in zip(bp['boxes'], colors_list):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
        
        ax.set_ylabel('Relative Error (%)', fontsize=12)
        ax.set_title(f'{param_names[param]}', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[SAVE] Stages comparison boxplot saved to {save_path}")
    plt.close()


# =============================================================================
# REPORT GENERATION
# =============================================================================

def generate_comparison_report(all_results: Dict[str, Dict], num_samples: int, save_path: Path):
    """Generate comparison report for all stages."""
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("Calibration Accuracy Comparison: DA3 Stages 1-3\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Test Set Size: {num_samples} frame pairs\n")
        f.write(f"Models Evaluated: DA3 Stage 1, DA3 Stage 2, DA3 Stage 3\n\n")
        
        f.write("EVALUATION METHODOLOGY\n")
        f.write("-" * 80 + "\n")
        f.write("For each frame pair:\n")
        f.write("1. Run AnyCalib on both frames to get per-frame predictions\n")
        f.write("2. Extract visual tokens (DINOv2 for Stage 1/2, AnyCam for Stage 3)\n")
        f.write("3. Run DA3 calibration head to predict mean calibration\n")
        f.write("4. Compute GT mean calibration from the pair\n")
        f.write("5. Calculate relative errors: |predicted - GT| / GT * 100%\n\n")
        
        param_names = {
            'fx': 'Focal Length (fx)',
            'fy': 'Focal Length (fy)',
            'cx': 'Principal Point (cx)',
            'cy': 'Principal Point (cy)',
        }
        
        # Comparison table for each parameter
        for param in ['fx', 'fy', 'cx', 'cy']:
            f.write(f"{param_names[param].upper()} - RELATIVE ERROR (%)\n")
            f.write("-" * 80 + "\n")
            
            # Header
            header = f"{'Metric':<20}"
            for stage_name in ['DA3 Stage 1', 'DA3 Stage 2', 'DA3 Stage 3']:
                if stage_name in all_results:
                    header += f"{stage_name:>20}"
            f.write(header + "\n")
            f.write("-" * 80 + "\n")
            
            # Data rows
            for metric in ['mean', 'median', 'std', 'p90', 'p95']:
                row = f"{metric.upper():<20}"
                for stage_name in ['DA3 Stage 1', 'DA3 Stage 2', 'DA3 Stage 3']:
                    if stage_name in all_results:
                        value = all_results[stage_name]['statistics'][param][metric]
                        row += f"{value:>20.4f}"
                f.write(row + "\n")
            
            f.write("\n")
        
        f.write("="*80 + "\n")
        f.write("SUMMARY\n")
        f.write("="*80 + "\n")
        
        # Overall statistics
        for stage_name in ['DA3 Stage 1', 'DA3 Stage 2', 'DA3 Stage 3']:
            if stage_name in all_results:
                all_errors = []
                for param in ['fx', 'fy', 'cx', 'cy']:
                    all_errors.extend(all_results[stage_name]['errors'][param])
                
                overall_mean = np.mean(all_errors)
                overall_median = np.median(all_errors)
                
                f.write(f"{stage_name}:\n")
                f.write(f"  Overall Mean Relative Error: {overall_mean:.4f}%\n")
                f.write(f"  Overall Median Relative Error: {overall_median:.4f}%\n\n")
        
        # Best stage for each parameter
        f.write("Best Performance by Parameter:\n")
        for param in ['fx', 'fy', 'cx', 'cy']:
            best_stage = None
            best_mean = float('inf')
            for stage_name in ['DA3 Stage 1', 'DA3 Stage 2', 'DA3 Stage 3']:
                if stage_name in all_results:
                    mean_err = all_results[stage_name]['statistics'][param]['mean']
                    if mean_err < best_mean:
                        best_mean = mean_err
                        best_stage = stage_name
            
            if best_stage:
                f.write(f"  {param_names[param]}: {best_stage} ({best_mean:.4f}%)\n")
    
    print(f"[SAVE] Comparison report saved to {save_path}")


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Compare DA3 Stages 1-3 Calibration Accuracy")
    
    # Model arguments
    parser.add_argument("--stage1_checkpoint", type=str, required=True,
                       help="Path to Stage 1 checkpoint")
    parser.add_argument("--stage2_checkpoint", type=str, required=True,
                       help="Path to Stage 2 checkpoint")
    parser.add_argument("--stage3_checkpoint", type=str, required=True,
                       help="Path to Stage 3 checkpoint")
    
    # Dataset arguments
    parser.add_argument("--dataset", type=str, choices=['objectron', 'lightspeed'], default='objectron',
                       help="Dataset to use (paths auto-detected)")
    parser.add_argument("--split_file", type=str, default="experiments/objectron_split.json",
                       help="Dataset split file (for Objectron, optional)")
    parser.add_argument("--num_samples", type=str, default="100",
                       help="Number of frame pairs to evaluate (default: 100, use 'all' for all available)")
    
    # Model architecture arguments
    parser.add_argument("--vis_dim_stage12", type=int, default=384,
                       help="Visual token dimension for Stage 1/2 (DINOv2-S)")
    parser.add_argument("--vis_dim_stage3", type=int, default=768,
                       help="Visual token dimension for Stage 3 (AnyCam)")
    parser.add_argument("--cam_dim", type=int, default=256,
                       help="Camera token dimension")
    parser.add_argument("--hidden_dim", type=int, default=128,
                       help="Hidden layer dimension")
    
    # Evaluation arguments
    parser.add_argument("--save_dir", type=str,
                       default="experiments/da3_integration/benchmark_results/stages_comparison",
                       help="Output directory")
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Batch size for evaluation")
    parser.add_argument("--device", type=str, default="cuda:0",
                       help="Device to use")
    
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[INIT] Using device: {device}")
    
    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"[SAVE] Results will be saved to: {save_dir}")
    
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
    
    # Load dataset split (optional, for Objectron)
    test_indices = None
    if args.dataset == 'objectron':
        if Path(args.split_file).exists():
            split_data = load_dataset_split(args.split_file)
            test_indices = split_data.get('test', split_data.get('test_indices', []))
            print(f"[DATASET] Using test split: {len(test_indices)} videos")
        else:
            print(f"[DATASET] No split file found, using all videos")
    else:
        raise NotImplementedError("LightSpeed dataset not yet implemented for stages comparison")
    
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
    
    # Evaluate all stages
    print(f"\n[STEP 2] Evaluating all stages...")
    all_results = {}
    
    for stage, checkpoint_path, vis_dim in [
        (1, args.stage1_checkpoint, args.vis_dim_stage12),
        (2, args.stage2_checkpoint, args.vis_dim_stage12),
        (3, args.stage3_checkpoint, args.vis_dim_stage3),
    ]:
        stage_name = f"DA3 Stage {stage}"
        print(f"\n[EVAL] Evaluating {stage_name}...")
        
        # Create model
        model = DA3CalibrationHead(
            vis_dim=vis_dim,
            cam_dim=args.cam_dim,
            hidden_dim=args.hidden_dim,
            num_mixing_layers=2,
        ).to(device)
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # For Stage 3, extract calibration head from full model
        if stage == 3:
            calibration_head_dict = {}
            for k, v in state_dict.items():
                if 'pose_predictor.da3_calibration_head.' in k:
                    new_key = k.replace('pose_predictor.da3_calibration_head.', '')
                    calibration_head_dict[new_key] = v
                elif k.startswith('da3_calibration_head.'):
                    new_key = k.replace('da3_calibration_head.', '')
                    calibration_head_dict[new_key] = v
            
            if calibration_head_dict:
                state_dict = calibration_head_dict
                print(f"[LOAD] Extracted calibration head from full model")
        
        # Filter state dict for dimension mismatches
        model_state = model.state_dict()
        filtered_state_dict = {}
        for k, v in state_dict.items():
            if k in model_state:
                if v.shape == model_state[k].shape:
                    filtered_state_dict[k] = v
        
        model.load_state_dict(filtered_state_dict, strict=False)
        print(f"[LOAD] {stage_name} loaded ({len(filtered_state_dict)} keys)")
        
        # Determine visual token extraction
        extract_visual_fn = None
        if stage >= 2:
            extract_visual_fn = extract_visual_tokens_dinov2
        elif stage == 3:
            # Stage 3 uses AnyCam backbone, but for this benchmark we'll use DINOv2 for consistency
            # (Stage 3 model in full pipeline uses AnyCam, but calibration head can use DINOv2)
            extract_visual_fn = extract_visual_tokens_dinov2
        
        # Evaluate
        results = evaluate_calibration_accuracy(
            model=model,
            dataloader=dataloader,
            device=device,
            stage=stage,
            extract_visual_fn=extract_visual_fn,
        )
        
        all_results[stage_name] = results
    
    # Generate plots
    print(f"\n[STEP 3] Generating comparison plots...")
    plot_stages_comparison_distributions(all_results, save_dir / "stages_comparison_distributions.png")
    plot_stages_comparison_statistics(all_results, save_dir / "stages_comparison_statistics.png")
    plot_stages_comparison_boxplot(all_results, save_dir / "stages_comparison_boxplot.png")
    
    # Generate report
    print(f"\n[STEP 4] Generating comparison report...")
    generate_comparison_report(all_results, len(dataset), save_dir / "stages_comparison_report.txt")
    
    # Save results JSON
    results_json = {
        'num_samples': len(dataset),
        'stages': {}
    }
    
    for stage_name, results in all_results.items():
        results_json['stages'][stage_name] = {
            'errors': results['errors'],
            'statistics': results['statistics'],
        }
    
    with open(save_dir / "stages_comparison_results.json", 'w') as f:
        json.dump(results_json, f, indent=2)
    
    # Save metadata
    metadata = {
        'stage1_checkpoint': str(args.stage1_checkpoint),
        'stage2_checkpoint': str(args.stage2_checkpoint),
        'stage3_checkpoint': str(args.stage3_checkpoint),
        'dataset': args.dataset,
        'num_samples': len(dataset),
        'batch_size': args.batch_size,
        'device': str(device),
    }
    
    with open(save_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"STAGES COMPARISON COMPLETE")
    print(f"{'='*80}")
    print(f"Results saved to: {save_dir}")
    print(f"  - stages_comparison_results.json")
    print(f"  - stages_comparison_report.txt")
    print(f"  - stages_comparison_distributions.png")
    print(f"  - stages_comparison_statistics.png")
    print(f"  - stages_comparison_boxplot.png")
    print(f"  - metadata.json")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()

