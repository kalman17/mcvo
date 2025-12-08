#!/usr/bin/env python3
"""
Hyperparameter Sweep for Experiment 2: max_ahead values 2-7

This script trains models with different max_ahead values sequentially,
then runs comprehensive benchmarking to compare all models against
AnyCam baseline on both Objectron test split and LightSpeed datasets.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
from typing import List, Dict

from experiments.dataset_paths import (
    get_objectron_videos, get_objectron_gt, get_lightspeed_root
)

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def run_training(max_ahead: int, base_dir: Path, args: argparse.Namespace) -> Path:
    """
    Run training for a specific max_ahead value.
    
    Args:
        max_ahead: Number of frames ahead to predict
        base_dir: Base directory for results
        args: Command line arguments with training configuration
    
    Returns:
        save_dir: Directory where model was saved
    """
    save_dir = base_dir / f"exp2_maxahead_{max_ahead}"
    
    print(f"\n{'='*80}")
    print(f"[SWEEP] Training model with max_ahead={max_ahead}")
    print(f"{'='*80}")
    
    # Build training command
    cmd = [
        sys.executable,
        "experiments/train_pose_head_anycalib_exp2.py",
        "--num_epochs", str(args.num_epochs),
        "--batch_size", str(args.batch_size),
        "--lr", str(args.lr),
        "--max_ahead", str(max_ahead),
        "--save_dir", str(save_dir),
        "--videos_dir", args.videos_dir,
        "--gt_dir", args.gt_dir,
        "--split_file", args.split_file,
        "--model_path", args.model_path,
        "--lightspeed_dir", args.lightspeed_dir,
    ]
    
    # Add flags
    if args.use_direct_flow:
        cmd.append("--use_direct_flow")
    if args.disable_composed_flow:
        cmd.append("--disable_composed_flow")
    if args.disable_composed_loss:
        cmd.append("--disable_composed_loss")
    
    print(f"[SWEEP] Command: {' '.join(cmd)}")
    print(f"[SWEEP] Results will be saved to: {save_dir}")
    
    # Run training
    result = subprocess.run(cmd, cwd=project_root)
    
    if result.returncode != 0:
        raise RuntimeError(f"Training failed for max_ahead={max_ahead} with return code {result.returncode}")
    
    print(f"[SWEEP] Training completed for max_ahead={max_ahead}")
    
    return save_dir


def run_benchmarking(
    model_paths: Dict[int, Path],
    baseline_checkpoint: Path,
    base_dir: Path,
    args: argparse.Namespace
):
    """
    Run benchmarking on all trained models.
    
    Args:
        model_paths: Dictionary mapping max_ahead -> model save directory
        baseline_checkpoint: Path to AnyCam baseline checkpoint
        base_dir: Base directory for results
        args: Command line arguments
    """
    print(f"\n{'='*80}")
    print(f"[SWEEP] Starting benchmarking for all models")
    print(f"{'='*80}")
    
    # Benchmark on Objectron test split
    print(f"\n[SWEEP] Benchmarking on Objectron test split...")
    objectron_results_dir = base_dir / "benchmark_results_objectron"
    objectron_results_dir.mkdir(exist_ok=True)
    
    cmd = [
        sys.executable,
        "experiments/benchmark_against_anycam.py",
        "--baseline_checkpoint", str(baseline_checkpoint),
        "--dataset", "objectron",
        "--max_samples", str(args.max_samples),
        "--save_dir", str(objectron_results_dir),
        "--videos_dir", args.videos_dir,
        "--gt_dir", args.gt_dir,
        "--split_file", args.split_file,
    ]
    
    # Add all exp2 models
    for max_ahead, model_dir in model_paths.items():
        model_path = model_dir / "final_model.pt"
        if model_path.exists():
            cmd.extend(["--exp2_model", str(model_path)])
    
    print(f"[SWEEP] Objectron command: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=project_root)
    
    if result.returncode != 0:
        print(f"[WARN] Objectron benchmarking failed with return code {result.returncode}")
    else:
        print(f"[SWEEP] Objectron benchmarking completed. Results: {objectron_results_dir}")
    
    # Benchmark on LightSpeed dataset
    print(f"\n[SWEEP] Benchmarking on LightSpeed dataset...")
    lightspeed_results_dir = base_dir / "benchmark_results_lightspeed"
    lightspeed_results_dir.mkdir(exist_ok=True)
    
    cmd = [
        sys.executable,
        "experiments/benchmark_against_anycam.py",
        "--baseline_checkpoint", str(baseline_checkpoint),
        "--dataset", "lightspeed",
        "--max_samples", str(args.max_samples),
        "--save_dir", str(lightspeed_results_dir),
        "--lightspeed_dir", args.lightspeed_dir,
    ]
    
    # Add all exp2 models
    for max_ahead, model_dir in model_paths.items():
        model_path = model_dir / "final_model.pt"
        if model_path.exists():
            cmd.extend(["--exp2_model", str(model_path)])
    
    print(f"[SWEEP] LightSpeed command: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=project_root)
    
    if result.returncode != 0:
        print(f"[WARN] LightSpeed benchmarking failed with return code {result.returncode}")
    else:
        print(f"[SWEEP] LightSpeed benchmarking completed. Results: {lightspeed_results_dir}")
    
    print(f"\n{'='*80}")
    print(f"[SWEEP] All benchmarking completed!")
    print(f"{'='*80}")
    print(f"[SWEEP] Objectron results: {objectron_results_dir}")
    print(f"[SWEEP] LightSpeed results: {lightspeed_results_dir}")


def main():
    parser = argparse.ArgumentParser(description="Hyperparameter sweep for Experiment 2: max_ahead values")
    
    # Dataset arguments
    parser.add_argument("--videos_dir", type=str,
                       default=get_objectron_videos(),
                       help="Directory containing Objectron videos")
    parser.add_argument("--gt_dir", type=str,
                       default=get_objectron_gt(),
                       help="Directory containing ground truth JSON files")
    parser.add_argument("--split_file", type=str,
                       default="experiments/objectron_split.json",
                       help="Path to dataset split file")
    parser.add_argument("--lightspeed_dir", type=str,
                       default=get_lightspeed_root(),
                       help="LightSpeed dataset directory")
    
    # Model arguments
    parser.add_argument("--model_path", type=str,
                       default="pretrained_models/anycam_seq8/",
                       help="Path to pretrained AnyCam model")
    parser.add_argument("--baseline_checkpoint", type=str,
                       default="pretrained_models/anycam_seq8/training_checkpoint_247500.pt",
                       help="Path to AnyCam baseline checkpoint for benchmarking")
    
    # Training arguments
    parser.add_argument("--num_epochs", type=int, default=10,
                       help="Number of training epochs per model")
    parser.add_argument("--batch_size", type=int, default=2,
                       help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-4,
                       help="Learning rate")
    
    # Sweep arguments
    parser.add_argument("--max_ahead_values", type=int, nargs="+",
                       default=[2, 3, 4, 5, 6, 7],
                       help="List of max_ahead values to train (default: 2 3 4 5 6 7)")
    parser.add_argument("--skip_training", action="store_true",
                       help="Skip training, only run benchmarking (assumes models already trained)")
    parser.add_argument("--skip_benchmarking", action="store_true",
                       help="Skip benchmarking, only run training")
    
    # Benchmarking arguments
    parser.add_argument("--max_samples", type=int, default=100,
                       help="Maximum samples for benchmarking")
    
    # Flow arguments
    parser.add_argument("--use_direct_flow", action="store_true",
                       help="Use direct UniMatch flows (default: composed flows)")
    parser.add_argument("--disable_composed_flow", action="store_true",
                       help="Disable composed flows")
    parser.add_argument("--disable_composed_loss", action="store_true",
                       help="Disable composed pose losses")
    
    # Output arguments
    parser.add_argument("--base_dir", type=str,
                       default="experiments/pose_head_experiment_results",
                       help="Base directory for all results")
    
    args = parser.parse_args()
    
    base_dir = Path(args.base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"[SWEEP] Experiment 2 Hyperparameter Sweep")
    print(f"{'='*80}")
    print(f"[SWEEP] max_ahead values: {args.max_ahead_values}")
    print(f"[SWEEP] num_epochs: {args.num_epochs}")
    print(f"[SWEEP] batch_size: {args.batch_size}")
    print(f"[SWEEP] Results base directory: {base_dir}")
    print(f"{'='*80}\n")
    
    # Step 1: Training
    model_paths = {}
    
    if not args.skip_training:
        for max_ahead in args.max_ahead_values:
            try:
                save_dir = run_training(max_ahead, base_dir, args)
                model_paths[max_ahead] = save_dir
            except Exception as e:
                print(f"[ERROR] Failed to train max_ahead={max_ahead}: {e}")
                import traceback
                traceback.print_exc()
                continue
    else:
        print(f"[SWEEP] Skipping training, loading existing models...")
        for max_ahead in args.max_ahead_values:
            model_dir = base_dir / f"exp2_maxahead_{max_ahead}"
            model_path = model_dir / "final_model.pt"
            if model_path.exists():
                model_paths[max_ahead] = model_dir
                print(f"[SWEEP] Found existing model: {model_path}")
            else:
                print(f"[WARN] Model not found: {model_path}")
    
    if not model_paths:
        print(f"[ERROR] No trained models available for benchmarking")
        return
    
    print(f"\n[SWEEP] Trained models: {list(model_paths.keys())}")
    
    # Step 2: Benchmarking
    if not args.skip_benchmarking:
        baseline_checkpoint = Path(args.baseline_checkpoint)
        if not baseline_checkpoint.exists():
            print(f"[ERROR] Baseline checkpoint not found: {baseline_checkpoint}")
            return
        
        run_benchmarking(model_paths, baseline_checkpoint, base_dir, args)
    else:
        print(f"[SWEEP] Skipping benchmarking")
    
    print(f"\n{'='*80}")
    print(f"[SWEEP] Hyperparameter sweep completed!")
    print(f"{'='*80}")
    print(f"[SWEEP] All results saved to: {base_dir}")


if __name__ == "__main__":
    main()

