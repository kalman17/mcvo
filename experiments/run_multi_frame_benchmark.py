#!/usr/bin/env python3
"""
Multi-Frame Benchmark Runner

Runs benchmark evaluation across different frame counts (2, 3, 4, 5, 6, 7, 8)
for the look_ahead_3 model compared against AnyCam baseline.

Each frame count is evaluated separately and results are saved in separate
directories under test-diff-num-frames/.

Usage:
    python experiments/run_multi_frame_benchmark.py \
        --model_path experiments/pose_head_experiment_results/look_ahead_3.pt \
        --baseline_checkpoint pretrained_models/anycam_seq8/training_checkpoint_247500.pt \
        --lightspeed_dir /data/thesis/LightSpeed
"""

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Run multi-frame benchmark evaluation")
    
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to look_ahead_3.pt model checkpoint")
    parser.add_argument("--baseline_checkpoint", type=str,
                       default="pretrained_models/anycam_seq8/training_checkpoint_247500.pt",
                       help="Path to baseline AnyCam checkpoint")
    parser.add_argument("--lightspeed_dir", type=str, default=None,
                       help="LightSpeed dataset directory (defaults to dataset_paths)")
    parser.add_argument("--base_output_dir", type=str,
                       default="experiments/pose_head_experiment_results/test-diff-num-frames",
                       help="Base directory for output (will create frames_N subdirectories)")
    parser.add_argument("--batch_size", type=int, default=2,
                       help="Batch size for evaluation")
    parser.add_argument("--device", type=str, default="cuda:0",
                       help="Device to use")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="Maximum number of samples to evaluate (for faster testing)")
    
    args = parser.parse_args()
    
    # Frame counts to evaluate
    frame_counts = [2, 3, 4, 5, 6, 7, 8]
    
    base_output_dir = Path(args.base_output_dir)
    base_output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("MULTI-FRAME BENCHMARK EVALUATION")
    print("="*80)
    print(f"\nModel: {args.model_path}")
    print(f"Baseline: {args.baseline_checkpoint}")
    print(f"Frame counts to evaluate: {frame_counts}")
    print(f"Output base directory: {base_output_dir}")
    print("="*80)
    
    # Run benchmark for each frame count
    for num_frames in frame_counts:
        print(f"\n{'='*80}")
        print(f"Evaluating with {num_frames} frames")
        print(f"{'='*80}")
        
        # Create output directory for this frame count
        output_dir = base_output_dir / f"frames_{num_frames}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Dynamic batch size: Reduce for higher frames to avoid OOM
        bs = 1 if num_frames > 4 else args.batch_size

        # Build command
        cmd = [
            sys.executable,
            "experiments/benchmark_against_anycam.py",
            "--exp2_model", args.model_path,
            "--baseline_checkpoint", args.baseline_checkpoint,
            "--dataset", "lightspeed",
            "--num_frames", str(num_frames),
            "--save_dir", str(output_dir),
            "--batch_size", str(bs),
            "--device", args.device,
        ]
        
        if args.lightspeed_dir:
            cmd.extend(["--lightspeed_dir", args.lightspeed_dir])
        
        if args.max_samples:
            cmd.extend(["--max_samples", str(args.max_samples)])
        
        print(f"\n[RUN] Command: {' '.join(cmd)}\n")
        
        # Run the benchmark
        try:
            result = subprocess.run(cmd, check=True, capture_output=False)
            print(f"\n[SUCCESS] Completed evaluation for {num_frames} frames (bs={bs})")
            print(f"Results saved to: {output_dir}")
        except subprocess.CalledProcessError as e:
            print(f"\n[ERROR] Failed to evaluate {num_frames} frames (bs={bs})")
            print(f"Exit code: {e.returncode}")
            print(f"Continuing with next frame count...")
            continue
    
    print(f"\n{'='*80}")
    print("MULTI-FRAME BENCHMARK COMPLETE")
    print(f"{'='*80}")
    print(f"\nAll results saved to: {base_output_dir}")
    print(f"Individual results in: frames_2/, frames_3/, ..., frames_8/")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()

