#!/usr/bin/env python3
"""
Focal Length Consistency Testing Script for AnyCam

This script performs focal length consistency analysis on AnyCam predictions by splitting
a video into batches of frames, running inference independently on each batch, and
comparing the predicted focal lengths across batches.

Usage:
    python focal_length_consistency_test.py --videos-dir /path/to/videos/
    python focal_length_consistency_test.py --videos-dir /path/to/videos/ --gt-dir /path/to/gt/
"""

import argparse
import json
import numpy as np
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Any

# PyTorch imports
import torch

# Matplotlib for visualization
import matplotlib.pyplot as plt

# Add project paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import common experiment modules
from common.data_loader import ExperimentDataManager
from common.anycam_inference import create_inference_engine
from anycam.scripts.anycam_demo import process_video, format_frames  # Import for sequence processing and resize

import cv2  # For resize
from minipytorch3d.rotation_conversions import matrix_to_axis_angle
import json as _json

def parse_args():
    """Parse command line arguments for focal length consistency testing"""
    parser = argparse.ArgumentParser(
        description="Focal Length Consistency Testing for AnyCam",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run on all sequences in videos directory
  python focal_length_consistency_test.py --videos-dir /home/kalman/TUM/thesis/Objectron/videos/
  
  # Custom experiment with ground truth
  python focal_length_consistency_test.py --videos-dir /home/kalman/TUM/thesis/Objectron/videos/ --name custom_exp --gt-dir /home/kalman/TUM/thesis/Objectron/processed_gt/
        """
    )
    
    # Core input/output arguments
    parser.add_argument('--videos-dir', type=str, required=True,
                        help='Directory containing all video sequences')
    parser.add_argument('--name', type=str, default=None,
                        help='Experiment name (auto-generated if not provided)')
    parser.add_argument('--output-dir', type=str, default='experiments/focal-length-consistency/results',
                        help='Base output directory for results')
    
    # Batch parameters
    parser.add_argument('--batch-length', type=int, default=2,
                        help='Number of frames per batch')
    
    # Model parameters
    parser.add_argument('--model-path', type=str, default='pretrained_models/anycam_seq8',
                        help='Path to AnyCam model directory')
    parser.add_argument('--ba-refinement', action='store_true',
                        help='Enable bundle adjustment refinement')
    
    # Ground truth integration
    parser.add_argument('--gt-dir', type=str, default=None,
                        help='Directory containing ground truth data for evaluation')
    
    return parser.parse_args()

def extract_focal_from_gt(gt_data) -> Optional[float]:
    """
    Extract focal length from ground truth data.
    
    Attempts to find focal length in various possible formats.
    Assumes focal is constant across the sequence.
    """
    if isinstance(gt_data, dict):
        # Try common keys for intrinsics
        for key in ['intrinsics', 'K', 'camera_matrix', 'cam_K', 'intrinsics_per_frame']:
            if key in gt_data and isinstance(gt_data[key], (np.ndarray, list)):
                K = np.array(gt_data[key])
                if K.ndim == 2 and K.shape == (3,3):
                    fx = float(K[0,0])
                    fy = float(K[1,1])
                    # Return average of fx and fy, or just fx if they're very close
                    if abs(fx - fy) < 1e-6:
                        return fx
                    else:
                        return (fx + fy) / 2.0
                elif K.ndim == 3 and K.shape[1:] == (3,3):
                    fx = float(K[0,0,0])
                    fy = float(K[0,1,1])
                    if abs(fx - fy) < 1e-6:
                        return fx
                    else:
                        return (fx + fy) / 2.0
                elif K.ndim == 3 and K.shape[1] == 9:  # 3x3 flattened
                    K_reshaped = K.reshape(-1, 3, 3)
                    fx = float(K_reshaped[0,0,0])
                    fy = float(K_reshaped[0,1,1])
                    if abs(fx - fy) < 1e-6:
                        return fx
                    else:
                        return (fx + fy) / 2.0
    elif isinstance(gt_data, np.ndarray):
        if gt_data.ndim == 2 and gt_data.shape == (3,3):
            fx = float(gt_data[0,0])
            fy = float(gt_data[1,1])
            if abs(fx - fy) < 1e-6:
                return fx
            else:
                return (fx + fy) / 2.0
        elif gt_data.ndim == 3 and gt_data.shape[1:] == (3,3):
            fx = float(gt_data[0,0,0])
            fy = float(gt_data[0,1,1])
            if abs(fx - fy) < 1e-6:
                return fx
            else:
                return (fx + fy) / 2.0
    elif isinstance(gt_data, list) and len(gt_data) > 0:
        return extract_focal_from_gt(gt_data[0])
    
    print("[WARN] Could not extract focal length from GT data")
    print(f"[DEBUG] GT data type: {type(gt_data)}")
    if isinstance(gt_data, dict):
        print(f"[DEBUG] GT data keys: {list(gt_data.keys())}")
    return None

def analyze_focal_distribution(focals: List[float]) -> Dict[str, Any]:
    """
    Analyze the distribution of focal lengths and compute comprehensive statistics.
    """
    if not focals:
        return {'count': 0, 'mean': 0, 'std': 0}
    
    focals_array = np.array(focals)
    
    # Basic statistics
    mean_val = float(np.mean(focals_array))
    std_val = float(np.std(focals_array))
    median_val = float(np.median(focals_array))
    
    # Coefficient of variation (CV) - measures relative variability
    cv = float(std_val / mean_val) if mean_val != 0 else 0
    
    # Range and percentiles
    range_val = float(np.max(focals_array) - np.min(focals_array))
    q25 = float(np.percentile(focals_array, 25))
    q75 = float(np.percentile(focals_array, 75))
    iqr = q75 - q25  # Interquartile range
    
    return {
        'count': len(focals),
        'mean': mean_val,
        'std': std_val,
        'median': median_val,
        'min': float(np.min(focals_array)),
        'max': float(np.max(focals_array)),
        'range': range_val,
        'q25': q25,
        'q75': q75,
        'iqr': float(iqr),
        'cv': cv,  # Coefficient of variation
        'std_percent': float((std_val / mean_val) * 100) if mean_val != 0 else 0  # Std as percentage of mean
    }

def se3_log_distance(T_pred: np.ndarray, T_gt: np.ndarray) -> float:
    """
    Compute SE(3) log-map distance between two 4x4 poses.
    Returns sqrt(||t||^2 + ||omega||^2), where omega is axis-angle vector from SO(3) log.
    """
    T_pred_t = torch.from_numpy(np.asarray(T_pred)).float()
    T_gt_t = torch.from_numpy(np.asarray(T_gt)).float()
    T_rel = torch.inverse(T_gt_t) @ T_pred_t
    R_rel = T_rel[:3, :3]
    t_rel = T_rel[:3, 3]
    omega = matrix_to_axis_angle(R_rel[None, ...]).squeeze(0)
    trans_norm = torch.norm(t_rel)
    rot_norm = torch.norm(omega)
    distance = torch.sqrt(trans_norm ** 2 + rot_norm ** 2)
    return float(distance.item())

class FocalConsistencyTester:
    """
    Focal length consistency testing class for AnyCam predictions.
    """
    
    def __init__(self, videos_dir: str,
                 experiment_name: Optional[str] = None,
                 model_path: str = "pretrained_models/anycam_seq8",
                 gt_dir: Optional[str] = None, output_dir: str = "experiments/focal-length-consistency/results",
                 ba_refinement: bool = False):
        """
        Initialize the focal consistency tester.
        """
        self.videos_dir = Path(videos_dir)
        self.model_path = model_path
        self.gt_dir = gt_dir
        self.ba_refinement = ba_refinement
        
        # Generate experiment name if not provided
        if experiment_name is None:
            experiment_name = f"all_sequences_videos_1754669323"  # From your run
        # Create unique output directory
        timestamp = int(time.time())
        self.experiment_name = f"{experiment_name}_{timestamp}"
        self.base_dir = Path(output_dir) / self.experiment_name
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"[INIT] Focal Length Consistency Experiment: {self.experiment_name}")
        print(f"[INIT] Output directory: {self.base_dir}")
        print(f"[INIT] BA Refinement: {self.ba_refinement}")
        
        # Initialize data manager and inference engine
        self.data_manager = ExperimentDataManager()
        self.inference_engine = create_inference_engine(model_path)
        
        # Preload GT focal summary if available
        self.gt_focal_summary = None
        if self.gt_dir:
            summary_path = Path(self.gt_dir) / 'summary.json'
            if summary_path.exists():
                try:
                    with open(summary_path, 'r') as f:
                        self.gt_focal_summary = _json.load(f)
                    print(f"[GT] Loaded focal summary with {len(self.gt_focal_summary)} entries")
                except Exception as e:
                    print(f"[WARN] Failed to load focal summary: {e}")
    
    def run_complete_experiment(self) -> bool:
        """Run the complete focal consistency experiment."""
        print(f"\n{'='*80}")
        print(f"FOCAL LENGTH CONSISTENCY EXPERIMENT")
        print(f"{'='*80}")
        print(f"Experiment: {self.experiment_name}")
        print(f"Input: {self.videos_dir}")
        print(f"Output: {self.base_dir}")
        print(f"{'='*80}")
        
        try:
            # Step 1: Find all video sequences
            video_files = self.data_manager.frame_loader.find_video_files(str(self.videos_dir))
            if not video_files:
                print("[FAIL] No videos found")
                return False
            
            print(f"[OK] Found {len(video_files)} video sequences")
            
            for video_path in video_files:
                self.process_sequence(video_path)
            
            return True
            
        except Exception as e:
            print(f"[FAIL] Experiment failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def process_sequence(self, video_path: Path):
        """Process a single sequence."""
        sequence_name = video_path.stem
        print(f"\n[SEQUENCE] Processing {sequence_name}")
        
        # Load all frames for this video
        frames = self.data_manager.frame_loader.load_frames(str(video_path), num_frames=0)  # 0 = all frames
        
        # Restrict to first 20 frames max
        max_frames = min(len(frames), 20)
        frames = frames[:max_frames]
        
        # Split into up to 10 two-frame batches: (0,1), (2,3), ..., (18,19)
        batches = []
        for i in range(0, max_frames, 2):
            if len(batches) >= 10:
                break
            if i + 1 < max_frames:
                batches.append(frames[i:i+2])
        
        print(f"[OK] Split into {len(batches)} 2-frame batches")
        
        # Load GT
        gt_focal = None
        gt_poses = None
        if self.gt_dir:
            base_name = sequence_name.replace('_video', '')
            gt_json_path = Path(self.gt_dir) / f"{base_name}.json"
            # Load GT focal from summary.json
            if self.gt_focal_summary and base_name in self.gt_focal_summary:
                gt_focal = float(self.gt_focal_summary[base_name])
            else:
                print(f"[WARN] No GT focal found in summary for {base_name}")
            # Load GT poses from per-sequence JSON
            if gt_json_path.exists():
                try:
                    with open(gt_json_path, 'r') as f:
                        gt_raw = _json.load(f)
                    poses_list = gt_raw.get('poses', [])
                    # Expect flattened 4x4 lists -> reshape
                    gt_poses = []
                    for p in poses_list:
                        arr = np.array(p, dtype=np.float32)
                        if arr.size == 16:
                            gt_poses.append(arr.reshape(4, 4))
                    gt_poses = np.stack(gt_poses, axis=0) if gt_poses else None
                    if gt_poses is not None:
                        print(f"[GT] Loaded {gt_poses.shape[0]} GT poses from {gt_json_path.name}")
                except Exception as e:
                    print(f"[WARN] Failed to load GT poses from {gt_json_path}: {e}")
        
        # Run inference on batches
        predicted_focals = []
        focal_errors = []
        predicted_pose_errors = []
        batch_indices = []
        start_frame = 0
        for idx, batch in enumerate(batches, 1):
            print(f"   [{idx}/{len(batches)}] Processing batch {idx} ({len(batch)} frames)...")
            result = self.run_inference_on_sequence(batch, f"{sequence_name}_batch{idx}", ba_refinement=self.ba_refinement)
            if result:
                predicted_focals.append(result['focal_length'])
                
                # Calculate pose error using relative motion between the two frames
                predicted_poses = np.array(result['trajectory'])  # (2, 4, 4)
                if gt_poses is not None and start_frame + 1 < len(gt_poses):
                    gt_p0, gt_p1 = gt_poses[start_frame], gt_poses[start_frame+1]
                    if predicted_poses.shape[:2] == (2, 4):
                        # Predicted relative (assumed c2w): T1 * inv(T0)
                        T_pred_rel = predicted_poses[1] @ np.linalg.inv(predicted_poses[0])
                        # GT relative in both conventions: pick the smaller error
                        T_gt_rel_c2w = gt_p1 @ np.linalg.inv(gt_p0)
                        T_gt_rel_w2c = np.linalg.inv(gt_p1) @ gt_p0
                        e1 = se3_log_distance(T_pred_rel, T_gt_rel_c2w)
                        e2 = se3_log_distance(T_pred_rel, T_gt_rel_w2c)
                        predicted_pose_errors.append(float(min(e1, e2)))
                    else:
                        predicted_pose_errors.append(float('nan'))
                else:
                    predicted_pose_errors.append(float('nan'))
                
                # Focal error per batch (absolute difference)
                if gt_focal is not None and result['focal_length'] is not None:
                    focal_errors.append(abs(float(result['focal_length']) - gt_focal))
                else:
                    focal_errors.append(float('nan'))
 
                batch_indices.append(idx)
                
                start_frame += 2
        
        # Save results for this sequence
        self.save_sequence_results(sequence_name, predicted_focals, predicted_pose_errors, gt_focal, batch_indices, focal_errors)
    
    def run_inference_on_sequence(self, frames: List[np.ndarray], seq_name: str = "sequence", ba_refinement: bool = False) -> Optional[Dict[str, Any]]:
        """Run AnyCam inference on a sequence of frames (2+)."""
        if len(frames) < 2:
            print(f"[ERROR] Expected at least 2 frames, got {len(frames)}")
            return None
        
        # Ensure model is loaded
        if not self.inference_engine.is_loaded:
            if not self.inference_engine.load_model():
                return None
        
        # Preprocess frames
        formatted_frames = []
        for frame in frames:
            if frame.max() > 1:
                formatted_frames.append(frame.astype(np.float32) / 255.0)
            else:
                formatted_frames.append(frame.astype(np.float32))
        
        formatted_frames = format_frames(formatted_frames)
        
        trajectory, projection_matrix, extras_dict, ba_extras = process_video(
            self.inference_engine.model,
            self.inference_engine.criterion,
            formatted_frames,
            ba_refinement=ba_refinement
        )
        
        trajectory_np = trajectory.cpu().numpy() if hasattr(trajectory, 'cpu') else np.array(trajectory)
        projection_np = projection_matrix.cpu().numpy() if hasattr(projection_matrix, 'cpu') else np.array(projection_matrix)
        
        # Extract focal length from projection matrix (K)
        focal = None
        if projection_np.shape == (3, 3):
            fx = float(projection_np[0, 0])
            fy = float(projection_np[1, 1])
            focal = (fx + fy) / 2.0 if abs(fx - fy) > 1e-6 else fx
        
        return {
            'trajectory': trajectory_np.tolist(),  # Predicted poses as list for JSON
            'focal_length': focal
        }
    
    def save_sequence_results(self, sequence_name: str, predicted_focals: List[float], predicted_pose_errors: List[float], gt_focal: Optional[float], batch_indices: List[int], focal_errors: List[float]):
        """Save results and plots for a sequence."""
        seq_dir = self.base_dir / sequence_name
        seq_dir.mkdir(parents=True, exist_ok=True)
        
        # Compute focal error percentage if GT focal is available
        focal_errors_pct = []
        if gt_focal is not None and gt_focal != 0:
            for pf in predicted_focals:
                if pf is None:
                    focal_errors_pct.append(float('nan'))
                else:
                    focal_errors_pct.append(abs(float(pf) - gt_focal) / gt_focal * 100.0)
        else:
            focal_errors_pct = [float('nan')] * len(predicted_focals)

        # Save JSON
        data = {
            'predicted_focals': predicted_focals,
            'predicted_pose_errors': predicted_pose_errors,
            'gt_focal': gt_focal,
            'focal_errors': focal_errors,
            'focal_errors_pct': focal_errors_pct,
            'batch_indices': batch_indices
        }
        with open(seq_dir / 'results.json', 'w') as f:
            json.dump(data, f, indent=4)
         
        # Per-batch dual-axis plot: focal error % (bars) and pose error SE3 (line)
        fig, ax1 = plt.subplots(figsize=(10, 5))
        x = np.array(batch_indices)
        focal_pct_arr = np.array(focal_errors_pct, dtype=np.float32)
        pose_arr = np.array(predicted_pose_errors, dtype=np.float32)
        # Bars for focal %
        ax1.bar(x, np.nan_to_num(focal_pct_arr, nan=0.0), color='b', alpha=0.6, label='Focal Error (%)')
        ax1.set_xlabel('Batch index (pairs of frames)')
        ax1.set_ylabel('Focal Error (%)', color='b')
        ax1.tick_params(axis='y', labelcolor='b')
        # Twin axis for pose error (SE3 log)
        ax2 = ax1.twinx()
        ax2.plot(x, pose_arr, 'r-o', label='Pose Error (SE3 log)')
        ax2.set_ylabel('Pose Error (SE3 log distance)', color='r')
        ax2.tick_params(axis='y', labelcolor='r')
        # Legends
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, loc='upper right')
        plt.title('Per-batch focal vs pose error')
        plt.tight_layout()
        plt.savefig(seq_dir / 'errors_per_batch_dual_axis.png')
        plt.close()

        # (Removed per user request: no separate histograms file)
         
        print(f"[OK] Results saved for {sequence_name} in {seq_dir}")

def main():
    """Main function for focal length consistency testing."""
    args = parse_args()
    
    print(f"[START] AnyCam Focal Length Consistency Testing")
    print(f"[INFO] Videos Directory: {args.videos_dir}")
    print(f"[INFO] Ground Truth Directory: {args.gt_dir if args.gt_dir else 'None'}")
    
    try:
        # Create and run experiment
        tester = FocalConsistencyTester(
            videos_dir=args.videos_dir,
            experiment_name=args.name,
            model_path=args.model_path,
            gt_dir=args.gt_dir,
            output_dir=args.output_dir,
            ba_refinement=args.ba_refinement
        )
        
        success = tester.run_complete_experiment()
        
        if success:
            print(f"[SUCCESS] Experiment completed successfully!")
            print(f"[OUTPUT] Results saved to: {tester.base_dir}")
        else:
            print(f"[FAIL] Experiment failed!")
            return 1
        
        return 0
        
    except KeyboardInterrupt:
        print(f"\n[STOP] Interrupted by user")
        return 1
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1 


if __name__ == "__main__":
    exit(main())