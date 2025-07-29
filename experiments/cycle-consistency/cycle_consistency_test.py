#!/usr/bin/env python3
"""
Cycle Consistency Testing Script for AnyCam

This script performs cycle consistency analysis on AnyCam pose estimates by testing
if composed pose chains match direct pose estimates and if full cycles return to identity.

Usage:
    python cycle_consistency_test.py --input /path/to/video.mp4 --frames 3
    python cycle_consistency_test.py --input /path/to/jpeg_folder/ --frames 5 --name custom_experiment
    python cycle_consistency_test.py --input /path/to/data --frames 4 --gt-dir /path/to/gt
"""

import argparse
import json
import math
import numpy as np
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

# PyTorch imports
import torch

# Add project paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import common experiment modules
from common.data_loader import ExperimentDataManager
from common.anycam_inference import create_pairwise_manager

# Import AnyCam modules for geometry operations
try:
    from minipytorch3d.rotation_conversions import matrix_to_axis_angle
    from anycam.utils.geometry import se3_ensure_numerical_accuracy
    from anycam.common.geometry import get_grid_xy
    from minipytorch3d.transform3d import Transform3d
    from minipytorch3d.rotation_conversions import matrix_to_axis_angle, axis_angle_to_matrix
except ImportError as e:
    print(f"Error importing AnyCam modules: {e}")
    print("Make sure you're running from the project root directory")
    sys.exit(1)


def parse_args():
    """Parse command line arguments for cycle consistency testing"""
    parser = argparse.ArgumentParser(
        description="Cycle Consistency Testing for AnyCam",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single video test
  python cycle_consistency_test.py --input /path/to/video.mp4 --frames 3
  
  # Custom experiment with ground truth
  python cycle_consistency_test.py --input /path/to/jpeg_folder/ --frames 4 --name custom_exp --gt-dir /path/to/gt
        """
    )
    
    # Core input/output arguments
    parser.add_argument('--input', type=str, required=True,
                        help='Input video file or image folder')
    parser.add_argument('--name', type=str, default=None,
                        help='Experiment name (auto-generated if not provided)')
    parser.add_argument('--output-dir', type=str, default='experiments/cycle-consistency/results',
                        help='Base output directory for results')
    
    # Frame extraction parameters
    parser.add_argument('--frames', type=int, default=3,
                        help='Number of frames to extract')
    parser.add_argument('--start-frame', type=int, default=0,
                        help='Starting frame index')
    parser.add_argument('--skip-frames', type=int, default=1,
                        help='Number of frames to skip between extractions')
    
    # Model parameters
    parser.add_argument('--model-path', type=str, default='pretrained_models/anycam_seq8',
                        help='Path to AnyCam model directory')
    
    # Ground truth integration
    parser.add_argument('--gt-dir', type=str, default=None,
                        help='Directory containing ground truth camera poses for evaluation')
    
    return parser.parse_args()


def se3_log_distance(T1: np.ndarray, T2: np.ndarray) -> float:
    """
    Compute SE(3) geodesic distance between two 4x4 pose matrices.
    
    This function measures the true geometric distance between poses in the SE(3) manifold using the Lie algebra representation.
    The SE(3) Lie algebra (se(3)) is a vector space where distances can be computed naturally as Euclidean norms.
    
    Detailed calculation:
    1. Convert input numpy arrays to PyTorch tensors for efficient computation.
    2. Compute the inverse of T1: T1_inv = torch.inverse(T1_torch)
       - This inverts the 4x4 transformation matrix.
    3. Compute relative transformation: T_rel = T1_inv @ T2_torch
       - T_rel represents the transformation from T1 to T2 in SE(3).
    4. Extract rotation component: R_rel = T_rel[:3, :3] (3x3 rotation matrix)
    5. Extract translation component: t_rel = T_rel[:3, 3] (3x1 translation vector)
    6. Compute SO(3) log map for rotation:
       - Convert rotation matrix to axis-angle: omega = matrix_to_axis_angle(R_rel.unsqueeze(0)).squeeze(0)
       - omega is a 3D vector where the direction is the rotation axis and magnitude is the angle in radians.
    7. Compute norms:
       - Translation norm: trans_norm = torch.norm(t_rel)  # Euclidean norm of translation vector
         - Measures the straight-line distance between origins in meters (assuming metric scale).
       - Rotation norm: rot_norm = torch.norm(omega)  # Magnitude of axis-angle vector (rotation angle in radians)
         - Measures the geodesic distance on the SO(3) manifold (shortest rotation angle).
    8. Combine into SE(3) distance: distance = torch.sqrt(trans_norm**2 + rot_norm**2)
       - This is the norm of the twist vector in se(3) Lie algebra.
       - Significance: Provides a single scalar metric that combines rotation and translation errors in a geometrically meaningful way.
         - Invariant to the choice of representation (unlike direct matrix norms).
         - Units: Roughly combines meters (translation) and radians (rotation), but interpretable as geodesic distance on the manifold.
    
    Args:
        T1, T2: 4x4 numpy arrays representing SE(3) transformations
        
    Returns:
        Scalar distance in SE(3) space (float)
    """
    # Convert numpy arrays to PyTorch tensors
    T1_torch = torch.from_numpy(T1).float()
    T2_torch = torch.from_numpy(T2).float()
    
    # Compute relative transformation: T_rel = T1^-1 @ T2
    T1_inv = torch.inverse(T1_torch)
    T_rel = T1_inv @ T2_torch
    
    # Extract rotation and translation components from relative transformation
    R_rel = T_rel[:3, :3]  # 3x3 rotation matrix
    t_rel = T_rel[:3, 3]   # 3x1 translation vector
    
    # Compute SO(3) log for rotation (axis-angle representation)
    omega = matrix_to_axis_angle(R_rel.unsqueeze(0)).squeeze(0)  # Shape: (3,)
    
    # For SE(3) log distance, we combine translation and rotation errors
    trans_norm = torch.norm(t_rel)          # Translation magnitude
    rot_norm = torch.norm(omega)            # Rotation angle in radians
    
    # Combined SE(3) distance
    distance = torch.sqrt(trans_norm**2 + rot_norm**2)
    
    return distance.item()


def so3_log_angle_deg(R1: np.ndarray, R2: np.ndarray) -> float:
    """
    Compute SO(3) log distance between two 3x3 rotation matrices.
    
    This function computes the geodesic distance on the SO(3) manifold, which is the shortest rotation angle
    between two orientations. Measured in the Lie algebra so(3), which is the tangent space at identity.
    
    Detailed calculation:
    1. Convert input numpy arrays to PyTorch tensors.
    2. Compute relative rotation: R_rel = R1^T @ R2
       - Transpose of R1 acts like inverse since rotations are orthogonal.
    3. Convert to axis-angle: omega = matrix_to_axis_angle(R_rel.unsqueeze(0)).squeeze(0)
       - omega: 3D vector, direction=axis, magnitude=angle in radians.
    4. Compute angle: angle_rad = torch.norm(omega)
       - Norm gives the rotation angle (geodesic distance on SO(3)).
    5. Convert to degrees: angle_deg = angle_rad * (180.0 / math.pi)
       - For human-interpretable units (degrees instead of radians).
    
    Significance:
    - Measures pure rotational difference, ignoring translation.
    - Geometrically meaningful: smallest angle needed to rotate from R1 to R2.
    - Units: Degrees, easy to interpret (e.g., 5° error is small, 90° is large).
    - Invariant to rotation representation; computed in Lie algebra for proper metric.
    
    Args:
        R1, R2: 3x3 numpy arrays representing rotation matrices
        
    Returns:
        Rotation angle in degrees (float)
    """
    # Convert to PyTorch tensors
    R1_torch = torch.from_numpy(R1).float()
    R2_torch = torch.from_numpy(R2).float()
    
    # Compute relative rotation: R_rel = R1^T @ R2
    R_rel = R1_torch.transpose(-2, -1) @ R2_torch
    
    # Convert to axis-angle representation (SO(3) log)
    omega = matrix_to_axis_angle(R_rel.unsqueeze(0)).squeeze(0)  # Shape: (3,)
    
    # The magnitude of omega is the rotation angle in radians
    angle_rad = torch.norm(omega)
    
    # Convert to degrees for interpretability
    angle_deg = angle_rad * (180.0 / math.pi)
    
    return angle_deg.item()


def translation_distance(T1: np.ndarray, T2: np.ndarray) -> float:
    """
    Compute Euclidean distance between translation components of two poses.
    
    This is a simple L2 norm between translation vectors, ignoring rotation.
    
    Detailed calculation:
    1. Extract translations: t1 = T1[:3, 3], t2 = T2[:3, 3]
       - These are the position vectors in the 4x4 matrices.
    2. Compute difference: t2 - t1
    3. Compute norm: np.linalg.norm(t2 - t1)
       - Euclidean distance between positions.
    
    Significance:
    - Measures pure positional difference in 3D space.
    - Units: Meters (assuming metric scale).
    - Useful for isolating translation error from rotation.
    - Not a full manifold distance; use with SE(3)/SO(3) metrics for complete analysis.
    
    Args:
        T1, T2: 4x4 numpy arrays representing SE(3) transformations
        
    Returns:
        Translation distance (float)
    """
    t1 = T1[:3, 3]  # Extract translation vector from T1
    t2 = T2[:3, 3]  # Extract translation vector from T2
    
    # Compute Euclidean distance between translations
    return np.linalg.norm(t2 - t1)


def analyze_error_distribution(errors: List[float]) -> Dict[str, Any]:
    """
    Analyze the distribution of errors and compute statistics.
    
    This function computes various statistical measures on a list of error values.
    Useful for understanding overall performance beyond single metrics.
    
    Args:
        errors: List of error values
        
    Returns:
        Dictionary with statistical analysis
    """
    if not errors:
        return {'count': 0, 'mean': 0, 'std': 0}
    
    errors_array = np.array(errors)
    
    return {
        'count': len(errors),
        'mean': float(np.mean(errors_array)),
        'std': float(np.std(errors_array)),
        'median': float(np.median(errors_array)),
        'min': float(np.min(errors_array)),
        'max': float(np.max(errors_array)),
        'q25': float(np.percentile(errors_array, 25)),
        'q75': float(np.percentile(errors_array, 75))
    }


class CycleConsistencyTester:
    """
    Cycle consistency testing class focused on geometric analysis of AnyCam pose estimates.
    """
    
    def __init__(self, input_path: str, num_frames: int = 3, start_frame: int = 0, 
                 skip_frames: int = 1, experiment_name: Optional[str] = None,
                 model_path: str = "pretrained_models/anycam_seq8",
                 gt_dir: Optional[str] = None, output_dir: str = "experiments/cycle-consistency/results"):
        """
        Initialize the cycle consistency tester.
        
        Args:
            input_path: Path to video file or image folder
            num_frames: Number of frames to extract and test
            start_frame: Starting frame index
            skip_frames: Number of frames to skip between extractions
            experiment_name: Custom name for this experiment
            model_path: Path to AnyCam model directory
            gt_dir: Optional ground truth directory
            output_dir: Base output directory
        """
        self.input_path = Path(input_path)
        self.num_frames = num_frames
        self.start_frame = start_frame
        self.skip_frames = skip_frames
        self.model_path = model_path
        self.gt_dir = gt_dir
        
        # Generate experiment name if not provided
        if experiment_name is None:
            input_name = self.input_path.stem
            experiment_name = f"{input_name}_f{num_frames}_s{start_frame}_k{skip_frames}"
        
        # Create unique output directory
        timestamp = int(time.time())
        self.experiment_name = f"{experiment_name}_{timestamp}"
        self.base_dir = Path(output_dir) / self.experiment_name
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"[INIT] Cycle Consistency Experiment: {self.experiment_name}")
        print(f"[INIT] Output directory: {self.base_dir}")
        
        # Initialize data manager and inference manager
        self.data_manager = ExperimentDataManager()
        self.inference_manager = create_pairwise_manager(model_path)
        
        # Load experiment data
        self.frames, self.gt_data = self.data_manager.load_experiment_data(
            str(self.input_path), num_frames, start_frame, skip_frames, gt_dir
        )
        
        # Extract poses from ground truth data if available
        self.gt_poses = None
        if self.gt_data and 'poses' in self.gt_data:
            # Convert poses to 4x4 format and extract only the frames we need
            raw_poses = self.gt_data['poses']  # Shape: (N, 3, 4) for DynPose
            poses_4x4 = []
            for i in range(num_frames):
                frame_idx = start_frame + i * skip_frames
                if frame_idx < len(raw_poses):
                    pose_4x4 = np.eye(4)
                    pose_4x4[:3, :] = raw_poses[frame_idx]  # Copy 3x4 into top part of 4x4
                    poses_4x4.append(pose_4x4)
            self.gt_poses = poses_4x4
        
        print(f"[OK] Loaded {len(self.frames)} frames")
        if self.gt_poses:
            print(f"[OK] Loaded {len(self.gt_poses)} ground truth poses")
    
    def run_inference(self) -> Dict[str, Dict]:
        """Run AnyCam inference on frame pairs for cycle consistency testing."""
        print(f"\n[INFERENCE] Running AnyCam on frame pairs...")
        
        # Use the pairwise inference manager to run cycle consistency inference
        results = self.inference_manager.run_cycle_consistency_inference(self.frames)
        
        print(f"[OK] Inference completed: {len(results)} successful pairs")
        return results
    
    def compute_cycle_consistency_metrics(self, inference_results: Dict[str, Dict]) -> Dict[str, Any]:
        """
        Comprehensive cycle consistency analysis.
        
        Tests performed:
        1. Forward Chain Test: Compose P01 @ P12 @ ... @ P(N-1)N vs Direct P0N
        2. Backward Chain Test: Compose P(N-1)(N-2) @ ... @ P10 vs Direct P(N-1)0  
        3. Full Cycle Close: Test P01 @ P12 @ ... @ P(N-1)0 ≈ Identity
        
        For each test, we compute:
        - SE(3) geodesic distance: Full manifold distance combining rotation and translation.
        - SO(3) rotation angle: Pure rotational error in degrees.
        - Translation distance: Pure positional error (Euclidean norm).
        
        These metrics are computed in Lie algebra spaces for geometric accuracy:
        - SE(3): Norm of twist vector (translation + rotation axis-angle).
        - SO(3): Norm of skew-symmetric matrix log (rotation angle).
        
        Args:
            inference_results: Results from AnyCam inference on all pairs
            
        Returns:
            Comprehensive metrics dictionary with all error measurements
        """
        print(f"\n[ANALYSIS] Computing cycle consistency metrics...")
        
        # Initialize metrics structure
        metrics = {
            'errors': {},
            'statistical_summary': {},
        }
        
        def extract_relative_pose(pair_name: str) -> Optional[np.ndarray]:
            """Extract relative pose transformation from inference results."""
            if pair_name in inference_results and inference_results[pair_name] is not None:
                trajectory = inference_results[pair_name]['trajectory']
                return self.inference_manager.engine.extract_relative_pose(trajectory)
            return None
        
        # =============================================================================
        # TEST 1: FORWARD CYCLE CONSISTENCY 
        # =============================================================================
        print(f"[TEST 1] Forward: P01 @ P12 @ ... @ P{self.num_frames-2}{self.num_frames-1} vs P0{self.num_frames-1}")
        
        # Build forward chain
        forward_chain = []
        for i in range(self.num_frames - 1):
            pose = extract_relative_pose(f"pair{i}{i+1}")
            if pose is not None:
                forward_chain.append(pose)
            else:
                print(f"   [WARN] Missing pair{i}{i+1}")
                
        # Get direct forward pose
        direct_forward = extract_relative_pose(f"pair0{self.num_frames-1}")
        
        if len(forward_chain) == self.num_frames - 1 and direct_forward is not None:
            composed_forward = self._compose_poses(forward_chain)
            error1 = se3_log_distance(composed_forward, direct_forward)
            rot_error1 = so3_log_angle_deg(composed_forward[:3, :3], direct_forward[:3, :3])
            trans_error1 = translation_distance(composed_forward, direct_forward)
            
            print(f"   SE(3) Error: {error1:.6f}")
            
            metrics['errors']['test1_se3'] = error1
            metrics['errors']['test1_rotation_deg'] = rot_error1
            metrics['errors']['test1_translation'] = trans_error1
            
            # Ground truth comparison if available:

            # Purpose: If ground truth poses exist, we perform two comparisons:
            # 1. GT consistency (gt_error1): Measures if the ground truth itself is cycle-consistent
            #    by comparing GT composed chain vs GT direct long-range pose. This should be near zero
            #    and serves as a sanity check for GT data quality.
            # 2. Prediction vs GT (pred_vs_gt1): Compares model's composed pose chain to the 
            #    corresponding GT composed chain. This evaluates how well the model's predictions
            #    match the actual ground truth transformations, beyond just internal consistency.
            # Why: Provides external validation - low pred_vs_gt indicates model accuracy relative
            #      to real poses, while gt_error verifies the benchmark itself.
            if self.gt_poses and len(self.gt_poses) >= self.num_frames:
                gt_forward_chain = []
                for i in range(self.num_frames - 1):
                    gt_rel = np.matmul(self.gt_poses[i+1], np.linalg.inv(self.gt_poses[i]))
                    gt_forward_chain.append(gt_rel)
                
                gt_composed = self._compose_poses(gt_forward_chain)
                gt_direct = np.matmul(self.gt_poses[self.num_frames-1], np.linalg.inv(self.gt_poses[0]))
                
                gt_error1 = se3_log_distance(gt_composed, gt_direct)
                pred_vs_gt1 = se3_log_distance(composed_forward, gt_composed)
                
                metrics['errors']['test1_gt_se3'] = gt_error1
                metrics['errors']['test1_pred_vs_gt_se3'] = pred_vs_gt1
                
                print(f"   GT SE(3) Error: {gt_error1:.6f}")
                print(f"   Pred vs GT SE(3): {pred_vs_gt1:.6f}")
        else:
            print(f"   [FAIL] Missing components for Test 1")
        
        # =============================================================================
        # TEST 2: BACKWARD CYCLE CONSISTENCY 
        # =============================================================================
        print(f"\n[TEST 2] Backward: P{self.num_frames-1}{self.num_frames-2} @ ... @ P21 @ P10 vs P{self.num_frames-1}0")
        
        # Build backward chain
        backward_chain = []
        for i in range(self.num_frames - 1, 0, -1):
            pose = extract_relative_pose(f"pair{i}{i-1}")
            if pose is not None:
                backward_chain.append(pose)
            else:
                print(f"   [WARN] Missing pair{i}{i-1}")
                
        # Get direct backward pose
        direct_backward = extract_relative_pose(f"pair{self.num_frames-1}0")
        
        if len(backward_chain) == self.num_frames - 1 and direct_backward is not None:
            composed_backward = self._compose_poses(backward_chain)
            error2 = se3_log_distance(composed_backward, direct_backward)
            rot_error2 = so3_log_angle_deg(composed_backward[:3, :3], direct_backward[:3, :3])
            trans_error2 = translation_distance(composed_backward, direct_backward)
            
            print(f"   SE(3) Error: {error2:.6f}")
            
            metrics['errors']['test2_se3'] = error2
            metrics['errors']['test2_rotation_deg'] = rot_error2
            metrics['errors']['test2_translation'] = trans_error2
            
            # Ground truth comparison if available:

            # Purpose: Similar to Test 1, but for backward direction:
            # 1. GT consistency (gt_error2): Verifies ground truth backward chain vs direct backward.
            #    Should be near zero; checks if GT maintains consistency in reverse traversal.
            # 2. Prediction vs GT (pred_vs_gt2): Compares model's backward composed chain to 
            #    GT's backward composed chain. Evaluates model accuracy in reverse direction.
            # Why: Backward consistency often reveals different error patterns (e.g., accumulation bias).
            #      Comparing directions helps identify if errors are symmetric or direction-dependent.
            if self.gt_poses and len(self.gt_poses) >= self.num_frames:
                gt_backward_chain = []
                for i in range(self.num_frames - 1, 0, -1):
                    gt_rel = np.matmul(self.gt_poses[i-1], np.linalg.inv(self.gt_poses[i]))
                    gt_backward_chain.append(gt_rel)
                
                gt_composed = self._compose_poses(gt_backward_chain)
                gt_direct = np.matmul(self.gt_poses[0], np.linalg.inv(self.gt_poses[self.num_frames-1]))
                
                gt_error2 = se3_log_distance(gt_composed, gt_direct)
                pred_vs_gt2 = se3_log_distance(composed_backward, gt_composed)
                
                metrics['errors']['test2_gt_se3'] = gt_error2
                metrics['errors']['test2_pred_vs_gt_se3'] = pred_vs_gt2
                
                print(f"   GT SE(3) Error: {gt_error2:.6f}")
                print(f"   Pred vs GT SE(3): {pred_vs_gt2:.6f}")
        else:
            print(f"   [FAIL] Missing components for Test 2")
        
        # =============================================================================
        # TEST 3: FULL CYCLE CLOSE
        # =============================================================================
        print(f"\n[TEST 3] Full cycle: P01 @ P12 @ ... @ P{self.num_frames-1}0 vs Identity")
        
        if len(forward_chain) == self.num_frames - 1 and direct_backward is not None:
            # Build cycle: forward chain + closing backward step
            cycle_chain = forward_chain + [direct_backward]
            full_cycle = self._compose_poses(cycle_chain)
            identity = np.eye(4)
            error3 = se3_log_distance(full_cycle, identity)
            rot_error3 = so3_log_angle_deg(full_cycle[:3, :3], identity[:3, :3])
            trans_error3 = translation_distance(full_cycle, identity)
            
            print(f"   SE(3) Error: {error3:.6f}")
            
            metrics['errors']['test3_se3'] = error3
            metrics['errors']['test3_rotation_deg'] = rot_error3
            metrics['errors']['test3_translation'] = trans_error3
            
            # Ground truth comparison if available:

            # Purpose: For the full cycle closure:
            # 1. GT consistency (gt_error3): Measures if GT full cycle (forward + backward closing)
            #    returns close to identity. Should be ~zero; validates GT loop closure quality.
            # 2. Prediction vs GT (pred_vs_gt3): Compares model's full cycle composition to 
            #    GT's full cycle composition. Evaluates overall trajectory closure accuracy.
            # Why: Cycle closure is a strong test of cumulative error; comparing to GT shows
            #      if model's drift matches real-world behavior or reveals systematic biases.
            if self.gt_poses and len(self.gt_poses) >= self.num_frames:
                gt_cycle_chain = []
                for i in range(self.num_frames - 1):
                    gt_rel = np.matmul(self.gt_poses[i+1], np.linalg.inv(self.gt_poses[i]))
                    gt_cycle_chain.append(gt_rel)
                gt_closing = np.matmul(self.gt_poses[0], np.linalg.inv(self.gt_poses[self.num_frames-1]))
                gt_cycle_chain.append(gt_closing)
                
                gt_full_cycle = self._compose_poses(gt_cycle_chain)
                gt_error3 = se3_log_distance(gt_full_cycle, identity)
                pred_vs_gt3 = se3_log_distance(full_cycle, gt_full_cycle)
                
                metrics['errors']['test3_gt_se3'] = gt_error3
                metrics['errors']['test3_pred_vs_gt_se3'] = pred_vs_gt3
                
                print(f"   GT SE(3) Error: {gt_error3:.6f}")
                print(f"   Pred vs GT SE(3): {pred_vs_gt3:.6f}")
        else:
            print(f"   [FAIL] Missing components for Test 3")
        
        # =============================================================================
        # STATISTICAL SUMMARY 
        # =============================================================================
        print(f"\n[ANALYSIS] Computing Statistical Summary...")
        
        # Collect all computed errors for analysis
        all_errors = []
        for key, value in metrics['errors'].items():
            if isinstance(value, (int, float)) and not np.isnan(value):
                all_errors.append(value)
        
        if all_errors:
            error_stats = analyze_error_distribution(all_errors)
            metrics['statistical_summary'] = error_stats
            
            
            print(f"   Mean error: {error_stats['mean']:.6f} ± {error_stats['std']:.6f}")
        
        print(f"\n[DONE] Cycle consistency analysis completed")
        return metrics
    
    def _compose_poses(self, pose_list: List[np.ndarray]) -> np.ndarray:
        """
        Compose (multiply) a list of 4x4 transformation matrices.
        
        Args:
            pose_list: List of 4x4 transformation matrices
            
        Returns:
            Single 4x4 matrix representing the composed transformation
        """
        # Start with identity matrix
        composed = np.eye(4)
        
        # Multiply each pose matrix in sequence
        for P in pose_list:
            composed = np.matmul(composed, P)
        
        return composed
    
    def save_results(self, metrics: Dict[str, Any]):
        """Save experiment results to files."""
        print(f"\n[SAVE] Saving results...")
        
        # Save comprehensive JSON data
        data_file = self.base_dir / "cycle_consistency_results.json"
        
        save_data = {
            'experiment_metadata': {
                'experiment_name': self.experiment_name,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'input_path': str(self.input_path),
                'num_frames': self.num_frames,
                'start_frame': self.start_frame,
                'skip_frames': self.skip_frames,
                'model_path': self.model_path,

                'ground_truth_available': self.gt_poses is not None
            },
            'cycle_consistency_metrics': self._convert_numpy(metrics)
        }
        
        with open(data_file, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)
        
        # Save markdown report
        self._save_markdown_report(metrics)
        
        print(f"[OK] Results saved to: {self.base_dir}")
    
    def _save_markdown_report(self, metrics: Dict[str, Any]):
        """Save a plain-text friendly markdown report of the cycle consistency analysis (no tables for basic text viewing)."""
        report_path = self.base_dir / "cycle_consistency_report.md"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            # Header (simple text block)
            f.write("# AnyCam Cycle Consistency Test Results\n\n")
            f.write(f"Experiment Name: {self.experiment_name}\n")
            f.write(f"Input: {self.input_path}\n")
            f.write(f"Frames Analyzed: {self.num_frames} (start: {self.start_frame}, skip: {self.skip_frames})\n")
            f.write(f"Ground Truth: {'Available' if self.gt_poses is not None else 'Not Available'}\n\n")
            
            f.write("---\n\n")  # Simple separator line
            
            # Test Results (bulleted structure)
            f.write("## Test Results\n\n")
            f.write("Quick Metric Notes:\n")
            f.write("- SE(3) Error: Geodesic distance combining rotation/translation (lower is better; unitless but sqrt(rad² + m²) interpretable).\n")
            f.write("- Rotation Error: Shortest angle difference in degrees (lower is better).\n")
            f.write("- Translation Error: Straight-line position difference (lower is better; in meters assuming scale).\n\n")
            
            # Test 1
            error1 = metrics['errors'].get('test1_se3', 'N/A')
            rot1 = metrics['errors'].get('test1_rotation_deg', 'N/A')
            trans1 = metrics['errors'].get('test1_translation', 'N/A')
            f.write("**Test 1 (Forward Chain):** Compose sequential vs. direct forward\n")
            f.write(f"  - SE(3) Error: {error1:.6f}\n")
            f.write(f"  - Rotation Error (°): {rot1:.3f}\n")
            f.write(f"  - Translation Error: {trans1:.6f}\n")
            if 'test1_gt_se3' in metrics['errors']:
                gt1 = metrics['errors']['test1_gt_se3']
                pred_gt1 = metrics['errors']['test1_pred_vs_gt_se3']
                f.write(f"  - GT SE(3) Error: {gt1:.6f}\n")
                f.write(f"  - Pred vs GT SE(3): {pred_gt1:.6f}\n")
            f.write("\n")
            
            # Test 2
            error2 = metrics['errors'].get('test2_se3', 'N/A')
            rot2 = metrics['errors'].get('test2_rotation_deg', 'N/A')
            trans2 = metrics['errors'].get('test2_translation', 'N/A')
            f.write("**Test 2 (Backward Chain):** Compose sequential vs. direct backward\n")
            f.write(f"  - SE(3) Error: {error2:.6f}\n")
            f.write(f"  - Rotation Error (°): {rot2:.3f}\n")
            f.write(f"  - Translation Error: {trans2:.6f}\n")
            if 'test2_gt_se3' in metrics['errors']:
                gt2 = metrics['errors']['test2_gt_se3']
                pred_gt2 = metrics['errors']['test2_pred_vs_gt_se3']
                f.write(f"  - GT SE(3) Error: {gt2:.6f}\n")
                f.write(f"  - Pred vs GT SE(3): {pred_gt2:.6f}\n")
            f.write("\n")
            
            # Test 3
            error3 = metrics['errors'].get('test3_se3', 'N/A')
            rot3 = metrics['errors'].get('test3_rotation_deg', 'N/A')
            trans3 = metrics['errors'].get('test3_translation', 'N/A')
            f.write("**Test 3 (Full Cycle):** Full loop vs. identity\n")
            f.write(f"  - SE(3) Error: {error3:.6f}\n")
            f.write(f"  - Rotation Error (°): {rot3:.3f}\n")
            f.write(f"  - Translation Error: {trans3:.6f}\n")
            if 'test3_gt_se3' in metrics['errors']:
                gt3 = metrics['errors']['test3_gt_se3']
                pred_gt3 = metrics['errors']['test3_pred_vs_gt_se3']
                f.write(f"  - GT SE(3) Error: {gt3:.6f}\n")
                f.write(f"  - Pred vs GT SE(3): {pred_gt3:.6f}\n")
            f.write("\n")
            
            f.write("---\n\n")
            
            # Statistical Summary (bulleted, with success rate explanation)
            if 'statistical_summary' in metrics:
                stats = metrics['statistical_summary']
                f.write("## Statistical Summary\n\n")
                f.write("Overview of all error values (across tests/metrics):\n")
                f.write(f"  - Total Tests/Metrics: {stats['count']}\n")
                f.write(f"  - Mean Error: {stats['mean']:.6f} ± {stats['std']:.6f} (SE(3) units)\n")
                f.write(f"  - Median Error: {stats['median']:.6f}\n")
                f.write(f"  - Error Range: [{stats['min']:.6f}, {stats['max']:.6f}]\n")
    
    def _convert_numpy(self, obj):
        """Recursive function for JSON serialization of numpy objects."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.number):
            return obj.item()
        elif isinstance(obj, dict):
            return {k: self._convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy(item) for item in obj]
        else:
            return obj
    
    def run_complete_experiment(self) -> bool:
        """
        Run the complete cycle consistency experiment.
        
        Returns:
            True if successful, False otherwise
        """
        print(f"\n{'='*80}")
        print(f"CYCLE CONSISTENCY EXPERIMENT")
        print(f"{'='*80}")
        print(f"Experiment: {self.experiment_name}")
        print(f"Input: {self.input_path}")
        print(f"Frames: {self.num_frames}")
        print(f"Output: {self.base_dir}")
        print(f"{'='*80}")
        
        try:
            # Step 1: Run AnyCam inference
            results = self.run_inference()
            if not results:
                print("[FAIL] No inference results obtained")
                return False
            
            # Step 2: Compute cycle consistency metrics
            metrics = self.compute_cycle_consistency_metrics(results)
            if not metrics:
                print("[FAIL] Failed to compute metrics")
                return False
            
            # Step 3: Save results
            self.save_results(metrics)
            
            return True
            
        except Exception as e:
            print(f"[FAIL] Experiment failed: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    """Main function for cycle consistency testing."""
    args = parse_args()
    
    print(f"[START] AnyCam Cycle Consistency Testing")
    print(f"[INFO] Input: {args.input}")
    print(f"[INFO] Ground Truth: {args.gt_dir if args.gt_dir else 'None'}")
    
    try:
        # Create and run experiment
        tester = CycleConsistencyTester(
            input_path=args.input,
            num_frames=args.frames,
            start_frame=args.start_frame,
            skip_frames=args.skip_frames,
            experiment_name=args.name,
            model_path=args.model_path,
            gt_dir=args.gt_dir,

            output_dir=args.output_dir
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