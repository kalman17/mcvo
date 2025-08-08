#!/usr/bin/env python3
"""
Focal Length Consistency Testing Script for AnyCam

This script performs focal length consistency analysis on AnyCam predictions by splitting
a video into batches of frames, running inference independently on each batch, and
comparing the predicted focal lengths across batches.

Usage:
    python focal_length_consistency_test.py --input /path/to/video.mp4 --batch-length 3
    python focal_length_consistency_test.py --input /path/to/jpeg_folder/ --batch-length 4 --max-batches 10 --name custom_experiment
    python focal_length_consistency_test.py --input /path/to/data --batch-length 3 --gt-dir /path/to/gt
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

def parse_args():
    """Parse command line arguments for focal length consistency testing"""
    parser = argparse.ArgumentParser(
        description="Focal Length Consistency Testing for AnyCam",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single video test with default batch length 2
  python focal_length_consistency_test.py --input /path/to/video.mp4
  
  # Custom experiment with ground truth and batch length 4
  python focal_length_consistency_test.py --input /path/to/jpeg_folder/ --batch-length 4 --name custom_exp --gt-dir /path/to/gt
        """
    )
    
    # Core input/output arguments
    parser.add_argument('--input', type=str, required=True,
                        help='Input video file or image folder')
    parser.add_argument('--name', type=str, default=None,
                        help='Experiment name (auto-generated if not provided)')
    parser.add_argument('--output-dir', type=str, default='experiments/focal-length-consistency/results',
                        help='Base output directory for results')
    
    # Batch parameters
    parser.add_argument('--batch-length', type=int, default=2,
                        help='Number of frames per batch')
    parser.add_argument('--max-batches', type=int, default=10,
                        help='Maximum number of batches to process (0 for all possible)')
    
    # Frame extraction parameters
    parser.add_argument('--start-frame', type=int, default=0,
                        help='Starting frame index')
    parser.add_argument('--skip-frames', type=int, default=1,
                        help='Number of frames to skip between extractions')
    
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
        for key in ['intrinsics', 'K', 'camera_matrix', 'cam_K']:
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

class FocalConsistencyTester:
    """
    Focal length consistency testing class for AnyCam predictions.
    """
    
    def __init__(self, input_path: str, batch_length: int = 3, max_batches: int = 20, start_frame: int = 0, 
                 skip_frames: int = 1, experiment_name: Optional[str] = None,
                 model_path: str = "pretrained_models/anycam_seq8",
                 gt_dir: Optional[str] = None, output_dir: str = "experiments/focal-length-consistency/results",
                 ba_refinement: bool = False):
        """
        Initialize the focal consistency tester.
        """
        self.input_path = Path(input_path)
        self.batch_length = batch_length
        self.max_batches = max_batches
        self.start_frame = start_frame
        self.skip_frames = skip_frames
        self.model_path = model_path
        self.gt_dir = gt_dir
        self.ba_refinement = ba_refinement
        
        # Generate experiment name if not provided
        if experiment_name is None:
            input_name = self.input_path.stem
            experiment_name = f"{input_name}_bl{batch_length}_mb{max_batches}_s{start_frame}_k{skip_frames}"
        
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
        
        # Calculate approximate number of frames needed
        frames_per_batch = batch_length
        total_frames_needed = frames_per_batch * max_batches if max_batches > 0 else 0  # 0 means load all
        
        # Load frames (load all if max_batches=0)
        self.frames, gt_data = self.data_manager.load_experiment_data(
            str(self.input_path), total_frames_needed, start_frame, skip_frames, gt_dir
        )
        
        print(f"[OK] Loaded {len(self.frames)} frames")
        self.gt_focal = None
        self.original_height = self.frames[0].shape[0] if self.frames else 1
        self.original_width = self.frames[0].shape[1] if self.frames else 1
        if gt_data:
            print(f"[DEBUG] GT data type: {type(gt_data)}")
            if isinstance(gt_data, dict):
                print(f"[DEBUG] GT data keys: {list(gt_data.keys())}")
            self.gt_focal = extract_focal_from_gt(gt_data)
            if self.gt_focal is not None:
                print(f"[OK] Extracted ground truth focal length: {self.gt_focal:.3f}")
            else:
                print(f"[WARN] Loaded GT data but could not extract focal length")
        else:
            print(f"[WARN] No GT data loaded")
    
    def split_into_batches(self) -> List[List[np.ndarray]]:
        """Split loaded frames into batches of specified length."""
        total_frames = len(self.frames)
        frames_per_batch = self.batch_length
        
        # Calculate maximum possible batches
        max_possible_batches = total_frames // frames_per_batch
        
        # Determine number of batches to use
        num_batches = min(self.max_batches, max_possible_batches) if self.max_batches > 0 else max_possible_batches
        
        batches = []
        for i in range(num_batches):
            start_idx = i * frames_per_batch
            end_idx = start_idx + frames_per_batch
            batch_frames = self.frames[start_idx:end_idx]
            if len(batch_frames) == frames_per_batch:
                batches.append(batch_frames)
            else:
                print(f"[WARN] Incomplete batch {i+1} discarded (only {len(batch_frames)} frames)")
                break  # Discard incomplete batch
        
        print(f"[OK] Split {total_frames} frames into {len(batches)} batches of {frames_per_batch} frames each")
        if total_frames % frames_per_batch != 0:
            discarded = total_frames % frames_per_batch
            print(f"[INFO] Discarded {discarded} leftover frames")
        
        return batches
    
    def run_inference_on_sequence(self, frames: List[np.ndarray], seq_name: str = "sequence", ba_refinement: bool = False) -> Optional[Dict[str, Any]]:
        """Run AnyCam inference on a sequence of frames (2+) using local hubconf interface."""
        if len(frames) < 2:
            print(f"[ERROR] Expected at least 2 frames, got {len(frames)}")
            return None
        
        try:
            # Preprocess frames: convert to float32 [0,1] format expected by AnyCam
            formatted_frames = []
            for frame in frames:
                if frame.max() > 1:
                    # Convert from uint8 [0,255] to float32 [0,1]
                    formatted_frames.append(frame.astype(np.float32) / 255.0)
                else:
                    # Already in [0,1] range
                    formatted_frames.append(frame.astype(np.float32))
            
            print(f"   [INFERENCE] Processing {seq_name} ({len(formatted_frames)} frames)...")
            
            # **SIMPLE APPROACH**: Use original inference engine but extract focal from projection matrix
            if not self.inference_engine.is_loaded:
                if not self.inference_engine.load_model():
                    print(f"[ERROR] Could not load model for inference")
                    return None
            
            # Apply proper resizing as in demo
            formatted_frames = format_frames(formatted_frames)
            
            print(f"   [INFERENCE] Processing {seq_name} ({len(formatted_frames)} frames)...")
            
            # **ANYCAM CALL**: Main inference - runs neural network on sequence
            trajectory, projection_matrix, extras_dict, ba_extras = process_video(
                self.inference_engine.model,          # The loaded AnyCam neural network model
                self.inference_engine.criterion,      # Loss criterion for the model  
                formatted_frames,    # Input frames
                ba_refinement=ba_refinement  # Bundle adjustment flag
            )
            
            # Convert results to numpy arrays
            trajectory_np = self.inference_engine._convert_to_numpy(trajectory)
            projection_np = self.inference_engine._convert_to_numpy(projection_matrix)
            
            # Extract focal length directly from the projection matrix
            # The projection matrix is a 3x3 camera intrinsics matrix where:
            # K = [[fx,  0, cx],
            #      [ 0, fy, cy], 
            #      [ 0,  0,  1]]
            # fx and fy are the focal lengths in pixel coordinates
            focal = None
            if projection_np is not None:
                K = projection_np
                
                if K.shape == (3, 3):
                    fx = float(K[0, 0])
                    fy = float(K[1, 1])
                    
                    # Use average of fx and fy, or just fx if they're very close
                    if abs(fx - fy) < 1e-6:
                        focal = fx
                    else:
                        focal = (fx + fy) / 2.0
                    
                    print(f"   [OK] Extracted focal length from projection matrix: {focal:.3f} (fx={fx:.3f}, fy={fy:.3f})")
                    
                    # Scale focal length to original image dimensions if needed
                    resized_width = formatted_frames[0].shape[1]
                    if resized_width != self.original_width:
                        focal_original = focal * (self.original_width / resized_width)
                        print(f"   [INFO] Scaled focal to original dimensions: {focal_original:.3f} (scale factor: {self.original_width / resized_width:.3f})")
                        focal = focal_original
                else:
                    print(f"   [ERROR] Unexpected projection matrix shape: {K.shape}")
                    return None
            else:
                print(f"   [ERROR] No projection matrix returned from inference")
                return None
            
            # Clear GPU memory after processing
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Convert trajectory to numpy if needed
            trajectory_np = []
            for pose in trajectory:
                if isinstance(pose, np.ndarray):
                    trajectory_np.append(pose)
                else:
                    trajectory_np.append(pose.cpu().numpy() if hasattr(pose, 'cpu') else np.array(pose))
            
            result = {
                'trajectory': trajectory_np,    # Camera pose transformations
                'projection': projection_np,    # Camera intrinsics matrix (3x3)
                'extras_dict': extras_dict,     # Additional inference outputs
                'ba_extras': ba_extras,         # Bundle adjustment outputs
                'focal_length': focal,          # Extracted focal length in pixels
                'seq_name': seq_name,           # Sequence identifier
                'frame_count': len(frames),     # Number of frames in sequence
                'original_width': self.original_width,  # Original image width
                'original_height': self.original_height,  # Original image height
                'ba_refinement': ba_refinement  # Whether BA was used
            }
            
            return result
            
        except Exception as e:
            print(f"   [ERROR] Inference failed for {seq_name}: {e}")
            print(f"   [DEBUG] Frame shapes: {[f.shape for f in formatted_frames[:3]]}")
            print(f"   [DEBUG] Frame dtypes: {[f.dtype for f in formatted_frames[:3]]}")
            print(f"   [DEBUG] CUDA available: {torch.cuda.is_available()}")
            if hasattr(self, '_hub_model'):
                try:
                    print(f"   [DEBUG] Model device: {next(iter(self._hub_model.parameters())).device}")
                except:
                    print(f"   [DEBUG] Could not determine model device")
            import traceback
            traceback.print_exc()
            # Clear GPU memory on error too
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return None
    
    def run_inference(self, batches: List[List[np.ndarray]]) -> List[float]:
        """Run AnyCam inference on each batch and extract focal lengths."""
        print(f"\n[INFERENCE] Running AnyCam on {len(batches)} batches...")
        
        focals = []
        for idx, batch_frames in enumerate(batches, 1):
            print(f"   [{idx}/{len(batches)}] Processing batch {idx} ({len(batch_frames)} frames)...")
            
            result = self.run_inference_on_sequence(
                batch_frames, f"batch{idx}", ba_refinement=self.ba_refinement
            )
            
            if result and 'focal_length' in result:
                focal = result['focal_length']
                focals.append(float(focal))
                print(f"   [OK] Batch {idx}: focal length {focal:.3f}")
            else:
                print(f"   [FAIL] Batch {idx}: no focal length found in results")
        
        print(f"[OK] Extracted {len(focals)} focal lengths")
        return focals
    
    def compute_consistency_metrics(self, focals: List[float]) -> Dict[str, Any]:
        """Compute comprehensive metrics on predicted focal lengths."""
        print(f"\n[ANALYSIS] Computing consistency metrics...")
        
        # Basic statistics on predicted focal lengths
        metrics = {
            'predicted_focals': focals,
            'stats': analyze_focal_distribution(focals),
        }
        
        # Print basic consistency analysis
        stats = metrics['stats']
        print(f"   Predicted Focal Statistics:")
        print(f"     Mean: {stats['mean']:.3f} ± {stats['std']:.3f}")
        print(f"     Median: {stats['median']:.3f}")
        print(f"     Range: {stats['min']:.3f} - {stats['max']:.3f}")
        print(f"     Coefficient of Variation: {stats['cv']:.3f} ({stats['std_percent']:.1f}%)")
        print(f"     IQR: {stats['iqr']:.3f}")
        
        if self.gt_focal is not None:
            # Compute errors relative to ground truth
            errors = [f - self.gt_focal for f in focals]  # Signed errors
            abs_errors = [abs(e) for e in errors]  # Absolute errors
            rel_errors = [abs(e / self.gt_focal) if self.gt_focal != 0 else 0 for e in errors]  # Relative errors
            
            # Error statistics
            error_stats = analyze_focal_distribution(abs_errors)
            rel_error_stats = analyze_focal_distribution(rel_errors)
            
            # Additional error metrics
            rmse = float(np.sqrt(np.mean(np.array(errors) ** 2)))  # Root Mean Square Error
            mae = float(np.mean(abs_errors))  # Mean Absolute Error
            mre = float(np.mean(rel_errors))  # Mean Relative Error
            
            # Consistency metrics
            consistency_score = 1.0 - (stats['cv'] / 2.0)  # Higher is more consistent (0-1 scale)
            
            metrics.update({
                'gt_focal': self.gt_focal,
                'errors': errors,  # Signed errors
                'absolute_errors': abs_errors,
                'relative_errors': rel_errors,
                'error_stats': error_stats,
                'relative_error_stats': rel_error_stats,
                'rmse': rmse,
                'mean_absolute_error': mae,
                'mean_relative_error': mre,
                'consistency_score': consistency_score,
                'accuracy_score': max(0, 1.0 - mre)  # Higher is more accurate (0-1 scale)
            })
            
            print(f"   Ground Truth Analysis:")
            print(f"     GT Focal: {self.gt_focal:.3f}")
            print(f"     RMSE: {rmse:.3f}")
            print(f"     MAE: {mae:.3f}")
            print(f"     MRE: {mre:.3f} ({mre*100:.1f}%)")
            print(f"     Consistency Score: {consistency_score:.3f}")
            print(f"     Accuracy Score: {metrics['accuracy_score']:.3f}")
            
            # Quality assessment
            if mre < 0.05:
                quality = "Excellent"
            elif mre < 0.10:
                quality = "Good"
            elif mre < 0.20:
                quality = "Fair"
            else:
                quality = "Poor"
            
            if stats['cv'] < 0.05:
                consistency = "Excellent"
            elif stats['cv'] < 0.10:
                consistency = "Good"
            elif stats['cv'] < 0.20:
                consistency = "Fair"
            else:
                consistency = "Poor"
            
            metrics['quality_assessment'] = {
                'accuracy_quality': quality,
                'consistency_quality': consistency
            }
            
            print(f"   Quality Assessment:")
            print(f"     Accuracy: {quality} (MRE: {mre*100:.1f}%)")
            print(f"     Consistency: {consistency} (CV: {stats['cv']:.3f})")
        
        print(f"[DONE] Analysis completed")
        return metrics
    
    def save_results(self, metrics: Dict[str, Any]):
        """Save experiment results to files."""
        print(f"\n[SAVE] Saving results...")
        
        # Save JSON data
        data_file = self.base_dir / "focal_consistency_results.json"
        
        save_data = {
            'experiment_metadata': {
                'experiment_name': self.experiment_name,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'input_path': str(self.input_path),
                'batch_length': self.batch_length,
                'max_batches': self.max_batches,
                'num_batches': len(metrics['predicted_focals']),
                'start_frame': self.start_frame,
                'skip_frames': self.skip_frames,
                'model_path': self.model_path,
                'gt_focal': self.gt_focal
            },
            'metrics': self._convert_numpy(metrics)
        }
        
        with open(data_file, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)
        
        # Save markdown report
        self._save_markdown_report(metrics)
        
        # Save visualization plot
        self._save_focal_plot(metrics)
        
        print(f"[OK] Results saved to: {self.base_dir}")
    
    def _save_markdown_report(self, metrics: Dict[str, Any]):
        """Save a markdown report of the results."""
        report_path = self.base_dir / "focal_consistency_report.md"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# AnyCam Focal Length Consistency Test Results\n\n")
            f.write(f"Experiment Name: {self.experiment_name}\n")
            f.write(f"Input: {self.input_path}\n")
            f.write(f"Batch Length: {self.batch_length}\n")
            f.write(f"Number of Batches: {len(metrics['predicted_focals'])}\n")
            f.write(f"Ground Truth Focal: {self.gt_focal if self.gt_focal is not None else 'N/A'}\n\n")
            
            f.write("## Predicted Focal Lengths\n\n")
            for i, focal in enumerate(metrics['predicted_focals'], 1):
                f.write(f"- Batch {i}: {focal:.3f}\n")
            
            if self.gt_focal is not None:
                f.write("\n## Absolute Errors\n\n")
                for i, err in enumerate(metrics.get('absolute_errors', []), 1):
                    f.write(f"- Batch {i}: {err:.3f}\n")
            
            f.write("\n## Statistical Summary (Predicted)\n\n")
            stats = metrics['stats']
            f.write(f"- Count: {stats['count']}\n")
            f.write(f"- Mean: {stats['mean']:.3f}\n")
            f.write(f"- Std: {stats['std']:.3f}\n")
            f.write(f"- Median: {stats['median']:.3f}\n")
            f.write(f"- Min: {stats['min']:.3f}\n")
            f.write(f"- Max: {stats['max']:.3f}\n")
            f.write(f"- Range: {stats['range']:.3f}\n")
            f.write(f"- Q25: {stats['q25']:.3f}\n")
            f.write(f"- Q75: {stats['q75']:.3f}\n")
            f.write(f"- IQR: {stats['iqr']:.3f}\n")
            f.write(f"- Coefficient of Variation: {stats['cv']:.3f}\n")
            f.write(f"- Std as % of Mean: {stats['std_percent']:.1f}%\n")
            
            if self.gt_focal is not None and 'error_stats' in metrics:
                f.write("\n## Ground Truth Comparison\n\n")
                f.write(f"- Ground Truth Focal: {self.gt_focal:.3f}\n")
                f.write(f"- RMSE: {metrics['rmse']:.3f}\n")
                f.write(f"- Mean Absolute Error: {metrics['mean_absolute_error']:.3f}\n")
                f.write(f"- Mean Relative Error: {metrics['mean_relative_error']:.3f} ({metrics['mean_relative_error']*100:.1f}%)\n")
                f.write(f"- Consistency Score: {metrics['consistency_score']:.3f}\n")
                f.write(f"- Accuracy Score: {metrics['accuracy_score']:.3f}\n")
                
                # Quality assessment
                if 'quality_assessment' in metrics:
                    f.write(f"- Accuracy Quality: {metrics['quality_assessment']['accuracy_quality']}\n")
                    f.write(f"- Consistency Quality: {metrics['quality_assessment']['consistency_quality']}\n")
                
                f.write("\n### Error Statistics\n\n")
                err_stats = metrics['error_stats']
                f.write(f"- Mean: {err_stats['mean']:.3f}\n")
                f.write(f"- Std: {err_stats['std']:.3f}\n")
                f.write(f"- Median: {err_stats['median']:.3f}\n")
                f.write(f"- Min: {err_stats['min']:.3f}\n")
                f.write(f"- Max: {err_stats['max']:.3f}\n")
                f.write(f"- Range: {err_stats['range']:.3f}\n")
                f.write(f"- Q25: {err_stats['q25']:.3f}\n")
                f.write(f"- Q75: {err_stats['q75']:.3f}\n")
                f.write(f"- IQR: {err_stats['iqr']:.3f}\n")
                
                f.write("\n### Relative Error Statistics\n\n")
                rel_err_stats = metrics['relative_error_stats']
                f.write(f"- Mean: {rel_err_stats['mean']:.3f} ({rel_err_stats['mean']*100:.1f}%)\n")
                f.write(f"- Std: {rel_err_stats['std']:.3f} ({rel_err_stats['std']*100:.1f}%)\n")
                f.write(f"- Median: {rel_err_stats['median']:.3f} ({rel_err_stats['median']*100:.1f}%)\n")
                f.write(f"- Min: {rel_err_stats['min']:.3f} ({rel_err_stats['min']*100:.1f}%)\n")
                f.write(f"- Max: {rel_err_stats['max']:.3f} ({rel_err_stats['max']*100:.1f}%)\n")
                f.write(f"- Range: {rel_err_stats['range']:.3f} ({rel_err_stats['range']*100:.1f}%)\n")
                f.write(f"- Q25: {rel_err_stats['q25']:.3f} ({rel_err_stats['q25']*100:.1f}%)\n")
                f.write(f"- Q75: {rel_err_stats['q75']:.3f} ({rel_err_stats['q75']*100:.1f}%)\n")
                f.write(f"- IQR: {rel_err_stats['iqr']:.3f} ({rel_err_stats['iqr']*100:.1f}%)\n")
    
    def _save_focal_plot(self, metrics: Dict[str, Any]):
        """Save a comprehensive plot visualizing focal length variation and errors."""
        plot_path = self.base_dir / "focal_variation_plot.png"
        
        focals = metrics['predicted_focals']
        batches = list(range(1, len(focals) + 1))
        
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Top plot: Focal length variation
        ax1.plot(batches, focals, marker='o', label='Predicted Focal', color='blue', linewidth=2, markersize=6)
        
        if self.gt_focal is not None:
            ax1.axhline(y=self.gt_focal, color='green', linestyle='--', label='GT Focal', linewidth=2)
            # Add error bars showing ±10% of GT
            gt_10_percent = self.gt_focal * 0.1
            ax1.fill_between(batches, self.gt_focal - gt_10_percent, self.gt_focal + gt_10_percent, 
                           alpha=0.2, color='green', label='±10% GT Range')
        
        ax1.set_xlabel('Batch Number')
        ax1.set_ylabel('Focal Length (pixels)')
        ax1.set_title('Focal Length Variation Across Batches')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Bottom plot: Errors relative to GT
        if self.gt_focal is not None and 'errors' in metrics:
            errors = metrics['errors']
            rel_errors = [e / self.gt_focal * 100 for e in errors]  # Convert to percentage
            
            ax2.bar(batches, rel_errors, color='red', alpha=0.7, label='Relative Error (%)')
            ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
            ax2.axhline(y=10, color='orange', linestyle='--', alpha=0.7, label='±10% Threshold')
            ax2.axhline(y=-10, color='orange', linestyle='--', alpha=0.7)
            
            ax2.set_xlabel('Batch Number')
            ax2.set_ylabel('Relative Error (%)')
            ax2.set_title('Relative Error vs Ground Truth')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Add statistics text
            stats_text = f"Mean Error: {np.mean(rel_errors):.1f}%\nStd Error: {np.std(rel_errors):.1f}%"
            ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[VIZ] Saved comprehensive focal variation plot to: {plot_path}")
    
    def _convert_numpy(self, obj):
        """Recursive numpy conversion for JSON serialization."""
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
        """Run the complete focal consistency experiment."""
        print(f"\n{'='*80}")
        print(f"FOCAL LENGTH CONSISTENCY EXPERIMENT")
        print(f"{'='*80}")
        print(f"Experiment: {self.experiment_name}")
        print(f"Input: {self.input_path}")
        print(f"Batch Length: {self.batch_length}")
        print(f"Max Batches: {self.max_batches}")
        print(f"Output: {self.base_dir}")
        print(f"{'='*80}")
        
        try:
            # Step 1: Split frames into batches
            batches = self.split_into_batches()
            if not batches:
                print("[FAIL] No batches created")
                return False
            
            # Step 2: Run inference on batches
            focals = self.run_inference(batches)
            if not focals:
                print("[FAIL] No focal lengths obtained")
                return False
            
            # Step 3: Compute metrics
            metrics = self.compute_consistency_metrics(focals)
            
            # Step 4: Save results
            self.save_results(metrics)
            
            return True
            
        except Exception as e:
            print(f"[FAIL] Experiment failed: {e}")
            import traceback
            traceback.print_exc()
            return False

def main():
    """Main function for focal length consistency testing."""
    args = parse_args()
    
    print(f"[START] AnyCam Focal Length Consistency Testing")
    print(f"[INFO] Input: {args.input}")
    print(f"[INFO] Ground Truth: {args.gt_dir if args.gt_dir else 'None'}")
    
    try:
        # Create and run experiment
        tester = FocalConsistencyTester(
            input_path=args.input,
            batch_length=args.batch_length,
            max_batches=args.max_batches,
            start_frame=args.start_frame,
            skip_frames=args.skip_frames,
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