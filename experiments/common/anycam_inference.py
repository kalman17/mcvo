#!/usr/bin/env python3
"""
AnyCam Inference Module for Experiments

This module provides reusable AnyCam inference functionality for various experiments,
including model loading, caching, and pose estimation on frame pairs.
"""

import numpy as np
import torch
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
import sys
import time

# Add project paths for AnyCam imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# AnyCam imports
from anycam.scripts.anycam_demo import load_anycam, process_video


class AnyCamInferenceEngine:
    """
    Handles AnyCam model loading and inference operations for experiments.
    """
    
    def __init__(self, model_path: str = "pretrained_models/anycam_seq8"):
        """
        Initialize the inference engine.
        
        Args:
            model_path: Path to AnyCam model directory
        """
        self.model_path = model_path
        self.model = None
        self.criterion = None
        self.is_loaded = False
        
        # Get absolute path to the workspace root directory
        self.workspace_root = Path(__file__).parent.parent.parent.resolve()
        self.full_model_path = self.workspace_root / model_path
        
        print(f"[INIT] AnyCam Inference Engine")
        print(f"[INIT] Model path: {self.full_model_path}")
    
    def load_model(self, force_reload: bool = False) -> bool:
        """
        Load the AnyCam model (cached after first load).
        
        Args:
            force_reload: Force reload even if already loaded
            
        Returns:
            True if successfully loaded, False otherwise
        """
        if self.is_loaded and not force_reload:
            print(f"[MODEL] Using cached AnyCam model")
            return True
        
        try:
            # Clear GPU memory before loading
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            print(f"[MODEL] Loading AnyCam model from {self.full_model_path}")
            
            # **ANYCAM CALL**: Load the pre-trained AnyCam model
            self.model, self.criterion = load_anycam(self.full_model_path)
            
            if self.model is not None:
                # Move model to GPU and set to evaluation mode
                if torch.cuda.is_available():
                    self.model = self.model.cuda().eval()
                    print(f"[MODEL] Model loaded on GPU")
                    if hasattr(torch.cuda, 'get_device_properties'):
                        gpu_props = torch.cuda.get_device_properties(0)
                        memory_total = gpu_props.total_memory / 1e9
                        memory_allocated = torch.cuda.memory_allocated() / 1e9
                        print(f"[MODEL] GPU memory: {memory_total:.1f}GB total, {memory_allocated:.1f}GB allocated")
                else:
                    self.model = self.model.eval()
                    print(f"[MODEL] Model loaded on CPU")
                
                self.is_loaded = True
                print(f"[OK] AnyCam model successfully loaded")
                return True
            else:
                print(f"[ERROR] Model is None after loading!")
                return False
                
        except Exception as e:
            print(f"[ERROR] Failed to load AnyCam model: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def run_inference_on_pair(self, frame_pair: List[np.ndarray], 
                             pair_name: str = "pair",
                             ba_refinement: bool = False) -> Optional[Dict[str, Any]]:
        """
        Run AnyCam inference on a single pair of frames.
        
        Args:
            frame_pair: List of 2 frames as numpy arrays
            pair_name: Name identifier for this pair
            ba_refinement: Whether to enable bundle adjustment refinement
            
        Returns:
            Dictionary with inference results or None if failed
        """
        if not self.is_loaded:
            if not self.load_model():
                print(f"[ERROR] Could not load model for inference")
                return None
        
        if len(frame_pair) != 2:
            print(f"[ERROR] Expected 2 frames, got {len(frame_pair)}")
            return None
        
        try:
            # Preprocess frames: convert to float32 [0,1] format expected by AnyCam
            formatted_frames = []
            for frame in frame_pair:
                if frame.max() > 1:
                    # Convert from uint8 [0,255] to float32 [0,1]
                    formatted_frames.append(frame.astype(np.float32) / 255.0)
                else:
                    # Already in [0,1] range
                    formatted_frames.append(frame.astype(np.float32))
            
            print(f"   [INFERENCE] Processing {pair_name}...")
            
            # **ANYCAM CALL**: Main inference - runs neural network on frame pair
            trajectory, projection, extras_dict, ba_extras = process_video(
                self.model,          # The loaded AnyCam neural network model
                self.criterion,      # Loss criterion for the model  
                formatted_frames,    # Our two input frames [frame1, frame2]
                ba_refinement=ba_refinement  # Bundle adjustment flag
            )
            
            # Convert results to numpy arrays
            trajectory_np = self._convert_to_numpy(trajectory)
            projection_np = self._convert_to_numpy(projection)
            
            # Clear GPU memory after processing
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return {
                'trajectory': trajectory_np,    # Camera pose transformations
                'projection': projection_np,    # Depth/projection information
                'extras_dict': extras_dict,     # Additional inference outputs
                'ba_extras': ba_extras,         # Bundle adjustment outputs
                'pair_name': pair_name,         # Identifier
                'ba_refinement': ba_refinement  # Whether BA was used
            }
            
        except Exception as e:
            print(f"   [ERROR] Inference failed for {pair_name}: {e}")
            # Clear GPU memory on error too
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return None
    
    def run_batch_inference(self, frame_pairs: List[Tuple[str, List[np.ndarray]]], 
                           ba_refinement: bool = False) -> Dict[str, Dict[str, Any]]:
        """
        Run AnyCam inference on multiple frame pairs.
        
        Args:
            frame_pairs: List of (pair_name, [frame1, frame2]) tuples
            ba_refinement: Whether to enable bundle adjustment refinement
            
        Returns:
            Dictionary mapping pair_name to inference results
        """
        if not self.is_loaded:
            if not self.load_model():
                print(f"[ERROR] Could not load model for batch inference")
                return {}
        
        print(f"[BATCH] Running inference on {len(frame_pairs)} frame pairs...")
        
        results = {}
        for i, (pair_name, frames) in enumerate(frame_pairs, 1):
            print(f"   [{i}/{len(frame_pairs)}] Processing {pair_name}...")
            
            result = self.run_inference_on_pair(frames, pair_name, ba_refinement)
            if result:
                results[pair_name] = result
                print(f"   [OK] {pair_name}: trajectory shape {result['trajectory'].shape}")
            else:
                print(f"   [FAIL] {pair_name}: inference failed")
        
        print(f"[BATCH] Completed: {len(results)}/{len(frame_pairs)} successful")
        return results
    
    def extract_relative_pose(self, trajectory: np.ndarray) -> np.ndarray:
        """
        Extract relative pose transformation from AnyCam trajectory output.
        
        AnyCam returns absolute poses in 'trajectory' with shape (2, 4, 4):
        - trajectory[0]: absolute pose of first frame  
        - trajectory[1]: absolute pose of second frame
        
        We compute the relative transformation: T_01 = trajectory[1] @ inv(trajectory[0])
        
        Args:
            trajectory: AnyCam trajectory output (2, 4, 4)
            
        Returns:
            Relative transformation matrix (4, 4)
        """
        if len(trajectory) >= 2:
            T_world_0 = trajectory[0]  # Absolute pose of frame 0
            T_world_1 = trajectory[1]  # Absolute pose of frame 1
            T_01 = T_world_1 @ np.linalg.inv(T_world_0)  # Relative transform 0→1
            return T_01
        else:
            raise ValueError(f"Expected trajectory with at least 2 poses, got {len(trajectory)}")
    
    def _convert_to_numpy(self, data) -> np.ndarray:
        """Convert PyTorch tensors or lists to numpy arrays."""
        if isinstance(data, list) and len(data) > 0:
            if hasattr(data[0], 'cpu'):  # Check if it's a PyTorch tensor
                # Convert each tensor to numpy and stack them
                return np.stack([item.cpu().numpy() for item in data])
            else:
                # Already numpy arrays
                return np.array(data)
        elif hasattr(data, 'cpu'):
            # Single PyTorch tensor
            return data.cpu().numpy()
        else:
            # Already numpy or other format
            return np.array(data)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return {
            'model_path': str(self.full_model_path),
            'is_loaded': self.is_loaded,
            'cuda_available': torch.cuda.is_available(),
            'device': 'cuda' if torch.cuda.is_available() and self.is_loaded else 'cpu'
        }
    
    def clear_cache(self):
        """Clear model cache and free memory."""
        if self.is_loaded:
            del self.model
            del self.criterion
            self.model = None
            self.criterion = None
            self.is_loaded = False
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            print(f"[CLEAR] Model cache cleared")


class PairwiseInferenceManager:
    """
    Higher-level manager for running inference on various frame pair combinations.
    """
    
    def __init__(self, inference_engine: AnyCamInferenceEngine):
        """
        Initialize with an inference engine.
        
        Args:
            inference_engine: AnyCamInferenceEngine instance
        """
        self.engine = inference_engine
    
    def generate_consecutive_pairs(self, frames: List[np.ndarray]) -> List[Tuple[str, List[np.ndarray]]]:
        """Generate consecutive frame pairs: (0,1), (1,2), (2,3), etc."""
        pairs = []
        for i in range(len(frames) - 1):
            pair_name = f"pair{i}{i+1}"
            pair_frames = [frames[i], frames[i+1]]
            pairs.append((pair_name, pair_frames))
        return pairs
    
    def generate_reverse_pairs(self, frames: List[np.ndarray]) -> List[Tuple[str, List[np.ndarray]]]:
        """Generate reverse consecutive pairs: (1,0), (2,1), (3,2), etc."""
        pairs = []
        for i in range(len(frames) - 1):
            pair_name = f"pair{i+1}{i}"
            pair_frames = [frames[i+1], frames[i]]
            pairs.append((pair_name, pair_frames))
        return pairs
    
    def generate_long_range_pairs(self, frames: List[np.ndarray]) -> List[Tuple[str, List[np.ndarray]]]:
        """Generate long-range pairs: (0, last), (last, 0)."""
        if len(frames) < 3:
            return []
        
        pairs = []
        last_idx = len(frames) - 1
        
        # Forward long-range: first to last
        pairs.append((f"pair0{last_idx}", [frames[0], frames[last_idx]]))
        
        # Backward long-range: last to first
        pairs.append((f"pair{last_idx}0", [frames[last_idx], frames[0]]))
        
        return pairs
    
    def generate_all_pairs(self, frames: List[np.ndarray]) -> List[Tuple[str, List[np.ndarray]]]:
        """Generate all possible frame pairs (can be expensive for many frames)."""
        pairs = []
        n = len(frames)
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    pair_name = f"pair{i}{j}"
                    pair_frames = [frames[i], frames[j]]
                    pairs.append((pair_name, pair_frames))
        
        return pairs
    
    def run_cycle_consistency_inference(self, frames: List[np.ndarray]) -> Dict[str, Dict[str, Any]]:
        """
        Run inference for cycle consistency testing.
        Generates the minimal set of pairs needed for comprehensive cycle analysis.
        """
        print(f"[CYCLE] Generating frame pairs for cycle consistency testing...")
        
        # Generate efficient pairs for cycle consistency
        pairs = []
        pairs.extend(self.generate_consecutive_pairs(frames))     # Forward consecutive
        pairs.extend(self.generate_reverse_pairs(frames))        # Backward consecutive  
        pairs.extend(self.generate_long_range_pairs(frames))     # Direct long-range
        
        print(f"[CYCLE] Generated {len(pairs)} essential pairs for cycle consistency")
        
        return self.engine.run_batch_inference(pairs, ba_refinement=False)


def create_inference_engine(model_path: str = "pretrained_models/anycam_seq8") -> AnyCamInferenceEngine:
    """
    Factory function to create and initialize an AnyCam inference engine.
    
    Args:
        model_path: Path to AnyCam model directory
        
    Returns:
        Initialized AnyCamInferenceEngine
    """
    engine = AnyCamInferenceEngine(model_path)
    return engine


def create_pairwise_manager(model_path: str = "pretrained_models/anycam_seq8") -> PairwiseInferenceManager:
    """
    Factory function to create a complete pairwise inference manager.
    
    Args:
        model_path: Path to AnyCam model directory
        
    Returns:
        PairwiseInferenceManager with initialized engine
    """
    engine = create_inference_engine(model_path)
    manager = PairwiseInferenceManager(engine)
    return manager 