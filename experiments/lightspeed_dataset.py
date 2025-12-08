#!/usr/bin/env python3
"""
LightSpeed Dataset Loader for Pose Evaluation

Loads the LightSpeed validation dataset from DynPose-100k for benchmarking
pose estimation accuracy.

Dataset structure:
- poses.pkl: Dict[seq_name, np.ndarray(num_frames, 3, 4)]
- frames-24fps/<seq_name>/images/<frame_num>.png

Author: AI Assistant for Kalman's Master's Thesis
Date: October 16, 2025
"""

import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import cv2

import torch
from torch.utils.data import Dataset

from experiments.dataset_paths import get_lightspeed_root


class LightSpeedDataset(Dataset):
    """
    PyTorch Dataset for LightSpeed validation sequences.
    
    Loads consecutive frame pairs with ground truth poses for evaluation.
    """
    
    def __init__(
        self,
        lightspeed_dir: str = None,
        num_frames: int = 2,
        image_size: Tuple[int, int] = (480, 640),  # (H, W)
        extract_all_pairs: bool = True,
        sequence_filter: Optional[List[str]] = None,
    ):
        """
        Args:
            lightspeed_dir: Root directory of LightSpeed dataset (defaults to dataset_paths.LIGHTSPEED_ROOT)
            num_frames: Number of consecutive frames per sample
            image_size: Target image size (H, W)
            extract_all_pairs: If True, extract all consecutive pairs
            sequence_filter: Optional list of sequence names to include
        """
        if lightspeed_dir is None:
            lightspeed_dir = get_lightspeed_root()
        self.lightspeed_dir = Path(lightspeed_dir)
        self.num_frames = num_frames
        self.image_size = image_size
        self.extract_all_pairs = extract_all_pairs
        
        # Load poses
        poses_file = self.lightspeed_dir / "poses.pkl"
        with open(poses_file, 'rb') as f:
            self.poses_dict = pickle.load(f)
        
        print(f"[LIGHTSPEED] Loaded poses for {len(self.poses_dict)} sequences")
        
        # Filter sequences if specified
        if sequence_filter is not None:
            self.poses_dict = {k: v for k, v in self.poses_dict.items() 
                              if k in sequence_filter}
            print(f"[LIGHTSPEED] Filtered to {len(self.poses_dict)} sequences")
        
        # Get sequence names
        self.sequence_names = sorted(self.poses_dict.keys())
        
        # Frames directory
        self.frames_dir = self.lightspeed_dir / "frames-24fps"
        
        # Build pair index
        if self.extract_all_pairs:
            self._build_pair_index()
        else:
            # One pair per sequence (frames 0-1)
            self.pair_info = [(i, 0) for i in range(len(self.sequence_names))]
        
        print(f"[LIGHTSPEED] Total frame pairs: {len(self.pair_info)}")
    
    def _build_pair_index(self):
        """Build index of all consecutive frame pairs."""
        self.pair_info = []  # List of (sequence_idx, start_frame)
        
        for seq_idx, seq_name in enumerate(self.sequence_names):
            poses = self.poses_dict[seq_name]
            num_sequence_frames = poses.shape[0]
            
            # Compute number of consecutive pairs
            n_pairs = (num_sequence_frames - self.num_frames + 1) if self.num_frames <= num_sequence_frames else 0
            
            # For consecutive pairs: (0,1), (2,3), (4,5), ...
            if self.extract_all_pairs:
                n_pairs = num_sequence_frames // self.num_frames
            
            for pair_idx in range(n_pairs):
                start_frame = pair_idx * self.num_frames if self.extract_all_pairs else pair_idx
                self.pair_info.append((seq_idx, start_frame))
        
        print(f"[LIGHTSPEED] Built pair index: {len(self.pair_info)} pairs from {len(self.sequence_names)} sequences")
    
    def __len__(self):
        return len(self.pair_info)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Load a frame pair and corresponding ground truth poses.
        
        Returns:
            dict with keys:
                - 'imgs': torch.Tensor [num_frames, 3, H, W] in range [0, 1]
                - 'poses': torch.Tensor [num_frames, 4, 4] - camera poses
                - 'projs': torch.Tensor [num_frames, 3, 3] - identity (no intrinsics provided)
                - 'sequence_name': str
                - 'frame_indices': List[int]
        """
        seq_idx, start_frame = self.pair_info[idx]
        seq_name = self.sequence_names[seq_idx]
        
        # Load frames
        frame_indices = list(range(start_frame, start_frame + self.num_frames))
        frames = self._load_frames(seq_name, frame_indices)
        
        # Convert frames to tensor
        imgs = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).float() / 255.0
        
        # Load ground truth poses
        poses_3x4 = self.poses_dict[seq_name][frame_indices]  # [num_frames, 3, 4]
        
        # Convert 3x4 to 4x4 homogeneous matrices
        poses_4x4 = self._convert_to_4x4(poses_3x4)
        poses = torch.from_numpy(poses_4x4).float()
        
        # Create dummy projection matrices (no intrinsics in LightSpeed)
        projs = torch.eye(3).unsqueeze(0).repeat(self.num_frames, 1, 1)
        
        return {
            'imgs': imgs,
            'poses': poses,
            'projs': projs,
            'sequence_name': seq_name,
            'frame_indices': frame_indices,
        }
    
    def _load_frames(self, seq_name: str, frame_indices: List[int]) -> List[np.ndarray]:
        """Load frame images for a sequence."""
        seq_dir = self.frames_dir / seq_name / "images"
        
        frames = []
        for frame_idx in frame_indices:
            frame_path = seq_dir / f"{frame_idx:05d}.png"
            
            if not frame_path.exists():
                raise FileNotFoundError(f"Frame not found: {frame_path}")
            
            # Load image
            frame = cv2.imread(str(frame_path))
            
            if frame is None:
                raise ValueError(f"Could not read frame: {frame_path}")
            
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Resize if needed
            if frame.shape[:2] != self.image_size:
                frame = cv2.resize(frame, (self.image_size[1], self.image_size[0]))
            
            frames.append(frame)
        
        return frames
    
    def _convert_to_4x4(self, poses_3x4: np.ndarray) -> np.ndarray:
        """
        Convert 3x4 poses to 4x4 homogeneous transformation matrices.
        
        Args:
            poses_3x4: Array of shape (num_frames, 3, 4)
            
        Returns:
            Array of shape (num_frames, 4, 4)
        """
        num_frames = poses_3x4.shape[0]
        poses_4x4 = np.zeros((num_frames, 4, 4), dtype=poses_3x4.dtype)
        
        # Copy 3x4 part
        poses_4x4[:, :3, :] = poses_3x4
        
        # Add bottom row [0, 0, 0, 1]
        poses_4x4[:, 3, 3] = 1.0
        
        return poses_4x4
    
    def get_sequence_info(self) -> Dict[str, int]:
        """Get information about sequences in the dataset."""
        info = {}
        for seq_name in self.sequence_names:
            num_frames = self.poses_dict[seq_name].shape[0]
            info[seq_name] = num_frames
        return info


if __name__ == "__main__":
    # Test the dataset loader
    print("Testing LightSpeed dataset loader...")
    
    dataset = LightSpeedDataset(
        num_frames=2,
        extract_all_pairs=True,
    )
    
    print(f"\nDataset size: {len(dataset)} pairs")
    
    # Load first sample
    print("\nLoading first sample...")
    sample = dataset[0]
    
    print(f"  imgs shape: {sample['imgs'].shape}")
    print(f"  poses shape: {sample['poses'].shape}")
    print(f"  projs shape: {sample['projs'].shape}")
    print(f"  sequence: {sample['sequence_name']}")
    print(f"  frames: {sample['frame_indices']}")
    
    print(f"\n  First pose:\n{sample['poses'][0]}")
    print(f"\n  Second pose:\n{sample['poses'][1]}")
    
    # Check a few more samples
    print(f"\nChecking samples...")
    for i in [0, len(dataset)//2, len(dataset)-1]:
        sample = dataset[i]
        print(f"  Sample {i}: {sample['sequence_name']}, frames {sample['frame_indices']}")
    
    print("\n✓ LightSpeed dataset loader working correctly!")

