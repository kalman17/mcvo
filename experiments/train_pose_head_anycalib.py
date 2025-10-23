#!/usr/bin/env python3
"""
=============================================================================
Pose Head Retraining Experiment with AnyCaLib Focal Length Injection
=============================================================================

EXPERIMENT GOAL:
----------------
Test whether we can successfully train a fresh pose head while using AnyCaLib 
for focal length prediction instead of AnyCam's original candidate system.

EXPERIMENT STEPS:
-----------------
1. Load pretrained AnyCam model
2. **DELETE existing pose_head** and create a **fresh randomly initialized** version
3. **FREEZE all layers** except the new pose_head (backbone, neck, other heads all frozen)
4. **INJECT AnyCaLib** focal length predictions directly into the training pipeline
5. Train on Objectron dataset (100 sequences, 2 frames per sequence initially)
6. Monitor loss convergence to verify the pose head can learn

ANYCALIB INTEGRATION:
---------------------
We will run AnyCaLib on frames to get focal length predictions, then use those
directly instead of the 32-candidate system. Two approaches are implemented:

  A) SINGLE FRAME: Run AnyCaLib only on first frame, assume constant focal length
     - Faster, simpler
     - Good for fixed camera sequences
     - **CURRENT DEFAULT**
  
  B) MULTI-FRAME AVERAGE: Run AnyCaLib on all frames, average the results
     - More robust
     - Slower
     - Use this if you want better focal length estimates

The injection point is marked with: ===== ANYCALIB INJECTION POINT =====

FOCAL LENGTH FLOW:
------------------
Original AnyCam:
  sequence_info_head → focal_enc → 32 candidates → weighted average

Modified (This Experiment):
  AnyCaLib.infer(image) → focal_px → single focal value → projection matrix

KEY ARCHITECTURAL CHANGES:
---------------------------
- anycam.py: Modified forward() to accept external focal length
- trainer.py: AnyCamWrapper now supports single focal length mode
- This script: Orchestrates training with AnyCaLib in the loop

Author: AI Assistant for Kalman's Master's Thesis
Date: October 10, 2025
Branch: experiment/pose-head-retraining-anycalib-focal

=============================================================================
TRAINING DATA CONFIGURATION (EASY TO CHANGE)
=============================================================================
To control training speed, change the number of frame pairs extracted per video:
- Go to ObjectronVideoDataset.__init__() around line 180
- Modify the parameter: max_pairs_per_video=5 (default)
  
  Examples:
  - max_pairs_per_video=5  → Extract 5 pairs per video (0-1, 2-3, 4-5, 6-7, 8-9)
  - max_pairs_per_video=10 → Extract 10 pairs per video (faster training, more data)
  - max_pairs_per_video=1  → Extract only 1 pair per video (very fast, less data)
  
This controls the total training samples: ~100 videos × max_pairs_per_video × 0.7 (train split)
=============================================================================
"""

import sys
import os
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler

# Add project paths
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "anycalib"))

# AnyCam imports
from anycam.models import make_pose_predictor, make_depth_predictor, make_depth_aligner
from anycam.loss import make_loss
from anycam.common.image_processor import make_image_processor
from anycam.trainer import make_proj_from_focal_length, induce_flow_dist, normalize_proj

# AnyCaLib import
try:
    from anycalib.model.anycalib_pretrained import AnyCalib
except ImportError:
    from anycalib.anycalib.model.anycalib_pretrained import AnyCalib

print("[INIT] Imports successful")


# =============================================================================
# DATASET SPLITTING UTILITIES
# =============================================================================

def create_train_val_test_split(
    num_videos: int,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[List[int], List[int], List[int]]:
    """
    Create deterministic train/val/test split for video indices.
    
    Args:
        num_videos: Total number of videos
        train_ratio: Fraction for training (default: 0.7)
        val_ratio: Fraction for validation (default: 0.15)
        test_ratio: Fraction for testing (default: 0.15)
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_indices, val_indices, test_indices)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
    
    # Create shuffled indices
    np.random.seed(seed)
    indices = np.random.permutation(num_videos).tolist()
    
    # Compute split points
    n_train = int(num_videos * train_ratio)
    n_val = int(num_videos * val_ratio)
    
    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]
    
    print(f"[SPLIT] Train: {len(train_indices)} videos, Val: {len(val_indices)} videos, Test: {len(test_indices)} videos")
    
    return train_indices, val_indices, test_indices


def save_dataset_split(split_dict: Dict, save_path: str):
    """Save train/val/test split indices to JSON file."""
    with open(save_path, 'w') as f:
        json.dump(split_dict, f, indent=2)
    print(f"[SPLIT] Saved dataset split to: {save_path}")


def load_dataset_split(split_path: str) -> Dict:
    """Load train/val/test split indices from JSON file."""
    with open(split_path, 'r') as f:
        split_dict = json.load(f)
    print(f"[SPLIT] Loaded dataset split from: {split_path}")
    return split_dict


# =============================================================================
# OBJECTRON DATASET LOADER
# =============================================================================

class ObjectronVideoDataset(Dataset):
    """
    PyTorch Dataset for Objectron video sequences with multi-pair extraction.
    
    Loads video frames and optionally ground truth camera poses/intrinsics from JSON files.
    Extracts ALL consecutive frame pairs from each video sequence.
    
    For example, a video with 10 frames generates pairs: (0-1), (2-3), (4-5), (6-7), (8-9)
    
    NOTE: Ground truth is NOT required for training (unsupervised flow reprojection loss).
          GT is only used for validation/monitoring if available.
    """
    
    def __init__(
        self, 
        videos_dir: str,
        gt_dir: Optional[str] = None,
        num_frames: int = 2,
        max_sequences: Optional[int] = None,
        image_size: Tuple[int, int] = (480, 640),  # (H, W)
        require_gt: bool = False,  # Set to False for unsupervised training
        video_indices: Optional[List[int]] = None,  # For train/val/test split
        extract_all_pairs: bool = True,  # Extract all consecutive pairs vs single pair
        max_pairs_per_video: int = 5,  # <<<< EASY TO CHANGE: Control how many pairs per video (e.g., 5 = pairs 0-1, 2-3, 4-5, 6-7, 8-9)
    ):
        """
        Args:
            videos_dir: Directory containing .MOV video files
            gt_dir: Directory containing .json ground truth files (optional)
            num_frames: Number of consecutive frames per sequence (default: 2)
            max_sequences: Limit dataset size (useful for debugging)
            image_size: Target image size (H, W)
            require_gt: If True, skip videos without GT. If False, load all videos.
            video_indices: Specific video indices to use (for train/val/test split)
            extract_all_pairs: If True, extract all consecutive pairs from each video
            max_pairs_per_video: Maximum number of frame pairs to extract per video (default: 5)
                                Example: 5 extracts pairs (0-1), (2-3), (4-5), (6-7), (8-9)
        """
        self.videos_dir = Path(videos_dir)
        self.gt_dir = Path(gt_dir) if gt_dir else None
        self.num_frames = num_frames
        self.image_size = image_size
        self.require_gt = require_gt
        self.extract_all_pairs = extract_all_pairs
        self.max_pairs_per_video = max_pairs_per_video  # <<<< Store parameter
        
        # Find all video files
        all_video_files = sorted(list(self.videos_dir.glob("*.MOV")) + 
                                 list(self.videos_dir.glob("*.mov")))
        
        # Apply video indices filter if provided (for train/val/test split)
        if video_indices is not None:
            all_video_files = [all_video_files[i] for i in video_indices if i < len(all_video_files)]
        
        if max_sequences is not None:
            all_video_files = all_video_files[:max_sequences]
        
        self.video_files = all_video_files
        print(f"[DATASET] Found {len(self.video_files)} video sequences")
        
        # Validate that GT files exist (optional)
        if self.require_gt and self.gt_dir:
            self._validate_dataset()
        else:
            print(f"[DATASET] Running in UNSUPERVISED mode (GT not required)")
        
        # ========================================================================
        # MULTI-PAIR EXTRACTION: Precompute all frame pairs
        # ========================================================================
        if self.extract_all_pairs:
            self._build_pair_index()
        else:
            # Legacy mode: one pair per video
            self.pair_info = [(i, 0) for i in range(len(self.video_files))]
        
        print(f"[DATASET] Total frame pairs available: {len(self.pair_info)}")
    
    def _build_pair_index(self):
        """
        Build index mapping from dataset index to (video_idx, start_frame).
        This enables extracting consecutive frame pairs from each video.
        
        Note: cv2.VideoCapture.get(CAP_PROP_FRAME_COUNT) is unreliable for some formats.
        We reduce the frame count by a small margin to avoid reading corrupted frames.
        
        The number of pairs extracted per video is limited by self.max_pairs_per_video.
        """
        self.pair_info = []  # List of (video_idx, start_frame_idx)
        
        for video_idx, video_path in enumerate(self.video_files):
            # Get video frame count
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                print(f"[WARN] Could not open video {video_path.name}, skipping")
                continue
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            
            # SAFETY MARGIN: Reduce frame count to avoid corrupted frames at end
            # cv2.VideoCapture.get(CAP_PROP_FRAME_COUNT) is unreliable for MOV files
            safe_total_frames = max(total_frames - 2, self.num_frames)
            
            # Compute number of consecutive pairs
            # For num_frames=2: pairs are (0,1), (2,3), (4,5), ..., (N-2, N-1)
            n_pairs_available = safe_total_frames // self.num_frames
            
            if n_pairs_available == 0:
                print(f"[WARN] Video {video_path.name} too short ({total_frames} frames), skipping")
                continue
            
            # <<<< LIMIT PAIRS PER VIDEO: Respect max_pairs_per_video parameter
            n_pairs = min(n_pairs_available, self.max_pairs_per_video)
            
            for pair_idx in range(n_pairs):
                start_frame = pair_idx * self.num_frames
                self.pair_info.append((video_idx, start_frame))
        
        print(f"[DATASET] Built pair index: {len(self.pair_info)} pairs from {len(self.video_files)} videos")
        print(f"[DATASET] Max pairs per video: {self.max_pairs_per_video}")
    
    def _validate_dataset(self):
        """Check that all videos have matching ground truth files."""
        valid_videos = []
        for video_path in self.video_files:
            # Try multiple naming patterns
            # Pattern 1: batch-10_0_video.MOV -> batch-10_0_video.json
            gt_path1 = self.gt_dir / f"{video_path.stem}.json"
            # Pattern 2: batch-10_0_video.MOV -> batch-10_0.json (remove "_video")
            stem_without_video = video_path.stem.replace("_video", "")
            gt_path2 = self.gt_dir / f"{stem_without_video}.json"
            
            if gt_path1.exists():
                valid_videos.append(video_path)
            elif gt_path2.exists():
                valid_videos.append(video_path)
            else:
                print(f"[WARN] No GT found for {video_path.name}, skipping")
        
        self.video_files = valid_videos
        print(f"[DATASET] {len(self.video_files)} sequences have valid GT")
    
    def __len__(self):
        """Return total number of frame pairs in the dataset."""
        return len(self.pair_info)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Load a sequence of frames and optionally ground truth data.
        
        Uses pair_info to map dataset index to (video_idx, start_frame).
        
        Returns:
            dict with keys:
                - 'imgs': torch.Tensor [num_frames, 3, H, W] in range [0, 1]
                - 'projs': torch.Tensor [num_frames, 3, 3] - dummy identity matrices (if no GT)
                - 'poses': torch.Tensor [num_frames, 4, 4] - dummy identity matrices (if no GT)
                - 'video_name': str - name of the video file
                - 'frame_indices': List[int] - indices of loaded frames
        """
        # Map dataset index to video and frame pair
        video_idx, start_frame = self.pair_info[idx]
        video_path = self.video_files[video_idx]
        
        # Load frames from video starting at start_frame
        frames, frame_indices = self._load_frames_from_video(video_path, start_frame=start_frame)
        
        # Convert frames to tensors
        imgs = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).float() / 255.0
        
        # Try to load ground truth (optional)
        projs = None
        poses = None
        
        if self.gt_dir:
            # Try multiple naming patterns
            gt_path1 = self.gt_dir / f"{video_path.stem}.json"
            stem_without_video = video_path.stem.replace("_video", "")
            gt_path2 = self.gt_dir / f"{stem_without_video}.json"
            
            gt_path = None
            if gt_path1.exists():
                gt_path = gt_path1
            elif gt_path2.exists():
                gt_path = gt_path2
            
            if gt_path:
                try:
                    with open(gt_path, 'r') as f:
                        gt_data = json.load(f)
                    projs = self._extract_projection_matrices(gt_data, frame_indices)
                    poses = self._extract_camera_poses(gt_data, frame_indices)
                    projs = torch.from_numpy(projs).float()
                    poses = torch.from_numpy(poses).float()
                except Exception as e:
                    print(f"[WARN] Failed to load GT for {video_path.name}: {e}")
        
        # If no GT available, create dummy placeholders (not used in unsupervised training)
        if projs is None:
            # Create dummy identity projection matrices
            projs = torch.eye(3).unsqueeze(0).repeat(self.num_frames, 1, 1)
        
        if poses is None:
            # Create dummy identity pose matrices
            poses = torch.eye(4).unsqueeze(0).repeat(self.num_frames, 1, 1)
        
        return {
            'imgs': imgs,  # [num_frames, 3, H, W]
            'projs': projs,  # [num_frames, 3, 3]
            'poses': poses,  # [num_frames, 4, 4]
            'video_name': video_path.stem,
            'frame_indices': frame_indices,
        }
    
    def _load_frames_from_video(self, video_path: Path, start_frame: int = 0) -> Tuple[List[np.ndarray], List[int]]:
        """
        Load consecutive frames from video starting at start_frame.
        
        Args:
            video_path: Path to video file
            start_frame: Starting frame index
            
        Returns:
            Tuple of (frames, frame_indices)
        """
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Generate frame indices
        frame_indices = list(range(start_frame, start_frame + self.num_frames))
        
        # Ensure we don't go beyond video length
        if frame_indices[-1] >= total_frames:
            frame_indices = list(range(max(0, total_frames - self.num_frames), total_frames))
        
        frames = []
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret:
                raise ValueError(f"Could not read frame {frame_idx} from {video_path}")
            
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Resize if needed
            if frame.shape[:2] != self.image_size:
                frame = cv2.resize(frame, (self.image_size[1], self.image_size[0]))
            
            frames.append(frame)
        
        cap.release()
        return frames, frame_indices
    
    def _extract_projection_matrices(self, gt_data: Dict, frame_indices: List[int]) -> np.ndarray:
        """Extract 3x3 projection matrices from ground truth."""
        # Objectron format: gt_data['frames'][i]['intrinsics'] = [fx, fy, cx, cy]
        projs = []
        
        for frame_idx in frame_indices:
            frame_data = gt_data['frames'][frame_idx]
            intrinsics = frame_data['intrinsics']
            
            fx, fy, cx, cy = intrinsics
            
            # Build projection matrix
            K = np.array([
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1]
            ], dtype=np.float32)
            
            projs.append(K)
        
        return np.stack(projs)
    
    def _extract_camera_poses(self, gt_data: Dict, frame_indices: List[int]) -> np.ndarray:
        """Extract 4x4 camera-to-world pose matrices from ground truth."""
        poses = []
        
        for frame_idx in frame_indices:
            frame_data = gt_data['frames'][frame_idx]
            c2w = np.array(frame_data['camera_to_world'], dtype=np.float32)
            
            # Ensure it's 4x4
            if c2w.shape != (4, 4):
                # Sometimes stored as flattened 16 elements
                c2w = c2w.reshape(4, 4)
            
            poses.append(c2w)
        
        return np.stack(poses)


# =============================================================================
# ANYCALIB WRAPPER FOR BATCH PROCESSING
# =============================================================================

class AnyCaLibBatchInference:
    """
    Wrapper for AnyCaLib to handle batch inference efficiently.
    """
    
    def __init__(self, device: torch.device, use_multi_frame: bool = False):
        """
        Args:
            device: torch.device for inference
            use_multi_frame: If True, run AnyCaLib on all frames and average.
                            If False, run only on first frame.
        """
        self.device = device
        self.use_multi_frame = use_multi_frame
        
        print(f"[ANYCALIB] Loading pretrained model...")
        self.model = AnyCalib(model_id="anycalib_pinhole").to(device).eval()
        print(f"[ANYCALIB] Model loaded on {device}")
        
        # ===== CONFIGURATION COMMENT =====
        # Change self.use_multi_frame to switch between approaches:
        #   False: Run AnyCalib only on first frame (FASTER, SIMPLER)
        #   True:  Run AnyCalib on all frames and average (SLOWER, MORE ROBUST)
        # ===== END CONFIGURATION =====
        
        if self.use_multi_frame:
            print(f"[ANYCALIB] Mode: Multi-frame averaging")
        else:
            print(f"[ANYCALIB] Mode: Single frame (first frame only)")
    
    def predict_focal_length(self, images: torch.Tensor) -> torch.Tensor:
        """
        Predict focal length for a batch of sequences.
        
        Args:
            images: torch.Tensor [batch, num_frames, 3, H, W] in range [0, 1]
        
        Returns:
            focal_lengths: torch.Tensor [batch] - focal length in pixels (normalized)
        """
        batch_size, num_frames, c, h, w = images.shape
        
        with torch.no_grad(), torch.autocast(device_type='cuda', enabled=False):
            if not self.use_multi_frame:
                # ===== SINGLE FRAME APPROACH =====
                # Run AnyCaLib only on the first frame of each sequence
                # Assumption: Focal length is constant across the sequence
                
                first_frames = images[:, 0].float()  # [batch, 3, H, W] - ensure float32
                pred = self.model.predict(first_frames, cam_id="pinhole")
                
                # pred["intrinsics"] is a list of intrinsic vectors, one per batch item
                # Each K is likely [fx, fy, cx, cy] or similar
                intrinsics_list = pred["intrinsics"]
                focal_px = torch.stack([K[0] if K.dim() > 0 else K for K in intrinsics_list])  # Extract fx
                
                # ===== END SINGLE FRAME APPROACH =====
            
            else:
                # ===== MULTI-FRAME AVERAGING APPROACH =====
                # Run AnyCaLib on all frames and average the predictions
                # More robust but slower
                
                focal_preds_all = []
                for frame_idx in range(num_frames):
                    frames = images[:, frame_idx].float()  # [batch, 3, H, W] - ensure float32
                    pred = self.model.predict(frames, cam_id="pinhole")
                    
                    # pred["intrinsics"] is a list of intrinsic vectors, one per batch item
                    intrinsics_list = pred["intrinsics"]
                    focal_pred = torch.stack([K[0] if K.dim() > 0 else K for K in intrinsics_list])  # Extract fx
                    focal_preds_all.append(focal_pred)
                
                # Average across frames
                focal_preds = torch.stack(focal_preds_all, dim=1).mean(dim=1)  # [batch]
                
                # focal_preds is already in pixels
                focal_px = focal_preds
                
                # ===== END MULTI-FRAME AVERAGING APPROACH =====
        
        return focal_px


# =============================================================================
# MODIFIED TRAINING WRAPPER WITH ANYCALIB INJECTION
# =============================================================================

class AnyCamWrapperWithAnyCaLib(nn.Module):
    """
    Modified AnyCam wrapper that uses AnyCaLib for focal length prediction
    instead of the original candidate system.
    
    KEY MODIFICATION: Instead of predicting 32 focal length candidates and
    selecting the best one via flow reprojection, we directly use AnyCaLib's
    prediction as a single focal length value.
    """
    
    def __init__(
        self,
        pose_predictor_config: Dict,
        depth_predictor_config: Dict,
        anycalib_model: AnyCaLibBatchInference,
        use_provided_depth: bool = False,
        use_provided_flow: bool = False,
    ):
        super().__init__()
        
        self.use_provided_depth = use_provided_depth
        self.use_provided_flow = use_provided_flow
        
        # ===== ANYCALIB INTEGRATION =====
        self.anycalib_model = anycalib_model
        print(f"[WRAPPER] AnyCaLib focal length injection enabled")
        # ===== END ANYCALIB INTEGRATION =====
        
        # Load models
        self.depth_predictor = make_depth_predictor(depth_predictor_config)
        self.pose_predictor = make_pose_predictor(pose_predictor_config)
        
        # Freeze depth predictor (it's just for preprocessing)
        for param in self.depth_predictor.parameters():
            param.requires_grad = False
        
        # Image processor for flow and occlusion
        self.image_processor = make_image_processor(
            {"type": "flow_occlusion"}, 
            flow_model="unimatch",
            use_provided_flow=self.use_provided_flow,
            pair_mode="sequential"
        )
        
        self.z_near = 0.1
        self.z_far = 10.0
        
        print(f"[WRAPPER] Initialized with AnyCaLib focal length prediction")
    
    def freeze_except_pose_head(self):
        """
        Freeze all model parameters except the pose_head.
        
        This is the KEY function for our experiment:
        - Backbone (DINOv2): FROZEN
        - Neck: FROZEN
        - Uncertainty head: FROZEN
        - Sequence info head (focal/scaling): FROZEN (we use AnyCaLib anyway)
        - Pose head: TRAINABLE (this is what we want to learn)
        """
        print(f"\n{'='*70}")
        print(f"[FREEZE] Freezing all layers except pose_head...")
        print(f"{'='*70}")
        
        # First, freeze everything
        for name, param in self.named_parameters():
            param.requires_grad = False
        
        # Then, unfreeze only the pose_head
        for name, param in self.pose_predictor.pose_head.named_parameters():
            param.requires_grad = True
            print(f"[UNFREEZE] {name}: {param.shape}")
        
        # Count parameters
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        
        print(f"\n[PARAMS] Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
        print(f"{'='*70}\n")
    
    def reinitialize_pose_head(self):
        """
        Delete the existing pose_head and create a fresh randomly initialized one.
        
        This ensures we're training from scratch and not relying on pretrained weights.
        """
        print(f"\n{'='*70}")
        print(f"[REINIT] Deleting old pose_head and creating fresh one...")
        print(f"{'='*70}")
        
        # Get the configuration from the existing head
        in_chn = self.pose_predictor.pose_head.proj0.in_features
        out_chn = self.pose_predictor.pose_head.proj1.out_features
        
        print(f"[REINIT] Old pose_head: in={in_chn}, out={out_chn}")
        
        # Import the head class
        from anycam.models.anycam_blocks import AnyCamPoseTokenHead
        
        # Create new head with random initialization
        self.pose_predictor.pose_head = AnyCamPoseTokenHead(in_chn, out_chn)
        
        # Move to same device as model
        device = next(self.pose_predictor.parameters()).device
        self.pose_predictor.pose_head = self.pose_predictor.pose_head.to(device)
        
        print(f"[REINIT] New pose_head created with random weights")
        print(f"{'='*70}\n")
    
    def forward(self, data: Dict) -> Dict:
        """
        Forward pass with AnyCaLib focal length injection.
        
        KEY MODIFICATION: At the point where focal length candidates would normally
        be generated, we instead use AnyCaLib's prediction.
        """
        images = data["imgs"]  # [B, num_frames, 3, H, W]
        gt_projs = data["projs"]  # [B, num_frames, 3, 3]
        
        n, f, c, h, w = images.shape
        device = images.device
        
        # Normalize projection matrices
        gt_projs = normalize_proj(gt_projs[:, 0], h, w)
        
        # ===== ANYCALIB FOCAL LENGTH PREDICTION =====
        # This is where we inject AnyCaLib instead of using the candidate system
        
        print(f"[FORWARD] Running AnyCaLib on batch...")
        focal_length_anycalib = self.anycalib_model.predict_focal_length(images)  # [B]
        print(f"[FORWARD] AnyCaLib focal lengths: {focal_length_anycalib}")
        
        # Normalize focal length (AnyCam expects focal / width)
        focal_length_normalized = focal_length_anycalib / w
        
        # Create projection matrix from focal length
        proj_candidates = make_proj_from_focal_length(
            focal_length_normalized.unsqueeze(1),  # [B, 1]
            aspect_ratio=h/w
        )
        
        # ===== END ANYCALIB INJECTION =====
        
        # Get depth predictions (frozen)
        if not self.use_provided_depth:
            depth_in = images.view(n * f, c, h, w)
            
            with torch.no_grad():
                depths, depth_features = self.depth_predictor(depth_in, return_features=True)
            
            depths = depths[0]
            depths = 1 / depths.clamp_min(1e-3).view(n, -1, 1, *depths.shape[-2:])
        else:
            depths = data["depths"]
        
        data["pred_depths"] = depths * 0.1
        data["pred_depths_list"] = [depths]
        
        # Get flow and occlusion
        images_ip_fwd, images_ip_bwd = self.image_processor(images * 2 - 1, data=data)
        flow_occ_fwd = images_ip_fwd[:, :, 3:6]
        
        # Run pose predictor
        pose_result = self.pose_predictor(
            images,
            flow_occs=flow_occ_fwd,
            depths=depths,
        )
        
        uncert = pose_result["uncert"]
        poses = pose_result["poses"]
        
        # Handle pose candidates: model may output 32 candidates but we only use 1 focal
        # Take only the first candidate if multiple exist
        if poses.dim() == 5 and poses.shape[2] > 1:
            poses = poses[:, :, 0:1]  # Keep only first candidate [B, F, 1, 4, 4]
            pose_result["poses"] = poses  # Update in pose_result
        if uncert.dim() == 6 and uncert.shape[2] > 1:
            uncert = uncert[:, :, 0:1]  # Keep only first candidate
            pose_result["uncert"] = uncert  # Update in pose_result
        
        # Compute induced flow for loss
        # aligned_depths needs shape [B, F, num_candidates, 1, H, W] to match poses
        num_candidates = 1
        aligned_depths = depths.view(n, f, 1, 1, h, w)
        induced_flow, dist = induce_flow_dist(
            aligned_depths, 
            proj_candidates, 
            poses, 
            flow_occ_fwd[..., :2, :, :]
        )
        
        # Select results (single focal candidate - already filtered above)
        selected_induced_flow = induced_flow[:, :, 0, :, :, :]
        selected_proj = proj_candidates[:, 0:1]
        selected_poses = poses[:, :, 0]  # [B, F, 4, 4]
        selected_aligned_depths = aligned_depths[:, :, 0]  # [B, F, 1, H, W]
        selected_uncert = uncert[:, :, 0]  # [B, F, ?, H, W]
        
        # Add flow_occs_in to pose_result for loss computation
        pose_result["flow_occs_in"] = flow_occ_fwd
        pose_result["aligned_depths"] = aligned_depths
        pose_result["induced_flow"] = induced_flow
        pose_result["dist"] = dist
        pose_result["proj_candidates"] = proj_candidates
        
        # Package results
        data["images_ip"] = images_ip_fwd
        data["induced_flow"] = selected_induced_flow
        data["induced_flow_list"] = [selected_induced_flow]
        data["valid"] = images_ip_fwd[:, :, 5:6] > 0.5
        data["proc_poses"] = selected_poses
        data["proc_projs"] = selected_proj
        data["uncertainties"] = selected_uncert
        data["weights_proc"] = selected_uncert
        data["scaled_depths"] = [selected_aligned_depths]
        data["z_near"] = torch.tensor(self.z_near, device=device)
        data["z_far"] = torch.tensor(self.z_far, device=device)
        data["pose_result"] = pose_result
        
        return data


# =============================================================================
# VISUALIZATION UTILITIES
# =============================================================================

def plot_loss_curve(loss_history: List[Dict], save_dir: Path):
    """
    Plot and save loss curve.
    """
    if not loss_history:
        print("[VIZ] No loss history to plot")
        return
    
    epochs = [item['epoch'] for item in loss_history]
    losses = [item['loss'] for item in loss_history]
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, losses, 'b-', linewidth=2, label='Training Loss')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training Loss Over Time', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    
    # Add annotations for first and last loss
    if len(losses) > 0:
        plt.annotate(f'Start: {losses[0]:.4f}', 
                    xy=(epochs[0], losses[0]), 
                    xytext=(10, 10), 
                    textcoords='offset points',
                    fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.5))
        plt.annotate(f'Final: {losses[-1]:.4f}', 
                    xy=(epochs[-1], losses[-1]), 
                    xytext=(-50, -20), 
                    textcoords='offset points',
                    fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.5))
    
    # Save plot
    plot_path = save_dir / "loss_curve.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"[VIZ] Loss curve saved: {plot_path}")


def save_training_summary(loss_history: List[Dict], batch_losses: List[float], save_dir: Path):
    """
    Save training summary statistics.
    """
    if not loss_history:
        return
    
    losses = [item['loss'] for item in loss_history]
    summary_path = save_dir / "training_summary.txt"
    
    with open(summary_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("TRAINING SUMMARY - Pose Head Experiment with AnyCaLib\n")
        f.write("=" * 70 + "\n\n")
        
        f.write(f"Total Epochs: {len(loss_history)}\n")
        f.write(f"Total Batches Processed: {len(batch_losses)}\n\n")
        
        f.write(f"Initial Loss (Epoch 1): {losses[0]:.6f}\n")
        f.write(f"Final Loss (Epoch {len(losses)}): {losses[-1]:.6f}\n")
        f.write(f"Best Loss: {min(losses):.6f} (Epoch {losses.index(min(losses)) + 1})\n")
        f.write(f"Worst Loss: {max(losses):.6f} (Epoch {losses.index(max(losses)) + 1})\n\n")
        
        improvement = losses[0] - losses[-1]
        improvement_pct = (improvement / abs(losses[0])) * 100 if losses[0] != 0 else 0
        f.write(f"Total Improvement: {improvement:.6f} ({improvement_pct:.2f}%)\n\n")
        
        f.write("Epoch-by-Epoch Progress:\n")
        f.write("-" * 70 + "\n")
        for item in loss_history:
            f.write(f"Epoch {item['epoch']:3d}: Loss = {item['loss']:.6f}\n")
    
    print(f"[VIZ] Training summary saved: {summary_path}")


# =============================================================================
# TRAINING LOOP
# =============================================================================

def train_pose_head(
    model: AnyCamWrapperWithAnyCaLib,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    num_epochs: int,
    device: torch.device,
    save_dir: Path,
    log_interval: int = 10,
):
    """
    Main training loop for pose head experiment.
    """
    model.train()
    scaler = GradScaler('cuda')
    
    # Initialize tracking
    loss_history = []
    batch_losses = []
    
    # Create log file
    log_file = save_dir / "training_log.txt"
    with open(log_file, 'w') as f:
        f.write(f"Training Log - Pose Head Experiment\n")
        f.write(f"{'='*70}\n")
        f.write(f"Start time: {pd.Timestamp.now()}\n" if 'pd' in dir() else f"")
        f.write(f"Num epochs: {num_epochs}\n")
        f.write(f"Batch size: {dataloader.batch_size}\n")
        f.write(f"Device: {device}\n")
        f.write(f"{'='*70}\n\n")
    
    print(f"\n{'='*70}")
    print(f"[TRAIN] Starting training for {num_epochs} epochs")
    print(f"[TRAIN] Batch size: {dataloader.batch_size}")
    print(f"[TRAIN] Total batches: {len(dataloader)}")
    print(f"[TRAIN] Logs will be saved to: {log_file}")
    print(f"{'='*70}\n")
    
    global_step = 0
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        
        for batch_idx, batch_data in enumerate(dataloader):
            # Move data to device
            batch_data = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                         for k, v in batch_data.items()}
            
            # Forward pass
            optimizer.zero_grad()
            
            with autocast(device_type='cuda', dtype=torch.float16):
                output_data = model(batch_data)
                
                # Compute loss
                loss_dict = criterion(output_data)
                loss = loss_dict.get('loss', loss_dict.get('total_loss', sum(loss_dict.values())))
            
            # Backward pass
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
            batch_losses.append(loss.item())
            global_step += 1
            
            # Logging
            if batch_idx % log_interval == 0:
                log_msg = (f"[TRAIN] Epoch {epoch+1}/{num_epochs} | "
                          f"Batch {batch_idx}/{len(dataloader)} | "
                          f"Loss: {loss.item():.6f}")
                print(log_msg)
                with open(log_file, 'a') as f:
                    f.write(f"{log_msg}\n")
        
        # Epoch summary
        avg_loss = epoch_loss / len(dataloader)
        loss_history.append({'epoch': epoch + 1, 'loss': avg_loss, 'step': global_step})
        
        log_msg = f"\n[EPOCH {epoch+1}] Average Loss: {avg_loss:.6f}\n"
        print(log_msg)
        with open(log_file, 'a') as f:
            f.write(f"{log_msg}\n")
        
        # Save checkpoint
        if (epoch + 1) % 5 == 0 or epoch == num_epochs - 1:
            checkpoint_path = save_dir / f"checkpoint_epoch_{epoch+1}.pt"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'loss_history': loss_history,
            }, checkpoint_path)
            print(f"[SAVE] Checkpoint saved to {checkpoint_path}")
    
    # Save loss history to JSON
    loss_json_path = save_dir / "loss_history.json"
    with open(loss_json_path, 'w') as f:
        json.dump({
            'epoch_losses': loss_history,
            'batch_losses': batch_losses,
        }, f, indent=2)
    print(f"[SAVE] Loss history saved: {loss_json_path}")
    
    # Generate visualizations and summary
    plot_loss_curve(loss_history, save_dir)
    save_training_summary(loss_history, batch_losses, save_dir)
    
    print(f"\n{'='*70}")
    print(f"[TRAIN] Training complete!")
    print(f"[TRAIN] All results saved to: {save_dir}")
    print(f"{'='*70}\n")


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Pose Head Retraining with AnyCaLib")
    parser.add_argument("--videos_dir", type=str, 
                       default="/home/kalman/TUM/thesis/Objectron/videos/",
                       help="Directory with Objectron video files")
    parser.add_argument("--gt_dir", type=str,
                       default="/home/kalman/TUM/thesis/Objectron/annotations/",
                       help="Directory with Objectron ground truth JSON files")
    parser.add_argument("--max_sequences", type=int, default=None,
                       help="Limit number of sequences (for debugging)")
    parser.add_argument("--num_frames", type=int, default=2,
                       help="Number of frames per sequence")
    parser.add_argument("--batch_size", type=int, default=1,
                       help="Batch size for training")
    parser.add_argument("--num_epochs", type=int, default=50,
                       help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4,
                       help="Learning rate")
    parser.add_argument("--save_dir", type=str, default="./experiments/pose_head_experiment_results",
                       help="Directory to save checkpoints and logs")
    parser.add_argument("--model_path", type=str, default="pretrained_models/anycam_seq8",
                       help="Path to pretrained AnyCam model")
    parser.add_argument("--anycalib_multi_frame", action="store_true",
                       help="Use multi-frame averaging for AnyCaLib (slower but more robust)")
    parser.add_argument("--split_file", type=str, default="experiments/objectron_split.json",
                       help="Path to dataset split file (will be created if doesn't exist)")
    parser.add_argument("--run_evaluation", action="store_true",
                       help="Run evaluation after training (requires GT)")
    parser.add_argument("--extract_all_pairs", action="store_true",
                       help="Extract all consecutive frame pairs from each video (more data)")
    parser.add_argument("--eval_only", action="store_true",
                       help="Skip training and only run evaluation")
    parser.add_argument("--eval_dataset", type=str, default="lightspeed", choices=["objectron", "lightspeed"],
                       help="Dataset to use for evaluation (default: lightspeed)")
    parser.add_argument("--lightspeed_dir", type=str, 
                       default="/home/kalman/TUM/thesis/dynpose-100k/lightspeed/",
                       help="Directory of LightSpeed validation dataset")
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"POSE HEAD RETRAINING EXPERIMENT WITH ANYCALIB")
    print(f"{'='*70}")
    print(f"Device: {device}")
    print(f"Videos: {args.videos_dir}")
    print(f"GT: {args.gt_dir}")
    print(f"Max sequences: {args.max_sequences or 'All'}")
    print(f"Frames per sequence: {args.num_frames}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.num_epochs}")
    print(f"Learning rate: {args.lr}")
    print(f"AnyCaLib mode: {'Multi-frame' if args.anycalib_multi_frame else 'Single frame'}")
    print(f"Multi-pair extraction: {args.extract_all_pairs}")
    print(f"{'='*70}\n")
    
    # 1. Load dataset and create train/val/test split
    print(f"[STEP 1] Loading Objectron dataset...")
    print(f"[INFO] This is UNSUPERVISED training - GT is NOT required for training!")
    print(f"[INFO] Training uses flow reprojection loss only")
    print(f"[INFO] GT is only needed for evaluation")
    
    # Create or load dataset split
    split_file = Path(args.split_file)
    if split_file.exists():
        print(f"[SPLIT] Loading existing split from {split_file}")
        split_dict = load_dataset_split(str(split_file))
        train_indices = split_dict['train']
        val_indices = split_dict['val']
        test_indices = split_dict['test']
    else:
        print(f"[SPLIT] Creating new dataset split...")
        # First load to count total videos
        temp_dataset = ObjectronVideoDataset(
            videos_dir=args.videos_dir,
            gt_dir=args.gt_dir,
            num_frames=args.num_frames,
            max_sequences=args.max_sequences,
            require_gt=False,
            extract_all_pairs=False,  # Just count videos
        )
        num_videos = len(temp_dataset.video_files)
        
        train_indices, val_indices, test_indices = create_train_val_test_split(
            num_videos=num_videos,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            seed=42,
        )
        
        # Save split for reproducibility
        split_dict = {
            'train': train_indices,
            'val': val_indices,
            'test': test_indices,
            'num_videos': num_videos,
        }
        save_dataset_split(split_dict, str(split_file))
    
    # Create training dataset
    print(f"\n[DATASET] Creating training dataset...")
    train_dataset = ObjectronVideoDataset(
        videos_dir=args.videos_dir,
        gt_dir=None,  # GT not needed for unsupervised training!
        num_frames=args.num_frames,
        video_indices=train_indices,
        require_gt=False,
        extract_all_pairs=args.extract_all_pairs,
    )
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    print(f"[STEP 1] Training dataset loaded: {len(train_dataset)} pairs\n")
    
    # Create test dataset (only if evaluation is requested)
    test_dataloader = None
    if args.run_evaluation or args.eval_only:
        if args.eval_dataset == "lightspeed":
            print(f"[DATASET] Creating LightSpeed evaluation dataset...")
            from experiments.lightspeed_dataset import LightSpeedDataset
            
            test_dataset = LightSpeedDataset(
                lightspeed_dir=args.lightspeed_dir,
                num_frames=args.num_frames,
                image_size=(480, 640),
                extract_all_pairs=args.extract_all_pairs,
            )
            print(f"[DATASET] LightSpeed dataset loaded: {len(test_dataset)} pairs")
            print(f"[INFO] LightSpeed contains {len(test_dataset.sequence_names)} sequences")
        else:
            print(f"[DATASET] Creating Objectron test dataset (with GT for evaluation)...")
            test_dataset = ObjectronVideoDataset(
                videos_dir=args.videos_dir,
                gt_dir=args.gt_dir,  # GT required for evaluation!
                num_frames=args.num_frames,
                video_indices=test_indices,
                require_gt=True,  # Skip videos without GT
                extract_all_pairs=args.extract_all_pairs,
            )
            print(f"[DATASET] Objectron test dataset loaded: {len(test_dataset)} pairs")
        
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )
        print(f"[EVAL] Evaluation will use: {args.eval_dataset.upper()} dataset\n")
    
    # 2. Initialize AnyCaLib
    print(f"[STEP 2] Initializing AnyCaLib...")
    anycalib_inference = AnyCaLibBatchInference(
        device=device,
        use_multi_frame=args.anycalib_multi_frame
    )
    print(f"[STEP 2] AnyCaLib ready\n")
    
    # 3. Load pretrained AnyCam
    print(f"[STEP 3] Loading pretrained AnyCam model...")
    
    # Load model checkpoint
    model_path = Path(args.model_path)
    checkpoint_file = model_path / "training_checkpoint_247500.pt"
    config_file = model_path / "training_config.yaml"
    
    if not checkpoint_file.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_file}")
    
    # Load config
    import yaml
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create wrapper with AnyCaLib
    model = AnyCamWrapperWithAnyCaLib(
        pose_predictor_config=config['model']['pose_predictor'],
        depth_predictor_config=config['model']['depth_predictor'],
        anycalib_model=anycalib_inference,
    )
    
    # Load pretrained weights
    checkpoint = torch.load(checkpoint_file, map_location=device)
    model.load_state_dict(checkpoint['model'], strict=False)
    model = model.to(device)
    
    print(f"[STEP 3] Pretrained model loaded\n")
    
    # 4. Reinitialize pose head
    print(f"[STEP 4] Reinitializing pose head...")
    model.reinitialize_pose_head()
    print(f"[STEP 4] Pose head reinitialized\n")
    
    # 5. Freeze all except pose head
    print(f"[STEP 5] Freezing layers...")
    model.freeze_except_pose_head()
    print(f"[STEP 5] Freezing complete\n")
    
    # 6. Setup optimizer and loss
    print(f"[STEP 6] Setting up optimizer and loss...")
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr
    )
    
    # Use flow reprojection loss
    loss_config = config['loss'][0].copy()
    # CRITICAL FIX: Disable fwd_bwd_consistency loss (causes NaN with forward-only training)
    loss_config['lambda_fwd_bwd_consistency'] = 0
    print(f"[FIX] Disabled fwd_bwd_consistency loss (was causing NaN)")
    criterion = make_loss(loss_config)
    
    print(f"[STEP 6] Optimizer and loss ready\n")
    
    # 7. Train (unless eval_only mode)
    if not args.eval_only:
        print(f"[STEP 7] Starting training...")
        train_pose_head(
            model=model,
            dataloader=train_dataloader,
            criterion=criterion,
            optimizer=optimizer,
            num_epochs=args.num_epochs,
            device=device,
            save_dir=save_dir,
        )
        print(f"[STEP 7] Training complete\n")
        
        # 8. Save final model
        final_model_path = save_dir / "final_model.pt"
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': config,
        }, final_model_path)
        print(f"[FINAL] Model saved to {final_model_path}")
    else:
        print(f"[SKIP] Training skipped (eval_only mode)")
        # Load trained model for evaluation
        final_model_path = save_dir / "final_model.pt"
        if final_model_path.exists():
            checkpoint = torch.load(final_model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"[LOAD] Loaded trained model from {final_model_path}")
        else:
            raise FileNotFoundError(f"No trained model found at {final_model_path}")
    
    # 9. Run evaluation (if requested and test data available)
    if args.run_evaluation and test_dataloader is not None:
        print(f"\n{'='*70}")
        print(f"[STEP 8] Running evaluation on test set...")
        print(f"{'='*70}\n")
        
        from experiments.pose_metrics import (
            rotation_error_degrees,
            translation_direction_error_degrees,
            pose_error,
            compute_error_statistics,
        )
        
        # Simple evaluation (full evaluation script available in evaluate_pose_model.py)
        eval_results_dir = save_dir / "evaluation"
        eval_results_dir.mkdir(exist_ok=True)
        
        print(f"[EVAL] Running simplified evaluation...")
        print(f"[EVAL] For full evaluation with baseline comparison, use evaluate_pose_model.py")
        
        model.eval()
        all_rot_errors = []
        all_trans_errors = []
        
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(test_dataloader):
                if batch_idx % 10 == 0:
                    print(f"[EVAL] Processing batch {batch_idx}/{len(test_dataloader)}")
                
                batch_data = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                             for k, v in batch_data.items()}
                
                # Get predictions
                output = model(batch_data)
                pred_poses = output['pose_result']['poses']  # [batch, pairs, candidates, 4, 4]
                gt_poses = batch_data['poses']  # [batch, frames, 4, 4]
                
                # Take first candidate
                if pred_poses.dim() == 5:
                    pred_poses = pred_poses[:, :, 0]  # [batch, pairs, 4, 4]
                
                # Compute relative GT poses (frame i to frame i+1)
                batch_size, num_frames = gt_poses.shape[:2]
                num_pairs = num_frames - 1
                
                for b in range(batch_size):
                    for p in range(num_pairs):
                        pred_pose = pred_poses[b, p].cpu().numpy()
                        # Compute GT relative pose
                        gt_pose1 = gt_poses[b, p].cpu().numpy()
                        gt_pose2 = gt_poses[b, p+1].cpu().numpy()
                        gt_rel_pose = np.linalg.inv(gt_pose2) @ gt_pose1
                        
                        rot_err, trans_err = pose_error(pred_pose, gt_rel_pose)
                        all_rot_errors.append(rot_err)
                        all_trans_errors.append(trans_err)
        
        # Compute statistics
        rot_errors = np.array(all_rot_errors)
        trans_errors = np.array(all_trans_errors)
        
        rot_stats = compute_error_statistics(rot_errors)
        trans_stats = compute_error_statistics(trans_errors)
        
        # Save results
        eval_results = {
            'rotation': rot_stats,
            'translation': trans_stats,
            'num_samples': len(rot_errors),
        }
        
        with open(eval_results_dir / 'evaluation_results.json', 'w') as f:
            json.dump(eval_results, f, indent=2)
        
        # Print summary
        print(f"\n{'='*70}")
        print(f"EVALUATION RESULTS")
        print(f"{'='*70}")
        print(f"Test Set Size: {len(rot_errors)} frame pairs")
        print(f"\nRotation Error (degrees):")
        print(f"  Mean:   {rot_stats['mean']:.4f}")
        print(f"  Median: {rot_stats['median']:.4f}")
        print(f"  Std:    {rot_stats['std']:.4f}")
        print(f"  P90:    {rot_stats['p90']:.4f}")
        print(f"\nTranslation Direction Error (degrees):")
        print(f"  Mean:   {trans_stats['mean']:.4f}")
        print(f"  Median: {trans_stats['median']:.4f}")
        print(f"  Std:    {trans_stats['std']:.4f}")
        print(f"  P90:    {trans_stats['p90']:.4f}")
        print(f"{'='*70}\n")
        
        print(f"[EVAL] Results saved to {eval_results_dir}")
    
    print(f"\n{'='*70}")
    print(f"EXPERIMENT COMPLETE!")
    print(f"Results saved to: {save_dir}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()

