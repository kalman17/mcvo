"""
Helper utilities for benchmark dataset loading and smart sampling.

Provides automatic dataset path detection and intelligent frame pair sampling.
"""

from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
import cv2
import json
from torch.utils.data import Dataset, Subset

from experiments.dataset_paths import (
    OBJECTRON_VIDEOS, OBJECTRON_GT, LIGHTSPEED_ROOT,
    get_objectron_videos, get_objectron_gt, get_lightspeed_root
)
from experiments.train_pose_head_anycalib import load_dataset_split
from experiments.lightspeed_dataset import LightSpeedDataset


def get_dataset_paths(dataset_name: str) -> Dict[str, Path]:
    """
    Automatically get dataset paths based on dataset name.
    
    Args:
        dataset_name: 'objectron' or 'lightspeed'
    
    Returns:
        Dictionary with paths:
        - For objectron: {'videos': Path, 'gt': Path}
        - For lightspeed: {'root': Path}
    """
    if dataset_name == 'objectron':
        return {
            'videos': Path(get_objectron_videos()),
            'gt': Path(get_objectron_gt()),
        }
    elif dataset_name == 'lightspeed':
        return {
            'root': Path(get_lightspeed_root()),
        }
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def count_available_pairs_objectron(
    videos_dir: Path,
    gt_dir: Path,
    video_indices: Optional[List[int]] = None,
) -> int:
    """
    Count total available frame pairs in Objectron dataset.
    
    Returns:
        Total number of frame pairs available (assuming at least 2 frames per video)
    """
    video_files = sorted(list(videos_dir.glob("*.MOV")) + list(videos_dir.glob("*.mov")))
    
    if video_indices is not None:
        video_files = [video_files[i] for i in video_indices if i < len(video_files)]
    
    total_pairs = 0
    
    for video_path in video_files:
        # Count frames in video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            continue
        
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        # Number of pairs = num_frames - 1 (consecutive pairs)
        if num_frames >= 2:
            total_pairs += num_frames - 1
    
    return total_pairs


def count_available_pairs_lightspeed(lightspeed_dir: Path) -> int:
    """
    Count total available frame pairs in LightSpeed dataset.
    
    Returns:
        Total number of frame pairs available
    """
    dataset = LightSpeedDataset(
        lightspeed_dir=str(lightspeed_dir),
        num_frames=2,
        extract_all_pairs=True,
    )
    return len(dataset)


def create_smart_sampled_dataset_objectron(
    videos_dir: Path,
    gt_dir: Path,
    num_samples: Union[int, str],
    video_indices: Optional[List[int]] = None,
    image_size: Tuple[int, int] = (480, 640),
) -> List[Dict]:
    """
    Create smart sampled frame pairs from Objectron dataset.
    
    Logic:
    - If num_samples is "all", use all available pairs
    - If num_samples > available, use all available
    - Otherwise, cycle through videos taking different frame pairs
    
    Returns:
        List of dictionaries with:
        - 'video_path': Path to video file
        - 'frame_indices': [frame_idx1, frame_idx2] for the pair
        - 'gt_calibration': Optional GT calibration for frames
    """
    video_files = sorted(list(videos_dir.glob("*.MOV")) + list(videos_dir.glob("*.mov")))
    
    if video_indices is not None:
        video_files = [video_files[i] for i in video_indices if i < len(video_files)]
    
    # First pass: collect all available pairs
    all_pairs = []
    
    for video_path in video_files:
        # Count frames
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            continue
        
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        if num_frames < 2:
            continue
        
        # Load GT calibrations if available
        gt_calibrations = _load_gt_calibrations_objectron(gt_dir, video_path, num_frames)
        
        # Add all consecutive pairs from this video
        for i in range(num_frames - 1):
            all_pairs.append({
                'video_path': video_path,
                'frame_indices': [i, i + 1],
                'gt_calibration': gt_calibrations[i:i+2] if gt_calibrations is not None else None,
            })
    
    total_available = len(all_pairs)
    
    # Determine how many to use
    if num_samples == "all":
        num_to_use = total_available
    elif isinstance(num_samples, int):
        num_to_use = min(num_samples, total_available)
    else:
        raise ValueError(f"Invalid num_samples: {num_samples}")
    
    # Sample without replacement
    if num_to_use >= total_available:
        # Use all
        return all_pairs
    else:
        # Random sample without replacement
        indices = np.random.choice(total_available, size=num_to_use, replace=False)
        return [all_pairs[i] for i in indices]


def create_smart_sampled_dataset_lightspeed(
    lightspeed_dir: Path,
    num_samples: Union[int, str],
) -> LightSpeedDataset:
    """
    Create smart sampled dataset from LightSpeed.
    
    Returns:
        LightSpeedDataset with appropriate sampling
    """
    # LightSpeed dataset already handles pair extraction
    dataset = LightSpeedDataset(
        lightspeed_dir=str(lightspeed_dir),
        num_frames=2,
        extract_all_pairs=True,
    )
    
    total_available = len(dataset)
    
    # Determine how many to use
    if num_samples == "all":
        num_to_use = total_available
    elif isinstance(num_samples, int):
        num_to_use = min(num_samples, total_available)
    else:
        raise ValueError(f"Invalid num_samples: {num_samples}")
    
    # If we need fewer, create a subset
    if num_to_use < total_available:
        from torch.utils.data import Subset
        indices = np.random.choice(total_available, size=num_to_use, replace=False)
        return Subset(dataset, indices.tolist())
    else:
        return dataset


def _load_gt_calibrations_objectron(
    gt_dir: Path,
    video_path: Path,
    num_frames: int,
) -> Optional[np.ndarray]:
    """Load GT calibrations for a video."""
    gt_path1 = gt_dir / f"{video_path.stem}.json"
    stem_without_video = video_path.stem.replace("_video", "")
    gt_path2 = gt_dir / f"{stem_without_video}.json"
    
    gt_path = gt_path1 if gt_path1.exists() else (gt_path2 if gt_path2.exists() else None)
    
    if gt_path is None:
        return None
    
    try:
        with open(gt_path, 'r') as f:
            gt_data = json.load(f)
        
        calibrations = []
        
        if 'intrinsics_per_frame' in gt_data:
            intr_list = gt_data['intrinsics_per_frame']
            for frame_idx in range(min(num_frames, len(intr_list))):
                K_flat = intr_list[frame_idx]
                fx, fy, cx, cy = K_flat[0], K_flat[4], K_flat[2], K_flat[5]
                calibrations.append([fx, fy, cx, cy])
        elif 'frames' in gt_data:
            for frame_idx in range(min(num_frames, len(gt_data['frames']))):
                frame_data = gt_data['frames'][frame_idx]
                intrinsics = frame_data.get('intrinsics', None)
                if intrinsics is None:
                    continue
                fx, fy, cx, cy = intrinsics[:4]
                calibrations.append([fx, fy, cx, cy])
        elif 'intrinsics' in gt_data:
            intr_list = gt_data['intrinsics']
            for frame_idx in range(min(num_frames, len(intr_list))):
                K = np.array(intr_list[frame_idx], dtype=np.float32).reshape(3, 3)
                fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
                calibrations.append([fx, fy, cx, cy])
        
        if len(calibrations) == 0:
            return None
        
        return np.array(calibrations, dtype=np.float32)
        
    except Exception:
        return None

