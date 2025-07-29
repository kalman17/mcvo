#!/usr/bin/env python3
"""
Data Loading Module for AnyCam Experiments

This module provides common data loading functionality for various AnyCam experiments,
including frame extraction from videos/images and ground truth pose loading.
"""

import numpy as np
import cv2
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
from glob import glob
import json
import pickle


class FrameLoader:
    """
    Handles loading and extraction of frames from videos or image sequences.
    """
    
    def __init__(self):
        self.supported_video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
        self.supported_image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    
    def load_frames(self, input_path: str, 
                   num_frames: int = 3,
                   start_frame: int = 0,
                   skip_frames: int = 1) -> List[np.ndarray]:
        """
        Load frames from video file or image folder with flexible extraction parameters.
        
        Args:
            input_path: Path to MP4 file or folder of images
            num_frames: Number of frames to extract
            start_frame: Starting frame index (0-based)
            skip_frames: Frame skip interval (1 = consecutive, 2 = every other frame, etc.)
        
        Returns:
            List of frames as numpy arrays (H, W, 3) in uint8 [0,255] range
        """
        input_path = Path(input_path)
        
        if self._is_video_file(input_path):
            return self._load_from_video(input_path, num_frames, start_frame, skip_frames)
        elif input_path.is_dir():
            return self._load_from_image_folder(input_path, num_frames, start_frame, skip_frames)
        else:
            raise ValueError(f"Unsupported input: {input_path} (must be video file or image folder)")
    
    def _is_video_file(self, path: Path) -> bool:
        """Check if path is a supported video file."""
        return path.is_file() and path.suffix.lower() in self.supported_video_extensions
    
    def _load_from_video(self, video_path: Path, num_frames: int, start_frame: int, skip_frames: int) -> List[np.ndarray]:
        """Load frames from video file."""
        print(f"Loading from video: {video_path}")
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        # Get total frame count
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"   Video has {total_frames} total frames")
        
        # Calculate required frames with skipping
        end_frame = start_frame + (num_frames - 1) * skip_frames
        if end_frame >= total_frames:
            raise ValueError(f"Not enough frames: need frame {end_frame}, but video has {total_frames} frames")
        
        # Extract specific frames
        frames = []
        for i in range(num_frames):
            frame_idx = start_frame + i * skip_frames
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret:
                cap.release()
                raise ValueError(f"Could not read frame {frame_idx}")
            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB
            frames.append(frame)
            print(f"   Extracted frame {frame_idx}: {frame.shape}")
        
        cap.release()
        print(f"[OK] Successfully loaded {len(frames)} frames from video")
        return frames
    
    def _load_from_image_folder(self, folder_path: Path, num_frames: int, start_frame: int, skip_frames: int) -> List[np.ndarray]:
        """Load frames from folder of images."""
        print(f"Loading from image folder: {folder_path}")
        
        # Find image files and sort them
        image_files = []
        for ext in self.supported_image_extensions:
            image_files.extend(glob(str(folder_path / ext)))
            image_files.extend(glob(str(folder_path / ext.upper())))
        
        if not image_files:
            raise ValueError(f"No image files found in folder: {folder_path}")
        
        # Sort by filename to maintain sequence order
        image_files = sorted(image_files)
        total_images = len(image_files)
        print(f"   Found {total_images} image files")
        
        # Calculate required frames with skipping
        end_frame = start_frame + (num_frames - 1) * skip_frames
        if end_frame >= total_images:
            raise ValueError(f"Not enough images: need image {end_frame}, but folder has {total_images} images")
        
        # Extract specific frames
        frames = []
        for i in range(num_frames):
            image_idx = start_frame + i * skip_frames
            image_path = image_files[image_idx]
            
            frame = cv2.imread(image_path)
            if frame is None:
                raise ValueError(f"Could not read image: {image_path}")
            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB
            frames.append(frame)
            print(f"   Loaded image {image_idx} ({Path(image_path).name}): {frame.shape}")
        
        print(f"[OK] Successfully loaded {len(frames)} frames from folder")
        return frames
    
    def get_video_info(self, video_path: str) -> Dict[str, Any]:
        """Get basic information about a video file."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {}
        
        info = {
            'total_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'duration_seconds': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / cap.get(cv2.CAP_PROP_FPS)
        }
        cap.release()
        return info
    
    def find_video_files(self, directory: str, max_files: Optional[int] = None) -> List[Path]:
        """Find all video files in a directory."""
        directory_path = Path(directory)
        video_files = []
        
        for ext in self.supported_video_extensions:
            video_files.extend(list(directory_path.glob(f'*{ext}')))
            video_files.extend(list(directory_path.glob(f'*{ext.upper()}')))
        
        video_files = sorted(video_files)
        
        if max_files:
            video_files = video_files[:max_files]
            
        return video_files


class GroundTruthLoader:
    """
    Handles loading of ground truth camera poses from various dataset formats.
    """
    
    def __init__(self):
        self.supported_formats = ['.pkl', '.npy', '.npz', '.txt', '.json']
    
    def load_poses(self, gt_file_path: str) -> Optional[List[np.ndarray]]:
        """
        Load ground truth camera poses from various dataset formats.
        
        Args:
            gt_file_path: Path to ground truth pose file
            
        Returns:
            List of 4x4 pose matrices, or None if loading fails
        """
        try:
            if not Path(gt_file_path).exists():
                print(f"[WARN] Ground truth file not found: {gt_file_path}")
                return None
            
            file_path = Path(gt_file_path)
            
            if file_path.suffix == '.pkl':
                return self._load_pickle_poses(gt_file_path)
            elif file_path.suffix in ['.npy', '.npz']:
                return self._load_numpy_poses(gt_file_path)
            elif file_path.suffix == '.json':
                return self._load_json_poses(gt_file_path)
            elif file_path.suffix == '.txt':
                return self._load_text_poses(gt_file_path)
            else:
                print(f"[WARN] Unsupported GT file format: {gt_file_path}")
                return None
                
        except Exception as e:
            print(f"[WARN] Failed to load ground truth poses: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def load_raw_gt_data(self, gt_file_path: str) -> Optional[Dict[str, Any]]:
        """
        Load raw ground truth data from various dataset formats.
        
        Args:
            gt_file_path: Path to ground truth file
            
        Returns:
            Raw ground truth data dictionary, or None if loading fails
        """
        try:
            if not Path(gt_file_path).exists():
                print(f"[WARN] Ground truth file not found: {gt_file_path}")
                return None
            
            file_path = Path(gt_file_path)
            
            if file_path.suffix == '.pkl':
                return self._load_raw_pickle_data(gt_file_path)
            elif file_path.suffix in ['.npy', '.npz']:
                return self._load_raw_numpy_data(gt_file_path)
            elif file_path.suffix == '.json':
                return self._load_raw_json_data(gt_file_path)
            else:
                print(f"[WARN] Unsupported GT file format for raw data: {gt_file_path}")
                return None
                
        except Exception as e:
            print(f"[WARN] Failed to load raw ground truth data: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _load_pickle_poses(self, file_path: str) -> Optional[List[np.ndarray]]:
        """Load poses from pickle file (DynPose-100K format)."""
        with open(file_path, 'rb') as f:
            gt_data = pickle.load(f)
        
        # Handle different pickle formats
        if isinstance(gt_data, dict):
            # Try common keys for DynPose format
            for key in ['c2w', 'w2c', 'poses', 'camera_poses', 'extrinsics', 'cam_poses']:
                if key in gt_data:
                    poses = gt_data[key]
                    break
            else:
                print(f"[WARN] No recognized pose key found in GT file: {file_path}")
                print(f"[DEBUG] Available keys: {list(gt_data.keys())}")
                return None
        elif isinstance(gt_data, (list, np.ndarray)):
            poses = gt_data
        else:
            print(f"[WARN] Unexpected GT data type: {type(gt_data)}")
            return None
        
        return self._convert_to_4x4_list(poses)
    
    def _load_numpy_poses(self, file_path: str) -> Optional[List[np.ndarray]]:
        """Load poses from numpy file."""
        gt_data = np.load(file_path, allow_pickle=True)
        
        if isinstance(gt_data, dict) or (hasattr(gt_data, 'files') and len(gt_data.files) > 0):
            # NPZ file or dict
            if hasattr(gt_data, 'files'):  # NPZ
                # Try to find poses in NPZ file
                for key in gt_data.files:
                    if 'pose' in key.lower() or 'cam' in key.lower():
                        poses = gt_data[key]
                        break
                else:
                    poses = gt_data[gt_data.files[0]]  # Use first array
            else:
                # Dictionary from .npy with allow_pickle=True
                for key in ['poses', 'camera_poses', 'c2w', 'w2c']:
                    if key in gt_data:
                        poses = gt_data[key]
                        break
                else:
                    print(f"[WARN] No recognized pose key found in GT file: {file_path}")
                    return None
        else:
            # Direct array
            poses = gt_data
        
        return self._convert_to_4x4_list(poses)
    
    def _load_json_poses(self, file_path: str) -> Optional[List[np.ndarray]]:
        """Load poses from JSON file."""
        with open(file_path, 'r') as f:
            gt_data = json.load(f)
        
        # Try to find poses in JSON structure
        if isinstance(gt_data, dict):
            for key in ['poses', 'camera_poses', 'c2w', 'w2c', 'extrinsics']:
                if key in gt_data:
                    poses = np.array(gt_data[key])
                    return self._convert_to_4x4_list(poses)
        
        # Try direct conversion
        poses = np.array(gt_data)
        return self._convert_to_4x4_list(poses)
    
    def _load_text_poses(self, file_path: str) -> Optional[List[np.ndarray]]:
        """Load poses from text file (assuming space/tab separated values)."""
        try:
            poses = np.loadtxt(file_path)
            return self._convert_to_4x4_list(poses)
        except Exception as e:
            print(f"[WARN] Failed to load text poses: {e}")
            return None
    
    def _convert_to_4x4_list(self, poses: np.ndarray) -> Optional[List[np.ndarray]]:
        """Convert various pose formats to list of 4x4 matrices."""
        poses = np.array(poses)
        
        if poses.ndim == 3 and poses.shape[-2:] == (4, 4):
            # Already (N, 4, 4)
            return [poses[i] for i in range(poses.shape[0])]
        elif poses.ndim == 2 and poses.shape == (4, 4):
            # Single pose (4, 4)
            return [poses]
        elif poses.ndim == 3 and poses.shape[-2:] == (3, 4):
            # DynPose format: (N, 3, 4) - convert to (N, 4, 4)
            print(f"[INFO] Converting DynPose format from {poses.shape} to (N, 4, 4)")
            num_poses = poses.shape[0]
            poses_4x4 = []
            for i in range(num_poses):
                pose_4x4 = np.eye(4)
                pose_4x4[:3, :] = poses[i]  # Copy 3x4 into top part of 4x4
                poses_4x4.append(pose_4x4)
            return poses_4x4
        elif poses.ndim == 2 and poses.shape[1] == 12:
            # Flattened 3x4 matrices: reshape to (N, 3, 4) then convert
            poses_3x4 = poses.reshape(-1, 3, 4)
            return self._convert_to_4x4_list(poses_3x4)
        elif poses.ndim == 2 and poses.shape[1] == 16:
            # Flattened 4x4 matrices: reshape to (N, 4, 4)
            poses_4x4 = poses.reshape(-1, 4, 4)
            return self._convert_to_4x4_list(poses_4x4)
        else:
            print(f"[WARN] Unexpected GT pose format: {poses.shape}")
            print(f"[DEBUG] Expected: (N, 4, 4), (4, 4), or (N, 3, 4), got: {poses.shape}")
            return None
    
    def _load_raw_pickle_data(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Load raw data from pickle file."""
        with open(file_path, 'rb') as f:
            gt_data = pickle.load(f)
        
        # Return the raw data as-is
        if isinstance(gt_data, dict):
            return gt_data
        else:
            print(f"[WARN] Pickle file does not contain dictionary data: {type(gt_data)}")
            return None
    
    def _load_raw_numpy_data(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Load raw data from numpy file."""
        gt_data = np.load(file_path, allow_pickle=True)
        
        if isinstance(gt_data, dict) or (hasattr(gt_data, 'files') and len(gt_data.files) > 0):
            # NPZ file or dict
            if hasattr(gt_data, 'files'):  # NPZ
                # Convert NPZ to dict
                return {key: gt_data[key] for key in gt_data.files}
            else:
                # Dictionary from .npy with allow_pickle=True
                return gt_data
        else:
            print(f"[WARN] Numpy file does not contain dictionary data: {type(gt_data)}")
            return None
    
    def _load_raw_json_data(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Load raw data from JSON file."""
        with open(file_path, 'r') as f:
            gt_data = json.load(f)
        
        if isinstance(gt_data, dict):
            return gt_data
        else:
            print(f"[WARN] JSON file does not contain dictionary data: {type(gt_data)}")
            return None
    
    def find_matching_gt_file(self, video_path: str, gt_dir: str) -> Optional[str]:
        """
        Find the corresponding ground truth file for a video.
        
        Args:
            video_path: Path to video file
            gt_dir: Directory containing ground truth files
            
        Returns:
            Path to matching GT file, or None if not found
        """
        video_name = Path(video_path).stem
        gt_dir_path = Path(gt_dir)
        
        # Try different possible extensions and naming conventions
        # Prioritize .pkl files for DynPose dataset
        possible_names = [
            f"{video_name}.pkl",           # DynPose format
            f"{video_name}.npy",
            f"{video_name}.npz", 
            f"{video_name}_poses.npy",
            f"{video_name}_poses.pkl",
            f"{video_name}_camera.npy",
            f"{video_name}_camera.pkl",
            f"{video_name}.txt",
            f"{video_name}.json"
        ]
        
        for name in possible_names:
            gt_file = gt_dir_path / name
            if gt_file.exists():
                return str(gt_file)
        
        print(f"[WARN] No matching ground truth found for video: {video_name}")
        print(f"[DEBUG] Searched for: {possible_names}")
        return None


class ExperimentDataManager:
    """
    High-level data manager that combines frame loading and ground truth handling.
    """
    
    def __init__(self):
        self.frame_loader = FrameLoader()
        self.gt_loader = GroundTruthLoader()
    
    def load_experiment_data(self, input_path: str, 
                           num_frames: int = 3,
                           start_frame: int = 0,
                           skip_frames: int = 1,
                           gt_dir: Optional[str] = None) -> Tuple[List[np.ndarray], Optional[Dict[str, Any]]]:
        """
        Load both frames and ground truth data for an experiment.
        
        Args:
            input_path: Path to video or image folder
            num_frames: Number of frames to extract
            start_frame: Starting frame index
            skip_frames: Frame skip interval
            gt_dir: Optional ground truth directory
            
        Returns:
            Tuple of (frames, gt_data) where gt_data is None if not available
        """
        # Load frames
        frames = self.frame_loader.load_frames(input_path, num_frames, start_frame, skip_frames)
        
        # Load ground truth if available
        gt_data = None
        if gt_dir:
            gt_file = self.gt_loader.find_matching_gt_file(input_path, gt_dir)
            if gt_file:
                print(f"[GT] Found matching ground truth: {gt_file}")
                gt_data = self.gt_loader.load_raw_gt_data(gt_file)
                if gt_data:
                    print(f"[OK] Loaded ground truth data with keys: {list(gt_data.keys())}")
                else:
                    print(f"[WARN] Failed to load ground truth data")
        
        return frames, gt_data
    
    def get_experiment_info(self, input_path: str) -> Dict[str, Any]:
        """Get information about the input data."""
        input_path = Path(input_path)
        
        if self.frame_loader._is_video_file(input_path):
            return self.frame_loader.get_video_info(str(input_path))
        elif input_path.is_dir():
            image_files = []
            for ext in self.frame_loader.supported_image_extensions:
                image_files.extend(glob(str(input_path / ext)))
            return {
                'type': 'image_folder',
                'total_images': len(image_files),
                'folder_path': str(input_path)
            }
        else:
            return {'type': 'unknown'} 