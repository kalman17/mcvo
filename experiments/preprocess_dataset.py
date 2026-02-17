#!/usr/bin/env python3
"""
Preprocessing Script for FAT and DA3 Training Pipelines

Preprocesses video datasets by running:
1. AnyCalib - on every frame (calibration)
2. UniDepth - on every frame (depth maps)
3. UniMatch - on consecutive frame pairs (optical flow)

Execution strategy:
- PARALLEL mode (default attempt): Load all models simultaneously, run each
  in its own thread processing frames sequentially at maximum speed.
- SERIAL mode (fallback): If all models don't fit in VRAM at once, load one
  model at a time, process all its data, unload, then load the next.

The --test flag determines which mode is possible on the current hardware.

Results are saved in per-frame .npz files (compressed numpy archives).

Data Organization (per supervisor discussion):
    output_dir/
      dataset_name/
        video_01/
          000000.npz  # First frame: forward_flow only (no backward, no depth, no calib)
          000001.npz  # Middle frames: forward_flow, backward_flow, depth, calib
          000002.npz
          ...
          NNNNNN.npz  # Last frame: backward_flow only (no forward, no depth, no calib)
        video_02/
          000000.npz
          ...

Each .data file contains (as applicable):
    - forward_flow: Flow from frame i to frame i+1 [2, H, W]
    - backward_flow: Flow from frame i to frame i-1 [2, H, W]
    - forward_occ: Forward flow occlusion mask [1, H, W]
    - backward_occ: Backward flow occlusion mask [1, H, W]
    - depth: Inverse depth map [1, H, W]
    - calib: Calibration intrinsics [4] (fx, fy, cx, cy)

Usage:
    # Test mode (validates models and determines parallel vs serial)
    python preprocess_dataset.py --dataset_path /data/videos --output_dir /data/preprocessed --test

    # Full preprocessing
    python preprocess_dataset.py --dataset_path /data/RealEstate10K --output_dir /data/preprocessed --dataset_name RealEstate10K

    # With visualization for debugging
    python preprocess_dataset.py --dataset_path /data/videos --output_dir /data/preprocessed --dataset_name Test --visualize

Author: Kalman Mahlich
Purpose: Master's Thesis - Preprocessing for cluster training
"""

import os
import sys
import argparse
import json
import cv2
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field, asdict
import logging
from tqdm import tqdm
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
import shutil
import requests
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless servers
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger(__name__)


# =============================================================================
# MODEL IMPORTS - ADJUST PATHS IF NEEDED
# =============================================================================
# These imports match the actual usage patterns in the anycam-extension codebase.
# Make sure the anycalib and unimatch submodules are present.

ANYCALIB_AVAILABLE = False
UNIDEPTH_AVAILABLE = False
UNIMATCH_AVAILABLE = False

# Add parent directory to path for submodule imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "anycalib"))
sys.path.insert(0, str(PROJECT_ROOT / "unimatch"))

try:
    # AnyCalib - from anycalib submodule
    from anycalib import AnyCalib
    ANYCALIB_AVAILABLE = True
    logger.info("AnyCalib imported successfully")
except ImportError as e:
    logger.warning(f"AnyCalib not available: {e}")
    logger.warning("  Make sure anycalib submodule is initialized: git submodule update --init")
    AnyCalib = None

try:
    # UniDepth - loaded via torch.hub (same as AnyCam does)
    # Will be loaded dynamically to avoid startup delays
    UNIDEPTH_AVAILABLE = True
    logger.info("UniDepth will be loaded via torch.hub")
except Exception as e:
    logger.warning(f"UniDepth setup issue: {e}")

try:
    # UniMatch - from unimatch submodule
    from unimatch.unimatch import UniMatch
    UNIMATCH_AVAILABLE = True
    logger.info("UniMatch imported successfully")
except ImportError as e:
    logger.warning(f"UniMatch not available: {e}")
    logger.warning("  Make sure unimatch submodule is initialized: git submodule update --init")
    UniMatch = None

# UniMatch checkpoint URL and path
UNIMATCH_CHECKPOINT_URL = "https://s3.eu-central-1.amazonaws.com/avg-projects/unimatch/pretrained/gmflow-scale2-regrefine6-mixdata-train320x576-4e7b215d.pth"
UNIMATCH_CHECKPOINT_NAME = "gmflow-scale2-regrefine6-mixdata-train320x576-4e7b215d.pth"

# =============================================================================


def compute_occlusions(flow_fwd: torch.Tensor, flow_bwd: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute occlusion masks from forward and backward flow using forward-backward consistency.
    Same implementation as AnyCam uses.

    Args:
        flow_fwd: Forward flow [B, 2, H, W]
        flow_bwd: Backward flow [B, 2, H, W]

    Returns:
        occ0: Occlusion mask for frame 0 [B, 1, H, W]
        occ1: Occlusion mask for frame 1 [B, 1, H, W]
    """
    import torch.nn.functional as F

    b, _, h, w = flow_fwd.shape

    # Create coordinate grid
    coords_y, coords_x = torch.meshgrid(
        torch.arange(h, device=flow_fwd.device, dtype=flow_fwd.dtype),
        torch.arange(w, device=flow_fwd.device, dtype=flow_fwd.dtype),
        indexing='ij'
    )
    coords = torch.stack([coords_x, coords_y], dim=0).unsqueeze(0).expand(b, -1, -1, -1)

    # Warp backward flow using forward flow
    grid_fwd = coords + flow_fwd
    grid_fwd_normalized = torch.stack([
        grid_fwd[:, 0] * 2 / (w - 1) - 1,
        grid_fwd[:, 1] * 2 / (h - 1) - 1
    ], dim=1).permute(0, 2, 3, 1)

    flow_bwd_warped = F.grid_sample(
        flow_bwd, grid_fwd_normalized,
        mode='bilinear', padding_mode='zeros', align_corners=True
    )

    # Forward-backward consistency check
    fb_diff = flow_fwd + flow_bwd_warped
    fb_error = torch.sum(fb_diff ** 2, dim=1, keepdim=True)
    fb_norm = torch.sum(flow_fwd ** 2, dim=1, keepdim=True) + torch.sum(flow_bwd_warped ** 2, dim=1, keepdim=True)

    # Occlusion threshold (same as RAFT/AnyCam)
    occ0 = (fb_error > 0.01 * fb_norm + 0.5).float()

    # Similarly for reverse direction
    grid_bwd = coords + flow_bwd
    grid_bwd_normalized = torch.stack([
        grid_bwd[:, 0] * 2 / (w - 1) - 1,
        grid_bwd[:, 1] * 2 / (h - 1) - 1
    ], dim=1).permute(0, 2, 3, 1)

    flow_fwd_warped = F.grid_sample(
        flow_fwd, grid_bwd_normalized,
        mode='bilinear', padding_mode='zeros', align_corners=True
    )

    bf_diff = flow_bwd + flow_fwd_warped
    bf_error = torch.sum(bf_diff ** 2, dim=1, keepdim=True)
    bf_norm = torch.sum(flow_bwd ** 2, dim=1, keepdim=True) + torch.sum(flow_fwd_warped ** 2, dim=1, keepdim=True)
    occ1 = (bf_error > 0.01 * bf_norm + 0.5).float()

    return occ0, occ1


@dataclass
class PreprocessingConfig:
    """Configuration for preprocessing."""
    dataset_path: Path
    output_dir: Path
    dataset_name: str

    # File extension for per-frame data
    data_extension: str = ".npz"

    # Video extensions to process
    video_extensions: Tuple[str, ...] = (".mp4", ".avi", ".mov", ".mkv", ".MOV", ".MP4", ".AVI")

    # Resolution: resize frames so short side = this value (None = no resize)
    resize_short_side: Optional[int] = None

    # Square resize: resize all frames to this square size (e.g. 336)
    # Overrides resize_short_side when set. Matches AnyCam training resolution.
    image_size: Optional[int] = None

    # Visualization
    visualize: bool = False
    vis_samples_per_video: int = 3  # Number of sample frames to visualize per video


@dataclass
class ProgressTracker:
    """Tracks preprocessing progress for resumption."""
    dataset_name: str
    total_videos: int = 0
    processed_videos: int = 0

    # Per-video tracking: {video_name: [frame_indices_done]}
    completed_frames: Dict[str, List[int]] = field(default_factory=dict)

    # Failed items for retry
    failed_frames: Dict[str, List[int]] = field(default_factory=dict)

    # Metadata
    last_updated: str = ""
    parallel_mode: bool = False

    def save(self, path: Path):
        """Save progress to JSON file."""
        self.last_updated = time.strftime("%Y-%m-%d %H:%M:%S")
        data = asdict(self)
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "ProgressTracker":
        """Load progress from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)

    def is_frame_done(self, video_name: str, frame_idx: int) -> bool:
        """Check if a frame has been processed."""
        return frame_idx in self.completed_frames.get(video_name, [])

    def mark_frame_done(self, video_name: str, frame_idx: int):
        """Mark a frame as successfully processed."""
        if video_name not in self.completed_frames:
            self.completed_frames[video_name] = []
        if frame_idx not in self.completed_frames[video_name]:
            self.completed_frames[video_name].append(frame_idx)

    def mark_frame_failed(self, video_name: str, frame_idx: int):
        """Mark a frame as failed (for retry)."""
        if video_name not in self.failed_frames:
            self.failed_frames[video_name] = []
        if frame_idx not in self.failed_frames[video_name]:
            self.failed_frames[video_name].append(frame_idx)


class VideoProcessor:
    """Handles video reading and frame extraction."""

    def __init__(self, video_path: Path):
        self.video_path = video_path
        self.cap = None
        self._frame_count = None
        self._fps = None
        self._width = None
        self._height = None

    def open(self):
        """Open video file."""
        self.cap = cv2.VideoCapture(str(self.video_path))
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open video: {self.video_path}")
        self._frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._fps = self.cap.get(cv2.CAP_PROP_FPS)
        self._width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def close(self):
        """Close video file."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def resolution(self) -> Tuple[int, int]:
        return (self._width, self._height)

    def get_frame(self, frame_idx: int) -> Optional[np.ndarray]:
        """Get a specific frame (BGR format)."""
        if self.cap is None:
            self.open()
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame

    def get_frames(self, frame_indices: List[int]) -> List[Optional[np.ndarray]]:
        """Get multiple frames efficiently."""
        frames = []
        for idx in frame_indices:
            frames.append(self.get_frame(idx))
        return frames

    def get_frame_pair(self, idx1: int, idx2: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Get a pair of frames."""
        return self.get_frame(idx1), self.get_frame(idx2)

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class OutputManager:
    """Manages output file paths and saving."""

    def __init__(self, config: PreprocessingConfig):
        self.config = config
        self.base_dir = config.output_dir / config.dataset_name
        self.base_dir.mkdir(parents=True, exist_ok=True)

        # Create visualization directory if needed
        if config.visualize:
            self.vis_dir = self.base_dir / "_visualizations"
            self.vis_dir.mkdir(exist_ok=True)

    def _sanitize_video_name(self, video_path: Path) -> str:
        """Create a safe, unique identifier for a video."""
        # Use relative path from dataset root, replace separators
        rel_path = video_path.relative_to(self.config.dataset_path)
        # Remove extension and replace path separators with underscores
        name = str(rel_path.with_suffix("")).replace(os.sep, "_").replace("/", "_")
        return name

    def get_video_dir(self, video_path: Path) -> Path:
        """Get output directory for a video."""
        video_name = self._sanitize_video_name(video_path)
        video_dir = self.base_dir / video_name
        video_dir.mkdir(parents=True, exist_ok=True)
        return video_dir

    def get_frame_data_path(self, video_path: Path, frame_idx: int) -> Path:
        """Get output path for a frame's data file."""
        video_dir = self.get_video_dir(video_path)
        return video_dir / f"{frame_idx:06d}{self.config.data_extension}"

    def save_frame_data(
        self,
        path: Path,
        forward_flow: Optional[np.ndarray] = None,
        backward_flow: Optional[np.ndarray] = None,
        forward_occ: Optional[np.ndarray] = None,
        backward_occ: Optional[np.ndarray] = None,
        depth: Optional[np.ndarray] = None,
        calib: Optional[np.ndarray] = None,
        metadata: Optional[Dict] = None
    ):
        """
        Save all data for a frame to a single compressed file.

        Args:
            path: Output path
            forward_flow: Flow from this frame to next [2, H, W]
            backward_flow: Flow from this frame to previous [2, H, W]
            forward_occ: Forward occlusion mask [1, H, W]
            backward_occ: Backward occlusion mask [1, H, W]
            depth: Inverse depth map [1, H, W]
            calib: Calibration intrinsics [4] (fx, fy, cx, cy)
            metadata: Optional metadata dict
        """
        save_dict = {}

        if forward_flow is not None:
            save_dict["forward_flow"] = forward_flow.astype(np.float16)
        if backward_flow is not None:
            save_dict["backward_flow"] = backward_flow.astype(np.float16)
        if forward_occ is not None:
            save_dict["forward_occ"] = forward_occ.astype(np.float16)
        if backward_occ is not None:
            save_dict["backward_occ"] = backward_occ.astype(np.float16)
        if depth is not None:
            save_dict["depth"] = depth.astype(np.float16)
        if calib is not None:
            save_dict["calib"] = calib.astype(np.float32)

        if metadata:
            save_dict["metadata_json"] = np.array(json.dumps(metadata))

        np.savez_compressed(path, **save_dict)

    def get_vis_path(self, video_path: Path, frame_idx: int) -> Path:
        """Get visualization output path."""
        video_name = self._sanitize_video_name(video_path)
        return self.vis_dir / f"{video_name}_{frame_idx:06d}_vis.png"


class Visualizer:
    """Creates visualizations of preprocessed data for debugging."""

    @staticmethod
    def visualize_flow(flow: np.ndarray) -> np.ndarray:
        """
        Convert optical flow to RGB visualization using HSV color wheel.

        Args:
            flow: Flow field [2, H, W] or [H, W, 2]

        Returns:
            RGB image [H, W, 3] in uint8
        """
        if flow.shape[0] == 2:
            flow = flow.transpose(1, 2, 0)  # [H, W, 2]

        h, w = flow.shape[:2]
        fx, fy = flow[:, :, 0], flow[:, :, 1]

        # Compute angle and magnitude
        angle = np.arctan2(fy, fx) + np.pi
        magnitude = np.sqrt(fx**2 + fy**2)

        # Normalize magnitude
        max_mag = np.percentile(magnitude, 99)
        if max_mag > 0:
            magnitude = np.clip(magnitude / max_mag, 0, 1)

        # Create HSV image
        hsv = np.zeros((h, w, 3), dtype=np.float32)
        hsv[:, :, 0] = angle / (2 * np.pi)  # Hue from angle
        hsv[:, :, 1] = 1.0  # Full saturation
        hsv[:, :, 2] = magnitude  # Value from magnitude

        # Convert to RGB
        rgb = matplotlib.colors.hsv_to_rgb(hsv)
        return (rgb * 255).astype(np.uint8)

    @staticmethod
    def visualize_depth(depth: np.ndarray) -> np.ndarray:
        """
        Convert depth map to colorized visualization.

        Args:
            depth: Depth map [1, H, W] or [H, W]

        Returns:
            RGB image [H, W, 3] in uint8
        """
        if depth.ndim == 3:
            depth = depth[0]

        # Cast to float32 to avoid overflow in np.percentile with large float16 arrays
        depth = depth.astype(np.float32)

        # Normalize depth
        valid_mask = (depth > 0) & np.isfinite(depth)
        if valid_mask.sum() > 0:
            d_min = np.percentile(depth[valid_mask], 1)
            d_max = np.percentile(depth[valid_mask], 99)
            depth_norm = np.clip((depth - d_min) / (d_max - d_min + 1e-8), 0, 1)
        else:
            depth_norm = np.zeros_like(depth)

        # Apply colormap
        cmap = plt.cm.viridis
        rgb = cmap(depth_norm)[:, :, :3]
        return (rgb * 255).astype(np.uint8)

    @staticmethod
    def create_visualization(
        frame: np.ndarray,
        forward_flow: Optional[np.ndarray] = None,
        backward_flow: Optional[np.ndarray] = None,
        depth: Optional[np.ndarray] = None,
        calib: Optional[np.ndarray] = None,
        frame_idx: int = 0
    ) -> np.ndarray:
        """
        Create a combined visualization of all preprocessed data.

        Layout:
            [Original Frame] [Forward Flow] [Backward Flow]
            [Depth Map]      [Calibration Info Panel]
        """
        h, w = frame.shape[:2]

        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Create panels
        panels = []

        # Row 1: Frame, Forward Flow, Backward Flow
        row1 = [frame_rgb]

        if forward_flow is not None:
            fwd_vis = Visualizer.visualize_flow(forward_flow)
            row1.append(fwd_vis)
        else:
            row1.append(np.zeros_like(frame_rgb))

        if backward_flow is not None:
            bwd_vis = Visualizer.visualize_flow(backward_flow)
            row1.append(bwd_vis)
        else:
            row1.append(np.zeros_like(frame_rgb))

        panels.append(np.concatenate(row1, axis=1))

        # Row 2: Depth, Info Panel
        row2 = []

        if depth is not None:
            depth_vis = Visualizer.visualize_depth(depth)
            row2.append(depth_vis)
        else:
            row2.append(np.zeros_like(frame_rgb))

        # Info panel
        info_panel = np.ones((h, w * 2, 3), dtype=np.uint8) * 40  # Dark gray

        # Add text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        color = (255, 255, 255)
        y_offset = 30

        cv2.putText(info_panel, f"Frame: {frame_idx}", (10, y_offset), font, font_scale, color, 1)
        y_offset += 30

        if calib is not None:
            cv2.putText(info_panel, f"fx: {calib[0]:.2f}", (10, y_offset), font, font_scale, color, 1)
            y_offset += 25
            cv2.putText(info_panel, f"fy: {calib[1]:.2f}", (10, y_offset), font, font_scale, color, 1)
            y_offset += 25
            cv2.putText(info_panel, f"cx: {calib[2]:.2f}", (10, y_offset), font, font_scale, color, 1)
            y_offset += 25
            cv2.putText(info_panel, f"cy: {calib[3]:.2f}", (10, y_offset), font, font_scale, color, 1)
        else:
            cv2.putText(info_panel, "No calibration", (10, y_offset), font, font_scale, (255, 100, 100), 1)
            y_offset += 30
            cv2.putText(info_panel, "(first/last frame)", (10, y_offset), font, font_scale, (255, 100, 100), 1)

        row2.append(info_panel)
        panels.append(np.concatenate(row2, axis=1))

        # Combine rows
        vis = np.concatenate(panels, axis=0)

        # Add labels
        label_color = (255, 255, 0)
        cv2.putText(vis, "Original", (10, 25), font, 0.7, label_color, 2)
        cv2.putText(vis, "Forward Flow", (w + 10, 25), font, 0.7, label_color, 2)
        cv2.putText(vis, "Backward Flow", (2*w + 10, 25), font, 0.7, label_color, 2)
        cv2.putText(vis, "Depth", (10, h + 25), font, 0.7, label_color, 2)
        cv2.putText(vis, "Calibration", (w + 10, h + 25), font, 0.7, label_color, 2)

        return vis


class ModelRunner:
    """Runs inference with the three models (one frame at a time)."""

    def __init__(self, config: PreprocessingConfig):
        self.config = config
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Models will be loaded on demand
        self._depth_model = None
        self._calib_model = None
        self._flow_model = None

        # UniDepth configuration
        self._unidepth_version = "v2"
        self._unidepth_backbone = "vits14"  # Use small backbone for speed, 'vitl14' for quality
        self._unidepth_scaling = 0.1  # Same as AnyCam uses

    # ----- Model loading -----

    def _load_depth_model(self):
        """Load UniDepth model via torch.hub (same as AnyCam)."""
        if not UNIDEPTH_AVAILABLE:
            raise RuntimeError("UniDepth not available.")

        logger.info(f"Loading UniDepth {self._unidepth_version} with {self._unidepth_backbone} backbone...")
        self._depth_model = torch.hub.load(
            "Brummi/UniDepth:stable",
            "UniDepth",
            version=self._unidepth_version,
            backbone=self._unidepth_backbone,
            pretrained=True,
            trust_repo=True,
            force_reload=False
        )
        self._depth_model = self._depth_model.to(self.device)
        self._depth_model.eval()
        for param in self._depth_model.parameters():
            param.requires_grad = False
        logger.info("UniDepth model loaded")

    def _load_calib_model(self):
        """Load AnyCalib model."""
        if not ANYCALIB_AVAILABLE:
            raise RuntimeError("AnyCalib not available. Check imports at top of file.")

        logger.info("Loading AnyCalib model (anycalib_pinhole)...")
        self._calib_model = AnyCalib(model_id="anycalib_pinhole")
        self._calib_model = self._calib_model.to(self.device)
        self._calib_model.eval()
        for param in self._calib_model.parameters():
            param.requires_grad = False
        logger.info("AnyCalib model loaded")

    def _load_flow_model(self):
        """Load UniMatch model with pretrained weights."""
        if not UNIMATCH_AVAILABLE:
            raise RuntimeError("UniMatch not available. Check imports at top of file.")

        logger.info("Loading UniMatch model...")

        # Download checkpoint if not present
        cache_dir = Path.home() / ".cache" / "torch" / "checkpoints"
        cache_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = cache_dir / UNIMATCH_CHECKPOINT_NAME

        if not ckpt_path.exists():
            logger.info(f"Downloading UniMatch checkpoint from {UNIMATCH_CHECKPOINT_URL}")
            r = requests.get(UNIMATCH_CHECKPOINT_URL)
            with open(ckpt_path, 'wb') as f:
                f.write(r.content)
            logger.info(f"Saved checkpoint to {ckpt_path}")

        # Initialize model with same config as AnyCam
        self._flow_model = UniMatch(
            feature_channels=128,
            num_scales=2,
            upsample_factor=4,
            ffn_dim_expansion=4,
            num_transformer_layers=6,
            reg_refine=True,
            task='flow'
        )

        # Load pretrained weights
        checkpoint = torch.load(ckpt_path, map_location=self.device)
        self._flow_model.load_state_dict(checkpoint['model'], strict=True)
        self._flow_model = self._flow_model.to(self.device)
        self._flow_model.eval()
        for param in self._flow_model.parameters():
            param.requires_grad = False
        logger.info("UniMatch model loaded with pretrained weights")

    def load_all_models(self):
        """
        Attempt to load all three models simultaneously.
        Raises torch.cuda.OutOfMemoryError if they don't all fit.
        """
        if self._flow_model is None and UNIMATCH_AVAILABLE:
            self._load_flow_model()
        if self._depth_model is None and UNIDEPTH_AVAILABLE:
            self._load_depth_model()
        if self._calib_model is None and ANYCALIB_AVAILABLE:
            self._load_calib_model()

    def unload_all_models(self):
        """Unload all models to free VRAM."""
        self._depth_model = None
        self._calib_model = None
        self._flow_model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def unload_flow_model(self):
        self._flow_model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def unload_depth_model(self):
        self._depth_model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def unload_calib_model(self):
        self._calib_model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @property
    def depth_loaded(self) -> bool:
        return self._depth_model is not None

    @property
    def calib_loaded(self) -> bool:
        return self._calib_model is not None

    @property
    def flow_loaded(self) -> bool:
        return self._flow_model is not None

    # ----- Preprocessing helpers -----

    def _preprocess_frame_for_depth(self, frame_bgr: np.ndarray) -> torch.Tensor:
        """Preprocess frame for UniDepth (expects [0, 1] normalized RGB)."""
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_tensor = torch.from_numpy(frame_rgb).float() / 255.0
        frame_tensor = frame_tensor.permute(2, 0, 1)
        return frame_tensor

    def _preprocess_frame_for_calib(self, frame_bgr: np.ndarray) -> torch.Tensor:
        """Preprocess frame for AnyCalib (expects [0, 1] normalized RGB)."""
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_tensor = torch.from_numpy(frame_rgb).float() / 255.0
        frame_tensor = frame_tensor.permute(2, 0, 1)
        return frame_tensor

    def _preprocess_frame_for_flow(self, frame_bgr: np.ndarray) -> torch.Tensor:
        """Preprocess frame for UniMatch (expects [0, 255] RGB)."""
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_tensor = torch.from_numpy(frame_rgb).float()
        frame_tensor = frame_tensor.permute(2, 0, 1)
        return frame_tensor

    # ----- Inference (single-frame / single-pair) -----

    def run_depth_single(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Run UniDepth on a single frame. Returns inverse depth [1, H, W] or None."""
        if self._depth_model is None:
            self._load_depth_model()

        if frame is None:
            return None

        img = self._preprocess_frame_for_depth(frame).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self._depth_model.infer(img)
            depth_map = outputs["depth"]  # [1, 1, H, W]
            depth_map = depth_map * self._unidepth_scaling
            depth_map = 1.0 / depth_map
            return depth_map[0].cpu().numpy()

    def run_calib_single(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Run AnyCalib on a single frame. Returns [fx, fy, cx, cy] or None."""
        if self._calib_model is None:
            self._load_calib_model()

        if frame is None:
            return None

        img = self._preprocess_frame_for_calib(frame).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self._calib_model.predict(img, cam_id="pinhole")
            intrinsics_list = output.get("intrinsics", [])
            if not intrinsics_list:
                return None
            intr = intrinsics_list[0]
            if isinstance(intr, torch.Tensor):
                intr = intr.cpu().numpy()
            return np.array(intr, dtype=np.float32)

    def run_flow_pair(self, frame1: np.ndarray, frame2: np.ndarray) -> Optional[Dict[str, np.ndarray]]:
        """
        Run UniMatch on a frame pair.

        Args:
            frame1: First frame (BGR)
            frame2: Second frame (BGR)

        Returns:
            Dict with:
                - 'flow_fwd': Flow from frame1 to frame2 [2, H, W]
                - 'flow_bwd': Flow from frame2 to frame1 [2, H, W]
                - 'occ_fwd': Forward occlusion [1, H, W]
                - 'occ_bwd': Backward occlusion [1, H, W]
        """
        import math
        import torch.nn.functional as F

        if self._flow_model is None:
            self._load_flow_model()

        if frame1 is None or frame2 is None:
            return None

        img0 = self._preprocess_frame_for_flow(frame1).unsqueeze(0).to(self.device)
        img1 = self._preprocess_frame_for_flow(frame2).unsqueeze(0).to(self.device)

        n, c, h, w = img0.shape

        # UniMatch preprocessing (same as AnyCam image_processor.py)
        max_size = 320
        smaller = min(h, w)

        if smaller > max_size:
            scale_factor = max_size / smaller
            target_h = h * scale_factor
            target_w = w * scale_factor
        else:
            target_h = h
            target_w = w

        target_h = int(math.ceil(target_h / 32) * 32)
        target_w = int(math.ceil(target_w / 32) * 32)

        if target_h != h or target_w != w:
            img0 = F.interpolate(img0, (target_h, target_w), mode='bilinear', align_corners=True)
            img1 = F.interpolate(img1, (target_h, target_w), mode='bilinear', align_corners=True)

        # Handle portrait orientation
        transposed = False
        if target_h > target_w:
            img0 = img0.permute(0, 1, 3, 2)
            img1 = img1.permute(0, 1, 3, 2)
            transposed = True

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=False):
                results_dict = self._flow_model(
                    img0, img1,
                    attn_type='swin',
                    attn_splits_list=[2, 8],
                    corr_radius_list=[-1, 4],
                    prop_radius_list=[-1, 1],
                    num_reg_refine=6,
                    task="flow",
                    pred_bidir_flow=True,
                )

            flows = results_dict['flow_preds'][-1]  # [2, 2, H, W] (fwd and bwd)

            if transposed:
                flows = flows.permute(0, 1, 3, 2)
                flows = flows[:, [1, 0], :, :]

            if target_h != h or target_w != w:
                flows = F.interpolate(flows, (h, w), mode='bilinear', align_corners=True)

            flow_fwd = flows[:n]  # [1, 2, H, W]
            flow_bwd = flows[n:]  # [1, 2, H, W]

            occ_fwd, occ_bwd = compute_occlusions(flow_fwd, flow_bwd)

            return {
                'flow_fwd': flow_fwd[0].cpu().numpy(),
                'flow_bwd': flow_bwd[0].cpu().numpy(),
                'occ_fwd': occ_fwd[0].cpu().numpy(),
                'occ_bwd': occ_bwd[0].cpu().numpy(),
            }

    # ----- Test mode -----

    def test_all_models(self, frame1: np.ndarray, frame2: np.ndarray) -> Tuple[bool, bool]:
        """
        Test all models and determine if parallel mode is possible.

        First tests each model individually (to validate they work at all).
        Then attempts to load all models simultaneously and run inference
        to determine if parallel mode fits in VRAM.

        Returns:
            (all_models_passed, parallel_possible)
        """
        all_passed = True

        # Phase 1: Test each model individually
        logger.info("\n" + "=" * 50)
        logger.info("PHASE 1: Testing each model individually")
        logger.info("=" * 50)

        # Test UniDepth
        logger.info("\n[1/3] Testing UniDepth...")
        if UNIDEPTH_AVAILABLE:
            try:
                result = self.run_depth_single(frame1)
                if result is not None:
                    logger.info(f"  PASSED - output shape: {result.shape}")
                else:
                    logger.error("  FAILED - returned None")
                    all_passed = False
            except Exception as e:
                logger.error(f"  FAILED: {e}")
                all_passed = False
        else:
            logger.warning("  SKIPPED - UniDepth not available")

        self.unload_all_models()

        # Test AnyCalib
        logger.info("\n[2/3] Testing AnyCalib...")
        if ANYCALIB_AVAILABLE:
            try:
                result = self.run_calib_single(frame1)
                if result is not None:
                    logger.info(f"  PASSED - intrinsics: {result}")
                else:
                    logger.error("  FAILED - returned None")
                    all_passed = False
            except Exception as e:
                logger.error(f"  FAILED: {e}")
                all_passed = False
        else:
            logger.warning("  SKIPPED - AnyCalib not available")

        self.unload_all_models()

        # Test UniMatch
        logger.info("\n[3/3] Testing UniMatch...")
        if UNIMATCH_AVAILABLE:
            try:
                result = self.run_flow_pair(frame1, frame2)
                if result is not None:
                    logger.info(f"  PASSED - flow shape: {result['flow_fwd'].shape}")
                else:
                    logger.error("  FAILED - returned None")
                    all_passed = False
            except Exception as e:
                logger.error(f"  FAILED: {e}")
                all_passed = False
        else:
            logger.warning("  SKIPPED - UniMatch not available")

        self.unload_all_models()

        if not all_passed:
            return False, False

        # Phase 2: Test parallel loading (all models at once)
        logger.info("\n" + "=" * 50)
        logger.info("PHASE 2: Testing parallel mode (all models loaded at once)")
        logger.info("=" * 50)

        parallel_possible = False
        try:
            logger.info("  Loading all models simultaneously...")
            self.load_all_models()

            if torch.cuda.is_available():
                allocated_gb = torch.cuda.memory_allocated(0) / (1024**3)
                total_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                logger.info(f"  VRAM usage after loading: {allocated_gb:.2f} / {total_gb:.2f} GB")

            # Run inference with all models loaded to confirm no OOM during actual work
            logger.info("  Running inference with all models loaded...")

            depth_ok = self.run_depth_single(frame1) is not None
            calib_ok = self.run_calib_single(frame1) is not None
            flow_ok = self.run_flow_pair(frame1, frame2) is not None

            if depth_ok and calib_ok and flow_ok:
                parallel_possible = True
                logger.info("  PARALLEL MODE: POSSIBLE")

                if torch.cuda.is_available():
                    peak_gb = torch.cuda.max_memory_allocated(0) / (1024**3)
                    logger.info(f"  Peak VRAM during parallel test: {peak_gb:.2f} / {total_gb:.2f} GB")
            else:
                logger.warning("  PARALLEL MODE: Some inference failed, falling back to serial")

        except torch.cuda.OutOfMemoryError:
            logger.warning("  PARALLEL MODE: NOT POSSIBLE (OOM)")
            logger.info("  Will use SERIAL mode (one model at a time)")
        except Exception as e:
            logger.warning(f"  PARALLEL MODE: NOT POSSIBLE ({e})")
            logger.info("  Will use SERIAL mode (one model at a time)")

        self.unload_all_models()

        return all_passed, parallel_possible


class PreprocessingPipeline:
    """Main preprocessing pipeline."""

    def __init__(self, config: PreprocessingConfig, parallel_mode: bool = False):
        self.config = config
        self.output_manager = OutputManager(config)
        self.model_runner = ModelRunner(config)
        self.visualizer = Visualizer() if config.visualize else None
        self.parallel_mode = parallel_mode

        # Progress tracking
        self.progress_file = config.output_dir / f".progress_{config.dataset_name}.json"
        if self.progress_file.exists():
            self.progress = ProgressTracker.load(self.progress_file)
            logger.info(f"Loaded existing progress from {self.progress_file}")
        else:
            self.progress = ProgressTracker(dataset_name=config.dataset_name)

        self.progress.parallel_mode = parallel_mode

    def _resize_frame(self, frame: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """Resize frame. Square resize (image_size) takes precedence over short-side resize."""
        if frame is None:
            return frame
        if self.config.image_size is not None:
            # Square resize — matches AnyCam training resolution
            sz = self.config.image_size
            return cv2.resize(frame, (sz, sz), interpolation=cv2.INTER_AREA)
        if self.config.resize_short_side is None:
            return frame
        h, w = frame.shape[:2]
        short_side = min(h, w)
        if short_side <= self.config.resize_short_side:
            return frame
        scale = self.config.resize_short_side / short_side
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

    def _discover_videos(self) -> List[Path]:
        """Find all video files in the dataset directory."""
        videos = []
        for ext in self.config.video_extensions:
            videos.extend(self.config.dataset_path.rglob(f"*{ext}"))
        videos = sorted(videos)
        logger.info(f"Found {len(videos)} videos in {self.config.dataset_path}")
        return videos

    def _save_progress(self):
        """Save current progress."""
        self.progress.save(self.progress_file)

    def run_test_mode(self) -> Tuple[bool, bool]:
        """
        Run quick test on first video's first frames.
        Tests each model individually, then tests if parallel mode is possible.

        Returns:
            (all_passed, parallel_possible)
        """
        logger.info("=" * 60)
        logger.info("RUNNING TEST MODE")
        logger.info("=" * 60)

        videos = self._discover_videos()
        if not videos:
            logger.error("No videos found in dataset path!")
            return False, False

        test_video = videos[0]
        logger.info(f"Using test video: {test_video}")

        with VideoProcessor(test_video) as vp:
            if vp.frame_count < 2:
                logger.error("Test video has less than 2 frames!")
                return False, False

            frame1 = vp.get_frame(0)
            frame2 = vp.get_frame(1)

            if frame1 is None or frame2 is None:
                logger.error("Failed to read test frames!")
                return False, False

            logger.info(f"Test frames loaded: {frame1.shape}")

        all_passed, parallel_possible = self.model_runner.test_all_models(frame1, frame2)

        logger.info("\n" + "=" * 60)
        if all_passed:
            logger.info("ALL MODEL TESTS PASSED")
            if parallel_possible:
                logger.info("EXECUTION MODE: PARALLEL (all models fit in VRAM)")
            else:
                logger.info("EXECUTION MODE: SERIAL (one model at a time)")
        else:
            logger.error("SOME TESTS FAILED - Check model imports and configurations")
        logger.info("=" * 60)

        return all_passed, parallel_possible

    def _process_video_parallel(self, video_path: Path):
        """
        Process a single video with all models loaded simultaneously.

        Three threads run concurrently, each saving to its own temp directory
        (no locks needed). After all threads finish, a merge pass combines
        temp files into final .npz files. This avoids the slow read-modify-write
        pattern on NAS and eliminates lock contention.
        """
        video_name = self.output_manager._sanitize_video_name(video_path)

        # Get frame count
        with VideoProcessor(video_path) as vp:
            total_frames = vp.frame_count

        if total_frames < 2:
            logger.warning(f"Video has less than 2 frames, skipping: {video_path}")
            return

        logger.info(f"  Processing {total_frames} frames (PARALLEL mode)...")

        video_dir = self.output_manager.get_video_dir(video_path)
        middle_frames = list(range(1, total_frames - 1))

        # Create temp directories for each model (no lock needed)
        tmp_flow = video_dir / '_tmp_flow'
        tmp_depth = video_dir / '_tmp_depth'
        tmp_calib = video_dir / '_tmp_calib'
        tmp_flow.mkdir(parents=True, exist_ok=True)
        tmp_depth.mkdir(parents=True, exist_ok=True)
        tmp_calib.mkdir(parents=True, exist_ok=True)

        # Ensure all models are loaded before spawning threads
        self.model_runner.load_all_models()

        def flow_worker():
            """Process all flow pairs, save each to temp dir."""
            with VideoProcessor(video_path) as vp_flow:
                for i in tqdm(range(total_frames - 1), desc="  Flow", leave=False):
                    frame_i = vp_flow.get_frame(i)
                    frame_i_plus_1 = vp_flow.get_frame(i + 1)

                    if frame_i is None or frame_i_plus_1 is None:
                        logger.warning(f"    Could not read frames {i} or {i+1}")
                        continue

                    frame_i = self._resize_frame(frame_i)
                    frame_i_plus_1 = self._resize_frame(frame_i_plus_1)

                    try:
                        result = self.model_runner.run_flow_pair(frame_i, frame_i_plus_1)
                        if result is not None:
                            # Save forward flow for frame i
                            np.savez(
                                tmp_flow / f"{i:06d}_fwd.npz",
                                forward_flow=result['flow_fwd'].astype(np.float16),
                                forward_occ=result['occ_fwd'].astype(np.float16),
                            )
                            # Save backward flow for frame i+1
                            np.savez(
                                tmp_flow / f"{i+1:06d}_bwd.npz",
                                backward_flow=result['flow_bwd'].astype(np.float16),
                                backward_occ=result['occ_bwd'].astype(np.float16),
                            )
                    except Exception as e:
                        logger.warning(f"    Flow error on pair ({i}, {i+1}): {e}")

        def depth_worker():
            """Process all middle frames for depth, save each to temp dir."""
            with VideoProcessor(video_path) as vp_depth:
                for idx in tqdm(middle_frames, desc="  Depth", leave=False):
                    frame = vp_depth.get_frame(idx)
                    if frame is None:
                        continue

                    frame = self._resize_frame(frame)
                    try:
                        result = self.model_runner.run_depth_single(frame)
                        if result is not None:
                            np.savez(
                                tmp_depth / f"{idx:06d}.npz",
                                depth=result.astype(np.float16),
                            )
                    except Exception as e:
                        logger.warning(f"    Depth error on frame {idx}: {e}")

        def calib_worker():
            """Process all middle frames for calibration, save each to temp dir."""
            with VideoProcessor(video_path) as vp_calib:
                for idx in tqdm(middle_frames, desc="  Calib", leave=False):
                    frame = vp_calib.get_frame(idx)
                    if frame is None:
                        continue

                    frame = self._resize_frame(frame)
                    try:
                        result = self.model_runner.run_calib_single(frame)
                        if result is not None:
                            np.savez(
                                tmp_calib / f"{idx:06d}.npz",
                                calib=result.astype(np.float32),
                            )
                    except Exception as e:
                        logger.warning(f"    Calib error on frame {idx}: {e}")

        # Run all three workers concurrently
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [
                executor.submit(flow_worker),
                executor.submit(depth_worker),
                executor.submit(calib_worker),
            ]
            # Wait for all to complete, propagate exceptions
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"  Worker error: {e}")

        # Merge temp files into final .npz files
        self._merge_temp_files(video_dir, total_frames, tmp_flow, tmp_depth, tmp_calib)

        # Save .jpg frames and visualizations
        self._save_jpg_frames(video_path, video_name, total_frames)

    def _process_video_serial(self, video_path: Path):
        """
        Process a single video with one model at a time (serial fallback).

        Loads each model, processes all relevant data, unloads, then moves
        to the next model. Minimizes peak VRAM usage.

        Each model pass saves results to its own temp directory per-frame.
        After all passes, a merge step combines temp files into final .npz.
        """
        video_name = self.output_manager._sanitize_video_name(video_path)

        with VideoProcessor(video_path) as vp:
            total_frames = vp.frame_count

            if total_frames < 2:
                logger.warning(f"Video has less than 2 frames, skipping: {video_path}")
                return

            logger.info(f"  Processing {total_frames} frames (SERIAL mode)...")

            video_dir = self.output_manager.get_video_dir(video_path)
            middle_frames = list(range(1, total_frames - 1))

            # Create temp directories for each model
            tmp_flow = video_dir / '_tmp_flow'
            tmp_depth = video_dir / '_tmp_depth'
            tmp_calib = video_dir / '_tmp_calib'
            tmp_flow.mkdir(parents=True, exist_ok=True)
            tmp_depth.mkdir(parents=True, exist_ok=True)
            tmp_calib.mkdir(parents=True, exist_ok=True)

            # Step 1: Flow (all consecutive pairs)
            logger.info("  [FLOW] Computing optical flow for all pairs...")
            for i in tqdm(range(total_frames - 1), desc="  Flow pairs", leave=False):
                frame_i = vp.get_frame(i)
                frame_i_plus_1 = vp.get_frame(i + 1)

                if frame_i is None or frame_i_plus_1 is None:
                    logger.warning(f"    Could not read frames {i} or {i+1}")
                    continue

                frame_i = self._resize_frame(frame_i)
                frame_i_plus_1 = self._resize_frame(frame_i_plus_1)

                try:
                    result = self.model_runner.run_flow_pair(frame_i, frame_i_plus_1)
                    if result is not None:
                        np.savez(
                            tmp_flow / f"{i:06d}_fwd.npz",
                            forward_flow=result['flow_fwd'].astype(np.float16),
                            forward_occ=result['occ_fwd'].astype(np.float16),
                        )
                        np.savez(
                            tmp_flow / f"{i+1:06d}_bwd.npz",
                            backward_flow=result['flow_bwd'].astype(np.float16),
                            backward_occ=result['occ_bwd'].astype(np.float16),
                        )
                except Exception as e:
                    logger.warning(f"    Error on pair ({i}, {i+1}): {e}")

            self.model_runner.unload_all_models()

            # Step 2: Depth (middle frames only)
            logger.info("  [DEPTH] Computing depth for middle frames...")
            for idx in tqdm(middle_frames, desc="  Depth", leave=False):
                frame = vp.get_frame(idx)
                if frame is None:
                    continue

                frame = self._resize_frame(frame)
                try:
                    result = self.model_runner.run_depth_single(frame)
                    if result is not None:
                        np.savez(
                            tmp_depth / f"{idx:06d}.npz",
                            depth=result.astype(np.float16),
                        )
                except Exception as e:
                    logger.warning(f"    Depth error on frame {idx}: {e}")

            self.model_runner.unload_all_models()

            # Step 3: Calibration (middle frames only)
            logger.info("  [CALIB] Computing calibration for middle frames...")
            for idx in tqdm(middle_frames, desc="  Calib", leave=False):
                frame = vp.get_frame(idx)
                if frame is None:
                    continue

                frame = self._resize_frame(frame)
                try:
                    result = self.model_runner.run_calib_single(frame)
                    if result is not None:
                        np.savez(
                            tmp_calib / f"{idx:06d}.npz",
                            calib=result.astype(np.float32),
                        )
                except Exception as e:
                    logger.warning(f"    Calib error on frame {idx}: {e}")

            self.model_runner.unload_all_models()

        # Merge temp files into final .npz files
        self._merge_temp_files(video_dir, total_frames, tmp_flow, tmp_depth, tmp_calib)

        # Final pass: save .jpg frames
        self._save_jpg_frames(video_path, video_name, total_frames)

    def _merge_temp_files(self, video_dir: Path, total_frames: int,
                          tmp_flow: Path, tmp_depth: Path, tmp_calib: Path):
        """
        Merge per-model temp files into final .npz files and clean up.

        Each model thread saved its outputs into separate temp directories.
        This method combines them into one .npz per frame, then removes
        the temp directories.
        """
        logger.info("  [MERGE] Combining model outputs into final .npz files...")

        for frame_idx in tqdm(range(total_frames), desc="  Merging", leave=False):
            merged = {}

            # Collect flow data (forward + backward in separate files)
            fwd_path = tmp_flow / f"{frame_idx:06d}_fwd.npz"
            bwd_path = tmp_flow / f"{frame_idx:06d}_bwd.npz"
            if fwd_path.exists():
                with np.load(fwd_path) as data:
                    for key in data.files:
                        merged[key] = data[key]
            if bwd_path.exists():
                with np.load(bwd_path) as data:
                    for key in data.files:
                        merged[key] = data[key]

            # Collect depth data
            depth_path = tmp_depth / f"{frame_idx:06d}.npz"
            if depth_path.exists():
                with np.load(depth_path) as data:
                    for key in data.files:
                        merged[key] = data[key]

            # Collect calib data
            calib_path = tmp_calib / f"{frame_idx:06d}.npz"
            if calib_path.exists():
                with np.load(calib_path) as data:
                    for key in data.files:
                        merged[key] = data[key]

            # Save merged result (skip frames with no data at all)
            if merged:
                npz_path = video_dir / f"{frame_idx:06d}{self.config.data_extension}"
                np.savez(npz_path, **merged)

        # Clean up temp directories
        for tmp_dir in [tmp_flow, tmp_depth, tmp_calib]:
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir)

        logger.info("  [MERGE] Done. Temp files cleaned up.")

    def _save_jpg_frames(self, video_path: Path, video_name: str, total_frames: int):
        """Save .jpg frames for all frames that don't already have them."""
        logger.info("  [JPG] Saving raw frames...")
        video_dir = self.output_manager.get_video_dir(video_path)

        with VideoProcessor(video_path) as vp_save:
            for frame_idx in tqdm(range(total_frames), desc="  Saving JPGs", leave=False):
                jpg_path = video_dir / f"{frame_idx:06d}.jpg"
                if not jpg_path.exists():
                    raw_frame = vp_save.get_frame(frame_idx)
                    if raw_frame is not None:
                        raw_frame = self._resize_frame(raw_frame)
                        cv2.imwrite(str(jpg_path), raw_frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

                self.progress.mark_frame_done(video_name, frame_idx)

        # Generate visualizations if enabled
        if self.config.visualize:
            self._save_visualizations_from_disk(video_path, video_name, total_frames)

        self._save_progress()

    def _save_visualizations_from_disk(self, video_path: Path, video_name: str, total_frames: int):
        """Generate visualization images by reading already-saved .npz files from disk."""
        video_dir = self.output_manager.get_video_dir(video_path)
        vis_frames_saved = 0

        vis_indices = [1, total_frames // 2, total_frames - 2]

        for frame_idx in vis_indices:
            if vis_frames_saved >= self.config.vis_samples_per_video:
                break

            npz_path = video_dir / f"{frame_idx:06d}{self.config.data_extension}"
            if not npz_path.exists():
                continue

            with VideoProcessor(video_path) as vp_vis:
                frame = vp_vis.get_frame(frame_idx)
            if frame is None:
                continue
            frame = self._resize_frame(frame)

            try:
                with np.load(npz_path) as data:
                    fwd_flow = data.get('forward_flow')
                    bwd_flow = data.get('backward_flow')
                    depth = data.get('depth')
                    calib = data.get('calib')

                vis = Visualizer.create_visualization(
                    frame,
                    forward_flow=fwd_flow,
                    backward_flow=bwd_flow,
                    depth=depth,
                    calib=calib,
                    frame_idx=frame_idx
                )
                vis_path = self.output_manager.get_vis_path(video_path, frame_idx)
                cv2.imwrite(str(vis_path), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
                vis_frames_saved += 1
            except Exception as e:
                logger.warning(f"  Visualization error for frame {frame_idx}: {e}")

    def run_full_preprocessing(self):
        """Run full preprocessing pipeline."""
        mode_str = "PARALLEL" if self.parallel_mode else "SERIAL"
        logger.info("=" * 60)
        logger.info("STARTING FULL PREPROCESSING")
        logger.info(f"Dataset: {self.config.dataset_name}")
        logger.info(f"Path: {self.config.dataset_path}")
        logger.info(f"Output: {self.config.output_dir}")
        logger.info(f"Mode: {mode_str}")
        if self.config.image_size:
            logger.info(f"Resize: square {self.config.image_size}x{self.config.image_size}px")
        elif self.config.resize_short_side:
            logger.info(f"Resize: short side = {self.config.resize_short_side}px")
        else:
            logger.info("Resize: DISABLED (original resolution)")
        if self.config.visualize:
            logger.info("Visualization: ENABLED")
        logger.info("=" * 60)

        videos = self._discover_videos()
        self.progress.total_videos = len(videos)

        for video_idx, video_path in enumerate(videos):
            logger.info(f"\n[{video_idx + 1}/{len(videos)}] {video_path.name}")

            try:
                if self.parallel_mode:
                    self._process_video_parallel(video_path)
                else:
                    self._process_video_serial(video_path)
            except torch.cuda.OutOfMemoryError:
                logger.warning(f"OOM during video {video_path.name}")
                if self.parallel_mode:
                    logger.warning("Falling back to SERIAL mode for remaining videos")
                    self.parallel_mode = False
                    self.progress.parallel_mode = False
                    self.model_runner.unload_all_models()
                    # Retry this video in serial mode
                    try:
                        self._process_video_serial(video_path)
                    except Exception as e:
                        logger.error(f"Serial fallback also failed: {e}")
                        traceback.print_exc()
                        continue
                else:
                    logger.error("OOM in serial mode - skipping video")
                    self.model_runner.unload_all_models()
                    continue
            except Exception as e:
                logger.error(f"Error processing video {video_path}: {e}")
                traceback.print_exc()
                self.model_runner.unload_all_models()
                continue

        self._save_progress()
        logger.info("\n" + "=" * 60)
        logger.info("PREPROCESSING COMPLETE")
        logger.info("=" * 60)

    def print_summary(self):
        """Print preprocessing summary."""
        mode_str = "PARALLEL" if self.parallel_mode else "SERIAL"
        logger.info("\nPREPROCESSING SUMMARY")
        logger.info("-" * 40)
        logger.info(f"Dataset: {self.config.dataset_name}")
        logger.info(f"Execution mode: {mode_str}")
        logger.info(f"Total videos: {self.progress.total_videos}")

        total_frames = sum(len(frames) for frames in self.progress.completed_frames.values())
        logger.info(f"Total frames processed: {total_frames}")

        if self.progress.failed_frames:
            total_failed = sum(len(frames) for frames in self.progress.failed_frames.values())
            logger.warning(f"Failed frames: {total_failed}")


# =============================================================================
# DATA LOADER FOR TRAINING
# =============================================================================

class PreprocessedDataLoader:
    """
    Loader for preprocessed data during training.

    Usage:
        loader = PreprocessedDataLoader(
            preprocessed_dir="/path/to/preprocessed",
            dataset_name="RealEstate10K"
        )

        # Load frame data
        data = loader.load_frame("video_name", frame_idx=5)
        # data contains: forward_flow, backward_flow, depth, calib (as available)

        # List available data
        videos = loader.list_videos()
        frames = loader.list_frames("video_name")
    """

    def __init__(self, preprocessed_dir: Union[str, Path], dataset_name: str):
        self.preprocessed_dir = Path(preprocessed_dir)
        self.dataset_name = dataset_name
        self.data_dir = self.preprocessed_dir / dataset_name

        if not self.data_dir.exists():
            raise ValueError(f"Preprocessed data directory not found: {self.data_dir}")

        # Index available files
        self._index = self._build_index()

    def _build_index(self) -> Dict[str, List[int]]:
        """Build index of available preprocessed files."""
        index = {}

        for video_dir in self.data_dir.iterdir():
            if not video_dir.is_dir() or video_dir.name.startswith("_"):
                continue

            video_name = video_dir.name
            frames = []

            for f in video_dir.glob("*.npz"):
                try:
                    frame_idx = int(f.stem)
                    frames.append(frame_idx)
                except ValueError:
                    continue

            if frames:
                index[video_name] = sorted(frames)

        return index

    def list_videos(self) -> List[str]:
        """List all video names with preprocessed data."""
        return sorted(self._index.keys())

    def list_frames(self, video_name: str) -> List[int]:
        """List available frame indices for a video."""
        return self._index.get(video_name, [])

    def _get_frame_path(self, video_name: str, frame_idx: int) -> Path:
        return self.data_dir / video_name / f"{frame_idx:06d}.npz"

    def load_frame(self, video_name: str, frame_idx: int) -> Optional[Dict[str, np.ndarray]]:
        """
        Load all data for a frame.

        Returns dict with (as available):
            - 'forward_flow': [2, H, W]
            - 'backward_flow': [2, H, W]
            - 'forward_occ': [1, H, W]
            - 'backward_occ': [1, H, W]
            - 'depth': [1, H, W]
            - 'calib': [4] (fx, fy, cx, cy)
        """
        path = self._get_frame_path(video_name, frame_idx)
        if not path.exists():
            return None

        data = np.load(path)
        result = {}

        for key in ['forward_flow', 'backward_flow', 'forward_occ', 'backward_occ', 'depth', 'calib']:
            if key in data:
                result[key] = data[key]

        return result

    def has_frame(self, video_name: str, frame_idx: int) -> bool:
        """Check if frame data exists."""
        return frame_idx in self._index.get(video_name, [])

    def get_video_stats(self) -> Dict[str, int]:
        """Get frame count per video."""
        return {video: len(frames) for video, frames in self._index.items()}

    def get_total_stats(self) -> Dict[str, int]:
        """Get total statistics."""
        total_videos = len(self._index)
        total_frames = sum(len(frames) for frames in self._index.values())
        return {"videos": total_videos, "frames": total_frames}


# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess video datasets for FAT and DA3 training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test mode (validates models, determines parallel vs serial)
  python preprocess_dataset.py --dataset_path /data/videos --output_dir /data/preprocessed --test

  # Full preprocessing for a dataset
  python preprocess_dataset.py --dataset_path /data/RealEstate10K --output_dir /data/preprocessed --dataset_name RealEstate10K

  # With visualization for debugging
  python preprocess_dataset.py --dataset_path /data/videos --output_dir /data/preprocessed --dataset_name Test --visualize

  # Resume interrupted preprocessing
  python preprocess_dataset.py --dataset_path /data/RealEstate10K --output_dir /data/preprocessed --dataset_name RealEstate10K --resume

  # Force serial mode (skip parallel attempt)
  python preprocess_dataset.py --dataset_path /data/videos --output_dir /data/preprocessed --serial

  # Show statistics about existing preprocessed data
  python preprocess_dataset.py --output_dir /data/preprocessed --dataset_name RealEstate10K --stats_only

Output structure:
  /data/preprocessed/
    RealEstate10K/
      video_01/
        000000.npz  # First frame: forward_flow only
        000001.npz  # Middle: forward_flow, backward_flow, depth, calib
        ...
        NNNNNN.npz  # Last frame: backward_flow only
      video_02/
        ...
      _visualizations/  # Only if --visualize is used
        video_01_000001_vis.png
        ...

Execution modes:
  PARALLEL: All models loaded simultaneously, each processes its data in
            its own thread. Faster but requires enough VRAM for all models.
  SERIAL:   One model at a time. Loads model, processes all data, unloads,
            then loads the next. Lower VRAM usage but slower.
  The --test flag determines which mode is possible on your hardware.
        """
    )

    parser.add_argument(
        "--dataset_path",
        type=Path,
        help="Path to dataset folder containing videos (required unless --stats_only)"
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Path to output directory for preprocessed results"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="unnamed_dataset",
        help="Name of the dataset (used for folder naming and progress tracking)"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run test mode: validate models and determine parallel vs serial execution"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from previous progress (default: start fresh)"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate visualization images for debugging"
    )
    parser.add_argument(
        "--serial",
        action="store_true",
        help="Force serial mode (one model at a time, skip parallel attempt)"
    )
    parser.add_argument(
        "--unidepth_backbone",
        type=str,
        default="vits14",
        choices=["vits14", "vitl14"],
        help="UniDepth backbone: 'vits14' (faster, less VRAM) or 'vitl14' (better quality)"
    )
    parser.add_argument(
        "--resize_short_side",
        type=int,
        default=None,
        help="Resize frames so short side equals this value (e.g. 518). "
             "Maintains aspect ratio. Reduces storage ~7x for 4K input. "
             "Default: no resize (store at original resolution)"
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=None,
        help="Resize all frames to this square size (e.g. 336). "
             "Overrides --resize_short_side. Ensures calibration intrinsics "
             "match the training resolution directly."
    )
    parser.add_argument(
        "--stats_only",
        action="store_true",
        help="Only show statistics about existing preprocessed data, don't process"
    )

    args = parser.parse_args()

    # Handle stats_only mode
    if args.stats_only:
        try:
            loader = PreprocessedDataLoader(
                preprocessed_dir=args.output_dir,
                dataset_name=args.dataset_name
            )
            stats = loader.get_video_stats()
            total = loader.get_total_stats()

            logger.info(f"\nPreprocessed data statistics for {args.dataset_name}:")
            logger.info("-" * 60)
            for video, num_frames in sorted(stats.items()):
                logger.info(f"  {video}: {num_frames} frames")
            logger.info("-" * 60)
            logger.info(f"  TOTAL: {total['videos']} videos, {total['frames']} frames")
            sys.exit(0)
        except Exception as e:
            logger.error(f"Failed to load stats: {e}")
            sys.exit(1)

    # Validate paths
    if not args.dataset_path:
        logger.error("--dataset_path is required unless using --stats_only")
        sys.exit(1)

    if not args.dataset_path.exists():
        logger.error(f"Dataset path does not exist: {args.dataset_path}")
        sys.exit(1)

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Create configuration
    config = PreprocessingConfig(
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        dataset_name=args.dataset_name,
        visualize=args.visualize,
        resize_short_side=args.resize_short_side,
        image_size=args.image_size,
    )

    # Handle resume logic
    progress_file = args.output_dir / f".progress_{args.dataset_name}.json"
    if progress_file.exists() and not args.resume and not args.test:
        logger.warning(f"Found existing progress file: {progress_file}")
        logger.warning("Use --resume to continue, or delete the file to start fresh")
        response = input("Continue from existing progress? [y/N]: ")
        if response.lower() != 'y':
            logger.info("Starting fresh - removing progress file")
            progress_file.unlink()

    # Add file logging
    log_file = args.output_dir / f"preprocessing_{args.dataset_name}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)

    if args.test:
        # Test mode: determine parallel vs serial
        pipeline = PreprocessingPipeline(config, parallel_mode=False)
        pipeline.model_runner._unidepth_backbone = args.unidepth_backbone
        all_passed, parallel_possible = pipeline.run_test_mode()
        sys.exit(0 if all_passed else 1)
    else:
        # Determine execution mode
        if args.serial:
            parallel_mode = False
            logger.info("Forced SERIAL mode via --serial flag")
        else:
            # Auto-detect: try loading all models
            logger.info("Determining execution mode...")
            test_runner = ModelRunner(config)
            test_runner._unidepth_backbone = args.unidepth_backbone
            try:
                test_runner.load_all_models()
                parallel_mode = True
                logger.info("All models fit in VRAM -> using PARALLEL mode")
                test_runner.unload_all_models()
            except torch.cuda.OutOfMemoryError:
                parallel_mode = False
                logger.info("Models don't all fit in VRAM -> using SERIAL mode")
                test_runner.unload_all_models()
            except Exception as e:
                parallel_mode = False
                logger.info(f"Could not load all models ({e}) -> using SERIAL mode")
                test_runner.unload_all_models()
            del test_runner

        # Create and run pipeline
        pipeline = PreprocessingPipeline(config, parallel_mode=parallel_mode)
        pipeline.model_runner._unidepth_backbone = args.unidepth_backbone

        try:
            pipeline.run_full_preprocessing()
            pipeline.print_summary()
        except KeyboardInterrupt:
            logger.info("\nInterrupted by user - saving progress...")
            pipeline._save_progress()
            pipeline.print_summary()
            sys.exit(130)
        except Exception as e:
            logger.error(f"Fatal error: {e}")
            traceback.print_exc()
            pipeline._save_progress()
            sys.exit(1)


if __name__ == "__main__":
    main()
