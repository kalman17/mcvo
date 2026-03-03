"""
Dataset class for loading preprocessed multi-frame sequences from .npz + .jpg files.

Expected directory structure:
    data_dir/
      dataset_name/
        video_name/
          000000.jpg      # Raw frame
          000000.npz      # forward_flow[2,H,W], depth[1,H,W], calib[4], etc.
          000001.jpg
          000001.npz
          ...

Each .npz contains (as available):
    - forward_flow: [2, H, W] float16  (pixel displacement to next frame)
    - backward_flow: [2, H, W] float16  (pixel displacement to previous frame)
    - forward_occ: [1, H, W] float16  (forward occlusion mask)
    - backward_occ: [1, H, W] float16  (backward occlusion mask)
    - depth: [1, H, W] float16  (inverse depth: 1.0 / (metric_depth * 0.1))
    - calib: [4] float32  (fx, fy, cx, cy)
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class PreprocessedMultiFrameDataset(Dataset):
    """
    PyTorch Dataset that loads multi-frame sequences from preprocessed data.

    Each sample is a sequence of (max_ahead + 1) consecutive frames from a single video.
    Only sequences where all required data is present are included.

    Args:
        data_dir: Root directory containing dataset subdirectories.
        datasets: List of dataset names to include. If None, all subdirectories are used.
        max_ahead: Maximum lookahead for flow composition. Sequence length = max_ahead + 1.
        image_size: Target image size (square) for resizing frames.
        phase: Training phase ('A', 'B1', 'B2', 'C'). Controls which data fields are required.
    """

    # Data fields required per phase.
    # Phase A needs depth+flow+calib (pose training). No FAT.
    # Phase B1 needs calib only (FAT pre-training on reprojection loss).
    # Phase B2/C need depth+flow+calib (combined training).
    PHASE_REQUIREMENTS = {
        'A':  {'depth', 'forward_flow', 'backward_flow', 'forward_occ', 'backward_occ', 'calib'},
        'B1': {'calib'},
        'B2': {'depth', 'forward_flow', 'backward_flow', 'forward_occ', 'backward_occ', 'calib'},
        'C':  {'depth', 'forward_flow', 'backward_flow', 'forward_occ', 'backward_occ', 'calib'},
        'Da': {'depth', 'forward_flow', 'backward_flow', 'forward_occ', 'backward_occ', 'calib'},
        'Db': {'depth', 'forward_flow', 'backward_flow', 'forward_occ', 'backward_occ', 'calib'},
    }

    def __init__(
        self,
        data_dir: str,
        datasets: Optional[List[str]] = None,
        max_ahead: int = 3,
        image_size: int = 336,
        phase: str = 'A',
    ):
        self.data_dir = Path(data_dir)
        self.max_ahead = max_ahead
        self.seq_len = max_ahead + 1
        self.image_size = image_size
        self.phase = phase

        if phase not in self.PHASE_REQUIREMENTS:
            raise ValueError(f"Unknown phase '{phase}'. Must be one of {list(self.PHASE_REQUIREMENTS.keys())}")

        self.required_fields = self.PHASE_REQUIREMENTS[phase]

        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")

        # Discover datasets
        if datasets is not None:
            self.dataset_names = datasets
        else:
            self.dataset_names = sorted([
                d.name for d in self.data_dir.iterdir()
                if d.is_dir() and not d.name.startswith('_') and not d.name.startswith('.')
            ])

        if not self.dataset_names:
            raise ValueError(f"No datasets found in {self.data_dir}")

        # Build index: list of (dataset_name, video_name, start_frame_idx)
        self.samples: List[Tuple[str, str, int]] = []
        self._frame_index: Dict[str, Dict[str, List[int]]] = {}
        self._build_index()

        logger.info(
            f"PreprocessedMultiFrameDataset: {len(self.samples)} samples from "
            f"{len(self.dataset_names)} datasets, seq_len={self.seq_len}, phase={phase}"
        )

    def _build_index(self):
        """Scan data directories and build sample index."""
        total_videos = 0
        total_frames = 0

        for ds_name in self.dataset_names:
            ds_dir = self.data_dir / ds_name
            if not ds_dir.exists():
                logger.warning(f"Dataset directory not found: {ds_dir}, skipping")
                continue

            self._frame_index[ds_name] = {}

            for video_dir in sorted(ds_dir.iterdir()):
                if not video_dir.is_dir() or video_dir.name.startswith('_'):
                    continue

                video_name = video_dir.name

                # Find all frame indices that have .npz files
                frame_indices = sorted([
                    int(f.stem) for f in video_dir.glob("*.npz")
                    if f.stem.isdigit()
                ])

                if not frame_indices:
                    continue

                self._frame_index[ds_name][video_name] = frame_indices
                total_videos += 1
                total_frames += len(frame_indices)

                # Build valid sequences from this video (skip per-file validation for speed)
                self._add_video_sequences(ds_name, video_name, frame_indices, validate=False)

        logger.info(
            f"Indexed {total_videos} videos, {total_frames} frames, "
            f"{len(self.samples)} valid sequences"
        )

    def _add_video_sequences(self, ds_name: str, video_name: str, frame_indices: List[int],
                             validate: bool = True):
        """Find all valid consecutive sequences of length seq_len in a video."""
        if len(frame_indices) < self.seq_len:
            return

        # Check for consecutive frame indices and required data availability
        idx_set = set(frame_indices)

        for i in range(len(frame_indices) - self.seq_len + 1):
            start = frame_indices[i]

            # Check that we have seq_len consecutive frames
            seq_frames = [start + j for j in range(self.seq_len)]
            if not all(f in idx_set for f in seq_frames):
                continue

            # Optionally validate that required data fields are present (slow on NFS)
            if validate:
                if not self._validate_sequence(ds_name, video_name, seq_frames):
                    continue

            self.samples.append((ds_name, video_name, start))

    def _validate_sequence(self, ds_name: str, video_name: str, seq_frames: List[int]) -> bool:
        """Check that all frames in a sequence have the required data fields."""
        for pos, frame_idx in enumerate(seq_frames):
            npz_path = self.data_dir / ds_name / video_name / f"{frame_idx:06d}.npz"
            if not npz_path.exists():
                return False

            # Determine which fields this frame position needs
            needed = set()

            if 'calib' in self.required_fields:
                needed.add('calib')
            if 'depth' in self.required_fields:
                needed.add('depth')

            # Flow requirements depend on position:
            # - All except last frame need forward_flow/forward_occ
            # - All except first frame need backward_flow/backward_occ
            if pos < len(seq_frames) - 1:
                if 'forward_flow' in self.required_fields:
                    needed.add('forward_flow')
                if 'forward_occ' in self.required_fields:
                    needed.add('forward_occ')
            if pos > 0:
                if 'backward_flow' in self.required_fields:
                    needed.add('backward_flow')
                if 'backward_occ' in self.required_fields:
                    needed.add('backward_occ')

            # Quick check using np.load with mmap for speed (only reads headers)
            try:
                with np.load(npz_path) as data:
                    available = set(data.files)
                    if not needed.issubset(available):
                        return False
            except Exception:
                return False

        return True

    def _load_image(self, ds_name: str, video_name: str, frame_idx: int) -> np.ndarray:
        """Load and resize an image frame. Returns [3, H, W] float32 in [0, 1]."""
        jpg_path = self.data_dir / ds_name / video_name / f"{frame_idx:06d}.jpg"
        png_path = self.data_dir / ds_name / video_name / f"{frame_idx:06d}.png"

        if jpg_path.exists():
            img = cv2.imread(str(jpg_path))
        elif png_path.exists():
            img = cv2.imread(str(png_path))
        else:
            raise FileNotFoundError(
                f"No image found for frame {frame_idx} in {ds_name}/{video_name}. "
                f"Checked: {jpg_path}, {png_path}"
            )

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        img = img.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)  # [H, W, 3] -> [3, H, W]
        return img

    def _load_npz(self, ds_name: str, video_name: str, frame_idx: int) -> Dict[str, np.ndarray]:
        """Load all available fields from a .npz file."""
        npz_path = self.data_dir / ds_name / video_name / f"{frame_idx:06d}.npz"
        data = np.load(npz_path)
        result = {}
        for key in ['forward_flow', 'backward_flow', 'forward_occ', 'backward_occ', 'depth', 'calib']:
            if key in data:
                result[key] = data[key].astype(np.float32)
        return result

    def _resize_spatial(self, arr: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
        """Resize a [C, H, W] array to [C, target_h, target_w] using bilinear interpolation."""
        c, h, w = arr.shape
        if h == target_h and w == target_w:
            return arr

        # Transpose to [H, W, C] for cv2
        arr_hwc = arr.transpose(1, 2, 0)
        resized = cv2.resize(arr_hwc, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        if resized.ndim == 2:
            resized = resized[:, :, np.newaxis]

        result = resized.transpose(2, 0, 1)  # [C, target_h, target_w]

        # Scale flow values proportionally to the resize
        if c == 2:  # likely flow
            result[0] *= target_w / w
            result[1] *= target_h / h

        return result

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Try loading this sample; on failure (missing fields), try up to 10 others
        for attempt in range(10):
            try:
                return self._load_sample((idx + attempt) % len(self.samples))
            except (KeyError, FileNotFoundError, ValueError):
                continue
        # Last resort: random index
        return self._load_sample(torch.randint(len(self.samples), (1,)).item())

    def _load_sample(self, idx: int) -> Dict[str, torch.Tensor]:
        ds_name, video_name, start_frame = self.samples[idx]
        seq_frames = [start_frame + j for j in range(self.seq_len)]

        images = []
        depths = []
        flows_fwd = []
        flows_bwd = []
        occs_fwd = []
        occs_bwd = []
        calibs = []

        target_size = self.image_size

        for pos, frame_idx in enumerate(seq_frames):
            # Load image
            img = self._load_image(ds_name, video_name, frame_idx)
            images.append(img)

            # Load npz data
            npz_data = self._load_npz(ds_name, video_name, frame_idx)

            # Calibration — scale fx, fy, cx, cy to match resized image resolution
            if 'calib' in self.required_fields:
                if 'calib' not in npz_data:
                    raise KeyError(f"Missing calib in {ds_name}/{video_name}/{frame_idx:06d}")
            if 'calib' in npz_data:
                calib = npz_data['calib'].copy()
                # Infer native .npz resolution from a spatial field
                ref_field = npz_data.get(
                    'depth', npz_data.get('forward_flow', npz_data.get('backward_flow'))
                )
                if ref_field is not None:
                    npz_h, npz_w = ref_field.shape[1], ref_field.shape[2]
                    if npz_h != target_size or npz_w != target_size:
                        calib[0] *= target_size / npz_w  # fx
                        calib[1] *= target_size / npz_h  # fy
                        calib[2] *= target_size / npz_w  # cx
                        calib[3] *= target_size / npz_h  # cy
                calibs.append(calib)

            # Depth: resize to target size
            if 'depth' in self.required_fields:
                if 'depth' not in npz_data:
                    raise KeyError(f"Missing depth in {ds_name}/{video_name}/{frame_idx:06d}")
                depth = self._resize_spatial(npz_data['depth'], target_size, target_size)
                depths.append(depth)
            elif 'depth' in npz_data:
                depth = self._resize_spatial(npz_data['depth'], target_size, target_size)
                depths.append(depth)

            # Forward flow/occ (not needed for last frame)
            if pos < self.seq_len - 1 and 'forward_flow' in self.required_fields:
                if 'forward_flow' not in npz_data:
                    raise KeyError(f"Missing forward_flow in {ds_name}/{video_name}/{frame_idx:06d}")
            if pos < self.seq_len - 1 and 'forward_flow' in npz_data:
                flow = self._resize_spatial(npz_data['forward_flow'], target_size, target_size)
                flows_fwd.append(flow)

                if 'forward_occ' in npz_data:
                    occ = self._resize_spatial(npz_data['forward_occ'], target_size, target_size)
                    occs_fwd.append(occ)
                else:
                    occs_fwd.append(np.ones((1, target_size, target_size), dtype=np.float32))

            # Backward flow/occ (not needed for first frame)
            if pos > 0 and 'backward_flow' in self.required_fields:
                if 'backward_flow' not in npz_data:
                    raise KeyError(f"Missing backward_flow in {ds_name}/{video_name}/{frame_idx:06d}")
            if pos > 0 and 'backward_flow' in npz_data:
                flow = self._resize_spatial(npz_data['backward_flow'], target_size, target_size)
                flows_bwd.append(flow)

                if 'backward_occ' in npz_data:
                    occ = self._resize_spatial(npz_data['backward_occ'], target_size, target_size)
                    occs_bwd.append(occ)
                else:
                    occs_bwd.append(np.ones((1, target_size, target_size), dtype=np.float32))

        # Stack and convert to tensors
        # Only include optional (non-required) fields if ALL frames contributed,
        # otherwise different samples would have different tensor sizes and
        # collate_fn would crash when stacking.
        N = self.seq_len
        N_flow = N - 1  # flows are between consecutive frames

        result = {
            'images': torch.from_numpy(np.stack(images)),           # [N, 3, H, W]
            'video_name': f"{ds_name}/{video_name}",
            'frame_indices': torch.tensor(seq_frames, dtype=torch.long),  # [N]
        }

        if len(depths) == N:
            result['depths'] = torch.from_numpy(np.stack(depths))       # [N, 1, H, W]
        if len(flows_fwd) == N_flow:
            result['flows_fwd'] = torch.from_numpy(np.stack(flows_fwd))  # [N-1, 2, H, W]
        if len(flows_bwd) == N_flow:
            result['flows_bwd'] = torch.from_numpy(np.stack(flows_bwd))  # [N-1, 2, H, W]
        if len(occs_fwd) == N_flow:
            result['occs_fwd'] = torch.from_numpy(np.stack(occs_fwd))    # [N-1, 1, H, W]
        if len(occs_bwd) == N_flow:
            result['occs_bwd'] = torch.from_numpy(np.stack(occs_bwd))    # [N-1, 1, H, W]
        if len(calibs) == N:
            result['calibs'] = torch.from_numpy(np.stack(calibs))        # [N, 4]

        return result


def collate_fn(batch: List[Dict]) -> Dict:
    """
    Custom collate function for DataLoader.

    Stacks all tensor fields along a new batch dimension.
    String fields are collected into lists.
    Only includes keys present in ALL samples (avoids size mismatches
    from optional fields that may be missing in some samples).
    """
    result = {}
    # Only collate keys that every sample has
    keys = set(batch[0].keys())
    for sample in batch[1:]:
        keys &= set(sample.keys())

    for key in keys:
        values = [sample[key] for sample in batch]
        if isinstance(values[0], torch.Tensor):
            result[key] = torch.stack(values)
        elif isinstance(values[0], str):
            result[key] = values
        else:
            result[key] = values

    return result
