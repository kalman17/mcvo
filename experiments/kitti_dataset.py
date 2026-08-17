"""
KITTI Odometry dataset loader for benchmarking.

Loads consecutive frame sequences from KITTI Odometry color sequences 00-10
(which have ground truth poses). Returns images, intrinsics, and poses in
the same format as AnyCam's SintelDataset/TUMRGBDDataset.

Expected directory structure:
    kitti_odom_color/
        sequences/
            00/
                image_2/000000.png, 000001.png, ...
                calib.txt
                times.txt
            01/ ...
            ...
            10/  (last sequence with GT poses)
        poses/
            00.txt, 01.txt, ..., 10.txt
"""

import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from torch.utils.data import Dataset

from anycam.datasets.common import get_target_size_and_crop, process_img, process_proj


# Sequences 00-10 have ground truth poses
GT_SEQUENCES = [f"{i:02d}" for i in range(11)]


class KITTIOdometryDataset(Dataset):
    NAME = "KITTI"

    def __init__(
        self,
        data_path: str,
        image_size: Optional[int] = None,
        frame_count: int = 2,
        dilation: int = 1,
        sequences: Optional[list] = None,
    ):
        self.data_path = data_path
        self.image_size = image_size
        self.frame_count = frame_count
        self.dilation = dilation

        self._sequences = sequences or GT_SEQUENCES
        self._calibrations = {}  # seq_id -> 3x3 K matrix
        self._poses = {}         # seq_id -> list of 4x4 poses
        self._datapoints = []    # list of (seq_id, start_frame_idx)

        self._load_all_metadata()
        self._build_datapoints()

    def _load_calibration(self, seq_id: str) -> np.ndarray:
        """Load P2 projection matrix from calib.txt, extract 3x3 intrinsics."""
        calib_path = os.path.join(self.data_path, "sequences", seq_id, "calib.txt")
        with open(calib_path, "r") as f:
            for line in f:
                if line.startswith("P2:"):
                    values = [float(x) for x in line.strip().split()[1:]]
                    P2 = np.array(values, dtype=np.float64).reshape(3, 4)
                    # Extract 3x3 intrinsics from P2
                    K = P2[:3, :3].astype(np.float32)
                    return K
        raise ValueError(f"P2 not found in {calib_path}")

    def _load_poses(self, seq_id: str) -> list:
        """Load GT poses from poses/XX.txt. Each line is a 3x4 matrix (12 values)."""
        pose_path = os.path.join(self.data_path, "poses", f"{seq_id}.txt")
        poses = []
        with open(pose_path, "r") as f:
            for line in f:
                values = [float(x) for x in line.strip().split()]
                if len(values) != 12:
                    continue
                T = np.eye(4, dtype=np.float32)
                T[:3, :4] = np.array(values, dtype=np.float32).reshape(3, 4)
                poses.append(T)
        return poses

    def _load_all_metadata(self):
        """Load calibrations and poses for all sequences."""
        for seq_id in self._sequences:
            seq_dir = os.path.join(self.data_path, "sequences", seq_id)
            pose_file = os.path.join(self.data_path, "poses", f"{seq_id}.txt")

            if not os.path.isdir(seq_dir) or not os.path.isfile(pose_file):
                continue

            self._calibrations[seq_id] = self._load_calibration(seq_id)
            self._poses[seq_id] = self._load_poses(seq_id)

    def _build_datapoints(self):
        """Build index of valid (sequence, start_frame) pairs."""
        span = (self.frame_count - 1) * self.dilation
        for seq_id in sorted(self._calibrations.keys()):
            num_poses = len(self._poses[seq_id])
            # Count actual images
            img_dir = os.path.join(self.data_path, "sequences", seq_id, "image_2")
            num_images = len([f for f in os.listdir(img_dir) if f.endswith((".png", ".jpg"))])
            max_frames = min(num_poses, num_images)

            for start_idx in range(max_frames - span):
                self._datapoints.append((seq_id, start_idx))

    def __len__(self):
        return len(self._datapoints)

    def __getitem__(self, index):
        seq_id, start_idx = self._datapoints[index]

        # Frame indices
        ids = [start_idx + i * self.dilation for i in range(self.frame_count)]

        # Load images
        imgs_raw = []
        for fid in ids:
            img_path_png = os.path.join(
                self.data_path, "sequences", seq_id, "image_2", f"{fid:06d}.png"
            )
            img_path_jpg = os.path.join(
                self.data_path, "sequences", seq_id, "image_2", f"{fid:06d}.jpg"
            )
            img_path = img_path_png if os.path.exists(img_path_png) else img_path_jpg
            img = cv2.imread(img_path)
            if img is None:
                raise FileNotFoundError(f"Could not read {img_path}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            imgs_raw.append(img)

        original_size = imgs_raw[0].shape[:2]  # (H, W)
        target_size, crop = get_target_size_and_crop(self.image_size, original_size)

        # Process images: resize, crop, CHW, float32 [0, 1]
        # NOTE (2026-08-17): until this line the stack stayed uint8 0..255 — the docstring
        # promised [0,1] but nothing divided. AnyCam's own Sintel/TUM loaders divide by 255,
        # so every model got 255x-too-large KITTI input (and the DA3 adapter, which does
        # (imgs*255).astype(uint8), got inverted images from uint8 overflow). All KITTI
        # rows produced through this loader before this fix are re-measured as *_kfix runs.
        imgs = np.stack([process_img(img, target_size, crop) for img in imgs_raw]).astype(np.float32) / 255.0

        # Intrinsics (same K for all frames in a sequence)
        K = self._calibrations[seq_id]
        projs = np.stack([
            process_proj(K.copy(), original_size, target_size, crop)
            for _ in ids
        ])

        # Poses (absolute camera-to-world)
        poses = np.stack([self._poses[seq_id][fid] for fid in ids])

        return {
            "imgs": imgs,
            "projs": projs,
            "poses": poses,
            "ids": np.array(ids, dtype=np.int64),
            "data_id": index,
        }
