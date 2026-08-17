"""Wrap PreprocessedMultiFrameDataset with cached AnyCam pseudo-poses (E4)."""
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class PseudoPoseDataset(Dataset):
    def __init__(self, base, pose_dir):
        self.base = base
        self.pose_dir = Path(pose_dir)
        keep = []
        for i, (ds_name, video, start) in enumerate(base.samples):
            if (self.pose_dir / ds_name / video / f"POSES_{start:06d}.npz").exists():
                keep.append(i)
        self.keep = keep
        print(f"[pseudo-ds] {len(keep)}/{len(base.samples)} sequences have pseudo-poses",
              flush=True)

    def __len__(self):
        return len(self.keep)

    def __getitem__(self, idx):
        i = self.keep[idx]
        s = self.base[i]
        ds_name, video, start = self.base.samples[i]
        d = np.load(self.pose_dir / ds_name / video / f"POSES_{start:06d}.npz")
        s["pseudo_poses"] = torch.from_numpy(d["rel_poses"].astype(np.float32))
        return s
