#!/usr/bin/env python3
"""
Precompute vanilla AnyCam pose predictions and AnyCalib calibrations on
validation sequences. Results are cached in a .pt file for per-epoch
divergence monitoring during training.

Only loads the AnyCam pose predictor (~30M params). No UniMatch, UniDepth,
or AnyCalib inference models are loaded.

Usage:
    cd /tmp  # avoid anycalib namespace conflict
    PYTHONPATH=~/TUM/thesis/anycam:$PYTHONPATH python3 \
        ~/TUM/thesis/anycam/experiments/precompute_vanilla_baselines.py \
        --data_dir ~/TUM/thesis/test_data/preprocessed \
        --output_path ~/TUM/thesis/test_data/preprocessed/val_baselines.pt
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def select_val_sequences(
    data_dir: str,
    seq_len: int = 4,
) -> List[Dict]:
    """Select the first valid sequence of length seq_len from each dataset."""
    data_path = Path(data_dir)
    sequences = []

    required_fields = {'depth', 'forward_flow', 'backward_flow',
                       'forward_occ', 'backward_occ', 'calib'}

    for ds_dir in sorted(data_path.iterdir()):
        if not ds_dir.is_dir() or ds_dir.name.startswith('_'):
            continue

        ds_name = ds_dir.name
        found = False

        for video_dir in sorted(ds_dir.iterdir()):
            if not video_dir.is_dir() or video_dir.name.startswith('_'):
                continue

            # Find frame indices
            frame_indices = sorted([
                int(f.stem) for f in video_dir.glob("*.npz")
                if f.stem.isdigit()
            ])

            if len(frame_indices) < seq_len:
                continue

            # Find first valid consecutive sequence
            idx_set = set(frame_indices)
            for start in frame_indices:
                seq_frames = [start + j for j in range(seq_len)]
                if not all(f in idx_set for f in seq_frames):
                    continue

                # Validate all required fields exist
                valid = True
                for pos, fidx in enumerate(seq_frames):
                    npz_path = video_dir / f"{fidx:06d}.npz"
                    try:
                        with np.load(npz_path) as data:
                            available = set(data.files)
                            needed = {'calib', 'depth'}
                            if pos < seq_len - 1:
                                needed |= {'forward_flow', 'forward_occ'}
                            if pos > 0:
                                needed |= {'backward_flow', 'backward_occ'}
                            if not needed.issubset(available):
                                valid = False
                                break
                    except Exception:
                        valid = False
                        break

                if valid:
                    sequences.append({
                        "dataset_name": ds_name,
                        "video_name": video_dir.name,
                        "start_frame": start,
                    })
                    found = True
                    logger.info(f"  {ds_name}/{video_dir.name} frame {start} (seq_len={seq_len})")
                    break

            if found:
                break

        if not found:
            logger.warning(f"  No valid sequence found in dataset: {ds_name}")

    return sequences


def load_image(path: Path, image_size: int) -> np.ndarray:
    """Load and resize an image. Returns [3, H, W] float32 in [0, 1]."""
    img = cv2.imread(str(path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float32) / 255.0
    return img.transpose(2, 0, 1)


def load_npz_field(path: Path, key: str) -> Optional[np.ndarray]:
    """Load a single field from an npz file as float32."""
    with np.load(path) as data:
        if key in data:
            return data[key].astype(np.float32)
    return None


def resize_spatial(arr: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Resize [C, H, W] array to [C, target_h, target_w]."""
    c, h, w = arr.shape
    if h == target_h and w == target_w:
        return arr
    arr_hwc = arr.transpose(1, 2, 0)
    resized = cv2.resize(arr_hwc, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    if resized.ndim == 2:
        resized = resized[:, :, np.newaxis]
    result = resized.transpose(2, 0, 1)
    if c == 2:  # flow: scale proportionally
        result[0] *= target_w / w
        result[1] *= target_h / h
    return result


def load_sequence_data(
    data_dir: str,
    dataset_name: str,
    video_name: str,
    start_frame: int,
    seq_len: int,
    image_size: int,
) -> Dict[str, torch.Tensor]:
    """Load a sequence from preprocessed data, matching PreprocessedMultiFrameDataset format."""
    base = Path(data_dir) / dataset_name / video_name
    seq_frames = [start_frame + j for j in range(seq_len)]

    images, depths, flows_fwd, occs_fwd, calibs = [], [], [], [], []

    for pos, fidx in enumerate(seq_frames):
        # Image
        jpg_path = base / f"{fidx:06d}.jpg"
        png_path = base / f"{fidx:06d}.png"
        img_path = jpg_path if jpg_path.exists() else png_path
        images.append(load_image(img_path, image_size))

        npz_path = base / f"{fidx:06d}.npz"

        # Calib
        calib = load_npz_field(npz_path, 'calib')
        if calib is not None:
            calibs.append(calib)

        # Depth
        depth = load_npz_field(npz_path, 'depth')
        if depth is not None:
            depths.append(resize_spatial(depth, image_size, image_size))

        # Forward flow/occ (not last frame)
        if pos < seq_len - 1:
            fwd = load_npz_field(npz_path, 'forward_flow')
            occ = load_npz_field(npz_path, 'forward_occ')
            if fwd is not None:
                flows_fwd.append(resize_spatial(fwd, image_size, image_size))
            if occ is not None:
                occs_fwd.append(resize_spatial(occ, image_size, image_size))

    return {
        'images': torch.from_numpy(np.stack(images)).unsqueeze(0),       # [1, N, 3, H, W]
        'depths': torch.from_numpy(np.stack(depths)).unsqueeze(0),       # [1, N, 1, H, W]
        'flows_fwd': torch.from_numpy(np.stack(flows_fwd)).unsqueeze(0), # [1, N-1, 2, H, W]
        'occs_fwd': torch.from_numpy(np.stack(occs_fwd)).unsqueeze(0),   # [1, N-1, 1, H, W]
        'calibs': torch.from_numpy(np.stack(calibs)).unsqueeze(0),       # [1, N, 4]
        'frame_indices': torch.tensor(seq_frames),
    }


def load_vanilla_anycam(
    config_path: str,
    checkpoint_path: str,
    device: torch.device,
) -> torch.nn.Module:
    """Load vanilla AnyCam pose predictor with pretrained weights."""
    from omegaconf import OmegaConf
    from anycam.models import make_pose_predictor

    config = OmegaConf.load(config_path)
    model = make_pose_predictor(config.model.pose_predictor)

    # Load pretrained weights — extract pose_predictor.* keys
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_state = ckpt["model"]

    prefix = "pose_predictor."
    filtered = {
        k[len(prefix):]: v
        for k, v in model_state.items()
        if k.startswith(prefix)
    }

    missing, unexpected = model.load_state_dict(filtered, strict=False)
    logger.info(f"Loaded vanilla AnyCam: {len(filtered)} keys, "
                f"{len(missing)} missing, {len(unexpected)} unexpected")

    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model.to(device)


def run_vanilla_anycam(
    model: torch.nn.Module,
    seq_data: Dict[str, torch.Tensor],
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Run vanilla AnyCam pose predictor on a sequence. Returns selected poses."""
    images = seq_data['images'].to(device)       # [1, N, 3, H, W]
    depths = seq_data['depths'].to(device)        # [1, N, 1, H, W]
    flows = seq_data['flows_fwd'].to(device)      # [1, N-1, 2, H, W]
    occs = seq_data['occs_fwd'].to(device)        # [1, N-1, 1, H, W]

    B, N, C, H, W = images.shape

    # Normalize flow to [-1, 1] (AnyCam convention)
    flows_norm_x = flows[:, :, 0:1] * 2.0 / W
    flows_norm_y = flows[:, :, 1:2] * 2.0 / H
    flow_occs = torch.cat([flows_norm_x, flows_norm_y, occs], dim=2)  # [1, N-1, 3, H, W]

    # Pad last frame
    flow_occs_padded = torch.cat([
        flow_occs,
        torch.zeros(1, 1, 3, H, W, device=device),
    ], dim=1)  # [1, N, 3, H, W]

    with torch.no_grad(), torch.amp.autocast("cuda"):
        result = model(
            images,
            flow_occs=flow_occs_padded,
            depths=depths,
        )

    poses = result["poses"]                    # [1, N, num_candidates, 4, 4]
    focal_probs = result["focal_length_probs"]  # [1, 1, num_candidates]

    # Select best candidate
    best_idx = torch.argmax(focal_probs[:, 0], dim=-1)  # [1]
    selected_poses = poses[0, :, best_idx[0]]            # [N, 4, 4]

    return {
        "poses": selected_poses.cpu(),
        "focal_length": result["focal_length"].cpu(),
    }


def main():
    parser = argparse.ArgumentParser(description="Precompute vanilla baselines for validation")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--anycam_config", type=str,
                        default="pretrained_models/anycam_seq8/training_config.yaml")
    parser.add_argument("--anycam_checkpoint", type=str,
                        default="pretrained_models/anycam_seq8/training_checkpoint_247500.pt")
    parser.add_argument("--max_ahead", type=int, default=3)
    parser.add_argument("--image_size", type=int, default=336)
    args = parser.parse_args()

    seq_len = args.max_ahead + 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("Selecting validation sequences...")
    sequences = select_val_sequences(args.data_dir, seq_len=seq_len)
    logger.info(f"Found {len(sequences)} validation sequences")

    if not sequences:
        logger.error("No valid sequences found!")
        return

    logger.info("Loading vanilla AnyCam pose predictor...")
    vanilla_model = load_vanilla_anycam(args.anycam_config, args.anycam_checkpoint, device)

    results = []
    for seq_info in sequences:
        ds = seq_info["dataset_name"]
        vid = seq_info["video_name"]
        start = seq_info["start_frame"]

        logger.info(f"Running vanilla AnyCam on {ds}/{vid} frame {start}...")

        seq_data = load_sequence_data(
            args.data_dir, ds, vid, start, seq_len, args.image_size,
        )

        vanilla_result = run_vanilla_anycam(vanilla_model, seq_data, device)

        results.append({
            "dataset_name": ds,
            "video_name": vid,
            "start_frame": start,
            "vanilla_poses": vanilla_result["poses"],           # [N, 4, 4]
            "vanilla_focal": vanilla_result["focal_length"],    # scalar or [1]
            "anycalib_calib": seq_data["calibs"][0],            # [N, 4]
        })

    # Save
    output = {
        "sequences": results,
        "config": {
            "max_ahead": args.max_ahead,
            "image_size": args.image_size,
            "anycam_config": args.anycam_config,
            "anycam_checkpoint": args.anycam_checkpoint,
        },
    }

    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, args.output_path)
    logger.info(f"Saved baselines to {args.output_path}")


if __name__ == "__main__":
    main()
