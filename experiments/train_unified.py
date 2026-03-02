#!/usr/bin/env python3
"""
Unified training script for the FAT + AnyCam pipeline.

Orchestrates all training phases:
  Phase A:  Pose head initialization with AnyCalib calibration
  Phase B1: FAT pre-training (isolated, reprojection loss)
  Phase B2: FAT end-to-end through frozen pose pipeline (flow loss)
  Phase C:  End-to-end joint training (all params unfrozen)

All expensive model outputs (depth, flow, calib) are loaded from preprocessed
.npz files. Only the DINOv2 backbones run live (frozen forward passes).

Usage:
    # Phase A
    python experiments/train_unified.py --phase A --data_dir /path/to/data \\
        --save_dir /path/to/output --num_epochs 50

    # Phase B1
    python experiments/train_unified.py --phase B1 --data_dir /path/to/data \\
        --save_dir /path/to/output --num_epochs 50

    # Phase B2 (requires A + B1 checkpoints)
    python experiments/train_unified.py --phase B2 --data_dir /path/to/data \\
        --save_dir /path/to/output --phase_a_checkpoint /path/to/a.pt \\
        --phase_b1_checkpoint /path/to/b1.pt

    # Phase C (requires B2 + A checkpoints)
    python experiments/train_unified.py --phase C --data_dir /path/to/data \\
        --save_dir /path/to/output --phase_b2_checkpoint /path/to/b2.pt \\
        --phase_a_checkpoint /path/to/a.pt

    # Quick test mode
    python experiments/train_unified.py --phase A --data_dir /path/to/data \\
        --save_dir /tmp/test --test
"""

import argparse
import csv
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, random_split

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_unified")


# ---------------------------------------------------------------------------
# Flow composition utilities (from train_pose_head_anycalib_exp2.py)
# ---------------------------------------------------------------------------

def compose_flows(
    flow_list: List[torch.Tensor],
    occlusion_list: List[torch.Tensor],
) -> tuple:
    """
    Compose consecutive flows by warping through intermediate frames.

    Args:
        flow_list: List of [B, 2, H, W] consecutive flows.
        occlusion_list: List of [B, 1, H, W] occlusion masks.

    Returns:
        (composed_flow [B, 2, H, W], composed_occ [B, 1, H, W])
    """
    composed_flow = flow_list[0].clone()
    composed_occ = occlusion_list[0].clone()
    _, _, h, w = composed_flow.shape
    device = composed_flow.device

    for i in range(1, len(flow_list)):
        curr_flow = flow_list[i]
        curr_occ = occlusion_list[i]

        # Build sampling grid from composed flow so far
        y_coords, x_coords = torch.meshgrid(
            torch.arange(h, device=device, dtype=torch.float32),
            torch.arange(w, device=device, dtype=torch.float32),
            indexing="ij",
        )

        # Warp coordinates
        warped_x = x_coords.unsqueeze(0) + composed_flow[:, 0]
        warped_y = y_coords.unsqueeze(0) + composed_flow[:, 1]

        # Normalize to [-1, 1] for grid_sample
        grid_x = (warped_x / (w - 1)) * 2 - 1
        grid_y = (warped_y / (h - 1)) * 2 - 1
        grid = torch.stack([grid_x, grid_y], dim=-1)  # [B, H, W, 2]

        # Sample current flow at warped locations
        sampled_flow = F.grid_sample(
            curr_flow, grid, mode="bilinear", padding_mode="zeros", align_corners=True,
        )

        composed_flow = composed_flow + sampled_flow

        # Compose occlusion
        valid = (warped_x >= 0) & (warped_x < w) & (warped_y >= 0) & (warped_y < h)
        sampled_occ = F.grid_sample(
            curr_occ, grid, mode="bilinear", padding_mode="zeros", align_corners=True,
        )
        composed_occ = composed_occ * sampled_occ * valid.float().unsqueeze(1)

    return composed_flow, composed_occ


def compose_poses(pose_list: List[torch.Tensor]) -> torch.Tensor:
    """Compose consecutive 4x4 poses: T_1->3 = T_1->2 @ T_2->3."""
    composed = pose_list[0]
    for p in pose_list[1:]:
        composed = composed @ p
    return composed


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def flow_to_color(flow: np.ndarray, max_flow: float = None) -> np.ndarray:
    """Convert [2, H, W] flow to [H, W, 3] RGB image."""
    u = flow[0]
    v = flow[1]
    mag = np.sqrt(u ** 2 + v ** 2)
    if max_flow is None:
        max_flow = max(mag.max(), 1e-5)
    ang = np.arctan2(v, u)

    hsv = np.zeros((*u.shape, 3), dtype=np.uint8)
    hsv[..., 0] = ((ang + np.pi) / (2 * np.pi) * 179).astype(np.uint8)
    hsv[..., 1] = 255
    hsv[..., 2] = np.clip(mag / max_flow * 255, 0, 255).astype(np.uint8)

    import cv2
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def save_visualization(
    save_path: str,
    images: torch.Tensor,
    induced_flow: Optional[torch.Tensor] = None,
    target_flow: Optional[torch.Tensor] = None,
    depths: Optional[torch.Tensor] = None,
    valid_mask: Optional[torch.Tensor] = None,
    calib_pred: Optional[torch.Tensor] = None,
    calib_gt: Optional[torch.Tensor] = None,
):
    """Save a visualization grid for one sample."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available, skipping visualization")
        return

    n_frames = images.shape[0]
    n_rows = 2  # frames row + flow/depth row
    n_cols = max(n_frames, 3)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]
    if n_cols == 1:
        axes = axes[:, np.newaxis]

    # Row 0: input frames
    for i in range(n_frames):
        img = images[i].cpu().numpy().transpose(1, 2, 0)
        img = np.clip(img, 0, 1)
        axes[0, i].imshow(img)
        axes[0, i].set_title(f"Frame {i}")
        axes[0, i].axis("off")

    # Row 1: flow visualizations
    if induced_flow is not None and target_flow is not None:
        n_pairs = min(induced_flow.shape[0], n_cols)
        for i in range(n_pairs):
            ind = induced_flow[i].cpu().numpy()
            tgt = target_flow[i].cpu().numpy()
            max_f = max(np.abs(ind).max(), np.abs(tgt).max(), 1e-5)
            combined = np.concatenate([
                flow_to_color(ind, max_f),
                flow_to_color(tgt, max_f),
            ], axis=1)
            axes[1, i].imshow(combined)
            axes[1, i].set_title(f"Induced | Target (pair {i})")
            axes[1, i].axis("off")

    # Hide unused axes
    for r in range(n_rows):
        for c in range(n_cols):
            if not axes[r, c].has_data():
                axes[r, c].axis("off")

    # Add calibration info as text
    if calib_pred is not None or calib_gt is not None:
        info = ""
        if calib_pred is not None:
            info += f"Pred: fx={calib_pred[0]:.1f} fy={calib_pred[1]:.1f} cx={calib_pred[2]:.1f} cy={calib_pred[3]:.1f}\n"
        if calib_gt is not None:
            info += f"GT:   fx={calib_gt[0]:.1f} fy={calib_gt[1]:.1f} cx={calib_gt[2]:.1f} cy={calib_gt[3]:.1f}"
        fig.suptitle(info, fontsize=10, family="monospace")

    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def save_loss_plot(save_path: str, loss_history: List[Dict]):
    """Save train + val loss curves."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    epochs = [h["epoch"] for h in loss_history]
    train_loss = [h.get("total", float("nan")) for h in loss_history]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_loss, "tab:blue", linewidth=1.5, label="Train loss")

    # Val loss (if available)
    val_loss = [h.get("val_loss", float("nan")) for h in loss_history]
    if any(not np.isnan(v) for v in val_loss):
        ax.plot(epochs, val_loss, "tab:orange", linewidth=1.5, label="Val loss")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss")
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Validation: divergence from vanilla baselines
# ---------------------------------------------------------------------------

def load_val_baselines(
    baselines_path: str,
    data_dir: str,
    image_size: int,
    seq_len: int,
) -> List[Dict]:
    """
    Load precomputed vanilla baselines and prepare input batches.

    Returns list of dicts, each with:
        - dataset_name, video_name, start_frame
        - vanilla_poses [N, 4, 4], vanilla_focal, anycalib_calib [N, 4]
        - input_data: dict with tensors ready for model forward
    """
    from experiments.precompute_vanilla_baselines import load_sequence_data

    cache = torch.load(baselines_path, map_location="cpu", weights_only=False)
    sequences = cache["sequences"]

    val_data = []
    for seq in sequences:
        try:
            input_data = load_sequence_data(
                data_dir=data_dir,
                dataset_name=seq["dataset_name"],
                video_name=seq["video_name"],
                start_frame=seq["start_frame"],
                seq_len=seq_len,
                image_size=image_size,
            )
            val_data.append({
                "dataset_name": seq["dataset_name"],
                "video_name": seq["video_name"],
                "start_frame": seq["start_frame"],
                "vanilla_poses": seq["vanilla_poses"],       # [N, 4, 4]
                "vanilla_focal": seq["vanilla_focal"],
                "anycalib_calib": seq["anycalib_calib"],      # [N, 4]
                "input_data": input_data,
            })
        except Exception as e:
            logger.warning(f"Failed to load val sequence {seq['dataset_name']}/{seq['video_name']}: {e}")

    logger.info(f"Loaded {len(val_data)} validation sequences from {baselines_path}")
    return val_data


def compute_pose_divergence(
    our_poses: torch.Tensor,
    vanilla_poses: torch.Tensor,
) -> Dict[str, float]:
    """
    Compute pose divergence between our poses and vanilla AnyCam poses.

    Both inputs: [N, 4, 4] absolute poses. We compute relative poses
    (frame 0 -> frame i) and compare rotation angles, translation direction,
    and translation magnitude.
    """
    from anycam.loss.metric import rotation_angle, translation_angle

    N = our_poses.shape[0]
    if N < 2:
        return {
            "rot_div_mean": 0.0, "rot_div_median": 0.0,
            "trans_dir_div_mean": 0.0, "trans_mag_div_mean": 0.0,
        }

    # Compute relative poses w.r.t. first frame
    our_rel_R = []
    van_rel_R = []
    our_rel_t = []
    van_rel_t = []
    our_inv0 = torch.inverse(our_poses[0])
    van_inv0 = torch.inverse(vanilla_poses[0])
    for i in range(1, N):
        our_rel_pose = our_inv0 @ our_poses[i]
        van_rel_pose = van_inv0 @ vanilla_poses[i]
        our_rel_R.append(our_rel_pose[:3, :3])
        van_rel_R.append(van_rel_pose[:3, :3])
        our_rel_t.append(our_rel_pose[:3, 3])
        van_rel_t.append(van_rel_pose[:3, 3])

    our_rots = torch.stack(our_rel_R)   # [N-1, 3, 3]
    van_rots = torch.stack(van_rel_R)   # [N-1, 3, 3]
    our_trans = torch.stack(our_rel_t)   # [N-1, 3]
    van_trans = torch.stack(van_rel_t)   # [N-1, 3]

    # Rotation divergence (degrees)
    rot_angles = rotation_angle(van_rots, our_rots)  # [N-1]

    # Translation direction divergence (degrees)
    trans_dir_angles = translation_angle(van_trans, our_trans)  # [N-1]
    # Filter out default error values (1e6) from near-zero translations
    valid_trans = trans_dir_angles < 1e5
    trans_dir_clean = trans_dir_angles[valid_trans] if valid_trans.any() else trans_dir_angles

    # Translation magnitude divergence (Euclidean distance)
    trans_mag_diff = (our_trans - van_trans).norm(dim=1)  # [N-1]

    return {
        "rot_div_mean": rot_angles.mean().item(),
        "rot_div_median": rot_angles.median().item(),
        "trans_dir_div_mean": trans_dir_clean.mean().item(),
        "trans_dir_div_median": trans_dir_clean.median().item(),
        "trans_mag_div_mean": trans_mag_diff.mean().item(),
    }


def compute_calib_divergence(
    our_calib: torch.Tensor,
    anycalib_calib: torch.Tensor,
) -> Dict[str, float]:
    """
    Compute calibration divergence: MAE of fx, fy in pixels.

    Both inputs: [N, 4] with [fx, fy, cx, cy] or [4] for single prediction.
    """
    if our_calib.dim() == 1:
        our_calib = our_calib.unsqueeze(0)
    if anycalib_calib.dim() == 1:
        anycalib_calib = anycalib_calib.unsqueeze(0)

    # Average over frames
    our_mean = our_calib.mean(dim=0)
    ref_mean = anycalib_calib.float().mean(dim=0)

    return {
        "calib_fx_mae": abs(our_mean[0].item() - ref_mean[0].item()),
        "calib_fy_mae": abs(our_mean[1].item() - ref_mean[1].item()),
    }


def run_validation(
    model,
    val_data: List[Dict],
    device: torch.device,
    phase: str,
) -> Dict[str, float]:
    """
    Run model on validation sequences, compute divergence metrics.

    Returns aggregated metrics dict.
    """
    model.eval()

    all_rot_divs = []
    all_trans_dir_divs = []
    all_trans_mag_divs = []
    all_calib_fx = []
    all_calib_fy = []
    per_dataset = {}

    for seq in val_data:
        ds_name = seq["dataset_name"]
        input_data = seq["input_data"]

        # Move to device
        data = {}
        for k, v in input_data.items():
            if isinstance(v, torch.Tensor):
                data[k] = v.to(device, non_blocking=True)
            else:
                data[k] = v

        try:
            with torch.no_grad(), torch.amp.autocast(device_type="cuda", enabled=True):
                result = model(data)
        except Exception as e:
            logger.warning(f"Validation forward failed for {ds_name}: {e}")
            continue

        ds_metrics = {}

        # Pose divergence (phases A, B2, C)
        if phase in ("A", "B2", "C") and "poses" in result:
            our_poses = result["poses"][0]  # [N, num_candidates, 4, 4] or [N, 4, 4]
            # If multi-candidate, select best
            if our_poses.dim() == 4:
                focal_probs = result.get("focal_length_probs")
                if focal_probs is not None:
                    best_idx = torch.argmax(focal_probs[0, 0], dim=-1)
                    our_poses = our_poses[:, best_idx]
                else:
                    our_poses = our_poses[:, 0]

            vanilla_poses = seq["vanilla_poses"].to(device)
            pose_div = compute_pose_divergence(our_poses, vanilla_poses)
            ds_metrics.update(pose_div)
            all_rot_divs.append(pose_div["rot_div_mean"])
            all_trans_dir_divs.append(pose_div["trans_dir_div_mean"])
            all_trans_mag_divs.append(pose_div["trans_mag_div_mean"])

        # Calibration divergence (phases B1, B2, C)
        if phase in ("B1", "B2", "C") and "fat_intrinsics" in result:
            our_calib = result["fat_intrinsics"][0].cpu()  # [N, 4] or [4]
            ref_calib = seq["anycalib_calib"]
            if ref_calib.dim() == 2:
                ref_calib = ref_calib[0]  # Use first frame's calib as ref
            calib_div = compute_calib_divergence(our_calib, ref_calib)
            ds_metrics.update(calib_div)
            all_calib_fx.append(calib_div["calib_fx_mae"])
            all_calib_fy.append(calib_div["calib_fy_mae"])

        per_dataset[ds_name] = ds_metrics

    model.train()

    # Aggregate
    metrics = {}
    if all_rot_divs:
        metrics["val_rot_div_mean"] = np.mean(all_rot_divs)
        metrics["val_rot_div_median"] = np.median(all_rot_divs)
    if all_trans_dir_divs:
        metrics["val_trans_dir_div_mean"] = np.mean(all_trans_dir_divs)
    if all_trans_mag_divs:
        metrics["val_trans_mag_div_mean"] = np.mean(all_trans_mag_divs)
    if all_calib_fx:
        metrics["val_calib_fx_mae"] = np.mean(all_calib_fx)
        metrics["val_calib_fy_mae"] = np.mean(all_calib_fy)

    # Per-dataset
    for ds_name, ds_m in per_dataset.items():
        for k, v in ds_m.items():
            metrics[f"val_{ds_name}_{k}"] = v

    return metrics


def save_divergence_plot(save_path: str, val_history: List[Dict]):
    """Save divergence plot with one panel per metric (own y-axis scale)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    epochs = [h["epoch"] for h in val_history]

    # Build list of (key, label, color, ylabel) for each available metric
    metric_defs = [
        ("val_rot_div_mean", "Rotation", "tab:blue", "Rotation div. (deg)"),
        ("val_trans_dir_div_mean", "Trans. direction", "tab:orange", "Trans. dir. div. (deg)"),
        ("val_trans_mag_div_mean", "Trans. magnitude", "tab:purple", "Trans. mag. div."),
        ("val_calib_fx_mae", "Calib fx MAE", "tab:red", "Calib MAE (px)"),
    ]

    # Filter to metrics that actually exist in the history
    active = [(key, label, color, ylabel) for key, label, color, ylabel in metric_defs
              if any(key in h for h in val_history)]

    if not active:
        return

    n = len(active)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4))
    if n == 1:
        axes = [axes]

    for i, (key, label, color, ylabel) in enumerate(active):
        ax = axes[i]
        vals = [h.get(key, float("nan")) for h in val_history]
        ax.plot(epochs, vals, color=color, marker="o", markersize=3, linewidth=1.5)

        # For calib panel, also plot fy
        if key == "val_calib_fx_mae":
            fy_vals = [h.get("val_calib_fy_mae", float("nan")) for h in val_history]
            ax.plot(epochs, fy_vals, color="tab:green", marker="s", markersize=3,
                    linewidth=1.5, linestyle="--")
            ax.legend(["fx", "fy"], fontsize=9)

        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(label)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Divergence vs vanilla AnyCam", fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_one_epoch(
    model,
    dataloader,
    optimizer,
    scaler,
    device,
    phase: str,
    lambda_calib: float = 1e-4,
    max_ahead: int = 3,
    lambda_comp: float = 0.1,
    # Intra-epoch checkpoint saving
    save_dir: str = None,
    epoch: int = 0,
    loss_history: List = None,
    config: Dict = None,
    intra_save_minutes: float = 30.0,
) -> Dict:
    """
    Train for one epoch.

    Returns dict with averaged loss values.
    """
    model.train()

    running_losses = {}
    n_batches = 0
    last_intra_save = time.time()

    for batch_idx, batch in enumerate(dataloader):
        # Move to device
        data = {}
        for key, val in batch.items():
            if isinstance(val, torch.Tensor):
                data[key] = val.to(device, non_blocking=True)
            else:
                data[key] = val

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type="cuda", enabled=True):
            result = model(data)

        # Compute total loss depending on phase
        if phase == "A":
            loss = result["loss"]
            losses_to_log = {
                "total": loss.item(),
                "flow": result["flow_loss"].item(),
                "flow_raw": result.get("flow_loss_raw", result["flow_loss"]).item(),
            }

        elif phase == "B1":
            loss = result["loss"]
            losses_to_log = {"total": loss.item(), "calib": result["calib_loss"].item()}

        elif phase in ("B2", "C"):
            flow_loss = result["flow_loss"]
            calib_loss = result["calib_loss"]
            loss = flow_loss + lambda_calib * calib_loss
            losses_to_log = {
                "total": loss.item(),
                "flow": flow_loss.item(),
                "flow_raw": result.get("flow_loss_raw", flow_loss).item(),
                "calib": calib_loss.item(),
            }
        else:
            raise ValueError(f"Unknown phase: {phase}")

        # Check for NaN/Inf
        if torch.isnan(loss) or torch.isinf(loss):
            logger.warning(f"NaN/Inf loss at batch {batch_idx}, skipping")
            continue

        # Backward pass
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.get_trainable_parameters(), max_norm=0.5)
        scaler.step(optimizer)
        scaler.update()

        # Accumulate losses
        for k, v in losses_to_log.items():
            running_losses[k] = running_losses.get(k, 0) + v
        n_batches += 1

        # Progress logging every 10 batches
        if (batch_idx + 1) % 10 == 0:
            avg = {k: v / n_batches for k, v in running_losses.items()}
            loss_str = " | ".join(f"{k}={v:.6f}" for k, v in avg.items())
            logger.info(f"  Batch {batch_idx + 1}/{len(dataloader)}: {loss_str}")

        # Per-batch loss logging every 500 batches (detect divergence early)
        if (batch_idx + 1) % 500 == 0:
            logger.info(f"  [per-batch] Batch {batch_idx + 1}: {' | '.join(f'{k}={v:.6f}' for k, v in losses_to_log.items())}")

        # Intra-epoch checkpoint saving (every ~30 minutes)
        if save_dir is not None and (time.time() - last_intra_save) >= intra_save_minutes * 60:
            intra_filename = f"intra_epoch{epoch + 1}_save.pt"
            save_checkpoint(
                save_dir, model, optimizer, scaler,
                epoch, phase, loss_history or [], config or {},
                filename=intra_filename,
            )
            logger.info(f"  Intra-epoch save at batch {batch_idx + 1}/{len(dataloader)}")
            last_intra_save = time.time()

    # Average over epoch
    if n_batches > 0:
        avg_losses = {k: v / n_batches for k, v in running_losses.items()}
    else:
        avg_losses = {"total": float("nan")}

    return avg_losses


def compute_val_loss(
    model,
    val_dataloader,
    device,
    phase: str,
    lambda_calib: float = 1e-4,
) -> float:
    """Compute average loss on the validation set (no gradient)."""
    model.eval()
    total_loss = 0.0
    n_batches = 0

    for batch in val_dataloader:
        data = {}
        for key, val in batch.items():
            if isinstance(val, torch.Tensor):
                data[key] = val.to(device, non_blocking=True)
            else:
                data[key] = val

        try:
            with torch.no_grad(), torch.amp.autocast(device_type="cuda", enabled=True):
                result = model(data)

            if phase == "A":
                loss = result["loss"]
            elif phase == "B1":
                loss = result["loss"]
            elif phase in ("B2", "C"):
                loss = result["flow_loss"] + lambda_calib * result["calib_loss"]
            else:
                loss = result["loss"]

            if not (torch.isnan(loss) or torch.isinf(loss)):
                total_loss += loss.item()
                n_batches += 1
        except Exception:
            continue

    model.train()
    return total_loss / n_batches if n_batches > 0 else float("nan")


# ---------------------------------------------------------------------------
# Checkpoint management
# ---------------------------------------------------------------------------

def save_checkpoint(
    save_dir: str,
    model,
    optimizer,
    scaler,
    epoch: int,
    phase: str,
    loss_history: List,
    config: Dict,
    filename: str = "latest.pt",
):
    """Save training checkpoint."""
    ckpt_dir = Path(save_dir) / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "phase": phase,
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "loss_history": loss_history,
        "config": config,
    }

    path = ckpt_dir / filename
    torch.save(checkpoint, path)
    logger.info(f"Saved checkpoint: {path}")

    # Save epoch-specific checkpoint only for end-of-epoch saves (not intra-epoch)
    if filename == "latest.pt":
        epoch_path = ckpt_dir / f"epoch_{epoch + 1:04d}.pt"
        torch.save(checkpoint, epoch_path)


def load_checkpoint(
    path: str,
    model,
    optimizer=None,
    scaler=None,
) -> Dict:
    """Load training checkpoint."""
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scaler is not None and "scaler_state_dict" in ckpt:
        scaler.load_state_dict(ckpt["scaler_state_dict"])
    logger.info(f"Loaded checkpoint from {path} (epoch {ckpt.get('epoch', '?')})")
    return ckpt


# ---------------------------------------------------------------------------
# Metrics CSV
# ---------------------------------------------------------------------------

def init_metrics_csv(save_dir: str, fieldnames: List[str]):
    """Initialize metrics CSV file (preserves existing rows on resume)."""
    path = Path(save_dir) / "metrics.csv"
    if path.exists():
        return path
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
    return path


def append_metrics_csv(path: str, row: Dict):
    """Append a row to the metrics CSV."""
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writerow(row)


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Unified training script for FAT + AnyCam pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required
    parser.add_argument("--phase", type=str, required=True, choices=["A", "B1", "B2", "C"],
                        help="Training phase")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to preprocessed data directory")
    parser.add_argument("--save_dir", type=str, required=True,
                        help="Path to save outputs (checkpoints, plots, logs)")

    # Dataset
    parser.add_argument("--datasets", type=str, default=None,
                        help="Comma-separated dataset names (default: all)")
    parser.add_argument("--max_ahead", type=int, default=3,
                        help="Max lookahead for flow composition")
    parser.add_argument("--image_size", type=int, default=336,
                        help="Target image size (square)")

    # Training
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--lambda_calib", type=float, default=1e-4,
                        help="Weight for calibration anchor loss (B2/C)")
    parser.add_argument("--lambda_comp", type=float, default=0.1,
                        help="Weight for composed flow pairs")
    parser.add_argument("--num_workers", type=int, default=4)

    # Model
    parser.add_argument("--anycam_config", type=str,
                        default="pretrained_models/anycam_seq8/training_config.yaml",
                        help="Path to AnyCam training config")

    # Checkpoint loading
    parser.add_argument("--pretrained_anycam", type=str, default=None,
                        help="Pretrained AnyCam checkpoint to initialize pose head (warm start)")
    parser.add_argument("--phase_a_checkpoint", type=str, default=None,
                        help="Phase A checkpoint (for B2/C)")
    parser.add_argument("--phase_b1_checkpoint", type=str, default=None,
                        help="Phase B1 checkpoint (for B2)")
    parser.add_argument("--phase_b2_checkpoint", type=str, default=None,
                        help="Phase B2 checkpoint (for C)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint (same phase)")

    # Validation
    parser.add_argument("--val_baselines", type=str, default=None,
                        help="Path to precomputed vanilla baselines (.pt) for per-epoch divergence monitoring")

    # Test mode
    parser.add_argument("--test", action="store_true",
                        help="Quick test: 10 samples, 2 epochs, validates pipeline")

    args = parser.parse_args()

    # Phase C default LR: 1e-5 (fine-tuning pretrained backbones)
    if args.phase == "C" and args.learning_rate == 1e-4:
        args.learning_rate = 1e-5
        logger.info("Phase C: using default lr=1e-5 (override with --learning_rate)")

    # Create output directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name()}")

    # ---- Dataset ----
    from experiments.datasets.preprocessed_dataset import (
        PreprocessedMultiFrameDataset,
        collate_fn,
    )

    datasets_list = args.datasets.split(",") if args.datasets else None

    if args.test:
        logger.info("=== TEST MODE: 10 samples, 2 epochs ===")
        args.num_epochs = 2
        args.batch_size = 1

    full_dataset = PreprocessedMultiFrameDataset(
        data_dir=args.data_dir,
        datasets=datasets_list,
        max_ahead=args.max_ahead,
        image_size=args.image_size,
        phase=args.phase,
    )

    if args.test and len(full_dataset) > 10:
        full_dataset.samples = full_dataset.samples[:10]

    # Train/val split (90/10, deterministic seed for reproducibility)
    n_total = len(full_dataset)
    n_val = max(1, int(n_total * 0.1))
    n_train = n_total - n_val
    train_dataset, val_dataset = random_split(
        full_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )
    logger.info(f"Dataset: {n_total} total, {n_train} train, {n_val} val")

    dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=False,
    )

    # ---- Model ----
    from experiments.models.unified_wrapper import UnifiedTrainingWrapper

    model = UnifiedTrainingWrapper(
        phase=args.phase,
        anycam_config_path=args.anycam_config,
        image_size=args.image_size,
    )

    # Initialize pose head from pretrained AnyCam (warm start)
    if args.pretrained_anycam and model.pose_predictor is not None:
        model.load_pretrained_pose_predictor(args.pretrained_anycam)

    # Load checkpoints from previous phases
    if args.phase == "B2":
        # B2 needs: trained pose head (Phase A) + pre-trained FAT (Phase B1)
        if args.phase_a_checkpoint:
            model.load_phase_checkpoint(args.phase_a_checkpoint, source_phase="A")
        else:
            logger.warning("Phase B2 started without Phase A checkpoint — pose head is randomly initialized")
        if args.phase_b1_checkpoint:
            model.load_phase_checkpoint(args.phase_b1_checkpoint, source_phase="B1")
        else:
            logger.warning("Phase B2 started without B1 checkpoint — FAT is randomly initialized")

    elif args.phase == "C":
        # C needs: pre-trained FAT (Phase B1) + trained pose head (Phase A)
        if args.phase_b1_checkpoint:
            model.load_phase_checkpoint(args.phase_b1_checkpoint, source_phase="B1")
        elif args.phase_b2_checkpoint:
            model.load_phase_checkpoint(args.phase_b2_checkpoint, source_phase="B2")
        else:
            logger.warning("Phase C started without B1/B2 checkpoint — FAT is randomly initialized")
        if args.phase_a_checkpoint:
            model.load_phase_checkpoint(args.phase_a_checkpoint, source_phase="A")
        else:
            logger.warning("Phase C started without Phase A checkpoint — pose head is randomly initialized")

    model = model.to(device)

    # ---- Validation baselines ----
    val_data = None
    val_history = []
    if args.val_baselines:
        seq_len = args.max_ahead + 1
        val_data = load_val_baselines(
            args.val_baselines, args.data_dir, args.image_size, seq_len,
        )
        if not val_data:
            logger.warning("No valid validation sequences loaded — disabling validation")
            val_data = None

    # ---- Optimizer ----
    trainable_params = model.get_trainable_parameters()
    logger.info(f"Trainable parameters: {sum(p.numel() for p in trainable_params):,}")
    optimizer = torch.optim.AdamW(
        trainable_params, lr=args.learning_rate, weight_decay=0.0, betas=(0.9, 0.999),
    )

    scaler = torch.amp.GradScaler("cuda")

    # ---- Resume ----
    start_epoch = 0
    loss_history = []

    if args.resume:
        ckpt = load_checkpoint(
            args.resume, model, optimizer, scaler,
        )
        start_epoch = ckpt.get("epoch", 0) + 1
        loss_history = ckpt.get("loss_history", [])
        logger.info(f"Resuming from epoch {start_epoch}")

    # ---- Metrics CSV ----
    csv_fields = ["epoch", "total", "val_loss", "flow", "calib", "lr", "time",
                  "val_rot_div_mean", "val_rot_div_median",
                  "val_trans_dir_div_mean", "val_trans_mag_div_mean",
                  "val_calib_fx_mae", "val_calib_fy_mae"]
    csv_path = init_metrics_csv(str(save_dir), csv_fields)

    # ---- Training loop ----
    config_dict = vars(args)
    logger.info(f"Starting Phase {args.phase} training for {args.num_epochs} epochs")
    logger.info(f"Config: {config_dict}")

    for epoch in range(start_epoch, args.num_epochs):
        epoch_start = time.time()

        # Train
        avg_losses = train_one_epoch(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            phase=args.phase,
            lambda_calib=args.lambda_calib,
            max_ahead=args.max_ahead,
            lambda_comp=args.lambda_comp,
            save_dir=str(save_dir),
            epoch=epoch,
            loss_history=loss_history,
            config=config_dict,
        )

        epoch_time = time.time() - epoch_start

        # Log
        loss_str = " | ".join(f"{k}={v:.6f}" for k, v in avg_losses.items())
        logger.info(f"Epoch {epoch + 1}/{args.num_epochs}: {loss_str} ({epoch_time:.1f}s)")

        # Record history
        record = {"epoch": epoch + 1, **avg_losses, "time": epoch_time}
        loss_history.append(record)

        # ---- Validation loss ----
        v_loss = compute_val_loss(model, val_dataloader, device, args.phase, args.lambda_calib)
        record["val_loss"] = v_loss
        logger.info(f"  Val loss: {v_loss:.6f}")

        # ---- Divergence vs vanilla baselines ----
        val_metrics = {}
        if val_data is not None:
            val_metrics = run_validation(model, val_data, device, args.phase)
            if val_metrics:
                # Log only aggregated metrics (skip per-dataset breakdown)
                agg_keys = ["val_rot_div_mean", "val_rot_div_median",
                            "val_trans_dir_div_mean", "val_trans_mag_div_mean",
                            "val_calib_fx_mae", "val_calib_fy_mae"]
                val_str = " | ".join(f"{k}={val_metrics[k]:.4f}"
                                     for k in agg_keys if k in val_metrics)
                logger.info(f"  Validation: {val_str}")
                record.update(val_metrics)
                val_history.append({"epoch": epoch + 1, **val_metrics})

        # Write to CSV
        csv_row = {
            "epoch": epoch + 1,
            "total": avg_losses.get("total", 0),
            "val_loss": v_loss if not np.isnan(v_loss) else "",
            "flow": avg_losses.get("flow", 0),
            "calib": avg_losses.get("calib", 0),
            "lr": args.learning_rate,
            "time": f"{epoch_time:.1f}",
            "val_rot_div_mean": val_metrics.get("val_rot_div_mean", ""),
            "val_rot_div_median": val_metrics.get("val_rot_div_median", ""),
            "val_trans_dir_div_mean": val_metrics.get("val_trans_dir_div_mean", ""),
            "val_trans_mag_div_mean": val_metrics.get("val_trans_mag_div_mean", ""),
            "val_calib_fx_mae": val_metrics.get("val_calib_fx_mae", ""),
            "val_calib_fy_mae": val_metrics.get("val_calib_fy_mae", ""),
        }
        append_metrics_csv(str(csv_path), csv_row)

        # Save checkpoint
        save_checkpoint(
            str(save_dir), model, optimizer, scaler,
            epoch, args.phase, loss_history, config_dict,
        )

        # Save loss plot
        if len(loss_history) > 1:
            save_loss_plot(str(save_dir / "losses.png"), loss_history)

        # Save divergence plot
        if len(val_history) > 1:
            save_divergence_plot(str(save_dir / "divergence.png"), val_history)

        # Save visualization (every 5 epochs or in test mode)
        if (epoch + 1) % 5 == 0 or args.test:
            try:
                # Grab one sample for visualization
                vis_batch = next(iter(dataloader))
                vis_data = {}
                for k, v in vis_batch.items():
                    if isinstance(v, torch.Tensor):
                        vis_data[k] = v[:1].to(device)
                    else:
                        vis_data[k] = v

                model.eval()
                with torch.no_grad():
                    vis_result = model(vis_data)
                model.train()

                save_visualization(
                    str(save_dir / f"vis_epoch_{epoch + 1:04d}.png"),
                    images=vis_data["images"][0],
                    induced_flow=vis_result.get("induced_flow", [None])[0] if "induced_flow" in vis_result else None,
                    target_flow=vis_result.get("target_flow", [None])[0] if "target_flow" in vis_result else None,
                    calib_pred=vis_result.get("fat_intrinsics", vis_result.get("focal_length", None)),
                    calib_gt=vis_data.get("calibs", [None])[0].mean(dim=0).cpu().numpy() if "calibs" in vis_data else None,
                )
            except Exception as e:
                logger.warning(f"Visualization failed: {e}")

        # Clear GPU cache
        torch.cuda.empty_cache()

    # ---- Final save ----
    save_checkpoint(
        str(save_dir), model, optimizer, scaler,
        args.num_epochs - 1, args.phase, loss_history, config_dict,
        filename="final.pt",
    )

    logger.info(f"Training complete! Results saved to {save_dir}")

    if args.test:
        final_loss = loss_history[-1].get("total", float("nan"))
        if not np.isnan(final_loss) and not np.isinf(final_loss):
            logger.info(f"TEST PASSED: Final loss = {final_loss:.6f}")
        else:
            logger.error(f"TEST FAILED: Final loss = {final_loss}")
            sys.exit(1)


if __name__ == "__main__":
    main()
