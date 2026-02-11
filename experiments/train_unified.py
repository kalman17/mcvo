#!/usr/bin/env python3
"""
Unified training script for the FAT + AnyCam pipeline.

Orchestrates all training phases:
  Phase A:  Pose head initialization with AnyCalib calibration
  Phase B1: FAT pre-training (isolated, reprojection loss)
  Phase B3: FAT + pose head joint training (combined loss)
  Phase C:  End-to-end alternating training

All expensive model outputs (depth, flow, calib) are loaded from preprocessed
.npz files. Only the DINOv2 backbones run live (frozen forward passes).

Usage:
    # Phase A
    python experiments/train_unified.py --phase A --data_dir /path/to/data \\
        --save_dir /path/to/output --num_epochs 50

    # Phase B1
    python experiments/train_unified.py --phase B1 --data_dir /path/to/data \\
        --save_dir /path/to/output --num_epochs 50

    # Phase B3 (requires B1 checkpoint)
    python experiments/train_unified.py --phase B3 --data_dir /path/to/data \\
        --save_dir /path/to/output --phase_b1_checkpoint /path/to/b1.pt

    # Phase C (requires B3 checkpoint)
    python experiments/train_unified.py --phase C --data_dir /path/to/data \\
        --save_dir /path/to/output --phase_b3_checkpoint /path/to/b3.pt

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
from torch.utils.data import DataLoader

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
    """Save loss curves over epochs."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    epochs = [h["epoch"] for h in loss_history]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot all available loss keys
    loss_keys = [k for k in loss_history[0] if k != "epoch" and k != "time"]
    for key in loss_keys:
        values = [h.get(key, 0) for h in loss_history]
        ax.plot(epochs, values, label=key)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
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
) -> Dict:
    """
    Train for one epoch.

    Returns dict with averaged loss values.
    """
    model.train()

    running_losses = {}
    n_batches = 0

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

        elif phase in ("B3", "C"):
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
        torch.nn.utils.clip_grad_norm_(model.get_trainable_parameters(), max_norm=1.0)
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

    # Average over epoch
    if n_batches > 0:
        avg_losses = {k: v / n_batches for k, v in running_losses.items()}
    else:
        avg_losses = {"total": float("nan")}

    return avg_losses


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

    # Also save epoch-specific checkpoint every 10 epochs
    if (epoch + 1) % 10 == 0:
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
    """Initialize metrics CSV file."""
    path = Path(save_dir) / "metrics.csv"
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
    parser.add_argument("--phase", type=str, required=True, choices=["A", "B1", "B3", "C"],
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
                        help="Weight for calibration anchor loss (B3/C)")
    parser.add_argument("--lambda_comp", type=float, default=0.1,
                        help="Weight for composed flow pairs")
    parser.add_argument("--num_workers", type=int, default=4)

    # Model
    parser.add_argument("--anycam_config", type=str,
                        default="pretrained_models/anycam_seq8/training_config.yaml",
                        help="Path to AnyCam training config")

    # Checkpoint loading
    parser.add_argument("--phase_a_checkpoint", type=str, default=None,
                        help="Phase A checkpoint (for B3/C)")
    parser.add_argument("--phase_b1_checkpoint", type=str, default=None,
                        help="Phase B1 checkpoint (for B3)")
    parser.add_argument("--phase_b3_checkpoint", type=str, default=None,
                        help="Phase B3 checkpoint (for C)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint (same phase)")

    # Phase C specific
    parser.add_argument("--calib_epochs_ratio", type=float, default=0.5,
                        help="Fraction of epochs dedicated to calib mode in Phase C")

    # Test mode
    parser.add_argument("--test", action="store_true",
                        help="Quick test: 10 samples, 2 epochs, validates pipeline")

    args = parser.parse_args()

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

    dataset = PreprocessedMultiFrameDataset(
        data_dir=args.data_dir,
        datasets=datasets_list,
        max_ahead=args.max_ahead,
        image_size=args.image_size,
        phase=args.phase,
    )

    if args.test and len(dataset) > 10:
        dataset.samples = dataset.samples[:10]

    logger.info(f"Dataset: {len(dataset)} samples")

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )

    # ---- Model ----
    from experiments.models.unified_wrapper import UnifiedTrainingWrapper

    model = UnifiedTrainingWrapper(
        phase=args.phase,
        anycam_config_path=args.anycam_config,
        image_size=args.image_size,
    )

    # Load checkpoints from previous phases
    if args.phase == "B3":
        if args.phase_b1_checkpoint:
            model.load_phase_checkpoint(args.phase_b1_checkpoint, source_phase="B1")
        else:
            logger.warning("Phase B3 started without B1 checkpoint — FAT is randomly initialized")

    elif args.phase == "C":
        if args.phase_b3_checkpoint:
            model.load_phase_checkpoint(args.phase_b3_checkpoint, source_phase="B3")
        elif args.phase_b1_checkpoint:
            model.load_phase_checkpoint(args.phase_b1_checkpoint, source_phase="B1")
            if args.phase_a_checkpoint:
                model.load_phase_checkpoint(args.phase_a_checkpoint, source_phase="A")
        else:
            logger.warning("Phase C started without prior checkpoints")

    model = model.to(device)

    # ---- Optimizer ----
    trainable_params = model.get_trainable_parameters()
    logger.info(f"Trainable parameters: {sum(p.numel() for p in trainable_params):,}")

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.learning_rate,
        weight_decay=0.0,
        betas=(0.9, 0.999),
    )
    scaler = torch.amp.GradScaler("cuda")

    # ---- Resume ----
    start_epoch = 0
    loss_history = []

    if args.resume:
        ckpt = load_checkpoint(args.resume, model, optimizer, scaler)
        start_epoch = ckpt.get("epoch", 0) + 1
        loss_history = ckpt.get("loss_history", [])
        logger.info(f"Resuming from epoch {start_epoch}")

    # ---- Metrics CSV ----
    csv_fields = ["epoch", "total", "flow", "calib", "lr", "time"]
    csv_path = init_metrics_csv(str(save_dir), csv_fields)

    # ---- Training loop ----
    config_dict = vars(args)
    logger.info(f"Starting Phase {args.phase} training for {args.num_epochs} epochs")
    logger.info(f"Config: {config_dict}")

    for epoch in range(start_epoch, args.num_epochs):
        epoch_start = time.time()

        # Phase C alternating
        if args.phase == "C":
            # Alternate between pose and calib modes
            if epoch % 2 == 0:
                model.set_training_mode("pose")
                # Rebuild optimizer for new set of trainable params
                trainable_params = model.get_trainable_parameters()
                optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate)
            else:
                model.set_training_mode("calib")
                trainable_params = model.get_trainable_parameters()
                optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate)

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
        )

        epoch_time = time.time() - epoch_start

        # Log
        loss_str = " | ".join(f"{k}={v:.6f}" for k, v in avg_losses.items())
        logger.info(f"Epoch {epoch + 1}/{args.num_epochs}: {loss_str} ({epoch_time:.1f}s)")

        # Record history
        record = {"epoch": epoch + 1, **avg_losses, "time": epoch_time}
        loss_history.append(record)

        # Write to CSV
        csv_row = {
            "epoch": epoch + 1,
            "total": avg_losses.get("total", 0),
            "flow": avg_losses.get("flow", 0),
            "calib": avg_losses.get("calib", 0),
            "lr": args.learning_rate,
            "time": f"{epoch_time:.1f}",
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
