#!/usr/bin/env python3
"""
=============================================================================
DA3 Calibration Head Training - Stage 3: End-to-End Flow Reprojection
=============================================================================

EXPERIMENT GOAL:
----------------
Integrate DA3 calibration head into full AnyCam pipeline and train with
flow reprojection loss (self-supervised).

STAGE 3 OBJECTIVE:
------------------
Train the DA3 calibration head end-to-end with the full AnyCam pipeline.
- Load Stage 2 checkpoint
- Freeze all except calibration head (or use LoRA)
- Train with flow reprojection loss (self-supervised)

LOSS FUNCTION:
--------------
L_stage3 = Flow_Reprojection_Loss(poses, focal_length, flows, depths)

TRAINING DATA:
--------------
- Objectron dataset with all available frames
- extract_all_pairs=True, max_pairs_per_video=None (unlimited)
- Self-supervised (no GT calibration or poses needed)
- GT directory argument is optional and not used in loss computation

Author: AI Assistant for Kalman's Master's Thesis
Date: December 2025
"""

import sys
import os
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from tqdm import tqdm

# Add project paths
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "anycalib"))

# DA3 imports
from experiments.train_pose_head_anycalib import (
    ObjectronVideoDataset,
    AnyCaLibBatchInference,
    AnyCamWrapperWithDA3Calibration,
    load_dataset_split,
)
from experiments.train_calibration_head_da3_stage1 import (
    plot_loss_curve,
    save_training_summary,
)
from experiments.dataset_paths import (
    get_objectron_videos, get_objectron_gt, get_lightspeed_root
)

# AnyCam imports
from anycam.models import make_pose_predictor, make_depth_predictor
from anycam.loss import make_loss

print("[INIT] Imports successful")


# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================

def plot_loss_curve_stage3(loss_history: List[Dict], val_loss_history: Optional[List[Dict]], save_dir: Path):
    """Plot and save training and validation loss curves for Stage 3."""
    if not loss_history:
        return
    
    epochs = [item['epoch'] for item in loss_history]
    train_losses = [item['loss'] for item in loss_history]
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, 'b-', linewidth=2, label='Training Loss')
    
    # Plot validation if available
    val_epochs = []
    val_losses = []
    if val_loss_history:
        val_epochs = [item['epoch'] for item in val_loss_history]
        val_losses = [item['loss'] for item in val_loss_history]
        plt.plot(val_epochs, val_losses, 'r-', linewidth=2, label='Validation Loss')
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss (Flow Reprojection)', fontsize=12)
    plt.title('DA3 Stage 3 Training Loss', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    
    if len(train_losses) > 0:
        plt.annotate(f'Train Start: {train_losses[0]:.4f}', 
                    xy=(epochs[0], train_losses[0]), 
                    xytext=(10, 10), 
                    textcoords='offset points',
                    fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.5))
        plt.annotate(f'Train Final: {train_losses[-1]:.4f}', 
                    xy=(epochs[-1], train_losses[-1]), 
                    xytext=(-60, -20), 
                    textcoords='offset points',
                    fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.5))
    
    if val_loss_history and len(val_losses) > 0:
        plt.annotate(f'Val Final: {val_losses[-1]:.4f}', 
                    xy=(val_epochs[-1], val_losses[-1]), 
                    xytext=(-60, 20), 
                    textcoords='offset points',
                    fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightcoral', alpha=0.5))
    
    plot_path = save_dir / "loss_curve.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[SAVE] Loss curve saved: {plot_path}")

def validate_epoch_stage3(
    model: AnyCamWrapperWithDA3Calibration,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Run validation for one epoch and return average loss."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(dataloader):
            batch_data = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                         for k, v in batch_data.items()}
            try:
                with autocast(device_type='cuda', dtype=torch.float16):
                    output_data = model(batch_data)
                    loss_dict = criterion(output_data)
                    loss = loss_dict.get('loss', loss_dict.get('total_loss', sum(loss_dict.values())))
                
                # Check for NaN or Inf
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"[WARN] Validation batch {batch_idx} produced NaN/Inf loss, skipping")
                    continue
                
                total_loss += loss.item()
                num_batches += 1
            except Exception as e:
                print(f"[WARN] Validation batch {batch_idx} failed: {e}")
                continue
    
    model.train()
    if num_batches == 0:
        print("[WARN] No valid validation batches processed, returning 0.0")
        return 0.0
    return total_loss / num_batches


def train_stage3(
    model: AnyCamWrapperWithDA3Calibration,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    num_epochs: int,
    device: torch.device,
    save_dir: Path,
    val_dataloader: Optional[DataLoader] = None,
    log_interval: int = 10,
):
    """
    Stage 3 training loop: End-to-end training with flow reprojection loss.
    
    Uses standalone DA3CalibrationHead with DINOv2-small visual tokens (vis_dim=384).
    This matches Stage 2's setup for consistent staged training.
    """
    model.train()
    
    # Freeze everything except calibration head
    for param in model.parameters():
        param.requires_grad = False
    
    # Unfreeze only the standalone calibration head (not pose_predictor's internal one)
    if hasattr(model, 'da3_calibration_head') and model.da3_calibration_head is not None:
        for param in model.da3_calibration_head.parameters():
            param.requires_grad = True
        print(f"[TRAIN] Standalone DA3 calibration head: UNFROZEN")
    else:
        raise ValueError("Standalone DA3 calibration head not found in model!")
    
    # Count trainable parameters
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[MODEL] Trainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    
    loss_history = []
    val_loss_history = []
    batch_losses = []
    
    log_file = save_dir / "training_log.txt"
    loss_json_path = save_dir / "loss_history.json"
    loss_plot_path = save_dir / "loss_curve.png"
    
    # Initialize log file
    with open(log_file, 'w') as f:
        f.write(f"DA3 Stage 3 Training Log\n")
        f.write(f"{'='*70}\n")
        f.write(f"Num epochs: {num_epochs}\n")
        f.write(f"Batch size: {dataloader.batch_size}\n")
        f.write(f"Validation: {'Enabled' if val_dataloader else 'Disabled'}\n")
        f.write(f"Device: {device}\n")
        f.write(f"{'='*70}\n\n")
    
    print(f"\n{'='*70}")
    print(f"[TRAIN] Starting Stage 3 training for {num_epochs} epochs")
    print(f"[TRAIN] Loss: Flow reprojection (self-supervised)")
    if val_dataloader:
        print(f"[TRAIN] Validation: Enabled ({len(val_dataloader)} batches)")
    print(f"{'='*70}\n")
    
    scaler = GradScaler('cuda')
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        
        for batch_idx, batch_data in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            # Move data to device
            batch_data = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                         for k, v in batch_data.items()}
            
            # Forward pass
            optimizer.zero_grad()
            
            try:
                with autocast(device_type='cuda', dtype=torch.float16):
                    output_data = model(batch_data)
                    
                    # Compute loss (flow reprojection loss)
                    loss_dict = criterion(output_data)
                    loss = loss_dict.get('loss', loss_dict.get('total_loss', sum(loss_dict.values())))
                
                # Check for NaN or Inf
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"[WARN] Batch {batch_idx} produced NaN/Inf loss, skipping")
                    torch.cuda.empty_cache()
                    continue
                
                # Backward pass
                scaler.scale(loss).backward()
                
                # Check gradients
                if batch_idx % 100 == 0:
                    total_grad_norm = 0.0
                    for name, param in model.named_parameters():
                        if param.requires_grad and param.grad is not None:
                            param_grad_norm = param.grad.data.norm(2)
                            total_grad_norm += param_grad_norm.item() ** 2
                    total_grad_norm = total_grad_norm ** (1. / 2)
                    if total_grad_norm > 0:
                        print(f"[GRAD] Gradient norm: {total_grad_norm:.6f}")
                
                scaler.step(optimizer)
                scaler.update()
                
                epoch_loss += loss.item()
                batch_losses.append(loss.item())
                
                if batch_idx % log_interval == 0:
                    log_msg = (f"[TRAIN] Epoch {epoch+1}/{num_epochs} | "
                              f"Batch {batch_idx}/{len(dataloader)} | "
                              f"Loss: {loss.item():.6f}")
                    print(log_msg)
                    with open(log_file, 'a') as f:
                        f.write(f"{log_msg}\n")
                        
            except Exception as e:
                print(f"[ERROR] Batch {batch_idx} failed: {e}")
                import traceback
                traceback.print_exc()
                # Clear GPU cache on error to recover memory
                torch.cuda.empty_cache()
                continue
            
            # Periodic GPU cache clear to manage memory (24GB constraint)
            if batch_idx % 50 == 0 and batch_idx > 0:
                torch.cuda.empty_cache()
        
        # Clear cache at end of epoch
        torch.cuda.empty_cache()
        
        # Epoch summary
        avg_loss = epoch_loss / max(len(dataloader), 1)
        loss_history.append({'epoch': epoch + 1, 'loss': avg_loss})
        
        # Validation
        val_loss = None
        if val_dataloader:
            val_loss = validate_epoch_stage3(model, val_dataloader, criterion, device)
            if not (torch.isnan(torch.tensor(val_loss)) or torch.isinf(torch.tensor(val_loss))):
                val_loss_history.append({'epoch': epoch + 1, 'loss': val_loss})
            else:
                print(f"[WARN] Validation loss is NaN/Inf, not adding to history")
                val_loss = None
            log_msg = f"\n[EPOCH {epoch+1}] Train Loss: {avg_loss:.6f} | Val Loss: {val_loss if val_loss is not None else 'nan':.6f}\n"
        else:
            log_msg = f"\n[EPOCH {epoch+1}] Average Loss: {avg_loss:.6f}\n"
        
        print(log_msg)
        with open(log_file, 'a') as f:
            f.write(f"{log_msg}\n")
        
        # Save loss history and plot after each epoch (overwrite previous)
        with open(loss_json_path, 'w') as f:
            json.dump({
                'epoch_losses': loss_history,
                'val_epoch_losses': val_loss_history,
                'batch_losses': batch_losses,
            }, f, indent=2)
        
        # Plot loss curve after each epoch (overwrite previous)
        plot_loss_curve_stage3(loss_history, val_loss_history if val_loss_history else None, save_dir)
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0 or epoch == num_epochs - 1:
            checkpoint_path = save_dir / "checkpoints" / f"checkpoint_epoch_{epoch+1}.pt"
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'val_loss': val_loss,
                'loss_history': loss_history,
                'val_loss_history': val_loss_history,
            }, checkpoint_path)
            print(f"[SAVE] Checkpoint saved to {checkpoint_path}")
        
        # Also save latest checkpoint after each epoch
        latest_checkpoint_path = save_dir / "checkpoints" / "latest_checkpoint.pt"
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
            'val_loss': val_loss,
            'loss_history': loss_history,
            'val_loss_history': val_loss_history,
        }, latest_checkpoint_path)
    
    # Save final model
    final_model_path = save_dir / "checkpoints" / "final_model.pt"
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss_history[-1]['loss'],
        'val_loss': val_loss_history[-1]['loss'] if val_loss_history else None,
        'loss_history': loss_history,
        'val_loss_history': val_loss_history,
    }, final_model_path)
    print(f"[SAVE] Final model saved to {final_model_path}")
    
    # Final loss history and plot (already saved each epoch, but save one more time)
    with open(loss_json_path, 'w') as f:
        json.dump({
            'epoch_losses': loss_history,
            'val_epoch_losses': val_loss_history,
            'batch_losses': batch_losses,
        }, f, indent=2)
    
    # Final plot (already saved each epoch, but save one more time)
    plot_loss_curve_stage3(loss_history, val_loss_history if val_loss_history else None, save_dir)
    save_training_summary(loss_history, batch_losses, save_dir, val_loss_history if val_loss_history else None)
    
    return loss_history, val_loss_history


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="DA3 Stage 3 Training: End-to-End Flow Reprojection")
    
    # Dataset arguments
    parser.add_argument("--objectron_videos", type=str, default=get_objectron_videos(),
                       help="Objectron videos directory")
    parser.add_argument("--objectron_gt", type=str, default=None,
                       help="Objectron GT directory (optional, not used in self-supervised training)")
    parser.add_argument("--split_file", type=str, default="experiments/objectron_split.json",
                       help="Dataset split file")
    parser.add_argument("--num_frames", type=int, default=2,
                       help="Number of frames per sequence")
    
    # Training arguments
    parser.add_argument("--num_epochs", type=int, default=50,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=2,
                       help="Batch size (memory intensive)")
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                       help="Learning rate (very low for fine-tuning)")
    parser.add_argument("--device", type=str, default="cuda:0",
                       help="Device to use")
    
    # Stage 2 checkpoint
    parser.add_argument("--stage2_checkpoint", type=str,
                       default="experiments/da3_integration/stage2_training/checkpoints/final_model.pt",
                       help="Path to Stage 2 checkpoint")
    
    # Config file
    parser.add_argument("--config_file", type=str,
                       default="pretrained_models/anycam_seq8/training_config.yaml",
                       help="Path to training config file")
    
    # Output arguments
    parser.add_argument("--save_dir", type=str,
                       default="experiments/da3_integration/stage3_training",
                       help="Directory to save results")
    
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[INIT] Using device: {device}")
    
    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / "checkpoints").mkdir(exist_ok=True)
    
    # Load config
    import yaml
    config_path = Path(args.config_file)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        full_config = yaml.safe_load(f)
    
    pose_predictor_config = full_config['model']['pose_predictor']
    depth_predictor_config = full_config['model']['depth_predictor']
    
    # Add use_da3_calibration to pose predictor config
    pose_predictor_config['use_da3_calibration'] = True
    
    # Load dataset split
    if Path(args.split_file).exists():
        split_data = load_dataset_split(args.split_file)
        train_indices = split_data.get('train', split_data.get('train_indices', []))
        val_indices = split_data.get('val', split_data.get('val_indices', []))
    else:
        raise FileNotFoundError(f"Split file not found: {args.split_file}")
    
    # Initialize AnyCalib
    anycalib_inference = AnyCaLibBatchInference(device=device)
    
    # Create dataset (with all available frames, unlimited pairs)
    print(f"\n[STEP 1] Creating dataset...")
    train_dataset = ObjectronVideoDataset(
        videos_dir=args.objectron_videos,
        gt_dir=args.objectron_gt,
        num_frames=args.num_frames,
        video_indices=train_indices,
        require_gt=False,  # Self-supervised, no GT needed
        extract_all_pairs=True,  # Extract all consecutive pairs
        max_pairs_per_video=999999,  # Effectively unlimited (extract all pairs)
    )
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )
    
    # Create validation dataset if val_indices are available
    val_dataloader = None
    if val_indices:
        val_dataset = ObjectronVideoDataset(
            videos_dir=args.objectron_videos,
            gt_dir=args.objectron_gt,
            num_frames=args.num_frames,
            video_indices=val_indices,
            require_gt=False,
            extract_all_pairs=True,
            max_pairs_per_video=999999,  # Effectively unlimited (extract all pairs)
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
        )
        print(f"[DATASET] Loaded {len(train_dataset)} training, {len(val_dataset)} validation sequences")
    else:
        print(f"[DATASET] Loaded {len(train_dataset)} training sequences (no validation set)")
    
    # Create model with DA3 calibration head
    # Uses standalone DINOv2-small (vis_dim=384) to match Stage 2's training setup
    print(f"\n[STEP 2] Creating model with DA3 calibration head...")
    print(f"[MODEL] Using standalone DA3CalibrationHead with vis_dim=384 (DINOv2-small)")
    print(f"[MODEL] This matches Stage 2's setup for consistent staged training")
    
    model = AnyCamWrapperWithDA3Calibration(
        pose_predictor_config=pose_predictor_config,
        depth_predictor_config=depth_predictor_config,
        anycalib_model=anycalib_inference,
        use_da3_calibration=True,
        vis_dim=384,  # Match Stage 2's DINOv2-small output dimension
    ).to(device)
    
    # Load Stage 2 checkpoint (calibration head weights)
    # Stage 2 checkpoint contains the DA3CalibrationHead trained with vis_dim=384
    if Path(args.stage2_checkpoint).exists():
        print(f"[LOAD] Loading Stage 2 checkpoint from {args.stage2_checkpoint}")
        checkpoint = torch.load(args.stage2_checkpoint, map_location=device, weights_only=False)
        
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        
        # Load into our standalone da3_calibration_head (not pose_predictor's internal one)
        try:
            model.da3_calibration_head.load_state_dict(state_dict, strict=True)
            print(f"[LOAD] Stage 2 calibration head weights loaded successfully (strict=True)")
        except RuntimeError as e:
            # Try non-strict loading if there are minor mismatches
            print(f"[WARN] Strict loading failed: {e}")
            print(f"[LOAD] Attempting non-strict loading...")
            model.da3_calibration_head.load_state_dict(state_dict, strict=False)
            print(f"[LOAD] Stage 2 calibration head weights loaded (strict=False)")
    else:
        print(f"[WARN] Stage 2 checkpoint not found: {args.stage2_checkpoint}")
        print(f"[WARN] Starting from random initialization")
    
    # Clear GPU cache after loading
    torch.cuda.empty_cache()
    
    # Create loss function (flow reprojection loss)
    # The config may have 'loss' as a list of dicts or a single dict
    loss_config_raw = full_config.get('loss', None)
    
    if isinstance(loss_config_raw, list) and len(loss_config_raw) > 0:
        # Take the first loss config from the list
        loss_config = loss_config_raw[0]
    elif isinstance(loss_config_raw, dict):
        loss_config = loss_config_raw
    else:
        # Use default loss config
        loss_config = {
            "type": "pose_loss",
            "flow_criterion": "l1",
            "dist_criterion": "l1",
            "lambda_flow": 1.0,
            "lambda_dist": 0.0,  # Match the config
            "use_flow_uncertainty": True,
            "use_dist_uncertainty": True,
        }
    
    print(f"[LOSS] Using loss config: {loss_config.get('type', 'pose_loss')}")
    criterion = make_loss(loss_config)
    print(f"[LOSS] Flow reprojection loss configured")
    
    # Create optimizer (only for calibration head)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(trainable_params, lr=args.learning_rate)
    
    # Train
    print(f"\n[STEP 3] Starting training...")
    loss_history, val_loss_history = train_stage3(
        model=model,
        dataloader=train_dataloader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=args.num_epochs,
        device=device,
        save_dir=save_dir,
        val_dataloader=val_dataloader,
    )
    
    print(f"\n{'='*70}")
    print(f"[COMPLETE] Stage 3 training complete!")
    print(f"[COMPLETE] Results saved to: {save_dir}")
    if val_loss_history:
        print(f"[COMPLETE] Final validation loss: {val_loss_history[-1]['loss']:.6f}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()

