#!/usr/bin/env python3
"""
=============================================================================
DA3 Calibration Head Training - Stage 1: Mean Calibration Learning
=============================================================================

EXPERIMENT GOAL:
----------------
Train the DA3 calibration head to output mean calibration from per-frame
AnyCalib predictions, without using visual tokens.

STAGE 1 OBJECTIVE:
------------------
Learn to aggregate per-frame AnyCalib predictions into sequence-level mean
calibration. This stage trains:
- Camera Encoder: ✓ Trainable
- Visual-Camera Mixing: ✗ Frozen (not used)
- Sequence Aggregation: ✓ Trainable
- Camera Decoder: ✓ Trainable

LOSS FUNCTION:
--------------
L_stage1 = MSE(predicted_mean_calibration, gt_mean_calibration)

TRAINING DATA:
--------------
- Objectron dataset with all available frames
- extract_all_pairs=True, max_pairs_per_video=None (unlimited)
- Extract AnyCalib predictions for all frames
- Compute GT mean calibration per sequence

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
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from tqdm import tqdm

# Add project paths
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "anycalib"))

# DA3 imports
from experiments.models.da3_calibration_head import DA3CalibrationHead

# Dataset imports
from experiments.dataset_paths import (
    get_objectron_videos, get_objectron_gt, get_lightspeed_root
)
from experiments.train_pose_head_anycalib import (
    ObjectronVideoDataset,
    AnyCaLibBatchInference,
    load_dataset_split,
    create_train_val_test_split,
    save_dataset_split,
)

# AnyCalib import
try:
    from anycalib.model.anycalib_pretrained import AnyCalib
except ImportError:
    from anycalib.anycalib.model.anycalib_pretrained import AnyCalib

print("[INIT] Imports successful")


# =============================================================================
# DATASET FOR STAGE 1 TRAINING
# =============================================================================

class DA3Stage1Dataset(Dataset):
    """
    Dataset for Stage 1 training: Mean calibration learning.
    
    For each sequence:
    - Loads all frames
    - Runs AnyCalib on all frames to get per-frame predictions
    - Extracts GT calibration from ground truth
    - Computes GT mean calibration per sequence
    """
    def __init__(
        self,
        videos_dir: str,
        gt_dir: str,
        anycalib_model: AnyCaLibBatchInference,
        video_indices: Optional[List[int]] = None,
        image_size: Tuple[int, int] = (480, 640),
        require_gt: bool = True,
    ):
        self.videos_dir = Path(videos_dir)
        self.gt_dir = Path(gt_dir) if gt_dir else None
        self.anycalib_model = anycalib_model
        self.image_size = image_size
        self.require_gt = require_gt
        
        # Find all video files
        all_video_files = sorted(list(self.videos_dir.glob("*.MOV")) + 
                                 list(self.videos_dir.glob("*.mov")))
        
        if video_indices is not None:
            all_video_files = [all_video_files[i] for i in video_indices if i < len(all_video_files)]
        
        self.video_files = all_video_files
        print(f"[DATASET] Found {len(self.video_files)} video sequences")
        
        # Precompute data for all sequences
        print(f"[DATASET] Precomputing AnyCalib predictions and GT calibrations...")
        self.sequences = []
        
        for video_idx, video_path in enumerate(tqdm(self.video_files, desc="Preprocessing")):
            try:
                # Load frames
                frames, frame_indices = self._load_frames_from_video(video_path)
                
                # Require at least 2 frames for meaningful mean calibration
                if len(frames) < 2:
                    print(f"[WARN] Skipping {video_path.name}: only {len(frames)} frame(s)")
                    continue
                
                # Load GT calibration
                gt_calibrations = self._load_gt_calibration(video_path, frame_indices)
                if gt_calibrations is None and require_gt:
                    continue
                
                # Run AnyCalib on all frames
                anycalib_predictions = []
                for frame in frames:
                    # Convert to tensor format expected by AnyCalib
                    # AnyCalib expects [1, 3, H, W] in range [0, 1]
                    frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0  # [3, H, W]
                    frame_tensor = frame_tensor.unsqueeze(0).to(self.anycalib_model.device)  # [1, 3, H, W]
                    
                    # Run AnyCalib
                    try:
                        pred = self.anycalib_model.model.predict(
                            frame_tensor, cam_id="pinhole"
                        )
                        intrinsics = pred["intrinsics"][0]  # [4] - fx, fy, cx, cy
                        # Convert to numpy if needed
                        if isinstance(intrinsics, torch.Tensor):
                            intrinsics = intrinsics.cpu().numpy()
                    except Exception as e:
                        print(f"[WARN] AnyCalib failed on {video_path.name}: {e}")
                        # Use default intrinsics
                        H, W = frame.shape[:2]
                        intrinsics = np.array([W * 0.7, H * 0.7, W / 2, H / 2], dtype=np.float32)
                    
                    anycalib_predictions.append(intrinsics)
                
                anycalib_predictions = np.stack(anycalib_predictions)  # [N, 4]
                
                # Compute GT mean calibration
                if gt_calibrations is not None:
                    gt_mean = np.mean(gt_calibrations, axis=0)  # [4]
                else:
                    # Use AnyCalib mean as proxy
                    gt_mean = np.mean(anycalib_predictions, axis=0)
                
                # Store sequence data
                self.sequences.append({
                    'video_path': video_path,
                    'frames': frames,
                    'frame_indices': frame_indices,
                    'anycalib_predictions': anycalib_predictions,  # [N, 4]
                    'gt_mean_calibration': gt_mean,  # [4]
                    'image_size': (frames[0].shape[0], frames[0].shape[1]),  # (H, W)
                })
                
            except Exception as e:
                print(f"[WARN] Failed to process {video_path.name}: {e}")
                continue
        
        print(f"[DATASET] Preprocessed {len(self.sequences)} sequences")
    
    def _load_frames_from_video(self, video_path: Path, start_frame: int = 0) -> Tuple[List[np.ndarray], List[int]]:
        """Load ALL frames from video file for training.
        
        Note: For Stage 1 training, we load all available frames from each video
        to compute the best possible mean calibration estimate. This is different
        from Stage 3 where num_frames controls how many frames go into the pose
        predictor at once.
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        frames = []
        frame_indices = []
        
        # Seek to start frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        frame_idx = start_frame
        while True:  # Load ALL frames until video ends
            ret, frame = cap.read()
            if not ret:
                break
            
            # Resize if needed
            if frame.shape[:2] != self.image_size:
                frame = cv2.resize(frame, (self.image_size[1], self.image_size[0]))
            
            frames.append(frame)
            frame_indices.append(frame_idx)
            frame_idx += 1
        
        cap.release()
        return frames, frame_indices
    
    def _load_gt_calibration(self, video_path: Path, frame_indices: List[int]) -> Optional[np.ndarray]:
        """Load GT calibration from ground truth file."""
        if self.gt_dir is None:
            return None
        
        # Try to find GT file
        gt_path1 = self.gt_dir / f"{video_path.stem}.json"
        stem_without_video = video_path.stem.replace("_video", "")
        gt_path2 = self.gt_dir / f"{stem_without_video}.json"
        
        gt_path = gt_path1 if gt_path1.exists() else (gt_path2 if gt_path2.exists() else None)
        
        if gt_path is None:
            return None
        
        try:
            with open(gt_path, 'r') as f:
                gt_data = json.load(f)
            
            calibrations = []
            
            if 'frames' in gt_data:
                # Original format
                for frame_idx in frame_indices:
                    if frame_idx < len(gt_data['frames']):
                        frame_data = gt_data['frames'][frame_idx]
                        intrinsics = frame_data.get('intrinsics', None)
                        if intrinsics is None:
                            continue
                        fx, fy, cx, cy = intrinsics[:4]
                        calibrations.append([fx, fy, cx, cy])
            elif 'intrinsics_per_frame' in gt_data:
                # Objectron processed format: flattened 3x3 matrices [fx, 0, cx, 0, fy, cy, 0, 0, 1]
                intr_list = gt_data['intrinsics_per_frame']
                for frame_idx in frame_indices:
                    if frame_idx < len(intr_list):
                        K_flat = intr_list[frame_idx]  # [9] flattened 3x3
                        fx, fy, cx, cy = K_flat[0], K_flat[4], K_flat[2], K_flat[5]
                        calibrations.append([fx, fy, cx, cy])
            elif 'intrinsics' in gt_data:
                # Alternative processed format
                intr_list = gt_data['intrinsics']
                for frame_idx in frame_indices:
                    if frame_idx < len(intr_list):
                        K = np.array(intr_list[frame_idx], dtype=np.float32).reshape(3, 3)
                        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
                        calibrations.append([fx, fy, cx, cy])
            
            if len(calibrations) == 0:
                return None
            
            return np.array(calibrations, dtype=np.float32)  # [N, 4]
            
        except Exception as e:
            print(f"[WARN] Failed to load GT from {gt_path}: {e}")
            return None
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Dict:
        seq_data = self.sequences[idx]
        
        # Get AnyCalib predictions for all frames
        anycalib_preds = seq_data['anycalib_predictions']  # [N, 4]
        
        # Get GT mean calibration
        gt_mean = seq_data['gt_mean_calibration']  # [4]
        
        # Get image size
        image_size = seq_data['image_size']  # (H, W)
        
        return {
            'anycalib_predictions': torch.from_numpy(anycalib_preds).float(),  # [N, 4]
            'gt_mean_calibration': torch.from_numpy(gt_mean).float().unsqueeze(0),  # [1, 4]
            'image_size': image_size,
            'num_frames': len(anycalib_preds),
        }


def collate_variable_length(batch: List[Dict]) -> Dict:
    """
    Custom collate function for variable-length sequences.
    
    Pads anycalib_predictions to the max length in the batch and creates
    an attention mask to indicate valid positions.
    """
    # Get max sequence length in this batch
    max_len = max(item['num_frames'] for item in batch)
    
    batch_size = len(batch)
    
    # Pad anycalib_predictions
    padded_preds = torch.zeros(batch_size, max_len, 4)
    attention_mask = torch.zeros(batch_size, max_len)
    
    gt_means = []
    image_sizes_h = []
    image_sizes_w = []
    num_frames_list = []
    
    for i, item in enumerate(batch):
        n_frames = item['num_frames']
        padded_preds[i, :n_frames, :] = item['anycalib_predictions']
        attention_mask[i, :n_frames] = 1.0
        gt_means.append(item['gt_mean_calibration'])
        image_sizes_h.append(item['image_size'][0])
        image_sizes_w.append(item['image_size'][1])
        num_frames_list.append(n_frames)
    
    return {
        'anycalib_predictions': padded_preds,  # [B, max_N, 4]
        'attention_mask': attention_mask,  # [B, max_N] - 1 for valid, 0 for padding
        'gt_mean_calibration': torch.stack(gt_means, dim=0),  # [B, 1, 4]
        'image_size': (torch.tensor(image_sizes_h), torch.tensor(image_sizes_w)),
        'num_frames': torch.tensor(num_frames_list),
    }


# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================

def validate_epoch(
    model: DA3CalibrationHead,
    dataloader: DataLoader,
    device: torch.device,
) -> float:
    """
    Run validation for one epoch and return average loss.
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch_data in dataloader:
            anycalib_preds = batch_data['anycalib_predictions'].to(device)
            attention_mask = batch_data['attention_mask'].to(device)
            gt_mean = batch_data['gt_mean_calibration'].to(device)
            image_sizes = batch_data['image_size']
            
            B, N, _ = anycalib_preds.shape
            dummy_visual = torch.zeros(B, N, model.vis_dim, device=device)
            
            H = image_sizes[0][0].item() if isinstance(image_sizes[0], torch.Tensor) else image_sizes[0]
            W = image_sizes[1][0].item() if isinstance(image_sizes[1], torch.Tensor) else image_sizes[1]
            
            pred_calibration = model(
                visual_tokens=dummy_visual,
                anycalib_predictions=anycalib_preds,
                image_size=(H, W),
                use_visual_conditioning=False,
                attention_mask=attention_mask,
            )
            
            loss = F.mse_loss(pred_calibration, gt_mean)
            total_loss += loss.item()
            num_batches += 1
    
    model.train()
    return total_loss / max(num_batches, 1)


def train_stage1(
    model: DA3CalibrationHead,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    num_epochs: int,
    device: torch.device,
    save_dir: Path,
    val_dataloader: Optional[DataLoader] = None,
    log_interval: int = 10,
):
    """
    Stage 1 training loop: Train calibration head to output mean calibration.
    """
    model.train()
    
    # Freeze visual-camera mixing (not used in stage 1)
    for param in model.visual_camera_mixing.parameters():
        param.requires_grad = False
    
    loss_history = []
    val_loss_history = []
    batch_losses = []
    
    # Create log file
    log_file = save_dir / "training_log.txt"
    with open(log_file, 'w') as f:
        f.write(f"DA3 Stage 1 Training Log\n")
        f.write(f"{'='*70}\n")
        f.write(f"Num epochs: {num_epochs}\n")
        f.write(f"Batch size: {dataloader.batch_size}\n")
        f.write(f"Validation: {'Enabled' if val_dataloader else 'Disabled'}\n")
        f.write(f"Device: {device}\n")
        f.write(f"{'='*70}\n\n")
    
    print(f"\n{'='*70}")
    print(f"[TRAIN] Starting Stage 1 training for {num_epochs} epochs")
    print(f"[TRAIN] Batch size: {dataloader.batch_size}")
    print(f"[TRAIN] Total batches: {len(dataloader)}")
    if val_dataloader:
        print(f"[TRAIN] Validation: Enabled ({len(val_dataloader)} batches)")
    print(f"{'='*70}\n")
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        
        for batch_idx, batch_data in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            # Move data to device
            anycalib_preds = batch_data['anycalib_predictions'].to(device)  # [B, max_N, 4]
            attention_mask = batch_data['attention_mask'].to(device)  # [B, max_N]
            gt_mean = batch_data['gt_mean_calibration'].to(device)  # [B, 1, 4]
            image_sizes = batch_data['image_size']  # Tuple of tensors: (H_tensor, W_tensor)
            num_frames = batch_data['num_frames']  # Tensor of ints
            
            # Forward pass (without visual tokens - Stage 1)
            # Create dummy visual tokens (won't be used)
            B, N, _ = anycalib_preds.shape
            dummy_visual = torch.zeros(B, N, model.vis_dim, device=device)
            
            # Get image size for first item in batch (assuming same size)
            # DataLoader collates tuples as (tensor([H1,H2,...]), tensor([W1,W2,...]))
            H = image_sizes[0][0].item() if isinstance(image_sizes[0], torch.Tensor) else image_sizes[0]
            W = image_sizes[1][0].item() if isinstance(image_sizes[1], torch.Tensor) else image_sizes[1]
            
            pred_calibration = model(
                visual_tokens=dummy_visual,
                anycalib_predictions=anycalib_preds,
                image_size=(H, W),
                use_visual_conditioning=False,  # Skip visual mixing
                attention_mask=attention_mask,  # Handle variable-length sequences
            )  # [B, 1, 4]
            
            # Loss: MSE between predicted and GT mean calibration
            loss = F.mse_loss(pred_calibration, gt_mean)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            batch_losses.append(loss.item())
            
            # Logging
            if batch_idx % log_interval == 0:
                log_msg = (f"[TRAIN] Epoch {epoch+1}/{num_epochs} | "
                          f"Batch {batch_idx}/{len(dataloader)} | "
                          f"Loss: {loss.item():.6f}")
                print(log_msg)
                with open(log_file, 'a') as f:
                    f.write(f"{log_msg}\n")
        
        # Epoch summary
        avg_loss = epoch_loss / len(dataloader)
        loss_history.append({'epoch': epoch + 1, 'loss': avg_loss})
        
        # Validation
        val_loss = None
        if val_dataloader:
            val_loss = validate_epoch(model, val_dataloader, device)
            val_loss_history.append({'epoch': epoch + 1, 'loss': val_loss})
            log_msg = f"\n[EPOCH {epoch+1}] Train Loss: {avg_loss:.6f} | Val Loss: {val_loss:.6f}\n"
        else:
            log_msg = f"\n[EPOCH {epoch+1}] Average Loss: {avg_loss:.6f}\n"
        
        print(log_msg)
        with open(log_file, 'a') as f:
            f.write(f"{log_msg}\n")
        
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
    
    # Save loss history
    loss_json_path = save_dir / "loss_history.json"
    with open(loss_json_path, 'w') as f:
        json.dump({
            'epoch_losses': loss_history,
            'val_epoch_losses': val_loss_history,
            'batch_losses': batch_losses,
        }, f, indent=2)
    
    # Generate visualizations
    plot_loss_curve(loss_history, val_loss_history, save_dir)
    save_training_summary(loss_history, batch_losses, save_dir, val_loss_history)
    
    return loss_history, val_loss_history


def evaluate_stage1(
    model: DA3CalibrationHead,
    dataloader: DataLoader,
    device: torch.device,
    save_dir: Path,
    prefix: str = "",
):
    """
    Evaluate Stage 1 model: Compute relative error to target mean calibration.
    
    Args:
        prefix: Optional prefix for output files (e.g., "val_" for validation results)
    """
    model.eval()
    
    all_errors = {
        'fx': [],
        'fy': [],
        'cx': [],
        'cy': [],
    }
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch_data in tqdm(dataloader, desc="Evaluating"):
            anycalib_preds = batch_data['anycalib_predictions'].to(device)  # [B, max_N, 4]
            attention_mask = batch_data['attention_mask'].to(device)  # [B, max_N]
            gt_mean = batch_data['gt_mean_calibration'].to(device)  # [B, 1, 4]
            image_sizes = batch_data['image_size']
            
            B, N, _ = anycalib_preds.shape
            dummy_visual = torch.zeros(B, N, model.vis_dim, device=device)
            # DataLoader collates tuples as (tensor([H1,H2,...]), tensor([W1,W2,...]))
            H = image_sizes[0][0].item() if isinstance(image_sizes[0], torch.Tensor) else image_sizes[0]
            W = image_sizes[1][0].item() if isinstance(image_sizes[1], torch.Tensor) else image_sizes[1]
            
            pred_calibration = model(
                visual_tokens=dummy_visual,
                anycalib_predictions=anycalib_preds,
                image_size=(H, W),
                use_visual_conditioning=False,
                attention_mask=attention_mask,
            )  # [B, 1, 4]
            
            # Compute relative errors
            pred_np = pred_calibration.cpu().numpy()  # [B, 1, 4]
            gt_np = gt_mean.cpu().numpy()  # [B, 1, 4]
            
            for b in range(B):
                pred = pred_np[b, 0]  # [4]
                gt = gt_np[b, 0]  # [4]
                
                # Relative errors
                rel_err_fx = abs(pred[0] - gt[0]) / (gt[0] + 1e-8)
                rel_err_fy = abs(pred[1] - gt[1]) / (gt[1] + 1e-8)
                rel_err_cx = abs(pred[2] - gt[2]) / (W + 1e-8)  # Normalize by image width
                rel_err_cy = abs(pred[3] - gt[3]) / (H + 1e-8)  # Normalize by image height
                
                all_errors['fx'].append(float(rel_err_fx))
                all_errors['fy'].append(float(rel_err_fy))
                all_errors['cx'].append(float(rel_err_cx))
                all_errors['cy'].append(float(rel_err_cy))
                
                all_predictions.append(pred.tolist())
                all_targets.append(gt.tolist())
    
    # Compute statistics
    stats = {}
    for key, errors in all_errors.items():
        errors = np.array(errors)
        stats[key] = {
            'mean': float(np.mean(errors)),
            'median': float(np.median(errors)),
            'std': float(np.std(errors)),
            'p90': float(np.percentile(errors, 90)),
        }
    
    # Overall relative error
    all_errors_flat = np.concatenate([np.array(errors) for errors in all_errors.values()])
    overall_error = {
        'mean': float(np.mean(all_errors_flat)),
        'median': float(np.median(all_errors_flat)),
    }
    
    # Save results
    results = {
        'focal_length_fx': stats['fx'],
        'focal_length_fy': stats['fy'],
        'principal_point_cx': stats['cx'],
        'principal_point_cy': stats['cy'],
        'overall_relative_error': overall_error,
        'predictions': all_predictions,
        'targets': all_targets,
    }
    
    accuracy_path = save_dir / f"{prefix}calibration_accuracy.json"
    with open(accuracy_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    set_name = "Validation" if prefix == "val_" else "Training"
    print(f"\n[EVAL] {set_name} Calibration Accuracy Results:")
    print(f"  Focal Length fx: mean={stats['fx']['mean']:.4f}, median={stats['fx']['median']:.4f}")
    print(f"  Focal Length fy: mean={stats['fy']['mean']:.4f}, median={stats['fy']['median']:.4f}")
    print(f"  Principal Point cx: mean={stats['cx']['mean']:.4f}, median={stats['cx']['median']:.4f}")
    print(f"  Principal Point cy: mean={stats['cy']['mean']:.4f}, median={stats['cy']['median']:.4f}")
    print(f"  Overall Relative Error: mean={overall_error['mean']:.4f}, median={overall_error['median']:.4f}")
    print(f"\n[SAVE] Results saved to {accuracy_path}")
    
    return results


def plot_loss_curve(loss_history: List[Dict], val_loss_history: List[Dict], save_dir: Path):
    """Plot and save training and validation loss curves."""
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
    plt.ylabel('Loss (MSE)', fontsize=12)
    plt.title('DA3 Stage 1 Training Loss', fontsize=14, fontweight='bold')
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


def save_training_summary(loss_history: List[Dict], batch_losses: List[float], save_dir: Path, 
                          val_loss_history: Optional[List[Dict]] = None):
    """Save training summary statistics."""
    if not loss_history:
        return
    
    losses = [item['loss'] for item in loss_history]
    val_losses = [item['loss'] for item in val_loss_history] if val_loss_history else []
    summary_path = save_dir / "training_summary.txt"
    
    with open(summary_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("DA3 STAGE 1 TRAINING SUMMARY\n")
        f.write("=" * 70 + "\n\n")
        
        f.write(f"Total Epochs: {len(loss_history)}\n")
        f.write(f"Total Batches Processed: {len(batch_losses)}\n")
        f.write(f"Validation: {'Enabled' if val_loss_history else 'Disabled'}\n\n")
        
        f.write(f"Initial Train Loss (Epoch 1): {losses[0]:.6f}\n")
        f.write(f"Final Train Loss (Epoch {len(losses)}): {losses[-1]:.6f}\n")
        f.write(f"Best Train Loss: {min(losses):.6f} (Epoch {losses.index(min(losses)) + 1})\n")
        f.write(f"Worst Train Loss: {max(losses):.6f} (Epoch {losses.index(max(losses)) + 1})\n\n")
        
        if val_losses:
            f.write(f"Initial Val Loss (Epoch 1): {val_losses[0]:.6f}\n")
            f.write(f"Final Val Loss (Epoch {len(val_losses)}): {val_losses[-1]:.6f}\n")
            f.write(f"Best Val Loss: {min(val_losses):.6f} (Epoch {val_losses.index(min(val_losses)) + 1})\n\n")
            
            # Overfitting detection
            if len(val_losses) > 5:
                recent_train = sum(losses[-5:]) / 5
                recent_val = sum(val_losses[-5:]) / 5
                gap = recent_val - recent_train
                f.write(f"Train-Val Gap (last 5 epochs): {gap:.6f}\n")
                if recent_val > 1.5 * recent_train:
                    f.write("WARNING: Potential overfitting detected!\n\n")
                else:
                    f.write("No significant overfitting detected.\n\n")
        
        improvement = losses[0] - losses[-1]
        improvement_pct = (improvement / abs(losses[0])) * 100 if losses[0] != 0 else 0
        f.write(f"Total Training Improvement: {improvement:.6f} ({improvement_pct:.2f}%)\n\n")
        
        f.write("Epoch-by-Epoch Progress:\n")
        f.write("-" * 70 + "\n")
        for i, item in enumerate(loss_history):
            if val_loss_history and i < len(val_loss_history):
                f.write(f"Epoch {item['epoch']:3d}: Train = {item['loss']:.6f} | Val = {val_loss_history[i]['loss']:.6f}\n")
            else:
                f.write(f"Epoch {item['epoch']:3d}: Train = {item['loss']:.6f}\n")
    
    print(f"[SAVE] Training summary saved: {summary_path}")


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="DA3 Stage 1 Training: Mean Calibration Learning")
    
    # Dataset arguments
    parser.add_argument("--objectron_videos", type=str, default=get_objectron_videos(),
                       help="Objectron videos directory")
    parser.add_argument("--objectron_gt", type=str, default=get_objectron_gt(),
                       help="Objectron GT directory")
    parser.add_argument("--split_file", type=str, default="experiments/objectron_split.json",
                       help="Dataset split file")
    # Note: Stage 1 always uses ALL frames from each video for training
    # (no --num_frames argument - that's only for Stage 3 pose prediction)
    
    # Training arguments
    parser.add_argument("--num_epochs", type=int, default=50,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                       help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda:0",
                       help="Device to use")
    
    # Model arguments
    parser.add_argument("--vis_dim", type=int, default=768,
                       help="Visual token dimension")
    parser.add_argument("--cam_dim", type=int, default=256,
                       help="Camera token dimension")
    parser.add_argument("--hidden_dim", type=int, default=128,
                       help="Hidden layer dimension")
    
    # Output arguments
    parser.add_argument("--save_dir", type=str,
                       default="experiments/da3_integration/stage1_training",
                       help="Directory to save results")
    
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[INIT] Using device: {device}")
    
    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / "checkpoints").mkdir(exist_ok=True)
    
    # Load dataset split
    if Path(args.split_file).exists():
        split_data = load_dataset_split(args.split_file)
        train_indices = split_data.get('train', split_data.get('train_indices', []))
        val_indices = split_data.get('val', split_data.get('val_indices', []))
    else:
        print(f"[WARN] Split file not found: {args.split_file}, creating new split...")
        # Create new split
        all_videos = sorted(list(Path(args.objectron_videos).glob("*.MOV")) + 
                           list(Path(args.objectron_videos).glob("*.mov")))
        train_indices, val_indices, test_indices = create_train_val_test_split(len(all_videos))
        split_data = {
            'train': train_indices,
            'val': val_indices,
            'test': test_indices,
        }
        save_dataset_split(split_data, args.split_file)
    
    # Initialize AnyCalib
    anycalib_inference = AnyCaLibBatchInference(device=device)
    
    # Create training dataset (loads ALL frames from each video)
    print(f"\n[STEP 1] Creating datasets (loading ALL frames per video)...")
    train_dataset = DA3Stage1Dataset(
        videos_dir=args.objectron_videos,
        gt_dir=args.objectron_gt,
        anycalib_model=anycalib_inference,
        video_indices=train_indices,
        require_gt=True,
    )
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        collate_fn=collate_variable_length,  # Handle variable-length sequences
    )
    
    # Create validation dataset if val_indices are available
    val_dataloader = None
    if val_indices:
        val_dataset = DA3Stage1Dataset(
            videos_dir=args.objectron_videos,
            gt_dir=args.objectron_gt,
            anycalib_model=anycalib_inference,
            video_indices=val_indices,
            require_gt=True,
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            collate_fn=collate_variable_length,  # Handle variable-length sequences
        )
        print(f"[DATASET] Loaded {len(train_dataset)} training, {len(val_dataset)} validation sequences")
    else:
        print(f"[DATASET] Loaded {len(train_dataset)} training sequences (no validation set)")
    
    # Create model
    print(f"\n[STEP 2] Creating DA3 calibration head...")
    model = DA3CalibrationHead(
        vis_dim=args.vis_dim,
        cam_dim=args.cam_dim,
        hidden_dim=args.hidden_dim,
        num_mixing_layers=2,
    ).to(device)
    
    # Freeze visual-camera mixing for Stage 1
    for param in model.visual_camera_mixing.parameters():
        param.requires_grad = False
    
    # Count parameters
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[MODEL] Trainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Train
    print(f"\n[STEP 3] Starting training...")
    loss_history, val_loss_history = train_stage1(
        model=model,
        dataloader=train_dataloader,
        optimizer=optimizer,
        num_epochs=args.num_epochs,
        device=device,
        save_dir=save_dir,
        val_dataloader=val_dataloader,
    )
    
    # Evaluate on training set
    print(f"\n[STEP 4] Evaluating model on training set...")
    evaluate_stage1(
        model=model,
        dataloader=train_dataloader,
        device=device,
        save_dir=save_dir,
    )
    
    # Evaluate on validation set if available
    if val_dataloader:
        print(f"\n[STEP 5] Evaluating model on validation set...")
        evaluate_stage1(
            model=model,
            dataloader=val_dataloader,
            device=device,
            save_dir=save_dir,
            prefix="val_",
        )
    
    print(f"\n{'='*70}")
    print(f"[COMPLETE] Stage 1 training complete!")
    print(f"[COMPLETE] Results saved to: {save_dir}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()

