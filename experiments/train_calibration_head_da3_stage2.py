#!/usr/bin/env python3
"""
=============================================================================
DA3 Calibration Head Training - Stage 2: Visual-Conditioned Calibration
=============================================================================

EXPERIMENT GOAL:
----------------
Train the DA3 calibration head with visual tokens to output mean calibration.

STAGE 2 OBJECTIVE:
------------------
Learn to leverage visual features for improved calibration. This stage:
- Camera Encoder: ✓ Trainable (fine-tune)
- Visual-Camera Mixing: ✓ Trainable (unfreeze)
- Sequence Aggregation: ✓ Trainable (fine-tune)
- Camera Decoder: ✓ Trainable (fine-tune)

LOSS FUNCTION:
--------------
L_stage2 = MSE(predicted_mean_calibration, gt_mean_calibration)

TRAINING DATA:
--------------
- Same as Stage 1 + extract visual tokens from AnyCam backbone
- Visual tokens: Extract from pose_tokens before pose-specific processing

Author: AI Assistant for Kalman's Master's Thesis
Date: December 2025
"""

import sys
import os

# Disable xFormers to ensure compatibility with newer GPUs (RTX 5090, etc.)
# xFormers memory-efficient attention doesn't support compute capability > 9.0
os.environ["XFORMERS_DISABLED"] = "1"
os.environ["XFORMERS_MORE_DETAILS"] = "0"

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
)

# AnyCam imports (for visual token extraction)
# Note: We use standalone DINOv2 for visual tokens, not the AnyCam backbone
from anycam.common.image_processor import make_image_processor

# AnyCalib import
try:
    from anycalib.model.anycalib_pretrained import AnyCalib
except ImportError:
    from anycalib.anycalib.model.anycalib_pretrained import AnyCalib

print("[INIT] Imports successful")


# =============================================================================
# DATASET FOR STAGE 2 TRAINING
# =============================================================================

class DA3Stage2Dataset(Dataset):
    """
    Dataset for Stage 2 training: Visual-conditioned calibration.
    
    For each sequence:
    - Loads all frames
    - Runs AnyCalib on all frames
    - Extracts visual tokens from AnyCam backbone
    - Extracts GT calibration
    - Computes GT mean calibration
    """
    def __init__(
        self,
        videos_dir: str,
        gt_dir: str,
        anycalib_model: AnyCaLibBatchInference,
        video_indices: Optional[List[int]] = None,
        image_size: Tuple[int, int] = (480, 640),
        require_gt: bool = True,
        device: torch.device = torch.device("cuda:0"),
    ):
        self.videos_dir = Path(videos_dir)
        self.gt_dir = Path(gt_dir) if gt_dir else None
        self.anycalib_model = anycalib_model
        self.image_size = image_size
        self.require_gt = require_gt
        self.device = device
        
        # Load DINOv2 backbone for visual token extraction
        # Note: We use a standalone DINOv2 model (3-channel input) instead of the 
        # AnyCam backbone (which expects 6-channel RGB+depth input)
        # Using HuggingFace transformers version which uses native PyTorch attention
        # (compatible with all GPUs including RTX 5090)
        print("[DATASET] Loading DINOv2 backbone (HuggingFace) for visual token extraction...")
        from transformers import AutoModel, AutoImageProcessor
        self.visual_backbone = AutoModel.from_pretrained('facebook/dinov2-small').to(device).eval()
        self.visual_processor = AutoImageProcessor.from_pretrained('facebook/dinov2-small')
        self.vis_dim = 384  # DINOv2-S output dimension
        
        # Freeze backbone (we only extract features, no training)
        for param in self.visual_backbone.parameters():
            param.requires_grad = False
        
        # Find all video files
        all_video_files = sorted(list(self.videos_dir.glob("*.MOV")) + 
                                 list(self.videos_dir.glob("*.mov")))
        
        if video_indices is not None:
            all_video_files = [all_video_files[i] for i in video_indices if i < len(all_video_files)]
        
        self.video_files = all_video_files
        print(f"[DATASET] Found {len(self.video_files)} video sequences")
        
        # Precompute data for all sequences
        # Use chunked processing to avoid OOM on 24GB VRAM cards
        self.chunk_size = 16  # Process 16 frames at a time through heavy models
        print(f"[DATASET] Precomputing visual tokens, AnyCalib predictions, and GT calibrations...")
        print(f"[DATASET] Using chunk_size={self.chunk_size} for memory efficiency")
        self.sequences = []
        
        for video_idx, video_path in enumerate(tqdm(self.video_files, desc="Preprocessing")):
            try:
                # Load frames (keep on CPU initially)
                frames, frame_indices = self._load_frames_from_video(video_path)
                
                # Require at least 2 frames for meaningful mean calibration
                if len(frames) < 2:
                    print(f"[WARN] Skipping {video_path.name}: only {len(frames)} frame(s)")
                    continue
                
                num_frames = len(frames)
                
                # Process visual tokens in chunks to avoid OOM
                visual_tokens_chunks = []
                
                with torch.no_grad():
                    for chunk_start in range(0, num_frames, self.chunk_size):
                        chunk_end = min(chunk_start + self.chunk_size, num_frames)
                        chunk_frames = frames[chunk_start:chunk_end]
                        
                        # Convert chunk to tensor
                        chunk_tensor = torch.stack([
                            torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
                            for frame in chunk_frames
                        ]).unsqueeze(0).to(device)  # [1, chunk_size, 3, H, W]
                        
                        # Extract visual tokens for chunk (no depth needed)
                        chunk_visual_tokens = self._extract_visual_tokens(chunk_tensor)
                        visual_tokens_chunks.append(chunk_visual_tokens.cpu())
                        
                        # Clean up GPU memory
                        del chunk_tensor, chunk_visual_tokens
                        torch.cuda.empty_cache()
                
                # Concatenate all visual token chunks
                visual_tokens = torch.cat(visual_tokens_chunks, dim=1)  # [1, N, D_vis]
                del visual_tokens_chunks
                
                # Load GT calibration
                gt_calibrations = self._load_gt_calibration(video_path, frame_indices)
                if gt_calibrations is None and require_gt:
                    continue
                
                # Run AnyCalib on all frames (in chunks to manage memory)
                anycalib_predictions = []
                for chunk_start in range(0, num_frames, self.chunk_size):
                    chunk_end = min(chunk_start + self.chunk_size, num_frames)
                    
                    for frame in frames[chunk_start:chunk_end]:
                        # Convert to tensor format expected by AnyCalib
                        frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0  # [3, H, W]
                        frame_tensor = frame_tensor.unsqueeze(0).to(self.device)  # [1, 3, H, W]
                        
                        try:
                            pred = self.anycalib_model.model.predict(
                                frame_tensor, cam_id="pinhole"
                            )
                            intrinsics = pred["intrinsics"][0]  # [4]
                            # Convert to numpy if needed
                            if isinstance(intrinsics, torch.Tensor):
                                intrinsics = intrinsics.cpu().numpy()
                        except Exception as e:
                            H, W = frame.shape[:2]
                            intrinsics = np.array([W * 0.7, H * 0.7, W / 2, H / 2], dtype=np.float32)
                        
                        anycalib_predictions.append(intrinsics)
                        del frame_tensor
                    
                    # Clear cache after each AnyCalib chunk
                    torch.cuda.empty_cache()
                
                anycalib_predictions = np.stack(anycalib_predictions)  # [N, 4]
                
                # Compute GT mean calibration
                if gt_calibrations is not None:
                    gt_mean = np.mean(gt_calibrations, axis=0)  # [4]
                else:
                    gt_mean = np.mean(anycalib_predictions, axis=0)
                
                # Store sequence data (all on CPU)
                self.sequences.append({
                    'video_path': video_path,
                    'visual_tokens': visual_tokens,  # [1, N, D_vis] - already on CPU
                    'anycalib_predictions': anycalib_predictions,  # [N, 4]
                    'gt_mean_calibration': gt_mean,  # [4]
                    'image_size': (frames[0].shape[0], frames[0].shape[1]),  # (H, W)
                })
                
                # Clean up after each video
                del frames, visual_tokens
                torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"[WARN] Failed to process {video_path.name}: {e}")
                torch.cuda.empty_cache()  # Clean up on failure too
                continue
        
        print(f"[DATASET] Preprocessed {len(self.sequences)} sequences")
    
    def _extract_visual_tokens(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract visual tokens from DINOv2 backbone (HuggingFace version).
        
        Uses the CLS token from DINOv2 as the visual representation for each frame.
        
        Args:
            images: [n, f, c, h, w] tensor of images (RGB, values in [0, 1])
        
        Returns:
            visual_tokens: [n, f, D_vis] visual feature tokens
        """
        n, f, c, h, w = images.shape
        
        # Reshape to batch all frames together
        inputs = images.view(n * f, c, h, w)
        
        # Normalize for DINOv2 (ImageNet normalization)
        mean = torch.tensor([0.485, 0.456, 0.406], device=inputs.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=inputs.device).view(1, 3, 1, 1)
        inputs = (inputs - mean) / std
        
        # Resize to DINOv2 expected size (224x224 for HuggingFace version)
        if h != 224 or w != 224:
            inputs = F.interpolate(inputs, size=(224, 224), mode='bilinear', align_corners=False)
        
        # Extract CLS token from DINOv2 (HuggingFace uses native PyTorch attention)
        with torch.no_grad():
            outputs = self.visual_backbone(inputs)
            # HuggingFace DINOv2 returns last_hidden_state, extract CLS token (first token)
            cls_tokens = outputs.last_hidden_state[:, 0, :]  # [n*f, D_vis]
        
        # Reshape back to [n, f, D_vis]
        visual_tokens = cls_tokens.view(n, f, -1)
        
        return visual_tokens
    
    def _load_frames_from_video(self, video_path: Path, start_frame: int = 0) -> Tuple[List[np.ndarray], List[int]]:
        """Load ALL frames from video file for training.
        
        Note: For Stage 2 training, we load all available frames from each video
        to compute the best possible mean calibration estimate with visual tokens.
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        frames = []
        frame_indices = []
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        frame_idx = start_frame
        while True:  # Load ALL frames until video ends
            ret, frame = cap.read()
            if not ret:
                break
            
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
                intr_list = gt_data['intrinsics']
                for frame_idx in frame_indices:
                    if frame_idx < len(intr_list):
                        K = np.array(intr_list[frame_idx], dtype=np.float32).reshape(3, 3)
                        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
                        calibrations.append([fx, fy, cx, cy])
            
            if len(calibrations) == 0:
                return None
            
            return np.array(calibrations, dtype=np.float32)
            
        except Exception as e:
            return None
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Dict:
        seq_data = self.sequences[idx]
        
        return {
            'visual_tokens': seq_data['visual_tokens'].squeeze(0),  # [N, D_vis]
            'anycalib_predictions': torch.from_numpy(seq_data['anycalib_predictions']).float(),  # [N, 4]
            'gt_mean_calibration': torch.from_numpy(seq_data['gt_mean_calibration']).float().unsqueeze(0),  # [1, 4]
            'image_size': seq_data['image_size'],
        }

# =============================================================================
# COLLATE FUNCTION FOR VARIABLE-LENGTH SEQUENCES
# =============================================================================

def collate_variable_length_stage2(batch: List[Dict]) -> Dict:
    """
    Custom collate function for variable-length sequences in Stage 2.
    
    Pads visual_tokens and anycalib_predictions to the max length in the batch
    and creates an attention mask to indicate valid positions.
    """
    # Get max sequence length in this batch
    max_len = max(item['visual_tokens'].shape[0] for item in batch)
    
    batch_size = len(batch)
    vis_dim = batch[0]['visual_tokens'].shape[1]  # D_vis
    
    # Pad visual_tokens and anycalib_predictions
    padded_visual = torch.zeros(batch_size, max_len, vis_dim)
    padded_preds = torch.zeros(batch_size, max_len, 4)
    attention_mask = torch.zeros(batch_size, max_len)
    
    gt_means = []
    image_sizes_h = []
    image_sizes_w = []
    num_frames_list = []
    
    for i, item in enumerate(batch):
        n_frames = item['visual_tokens'].shape[0]
        padded_visual[i, :n_frames, :] = item['visual_tokens']
        padded_preds[i, :n_frames, :] = item['anycalib_predictions']
        attention_mask[i, :n_frames] = 1.0
        gt_means.append(item['gt_mean_calibration'])
        image_sizes_h.append(item['image_size'][0])
        image_sizes_w.append(item['image_size'][1])
        num_frames_list.append(n_frames)
    
    return {
        'visual_tokens': padded_visual,  # [B, max_N, D_vis]
        'anycalib_predictions': padded_preds,  # [B, max_N, 4]
        'attention_mask': attention_mask,  # [B, max_N] - 1 for valid, 0 for padding
        'gt_mean_calibration': torch.stack(gt_means, dim=0),  # [B, 1, 4]
        'image_size': (torch.tensor(image_sizes_h), torch.tensor(image_sizes_w)),
        'num_frames': torch.tensor(num_frames_list),
    }


# =============================================================================
# TRAINING FUNCTIONS (Similar to Stage 1, but with visual tokens)
# =============================================================================

def validate_epoch_stage2(
    model: DA3CalibrationHead,
    dataloader: DataLoader,
    device: torch.device,
) -> float:
    """Run validation for one epoch and return average loss."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch_data in dataloader:
            visual_tokens = batch_data['visual_tokens'].to(device)  # [B, max_N, D_vis]
            anycalib_preds = batch_data['anycalib_predictions'].to(device)  # [B, max_N, 4]
            attention_mask = batch_data['attention_mask'].to(device)  # [B, max_N]
            gt_mean = batch_data['gt_mean_calibration'].to(device)  # [B, 1, 4]
            image_sizes = batch_data['image_size']
            
            H = image_sizes[0][0].item() if isinstance(image_sizes[0], torch.Tensor) else image_sizes[0]
            W = image_sizes[1][0].item() if isinstance(image_sizes[1], torch.Tensor) else image_sizes[1]
            
            pred_calibration = model(
                visual_tokens=visual_tokens,
                anycalib_predictions=anycalib_preds,
                image_size=(H, W),
                use_visual_conditioning=True,
                attention_mask=attention_mask,
            )
            
            loss = F.mse_loss(pred_calibration, gt_mean)
            total_loss += loss.item()
            num_batches += 1
    
    model.train()
    return total_loss / max(num_batches, 1)


def train_stage2(
    model: DA3CalibrationHead,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    num_epochs: int,
    device: torch.device,
    save_dir: Path,
    val_dataloader: Optional[DataLoader] = None,
    log_interval: int = 10,
):
    """Stage 2 training loop: Train with visual tokens."""
    model.train()
    
    # Unfreeze visual-camera mixing for Stage 2
    for param in model.visual_camera_mixing.parameters():
        param.requires_grad = True
    
    loss_history = []
    val_loss_history = []
    batch_losses = []
    
    log_file = save_dir / "training_log.txt"
    with open(log_file, 'w') as f:
        f.write(f"DA3 Stage 2 Training Log\n")
        f.write(f"{'='*70}\n")
        f.write(f"Num epochs: {num_epochs}\n")
        f.write(f"Batch size: {dataloader.batch_size}\n")
        f.write(f"Validation: {'Enabled' if val_dataloader else 'Disabled'}\n")
        f.write(f"Device: {device}\n")
        f.write(f"{'='*70}\n\n")
    
    print(f"\n{'='*70}")
    print(f"[TRAIN] Starting Stage 2 training for {num_epochs} epochs")
    print(f"[TRAIN] Visual-camera mixing: UNFROZEN")
    if val_dataloader:
        print(f"[TRAIN] Validation: Enabled ({len(val_dataloader)} batches)")
    print(f"{'='*70}\n")
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        
        for batch_idx, batch_data in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            visual_tokens = batch_data['visual_tokens'].to(device)  # [B, max_N, D_vis]
            anycalib_preds = batch_data['anycalib_predictions'].to(device)  # [B, max_N, 4]
            attention_mask = batch_data['attention_mask'].to(device)  # [B, max_N]
            gt_mean = batch_data['gt_mean_calibration'].to(device)  # [B, 1, 4]
            image_sizes = batch_data['image_size']
            
            # DataLoader collates tuples as (tensor([H1,H2,...]), tensor([W1,W2,...]))
            H = image_sizes[0][0].item() if isinstance(image_sizes[0], torch.Tensor) else image_sizes[0]
            W = image_sizes[1][0].item() if isinstance(image_sizes[1], torch.Tensor) else image_sizes[1]
            
            # Forward pass (with visual tokens - Stage 2)
            pred_calibration = model(
                visual_tokens=visual_tokens,
                anycalib_predictions=anycalib_preds,
                image_size=(H, W),
                use_visual_conditioning=True,  # Enable visual conditioning
                attention_mask=attention_mask,  # Mask for padded positions
            )  # [B, 1, 4]
            
            # Loss: MSE between predicted and GT mean calibration
            loss = F.mse_loss(pred_calibration, gt_mean)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            batch_losses.append(loss.item())
            
            if batch_idx % log_interval == 0:
                log_msg = (f"[TRAIN] Epoch {epoch+1}/{num_epochs} | "
                          f"Batch {batch_idx}/{len(dataloader)} | "
                          f"Loss: {loss.item():.6f}")
                print(log_msg)
                with open(log_file, 'a') as f:
                    f.write(f"{log_msg}\n")
        
        avg_loss = epoch_loss / len(dataloader)
        loss_history.append({'epoch': epoch + 1, 'loss': avg_loss})
        
        # Validation
        val_loss = None
        if val_dataloader:
            val_loss = validate_epoch_stage2(model, val_dataloader, device)
            val_loss_history.append({'epoch': epoch + 1, 'loss': val_loss})
            log_msg = f"\n[EPOCH {epoch+1}] Train Loss: {avg_loss:.6f} | Val Loss: {val_loss:.6f}\n"
        else:
            log_msg = f"\n[EPOCH {epoch+1}] Average Loss: {avg_loss:.6f}\n"
        
        print(log_msg)
        with open(log_file, 'a') as f:
            f.write(f"{log_msg}\n")
        
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
    
    # Save loss history and generate visualizations
    loss_json_path = save_dir / "loss_history.json"
    with open(loss_json_path, 'w') as f:
        json.dump({
            'epoch_losses': loss_history,
            'val_epoch_losses': val_loss_history,
            'batch_losses': batch_losses,
        }, f, indent=2)
    
    # Import visualization functions from Stage 1
    from experiments.train_calibration_head_da3_stage1 import plot_loss_curve, save_training_summary
    plot_loss_curve(loss_history, val_loss_history, save_dir)
    save_training_summary(loss_history, batch_losses, save_dir, val_loss_history)
    
    return loss_history, val_loss_history


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="DA3 Stage 2 Training: Visual-Conditioned Calibration")
    
    # Dataset arguments
    parser.add_argument("--objectron_videos", type=str, default=get_objectron_videos(),
                       help="Objectron videos directory")
    parser.add_argument("--objectron_gt", type=str, default=get_objectron_gt(),
                       help="Objectron GT directory")
    parser.add_argument("--split_file", type=str, default="experiments/objectron_split.json",
                       help="Dataset split file")
    # Note: Stage 2 always uses ALL frames from each video for training
    # (no --num_frames argument - that's only for Stage 3 pose prediction)
    
    # Training arguments
    parser.add_argument("--num_epochs", type=int, default=50,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Batch size (smaller due to visual tokens)")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                       help="Learning rate (lower than Stage 1)")
    parser.add_argument("--device", type=str, default="cuda:0",
                       help="Device to use")
    
    # Model arguments
    parser.add_argument("--vis_dim", type=int, default=384,
                       help="Visual token dimension (384 for DINOv2-S)")
    parser.add_argument("--cam_dim", type=int, default=256,
                       help="Camera token dimension")
    parser.add_argument("--hidden_dim", type=int, default=128,
                       help="Hidden layer dimension")
    
    # Stage 1 checkpoint
    parser.add_argument("--stage1_checkpoint", type=str,
                       default="experiments/da3_integration/stage1_training/checkpoints/final_model.pt",
                       help="Path to Stage 1 checkpoint")
    
    # Output arguments
    parser.add_argument("--save_dir", type=str,
                       default="experiments/da3_integration/stage2_training",
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
        raise FileNotFoundError(f"Split file not found: {args.split_file}")
    
    # Initialize AnyCalib
    anycalib_inference = AnyCaLibBatchInference(device=device)
    
    # Create dataset
    # Create training dataset (loads ALL frames from each video)
    print(f"\n[STEP 1] Creating datasets (loading ALL frames per video)...")
    train_dataset = DA3Stage2Dataset(
        videos_dir=args.objectron_videos,
        gt_dir=args.objectron_gt,
        anycalib_model=anycalib_inference,
        video_indices=train_indices,
        require_gt=True,
        device=device,
    )
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        collate_fn=collate_variable_length_stage2,
    )
    
    # Create validation dataset if val_indices are available
    val_dataloader = None
    if val_indices:
        val_dataset = DA3Stage2Dataset(
            videos_dir=args.objectron_videos,
            gt_dir=args.objectron_gt,
            anycalib_model=anycalib_inference,
            video_indices=val_indices,
            require_gt=True,
            device=device,
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            collate_fn=collate_variable_length_stage2,
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
    
    # Load Stage 1 checkpoint
    if Path(args.stage1_checkpoint).exists():
        print(f"[LOAD] Loading Stage 1 checkpoint from {args.stage1_checkpoint}")
        checkpoint = torch.load(args.stage1_checkpoint, map_location=device, weights_only=False)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        
        # Filter out visual_camera_mixing weights (they weren't trained in Stage 1 anyway)
        # This handles dimension mismatch when Stage 1 used vis_dim=768 but Stage 2 uses vis_dim=384
        model_state = model.state_dict()
        filtered_state_dict = {}
        skipped_keys = []
        for k, v in state_dict.items():
            if k in model_state:
                if v.shape == model_state[k].shape:
                    filtered_state_dict[k] = v
                else:
                    skipped_keys.append(f"{k}: checkpoint {v.shape} vs model {model_state[k].shape}")
            else:
                skipped_keys.append(f"{k}: not in model")
        
        if skipped_keys:
            print(f"[LOAD] Skipping {len(skipped_keys)} mismatched keys (visual_camera_mixing was frozen in Stage 1):")
            for sk in skipped_keys:
                print(f"       - {sk}")
        
        model.load_state_dict(filtered_state_dict, strict=False)
        print(f"[LOAD] Stage 1 weights loaded ({len(filtered_state_dict)} keys)")
    else:
        print(f"[WARN] Stage 1 checkpoint not found: {args.stage1_checkpoint}")
        print(f"[WARN] Starting from random initialization")
    
    # Unfreeze visual-camera mixing for Stage 2
    for param in model.visual_camera_mixing.parameters():
        param.requires_grad = True
    
    # Count parameters
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[MODEL] Trainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Train
    print(f"\n[STEP 3] Starting training...")
    loss_history, val_loss_history = train_stage2(
        model=model,
        dataloader=train_dataloader,
        optimizer=optimizer,
        num_epochs=args.num_epochs,
        device=device,
        save_dir=save_dir,
        val_dataloader=val_dataloader,
    )
    
    print(f"\n{'='*70}")
    print(f"[COMPLETE] Stage 2 training complete!")
    print(f"[COMPLETE] Results saved to: {save_dir}")
    if val_loss_history:
        print(f"[COMPLETE] Final validation loss: {val_loss_history[-1]['loss']:.6f}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()

