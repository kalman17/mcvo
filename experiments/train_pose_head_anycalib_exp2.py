#!/usr/bin/env python3
"""
=============================================================================
Experiment 2: Multi-Frame Pose Prediction with Composed Consistency
=============================================================================

EXPERIMENT GOAL:
----------------
Extend Experiment 1 by adding multi-frame constraints to improve pose prediction 
accuracy, convergence speed, or quality. The goal is to leverage multiple frames 
in a sequence to enforce consistency across predicted poses, testing the hypothesis 
that composing consecutive poses and aligning them with long-range flows will 
enhance performance over consecutive-only reprojections.

KEY DIFFERENCES FROM EXPERIMENT 1:
-----------------------------------
1. Loads max_ahead + 1 frames per sequence (e.g., 4 frames for max_ahead=3)
2. Predicts poses for consecutive pairs (1→2, 2→3, 3→4)
3. Composes poses to get long-range predictions (1→3, 1→4)
4. Computes reprojection loss for both consecutive and composed poses
5. Enforces consistency across the entire sequence

APPROACH:
---------
- Data Preparation: Select video sequences with configurable number of future frames
- Model Setup: Use AnyCam with new pose head (trainable), all other components frozen
- Pose Composition: Mathematically compose consecutive poses to derive long-range poses
- Reprojection Loss: Compare predicted flows to observed flows for both consecutive and composed poses
- Training: Optimize pose head using total reprojection loss with multi-frame constraints

KEY FEATURES:
-------------
- Unsupervised Learning: No ground truth poses required
- Consistency Enforcement: Composed pose reprojection ensures coherent trajectory
- Flexibility: max_ahead parameter allows testing different ranges
- AnyCaLib Integration: Uses AnyCaLib focal length prediction (first frame only)

EXPECTED OUTCOME:
-----------------
This experiment tests whether composing consecutive poses and enforcing alignment 
with long-range flows leads to better pose predictions, faster convergence, or 
improved robustness compared to Experiment 1.

Author: AI Assistant
Date: October 23, 2025
"""

import os
import sys
import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings("ignore")

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np
import matplotlib.pyplot as plt
import cv2
import math
import os
import requests
from tqdm import tqdm
import torch.nn.functional as F

# AnyCam imports
from anycam.trainer import AnyCamWrapper, induce_flow_dist, normalize_proj, make_proj_from_focal_length
from anycam.models import make_pose_predictor, make_depth_predictor
from anycam.common.image_processor import make_image_processor
from anycam.loss import make_loss

# AnyCaLib imports
from anycalib.model.anycalib_pretrained import AnyCalib

# Local imports
from experiments.common.data_loader import FrameLoader, GroundTruthLoader, ExperimentDataManager
from experiments.train_pose_head_anycalib import (
    ObjectronVideoDataset, 
    AnyCaLibBatchInference,
    AnyCamWrapperWithAnyCaLib,
    create_train_val_test_split,
    save_dataset_split,
    load_dataset_split,
    plot_loss_curve,
    save_training_summary
)
from experiments.dataset_paths import (
    get_objectron_videos, get_objectron_gt, get_lightspeed_root
)

print("[INIT] Imports successful")

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def compose_poses(pose_list: List[torch.Tensor]) -> torch.Tensor:
    """
    Compose consecutive poses: pose_1->3 = pose_1->2 @ pose_2->3
    
    Args:
        pose_list: List of 4x4 transformation matrices [T_1->2, T_2->3, ...]
    
    Returns:
        composed: 4x4 transformation matrix representing cumulative transformation
    """
    if len(pose_list) == 0:
        raise ValueError("pose_list cannot be empty")
    
    composed = pose_list[0]
    for pose in pose_list[1:]:
        composed = composed @ pose
    
    return composed

def compose_flows(flow_list: List[torch.Tensor], occlusion_list: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compose consecutive flows by warping through intermediate frames.
    
    Args:
        flow_list: List of consecutive flows [B, 2, H, W]
        occlusion_list: List of occlusion masks [B, H, W]
    
    Returns:
        composed_flow: Composed long-range flow [B, 2, H, W]
        composed_occlusion: Composed occlusion mask [B, H, W]
    """
    if len(flow_list) == 0:
        raise ValueError("flow_list cannot be empty")
    
    batch_size, _, h, w = flow_list[0].shape
    device = flow_list[0].device
    
    # Start with first flow
    composed_flow = flow_list[0].clone()
    composed_occlusion = occlusion_list[0].clone()
    
    # Compose remaining flows
    for i in range(1, len(flow_list)):
        current_flow = flow_list[i]
        current_occlusion = occlusion_list[i]
        
        # Create coordinate grids for warping
        y_coords, x_coords = torch.meshgrid(
            torch.arange(h, device=device, dtype=torch.float32),
            torch.arange(w, device=device, dtype=torch.float32),
            indexing='ij'
        )
        
        # Warp coordinates by current flow
        warped_x = x_coords + current_flow[:, 0]  # [B, H, W]
        warped_y = y_coords + current_flow[:, 1]  # [B, H, W]
        
        # Check bounds
        valid_mask = (
            (warped_x >= 0) & (warped_x < w) & 
            (warped_y >= 0) & (warped_y < h)
        )
        
        # Interpolate previous flow at warped coordinates
        warped_flow_x = torch.zeros_like(composed_flow[:, 0])
        warped_flow_y = torch.zeros_like(composed_flow[:, 1])
        
        for b in range(batch_size):
            if valid_mask[b].any():
                # Use grid_sample for interpolation
                grid = torch.stack([
                    (warped_x[b] / (w - 1)) * 2 - 1,  # Normalize to [-1, 1]
                    (warped_y[b] / (h - 1)) * 2 - 1,
                ], dim=-1).unsqueeze(0)  # [1, H, W, 2]
                
                # Sample previous flow
                prev_flow_x = composed_flow[b:b+1, 0:1]  # [1, 1, H, W]
                prev_flow_y = composed_flow[b:b+1, 1:2]  # [1, 1, H, W]
                
                warped_flow_x[b] = torch.nn.functional.grid_sample(
                    prev_flow_x, grid, mode='bilinear', padding_mode='zeros', align_corners=True
                ).squeeze()
                warped_flow_y[b] = torch.nn.functional.grid_sample(
                    prev_flow_y, grid, mode='bilinear', padding_mode='zeros', align_corners=True
                ).squeeze()
        
        # Add current flow to warped previous flow
        composed_flow[:, 0] = warped_flow_x + current_flow[:, 0]
        composed_flow[:, 1] = warped_flow_y + current_flow[:, 1]
        
        # Update occlusion mask (AND operation - all intermediate flows must be valid)
        composed_occlusion = composed_occlusion * current_occlusion * valid_mask.float()
    
    return composed_flow, composed_occlusion

def extract_frame_pair(data: Dict, frame_idx1: int, frame_idx2: int) -> Dict:
    """
    Extract a pair of frames from multi-frame data.
    
    Args:
        data: Dictionary containing 'imgs' [B, F, 3, H, W] and other data
        frame_idx1: Index of first frame
        frame_idx2: Index of second frame
    
    Returns:
        pair_data: Dictionary with 2-frame data suitable for AnyCam
    """
    pair_data = {}
    
    # Extract 2 frames
    pair_data['imgs'] = data['imgs'][:, [frame_idx1, frame_idx2]]  # [B, 2, 3, H, W]
    
    # Extract corresponding projection matrices if available
    if 'projs' in data:
        pair_data['projs'] = data['projs'][:, [frame_idx1, frame_idx2]]  # [B, 2, 3, 3]
    
    # Copy other data
    for key in ['poses', 'sequence_name', 'frame_indices']:
        if key in data:
            pair_data[key] = data[key]
    
    return pair_data

# =============================================================================
# MULTI-FRAME DATASET
# =============================================================================

class ObjectronVideoDatasetMultiFrame(ObjectronVideoDataset):
    """
    Extended ObjectronVideoDataset that loads multiple consecutive frames
    for multi-frame pose prediction experiments.
    """
    
    def __init__(
        self,
        videos_dir: str,
        gt_dir: str,
        num_frames: int = 4,  # Changed default to 4 for max_ahead=3
        video_indices: Optional[List[int]] = None,
        require_gt: bool = False,
        extract_all_pairs: bool = False,
        max_pairs_per_video: int = 5,
        max_ahead: int = 3,  # NEW: Number of frames ahead to predict
    ):
        """
        Args:
            max_ahead: Number of frames ahead to predict (default: 3)
                      For max_ahead=3, loads frames [0,1,2,3] and predicts 0→1, 1→2, 2→3, 0→3, 0→2
        """
        self.max_ahead = max_ahead
        # Load max_ahead + 1 frames total
        super().__init__(
            videos_dir=videos_dir,
            gt_dir=gt_dir,
            num_frames=num_frames,
            video_indices=video_indices,
            require_gt=require_gt,
            extract_all_pairs=extract_all_pairs,
            max_pairs_per_video=max_pairs_per_video,
        )
        
        print(f"[DATASET] Multi-frame mode: max_ahead={max_ahead}, loading {max_ahead + 1} frames per sequence")
    
    def _build_pair_index(self):
        """
        Build index of frame sequences for multi-frame training.
        For max_ahead=3, creates sequences like [0,1,2,3], [4,5,6,7], etc.
        """
        self.pair_info = []
        
        for video_idx in self.video_indices:
            video_path = self.video_files[video_idx]
            
            # Get total frames safely
            cap = cv2.VideoCapture(str(video_path))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            
            # Safety margin to avoid corrupted frames at the end
            safe_total_frames = max(0, total_frames - 2)
            
            if safe_total_frames < self.max_ahead + 1:
                print(f"[WARN] Video {video_path.name} has only {safe_total_frames} frames, skipping")
                continue
            
            # Create overlapping sequences to use all available frames
            # For max_ahead=3: sequences start at 0, 1, 2, 3, 4, 5, ... (step_size=1)
            step_size = 1  # Use all possible sequences (changed from 10 to maximize data usage)
            for start_frame in range(0, safe_total_frames - self.max_ahead, step_size):
                frame_indices = list(range(start_frame, start_frame + self.max_ahead + 1))
                
                self.pair_info.append({
                    'video_idx': video_idx,
                    'start_frame': start_frame,
                    'frame_indices': frame_indices,
                    'video_path': video_path,
                })
        
        print(f"[DATASET] Built multi-frame index: {len(self.pair_info)} sequences from {len(self.video_indices)} videos")
        print(f"[DATASET] Each sequence contains {self.max_ahead + 1} frames (max_ahead={self.max_ahead})")

# =============================================================================
# MULTI-FRAME WRAPPER
# =============================================================================

class AnyCamWrapperMultiFrame(AnyCamWrapperWithAnyCaLib):
    """
    Extended wrapper that handles multi-frame pose prediction with composition.
    
    This wrapper:
    1. Runs AnyCam on consecutive frame pairs (1→2, 2→3, 3→4)
    2. Composes poses to get long-range predictions (1→3, 1→4)
    3. Returns both consecutive and composed results for loss computation
    """
    
    def __init__(
        self,
        pose_predictor_config: Dict,
        depth_predictor_config: Dict,
        anycalib_model: AnyCaLibBatchInference,
        max_ahead: int = 3,
        use_provided_depth: bool = False,
        use_provided_flow: bool = False,
    ):
        super().__init__(
            pose_predictor_config=pose_predictor_config,
            depth_predictor_config=depth_predictor_config,
            anycalib_model=anycalib_model,
            use_provided_depth=use_provided_depth,
            use_provided_flow=use_provided_flow,
        )
        
        self.max_ahead = max_ahead
        print(f"[WRAPPER] Multi-frame mode: max_ahead={max_ahead}")
    
    def forward(self, data: Dict) -> Dict:
        """
        Multi-frame forward pass with pose composition.
        
        Args:
            data: Dictionary containing 'imgs' [B, max_ahead+1, 3, H, W]
        
        Returns:
            result: Dictionary containing:
                - consecutive_results: List of results for consecutive pairs
                - composed_poses: List of composed long-range poses
                - all intermediate data for loss computation
        """
        images = data["imgs"]  # [B, max_ahead+1, 3, H, W]
        batch_size, num_frames, c, h, w = images.shape
        
        # Handle 2-frame evaluation (LightSpeed dataset)
        if num_frames == 2:
            # Fall back to parent class behavior for 2-frame evaluation
            return super().forward(data)
        
        if num_frames != self.max_ahead + 1:
            raise ValueError(f"Expected {self.max_ahead + 1} frames, got {num_frames}")
        
        device = images.device
        
        # ===== STEP 1: RUN CONSECUTIVE PAIRS =====
        consecutive_results = []
        consecutive_flows = []
        consecutive_occlusions = []
        
        for i in range(self.max_ahead):
            # Extract pair (i, i+1)
            pair_data = extract_frame_pair(data, i, i + 1)
            
            # Run AnyCam on this pair
            result = super().forward(pair_data)
            consecutive_results.append(result)
            
            # Extract flows and occlusions for composition
            if 'pose_result' in result and 'flow_occs_in' in result['pose_result']:
                flow_occs = result['pose_result']['flow_occs_in']  # [B, 2, 3, H, W]
                flow = flow_occs[:, 0, :2]  # [B, 2, H, W] - forward flow
                occlusion = flow_occs[:, 0, 2]  # [B, H, W] - occlusion mask
                consecutive_flows.append(flow)
                consecutive_occlusions.append(occlusion)
        
        # ===== STEP 2: COMPOSE LONG-RANGE POSES =====
        composed_poses = []
        
        for ahead in range(2, self.max_ahead + 1):
            # Get consecutive poses for composition
            pose_list = []
            for j in range(ahead):
                pose = consecutive_results[j]['proc_poses'][:, 0]  # [B, 4, 4] - first frame pair
                pose_list.append(pose)
            
            # Compose poses
            composed_pose = compose_poses(pose_list)  # [B, 4, 4]
            composed_poses.append(composed_pose)
        
        # ===== STEP 3: COMPOSE LONG-RANGE FLOWS =====
        composed_flows = []
        composed_occlusions = []
        
        for ahead in range(2, self.max_ahead + 1):
            # Compose flows for this range
            range_flows = consecutive_flows[:ahead]
            range_occlusions = consecutive_occlusions[:ahead]
            
            if range_flows and range_occlusions:
                composed_flow, composed_occlusion = compose_flows(range_flows, range_occlusions)
                composed_flows.append(composed_flow)
                composed_occlusions.append(composed_occlusion)
        
        # ===== STEP 4: PREPARE RESULT =====
        result = {
            'consecutive_results': consecutive_results,
            'composed_poses': composed_poses,
            'consecutive_flows': consecutive_flows,
            'consecutive_occlusions': consecutive_occlusions,
            'composed_flows': composed_flows,
            'composed_occlusions': composed_occlusions,
            'max_ahead': self.max_ahead,
            'batch_size': batch_size,
            'device': device,
        }
        
        # Copy some data from the first consecutive result for compatibility
        if consecutive_results:
            first_result = consecutive_results[0]
            result.update({
                'images_ip': first_result.get('images_ip'),
                'pred_depths': first_result.get('pred_depths'),
                'pred_depths_list': first_result.get('pred_depths_list'),
                'z_near': first_result.get('z_near'),
                'z_far': first_result.get('z_far'),
            })
            
            # Add focal_length_probs for loss computation compatibility
            # Since we use single focal length from AnyCaLib, set all probabilities to 1.0
            if 'pose_result' in first_result:
                batch_size = first_result['pose_result']['induced_flow'].shape[0]
                result['pose_result'] = first_result['pose_result'].copy()
                result['pose_result']['focal_length_probs'] = torch.ones(batch_size, 1, device=device)
        
        return result

# =============================================================================
# MULTI-FRAME LOSS FUNCTION WITH UNIMATCH DIRECT FLOW
# =============================================================================

class MultiFramePoseLoss(nn.Module):
    """
    Loss function that combines consecutive and composed pose reprojection losses.
    
    This loss:
    1. Computes reprojection loss for consecutive pairs (1→2, 2→3, 3→4)
    2. Computes reprojection loss for composed long-range poses (1→3, 1→4)
    3. Uses UniMatch to compute direct GT flow for long-range pairs
    4. Combines all losses with equal weighting
    """
    
    def __init__(self, base_loss_config: Dict, max_ahead: int = 3, use_direct_flow: bool = True, use_composed_loss: bool = True):
        super().__init__()
        
        self.base_loss = make_loss(base_loss_config)
        self.max_ahead = max_ahead
        self.use_direct_flow = use_direct_flow
        self.use_composed_loss = use_composed_loss
        
        # Initialize UniMatch for direct flow computation
        if self.use_direct_flow:
            self.unimatch_model = self._load_unimatch_model()
        
        print(f"[LOSS] Multi-frame loss: max_ahead={max_ahead}")
        print(f"[LOSS] Direct flow computation: {use_direct_flow}")
        print(f"[LOSS] Will compute loss for {max_ahead} consecutive pairs + {max_ahead - 1} composed poses")
    
    def _load_unimatch_model(self):
        """Load UniMatch model for direct flow computation."""
        try:
            from unimatch.unimatch import UniMatch
        except ModuleNotFoundError:
            import sys
            repo_root = Path(__file__).resolve().parents[1]
            unimatch_root = repo_root / "unimatch"
            if str(unimatch_root) not in sys.path:
                sys.path.insert(0, str(unimatch_root))
            from unimatch.unimatch import UniMatch
        
        # Use the same checkpoint path as AnyCam
        ckpt_path = Path(os.environ.get("HOME", str(Path.home()))) / \
            ".cache/torch/checkpoints/gmflow-scale2-regrefine6-mixdata-train320x576-4e7b215d.pth"
        
        if not ckpt_path.exists():
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            import requests
            url = "https://s3.eu-central-1.amazonaws.com/avg-projects/unimatch/pretrained/gmflow-scale2-regrefine6-mixdata-train320x576-4e7b215d.pth"
            print(f"[UniMatch] Downloading pretrained weights to {ckpt_path} ...")
            r = requests.get(url)
            r.raise_for_status()
            with open(ckpt_path, 'wb') as f:
                f.write(r.content)
            print(f"[UniMatch] Downloaded.")
        
        model = UniMatch(
            feature_channels=128,
            num_scales=2,
            upsample_factor=4,
            ffn_dim_expansion=4,
            num_transformer_layers=6,
            reg_refine=True,
            task='flow',
        ).eval()
        
        state = torch.load(str(ckpt_path), map_location='cpu')
        model.load_state_dict(state['model'], strict=False)
        
        print(f"[LOSS] UniMatch model loaded for direct flow computation")
        return model
    
    def _compute_direct_flow(self, img0: torch.Tensor, img1: torch.Tensor, device: torch.device) -> torch.Tensor:
        """
        Compute direct optical flow between two frames using UniMatch.
        
        Args:
            img0: First frame [B, 3, H, W]
            img1: Second frame [B, 3, H, W]
            device: Device to run on
        
        Returns:
            flow: Optical flow [B, 2, H, W]
        """
        # Move UniMatch to device
        self.unimatch_model = self.unimatch_model.to(device)
        
        # Preprocess images exactly like AnyCam does
        n, c, h, w = img0.shape
        max_size = 320
        smaller = min(h, w)
        
        if smaller > max_size:
            scale_factor = max_size / smaller
            target_h = h * scale_factor
            target_w = w * scale_factor
        else:
            target_h = h
            target_w = w
        
        target_h = math.ceil(target_h / 32) * 32
        target_w = math.ceil(target_w / 32) * 32
        
        if target_h != h or target_w != w:
            img0 = F.interpolate(img0, (target_h, target_w), mode='bilinear', align_corners=True)
            img1 = F.interpolate(img1, (target_h, target_w), mode='bilinear', align_corners=True)
        
        # Transform to UniMatch format: [0,1] -> [-0.5, 0.5] -> [0, 255]
        img0 = (img0 * 0.5 + 0.5) * 255
        img1 = (img1 * 0.5 + 0.5) * 255
        
        # Handle portrait orientation
        if target_h > target_w:
            img0 = img0.permute(0, 1, 3, 2)
            img1 = img1.permute(0, 1, 3, 2)
        
        with torch.no_grad():
            # UniMatch parameters (matching AnyCam's settings)
            attn_type = 'swin'
            attn_splits_list = [2, 8]
            corr_radius_list = [-1, 4]
            prop_radius_list = [-1, 1]
            num_reg_refine = 6
            
            # Run UniMatch
            results_dict = self.unimatch_model(
                img0, img1,
                attn_type=attn_type,
                attn_splits_list=attn_splits_list,
                corr_radius_list=corr_radius_list,
                prop_radius_list=prop_radius_list,
                num_reg_refine=num_reg_refine,
            )
            
            flow_pred = results_dict['flow_preds'][-1]  # Take the finest scale
            
            # Handle portrait orientation (undo permutation)
            if target_h > target_w:
                flow_pred = flow_pred.permute(0, 1, 3, 2)
            
            # Resize back to original size if needed
            if target_h != h or target_w != w:
                flow_pred = F.interpolate(flow_pred, (h, w), mode='bilinear', align_corners=True)
                # Scale flow vectors accordingly
                scale_h = h / target_h
                scale_w = w / target_w
                flow_pred[:, 0] *= scale_w  # x component
                flow_pred[:, 1] *= scale_h  # y component
        
        return flow_pred
    
    def forward(self, data: Dict, model_result: Dict) -> Tuple[torch.Tensor, Dict, Dict]:
        """
        Compute multi-frame pose loss.
        
        Args:
            data: Original input data
            model_result: Result from AnyCamWrapperMultiFrame (or parent class for 2-frame case)
        
        Returns:
            total_loss: Combined loss from all reprojections
            losses: Dictionary of individual loss components
            extra_data: Additional data for logging
        """
        # Handle 2-frame case (LightSpeed dataset) - model_result is from parent class
        if 'consecutive_results' not in model_result:
            # This is a 2-frame evaluation, use base loss directly
            loss_dict = self.base_loss(model_result)
            loss = loss_dict.get('loss', loss_dict.get('total_loss', sum(loss_dict.values())))
            
            losses = {
                'total': loss.item(),
                'consecutive_total': loss.item(),
                'composed_total': 0.0,
            }
            extra_data = {'consecutive_losses': [loss.item()], 'composed_losses': []}
            
            return loss, losses, extra_data
        
        # Multi-frame case (normal operation)
        consecutive_results = model_result['consecutive_results']
        composed_poses = model_result['composed_poses']
        images = data['imgs']  # [B, max_ahead+1, 3, H, W]
        device = images.device
        
        total_loss = 0.0
        losses = {}
        extra_data = {}
        
        # ===== LOSS FOR CONSECUTIVE PAIRS =====
        consecutive_losses = []
        
        for i, result in enumerate(consecutive_results):
            loss_dict = self.base_loss(result)
            loss = loss_dict.get('loss', loss_dict.get('total_loss', sum(loss_dict.values())))
            total_loss += loss
            consecutive_losses.append(loss.item())
            losses[f'consecutive_{i}to{i+1}'] = loss.item()
        
        losses['consecutive_total'] = sum(consecutive_losses)
        extra_data['consecutive_losses'] = consecutive_losses
        
        # ===== LOSS FOR COMPOSED POSES =====
        composed_losses = []
        
        if self.use_composed_loss:
            # Weight composed losses to balance with consecutive losses
            composed_weight = 0.1  # Reduce composed loss influence
            
            for ahead_idx, composed_pose in enumerate(composed_poses):
                ahead = ahead_idx + 2  # ahead = 2, 3, 4, ...
                
                if self.use_direct_flow:
                    # Use UniMatch to compute direct GT flow
                    reprojection_data = self._prepare_composed_reprojection_with_direct_flow(
                        data, composed_pose, 0, ahead, device
                    )
                else:
                    # Use composed flows from consecutive pairs
                    if ahead_idx < len(model_result.get('composed_flows', [])):
                        composed_flow = model_result['composed_flows'][ahead_idx]
                        composed_occlusion = model_result['composed_occlusions'][ahead_idx]
                        reprojection_data = self._prepare_composed_reprojection_with_composed_flow(
                            data, composed_pose, 0, ahead, composed_flow, composed_occlusion
                        )
                    else:
                        # Fallback to empty flow data
                        reprojection_data = self._prepare_composed_reprojection(
                            data, composed_pose, 0, ahead
                        )
                
                # Compute loss
                loss_dict = self.base_loss(reprojection_data)
                loss = loss_dict.get('loss', loss_dict.get('total_loss', sum(loss_dict.values())))
                weighted_loss = loss * composed_weight  # Apply weighting
                total_loss += weighted_loss
                composed_losses.append(loss.item())
                losses[f'composed_0to{ahead}'] = loss.item()
                losses[f'composed_0to{ahead}_weighted'] = weighted_loss.item()
        
        losses['composed_total'] = sum(composed_losses)
        extra_data['composed_losses'] = composed_losses
        
        # ===== NORMALIZE LOSS =====
        num_losses = len(consecutive_losses) + len(composed_losses)
        total_loss = total_loss / num_losses
        
        losses['total'] = total_loss.item()
        extra_data['num_losses'] = num_losses
        
        return total_loss, losses, extra_data
    
    def _prepare_composed_reprojection_with_direct_flow(
        self, 
        data: Dict, 
        composed_pose: torch.Tensor, 
        frame_from: int, 
        frame_to: int,
        device: torch.device
    ) -> Dict:
        """
        Prepare data for reprojection loss computation using composed pose and direct UniMatch flow.
        
        Args:
            data: Original multi-frame data
            composed_pose: Composed transformation matrix [B, 4, 4]
            frame_from: Source frame index
            frame_to: Target frame index
            device: Device to run on
        
        Returns:
            reprojection_data: Data suitable for base loss computation
        """
        images = data['imgs']  # [B, F, 3, H, W]
        batch_size, num_frames, c, h, w = images.shape
        
        # Extract source and target frames
        source_frame = images[:, frame_from]  # [B, 3, H, W]
        target_frame = images[:, frame_to]    # [B, 3, H, W]
        
        # Compute direct flow using UniMatch
        direct_flow = self._compute_direct_flow(source_frame, target_frame, device)  # [B, 2, H, W]
        
        # Create 2-frame data
        reprojection_data = {
            'imgs': torch.stack([source_frame, target_frame], dim=1),  # [B, 2, 3, H, W]
        }
        
        # Copy other data
        for key in ['poses', 'sequence_name', 'frame_indices']:
            if key in data:
                reprojection_data[key] = data[key]
        
        # Create pose_result structure with proper shapes for AnyCam loss
        # The loss expects: [B, F, num_candidates, ...] where num_candidates = 1 for our case
        pose_result = {
            'poses': composed_pose.unsqueeze(1).unsqueeze(1),  # [B, 1, 1, 4, 4]
            'uncert': torch.ones(batch_size, 1, 1, 2, h, w, device=device),  # [B, F, 1, 2, H, W]
            'flow_occs_in': torch.zeros(batch_size, 2, 3, h, w, device=device),  # [B, F, 3, H, W] - flow + occlusion
            'aligned_depths': torch.ones(batch_size, 2, 1, 1, h, w, device=device),  # [B, F, 1, 1, H, W]
            'induced_flow': torch.zeros(batch_size, 2, 1, 2, h, w, device=device),  # [B, F, 1, 2, H, W]
            'dist': torch.zeros(batch_size, 2, 1, h, w, device=device),  # [B, F, 1, H, W]
            'proj_candidates': torch.eye(3, device=device).unsqueeze(0).unsqueeze(0),  # [B, 1, 3, 3]
            'focal_length_probs': torch.ones(batch_size, 1, device=device),  # [B, 1]
            'focal_length': torch.ones(batch_size, 1, device=device),  # [B, 1] - focal length
            'focal_length_candidates': torch.ones(batch_size, 1, device=device),  # [B, 1] - focal candidates
            'scaling_feature': torch.randn(batch_size, 1, 16, device=device),  # [B, 1, 16] - scaling feature
            'wd_pose_token_1': torch.randn(2, 128, device=device),  # [F, 128] - pose tokens
            'wd_pose_token_2': torch.randn(2, 1, 128, device=device),  # [F, 1, 128] - pose tokens
        }
        
        # Replace the flow_occs_in with our direct flow
        # The base loss expects flow_occs_in to have shape [B, 2, 3, H, W]
        # We need to put our direct flow in the right place
        pose_result['flow_occs_in'][:, 0, :2] = direct_flow  # Forward flow in first frame pair
        pose_result['flow_occs_in'][:, 0, 2] = 1.0  # Valid occlusion mask
        pose_result['flow_occs_in'][:, 1, :2] = 0.0  # No flow for second frame
        pose_result['flow_occs_in'][:, 1, 2] = 1.0   # Valid occlusion mask
        
        reprojection_data['pose_result'] = pose_result
        
        return reprojection_data
    
    def _prepare_composed_reprojection_with_composed_flow(
        self, 
        data: Dict, 
        composed_pose: torch.Tensor, 
        frame_from: int, 
        frame_to: int,
        composed_flow: torch.Tensor,
        composed_occlusion: torch.Tensor
    ) -> Dict:
        """
        Prepare data for reprojection loss computation using composed pose and composed flow.
        
        Args:
            data: Original multi-frame data
            composed_pose: Composed transformation matrix [B, 4, 4]
            frame_from: Source frame index
            frame_to: Target frame index
            composed_flow: Composed flow [B, 2, H, W]
            composed_occlusion: Composed occlusion mask [B, H, W]
        
        Returns:
            reprojection_data: Data suitable for base loss computation
        """
        images = data['imgs']  # [B, F, 3, H, W]
        batch_size, num_frames, c, h, w = images.shape
        
        # Extract source and target frames
        source_frame = images[:, frame_from]  # [B, 3, H, W]
        target_frame = images[:, frame_to]    # [B, 3, H, W]
        
        # Create 2-frame data
        reprojection_data = {
            'imgs': torch.stack([source_frame, target_frame], dim=1),  # [B, 2, 3, H, W]
        }
        
        # Copy other data
        for key in ['poses', 'sequence_name', 'frame_indices']:
            if key in data:
                reprojection_data[key] = data[key]
        
        # Create pose_result structure with proper shapes for AnyCam loss
        pose_result = {
            'poses': composed_pose.unsqueeze(1).unsqueeze(1),  # [B, 1, 1, 4, 4]
            'uncert': torch.ones(batch_size, 1, 1, 2, h, w, device=images.device),  # [B, F, 1, 2, H, W]
            'flow_occs_in': torch.zeros(batch_size, 2, 3, h, w, device=images.device),  # [B, F, 3, H, W]
            'aligned_depths': torch.ones(batch_size, 2, 1, 1, h, w, device=images.device),  # [B, F, 1, 1, H, W]
            'induced_flow': torch.zeros(batch_size, 2, 1, 2, h, w, device=images.device),  # [B, F, 1, 2, H, W]
            'dist': torch.zeros(batch_size, 2, 1, h, w, device=images.device),  # [B, F, 1, H, W]
            'proj_candidates': torch.eye(3, device=images.device).unsqueeze(0).unsqueeze(0),  # [B, 1, 3, 3]
            'focal_length_probs': torch.ones(batch_size, 1, device=images.device),  # [B, 1]
            'focal_length': torch.ones(batch_size, 1, device=images.device),  # [B, 1]
            'focal_length_candidates': torch.ones(batch_size, 1, device=images.device),  # [B, 1]
            'scaling_feature': torch.randn(batch_size, 1, 16, device=images.device),  # [B, 1, 16]
            'wd_pose_token_1': torch.randn(2, 128, device=images.device),  # [F, 128]
            'wd_pose_token_2': torch.randn(2, 1, 128, device=images.device),  # [F, 1, 128]
        }
        
        # Put composed flow in the right place
        pose_result['flow_occs_in'][:, 0, :2] = composed_flow  # Forward flow
        pose_result['flow_occs_in'][:, 0, 2] = composed_occlusion  # Occlusion mask
        pose_result['flow_occs_in'][:, 1, :2] = 0.0  # No flow for second frame
        pose_result['flow_occs_in'][:, 1, 2] = 1.0   # Valid occlusion mask
        
        reprojection_data['pose_result'] = pose_result
        
        return reprojection_data
    
    def _prepare_composed_reprojection(
        self, 
        data: Dict, 
        composed_pose: torch.Tensor, 
        frame_from: int, 
        frame_to: int
    ) -> Dict:
        """
        Prepare data for reprojection loss computation using composed pose (fallback method).
        
        Args:
            data: Original multi-frame data
            composed_pose: Composed transformation matrix [B, 4, 4]
            frame_from: Source frame index
            frame_to: Target frame index
        
        Returns:
            reprojection_data: Data suitable for base loss computation
        """
        # Extract frames
        images = data['imgs']  # [B, F, 3, H, W]
        source_frame = images[:, frame_from]  # [B, 3, H, W]
        target_frame = images[:, frame_to]    # [B, 3, H, W]
        
        # Create 2-frame data
        reprojection_data = {
            'imgs': torch.stack([source_frame, target_frame], dim=1),  # [B, 2, 3, H, W]
        }
        
        # Copy other data
        for key in ['poses', 'sequence_name', 'frame_indices']:
            if key in data:
                reprojection_data[key] = data[key]
        
        # Create a minimal pose_result structure
        batch_size = images.shape[0]
        device = images.device
        
        # Create minimal pose_result with proper shapes for AnyCam loss
        pose_result = {
            'poses': composed_pose.unsqueeze(1).unsqueeze(1),  # [B, 1, 1, 4, 4]
            'uncert': torch.ones(batch_size, 1, 1, 2, *images.shape[-2:], device=device),  # [B, F, 1, 2, H, W]
            'flow_occs_in': torch.zeros(batch_size, 2, 3, *images.shape[-2:], device=device),  # [B, F, 3, H, W]
            'aligned_depths': torch.ones(batch_size, 2, 1, 1, *images.shape[-2:], device=device),  # [B, F, 1, 1, H, W]
            'induced_flow': torch.zeros(batch_size, 2, 1, 2, *images.shape[-2:], device=device),  # [B, F, 1, 2, H, W]
            'dist': torch.zeros(batch_size, 2, 1, *images.shape[-2:], device=device),  # [B, F, 1, H, W]
            'proj_candidates': torch.eye(3, device=device).unsqueeze(0).unsqueeze(0),  # [B, 1, 3, 3]
            'focal_length_probs': torch.ones(batch_size, 1, device=device),  # [B, 1]
            'focal_length': torch.ones(batch_size, 1, device=device),  # [B, 1] - focal length
            'focal_length_candidates': torch.ones(batch_size, 1, device=device),  # [B, 1] - focal candidates
            'scaling_feature': torch.randn(batch_size, 1, 16, device=device),  # [B, 1, 16] - scaling feature
            'wd_pose_token_1': torch.randn(2, 128, device=device),  # [F, 128] - pose tokens
            'wd_pose_token_2': torch.randn(2, 1, 128, device=device),  # [F, 1, 128] - pose tokens
        }
        
        reprojection_data['pose_result'] = pose_result
        
        return reprojection_data

# =============================================================================
# VALIDATION FUNCTION
# =============================================================================

def evaluate_model_multiframe(
    model: AnyCamWrapperMultiFrame,
    dataloaders: Dict[str, DataLoader],
    criterion: MultiFramePoseLoss,
    device: torch.device,
    max_ahead: int,
    max_samples: int = 50,  # Limit validation samples for speed
) -> Dict[str, float]:
    """
    Evaluate model on multiple validation datasets.
    
    Args:
        model: Multi-frame AnyCam wrapper
        dataloaders: Dictionary of dataset_name -> DataLoader
        criterion: Multi-frame loss function
        device: Device to run on
        max_ahead: Number of frames ahead (for handling different datasets)
        max_samples: Maximum number of samples to evaluate per dataset (for speed)
    
    Returns:
        val_losses: Dictionary of dataset_name -> average loss
    """
    model.eval()
    val_losses = {}
    
    with torch.no_grad():
        for dataset_name, dataloader in dataloaders.items():
            total_loss = 0.0
            num_batches = 0
            
            total_dataset_samples = min(max_samples, len(dataloader))
            print(f"[EVAL] Evaluating on {dataset_name} ({total_dataset_samples} samples)...")
            
            for batch_idx, batch_data in enumerate(dataloader):
                if batch_idx >= max_samples:
                    break
                
                # Show progress every 10 samples
                if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == total_dataset_samples:
                    print(f"[EVAL] {dataset_name}: {batch_idx + 1}/{total_dataset_samples} samples...")
                
                try:
                    # Move data to device
                    batch_data = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                                 for k, v in batch_data.items()}
                    
                    # Forward pass
                    model_result = model(batch_data)
                    
                    # Compute loss
                    loss, losses, extra_data = criterion(batch_data, model_result)
                    total_loss += loss.item()
                    num_batches += 1
                    
                except Exception as e:
                    print(f"[EVAL] {dataset_name} batch failed: {e}")
                    continue
            
            if num_batches > 0:
                val_losses[dataset_name] = total_loss / num_batches
            else:
                val_losses[dataset_name] = float('inf')
    
    model.train()
    return val_losses

# =============================================================================
# TRAINING FUNCTION
# =============================================================================

def train_pose_head_multiframe(
    model: AnyCamWrapperMultiFrame,
    dataloader: DataLoader,
    criterion: MultiFramePoseLoss,
    optimizer: optim.Optimizer,
    num_epochs: int,
    device: torch.device,
    save_dir: Path,
    val_dataloaders: Optional[Dict[str, DataLoader]] = None,
) -> List[Dict]:
    """
    Train the multi-frame pose head.
    
    Args:
        model: Multi-frame AnyCam wrapper
        dataloader: Data loader for training
        criterion: Multi-frame loss function
        optimizer: Optimizer
        num_epochs: Number of training epochs
        device: Device to train on
        save_dir: Directory to save checkpoints
    
    Returns:
        loss_history: List of loss dictionaries for each epoch
    """
    model.train()
    loss_history = []
    
    print(f"\n{'='*70}")
    print(f"[TRAIN] Starting multi-frame training for {num_epochs} epochs")
    print(f"{'='*70}")
    
    # Get total batches for mid-epoch evaluation
    total_batches = len(dataloader)
    mid_epoch_batch = total_batches // 2
    
    for epoch in range(num_epochs):
        epoch_losses = []
        batch_losses = []
        val_losses_mid = {}
        val_losses_end = {}
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch_idx, batch_data in enumerate(progress_bar):
            # Move data to device
            batch_data = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                         for k, v in batch_data.items()}
            
            # Forward pass
            optimizer.zero_grad()
            
            try:
                model_result = model(batch_data)
                loss, losses, extra_data = criterion(batch_data, model_result)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                # Track losses
                epoch_losses.append(loss.item())
                batch_losses.append(losses)
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'consec': f"{losses.get('consecutive_total', 0):.4f}",
                    'composed': f"{losses.get('composed_total', 0):.4f}",
                })
                
            except Exception as e:
                print(f"[ERROR] Batch {batch_idx} failed: {e}")
                import traceback
                traceback.print_exc()
                continue
            
            # Evaluate validation sets at mid-epoch (50% progress)
            if val_dataloaders is not None and batch_idx == mid_epoch_batch:
                print(f"\n[EVAL] Mid-epoch evaluation at batch {batch_idx}/{total_batches}...")
                val_losses_mid = evaluate_model_multiframe(
                    model, val_dataloaders, criterion, device, model.max_ahead
                )
                for dataset_name, val_loss in val_losses_mid.items():
                    print(f"[EVAL] {dataset_name} (mid): {val_loss:.6f}")
        
        # Evaluate validation sets at end of epoch
        if val_dataloaders is not None:
            print(f"\n[EVAL] End-of-epoch evaluation...")
            val_losses_end = evaluate_model_multiframe(
                model, val_dataloaders, criterion, device, model.max_ahead
            )
            for dataset_name, val_loss in val_losses_end.items():
                print(f"[EVAL] {dataset_name} (end): {val_loss:.6f}")
        
        # Epoch summary
        avg_loss = np.mean(epoch_losses) if epoch_losses else float('inf')
        
        epoch_summary = {
            'epoch': epoch + 1,
            'loss': avg_loss,  # Use 'loss' key for compatibility with plot_loss_curve
            'avg_loss': avg_loss,
            'batch_losses': batch_losses,
        }
        
        # Add validation losses (use end-of-epoch values)
        for dataset_name, val_loss in val_losses_end.items():
            epoch_summary[f'val_{dataset_name}'] = val_loss
        
        # Also store mid-epoch values with different key
        for dataset_name, val_loss in val_losses_mid.items():
            epoch_summary[f'val_{dataset_name}_mid'] = val_loss
        
        loss_history.append(epoch_summary)
        
        print(f"[EPOCH {epoch+1}] Average loss: {avg_loss:.6f}")
        for dataset_name, val_loss in val_losses_end.items():
            print(f"[EPOCH {epoch+1}] {dataset_name} validation loss: {val_loss:.6f}")
        
        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            checkpoint_path = save_dir / f"checkpoint_epoch_{epoch+1}.pt"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss_history': loss_history,
            }, checkpoint_path)
            print(f"[CHECKPOINT] Saved to {checkpoint_path}")
    
    print(f"\n{'='*70}")
    print(f"[TRAIN] Multi-frame training complete!")
    print(f"{'='*70}")
    
    return loss_history

def plot_loss_curve_with_validation(loss_history: List[Dict], save_dir: Path):
    """
    Plot training and validation loss curves.
    """
    if not loss_history:
        print("[VIZ] No loss history to plot")
        return
    
    epochs = [item['epoch'] for item in loss_history]
    train_losses = [item['loss'] for item in loss_history]
    
    plt.figure(figsize=(12, 7))
    plt.plot(epochs, train_losses, 'b-', linewidth=2, label='Training Loss', marker='o', markersize=4)
    
    # Plot validation losses if available
    val_keys = []
    if loss_history and len(loss_history) > 0:
        val_keys = [key for key in loss_history[0].keys() if key.startswith('val_') and not key.endswith('_mid')]
    colors = ['g', 'r', 'm', 'c', 'y']
    
    for idx, val_key in enumerate(val_keys):
        val_losses = [item.get(val_key, None) for item in loss_history]
        # Filter out None values
        valid_epochs = [e for e, v in zip(epochs, val_losses) if v is not None]
        valid_losses = [v for v in val_losses if v is not None]
        
        if valid_losses:
            dataset_name = val_key.replace('val_', '').replace('_', ' ').title()
            color = colors[idx % len(colors)]
            plt.plot(valid_epochs, valid_losses, color=color, linewidth=2, 
                    label=f'Validation {dataset_name}', marker='s', markersize=4, linestyle='--')
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training and Validation Loss Over Time', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10, loc='best')
    
    # Add annotations for first and last training loss
    if len(train_losses) > 0:
        plt.annotate(f'Start: {train_losses[0]:.4f}', 
                    xy=(epochs[0], train_losses[0]), 
                    xytext=(10, 10), 
                    textcoords='offset points',
                    fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.5))
        plt.annotate(f'Final: {train_losses[-1]:.4f}', 
                    xy=(epochs[-1], train_losses[-1]), 
                    xytext=(-50, -20), 
                    textcoords='offset points',
                    fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.5))
    
    # Save plot
    plot_path = save_dir / "loss_curve.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"[VIZ] Loss curve with validation saved: {plot_path}")

# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Experiment 2: Multi-Frame Pose Prediction")
    
    # Dataset arguments
    parser.add_argument("--videos_dir", type=str,
                       default=get_objectron_videos(),
                       help="Directory containing Objectron videos")
    parser.add_argument("--gt_dir", type=str,
                       default=get_objectron_gt(),
                       help="Directory containing ground truth JSON files")
    parser.add_argument("--split_file", type=str,
                       default="experiments/objectron_split.json",
                       help="Path to dataset split file")
    
    # Model arguments
    parser.add_argument("--model_path", type=str,
                       default="pretrained_models/anycam_seq8/",
                       help="Path to pretrained AnyCam model")
    
    # Training arguments
    parser.add_argument("--num_epochs", type=int, default=10,
                       help="Number of training epochs (default: 10, reduced due to increased training data)")
    parser.add_argument("--batch_size", type=int, default=1,
                       help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-4,
                       help="Learning rate")
    
    # Multi-frame arguments
    parser.add_argument("--max_ahead", type=int, default=3,
                       help="Number of frames ahead to predict (default: 3)")
    parser.add_argument("--use_direct_flow", action="store_true",
                       help="Use UniMatch for direct GT flow computation (default: False, composed flows used by default)")
    parser.add_argument("--disable_composed_flow", action="store_true",
                       help="Disable composed flows and use direct UniMatch instead (default: composed flows enabled)")
    parser.add_argument("--disable_composed_loss", action="store_true",
                       help="Disable composed pose losses (only consecutive pairs)")
    
    # Experiment arguments
    parser.add_argument("--save_dir", type=str, default="experiments/pose_head_experiment_results/exp2_full_run",
                       help="Directory to save results")
    parser.add_argument("--max_sequences", type=int, default=None,
                       help="Maximum number of sequences to use (for testing)")
    parser.add_argument("--run_evaluation", action="store_true",
                       help="Run evaluation after training")
    parser.add_argument("--eval_dataset", type=str, choices=['objectron', 'lightspeed'], 
                       default='lightspeed',
                       help="Dataset to use for evaluation")
    parser.add_argument("--lightspeed_dir", type=str,
                       default=get_lightspeed_root(),
                       help="LightSpeed dataset directory")
    parser.add_argument("--eval_only", action="store_true",
                       help="Skip training, only run evaluation")
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] Using device: {device}")
    
    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"[SAVE] Results will be saved to: {save_dir}")
    
    # =============================================================================
    # Load Dataset Split
    # =============================================================================
    print(f"\n[STEP 1] Loading dataset split...")
    
    if Path(args.split_file).exists():
        split_data = load_dataset_split(args.split_file)
        print(f"[SPLIT] Loaded existing split: {len(split_data['train'])} train, "
              f"{len(split_data['val'])} val, {len(split_data['test'])} test")
    else:
        print(f"[SPLIT] Creating new split...")
        split_data = create_train_val_test_split(
            videos_dir=args.videos_dir,
            test_size=0.15,
            val_size=0.15,
            random_seed=42
        )
        save_dataset_split(split_data, args.split_file)
        print(f"[SPLIT] Saved split to {args.split_file}")
    
    # Limit sequences if requested
    if args.max_sequences:
        split_data['train'] = split_data['train'][:args.max_sequences]
        print(f"[LIMIT] Limited to {args.max_sequences} training sequences")
    
    # =============================================================================
    # Create Datasets
    # =============================================================================
    print(f"\n[STEP 2] Creating multi-frame datasets...")
    
    train_dataset = ObjectronVideoDatasetMultiFrame(
        videos_dir=args.videos_dir,
        gt_dir=args.gt_dir,
        num_frames=args.max_ahead + 1,
        video_indices=split_data['train'],
        require_gt=False,  # Unsupervised training
        extract_all_pairs=False,  # We handle multi-frame in the dataset
        max_ahead=args.max_ahead,
    )
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )
    
    print(f"[DATASET] Created training dataset with {len(train_dataset)} sequences")
    
    # =============================================================================
    # Create Validation Datasets
    # =============================================================================
    print(f"\n[STEP 2b] Creating validation datasets...")
    
    val_dataloaders = {}
    
    # Objectron test split
    try:
        objectron_test_dataset = ObjectronVideoDatasetMultiFrame(
            videos_dir=args.videos_dir,
            gt_dir=args.gt_dir,
            num_frames=args.max_ahead + 1,
            video_indices=split_data['test'],
            require_gt=False,  # For loss computation, GT not needed
            extract_all_pairs=False,
            max_ahead=args.max_ahead,
        )
        objectron_test_dataloader = DataLoader(
            objectron_test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
        )
        val_dataloaders['objectron_test'] = objectron_test_dataloader
        print(f"[VAL] Objectron test dataset: {len(objectron_test_dataset)} sequences")
    except Exception as e:
        print(f"[WARN] Could not create Objectron test dataset: {e}")
    
    # LightSpeed dataset
    try:
        from experiments.lightspeed_dataset import LightSpeedDataset
        lightspeed_dataset = LightSpeedDataset(
            lightspeed_dir=args.lightspeed_dir,
            num_frames=2,  # LightSpeed always uses 2 frames
            image_size=(480, 640),
            extract_all_pairs=True,
        )
        lightspeed_dataloader = DataLoader(
            lightspeed_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
        )
        val_dataloaders['lightspeed'] = lightspeed_dataloader
        print(f"[VAL] LightSpeed dataset: {len(lightspeed_dataset)} pairs")
    except Exception as e:
        print(f"[WARN] Could not create LightSpeed dataset: {e}")
    
    if not val_dataloaders:
        print(f"[WARN] No validation datasets available, training without validation")
        val_dataloaders = None
    
    # =============================================================================
    # Setup AnyCaLib
    # =============================================================================
    print(f"\n[STEP 3] Setting up AnyCaLib...")
    
    anycalib_inference = AnyCaLibBatchInference(device=device)
    print(f"[ANYCALIB] Ready for multi-frame inference")
    
    # =============================================================================
    # Load Pretrained AnyCam
    # =============================================================================
    print(f"\n[STEP 4] Loading pretrained AnyCam model...")
    
    # Load model checkpoint
    model_path = Path(args.model_path)
    checkpoint_file = model_path / "training_checkpoint_247500.pt"
    config_file = model_path / "training_config.yaml"
    
    if not checkpoint_file.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_file}")
    
    # Load config
    import yaml
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create multi-frame wrapper
    model = AnyCamWrapperMultiFrame(
        pose_predictor_config=config['model']['pose_predictor'],
        depth_predictor_config=config['model']['depth_predictor'],
        anycalib_model=anycalib_inference,
        max_ahead=args.max_ahead,
    )
    
    # Load pretrained weights
    checkpoint = torch.load(checkpoint_file, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model'], strict=False)
    model = model.to(device)
    
    print(f"[MODEL] Pretrained model loaded")
    
    # =============================================================================
    # Reinitialize Pose Head
    # =============================================================================
    print(f"\n[STEP 5] Reinitializing pose head...")
    model.reinitialize_pose_head()
    
    # =============================================================================
    # Freeze Layers
    # =============================================================================
    print(f"\n[STEP 6] Freezing layers...")
    model.freeze_except_pose_head()
    
    # =============================================================================
    # Setup Optimizer and Loss
    # =============================================================================
    print(f"\n[STEP 7] Setting up optimizer and loss...")
    
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr
    )
    
    # Use multi-frame loss
    loss_config = config['loss'][0].copy()
    loss_config['lambda_fwd_bwd_consistency'] = 0  # Disable for forward-only training
    loss_config['use_flow_uncertainty'] = False  # Disable uncertainty-based loss
    loss_config['use_dist_uncertainty'] = False  # Disable uncertainty-based loss
    
    # Choose flow method: direct UniMatch vs composed flows
    # Default is composed flows (disable_composed_flow=False means composed flows enabled)
    if args.disable_composed_flow:
        use_direct_flow = True
        print(f"[LOSS] Using direct UniMatch flows (composed flows disabled)")
    else:
        use_direct_flow = args.use_direct_flow  # Can still force direct flow even if composed is enabled
        if use_direct_flow:
            print(f"[LOSS] Using direct UniMatch flows (forced by flag)")
        else:
            print(f"[LOSS] Using composed flows from consecutive pairs (default)")
    
    criterion = MultiFramePoseLoss(
        loss_config, 
        max_ahead=args.max_ahead, 
        use_direct_flow=use_direct_flow,
        use_composed_loss=not args.disable_composed_loss  # Enable composed losses unless disabled
    )
    
    print(f"[SETUP] Optimizer and loss ready")
    
    # =============================================================================
    # Training
    # =============================================================================
    if not args.eval_only:
        print(f"\n[STEP 8] Starting multi-frame training...")
        
        loss_history = train_pose_head_multiframe(
            model=model,
            dataloader=train_dataloader,
            criterion=criterion,
            optimizer=optimizer,
            num_epochs=args.num_epochs,
            device=device,
            save_dir=save_dir,
            val_dataloaders=val_dataloaders,
        )
        
        # Save final model
        final_model_path = save_dir / "final_model.pt"
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': config,
            'loss_history': loss_history,
            'max_ahead': args.max_ahead,
        }, final_model_path)
        print(f"[FINAL] Model saved to {final_model_path}")
        
        # Plot loss curve with validation
        plot_loss_curve_with_validation(loss_history, save_dir)
        
        # Save training summary
        save_training_summary(loss_history, [], save_dir)  # Empty batch_losses for now
        
    else:
        print(f"[SKIP] Training skipped (eval_only mode)")
        # Load trained model
        final_model_path = save_dir / "final_model.pt"
        if final_model_path.exists():
            checkpoint = torch.load(final_model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"[LOAD] Loaded trained model from {final_model_path}")
        else:
            raise FileNotFoundError(f"No trained model found at {final_model_path}")
    
    # =============================================================================
    # Evaluation (if requested)
    # =============================================================================
    if args.run_evaluation:
        print(f"\n[STEP 9] Running evaluation...")
        
        if args.eval_dataset == 'lightspeed':
            from experiments.lightspeed_dataset import LightSpeedDataset
            
            test_dataset = LightSpeedDataset(
                lightspeed_dir=args.lightspeed_dir,
                num_frames=2,  # LightSpeed evaluation uses 2 frames
                image_size=(480, 640),
            )
        else:  # objectron
            test_dataset = ObjectronVideoDatasetMultiFrame(
                videos_dir=args.videos_dir,
                gt_dir=args.gt_dir,
                num_frames=args.max_ahead + 1,  # Match training frame count
                video_indices=split_data['test'],
                require_gt=True,
                extract_all_pairs=False,
                max_ahead=args.max_ahead,  # Use same max_ahead as training
            )
        
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
        )
        
        print(f"[EVAL] Evaluating on {len(test_dataset)} samples...")
        
        # Simple evaluation loop
        model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch_data in tqdm(test_dataloader, desc="Evaluating"):
                batch_data = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                             for k, v in batch_data.items()}
                
                try:
                    model_result = model(batch_data)
                    loss, losses, extra_data = criterion(batch_data, model_result)
                    total_loss += loss.item()
                    num_batches += 1
                except Exception as e:
                    print(f"[ERROR] Evaluation batch failed: {e}")
                    continue
        
        avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
        print(f"[EVAL] Average evaluation loss: {avg_loss:.6f}")
        
        # Save evaluation results
        eval_results = {
            'avg_loss': avg_loss,
            'num_batches': num_batches,
            'max_ahead': args.max_ahead,
        }
        
        eval_path = save_dir / "evaluation" / "evaluation_results.json"
        eval_path.parent.mkdir(exist_ok=True)
        
        with open(eval_path, 'w') as f:
            json.dump(eval_results, f, indent=2)
        
        print(f"[EVAL] Results saved to {eval_path}")
    
    print(f"\n{'='*70}")
    print(f"[COMPLETE] Experiment 2 finished successfully!")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()
