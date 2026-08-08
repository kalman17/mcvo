"""
AnyCam Wrapper with FAT-Enhanced AnyCalib for Phase 3 Training.

This wrapper integrates FAT-enhanced AnyCalib into the full AnyCam pipeline
for end-to-end training with flow reprojection loss.

Architecture:
    1. FAT-Enhanced AnyCalib: Multi-frame calibration → intrinsics [B, 4]
    2. Extract focal length (fx) from intrinsics
    3. Depth Predictor (frozen): Predict depths
    4. Flow Processor (frozen): Compute flow and occlusion
    5. Pose Head (trainable): Predict poses using focal length
    6. Flow Reprojection Loss: Compare induced vs observed flow
"""

import torch
import torch.nn as nn
from typing import Dict, Optional
from anycam.models import make_pose_predictor, make_depth_predictor
from anycam.common.image_processor import make_image_processor
from anycam.trainer import induce_flow_dist, make_proj_from_focal_length

from experiments.models.anycalib_with_fat import AnyCalibWithMCT


class AnyCamWrapperWithMCTCalibration(nn.Module):
    """
    AnyCam wrapper using FAT-enhanced AnyCalib for focal length prediction.
    
    This integrates AnyCalibWithMCT into the AnyCam pipeline for end-to-end
    training with flow reprojection loss.
    
    Args:
        fat_model: AnyCalibWithMCT instance (with FAT aggregation)
        pose_predictor_config: Config dict for pose predictor
        depth_predictor_config: Config dict for depth predictor
        use_provided_depth: Use provided depths instead of predicting
        use_provided_flow: Use provided flow instead of computing
    """
    
    def __init__(
        self,
        fat_model: AnyCalibWithMCT,
        pose_predictor_config: Dict,
        depth_predictor_config: Dict,
        use_provided_depth: bool = False,
        use_provided_flow: bool = False,
    ):
        super().__init__()
        
        self.fat_model = fat_model
        self.use_provided_depth = use_provided_depth
        self.use_provided_flow = use_provided_flow
        
        # Load AnyCam components
        self.depth_predictor = make_depth_predictor(depth_predictor_config)
        # Our checkpoints condition the pose head on the FAT focal (input 128+8);
        # vanilla AnyCam uses focal_embed_dim=0.
        pose_predictor_config = dict(pose_predictor_config)
        pose_predictor_config.setdefault("focal_embed_dim", 8)
        self.pose_predictor = make_pose_predictor(pose_predictor_config)
        
        # Freeze depth predictor (it's just for preprocessing)
        for param in self.depth_predictor.parameters():
            param.requires_grad = False
        
        # Image processor for flow and occlusion
        self.image_processor = make_image_processor(
            {"type": "flow_occlusion"},
            flow_model="unimatch",
            use_provided_flow=self.use_provided_flow,
            pair_mode="sequential"
        )
        
        self.z_near = 0.1
        self.z_far = 10.0
        
        # Note: Components will be moved to device when model.to(device) is called
        # All components are registered as submodules, so .to(device) will move them recursively
        
        print(f"[WRAPPER] AnyCamWrapperWithMCTCalibration initialized")
        print(f"[WRAPPER] FAT model: {type(fat_model).__name__}")
        print(f"[WRAPPER] Depth predictor: frozen")
        print(f"[WRAPPER] Flow processor: frozen")
    
    def freeze_except_fat_and_pose(self):
        """
        Freeze all parameters except FAT and pose head.
        
        This is the KEY function for Phase 3 training:
        - DINOv2 backbone: FROZEN (in FAT model)
        - DPT decoder + Ray Head: FROZEN (in FAT model)
        - FAT aggregation: TRAINABLE
        - Depth predictor: FROZEN
        - Flow processor: FROZEN
        - Pose head: TRAINABLE
        """
        print(f"\n{'='*70}")
        print(f"[FREEZE] Freezing all layers except FAT and pose_head...")
        print(f"{'='*70}")
        
        # First, freeze everything
        for name, param in self.named_parameters():
            param.requires_grad = False
        
        # Unfreeze FAT components (if FAT is enabled)
        if hasattr(self.fat_model, 'fat') and self.fat_model.fat is not None:
            for name, param in self.fat_model.fat.named_parameters():
                param.requires_grad = True
                print(f"[UNFREEZE] FAT.{name}: {param.shape}")
        
        # Unfreeze visual projection if visual conditioning is used
        if hasattr(self.fat_model, 'visual_proj') and self.fat_model.visual_proj is not None:
            for name, param in self.fat_model.visual_proj.named_parameters():
                param.requires_grad = True
                print(f"[UNFREEZE] FAT.visual_proj.{name}: {param.shape}")
        
        # Unfreeze pose head
        if hasattr(self.pose_predictor, 'pose_head'):
            for name, param in self.pose_predictor.pose_head.named_parameters():
                param.requires_grad = True
                print(f"[UNFREEZE] pose_head.{name}: {param.shape}")
        
        # Count parameters
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        
        print(f"\n[PARAMS] Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
        print(f"{'='*70}\n")
    
    def freeze_fat_only(self):
        """Freeze everything except FAT (for alternating training)."""
        print(f"\n{'='*70}")
        print(f"[FREEZE] Freezing all layers except FAT...")
        print(f"{'='*70}")
        
        # Freeze everything
        for name, param in self.named_parameters():
            param.requires_grad = False
        
        # Unfreeze FAT only
        if hasattr(self.fat_model, 'fat') and self.fat_model.fat is not None:
            for name, param in self.fat_model.fat.named_parameters():
                param.requires_grad = True
                print(f"[UNFREEZE] FAT.{name}: {param.shape}")
        
        if hasattr(self.fat_model, 'visual_proj') and self.fat_model.visual_proj is not None:
            for name, param in self.fat_model.visual_proj.named_parameters():
                param.requires_grad = True
                print(f"[UNFREEZE] FAT.visual_proj.{name}: {param.shape}")
        
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"\n[PARAMS] Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
        print(f"{'='*70}\n")
    
    def freeze_pose_only(self):
        """Freeze everything except pose head (for alternating training)."""
        print(f"\n{'='*70}")
        print(f"[FREEZE] Freezing all layers except pose_head...")
        print(f"{'='*70}")
        
        # Freeze everything
        for name, param in self.named_parameters():
            param.requires_grad = False
        
        # Unfreeze pose head only
        if hasattr(self.pose_predictor, 'pose_head'):
            for name, param in self.pose_predictor.pose_head.named_parameters():
                param.requires_grad = True
                print(f"[UNFREEZE] pose_head.{name}: {param.shape}")
        
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"\n[PARAMS] Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
        print(f"{'='*70}\n")
    
    def forward(self, data: Dict) -> Dict:
        """
        Forward pass through full pipeline.
        
        Pipeline:
        1. FAT-Enhanced AnyCalib: images [B, N, 3, H, W] → intrinsics [B, 4]
        2. Extract focal length (fx) from intrinsics
        3. Depth prediction (frozen)
        4. Flow computation (frozen)
        5. Pose prediction (trainable)
        6. Flow reprojection (for loss computation)
        
        Args:
            data: Dict with 'imgs' [B, N, 3, H, W] in range [0, 1]
        
        Returns:
            Dict with pose_result and all intermediate results for loss computation
        """
        images = data["imgs"]  # [B, N, 3, H, W]
        n, f, c, h, w = images.shape
        device = images.device
        
        # ===== STEP 1: FAT-Enhanced AnyCalib Calibration =====
        # Process all N frames through FAT to get single intrinsics per batch
        # FAT model expects [N, 3, H, W] per sequence
        intrinsics_list = []
        image_sizes = []

        for b in range(n):
            seq_images = images[b]  # [N, 3, H, W]

            # Run FAT-enhanced AnyCalib
            # Disable autocast for AnyCalib (requires FP32)
            with torch.cuda.amp.autocast(enabled=False):
                calib_result = self.fat_model(seq_images, cam_id="pinhole")

            # Extract intrinsics [4] = [fx, fy, cx, cy]
            intrinsics = calib_result["intrinsics"][0]  # First (and only) result

            # Convert to tensor if needed
            if isinstance(intrinsics, torch.Tensor):
                intrinsics_tensor = intrinsics
            else:
                intrinsics_tensor = torch.tensor(intrinsics, device=device, dtype=torch.float32)

            intrinsics_list.append(intrinsics_tensor)
            image_sizes.append(calib_result["image_size"])

        # Stack to [B, 4]
        batch_intrinsics = torch.stack(intrinsics_list, dim=0)  # [B, 4]

        # Extract focal length (fx) from intrinsics
        focal_length = batch_intrinsics[:, 0]  # [B] - fx
        
        # ===== STEP 2: Depth Prediction (frozen) =====
        if not self.use_provided_depth:
            depth_in = images.view(n * f, c, h, w)  # [B*N, 3, H, W]
            with torch.no_grad():
                depths, depth_features = self.depth_predictor(depth_in, return_features=True)
            
            # Handle list output from depth predictor
            if isinstance(depths, list):
                depths = depths[0]
            
            # Convert to inverse depth and reshape
            depths = 1 / depths.clamp_min(1e-3).view(n, f, 1, *depths.shape[-2:])
        else:
            depths = data.get("depths", None)
        
        data["pred_depths"] = depths * 0.1
        data["pred_depths_list"] = [depths]
        
        # ===== STEP 3: Flow and Occlusion (frozen) =====
        if not self.use_provided_flow:
            # Image processor expects images in range [-1, 1]
            images_normalized = (images * 2 - 1).contiguous()
            images_ip_fwd, images_ip_bwd = self.image_processor(images_normalized, data=data)
            flow_occs = images_ip_fwd[:, :, 3:6]  # [B, F, 3, H, W] - flow + occlusion
        else:
            flow_occs = data.get("flow_occs", None)
            images_ip_fwd = None

        # ===== STEP 4: Pose Prediction =====
        # Normalize focal length to AnyCam's NDC [-1, 1] convention.
        # focal_length is FAT fx in ray resolution pixels (~48px).
        # Must normalize in ray space: fx_norm = 2 * fx_ray / W_ray
        H_ray, W_ray = image_sizes[0]
        focal_length_normalized = 2.0 * focal_length / W_ray

        # Create projection matrix from focal length
        proj_candidates = make_proj_from_focal_length(
            focal_length_normalized.unsqueeze(1),  # [B, 1]
            aspect_ratio=h/w
        )

        # Forward pass through pose predictor
        pose_result = self.pose_predictor(
            images,
            depths=depths,
            flow_occs=flow_occs,
            anycalib_predictions=None,  # Don't use internal calibration
            external_focal_norm=focal_length_normalized,
        )

        # Override focal_length with FAT prediction
        pose_result["focal_length"] = focal_length

        # Handle pose candidates: model may output multiple candidates but we only use 1 focal
        poses = pose_result["poses"]
        uncert = pose_result["uncert"]
        if poses.dim() == 5 and poses.shape[2] > 1:
            poses = poses[:, :, 0:1]  # Keep only first candidate [B, F, 1, 4, 4]
            pose_result["poses"] = poses
        if uncert.dim() == 6 and uncert.shape[2] > 1:
            uncert = uncert[:, :, 0:1]  # Keep only first candidate
            pose_result["uncert"] = uncert
        
        # ===== STEP 5: Flow Reprojection (for loss computation) =====
        # Align depths for flow computation - need shape [B, F, 1, 1, H, W]
        num_candidates = 1
        aligned_depths = depths.view(n, f, 1, 1, *depths.shape[-2:])
        alignment_params = torch.zeros(aligned_depths.shape[0], f, 1, 1, device=device)
        
        # Induce flow and compute distance
        induced_flow, dist = induce_flow_dist(
            aligned_depths,
            proj_candidates,
            pose_result["poses"],
            flow_occs[..., :2, :, :] if flow_occs is not None else None
        )
        
        # Package results
        pose_result["flow_occs_in"] = flow_occs
        pose_result["aligned_depths"] = aligned_depths
        pose_result["alignment_params"] = alignment_params
        pose_result["induced_flow"] = induced_flow
        pose_result["dist"] = dist
        pose_result["proj_candidates"] = proj_candidates
        
        # Add focal_length_probs for loss computation compatibility
        batch_size = induced_flow.shape[0]
        pose_result["focal_length_probs"] = torch.ones(batch_size, 1, device=device)
        
        # Select results (single focal candidate)
        selected_induced_flow = induced_flow[:, :, 0, :, :, :]
        selected_proj = proj_candidates[:, 0:1]
        selected_poses = poses[:, :, 0]  # [B, F, 4, 4]
        selected_aligned_depths = aligned_depths[:, :, 0]  # [B, F, 1, H, W]
        selected_uncert = uncert[:, :, 0]  # [B, F, ?, H, W]
        
        # Package final results
        data["images_ip"] = images_ip_fwd if images_ip_fwd is not None else images
        data["induced_flow"] = selected_induced_flow
        data["induced_flow_list"] = [selected_induced_flow]
        data["valid"] = flow_occs[:, :, 2:3] > 0.5 if flow_occs is not None else torch.ones(n, f, 1, h, w, device=device, dtype=torch.bool)
        data["proc_poses"] = selected_poses
        data["proc_projs"] = selected_proj
        data["uncertainties"] = selected_uncert
        data["weights_proc"] = selected_uncert
        data["scaled_depths"] = [selected_aligned_depths]
        data["z_near"] = torch.tensor(self.z_near, device=device)
        data["z_far"] = torch.tensor(self.z_far, device=device)
        data["pose_result"] = pose_result
        data["intrinsics"] = batch_intrinsics  # Store FAT intrinsics for benchmarking

        return data

    def forward_with_calibration_info(self, data: Dict) -> Dict:
        """
        Forward pass that also returns information needed for calibration loss.

        This extends the standard forward() to capture:
        - FAT rays with gradients (for calibration reprojection loss)
        - Per-frame AnyCalib intrinsics (for computing average as calibration anchor)

        Pipeline:
        1. FAT-Enhanced AnyCalib: images [B, N, 3, H, W] → intrinsics [B, 4] + rays [B, H*W, 3]
        2. Extract focal length (fx) from intrinsics
        3. Depth prediction (frozen)
        4. Flow computation (frozen)
        5. Pose prediction (trainable)
        6. Flow reprojection (for loss computation)

        Args:
            data: Dict with 'imgs' [B, N, 3, H, W] in range [0, 1]

        Returns:
            Dict with pose_result and all intermediate results for loss computation,
            plus calibration info:
                - 'fat_rays': [B, H*W, 3] aggregated rays from FAT (WITH gradients)
                - 'fat_image_size': (H_ray, W_ray) tuple
                - 'per_frame_intrinsics': [B, N, 4] per-frame AnyCalib predictions
                - 'average_intrinsics': [B, 4] averaged intrinsics (detached)
                - 'original_image_size': (H_orig, W_orig) tuple
        """
        images = data["imgs"]  # [B, N, 3, H, W]
        n, f, c, h, w = images.shape
        device = images.device

        # ===== STEP 1: FAT-Enhanced AnyCalib Calibration with Ray Capture =====
        # Process all N frames through FAT to get single intrinsics per batch
        # FAT model expects [N, 3, H, W] per sequence
        intrinsics_list = []
        rays_list = []
        image_sizes = []
        per_frame_intrinsics_list = []

        for b in range(n):
            seq_images = images[b]  # [N, 3, H, W]

            # Run FAT-enhanced AnyCalib (returns rays WITH gradients)
            # Disable autocast for AnyCalib (requires FP32)
            with torch.cuda.amp.autocast(enabled=False):
                calib_result = self.fat_model(seq_images.float(), cam_id="pinhole")

            # Extract intrinsics [4] = [fx, fy, cx, cy]
            intrinsics = calib_result["intrinsics"][0]  # First (and only) result

            # Convert to tensor if needed
            if isinstance(intrinsics, torch.Tensor):
                intrinsics_tensor = intrinsics
            else:
                intrinsics_tensor = torch.tensor(intrinsics, device=device, dtype=torch.float32)

            intrinsics_list.append(intrinsics_tensor)

            # Extract rays [1, H*W, 3] - WITH GRADIENTS for calibration loss
            rays = calib_result["rays"]  # [1, H*W, 3]
            rays_list.append(rays)

            # Store image size (ray resolution)
            image_sizes.append(calib_result["image_size"])

            # Get per-frame AnyCalib predictions (detached - no gradients)
            # IMPORTANT: Disable autocast because AnyCalib calibrator uses torch.linalg.solve_ex
            # which doesn't support FP16 ("lu_factor_cusolver" not implemented for 'Half')
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=False):
                    per_frame = self.fat_model.get_per_frame_intrinsics(seq_images.float(), cam_id="pinhole")
            per_frame_intrinsics_list.append(per_frame)

        # Stack to [B, 4]
        batch_intrinsics = torch.stack(intrinsics_list, dim=0)  # [B, 4]

        # Stack rays to [B, H*W, 3] - these have gradients
        batch_rays = torch.cat(rays_list, dim=0)  # [B, H*W, 3]

        # Stack per-frame intrinsics [B, N, 4]
        batch_per_frame = torch.stack(per_frame_intrinsics_list, dim=0)  # [B, N, 4]

        # Compute average intrinsics (detached)
        average_intrinsics = batch_per_frame.mean(dim=1)  # [B, 4]

        # Extract focal length (fx) from FAT intrinsics
        focal_length = batch_intrinsics[:, 0]  # [B] - fx

        # ===== STEP 2: Depth Prediction (frozen) =====
        if not self.use_provided_depth:
            depth_in = images.view(n * f, c, h, w)  # [B*N, 3, H, W]
            with torch.no_grad():
                depths, depth_features = self.depth_predictor(depth_in, return_features=True)

            # Handle list output from depth predictor
            if isinstance(depths, list):
                depths = depths[0]

            # Convert to inverse depth and reshape
            depths = 1 / depths.clamp_min(1e-3).view(n, f, 1, *depths.shape[-2:])
        else:
            depths = data.get("depths", None)

        data["pred_depths"] = depths * 0.1
        data["pred_depths_list"] = [depths]

        # ===== STEP 3: Flow and Occlusion (frozen) =====
        if not self.use_provided_flow:
            # Image processor expects images in range [-1, 1]
            images_normalized = (images * 2 - 1).contiguous()
            images_ip_fwd, images_ip_bwd = self.image_processor(images_normalized, data=data)
            flow_occs = images_ip_fwd[:, :, 3:6]  # [B, F, 3, H, W] - flow + occlusion
        else:
            flow_occs = data.get("flow_occs", None)
            images_ip_fwd = None

        # ===== STEP 4: Pose Prediction =====
        # Normalize focal length to AnyCam's NDC [-1, 1] convention.
        # focal_length is FAT fx in ray resolution pixels (~48px).
        # Must normalize in ray space: fx_norm = 2 * fx_ray / W_ray
        H_ray_pose, W_ray_pose = image_sizes[0]
        focal_length_normalized = 2.0 * focal_length / W_ray_pose

        # Create projection matrix from focal length
        proj_candidates = make_proj_from_focal_length(
            focal_length_normalized.unsqueeze(1),  # [B, 1]
            aspect_ratio=h/w
        )

        # Forward pass through pose predictor
        pose_result = self.pose_predictor(
            images,
            depths=depths,
            flow_occs=flow_occs,
            anycalib_predictions=None,  # Don't use internal calibration
            external_focal_norm=focal_length_normalized,
        )

        # Override focal_length with FAT prediction
        pose_result["focal_length"] = focal_length

        # Handle pose candidates: model may output multiple candidates but we only use 1 focal
        poses = pose_result["poses"]
        uncert = pose_result["uncert"]
        if poses.dim() == 5 and poses.shape[2] > 1:
            poses = poses[:, :, 0:1]  # Keep only first candidate [B, F, 1, 4, 4]
            pose_result["poses"] = poses
        if uncert.dim() == 6 and uncert.shape[2] > 1:
            uncert = uncert[:, :, 0:1]  # Keep only first candidate
            pose_result["uncert"] = uncert

        # ===== STEP 5: Flow Reprojection (for loss computation) =====
        # Align depths for flow computation - need shape [B, F, 1, 1, H, W]
        num_candidates = 1
        aligned_depths = depths.view(n, f, 1, 1, *depths.shape[-2:])
        alignment_params = torch.zeros(aligned_depths.shape[0], f, 1, 1, device=device)

        # Induce flow and compute distance
        induced_flow, dist = induce_flow_dist(
            aligned_depths,
            proj_candidates,
            pose_result["poses"],
            flow_occs[..., :2, :, :] if flow_occs is not None else None
        )

        # Package results
        pose_result["flow_occs_in"] = flow_occs
        pose_result["aligned_depths"] = aligned_depths
        pose_result["alignment_params"] = alignment_params
        pose_result["induced_flow"] = induced_flow
        pose_result["dist"] = dist
        pose_result["proj_candidates"] = proj_candidates

        # Add focal_length_probs for loss computation compatibility
        batch_size = induced_flow.shape[0]
        pose_result["focal_length_probs"] = torch.ones(batch_size, 1, device=device)

        # Select results (single focal candidate)
        selected_induced_flow = induced_flow[:, :, 0, :, :, :]
        selected_proj = proj_candidates[:, 0:1]
        selected_poses = poses[:, :, 0]  # [B, F, 4, 4]
        selected_aligned_depths = aligned_depths[:, :, 0]  # [B, F, 1, H, W]
        selected_uncert = uncert[:, :, 0]  # [B, F, ?, H, W]

        # Package final results for flow loss
        data["images_ip"] = images_ip_fwd if images_ip_fwd is not None else images
        data["induced_flow"] = selected_induced_flow
        data["induced_flow_list"] = [selected_induced_flow]
        data["valid"] = flow_occs[:, :, 2:3] > 0.5 if flow_occs is not None else torch.ones(n, f, 1, h, w, device=device, dtype=torch.bool)
        data["proc_poses"] = selected_poses
        data["proc_projs"] = selected_proj
        data["uncertainties"] = selected_uncert
        data["weights_proc"] = selected_uncert
        data["scaled_depths"] = [selected_aligned_depths]
        data["z_near"] = torch.tensor(self.z_near, device=device)
        data["z_far"] = torch.tensor(self.z_far, device=device)
        data["pose_result"] = pose_result
        data["intrinsics"] = batch_intrinsics  # Store FAT intrinsics for benchmarking

        # ===== CALIBRATION INFO for combined loss =====
        # These are needed for the calibration reprojection loss
        data["fat_rays"] = batch_rays  # [B, H*W, 3] - WITH GRADIENTS for calibration loss
        data["fat_image_size"] = image_sizes[0]  # (H_ray, W_ray) - assume same for all batches
        data["per_frame_intrinsics"] = batch_per_frame  # [B, N, 4]
        data["average_intrinsics"] = average_intrinsics  # [B, 4] - detached reference
        data["original_image_size"] = (h, w)  # (H_orig, W_orig)

        return data


# Back-compat alias after the FAT -> MCT rename.
AnyCamWrapperWithFATCalibration = AnyCamWrapperWithMCTCalibration
