"""
AnyCalib with Feature Aggregation Transformer (FAT).

This module wraps AnyCalib to support multi-frame calibration by inserting
the FAT between Step 1 (DINOv2 feature extraction) and Step 2 (DPT decoder).

Architecture:
    1. DINOv2 backbone (frozen) extracts features per frame
    2. FAT (trainable) aggregates multi-frame features
    3. DPT decoder + ray head (frozen) produce rays
    4. Calibrator (unchanged) fits camera model
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Dict, List, Optional, Tuple

# Set environment to disable xFormers for GPU compatibility
os.environ["XFORMERS_DISABLED"] = "1"

from .feature_aggregation_transformer import FeatureAggregationTransformer


class AnyCalibWithFAT(nn.Module):
    """
    AnyCalib with Feature Aggregation Transformer for multi-frame calibration.

    This wrapper modifies AnyCalib to process multiple frames and aggregate
    their DINOv2 features using FAT before passing to the DPT decoder.

    Args:
        model_id: AnyCalib model variant (e.g., "anycalib_pinhole")
        use_fat: Whether to use FAT for aggregation (False = mean pooling fallback)
        fat_config: Configuration dict for FeatureAggregationTransformer
        use_dinov2_small: Use HuggingFace DINOv2-small for visual tokens (local dev)
        use_dinov2_full: Use torch.hub DINOv2 for visual tokens (cluster with xFormers)
        freeze_backbone: Freeze DINOv2 backbone
        freeze_decoder: Freeze DPT decoder and ray head
        freeze_calibrator: Freeze calibrator (always True, non-differentiable)
    """

    # Constants from AnyCalib
    EDGE_DIVISIBLE_BY = 14
    AR_RANGE = (0.5, 2)  # H/W range seen during training
    RESOLUTION = 102_400  # resolution seen during training

    def __init__(
        self,
        model_id: str = "anycalib_pinhole",
        use_fat: bool = True,
        fat_config: Optional[Dict] = None,
        use_dinov2_small: bool = True,
        use_dinov2_full: bool = False,
        freeze_backbone: bool = True,
        freeze_decoder: bool = True,
        freeze_calibrator: bool = True,
    ):
        super().__init__()

        self.model_id = model_id
        self.use_fat = use_fat
        self.use_dinov2_small = use_dinov2_small
        self.use_dinov2_full = use_dinov2_full
        self.freeze_backbone = freeze_backbone
        self.freeze_decoder = freeze_decoder
        self.freeze_calibrator = freeze_calibrator

        # Import AnyCalib components
        from anycalib.cameras import CameraFactory
        from anycalib.model.dinov2 import DINOv2
        from anycalib.model.dpt_light_decoder import LightDPTDecoder
        from anycalib.model.ray_decoder import ConvexTangentDecoder
        from anycalib.model.anycalib_pretrained import Calibrator

        self.CameraFactory = CameraFactory

        # Load AnyCalib components
        self.backbone = DINOv2(model_name="dinov2_vitl14")
        self.decoder = LightDPTDecoder(embed_dim=self.backbone.embed_dim)
        self.head = ConvexTangentDecoder(in_channels=self.decoder.out_channels)
        self.calibrator = Calibrator(
            nonlin_opt_method="gauss_newton",
            fallback_to_sac=True,
        )

        # Load pretrained weights
        url = f"https://github.com/javrtg/AnyCalib/releases/download/v1.0.0/{model_id}.pt"
        model_dir = f"{torch.hub.get_dir()}/anycalib"
        state_dict = torch.hub.load_state_dict_from_url(
            url, model_dir, map_location="cpu", file_name=f"{model_id}.pt"
        )

        # Load state dict (may have some missing keys for FAT)
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"[AnyCalibWithFAT] Missing keys (expected for FAT): {len(missing)}")

        # Feature Aggregation Transformer
        if use_fat:
            default_fat_config = {
                "embed_dim": 1024,  # DINOv2 vitL dimension
                "num_heads": 8,
                "num_layers": 2,
                "dropout": 0.1,
                "use_learnable_agg_token": False,
                "use_visual_conditioning": False,  # Can be enabled later
                "visual_token_dim": 384,
                "num_scales": 4,
            }
            if fat_config:
                default_fat_config.update(fat_config)
            self.fat = FeatureAggregationTransformer(**default_fat_config)
        else:
            self.fat = None

        # Optional DINOv2-small for visual tokens (separate from backbone)
        self.visual_backbone = None
        self.visual_dim = 384
        if use_dinov2_small:
            self._init_dinov2_small()
        elif use_dinov2_full:
            self._init_dinov2_full()

        # Apply freezing
        self._apply_freezing()

    def _init_dinov2_small(self):
        """Initialize HuggingFace DINOv2-small for visual tokens."""
        try:
            from transformers import AutoModel, AutoImageProcessor
            self.visual_backbone = AutoModel.from_pretrained('facebook/dinov2-small')
            self.visual_backbone.eval()
            for param in self.visual_backbone.parameters():
                param.requires_grad = False
            self.visual_dim = 384
            print("[AnyCalibWithFAT] Loaded DINOv2-small for visual tokens")
        except ImportError:
            print("[WARNING] transformers not installed, visual tokens disabled")
            self.visual_backbone = None

    def _init_dinov2_full(self):
        """Initialize torch.hub DINOv2 for visual tokens (requires xFormers)."""
        try:
            self.visual_backbone = torch.hub.load(
                'facebookresearch/dinov2', 'dinov2_vits14'
            )
            self.visual_backbone.eval()
            for param in self.visual_backbone.parameters():
                param.requires_grad = False
            self.visual_dim = 384  # vits14 has 384 dim
            print("[AnyCalibWithFAT] Loaded DINOv2 vits14 for visual tokens")
        except Exception as e:
            print(f"[WARNING] Failed to load DINOv2 from hub: {e}")
            self.visual_backbone = None

    def _apply_freezing(self):
        """Apply freezing based on configuration."""
        if self.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.backbone.eval()

        if self.freeze_decoder:
            for param in self.decoder.parameters():
                param.requires_grad = False
            for param in self.head.parameters():
                param.requires_grad = False
            self.decoder.eval()
            self.head.eval()

        # Calibrator is always frozen (non-differentiable)
        # No parameters to freeze

    def get_trainable_parameters(self) -> List[nn.Parameter]:
        """Get list of trainable parameters (FAT only by default)."""
        params = []
        if self.fat is not None:
            params.extend(self.fat.parameters())
        if not self.freeze_decoder:
            params.extend(self.decoder.parameters())
            params.extend(self.head.parameters())
        return params

    def _extract_visual_tokens(self, images: Tensor) -> Optional[Tensor]:
        """
        Extract visual tokens from DINOv2-small.

        Args:
            images: [N, 3, H, W] images normalized to [0, 1]

        Returns:
            [N, D_vis] CLS tokens or None if visual backbone not available
        """
        if self.visual_backbone is None:
            return None

        N = images.shape[0]
        device = images.device

        # Normalize for DINOv2 (ImageNet stats)
        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        images_norm = (images - mean) / std

        # Resize to 224x224 for DINOv2-small (if using HuggingFace)
        if self.use_dinov2_small:
            images_resized = nn.functional.interpolate(
                images_norm, size=(224, 224), mode='bilinear', align_corners=False
            )
            with torch.no_grad():
                outputs = self.visual_backbone(images_resized)
                # CLS token is first token in last_hidden_state
                visual_tokens = outputs.last_hidden_state[:, 0, :]  # [N, 384]
        else:
            # torch.hub DINOv2 - can handle various sizes
            with torch.no_grad():
                visual_tokens = self.visual_backbone(images_norm)  # [N, 384]

        return visual_tokens

    def _pad_to_multiple(self, images: Tensor, multiple: int = 14) -> Tuple[Tensor, Tuple[int, int, int, int]]:
        """
        Pad images to make dimensions divisible by multiple.

        Args:
            images: [N, 3, H, W] input images
            multiple: Target multiple (14 for DINOv2 patch size)

        Returns:
            padded_images: [N, 3, H', W'] where H' and W' are multiples of `multiple`
            padding: (pad_left, pad_right, pad_top, pad_bottom)
        """
        N, C, H, W = images.shape

        # Calculate padding needed
        pad_h = (multiple - H % multiple) % multiple
        pad_w = (multiple - W % multiple) % multiple

        if pad_h == 0 and pad_w == 0:
            return images, (0, 0, 0, 0)

        # Center padding
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        # Apply padding (left, right, top, bottom for F.pad)
        padded = torch.nn.functional.pad(
            images,
            (pad_left, pad_right, pad_top, pad_bottom),
            mode='reflect'
        )

        return padded, (pad_left, pad_right, pad_top, pad_bottom)

    def forward_backbone(self, images: Tensor) -> Dict[str, List[Tensor]]:
        """
        Extract features from DINOv2 backbone for each frame.

        Args:
            images: [N, 3, H, W] images

        Returns:
            Dict with 'outputs' list of [N, 1024, h, w] and 'class_tokens'
        """
        N, C, H, W = images.shape

        # DEBUG: Check device placement (only once at start)
        if not hasattr(self, '_debug_printed'):
            print(f"[DEBUG] Input images device: {images.device}")
            print(f"[DEBUG] Backbone first param device: {next(self.backbone.parameters()).device}")
            print(f"[DEBUG] Backbone image_mean device: {self.backbone.image_mean.device}")
            self._debug_printed = True

        # Pad images to multiples of 14 for DINOv2
        images_padded, padding = self._pad_to_multiple(images, self.EDGE_DIVISIBLE_BY)

        # Only print padding info once
        if not hasattr(self, '_padding_printed') and padding != (0, 0, 0, 0):
            _, _, H_pad, W_pad = images_padded.shape
            print(f"[DEBUG] Padded images from {H}x{W} to {H_pad}x{W_pad} (padding: {padding})")
            self._padding_printed = True

        # OPTIMIZED: Process all frames in one batch instead of frame-by-frame
        # NOTE: Even though backbone parameters are frozen (requires_grad=False),
        # we still need gradients to flow THROUGH it for FAT training.
        # The frozen parameters won't be updated, but FAT needs gradients from backbone outputs.
        out = self.backbone(images_padded)  # Process ALL N frames at once
        # out['outputs'] is already List of [N, 1024, h, w] for each scale
        # out['class_tokens'] is already List of [N, ...] for each scale
        stacked_outputs = out['outputs']
        stacked_class_tokens = out['class_tokens']

        return {
            'outputs': stacked_outputs,
            'class_tokens': stacked_class_tokens,
        }

        # OLD SLOW CODE (kept for reference):
        # all_outputs = []
        # all_class_tokens = []
        #
        # # Process each frame
        # for i in range(N):
        #     with torch.no_grad() if self.freeze_backbone else torch.enable_grad():
        #         out = self.backbone(images[i:i+1])  # [1, 1024, h, w] × 4
        #     all_outputs.append(out['outputs'])
        #     all_class_tokens.append(out['class_tokens'])
        #
        # # Stack per scale: List of [N, 1024, h, w]
        # stacked_outputs = [
        #     torch.cat([out[scale] for out in all_outputs], dim=0)
        #     for scale in range(4)
        # ]
        # stacked_class_tokens = [
        #     torch.cat([ct[scale] for ct in all_class_tokens], dim=0)
        #     for scale in range(4)
        # ]

    def forward(
        self,
        images: Tensor,
        cam_id: str = "pinhole",
        return_intermediate: bool = False,
        return_calibration_info: bool = False,
    ) -> Dict:
        """
        Forward pass with multi-frame FAT aggregation.

        Args:
            images: [N, 3, H, W] images for a single sequence
            cam_id: Camera model ID for calibrator
            return_intermediate: If True, return intermediate features
            return_calibration_info: If True, return calibration details (inliers, residuals) for training

        Returns:
            Dict with 'intrinsics', 'rays', 'success', etc.
        """
        N, _, H_orig, W_orig = images.shape
        device = images.device

        # Step 1: DINOv2 backbone (per-frame) - handles padding internally
        backbone_out = self.forward_backbone(images)
        multi_scale_features = backbone_out['outputs']  # List of [N, 1024, h, w]

        # Get padded dimensions from backbone output
        # The decoder output will match the padded feature map dimensions
        _, _, h_feat, w_feat = multi_scale_features[0].shape
        # DPT upsamples by ~2x from the feature resolution
        H_padded = h_feat * 14  # DINOv2 downsamples by 14
        W_padded = w_feat * 14

        # Extract visual tokens if enabled
        visual_tokens = None
        if self.fat is not None and self.fat.use_visual_conditioning:
            visual_tokens = self._extract_visual_tokens(images)

        # Step 1.5: FAT aggregation
        # CRITICAL: Run FAT in FP32 for numerical stability during training
        # Mixed precision (FP16) causes NaN gradients in transformer layers
        if self.fat is not None:
            with torch.amp.autocast(device_type='cuda', enabled=False):
                # Convert to FP32
                multi_scale_fp32 = [f.float() for f in multi_scale_features]
                visual_tokens_fp32 = visual_tokens.float() if visual_tokens is not None else None

                # FAT forward in FP32
                aggregated = self.fat(multi_scale_fp32, visual_tokens_fp32)
        else:
            # Fallback: mean pooling across frames
            aggregated = [f.mean(dim=0, keepdim=True) for f in multi_scale_features]

        # Step 2: DPT decoder + ray head
        decoder_input = {'outputs': aggregated, 'class_tokens': None}

        # NOTE: Even though decoder/head parameters are frozen (requires_grad=False),
        # we still need gradients to flow THROUGH them for the loss computation.
        # The frozen parameters won't be updated, but intermediate activations need gradients.
        decoded = self.decoder(decoder_input)  # [1, 256, H_pad/7, W_pad/7]
        ray_output = self.head(decoded)  # rays [1, 3, H_pad, W_pad], tangent_coords

        # Get actual output dimensions from ray head
        _, _, H_ray, W_ray = ray_output["rays"].shape

        # Reshape rays for calibrator using ACTUAL dimensions
        b = 1  # Always 1 after aggregation
        rays_spatial = ray_output["rays"].permute(0, 2, 3, 1).view(b, H_ray * W_ray, 3)
        tangent_spatial = ray_output["tangent_coords"].permute(0, 2, 3, 1).view(b, H_ray * W_ray, 2)

        # Step 3: Calibrator (with timing for RANSAC)
        # Pass padded dimensions to calibrator (it needs correct image dimensions)
        # Create a dummy image with the right shape for calibrator
        dummy_image = torch.zeros(1, 3, H_ray, W_ray, device=device)
        data = {"image": dummy_image, "cam_id": [cam_id]}

        # Calibrator needs FP32 for LU factorization (not supported in FP16)
        # Also, calibrator is non-differentiable, so no autocast needed
        import time
        ransac_start = time.time()

        with torch.cuda.amp.autocast(enabled=False):
            # Convert rays to FP32 if needed
            rays_fp32 = rays_spatial.float()
            tangent_fp32 = tangent_spatial.float()
            ray_output_fp32 = {
                "rays": rays_fp32,
                "tangent_coords": tangent_fp32,
                "fov_field": tangent_fp32,
            }
            calibration = self.calibrator(ray_output_fp32, data)

        ransac_time = (time.time() - ransac_start) * 1000  # Convert to ms

        result = {
            "intrinsics": calibration["intrinsics"],
            "success": calibration["success"],
            "rays": rays_spatial,  # Keep original precision for training
            "tangent_coords": tangent_spatial,
            "image_size": (H_ray, W_ray),
            "ransac_time_ms": ransac_time,
        }

        if return_calibration_info:
            # Return additional info for training loss computation
            result["calibration_result"] = calibration

        if return_intermediate:
            result["backbone_features"] = multi_scale_features
            result["aggregated_features"] = aggregated
            result["visual_tokens"] = visual_tokens

        return result

    def predict_intrinsics(self, images: Tensor, cam_id: str = "pinhole") -> Tensor:
        """
        Convenience method to get intrinsics as a tensor.

        Args:
            images: [N, 3, H, W] images
            cam_id: Camera model ID

        Returns:
            [4] tensor of (fx, fy, cx, cy)
        """
        result = self.forward(images, cam_id=cam_id)
        intrinsics = result["intrinsics"][0]
        if isinstance(intrinsics, Tensor):
            return intrinsics
        else:
            return torch.tensor(intrinsics, device=images.device, dtype=images.dtype)

    def forward_single_frame(self, image: Tensor, cam_id: str = "pinhole") -> Dict:
        """
        Forward pass for a single frame WITHOUT FAT aggregation.
        Uses standard AnyCalib pipeline for baseline comparison.

        Args:
            image: [1, 3, H, W] single image
            cam_id: Camera model ID

        Returns:
            Dict with 'intrinsics', 'rays', 'success'
        """
        device = image.device
        _, _, H_orig, W_orig = image.shape  # Original dimensions

        # Step 1: DINOv2 backbone - handles padding internally
        backbone_out = self.forward_backbone(image)
        multi_scale_features = backbone_out['outputs']  # List of [1, 1024, h, w]

        # Step 2: DPT decoder - NO FAT aggregation
        # Wrap in dict like regular forward() method
        decoder_input = {'outputs': multi_scale_features, 'class_tokens': None}
        decoded = self.decoder(decoder_input)  # [1, 256, H_pad/7, W_pad/7]

        # Step 3: Ray head
        # Head returns dict with "rays" and "tangent_coords"
        ray_output = self.head(decoded)  # Returns {"rays": ..., "tangent_coords": ...}
        rays = ray_output["rays"]  # [1, 3, H_pad, W_pad] - at padded resolution

        # Step 4: Crop rays back to original image dimensions
        # Compute padding amounts (same logic as _pad_to_multiple)
        pad_h = (self.EDGE_DIVISIBLE_BY - H_orig % self.EDGE_DIVISIBLE_BY) % self.EDGE_DIVISIBLE_BY
        pad_w = (self.EDGE_DIVISIBLE_BY - W_orig % self.EDGE_DIVISIBLE_BY) % self.EDGE_DIVISIBLE_BY
        
        if pad_h > 0 or pad_w > 0:
            # Calculate padding offsets (center padding)
            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left
            
            # Crop rays: remove padding from all sides
            rays = rays[:, :, pad_top:rays.shape[2]-pad_bottom, pad_left:rays.shape[3]-pad_right]
        
        # Now rays match original image dimensions
        H_ray, W_ray = rays.shape[2], rays.shape[3]
        assert H_ray == H_orig and W_ray == W_orig, \
            f"Ray dimensions {H_ray}x{W_ray} don't match original {H_orig}x{W_orig}"
        
        rays_flat = rays.permute(0, 2, 3, 1).reshape(1, -1, 3)  # [1, H*W, 3]

        # Step 5: Calibrator - now image and rays have matching dimensions
        with torch.no_grad():
            rays_fp32 = rays_flat.float()
            calibration = self.calibrator(
                {"rays": rays_fp32},
                {
                    "image": image,  # Original image, now matches ray dimensions
                    "cam_id": [cam_id],
                },
            )

        return {
            "intrinsics": calibration["intrinsics"],
            "success": calibration["success"],
            "rays": rays,
            "image_size": (H_ray, W_ray),
        }

    def get_per_frame_intrinsics(
        self,
        images: Tensor,  # [N, 3, H, W]
        cam_id: str = "pinhole",
    ) -> Tensor:
        """
        Get per-frame AnyCalib intrinsics predictions.

        Args:
            images: [N, 3, H, W] images
            cam_id: Camera model ID

        Returns:
            intrinsics: [N, 4] tensor of (fx, fy, cx, cy) per frame
        """
        N = images.shape[0]
        intrinsics_list = []

        for n in range(N):
            frame = images[n:n+1]  # [1, 3, H, W]
            result = self.forward_single_frame(frame, cam_id=cam_id)
            intr = result['intrinsics'][0]  # [4]
            if isinstance(intr, list):
                intr = intr[0]
            if not isinstance(intr, torch.Tensor):
                intr = torch.tensor(intr, device=images.device, dtype=torch.float32)
            intrinsics_list.append(intr)

        return torch.stack(intrinsics_list, dim=0)  # [N, 4]

    @staticmethod
    def compute_reprojection_loss(
        predicted_rays: Tensor,  # [H*W, 3] normalized rays
        average_intrinsics: Tensor,  # [4] (fx, fy, cx, cy) for original image
        ray_image_size: Tuple[int, int],  # (H_ray, W_ray) - actual ray resolution
        original_image_size: Tuple[int, int],  # (H_orig, W_orig) - original image size
    ) -> Tuple[Tensor, Dict]:
        """
        Compute reprojection loss: project rays back to image plane and compare with pixel grid.

        Args:
            predicted_rays: [H*W, 3] normalized ray directions (differentiable)
            average_intrinsics: [4] (fx, fy, cx, cy) for original image resolution
            ray_image_size: (H_ray, W_ray) - resolution of ray predictions
            original_image_size: (H_orig, W_orig) - original image resolution

        Returns:
            loss: Scalar tensor (differentiable)
            info: Dict with loss components
        """
        H_ray, W_ray = ray_image_size
        H_orig, W_orig = original_image_size

        # Scale intrinsics to ray resolution
        scale_x = W_ray / W_orig
        scale_y = H_ray / H_orig
        fx_down = average_intrinsics[0] * scale_x
        fy_down = average_intrinsics[1] * scale_y
        cx_down = average_intrinsics[2] * scale_x
        cy_down = average_intrinsics[3] * scale_y

        # Project rays to image plane: u = fx * (rx/rz) + cx, v = fy * (ry/rz) + cy
        rz = predicted_rays[:, 2:3].clamp(min=1e-6)
        projected_x = fx_down * (predicted_rays[:, 0] / rz.squeeze()) + cx_down
        projected_y = fy_down * (predicted_rays[:, 1] / rz.squeeze()) + cy_down
        projected = torch.stack([projected_x, projected_y], dim=-1)  # [H*W, 2]

        # Create actual pixel grid using point_grid helper
        y_coords, x_coords = torch.meshgrid(
            torch.arange(H_ray, device=predicted_rays.device, dtype=predicted_rays.dtype),
            torch.arange(W_ray, device=predicted_rays.device, dtype=predicted_rays.dtype),
            indexing='ij'
        )
        actual_coords = torch.stack([x_coords.flatten(), y_coords.flatten()], dim=-1)  # [H*W, 2]

        # Compute MSE loss
        loss = F.mse_loss(projected, actual_coords.float())

        return loss, {
            'reprojection_loss': loss.item(),
            'mean_projected_x': projected_x.mean().item(),
            'mean_projected_y': projected_y.mean().item(),
        }

    @staticmethod
    def compute_inlier_mask_from_residuals(
        predicted_rays: Tensor,
        intrinsics: Tensor,
        image_size: Tuple[int, int],
        threshold_degrees: float = 5.0,  # More lenient: 5° instead of 1°
        use_soft_weights: bool = True,
    ) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        """
        Compute inlier mask and residuals by measuring angular error between
        predicted rays and expected rays from calibration.

        This replicates what RANSAC does: rays with small angular error are inliers.

        Args:
            predicted_rays: [H*W, 3] predicted ray directions
            intrinsics: [4] fitted intrinsics (fx, fy, cx, cy)
            image_size: (H, W) tuple
            threshold_degrees: Angular error threshold in degrees (default 5.0°)
            use_soft_weights: If True, use soft exponential weights instead of hard threshold

        Returns:
            inlier_mask: [H*W] boolean tensor (True for inliers)
            residuals: [H*W] angular errors in radians
            soft_weights: [H*W] optional soft weights (if use_soft_weights=True)
        """
        device = predicted_rays.device

        # Validate predicted rays (must be normalized, non-zero)
        ray_norms = predicted_rays.norm(dim=-1)
        if (ray_norms < 0.1).any():
            raise ValueError(f"Degenerate predicted rays detected (near-zero magnitude)")
        if torch.isnan(ray_norms).any() or torch.isinf(ray_norms).any():
            raise ValueError(f"NaN/Inf in predicted rays")

        # Re-normalize predicted rays (ensure unit vectors)
        predicted_rays = torch.nn.functional.normalize(predicted_rays, dim=-1)

        # Get expected rays from calibration
        expected_rays = AnyCalibWithFAT.compute_expected_rays_from_calibration(
            intrinsics, image_size, device
        )

        # Compute angular error using numerically stable atan2 instead of acos
        # This avoids the gradient explosion issue with acos near ±1
        # Formula: theta = atan2(||cross||, dot) where cross = ray1 × ray2
        dot_product = (predicted_rays * expected_rays).sum(dim=-1)  # [H*W]
        cross_product = torch.cross(predicted_rays, expected_rays, dim=-1)  # [H*W, 3]
        cross_norm = cross_product.norm(dim=-1)  # [H*W]
        angular_error = torch.atan2(cross_norm, dot_product)  # Radians, more stable than acos

        # Convert threshold to radians
        threshold_rad = threshold_degrees * (3.14159265359 / 180.0)

        # Inliers are rays with error below threshold
        inlier_mask = angular_error < threshold_rad

        # Soft weights: exponential decay based on angular error
        # Higher error → lower weight
        soft_weights = None
        if use_soft_weights:
            # exp(-error^2 / (2 * sigma^2)) where sigma = threshold/2
            sigma = threshold_rad / 2.0
            soft_weights = torch.exp(-(angular_error ** 2) / (2 * sigma ** 2))

        return inlier_mask, angular_error, soft_weights

    @staticmethod
    def compute_expected_rays_from_calibration(intrinsics: Tensor, image_size: Tuple[int, int], device: str) -> Tensor:
        """
        Compute expected ray directions from fitted calibration parameters.

        For a pinhole camera, given intrinsics (fx, fy, cx, cy) and pixel coordinates (u, v),
        the ray direction is:
            ray = normalize([(u - cx) / fx, (v - cy) / fy, 1])

        Args:
            intrinsics: [4] tensor (fx, fy, cx, cy) or list
            image_size: (H, W) tuple
            device: torch device

        Returns:
            expected_rays: [H*W, 3] normalized ray directions
        """
        H, W = image_size

        # Extract intrinsics
        if isinstance(intrinsics, (list, tuple)):
            fx, fy, cx, cy = intrinsics
        else:
            fx, fy, cx, cy = intrinsics[0], intrinsics[1], intrinsics[2], intrinsics[3]

        # Validate intrinsics (prevent division by zero)
        if abs(fx) < 1.0 or abs(fy) < 1.0:
            raise ValueError(f"Degenerate intrinsics: fx={fx}, fy={fy} (too close to zero)")

        # Create pixel grid
        v_coords, u_coords = torch.meshgrid(
            torch.arange(H, device=device, dtype=torch.float32),
            torch.arange(W, device=device, dtype=torch.float32),
            indexing='ij'
        )
        u_coords = u_coords.flatten()  # [H*W]
        v_coords = v_coords.flatten()  # [H*W]

        # Compute ray directions from camera model
        ray_x = (u_coords - cx) / fx
        ray_y = (v_coords - cy) / fy
        ray_z = torch.ones_like(ray_x)

        # Stack and normalize
        expected_rays = torch.stack([ray_x, ray_y, ray_z], dim=-1)  # [H*W, 3]
        expected_rays = torch.nn.functional.normalize(expected_rays, dim=-1)

        return expected_rays

    @staticmethod
    def compute_ray_consistency_loss(
        predicted_rays: Tensor,
        intrinsics: Tensor,
        image_size: Tuple[int, int],
        soft_weights: Optional[Tensor] = None,
        inlier_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Dict]:
        """
        Compute differentiable loss that encourages predicted rays to be consistent
        with the fitted calibration.

        This implements the supervisor's approach:
        1. Use calibrator to fit intrinsics (non-differentiable, detached)
        2. Compute expected rays from those intrinsics
        3. Measure difference between predicted and expected rays
        4. Weight by soft weights (exponential decay based on angular error)

        Args:
            predicted_rays: [B, H*W, 3] predicted ray directions (has grad_fn)
            intrinsics: [B, 4] or list of intrinsics from calibrator (detached)
            image_size: (H, W) tuple
            soft_weights: [B, H*W] optional soft weights (0-1 range)
            inlier_mask: [B, H*W] boolean tensor (True for inliers), for logging only

        Returns:
            loss: Scalar tensor (differentiable w.r.t. predicted_rays)
            info: Dict with loss components for logging
        """
        B = predicted_rays.shape[0]
        device = predicted_rays.device

        total_loss = 0.0
        total_inlier_count = 0
        total_outlier_count = 0

        for b in range(B):
            # Get expected rays from calibration
            intrinsics_b = intrinsics[b] if B > 1 else intrinsics
            expected_rays = AnyCalibWithFAT.compute_expected_rays_from_calibration(
                intrinsics_b, image_size, device
            )  # [H*W, 3]

            # Compute per-ray error (angular difference)
            ray_errors = (predicted_rays[b] - expected_rays).pow(2).sum(dim=-1)  # [H*W]

            # Apply soft weights if provided
            if soft_weights is not None:
                weights = soft_weights[b]
            else:
                weights = torch.ones_like(ray_errors)

            # Count inliers for logging
            if inlier_mask is not None:
                inlier_count = inlier_mask[b].sum().item()
                outlier_count = (~inlier_mask[b]).sum().item()
                total_inlier_count += inlier_count
                total_outlier_count += outlier_count

            # Weighted mean
            weighted_loss = (weights * ray_errors).sum() / (weights.sum() + 1e-8)
            total_loss += weighted_loss

        # Average over batch
        loss = total_loss / B

        info = {
            "ray_consistency_loss": loss.item(),
            "total_inliers": total_inlier_count,
            "total_outliers": total_outlier_count,
        }

        return loss, info

    def forward_with_differentiable_calibration(
        self,
        images: Tensor,
        cam_id: str = "pinhole",
        inlier_threshold_degrees: float = 1.0,
        inlier_weight: float = 1.0,
        outlier_weight: float = 1e-6,
    ) -> Dict:
        """
        Forward pass with differentiable calibration using implicit differentiation.

        This method:
        1. Runs the full pipeline to get rays (with gradients)
        2. Runs RANSAC to get inlier mask (no gradients)
        3. Computes differentiable intrinsics via weighted least squares

        The gradient flows through the least-squares solution, not RANSAC.

        Args:
            images: [N, 3, H, W] images
            cam_id: Camera model ID
            inlier_threshold_degrees: Angular threshold for RANSAC inliers
            inlier_weight: Weight for inlier rays (default 1.0)
            outlier_weight: Weight for outlier rays (default 1e-6)

        Returns:
            Dict with:
                'intrinsics_diff': [4] differentiable intrinsics (has gradients)
                'intrinsics_ransac': [4] RANSAC intrinsics (detached)
                'rays': [1, H*W, 3] predicted rays
                'inlier_mask': [H*W] boolean mask
                'inlier_ratio': float
                'ransac_time_ms': float
        """
        from .differentiable_calibrator import DifferentiableCalibrator

        # Run normal forward to get rays and RANSAC intrinsics
        result = self.forward(images, cam_id=cam_id, return_calibration_info=True)

        rays = result['rays']  # [1, H*W, 3] - has gradients
        ransac_intrinsics = result['intrinsics'][0]  # [4] - from RANSAC
        image_size = result['image_size']

        # Convert RANSAC intrinsics to tensor
        if not isinstance(ransac_intrinsics, Tensor):
            ransac_intrinsics = torch.tensor(
                ransac_intrinsics, device=rays.device, dtype=torch.float32
            )

        # Get inlier mask from RANSAC output (no gradients)
        with torch.no_grad():
            inlier_mask, angular_error, _ = self.compute_inlier_mask_from_residuals(
                rays[0],  # [H*W, 3]
                ransac_intrinsics,
                image_size,
                threshold_degrees=inlier_threshold_degrees,
                use_soft_weights=False,
            )

        # Create differentiable calibrator
        diff_calibrator = DifferentiableCalibrator(
            inlier_weight=inlier_weight,
            outlier_weight=outlier_weight,
        )

        # Compute differentiable intrinsics (gradients flow through this!)
        # CRITICAL: rays must have gradients, inlier_mask is detached
        diff_result = diff_calibrator(
            predicted_rays=rays[0],  # [H*W, 3] - WITH gradients
            image_size=image_size,
            ransac_intrinsics=ransac_intrinsics.detach(),  # NO gradients
            inlier_mask=inlier_mask.detach(),  # NO gradients
        )

        return {
            'intrinsics_diff': diff_result['intrinsics'],  # [4] - differentiable!
            'intrinsics_ransac': ransac_intrinsics.detach(),  # [4] - for regularization
            'rays': rays,
            'inlier_mask': inlier_mask,
            'inlier_ratio': diff_result['inlier_ratio'],
            'ransac_time_ms': result['ransac_time_ms'],
            'image_size': image_size,
        }

    def train(self, mode: bool = True):
        """Override train to respect freezing."""
        super().train(mode)

        # Always keep frozen components in eval mode
        if self.freeze_backbone:
            self.backbone.eval()
        if self.freeze_decoder:
            self.decoder.eval()
            self.head.eval()
        if self.visual_backbone is not None:
            self.visual_backbone.eval()

        return self


class AnyCamWrapperWithFAT(nn.Module):
    """
    AnyCam wrapper using FAT-enhanced AnyCalib for focal length prediction.

    This integrates AnyCalibWithFAT into the AnyCam pipeline for end-to-end
    training with flow reprojection loss.

    Args:
        anycam_checkpoint: Path to AnyCam checkpoint
        anycalib_fat: AnyCalibWithFAT instance or config
        freeze_pose_predictor: Freeze AnyCam pose predictor
        freeze_depth_predictor: Freeze depth predictor
    """

    def __init__(
        self,
        anycam_checkpoint: str,
        anycalib_fat: Optional[AnyCalibWithFAT] = None,
        fat_config: Optional[Dict] = None,
        freeze_pose_predictor: bool = True,
        freeze_depth_predictor: bool = True,
        device: str = "cuda:0",
    ):
        super().__init__()

        self.device = device
        self.freeze_pose_predictor = freeze_pose_predictor
        self.freeze_depth_predictor = freeze_depth_predictor

        # Initialize AnyCalibWithFAT
        if anycalib_fat is not None:
            self.anycalib_fat = anycalib_fat
        else:
            self.anycalib_fat = AnyCalibWithFAT(
                use_fat=True,
                fat_config=fat_config or {},
                use_dinov2_small=True,
            )

        # Load AnyCam components
        self._load_anycam(anycam_checkpoint)

        # Apply freezing
        self._apply_freezing()

    def _load_anycam(self, checkpoint_path: str):
        """Load AnyCam pose predictor and depth predictor."""
        from anycam.models import make_pose_predictor, make_depth_predictor
        from anycam.common.io.configs import load_config
        from omegaconf import OmegaConf

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        # Get config from checkpoint or use default
        if "config" in checkpoint:
            config = checkpoint["config"]
            if isinstance(config, dict):
                config = OmegaConf.create(config)
        else:
            # Use default config
            config = load_config("anycam/configs/anycam_training.yaml")

        # Create pose predictor
        self.pose_predictor = make_pose_predictor(config.pose_predictor)

        # Load pose predictor weights
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
            # Filter for pose_predictor keys
            pose_state = {
                k.replace("pose_predictor.", ""): v
                for k, v in state_dict.items()
                if k.startswith("pose_predictor.")
            }
            if pose_state:
                self.pose_predictor.load_state_dict(pose_state, strict=False)

        # Create depth predictor
        self.depth_predictor = make_depth_predictor(config.depth_predictor)

        # Create image processor for flow
        from anycam.common.image_processor import make_image_processor
        ip_config = config.image_processor if hasattr(config, 'image_processor') else {"type": "flow_occlusion"}
        self.image_processor = make_image_processor(ip_config)

        # Move to device
        self.pose_predictor.to(self.device)
        self.depth_predictor.to(self.device)
        self.image_processor.to(self.device)

    def _apply_freezing(self):
        """Apply freezing to AnyCam components."""
        if self.freeze_pose_predictor:
            for param in self.pose_predictor.parameters():
                param.requires_grad = False
            self.pose_predictor.eval()

        if self.freeze_depth_predictor:
            for param in self.depth_predictor.parameters():
                param.requires_grad = False
            self.depth_predictor.eval()

        # Image processor is always frozen
        for param in self.image_processor.parameters():
            param.requires_grad = False
        self.image_processor.eval()

    def get_trainable_parameters(self) -> List[nn.Parameter]:
        """Get all trainable parameters."""
        params = list(self.anycalib_fat.get_trainable_parameters())

        if not self.freeze_pose_predictor:
            params.extend(self.pose_predictor.parameters())

        return params

    def forward(self, batch: Dict) -> Dict:
        """
        Forward pass through full pipeline.

        Args:
            batch: Dict with 'imgs' [B, N, 3, H, W]

        Returns:
            Dict with pose results and focal length for loss computation
        """
        images = batch['imgs']  # [B, N, 3, H, W]
        B, N, C, H, W = images.shape

        results = []

        for b in range(B):
            seq_images = images[b]  # [N, 3, H, W]

            # Get aggregated calibration from FAT
            calib_result = self.anycalib_fat(seq_images, cam_id="pinhole")
            intrinsics = calib_result["intrinsics"][0]

            # Extract focal length (fx)
            if isinstance(intrinsics, Tensor):
                focal_length = intrinsics[0]
            else:
                focal_length = torch.tensor(intrinsics[0], device=images.device)

            # Run through AnyCam pose predictor
            # This requires proper formatting of the batch
            seq_batch = {
                'imgs': seq_images.unsqueeze(0),  # [1, N, 3, H, W]
                'focal_length': focal_length.unsqueeze(0),  # [1]
            }

            # Get depth predictions
            with torch.no_grad():
                depth_result = self.depth_predictor(seq_batch)
            seq_batch.update(depth_result)

            # Get flow predictions
            with torch.no_grad():
                flow_result = self.image_processor(seq_batch)
            seq_batch.update(flow_result)

            # Get pose predictions
            with torch.no_grad() if self.freeze_pose_predictor else torch.enable_grad():
                pose_result = self.pose_predictor(seq_batch)

            # Override focal length with FAT prediction
            pose_result['focal_length'] = focal_length.unsqueeze(0)
            pose_result['intrinsics'] = intrinsics

            results.append(pose_result)

        # Combine batch results
        combined = {}
        for key in results[0].keys():
            if isinstance(results[0][key], Tensor):
                combined[key] = torch.stack([r[key] for r in results])
            else:
                combined[key] = [r[key] for r in results]

        return combined

    def train(self, mode: bool = True):
        """Override train to respect freezing."""
        super().train(mode)

        if self.freeze_pose_predictor:
            self.pose_predictor.eval()
        if self.freeze_depth_predictor:
            self.depth_predictor.eval()
        self.image_processor.eval()

        return self
