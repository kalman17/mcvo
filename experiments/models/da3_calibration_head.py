"""
DA3 Calibration Head: Complete calibration head combining all components.

This module implements the full Depth Anything 3-inspired calibration head
that combines camera encoding, visual-camera mixing, sequence aggregation,
and camera decoding.
"""

import torch
import torch.nn as nn

from .camera_encoder import CameraEncoder
from .visual_camera_mixing import VisualCameraMixing
from .sequence_aggregation import SequenceCameraAggregation
from .camera_decoder import CameraDecoder


class DA3CalibrationHead(nn.Module):
    """
    Complete DA3-inspired calibration head for multi-frame camera conditioning.
    
    Architecture:
        Visual Tokens + AnyCalib Init → Camera Encoder → Visual-Camera Mixing
        → Sequence Aggregation → Camera Decoder → Camera Parameters
    """
    def __init__(
        self,
        vis_dim=768,  # Visual token dimension (from backbone)
        cam_dim=256,  # Camera token dimension
        hidden_dim=128,  # Hidden layer dimension
        num_mixing_layers=2,  # Number of attention layers in mixing
        use_learnable_token=False,  # Use learnable token for aggregation (default: mean pooling)
    ):
        super().__init__()
        
        self.vis_dim = vis_dim
        self.cam_dim = cam_dim
        self.hidden_dim = hidden_dim
        
        # Components
        self.camera_encoder = CameraEncoder(cam_dim, hidden_dim)
        self.visual_camera_mixing = VisualCameraMixing(
            vis_dim, cam_dim, num_layers=num_mixing_layers
        )
        self.sequence_aggregation = SequenceCameraAggregation(
            cam_dim, use_learnable_token=use_learnable_token
        )
        self.camera_decoder = CameraDecoder(cam_dim, hidden_dim)
    
    def forward(self, visual_tokens, anycalib_predictions, image_size, use_visual_conditioning=True, attention_mask=None):
        """
        Args:
            visual_tokens: [B, N, D_vis] - Visual features from backbone
            anycalib_predictions: [B, N, 4] - AnyCalib predictions (fx, fy, cx, cy)
            image_size: (H, W) tuple
            use_visual_conditioning: If False, skip visual mixing (Stage 1 training)
            attention_mask: [B, N] - Optional mask (1 for valid, 0 for padding)
        
        Returns:
            camera_params: [B, 1, 4] - Predicted camera parameters (fx, fy, cx, cy)
        """
        # Stage 1: Encode camera parameters
        camera_tokens = self.camera_encoder(anycalib_predictions, image_size)  # [B, N, D_cam]
        
        # Stage 2: Update camera tokens with visual information (reverse conditioning)
        if use_visual_conditioning:
            # Visual-conditioned camera tokens
            updated_camera_tokens = self.visual_camera_mixing(visual_tokens, camera_tokens)  # [B, N, D_cam]
        else:
            # Stage 1 training: skip visual conditioning
            updated_camera_tokens = camera_tokens  # [B, N, D_cam]
        
        # Stage 3: Aggregate to sequence level (with optional mask for variable-length sequences)
        aggregated_camera = self.sequence_aggregation(updated_camera_tokens, attention_mask)  # [B, 1, D_cam]
        
        # Stage 4: Decode to camera parameters
        camera_params = self.camera_decoder(aggregated_camera, image_size)  # [B, 1, 4]
        
        return camera_params

