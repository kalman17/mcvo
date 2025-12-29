"""
Camera Decoder: Decodes aggregated camera token back to camera parameters.

This module implements the decoder that maps sequence-level camera tokens
back to camera intrinsics (fx, fy, cx, cy).
"""

import torch
import torch.nn as nn


class CameraDecoder(nn.Module):
    """
    Decodes aggregated camera token back to camera parameters.
    
    Architecture:
        Aggregated Token [B, 1, D_cam] → MLP → Parameters [B, 1, 4]
    """
    def __init__(self, cam_dim, hidden_dim=128):
        super().__init__()
        self.cam_dim = cam_dim
        self.hidden_dim = hidden_dim
        
        # MLP Decoder
        self.decoder = nn.Sequential(
            nn.Linear(cam_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4)  # Output: fx, fy, cx, cy
        )
    
    def forward(self, aggregated_token, image_size):
        """
        Args:
            aggregated_token: [B, 1, D_cam] - Sequence-level camera token
            image_size: (H, W) tuple for denormalization
        
        Returns:
            camera_params: [B, 1, 4] (fx, fy, cx, cy) in pixels
        """
        # Decode
        normalized_params = self.decoder(aggregated_token)  # [B, 1, 4]
        
        # Denormalize
        H, W = image_size
        fx = normalized_params[:, :, 0] * (max(H, W) / 2)
        fy = normalized_params[:, :, 1] * (max(H, W) / 2)
        cx = normalized_params[:, :, 2] * W
        cy = normalized_params[:, :, 3] * H
        
        camera_params = torch.stack([fx, fy, cx, cy], dim=-1)  # [B, 1, 4]
        
        return camera_params

