"""
Camera Encoder: Encodes camera parameters to embedding space.

This module implements the camera parameter encoder that maps camera intrinsics
(fx, fy, cx, cy) to embedding tokens for use in the DA3 calibration head.
"""

import torch
import torch.nn as nn


class CameraEncoder(nn.Module):
    """
    Encodes camera parameters (fx, fy, cx, cy) to embedding space.
    
    Architecture:
        Input: [B, N, 4] (fx, fy, cx, cy)
        Normalize → MLP → Output: [B, N, D_cam]
    """
    def __init__(self, cam_dim=256, hidden_dim=128):
        super().__init__()
        self.cam_dim = cam_dim
        self.hidden_dim = hidden_dim
        
        # Normalization: assumes image size provided
        self.normalize = True
        
        # MLP Encoder
        self.encoder = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, cam_dim)
        )
    
    def forward(self, camera_params, image_size):
        """
        Args:
            camera_params: [B, N, 4] (fx, fy, cx, cy) in pixels
            image_size: (H, W) tuple for normalization
        
        Returns:
            camera_tokens: [B, N, cam_dim]
        """
        B, N, _ = camera_params.shape
        H, W = image_size
        
        # Normalize camera parameters
        if self.normalize:
            fx_norm = camera_params[:, :, 0] / (max(H, W) / 2)  # Normalize fx
            fy_norm = camera_params[:, :, 1] / (max(H, W) / 2)  # Normalize fy
            cx_norm = camera_params[:, :, 2] / W  # Normalize cx
            cy_norm = camera_params[:, :, 3] / H  # Normalize cy
            
            normalized = torch.stack([fx_norm, fy_norm, cx_norm, cy_norm], dim=-1)
        else:
            normalized = camera_params
        
        # Encode
        camera_tokens = self.encoder(normalized)  # [B, N, cam_dim]
        
        return camera_tokens

