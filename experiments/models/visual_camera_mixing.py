"""
Visual-Camera Mixing: Updates camera tokens with visual information.

This module implements DA3-style mixing where camera and visual tokens are
concatenated and processed with self-attention, then only camera tokens
are extracted (visual token updates are discarded).

Supervisor's approach: "Put all tokens to the same set and run regular attention.
Then forget about updates of visual tokens. Like the depth anything 3 style."
"""

import torch
import torch.nn as nn


class VisualCameraMixing(nn.Module):
    """
    Updates camera tokens with visual information using DA3-style self-attention.
    
    Architecture:
        1. Project visual tokens to camera dimension
        2. Concatenate: [camera_tokens, visual_tokens] → all_tokens
        3. Run self-attention on all_tokens
        4. Extract only camera tokens (discard visual token updates)
    """
    def __init__(self, vis_dim, cam_dim, num_layers=1):
        super().__init__()
        self.vis_dim = vis_dim
        self.cam_dim = cam_dim
        self.num_layers = num_layers
        
        # Project visual tokens to camera dimension for concatenation
        self.visual_projection = nn.Linear(vis_dim, cam_dim)
        
        # Self-attention layers on combined tokens (DA3 style)
        self.self_attention_layers = nn.ModuleList([
            nn.MultiheadAttention(cam_dim, num_heads=8, batch_first=True)
            for _ in range(num_layers)
        ])
        
        # Layer norms for each attention layer
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(cam_dim) for _ in range(num_layers)
        ])
    
    def forward(self, visual_tokens, camera_tokens):
        """
        Args:
            visual_tokens: [B, N, D_vis] - Visual features from backbone
            camera_tokens: [B, N, D_cam] - Camera tokens from encoder
        
        Returns:
            updated_camera_tokens: [B, N, D_cam] - Visual-conditioned camera tokens
        """
        B, N_cam, _ = camera_tokens.shape
        
        # Project visual tokens to same dimension as camera tokens
        visual_proj = self.visual_projection(visual_tokens)  # [B, N, D_cam]
        
        # Concatenate: camera tokens first, then visual tokens
        # "Put all of the tokens to the same set"
        all_tokens = torch.cat([camera_tokens, visual_proj], dim=1)  # [B, 2*N, D_cam]
        
        # Run self-attention on combined tokens
        # "To combine the visual tokens and the camera tokens you can just use regular attention"
        for self_attn, norm in zip(self.self_attention_layers, self.layer_norms):
            residual = all_tokens
            all_tokens_out, _ = self_attn(all_tokens, all_tokens, all_tokens)
            all_tokens = norm(all_tokens_out + residual)
        
        # "Forget about updates of visual tokens" - extract only camera tokens
        updated_camera_tokens = all_tokens[:, :N_cam, :]  # [B, N, D_cam]
        
        return updated_camera_tokens
