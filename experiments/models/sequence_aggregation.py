"""
Sequence Aggregation: Aggregates per-frame camera tokens to sequence level.

This module implements sequence-level aggregation using either simple mean pooling
or learnable token with attention.
"""

import torch
import torch.nn as nn


class SequenceCameraAggregation(nn.Module):
    """
    Aggregates per-frame camera tokens into sequence-level representation.
    
    Architecture:
        Per-Frame Tokens [B, N, D_cam] + Learnable Token [1, D_cam] (optional)
        → Cross-Attention or Mean Pooling → Aggregated Token [B, 1, D_cam]
    """
    def __init__(self, cam_dim, use_learnable_token=False):
        super().__init__()
        self.cam_dim = cam_dim
        self.use_learnable_token = use_learnable_token
        
        if use_learnable_token:
            # Learnable sequence-level camera token
            self.learnable_token = nn.Parameter(torch.randn(1, 1, cam_dim))
            
            # Cross-attention for aggregation
            self.attention = nn.MultiheadAttention(
                cam_dim, num_heads=8, batch_first=True
            )
            
            # Layer norm
            self.norm = nn.LayerNorm(cam_dim)
        # If use_learnable_token=False, just use mean pooling (simpler, supervisor's suggestion)
    
    def forward(self, per_frame_tokens, attention_mask=None):
        """
        Args:
            per_frame_tokens: [B, N, D_cam] - Per-frame camera tokens (visual-conditioned)
            attention_mask: [B, N] - Optional mask (1 for valid, 0 for padding)
        
        Returns:
            aggregated_token: [B, 1, D_cam] - Sequence-level camera token
        """
        if self.use_learnable_token:
            B, N, _ = per_frame_tokens.shape
            
            # Expand learnable token to batch size
            learnable = self.learnable_token.expand(B, -1, -1)  # [B, 1, D_cam]
            
            # Convert attention_mask to key_padding_mask format (True = ignore)
            key_padding_mask = None
            if attention_mask is not None:
                key_padding_mask = (attention_mask == 0)  # [B, N] - True where padding
            
            # Cross-attention: query from learnable token, key/value from per-frame tokens
            aggregated, _ = self.attention(
                query=learnable,  # [B, 1, D_cam]
                key=per_frame_tokens,  # [B, N, D_cam]
                value=per_frame_tokens,  # [B, N, D_cam]
                key_padding_mask=key_padding_mask,
            )  # [B, 1, D_cam]
            
            # Layer norm
            aggregated = self.norm(aggregated)
            
            return aggregated
        else:
            # Masked mean pooling
            if attention_mask is not None:
                # attention_mask: [B, N] - 1 for valid, 0 for padding
                mask = attention_mask.unsqueeze(-1)  # [B, N, 1]
                masked_tokens = per_frame_tokens * mask  # [B, N, D_cam]
                sum_tokens = masked_tokens.sum(dim=1, keepdim=True)  # [B, 1, D_cam]
                count = mask.sum(dim=1, keepdim=True).clamp(min=1)  # [B, 1, 1]
                return sum_tokens / count  # [B, 1, D_cam]
            else:
                # Simple mean pooling (no padding)
                return per_frame_tokens.mean(dim=1, keepdim=True)  # [B, 1, D_cam]

