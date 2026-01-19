"""
Feature Aggregation Transformer (FAT): Aggregates multi-frame DINOv2 features.

This module implements a transformer that operates between AnyCalib's Step 1
(DINOv2 feature extraction) and Step 2 (DPT decoder + ray transformation).
It aggregates spatial features from N frames into a single aggregated representation.

Key difference from DA3:
- DA3: Works on AnyCalib's final scalar outputs [fx, fy, cx, cy]
- FAT: Works on AnyCalib's intermediate spatial features [1024, H/14, W/14]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class FeatureAggregationTransformer(nn.Module):
    """
    Aggregates multi-frame DINOv2 features at each spatial position.

    For each spatial position (h, w), we have N frame tokens of dimension embed_dim.
    The transformer aggregates these N tokens into a single output token.

    Architecture:
        1. Flatten spatial: [N, C, h, w] -> [S, N, C] where S=h*w
        2. Optional: Add visual tokens from DINOv2-small CLS
        3. Self-attention across frames at each spatial position
        4. Aggregate (mean pooling or learnable token)
        5. Reshape: [S, C] -> [1, C, h, w]
    """

    def __init__(
        self,
        embed_dim: int = 1024,              # DINOv2 vitL feature dimension
        num_heads: int = 8,                 # Attention heads
        num_layers: int = 2,                # Transformer layers
        dropout: float = 0.1,               # Dropout rate
        use_learnable_agg_token: bool = False,  # Mean pooling vs learnable token
        visual_token_dim: int = 384,        # DINOv2-small CLS token dim
        use_visual_conditioning: bool = True,
        num_scales: int = 4,                # Number of DINOv2 output scales
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.use_learnable_agg_token = use_learnable_agg_token
        self.use_visual_conditioning = use_visual_conditioning
        self.num_scales = num_scales

        # Optional visual token projection (DINOv2-small 384 -> embed_dim)
        if use_visual_conditioning:
            self.visual_proj = nn.Sequential(
                nn.Linear(visual_token_dim, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.GELU(),
            )

        # Learnable aggregation token (optional)
        if use_learnable_agg_token:
            self.agg_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
            # Cross-attention for learnable token aggregation
            self.agg_cross_attn = nn.MultiheadAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True,
            )
            self.agg_norm = nn.LayerNorm(embed_dim)

        # Transformer encoder layers for self-attention
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=embed_dim * 4,
                dropout=dropout,
                activation='gelu',
                batch_first=True,
                norm_first=True,  # Pre-norm for stability
            )
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(embed_dim)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with small values for stability."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        multi_scale_features: List[torch.Tensor],  # List of [N, 1024, h, w]
        visual_tokens: Optional[torch.Tensor] = None,  # [N, 384] CLS tokens
    ) -> List[torch.Tensor]:
        """
        Aggregate multi-frame features at each spatial position.

        Args:
            multi_scale_features: List of 4 tensors, each [N, C, h, w]
                                  where N = num_frames, C = embed_dim (1024)
            visual_tokens: Optional [N, D_vis] visual tokens from DINOv2-small
                          where D_vis = visual_token_dim (384)

        Returns:
            List of 4 tensors, each [1, C, h, w] - aggregated features per scale
        """
        aggregated_features = []

        for scale_idx, scale_features in enumerate(multi_scale_features):
            N, C, h, w = scale_features.shape
            S = h * w  # Number of spatial positions

            # Reshape: [N, C, h, w] -> [S, N, C]
            # Each spatial position becomes a separate "batch" for the transformer
            features_flat = scale_features.permute(2, 3, 0, 1)  # [h, w, N, C]
            features_flat = features_flat.reshape(S, N, C)  # [S, N, C]

            # Add visual conditioning if enabled
            num_feature_tokens = N
            if self.use_visual_conditioning and visual_tokens is not None:
                # Project visual tokens: [N, D_vis] -> [N, C]
                vis_projected = self.visual_proj(visual_tokens)  # [N, C]

                # Expand to match spatial positions: [S, N, C]
                vis_expanded = vis_projected.unsqueeze(0).expand(S, -1, -1)

                # Concatenate: [S, 2N, C] - spatial features + visual context
                features_flat = torch.cat([features_flat, vis_expanded], dim=1)

            # Apply transformer layers
            # features_flat: [S, N', C] where N' = N or 2N (with visual tokens)
            tokens = features_flat
            for layer in self.transformer_layers:
                tokens = layer(tokens)
            tokens = self.final_norm(tokens)

            # Extract only the spatial feature tokens (not visual tokens)
            if self.use_visual_conditioning and visual_tokens is not None:
                tokens = tokens[:, :num_feature_tokens, :]  # [S, N, C]

            # Aggregate across frames
            if self.use_learnable_agg_token:
                # Cross-attention: learnable token queries frame tokens
                agg = self.agg_token.expand(S, -1, -1)  # [S, 1, C]
                agg_out, _ = self.agg_cross_attn(
                    query=agg,
                    key=tokens,
                    value=tokens,
                )
                aggregated = self.agg_norm(agg_out.squeeze(1))  # [S, C]
            else:
                # Mean pooling across frames
                aggregated = tokens.mean(dim=1)  # [S, C]

            # Reshape back to spatial: [S, C] -> [1, C, h, w]
            aggregated = aggregated.reshape(h, w, C)  # [h, w, C]
            aggregated = aggregated.permute(2, 0, 1).unsqueeze(0)  # [1, C, h, w]

            aggregated_features.append(aggregated)

        return aggregated_features


class FeatureAggregationTransformerV2(nn.Module):
    """
    Alternative FAT implementation with shared weights across scales.

    This version uses the same transformer for all 4 scales, reducing parameters.
    The scale information can be optionally encoded via a scale embedding.
    """

    def __init__(
        self,
        embed_dim: int = 1024,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
        use_learnable_agg_token: bool = False,
        visual_token_dim: int = 384,
        use_visual_conditioning: bool = True,
        num_scales: int = 4,
        use_scale_embedding: bool = False,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.use_learnable_agg_token = use_learnable_agg_token
        self.use_visual_conditioning = use_visual_conditioning
        self.num_scales = num_scales
        self.use_scale_embedding = use_scale_embedding

        # Optional scale embedding
        if use_scale_embedding:
            self.scale_embed = nn.Embedding(num_scales, embed_dim)

        # Visual projection
        if use_visual_conditioning:
            self.visual_proj = nn.Sequential(
                nn.Linear(visual_token_dim, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.GELU(),
            )

        # Learnable aggregation token
        if use_learnable_agg_token:
            self.agg_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
            self.agg_cross_attn = nn.MultiheadAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True,
            )
            self.agg_norm = nn.LayerNorm(embed_dim)

        # Shared transformer layers
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=embed_dim * 4,
                dropout=dropout,
                activation='gelu',
                batch_first=True,
                norm_first=True,
            )
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(embed_dim)
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def _process_single_scale(
        self,
        scale_features: torch.Tensor,  # [N, C, h, w]
        visual_tokens: Optional[torch.Tensor],  # [N, D_vis]
        scale_idx: int,
    ) -> torch.Tensor:
        """Process a single scale."""
        N, C, h, w = scale_features.shape
        S = h * w

        # Reshape: [N, C, h, w] -> [S, N, C]
        features_flat = scale_features.permute(2, 3, 0, 1).reshape(S, N, C)

        # Add scale embedding if enabled
        if self.use_scale_embedding:
            scale_emb = self.scale_embed(
                torch.tensor([scale_idx], device=scale_features.device)
            )  # [1, C]
            features_flat = features_flat + scale_emb.unsqueeze(0)  # [S, N, C]

        # Add visual conditioning
        num_feature_tokens = N
        if self.use_visual_conditioning and visual_tokens is not None:
            vis_projected = self.visual_proj(visual_tokens)
            vis_expanded = vis_projected.unsqueeze(0).expand(S, -1, -1)
            features_flat = torch.cat([features_flat, vis_expanded], dim=1)

        # Apply transformer
        tokens = features_flat
        for layer in self.transformer_layers:
            tokens = layer(tokens)
        tokens = self.final_norm(tokens)

        # Extract feature tokens only
        if self.use_visual_conditioning and visual_tokens is not None:
            tokens = tokens[:, :num_feature_tokens, :]

        # Aggregate
        if self.use_learnable_agg_token:
            agg = self.agg_token.expand(S, -1, -1)
            agg_out, _ = self.agg_cross_attn(query=agg, key=tokens, value=tokens)
            aggregated = self.agg_norm(agg_out.squeeze(1))
        else:
            aggregated = tokens.mean(dim=1)

        # Reshape: [S, C] -> [1, C, h, w]
        return aggregated.reshape(h, w, C).permute(2, 0, 1).unsqueeze(0)

    def forward(
        self,
        multi_scale_features: List[torch.Tensor],
        visual_tokens: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        """Process all scales."""
        return [
            self._process_single_scale(feat, visual_tokens, idx)
            for idx, feat in enumerate(multi_scale_features)
        ]


# Convenience function to create FAT with common configurations
def create_fat(
    config: str = "default",
    **kwargs,
) -> FeatureAggregationTransformer:
    """
    Create FAT with preset configurations.

    Args:
        config: One of "default", "small", "large", "shared"
        **kwargs: Override any config parameter

    Returns:
        FeatureAggregationTransformer instance
    """
    configs = {
        "default": {
            "embed_dim": 1024,
            "num_heads": 8,
            "num_layers": 2,
            "dropout": 0.1,
            "use_learnable_agg_token": False,
            "use_visual_conditioning": True,
            "visual_token_dim": 384,
        },
        "small": {
            "embed_dim": 1024,
            "num_heads": 8,
            "num_layers": 1,
            "dropout": 0.1,
            "use_learnable_agg_token": False,
            "use_visual_conditioning": False,
        },
        "large": {
            "embed_dim": 1024,
            "num_heads": 16,
            "num_layers": 4,
            "dropout": 0.1,
            "use_learnable_agg_token": True,
            "use_visual_conditioning": True,
            "visual_token_dim": 384,
        },
    }

    if config not in configs:
        raise ValueError(f"Unknown config: {config}. Available: {list(configs.keys())}")

    cfg = configs[config].copy()
    cfg.update(kwargs)

    return FeatureAggregationTransformer(**cfg)
