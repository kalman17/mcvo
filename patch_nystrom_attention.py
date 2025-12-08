import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class NystromBlock(nn.Module):
    """Replacement NystromBlock using PyTorch native SDPA instead of xformers"""
    def __init__(self, dim, num_heads=8, mlp_ratio=4.0, drop=0.0, **kwargs):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(drop),
        )
    
    def forward(self, x, pos_embed=None, **kwargs):
        # Add positional embedding if provided
        if pos_embed is not None:
            x = x + pos_embed
        
        # Self-attention with residual
        residual = x
        x = self.norm1(x)
        B, N, C = x.shape
        
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        
        # Use fp16 for flash attention compatibility
        orig_dtype = q.dtype
        if orig_dtype == torch.float32:
            q, k, v = q.half(), k.half(), v.half()
        
        out = F.scaled_dot_product_attention(q, k, v)
        
        if orig_dtype == torch.float32:
            out = out.float()
        
        out = out.transpose(1, 2).reshape(B, N, C)
        x = residual + self.proj(out)
        
        # MLP with residual
        x = x + self.mlp(self.norm2(x))
        return x

