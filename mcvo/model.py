"""MCVO: image-only multi-frame visual odometry (+calibration later), FVO-style,
trained with AnyCam-style self-supervision (cached flow/depth teachers in the loss).

Architecture (E1 scope):
    frozen DINOv2 backbone (per-frame patch tokens)
      -> linear proj to d_model
      -> L x [temporal attention (across frames, per spatial location)
              + spatial attention (within frame, incl. a per-frame camera token)
              + MLP]
      -> pose head: per adjacent pair (cam_i, cam_{i+1}) -> 7D (t, quaternion) * 0.01
      -> per-patch uncertainty head (upsampled to image res) for the Laplacian NLL loss

Camera tokens participate ONLY in spatial attention (FVO ablation finding).
Inputs at inference: raw images only. No depth, no flow, no intrinsics.
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from minipytorch3d.rotation_conversions import quaternion_to_matrix


class Block(nn.Module):
    """One time-space attention block."""

    def __init__(self, d: int, heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.t_norm = nn.LayerNorm(d)
        self.t_attn = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.s_norm = nn.LayerNorm(d)
        self.s_attn = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.m_norm = nn.LayerNorm(d)
        self.mlp = nn.Sequential(
            nn.Linear(d, int(d * mlp_ratio)), nn.GELU(), nn.Linear(int(d * mlp_ratio), d)
        )

    def forward(self, x: torch.Tensor, cam: torch.Tensor):
        """x: [B, N, S, D] patch tokens; cam: [B, N, D] camera tokens."""
        B, N, S, D = x.shape

        # temporal attention over frames, per spatial location (patch tokens only)
        t = x.permute(0, 2, 1, 3).reshape(B * S, N, D)
        t = self.t_norm(t)
        t_out, _ = self.t_attn(t, t, t, need_weights=False)
        x = x + t_out.reshape(B, S, N, D).permute(0, 2, 1, 3)

        # spatial attention within each frame, camera token included
        s_in = torch.cat([cam.unsqueeze(2), x], dim=2)  # [B, N, 1+S, D]
        s = s_in.reshape(B * N, 1 + S, D)
        s = self.s_norm(s)
        s_out, _ = self.s_attn(s, s, s, need_weights=False)
        s_out = s_out.reshape(B, N, 1 + S, D)
        cam = cam + s_out[:, :, 0]
        x = x + s_out[:, :, 1:]

        # MLP on everything
        cam = cam + self.mlp(self.m_norm(cam))
        x = x + self.mlp(self.m_norm(x))
        return x, cam


class MCVO(nn.Module):
    POSE_FACTOR = 0.01  # same output scaling trick as AnyCam
    LOGF_PRIOR = 0.30   # log(f/W) ~ log(1.35): typical casual-video focal at square crops

    def __init__(
        self,
        backbone: str = "facebook/dinov2-small",
        d_model: int = 384,
        depth: int = 6,
        heads: int = 6,
        dropout: float = 0.0,
        freeze_backbone: bool = True,
        precomputed_features: bool = False,
    ):
        super().__init__()
        self.backbone_name = backbone
        self.precomputed_features = precomputed_features

        if not precomputed_features:
            from transformers import AutoModel
            self.backbone = AutoModel.from_pretrained(backbone)
            bdim = self.backbone.config.hidden_size
            if freeze_backbone:
                for p in self.backbone.parameters():
                    p.requires_grad = False
                self.backbone.eval()
        else:
            self.backbone = None
            bdim = d_model

        self.proj = nn.Linear(bdim, d_model) if bdim != d_model else nn.Identity()
        self.cam_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        # small learnable temporal position embedding (up to 32 frames)
        self.time_embed = nn.Parameter(torch.zeros(1, 32, 1, d_model))

        self.blocks = nn.ModuleList(Block(d_model, heads, dropout=dropout) for _ in range(depth))

        self.pose_head = nn.Sequential(
            nn.Linear(2 * d_model, d_model), nn.GELU(), nn.Linear(d_model, 7)
        )
        # per-patch uncertainty for pair i (uses frame i tokens): softplus -> positive
        self.uncert_head = nn.Linear(d_model, 1)
        # optional calibration head on the per-frame camera token: predicts
        # [log(f / W) - LOGF_PRIOR, cx offset / W, cy offset / H]. Trained by distillation
        # from cached AnyCalib intrinsics (a training-time teacher, like depth and flow) and/or
        # by feeding the prediction into the flow-reprojection loss. Zero-initialised so an
        # untrained head predicts the prior (f = W * exp(LOGF_PRIOR), centred principal point).
        self.calib_head = nn.Linear(d_model, 3)
        nn.init.zeros_(self.calib_head.weight); nn.init.zeros_(self.calib_head.bias)

        # ImageNet normalization (backbone expects it; dataset gives [0,1])
        self.register_buffer("im_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("im_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """images: [B, N, 3, H, W] in [0,1] -> patch tokens [B, N, S, D_backbone]."""
        B, N, C, H, W = images.shape
        x = images.reshape(B * N, C, H, W)
        x = (x - self.im_mean) / self.im_std
        with torch.no_grad():
            out = self.backbone(pixel_values=x)
        tok = out.last_hidden_state[:, 1:]  # drop CLS
        return tok.reshape(B, N, tok.shape[1], tok.shape[2])

    def forward(self, images: Optional[torch.Tensor] = None,
                features: Optional[torch.Tensor] = None,
                image_hw: Optional[tuple] = None) -> Dict:
        """Either images [B,N,3,H,W] or precomputed features [B,N,S,D_backbone]."""
        if features is None:
            assert images is not None
            features = self.extract_features(images)
            image_hw = images.shape[-2:]
        B, N, S, _ = features.shape

        x = self.proj(features)
        x = x + self.time_embed[:, :N]
        cam = self.cam_token.expand(B, N, -1).contiguous()
        cam = cam + self.time_embed[:, :N, 0]

        for blk in self.blocks:
            x, cam = blk(x, cam)

        # poses per adjacent pair
        pair = torch.cat([cam[:, :-1], cam[:, 1:]], dim=-1)  # [B, N-1, 2D]
        enc = self.pose_head(pair) * self.POSE_FACTOR         # [B, N-1, 7]
        t = enc[..., :3]
        quat = enc[..., 3:]
        quat = torch.cat([quat[..., :1] + 1.0, quat[..., 1:]], dim=-1)  # bias toward identity
        quat = F.normalize(quat, dim=-1)
        R = quaternion_to_matrix(quat)                         # [B, N-1, 3, 3]
        poses = torch.eye(4, device=enc.device).repeat(B, N, 1, 1)
        poses[:, :-1, :3, :3] = R
        poses[:, :-1, :3, 3] = t
        # last pose stays identity (padding; excluded by the loss)

        # per-patch uncertainty -> per-pixel map
        g = int(S ** 0.5)
        u = F.softplus(self.uncert_head(x)).squeeze(-1)        # [B, N, S]
        u = u.reshape(B, N, 1, g, g)
        if image_hw is not None:
            u = F.interpolate(u.reshape(B * N, 1, g, g), size=image_hw,
                              mode="bilinear", align_corners=False).reshape(B, N, 1, *image_hw)

        # scalar per-pair confidence proxy (mean uncertainty of source frame)
        pair_conf = u.reshape(B, N, -1).mean(-1)[:, :-1]        # [B, N-1]

        # per-frame calibration from the camera token -> pixels of the input resolution
        c = self.calib_head(cam)                                # [B, N, 3]
        if image_hw is not None:
            Hh, Ww = float(image_hw[0]), float(image_hw[1])
        else:
            Hh = Ww = float(g * 14)
        f = Ww * torch.exp(self.LOGF_PRIOR + c[..., 0])
        cx = Ww * (0.5 + c[..., 1])
        cy = Hh * (0.5 + c[..., 2])
        calib = torch.stack([f, f, cx, cy], dim=-1)             # [B, N, 4] fx fy cx cy

        return {
            "poses": poses.unsqueeze(2),       # [B, N, 1, 4, 4] (candidate dim = 1)
            "uncert": u.unsqueeze(2),  # [B, N, 1(candidate), 1(channel), H, W]
            "pair_conf": pair_conf,
            "calib": calib,                    # [B, N, 4] per-frame intrinsics (px)
            "calib_raw": c,                    # [B, N, 3] head output (for the loss)
            "logf_prior": self.LOGF_PRIOR,
        }
