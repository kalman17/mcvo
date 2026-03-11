"""
Unified Training Wrapper for all training phases (A, B1, B2, C, Da, Db).

Configures model components, freezing, and forward passes per phase:
  - Phase A:  Pose head only. AnyCam DINOv2-small (frozen) runs live.
              Depth, flow, calib loaded from preprocessed .npz.
  - Phase B1: FAT pre-training. AnyCalib DINOv2 ViT-L (frozen) runs live.
              Calib from .npz used as pseudo GT for reprojection loss.
  - Phase B2: FAT end-to-end through frozen pose pipeline. Both backbones
              run live (frozen). Only FAT adapter trainable. Flow reprojection
              loss teaches FAT to produce calibrations good for pose estimation.
  - Phase C:  End-to-end alternating. Backbones unfrozen during their
              respective component's training turn.
  - Phase Da: Pose-only fine-tuning from Phase C. Only pose_head trainable (~21K).
  - Phase Db: Pose-path fine-tuning from Phase C. Unfreezes pose_head +
              interframe attention + feature fusion + reassembly (~2.5M).
"""

import logging
import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

os.environ["XFORMERS_DISABLED"] = "1"

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Flow / pose composition utilities for multi-frame consistency loss
# ---------------------------------------------------------------------------

def _compose_flows(
    flow_list: List[torch.Tensor],
    occ_list: List[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compose consecutive pixel-space flows via bilinear warping.

    Args:
        flow_list: List of [B, 2, H, W] consecutive flows (pixel space).
        occ_list:  List of [B, 1, H, W] occlusion masks.

    Returns:
        (composed_flow [B, 2, H, W], composed_occ [B, 1, H, W])
    """
    composed_flow = flow_list[0].clone()
    composed_occ = occ_list[0].clone()
    _, _, h, w = composed_flow.shape
    device = composed_flow.device

    for i in range(1, len(flow_list)):
        curr_flow = flow_list[i]
        curr_occ = occ_list[i]

        y_coords, x_coords = torch.meshgrid(
            torch.arange(h, device=device, dtype=torch.float32),
            torch.arange(w, device=device, dtype=torch.float32),
            indexing="ij",
        )

        warped_x = x_coords.unsqueeze(0) + composed_flow[:, 0]
        warped_y = y_coords.unsqueeze(0) + composed_flow[:, 1]

        grid_x = (warped_x / (w - 1)) * 2 - 1
        grid_y = (warped_y / (h - 1)) * 2 - 1
        grid = torch.stack([grid_x, grid_y], dim=-1)  # [B, H, W, 2]

        sampled_flow = F.grid_sample(
            curr_flow, grid, mode="bilinear", padding_mode="zeros", align_corners=True,
        )
        composed_flow = composed_flow + sampled_flow

        valid = (warped_x >= 0) & (warped_x < w) & (warped_y >= 0) & (warped_y < h)
        sampled_occ = F.grid_sample(
            curr_occ, grid, mode="bilinear", padding_mode="zeros", align_corners=True,
        )
        composed_occ = composed_occ * sampled_occ * valid.float().unsqueeze(1)

    return composed_flow, composed_occ


def _compose_poses(pose_list: List[torch.Tensor]) -> torch.Tensor:
    """Compose consecutive 4×4 poses: T_{0→2} = T_{0→1} @ T_{1→2}."""
    composed = pose_list[0]
    for p in pose_list[1:]:
        composed = composed @ p
    return composed


class UnifiedTrainingWrapper(nn.Module):
    """
    Unified model wrapper that configures itself per training phase.

    Args:
        phase: Training phase ('A', 'B1', 'B2', 'C', 'Da', 'Db').
        anycam_config_path: Path to AnyCam training config YAML.
        image_size: Target image size for model inputs (default 336).
    """

    def __init__(
        self,
        phase: str,
        anycam_config_path: str = "pretrained_models/anycam_seq8/training_config.yaml",
        image_size: int = 336,
    ):
        super().__init__()
        self.phase = phase
        self.image_size = image_size
        self.lambda_comp = 0.1          # Composed flow loss weight (can be set to 0 to disable)

        # These will be initialized per-phase
        self.pose_predictor = None      # AnyCam model (DINOv2-small backbone + pose head)
        self.fat_model = None           # AnyCalibWithFAT (DINOv2 ViT-L + FAT + decoder)
        self._training_mode = None      # Phase C alternating: 'pose' or 'calib'

        if phase == 'A':
            self._init_phase_a(anycam_config_path)
        elif phase == 'B1':
            self._init_phase_b1()
        elif phase == 'B2':
            self._init_phase_b2(anycam_config_path)
        elif phase in ('C', 'Ca'):
            self._init_phase_c(anycam_config_path)
        elif phase == 'Cb':
            self._init_phase_cb(anycam_config_path)
        elif phase in ('Da', 'Db'):
            self._init_phase_d(anycam_config_path)
        else:
            raise ValueError(f"Unknown phase: {phase}")

        self._print_param_summary()

    # ------------------------------------------------------------------
    # Initialization helpers
    # ------------------------------------------------------------------

    def _init_phase_a(self, config_path: str):
        """Phase A: Only AnyCam (DINOv2-small + pose head)."""
        from anycam.models import make_pose_predictor
        from omegaconf import OmegaConf

        config = OmegaConf.load(config_path)
        self.pose_predictor = make_pose_predictor(config.model.pose_predictor)

        # Freeze backbone, unfreeze pose head
        for param in self.pose_predictor.parameters():
            param.requires_grad = False
        for param in self.pose_predictor.pose_head.parameters():
            param.requires_grad = True

        logger.info("[Phase A] AnyCam loaded. Backbone frozen, pose head trainable.")

    def _init_phase_b1(self):
        """Phase B1: Only AnyCalibWithFAT (DINOv2 ViT-L + FAT + decoder)."""
        from experiments.models.anycalib_with_fat import AnyCalibWithFAT

        self.fat_model = AnyCalibWithFAT(
            model_id="anycalib_pinhole",
            use_fat=True,
            fat_config={
                "embed_dim": 1024,
                "num_heads": 8,
                "num_layers": 2,
                "dropout": 0.1,
                "use_visual_conditioning": False,
                "num_scales": 4,
            },
            use_dinov2_small=False,   # No visual conditioning
            use_dinov2_full=False,
            freeze_backbone=True,
            freeze_decoder=True,
        )

        logger.info("[Phase B1] AnyCalibWithFAT loaded. Only FAT trainable.")

    def _init_phase_c(self, config_path: str):
        """Phase C: Joint training — everything unfrozen."""
        from anycam.models import make_pose_predictor
        from omegaconf import OmegaConf
        from experiments.models.anycalib_with_fat import AnyCalibWithFAT

        # Load AnyCam
        config = OmegaConf.load(config_path)
        self.pose_predictor = make_pose_predictor(config.model.pose_predictor)

        # Load AnyCalibWithFAT
        self.fat_model = AnyCalibWithFAT(
            model_id="anycalib_pinhole",
            use_fat=True,
            fat_config={
                "embed_dim": 1024,
                "num_heads": 8,
                "num_layers": 2,
                "dropout": 0.1,
                "use_visual_conditioning": False,
                "num_scales": 4,
            },
            use_dinov2_small=False,
            use_dinov2_full=False,
            freeze_backbone=True,
            freeze_decoder=True,
        )

        # Freeze everything first
        for param in self.parameters():
            param.requires_grad = False

        # Unfreeze only task-specific heads:
        # 1. Pose head (AnyCam pose prediction)
        for param in self.pose_predictor.pose_head.parameters():
            param.requires_grad = True
        # 2. FAT adapter (calibration prediction)
        for param in self.fat_model.fat.parameters():
            param.requires_grad = True

        logger.info("[Phase C] Both pipelines loaded. Only pose_head + FAT trainable (backbones frozen).")

    def _init_phase_cb(self, config_path: str):
        """Phase Cb: Like Ca/C but also unfreezes pose neck (reassemble, fusion, interframe attn)."""
        # Start with same setup as Phase C
        self._init_phase_c(config_path)

        # Additionally unfreeze pose-path neck layers
        for name, param in self.pose_predictor.named_parameters():
            if any(name.startswith(prefix) for prefix in (
                "pose_reassemble_stage.",
                "pose_feature_fusion_stage.",
                "pose_interframe_attention.",
                "sequence_token_attention.",
                "sequence_token",
                "sequence_info_head.",
                "focal_embedding.",
            )):
                param.requires_grad = True

        logger.info("[Phase Cb] Both pipelines loaded. pose_head + FAT + pose neck trainable (backbones frozen).")

    def _init_phase_d(self, config_path: str):
        """Phase D (Da/Db): Pose-only fine-tuning from Phase C checkpoint.

        Same model loading as Phase C (both pose_predictor + fat_model).
        Freezes everything, then selectively unfreezes pose-path components:
          - Da: Only pose_head (~21K params) — minimal, conservative
          - Db: All pose-path layers (~2.5M params) — interframe attention,
                feature fusion, reassembly, sequence token, etc.
        """
        from anycam.models import make_pose_predictor
        from omegaconf import OmegaConf
        from experiments.models.anycalib_with_fat import AnyCalibWithFAT

        # Load AnyCam
        config = OmegaConf.load(config_path)
        self.pose_predictor = make_pose_predictor(config.model.pose_predictor)

        # Load AnyCalibWithFAT
        self.fat_model = AnyCalibWithFAT(
            model_id="anycalib_pinhole",
            use_fat=True,
            fat_config={
                "embed_dim": 1024,
                "num_heads": 8,
                "num_layers": 2,
                "dropout": 0.1,
                "use_visual_conditioning": False,
                "num_scales": 4,
            },
            use_dinov2_small=False,
            use_dinov2_full=False,
            freeze_backbone=True,
            freeze_decoder=True,
        )

        # Freeze everything first
        for param in self.parameters():
            param.requires_grad = False

        if self.phase == 'Da':
            # Da: Only pose_head trainable (~21K params)
            for param in self.pose_predictor.pose_head.parameters():
                param.requires_grad = True
            logger.info("[Phase Da] Pose-only fine-tuning: only pose_head trainable (~21K params).")

        elif self.phase == 'Db':
            # Db: All pose-path layers trainable (~2.5M params)
            for name, param in self.pose_predictor.named_parameters():
                if any(name.startswith(prefix) for prefix in (
                    "pose_head.",
                    "pose_reassemble_stage.",
                    "pose_feature_fusion_stage.",
                    "pose_interframe_attention.",
                    "sequence_token_attention.",
                    "sequence_token",
                    "sequence_info_head.",
                )):
                    param.requires_grad = True
            logger.info("[Phase Db] Pose-path fine-tuning: pose_head + reassemble + fusion + "
                        "interframe attention + sequence token (~2.5M params).")

    def _init_phase_b2(self, config_path: str):
        """Phase B2: Both pipelines loaded, only FAT trainable, pose head frozen."""
        from anycam.models import make_pose_predictor
        from omegaconf import OmegaConf
        from experiments.models.anycalib_with_fat import AnyCalibWithFAT

        # Load AnyCam (will be fully frozen)
        config = OmegaConf.load(config_path)
        self.pose_predictor = make_pose_predictor(config.model.pose_predictor)

        # Load AnyCalibWithFAT (only FAT adapter trainable)
        self.fat_model = AnyCalibWithFAT(
            model_id="anycalib_pinhole",
            use_fat=True,
            fat_config={
                "embed_dim": 1024,
                "num_heads": 8,
                "num_layers": 2,
                "dropout": 0.1,
                "use_visual_conditioning": False,
                "num_scales": 4,
            },
            use_dinov2_small=False,
            use_dinov2_full=False,
            freeze_backbone=True,
            freeze_decoder=True,
        )

        self._freeze_for_b2()
        logger.info("[Phase B2] Both pipelines loaded. Only FAT trainable, pose head frozen.")

    def _freeze_for_b2(self):
        """B2: freeze everything, unfreeze only FAT adapter."""
        for param in self.parameters():
            param.requires_grad = False

        if self.fat_model is not None and self.fat_model.fat is not None:
            for param in self.fat_model.fat.parameters():
                param.requires_grad = True

    # ------------------------------------------------------------------
    # Checkpoint loading
    # ------------------------------------------------------------------

    def _filter_shape_mismatches(self, state_dict: dict) -> dict:
        """Filter out keys whose shapes don't match the current model."""
        current = self.state_dict()
        filtered = {}
        skipped = []
        for k, v in state_dict.items():
            if k in current and current[k].shape != v.shape:
                skipped.append(f"{k}: ckpt {v.shape} vs model {current[k].shape}")
            else:
                filtered[k] = v
        if skipped:
            logger.info(f"Skipped {len(skipped)} shape-mismatched keys: {skipped}")
        return filtered

    def load_pretrained_pose_predictor(self, checkpoint_path: str):
        """
        Initialize pose_predictor from an original pretrained AnyCam checkpoint.

        The pretrained checkpoint stores the full AnyCam model under 'model' key.
        We extract pose_predictor.* keys and load them, giving a warm start
        closer to a good local minimum.
        """
        if self.pose_predictor is None:
            logger.warning("No pose_predictor to initialize — skipping pretrained loading")
            return

        ckpt = torch.load(checkpoint_path, map_location="cpu")
        # Original AnyCam checkpoints use 'model' key (not 'model_state_dict')
        state = ckpt.get("model", ckpt.get("model_state_dict", ckpt))

        pose_keys = {k: v for k, v in state.items() if k.startswith("pose_predictor.")}
        pose_keys = self._filter_shape_mismatches(pose_keys)
        if pose_keys:
            missing, unexpected = self.load_state_dict(pose_keys, strict=False)
            loaded = len(pose_keys) - len(unexpected)
            logger.info(f"Loaded pretrained pose_predictor: {loaded}/{len(pose_keys)} keys loaded, "
                        f"{len(missing)} missing in current model")
        else:
            logger.warning(f"No pose_predictor keys found in {checkpoint_path}")

    def load_phase_checkpoint(self, checkpoint_path: str, source_phase: str):
        """
        Load weights from a previous phase checkpoint.

        Args:
            checkpoint_path: Path to .pt checkpoint file.
            source_phase: Which phase produced this checkpoint ('A', 'B1', 'B2').
        """
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        state = ckpt.get("model_state_dict", ckpt)

        if source_phase == 'A':
            # Load pose head weights into pose_predictor
            pose_keys = {k: v for k, v in state.items() if k.startswith("pose_predictor.")}
            pose_keys = self._filter_shape_mismatches(pose_keys)
            if pose_keys:
                missing, unexpected = self.load_state_dict(pose_keys, strict=False)
                logger.info(f"Loaded Phase A pose head: {len(pose_keys)} keys, "
                            f"{len(missing)} missing, {len(unexpected)} unexpected")
            else:
                logger.warning("No pose_predictor keys found in Phase A checkpoint")

        elif source_phase == 'B1':
            # Load FAT weights into fat_model
            fat_keys = {k: v for k, v in state.items() if 'fat_model.fat.' in k or 'fat.' in k}
            if not fat_keys:
                # Try loading directly into fat_model.fat
                fat_keys = {f"fat_model.fat.{k}": v for k, v in state.items()
                            if not k.startswith("fat_model.") and not k.startswith("pose_predictor.")}
            fat_keys = self._filter_shape_mismatches(fat_keys)
            if fat_keys:
                missing, unexpected = self.load_state_dict(fat_keys, strict=False)
                logger.info(f"Loaded Phase B1 FAT: {len(fat_keys)} keys, "
                            f"{len(missing)} missing, {len(unexpected)} unexpected")
            else:
                logger.warning("No FAT keys found in Phase B1 checkpoint")

        elif source_phase == 'B2':
            # Load FAT weights from Phase B2 (same structure as B1 but trained end-to-end)
            fat_keys = {k: v for k, v in state.items() if k.startswith("fat_model.")}
            fat_keys = self._filter_shape_mismatches(fat_keys)
            if fat_keys:
                missing, unexpected = self.load_state_dict(fat_keys, strict=False)
                logger.info(f"Loaded Phase B2 fat_model: {len(fat_keys)} keys, "
                            f"{len(missing)} missing, {len(unexpected)} unexpected")
            else:
                logger.warning("No fat_model keys found in Phase B2 checkpoint")

        elif source_phase == 'C':
            # Load full Phase C state (pose_predictor + fat_model)
            relevant_keys = {k: v for k, v in state.items()
                             if k.startswith("pose_predictor.") or k.startswith("fat_model.")}
            relevant_keys = self._filter_shape_mismatches(relevant_keys)
            if relevant_keys:
                missing, unexpected = self.load_state_dict(relevant_keys, strict=False)
                logger.info(f"Loaded Phase C: {len(relevant_keys)} keys, "
                            f"{len(missing)} missing, {len(unexpected)} unexpected")
            else:
                logger.warning("No pose_predictor/fat_model keys found in Phase C checkpoint")

        else:
            raise ValueError(f"Unknown source_phase: {source_phase}")

    # ------------------------------------------------------------------
    # Forward passes per phase
    # ------------------------------------------------------------------

    def forward(self, data: Dict) -> Dict:
        """Route to phase-specific forward pass."""
        if self.phase == 'A':
            return self._forward_phase_a(data)
        elif self.phase == 'B1':
            return self._forward_phase_b1(data)
        elif self.phase in ('B2', 'C', 'Ca', 'Cb', 'Da', 'Db'):
            return self._forward_combined(data)
        else:
            raise ValueError(f"Unknown phase: {self.phase}")

    def _forward_phase_a(self, data: Dict) -> Dict:
        """
        Phase A: Pose head training with preprocessed data.

        Inputs (from dataset):
            images:    [B, N, 3, H, W]
            depths:    [B, N, 1, H, W]  (inverse depth)
            flows_fwd: [B, N-1, 2, H, W]
            occs_fwd:  [B, N-1, 1, H, W]
            calibs:    [B, N, 4]  (fx, fy, cx, cy)

        Returns dict with loss components.
        """
        from anycam.trainer import induce_flow_dist, make_proj_from_focal_length

        images = data["images"]       # [B, N, 3, H, W]
        depths = data["depths"]       # [B, N, 1, H, W]
        flows_fwd = data["flows_fwd"] # [B, N-1, 2, H, W]
        occs_fwd = data["occs_fwd"]   # [B, N-1, 1, H, W]
        calibs = data["calibs"]       # [B, N, 4]

        B, N, C, H, W = images.shape
        device = images.device

        # Average calibration across frames → single [fx, fy, cx, cy]
        avg_calib = calibs.mean(dim=1)  # [B, 4]
        focal_length = avg_calib[:, 0]  # [B] — use fx

        # Build flow_occs: [B, N-1, 3, H, W] (flow_x, flow_y, occ_mask)
        # Normalize flow to [-1, 1] range (AnyCam convention)
        flows_norm_x = flows_fwd[:, :, 0:1] * 2.0 / W  # [B, N-1, 1, H, W]
        flows_norm_y = flows_fwd[:, :, 1:2] * 2.0 / H
        flow_occs = torch.cat([flows_norm_x, flows_norm_y, occs_fwd], dim=2)  # [B, N-1, 3, H, W]

        # Pad flow_occs with zeros for last frame (convention: last frame has no flow)
        flow_occs_padded = torch.cat([
            flow_occs,
            torch.zeros(B, 1, 3, H, W, device=device)
        ], dim=1)  # [B, N, 3, H, W]

        # Convert inverse depth to the format AnyCam expects
        # Preprocessed: inverse_depth = 1 / (metric_depth * 0.1)
        # AnyCam: pred_depths = metric_depth * 0.1 = 1 / inverse_depth
        metric_depths_scaled = 1.0 / depths.clamp(min=1e-6)  # metric_depth * 0.1
        # AnyCam then multiplies by 0.1: pred_depths = depths * 0.1
        # But depths are already stored as inverse depth, and AnyCam does
        # depths = 1 / raw_depth  and then  pred_depths = depths * 0.1
        # So we need depths in the "raw inverse depth" format that when
        # multiplied by 0.1 gives the correct scale.
        # Actually: AnyCam stores raw inverse depth, then does pred_depths = depths * 0.1
        # Our preprocessed data stores: 1 / (metric * 0.1) which is the raw inverse depth.
        # So we pass it directly, and the multiplication by 0.1 happens in the trainer.
        anycam_depths = depths  # [B, N, 1, H, W] — raw inverse depth

        # Normalize focal length to AnyCam's NDC [-1, 1] convention.
        # AnyCam's normalize_proj does: fx_norm = 2 * fx_pixel / width
        # focal_length here is GT fx in pixel space at resolution W.
        focal_norm = 2.0 * focal_length / W

        # Create projection matrix
        proj = make_proj_from_focal_length(
            focal_norm.unsqueeze(1),  # [B, 1]
            aspect_ratio=H / W,
        )  # [B, 1, 3, 3]

        # Run pose predictor
        pose_result = self.pose_predictor(
            images,
            flow_occs=flow_occs_padded,
            depths=anycam_depths,
            external_focal_norm=focal_norm,
        )

        poses = pose_result["poses"]     # [B, N, nc, 4, 4]
        uncert = pose_result["uncert"]   # [B, N, nc, 2, H, W]

        # Keep single candidate
        if poses.dim() == 5 and poses.shape[2] > 1:
            poses = poses[:, :, 0:1]
        if uncert.dim() == 6 and uncert.shape[2] > 1:
            uncert = uncert[:, :, 0:1]

        # Induce flow from depths + projection + poses
        # depths needs shape [B, N, 1, 1, H, W] = [n, f, _, c, h, w]
        # anycam_depths is [B, N, 1, H, W], insert candidate dim at pos 2
        aligned_depths = anycam_depths.unsqueeze(2)  # [B, N, 1, 1, H, W]

        induced_flow, dist = induce_flow_dist(
            aligned_depths * 0.1,  # Scale as AnyCam expects
            proj,
            poses,
            flow_occs_padded[:, :, :2],
        )

        # Compute flow reprojection loss (matching AnyCam's PoseLoss exactly)
        EPS = 1e-4

        target_flow = flow_occs_padded[:, :-1, :2]  # [B, N-1, 2, H, W]
        induced_flow_sel = induced_flow[:, :-1, 0]   # [B, N-1, 2, H, W]
        invalid = flow_occs_padded[:, :-1, 2:3] < 0.5  # [B, N-1, 1, H, W]

        induced_clamped = induced_flow_sel.clamp(-1, 1)
        flow_error = F.l1_loss(induced_clamped, target_flow, reduction='none')
        flow_error = flow_error.mean(dim=2, keepdim=True).to(torch.float32)  # [B, N-1, 1, H, W]

        flow_loss_raw = flow_error.mean().detach()  # For monitoring (pre-uncertainty)

        # Uncertainty weighting (Laplacian NLL, same as AnyCam PoseLoss.compute_pose_loss)
        # uncert: [B, N, 1, 2, H, W] — channel 0 = flow uncertainty, channel 1 = dist uncertainty
        flow_uncert = uncert[:, :-1, 0, :1, :, :].to(torch.float32)  # [B, N-1, 1, H, W]
        flow_uncert = flow_uncert.clamp(min=0.01, max=10.0)
        flow_error = flow_error * (2 ** 0.5) / (flow_uncert + EPS) + (flow_uncert + EPS).log()
        flow_error = flow_error.clamp(max=10.0)  # Cap weighted error to prevent loss spikes

        # Apply occlusion mask (set invalid to 0, matching original PoseLoss)
        flow_error[invalid.expand_as(flow_error)] = 0
        flow_error[torch.isinf(flow_error) | torch.isnan(flow_error)] = 0

        consec_loss = flow_error.mean()

        # --- Composed flow loss (multi-frame consistency) ---
        # For sequences with N > 2, compose long-range flows and poses
        # to add supervision on pairs (0→2, 0→3, ...).
        # Uses same uncertainty weighting as consecutive loss for fair scaling.
        N_pairs = N - 1  # number of consecutive pairs
        composed_loss = torch.tensor(0.0, device=device)
        n_composed = 0

        # Source frame (frame 0) uncertainty for all composed pairs
        src_uncert = uncert[:, 0, 0, :1, :, :].to(torch.float32).clamp(min=0.01, max=10.0)  # [B, 1, H, W]

        if N_pairs > 1:
            for ahead in range(2, N_pairs + 1):
                # Compose observed flows: pixel-space flows_fwd[:, 0..ahead-1]
                range_flows = [flows_fwd[:, j] for j in range(ahead)]
                range_occs = [occs_fwd[:, j] for j in range(ahead)]
                comp_flow, comp_occ = _compose_flows(range_flows, range_occs)
                # comp_flow: [B, 2, H, W] pixel space, comp_occ: [B, 1, H, W]

                # Compose predicted poses: poses[:, 0..ahead-1, 0]
                pose_list = [poses[:, j, 0] for j in range(ahead)]
                comp_pose = _compose_poses(pose_list)  # [B, 4, 4]

                # Induce flow from composed pose + source depth (frame 0)
                # Set up 2-frame input: [composed_pose, identity]
                identity = torch.eye(4, device=device).unsqueeze(0).expand(B, -1, -1)
                pair_poses = torch.stack([comp_pose, identity], dim=1).unsqueeze(2)  # [B, 2, 1, 4, 4]
                src_depth = aligned_depths[:, 0:1] * 0.1  # [B, 1, 1, 1, H, W]
                pair_depths = torch.cat([src_depth, src_depth], dim=1)  # [B, 2, 1, 1, H, W]

                # Normalize composed flow for comparison
                comp_flow_norm_x = comp_flow[:, 0:1] * 2.0 / W
                comp_flow_norm_y = comp_flow[:, 1:2] * 2.0 / H
                comp_flow_norm = torch.cat([comp_flow_norm_x, comp_flow_norm_y], dim=1)  # [B, 2, H, W]
                pair_flow = torch.stack([comp_flow_norm, torch.zeros_like(comp_flow_norm)], dim=1)  # [B, 2, 2, H, W]

                comp_induced, _ = induce_flow_dist(pair_depths, proj, pair_poses, pair_flow)
                # comp_induced: [B, 2, 1, 2, H, W] — take frame 0
                comp_induced_sel = comp_induced[:, 0, 0].clamp(-1, 1)  # [B, 2, H, W]

                comp_err = F.l1_loss(comp_induced_sel, comp_flow_norm, reduction='none')
                comp_err = comp_err.mean(dim=1, keepdim=True).to(torch.float32)  # [B, 1, H, W]

                # Uncertainty weighting (same Laplacian NLL as consecutive loss)
                comp_err = comp_err * (2 ** 0.5) / (src_uncert + EPS) + (src_uncert + EPS).log()
                comp_err = comp_err.clamp(max=10.0)

                # Occlusion mask from composed occlusions
                comp_invalid = comp_occ < 0.5  # [B, 1, H, W]
                comp_err[comp_invalid.expand_as(comp_err)] = 0
                comp_err[torch.isinf(comp_err) | torch.isnan(comp_err)] = 0

                composed_loss = composed_loss + comp_err.mean()
                n_composed += 1

        if n_composed > 0:
            composed_loss = composed_loss / n_composed

        # Total flow loss: weighted combination of consecutive and composed
        # With uncertainty weighting, both are on the same scale.
        # lambda_comp=1.0 means equal weight per pair.
        flow_loss = consec_loss + self.lambda_comp * composed_loss

        result = {
            "loss": flow_loss,
            "flow_loss": flow_loss.detach(),
            "flow_loss_raw": flow_loss_raw,
            "consec_loss": consec_loss.detach(),
            "composed_loss": composed_loss.detach(),
            "poses": poses.detach(),
            "focal_length": focal_length.detach(),
            "induced_flow": induced_flow_sel.detach(),
            "target_flow": target_flow.detach(),
            "valid_mask": (~invalid).detach(),
        }

        return result

    def _forward_phase_b1(self, data: Dict) -> Dict:
        """
        Phase B1: FAT pre-training with reprojection loss.

        Inputs:
            images: [B, N, 3, H, W]
            calibs: [B, N, 4]  (fx, fy, cx, cy) — used as pseudo GT

        Returns dict with loss components.
        """
        images = data["images"]  # [B, N, 3, H, W]
        calibs = data["calibs"]  # [B, N, 4]

        B, N, C, H, W = images.shape
        device = images.device

        # Average calibration = pseudo ground truth
        avg_calib = calibs.mean(dim=1)  # [B, 4]

        total_loss = torch.tensor(0.0, device=device)
        all_fat_intrinsics = []

        for b in range(B):
            seq_images = images[b]  # [N, 3, H, W]

            # Run FAT pipeline (DINOv2 ViT-L → FAT → decoder → rays)
            with torch.amp.autocast(device_type='cuda', enabled=False):
                result = self.fat_model(seq_images.float(), cam_id="pinhole")

            rays = result["rays"]           # [1, H_ray*W_ray, 3]
            image_size = result["image_size"]  # (H_ray, W_ray)

            # Extract predicted intrinsics for validation monitoring
            intrinsics = result["intrinsics"][0]
            if isinstance(intrinsics, Tensor):
                all_fat_intrinsics.append(intrinsics.detach())
            else:
                all_fat_intrinsics.append(
                    torch.tensor(intrinsics, device=device, dtype=torch.float32)
                )

            # Compute reprojection loss against pseudo GT calibration
            loss, info = self.fat_model.compute_reprojection_loss(
                predicted_rays=rays[0],
                average_intrinsics=avg_calib[b],
                ray_image_size=image_size,
                original_image_size=(H, W),
            )
            total_loss = total_loss + loss

        total_loss = total_loss / B

        return {
            "loss": total_loss,
            "calib_loss": total_loss.detach(),
            "fat_intrinsics": torch.stack(all_fat_intrinsics),  # [B, 4]
        }

    def _forward_combined(self, data: Dict) -> Dict:
        """
        Phase B2/C: Combined FAT calibration + pose pipeline forward pass.

        Inputs:
            images:    [B, N, 3, H, W]
            depths:    [B, N, 1, H, W]
            flows_fwd: [B, N-1, 2, H, W]
            occs_fwd:  [B, N-1, 1, H, W]
            calibs:    [B, N, 4]

        Returns dict with loss components.
        """
        from anycam.trainer import induce_flow_dist, make_proj_from_focal_length

        images = data["images"]
        depths = data["depths"]
        flows_fwd = data["flows_fwd"]
        occs_fwd = data["occs_fwd"]
        calibs = data["calibs"]

        B, N, C, H, W = images.shape
        device = images.device

        # Average calib for anchor loss
        avg_calib = calibs.mean(dim=1)  # [B, 4]

        # --- Step 1: FAT calibration ---
        all_fat_intrinsics = []
        all_rays = []
        all_image_sizes = []

        fat_success = []
        for b in range(B):
            seq_images = images[b]  # [N, 3, H, W]

            with torch.amp.autocast(device_type='cuda', enabled=False):
                calib_result = self.fat_model(seq_images.float(), cam_id="pinhole")

            success_b = calib_result["success"].all().item()
            fat_success.append(success_b)

            intrinsics = calib_result["intrinsics"][0]
            if isinstance(intrinsics, Tensor):
                intr_tensor = intrinsics
            else:
                intr_tensor = torch.tensor(intrinsics, device=device, dtype=torch.float32)

            # If calibrator failed, fall back to average GT calibration (detached)
            if not success_b:
                intr_tensor = avg_calib[b].detach().clone()

            all_fat_intrinsics.append(intr_tensor)
            all_rays.append(calib_result["rays"])        # [1, H*W, 3]
            all_image_sizes.append(calib_result["image_size"])

        batch_intrinsics = torch.stack(all_fat_intrinsics)  # [B, 4]
        focal_length = batch_intrinsics[:, 0]               # [B] — fx from FAT

        # --- Step 2: Flow reprojection loss (pose) ---
        # Normalize flow
        flows_norm_x = flows_fwd[:, :, 0:1] * 2.0 / W
        flows_norm_y = flows_fwd[:, :, 1:2] * 2.0 / H
        flow_occs = torch.cat([flows_norm_x, flows_norm_y, occs_fwd], dim=2)
        flow_occs_padded = torch.cat([
            flow_occs,
            torch.zeros(B, 1, 3, H, W, device=device)
        ], dim=1)

        anycam_depths = depths

        # Normalize focal length to AnyCam's NDC [-1, 1] convention.
        # focal_length is FAT fx in pixel space at H_ray x W_ray resolution.
        # For 336x336 input (divisible by 14): H_ray = W_ray = 336 (same as input).
        H_ray, W_ray = all_image_sizes[0]
        focal_norm = 2.0 * focal_length / W_ray
        proj = make_proj_from_focal_length(
            focal_norm.unsqueeze(1),
            aspect_ratio=H / W,
        )

        pose_result = self.pose_predictor(
            images,
            flow_occs=flow_occs_padded,
            depths=anycam_depths,
            external_focal_norm=focal_norm,
        )

        poses = pose_result["poses"]
        uncert = pose_result["uncert"]

        if poses.dim() == 5 and poses.shape[2] > 1:
            poses = poses[:, :, 0:1]
        if uncert.dim() == 6 and uncert.shape[2] > 1:
            uncert = uncert[:, :, 0:1]

        # depths needs shape [B, N, 1, 1, H, W] = [n, f, _, c, h, w]
        # anycam_depths is [B, N, 1, H, W], insert candidate dim at pos 2
        aligned_depths = anycam_depths.unsqueeze(2)

        induced_flow, dist = induce_flow_dist(
            aligned_depths * 0.1,
            proj,
            poses,
            flow_occs_padded[:, :, :2],
        )

        # Compute flow reprojection loss (matching AnyCam's PoseLoss exactly)
        EPS = 1e-4

        target_flow = flow_occs_padded[:, :-1, :2]  # [B, N-1, 2, H, W]
        induced_flow_sel = induced_flow[:, :-1, 0]   # [B, N-1, 2, H, W]
        invalid = flow_occs_padded[:, :-1, 2:3] < 0.5  # [B, N-1, 1, H, W]

        induced_clamped = induced_flow_sel.clamp(-1, 1)
        flow_error = F.l1_loss(induced_clamped, target_flow, reduction='none')
        flow_error = flow_error.mean(dim=2, keepdim=True).to(torch.float32)  # [B, N-1, 1, H, W]

        flow_loss_raw = flow_error.mean().detach()  # For monitoring (pre-uncertainty)

        # Uncertainty weighting (Laplacian NLL, same as AnyCam PoseLoss.compute_pose_loss)
        flow_uncert = uncert[:, :-1, 0, :1, :, :].to(torch.float32)  # [B, N-1, 1, H, W]
        flow_uncert = flow_uncert.clamp(min=0.01, max=10.0)
        flow_error = flow_error * (2 ** 0.5) / (flow_uncert + EPS) + (flow_uncert + EPS).log()
        flow_error = flow_error.clamp(max=10.0)  # Cap weighted error to prevent loss spikes

        # Apply occlusion mask (set invalid to 0, matching original PoseLoss)
        flow_error[invalid.expand_as(flow_error)] = 0
        flow_error[torch.isinf(flow_error) | torch.isnan(flow_error)] = 0

        consec_loss = flow_error.mean()

        # --- Composed flow loss (multi-frame consistency) ---
        N_pairs = N - 1
        composed_loss = torch.tensor(0.0, device=device)
        n_composed = 0

        if N_pairs > 1:
            # Source frame (frame 0) uncertainty for all composed pairs
            src_uncert = uncert[:, 0, 0, :1, :, :].to(torch.float32).clamp(min=0.01, max=10.0)

            for ahead in range(2, N_pairs + 1):
                range_flows = [flows_fwd[:, j] for j in range(ahead)]
                range_occs = [occs_fwd[:, j] for j in range(ahead)]
                comp_flow, comp_occ = _compose_flows(range_flows, range_occs)

                pose_list = [poses[:, j, 0] for j in range(ahead)]
                comp_pose = _compose_poses(pose_list)

                identity = torch.eye(4, device=device).unsqueeze(0).expand(B, -1, -1)
                pair_poses = torch.stack([comp_pose, identity], dim=1).unsqueeze(2)
                src_depth = aligned_depths[:, 0:1] * 0.1
                pair_depths = torch.cat([src_depth, src_depth], dim=1)

                comp_flow_norm_x = comp_flow[:, 0:1] * 2.0 / W
                comp_flow_norm_y = comp_flow[:, 1:2] * 2.0 / H
                comp_flow_norm = torch.cat([comp_flow_norm_x, comp_flow_norm_y], dim=1)
                pair_flow = torch.stack([comp_flow_norm, torch.zeros_like(comp_flow_norm)], dim=1)

                comp_induced, _ = induce_flow_dist(pair_depths, proj, pair_poses, pair_flow)
                comp_induced_sel = comp_induced[:, 0, 0].clamp(-1, 1)

                comp_err = F.l1_loss(comp_induced_sel, comp_flow_norm, reduction='none')
                comp_err = comp_err.mean(dim=1, keepdim=True).to(torch.float32)

                # Uncertainty weighting (Laplacian NLL, same scale as consecutive loss)
                comp_err = comp_err * (2 ** 0.5) / (src_uncert + EPS) + (src_uncert + EPS).log()
                comp_err = comp_err.clamp(max=10.0)

                comp_invalid = comp_occ < 0.5
                comp_err[comp_invalid.expand_as(comp_err)] = 0
                comp_err[torch.isinf(comp_err) | torch.isnan(comp_err)] = 0

                composed_loss = composed_loss + comp_err.mean()
                n_composed += 1

        if n_composed > 0:
            composed_loss = composed_loss / n_composed

        flow_loss = consec_loss + self.lambda_comp * composed_loss

        # --- Step 3: Calibration anchor loss ---
        calib_loss = torch.tensor(0.0, device=device)
        n_calib_valid = 0
        for b in range(B):
            if not fat_success[b]:
                continue  # Skip failed calibrations — don't backprop garbage
            rays = all_rays[b]
            image_size = all_image_sizes[b]
            loss_b, _ = self.fat_model.compute_reprojection_loss(
                predicted_rays=rays[0],
                average_intrinsics=avg_calib[b],
                ray_image_size=image_size,
                original_image_size=(H, W),
            )
            calib_loss = calib_loss + loss_b
            n_calib_valid += 1
        if n_calib_valid > 0:
            calib_loss = calib_loss / n_calib_valid

        return {
            "flow_loss": flow_loss,
            "flow_loss_raw": flow_loss_raw,
            "consec_loss": consec_loss.detach(),
            "composed_loss": composed_loss.detach(),
            "calib_loss": calib_loss,
            "poses": poses.detach(),
            "focal_length": focal_length.detach(),
            "fat_intrinsics": batch_intrinsics.detach(),
            "avg_calib": avg_calib.detach(),
            "induced_flow": induced_flow_sel.detach(),
            "target_flow": target_flow.detach(),
            "valid_mask": (~invalid).detach(),
        }

    # ------------------------------------------------------------------
    # Trainable parameters
    # ------------------------------------------------------------------

    def get_trainable_parameters(self) -> List[nn.Parameter]:
        """Return only the parameters that should be optimized."""
        return [p for p in self.parameters() if p.requires_grad]

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _print_param_summary(self):
        """Print trainable/total parameter counts."""
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        pct = 100 * trainable / total if total > 0 else 0
        logger.info(f"[{self.phase}] Trainable: {trainable:,} / {total:,} ({pct:.2f}%)")

    def train(self, mode: bool = True):
        """Override to respect freezing — frozen modules stay in eval mode."""
        super().train(mode)

        if self.phase == 'A':
            # Backbone always eval
            if self.pose_predictor is not None:
                self.pose_predictor.backbone.eval()

        elif self.phase == 'B1':
            if self.fat_model is not None:
                self.fat_model.backbone.eval()
                self.fat_model.decoder.eval()
                self.fat_model.head.eval()

        elif self.phase == 'B2':
            # Everything frozen except FAT adapter
            if self.fat_model is not None:
                self.fat_model.backbone.eval()
                self.fat_model.decoder.eval()
                self.fat_model.head.eval()
            if self.pose_predictor is not None:
                self.pose_predictor.eval()  # Entire pose pipeline in eval

        elif self.phase in ('C', 'Ca'):
            # Joint training: only pose_head + FAT trainable, backbones in eval
            if self.pose_predictor is not None:
                self.pose_predictor.backbone.eval()
                self.pose_predictor.neck.eval()
                self.pose_predictor.head.eval()
            if self.fat_model is not None:
                self.fat_model.backbone.eval()
                self.fat_model.decoder.eval()
                self.fat_model.head.eval()

        elif self.phase == 'Cb':
            # Like Ca but pose neck is trainable — only backbones + depth head in eval
            if self.pose_predictor is not None:
                self.pose_predictor.backbone.eval()
                self.pose_predictor.head.eval()  # depth uncertainty head stays eval
            if self.fat_model is not None:
                self.fat_model.backbone.eval()
                self.fat_model.decoder.eval()
                self.fat_model.head.eval()

        elif self.phase in ('Da', 'Db'):
            # Pose-only: both backbones eval, uncertainty eval, all FAT eval
            if self.pose_predictor is not None:
                self.pose_predictor.backbone.eval()
                self.pose_predictor.neck.eval()
                self.pose_predictor.head.eval()
            if self.fat_model is not None:
                self.fat_model.eval()  # Entire FAT pipeline in eval

        return self
