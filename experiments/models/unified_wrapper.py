"""
Unified Training Wrapper for all training phases (A, B1, B3, C).

Configures model components, freezing, and forward passes per phase:
  - Phase A:  Pose head only. AnyCam DINOv2-small (frozen) runs live.
              Depth, flow, calib loaded from preprocessed .npz.
  - Phase B1: FAT pre-training. AnyCalib DINOv2 ViT-L (frozen) runs live.
              Calib from .npz used as pseudo GT for reprojection loss.
  - Phase B3: FAT + pose head joint training. Both backbones run live (frozen).
              Depth, flow from .npz. Combined flow + calib loss.
  - Phase C:  End-to-end alternating. Same as B3 but backbones unfrozen
              during their respective component's training turn.
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


class UnifiedTrainingWrapper(nn.Module):
    """
    Unified model wrapper that configures itself per training phase.

    Args:
        phase: Training phase ('A', 'B1', 'B3', 'C').
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

        # These will be initialized per-phase
        self.pose_predictor = None      # AnyCam model (DINOv2-small backbone + pose head)
        self.fat_model = None           # AnyCalibWithFAT (DINOv2 ViT-L + FAT + decoder)

        if phase == 'A':
            self._init_phase_a(anycam_config_path)
        elif phase == 'B1':
            self._init_phase_b1()
        elif phase in ('B3', 'C'):
            self._init_phase_b3_c(anycam_config_path)
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

    def _init_phase_b3_c(self, config_path: str):
        """Phase B3/C: Both AnyCam + AnyCalibWithFAT."""
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

        # Default freezing for B3: everything frozen except FAT + pose head
        self._freeze_for_b3()

        mode = "B3" if self.phase == "B3" else "C"
        logger.info(f"[Phase {mode}] Both pipelines loaded.")

    def _freeze_for_b3(self):
        """B3 default: freeze everything, unfreeze FAT + pose head."""
        for param in self.parameters():
            param.requires_grad = False

        if self.fat_model is not None and self.fat_model.fat is not None:
            for param in self.fat_model.fat.parameters():
                param.requires_grad = True

        if self.pose_predictor is not None:
            for param in self.pose_predictor.pose_head.parameters():
                param.requires_grad = True

    # ------------------------------------------------------------------
    # Phase C alternating training modes
    # ------------------------------------------------------------------

    def set_training_mode(self, mode: str):
        """
        Configure parameter freezing for Phase C alternating training.

        Args:
            mode: 'pose' — unfreeze AnyCam DINOv2-small + pose head, freeze calib.
                  'calib' — unfreeze AnyCalib DINOv2 ViT-L + FAT + decoder, freeze AnyCam.
        """
        if self.phase != 'C':
            logger.warning(f"set_training_mode called outside Phase C (current: {self.phase})")
            return

        # Freeze everything first
        for param in self.parameters():
            param.requires_grad = False

        if mode == 'pose':
            # Unfreeze AnyCam backbone + pose head
            if self.pose_predictor is not None:
                for param in self.pose_predictor.parameters():
                    param.requires_grad = True
            logger.info("[Phase C] Mode: POSE — AnyCam unfrozen, calibration frozen.")

        elif mode == 'calib':
            # Unfreeze AnyCalib ViT-L backbone + FAT + decoder + ray head
            if self.fat_model is not None:
                for param in self.fat_model.backbone.parameters():
                    param.requires_grad = True
                if self.fat_model.fat is not None:
                    for param in self.fat_model.fat.parameters():
                        param.requires_grad = True
                for param in self.fat_model.decoder.parameters():
                    param.requires_grad = True
                for param in self.fat_model.head.parameters():
                    param.requires_grad = True
            logger.info("[Phase C] Mode: CALIB — calibration pipeline unfrozen, AnyCam frozen.")
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'pose' or 'calib'.")

    # ------------------------------------------------------------------
    # Checkpoint loading
    # ------------------------------------------------------------------

    def load_phase_checkpoint(self, checkpoint_path: str, source_phase: str):
        """
        Load weights from a previous phase checkpoint.

        Args:
            checkpoint_path: Path to .pt checkpoint file.
            source_phase: Which phase produced this checkpoint ('A', 'B1', 'B3').
        """
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        state = ckpt.get("model_state_dict", ckpt)

        if source_phase == 'A':
            # Load pose head weights into pose_predictor
            pose_keys = {k: v for k, v in state.items() if k.startswith("pose_predictor.")}
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
            if fat_keys:
                missing, unexpected = self.load_state_dict(fat_keys, strict=False)
                logger.info(f"Loaded Phase B1 FAT: {len(fat_keys)} keys, "
                            f"{len(missing)} missing, {len(unexpected)} unexpected")
            else:
                logger.warning("No FAT keys found in Phase B1 checkpoint")

        elif source_phase == 'B3':
            # Load full model (FAT + pose head)
            missing, unexpected = self.load_state_dict(state, strict=False)
            logger.info(f"Loaded Phase B3 full model: {len(missing)} missing, "
                        f"{len(unexpected)} unexpected")
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
        elif self.phase in ('B3', 'C'):
            return self._forward_phase_b3(data)
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

        # Normalize focal length: AnyCam expects focal / width style
        focal_norm = focal_length / W

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
        flow_uncert = flow_uncert.clamp_min(EPS)
        flow_error = flow_error * (2 ** 0.5) / (flow_uncert + EPS) + (flow_uncert + EPS).log()

        # Apply occlusion mask (set invalid to 0, matching original PoseLoss)
        flow_error[invalid.expand_as(flow_error)] = 0
        flow_error[torch.isinf(flow_error) | torch.isnan(flow_error)] = 0

        flow_loss = flow_error.mean()

        result = {
            "loss": flow_loss,
            "flow_loss": flow_loss.detach(),
            "flow_loss_raw": flow_loss_raw,
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
        batch_info = {}

        for b in range(B):
            seq_images = images[b]  # [N, 3, H, W]

            # Run FAT pipeline (DINOv2 ViT-L → FAT → decoder → rays)
            with torch.amp.autocast(device_type='cuda', enabled=False):
                result = self.fat_model(seq_images.float(), cam_id="pinhole")

            rays = result["rays"]           # [1, H_ray*W_ray, 3]
            image_size = result["image_size"]  # (H_ray, W_ray)

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
        }

    def _forward_phase_b3(self, data: Dict) -> Dict:
        """
        Phase B3/C: Joint FAT + pose training with combined loss.

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

        for b in range(B):
            seq_images = images[b]  # [N, 3, H, W]

            with torch.amp.autocast(device_type='cuda', enabled=False):
                calib_result = self.fat_model(seq_images.float(), cam_id="pinhole")

            intrinsics = calib_result["intrinsics"][0]
            if isinstance(intrinsics, Tensor):
                intr_tensor = intrinsics
            else:
                intr_tensor = torch.tensor(intrinsics, device=device, dtype=torch.float32)

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

        focal_norm = focal_length / W
        proj = make_proj_from_focal_length(
            focal_norm.unsqueeze(1),
            aspect_ratio=H / W,
        )

        pose_result = self.pose_predictor(
            images,
            flow_occs=flow_occs_padded,
            depths=anycam_depths,
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
        flow_uncert = flow_uncert.clamp_min(EPS)
        flow_error = flow_error * (2 ** 0.5) / (flow_uncert + EPS) + (flow_uncert + EPS).log()

        # Apply occlusion mask (set invalid to 0, matching original PoseLoss)
        flow_error[invalid.expand_as(flow_error)] = 0
        flow_error[torch.isinf(flow_error) | torch.isnan(flow_error)] = 0

        flow_loss = flow_error.mean()

        # --- Step 3: Calibration anchor loss ---
        calib_loss = torch.tensor(0.0, device=device)
        for b in range(B):
            rays = all_rays[b]
            image_size = all_image_sizes[b]
            loss_b, _ = self.fat_model.compute_reprojection_loss(
                predicted_rays=rays[0],
                average_intrinsics=avg_calib[b],
                ray_image_size=image_size,
                original_image_size=(H, W),
            )
            calib_loss = calib_loss + loss_b
        calib_loss = calib_loss / B

        return {
            "flow_loss": flow_loss,
            "flow_loss_raw": flow_loss_raw,
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

        elif self.phase == 'B3':
            if self.fat_model is not None:
                self.fat_model.backbone.eval()
                self.fat_model.decoder.eval()
                self.fat_model.head.eval()
            if self.pose_predictor is not None:
                self.pose_predictor.backbone.eval()

        elif self.phase == 'C':
            # In Phase C, eval/train depends on which mode is active.
            # After set_training_mode(), frozen components are eval.
            # We just ensure frozen params don't accumulate BN stats.
            pass

        return self
