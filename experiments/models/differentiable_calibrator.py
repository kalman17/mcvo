"""
Differentiable Camera Calibrator using Implicit Differentiation.

This module implements a differentiable camera intrinsics estimation where:
- RANSAC is used only to determine inliers (no gradients)
- Final intrinsics are computed via weighted least squares
- Gradients flow only through the least-squares solution, not RANSAC

Key insight: The solution of a least-squares problem is differentiable w.r.t.
its inputs even if those inputs were selected by a non-differentiable process,
as long as the selection is treated as constant.

Mathematical formulation:
    x = argmin_x ||W(Ax - b)||²
    x = (A^T W A)^{-1} A^T W b

Where:
    A = design matrix from pixel coordinates (constant)
    b = tangent coords from predicted rays (differentiable)
    W = diagonal weight matrix from RANSAC (constant, no gradients)
    x = intrinsic parameters (differentiable w.r.t. rays)
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple, Dict, Optional


class DifferentiableCalibrator(nn.Module):
    """
    Differentiable pinhole camera calibrator.

    Uses RANSAC for robust inlier detection (no gradients),
    then solves weighted least squares for intrinsics (with gradients).
    """

    def __init__(
        self,
        inlier_weight: float = 1.0,
        outlier_weight: float = 1e-6,
        ransac_threshold_degrees: float = 1.0,
        regularization_lambda: float = 0.01,
        damping: float = 1e-6,  # For numerical stability in matrix inversion
    ):
        super().__init__()
        self.inlier_weight = inlier_weight
        self.outlier_weight = outlier_weight
        self.ransac_threshold_degrees = ransac_threshold_degrees
        self.regularization_lambda = regularization_lambda
        self.damping = damping

    def build_design_matrix(self, H: int, W: int, device: torch.device) -> Tensor:
        """
        Build design matrix A from pixel coordinates.

        For pinhole model, we solve for [p, q, r, s] where:
            p = 1/fx, q = cx/fx, r = 1/fy, s = cy/fy

        The equations are:
            rx/rz = u*p - q  →  [u, -1, 0, 0] @ [p,q,r,s] = rx/rz
            ry/rz = v*r - s  →  [0, 0, v, -1] @ [p,q,r,s] = ry/rz

        Args:
            H, W: Image dimensions
            device: torch device

        Returns:
            A: [2*H*W, 4] design matrix
        """
        # Create pixel coordinate grids
        v_coords, u_coords = torch.meshgrid(
            torch.arange(H, device=device, dtype=torch.float32),
            torch.arange(W, device=device, dtype=torch.float32),
            indexing='ij'
        )
        u_flat = u_coords.flatten()  # [H*W]
        v_flat = v_coords.flatten()  # [H*W]

        N = H * W

        # Build design matrix rows:
        # Row for x: [u, -1, 0, 0]
        # Row for y: [0, 0, v, -1]
        A = torch.zeros(2 * N, 4, device=device, dtype=torch.float32)

        # X equations: [u, -1, 0, 0]
        A[:N, 0] = u_flat      # u coefficient for p
        A[:N, 1] = -1.0        # coefficient for q

        # Y equations: [0, 0, v, -1]
        A[N:, 2] = v_flat      # v coefficient for r
        A[N:, 3] = -1.0        # coefficient for s

        return A

    def rays_to_tangent(self, rays: Tensor) -> Tensor:
        """
        Convert 3D rays to 2D tangent coordinates.

        Args:
            rays: [N, 3] normalized ray directions

        Returns:
            tangent: [N, 2] tangent coordinates (rx/rz, ry/rz)
        """
        # Avoid division by zero
        rz = rays[:, 2:3].clamp(min=1e-6)
        tangent = rays[:, :2] / rz
        return tangent

    def solve_weighted_lstsq(
        self,
        A: Tensor,  # [M, 4] design matrix (constant)
        b: Tensor,  # [M] target (differentiable)
        W: Tensor,  # [M] weights (constant)
    ) -> Tensor:
        """
        Solve weighted least squares: x = argmin ||W(Ax - b)||²

        Closed-form solution: x = (A^T W² A)^{-1} A^T W² b

        This is DIFFERENTIABLE w.r.t. b because:
        - A and W are treated as constants
        - The inverse and matrix multiplications are differentiable

        Args:
            A: [M, 4] design matrix
            b: [M] target vector (has gradients)
            W: [M] weight vector (no gradients)

        Returns:
            x: [4] solution vector (has gradients w.r.t. b)
        """
        # W² for weighted least squares (since ||Wx||² = x^T W² x)
        W_sq = W * W  # [M]

        # A^T W² A  [4, 4]
        # We compute: (A.T @ diag(W²) @ A)
        AW = A * W_sq.unsqueeze(1)  # [M, 4] - each row of A weighted by W²
        AtWA = A.t() @ AW  # [4, 4]

        # Add damping for numerical stability (Tikhonov regularization)
        AtWA = AtWA + self.damping * torch.eye(4, device=A.device, dtype=A.dtype)

        # A^T W² b  [4]
        AtWb = A.t() @ (W_sq * b)  # [4]

        # Solve: x = (A^T W² A)^{-1} A^T W² b
        # Use torch.linalg.solve for numerical stability
        x = torch.linalg.solve(AtWA, AtWb)

        return x

    def params_to_intrinsics(self, params: Tensor) -> Tensor:
        """
        Convert least-squares parameters to camera intrinsics.

        Args:
            params: [4] tensor [p, q, r, s] where:
                p = 1/fx, q = cx/fx, r = 1/fy, s = cy/fy

        Returns:
            intrinsics: [4] tensor [fx, fy, cx, cy]
        """
        p, q, r, s = params[0], params[1], params[2], params[3]

        # Avoid division by zero
        p = p.clamp(min=1e-8)
        r = r.clamp(min=1e-8)

        fx = 1.0 / p
        fy = 1.0 / r
        cx = q / p  # q = cx/fx → cx = q * fx = q/p
        cy = s / r  # s = cy/fy → cy = s * fy = s/r

        return torch.stack([fx, fy, cx, cy])

    def forward(
        self,
        predicted_rays: Tensor,  # [H*W, 3] normalized rays (has gradients)
        image_size: Tuple[int, int],  # (H, W)
        ransac_intrinsics: Optional[Tensor] = None,  # [4] from RANSAC (detached)
        inlier_mask: Optional[Tensor] = None,  # [H*W] boolean (detached)
    ) -> Dict:
        """
        Compute differentiable camera intrinsics from predicted rays.

        Args:
            predicted_rays: [H*W, 3] predicted ray directions (WITH gradients)
            image_size: (H, W) tuple
            ransac_intrinsics: [4] intrinsics from RANSAC (for regularization, detached)
            inlier_mask: [H*W] boolean inlier mask from RANSAC (detached)

        Returns:
            Dict with:
                'intrinsics': [4] differentiable intrinsics [fx, fy, cx, cy]
                'params': [4] raw least-squares parameters
                'inlier_ratio': float
        """
        H, W = image_size
        N = H * W
        device = predicted_rays.device

        # Normalize rays
        predicted_rays = torch.nn.functional.normalize(predicted_rays, dim=-1)

        # Build design matrix (constant, no gradients)
        A = self.build_design_matrix(H, W, device)  # [2N, 4]

        # Convert rays to tangent coordinates (DIFFERENTIABLE)
        tangent = self.rays_to_tangent(predicted_rays)  # [N, 2]

        # Build target vector b = [tx_1, tx_2, ..., tx_N, ty_1, ty_2, ..., ty_N]
        b = torch.cat([tangent[:, 0], tangent[:, 1]], dim=0)  # [2N]

        # Create weight vector from inlier mask (constant, no gradients)
        if inlier_mask is not None:
            # Expand mask for both x and y equations
            inlier_mask_2d = torch.cat([inlier_mask, inlier_mask], dim=0)  # [2N]
            W = torch.where(
                inlier_mask_2d,
                torch.tensor(self.inlier_weight, device=device),
                torch.tensor(self.outlier_weight, device=device)
            )
        else:
            # All rays weighted equally
            W = torch.ones(2 * N, device=device)

        # Solve weighted least squares (DIFFERENTIABLE w.r.t. b)
        params = self.solve_weighted_lstsq(A, b, W)  # [4]

        # Convert to intrinsics (DIFFERENTIABLE)
        intrinsics = self.params_to_intrinsics(params)  # [4]

        # Compute inlier ratio for logging
        inlier_ratio = inlier_mask.float().mean().item() if inlier_mask is not None else 1.0

        return {
            'intrinsics': intrinsics,
            'params': params,
            'inlier_ratio': inlier_ratio,
        }


def compute_differentiable_calibration_loss(
    predicted_intrinsics: Tensor,  # [4] - differentiable
    ransac_intrinsics: Tensor,  # [4] - detached
    gt_intrinsics: Optional[Tensor] = None,  # [4] - ground truth if available
    regularization_lambda: float = 0.1,
) -> Tuple[Tensor, Dict]:
    """
    Compute loss for differentiable calibration.

    The main loss can be either:
    1. MSE to ground truth intrinsics (if available)
    2. Some downstream task loss (e.g., flow reprojection)

    We always add regularization to anchor the solution near RANSAC output.

    Args:
        predicted_intrinsics: [4] differentiable intrinsics
        ransac_intrinsics: [4] RANSAC intrinsics (detached, for regularization)
        gt_intrinsics: [4] optional ground truth
        regularization_lambda: weight for regularization term

    Returns:
        loss: scalar tensor
        info: dict with loss components
    """
    info = {}

    # Main loss
    if gt_intrinsics is not None:
        main_loss = torch.nn.functional.mse_loss(predicted_intrinsics, gt_intrinsics)
        info['gt_loss'] = main_loss.item()
    else:
        # Placeholder - in practice, use downstream loss (e.g., pose estimation)
        main_loss = torch.tensor(0.0, device=predicted_intrinsics.device)

    # Regularization: anchor to RANSAC solution (no gradients through ransac_intrinsics)
    reg_loss = torch.nn.functional.mse_loss(
        predicted_intrinsics,
        ransac_intrinsics.detach()
    )
    info['reg_loss'] = reg_loss.item()

    # Total loss
    total_loss = main_loss + regularization_lambda * reg_loss
    info['total_loss'] = total_loss.item()

    return total_loss, info
