"""Self-supervised flow-reprojection loss for MCVO.

Same math as AnyCam / unified_wrapper phase A: induce flow from (cached teacher
depth, focal, predicted pose), compare against cached teacher flow with a learned
per-pixel Laplacian-NLL uncertainty weighting. Teachers appear ONLY here — the model
input is raw images.
"""

from typing import Dict

import torch
import torch.nn.functional as F

from anycam.trainer import induce_flow_dist, make_proj_from_focal_length

EPS = 1e-4


def mcvo_selfsup_loss(out: Dict, data: Dict) -> Dict:
    """out: MCVO forward output. data: batch from PreprocessedMultiFrameDataset
    (phase-A fields: images, depths, flows_fwd, occs_fwd, calibs)."""
    images = data["images"]
    depths = data["depths"]          # [B, N, 1, H, W] raw inverse depth
    flows_fwd = data["flows_fwd"]    # [B, N-1, 2, H, W] pixel flow
    occs_fwd = data["occs_fwd"]      # [B, N-1, 1, H, W]
    calibs = data["calibs"]          # [B, N, 4]

    B, N, C, H, W = images.shape
    device = images.device

    # teacher focal (cached per-frame AnyCalib) -> normalized proj
    focal_norm = 2.0 * calibs.mean(dim=1)[:, 0] / W
    proj = make_proj_from_focal_length(focal_norm.unsqueeze(1), aspect_ratio=H / W)

    # normalized flow + occ (AnyCam convention), padded to N
    flow_occs = torch.cat([
        flows_fwd[:, :, 0:1] * 2.0 / W,
        flows_fwd[:, :, 1:2] * 2.0 / H,
        occs_fwd,
    ], dim=2)
    flow_occs = torch.cat([flow_occs, torch.zeros(B, 1, 3, H, W, device=device)], dim=1)

    poses = out["poses"]             # [B, N, 1, 4, 4]
    uncert = out["uncert"]           # [B, N, 1, 1, H, W]

    aligned_depths = depths.unsqueeze(2)  # [B, N, 1, 1, H, W]
    induced_flow, _ = induce_flow_dist(
        aligned_depths * 0.1, proj, poses, flow_occs[:, :, :2],
    )

    target_flow = flow_occs[:, :-1, :2]
    induced_sel = induced_flow[:, :-1, 0].clamp(-1, 1)
    # teacher data can contain NaNs (e.g. gaps in cached flow/depth) — mask them out
    bad_teacher = (~torch.isfinite(target_flow)).any(dim=2, keepdim=True) \
        | (~torch.isfinite(induced_sel)).any(dim=2, keepdim=True)
    target_flow = torch.nan_to_num(target_flow, nan=0.0, posinf=0.0, neginf=0.0)
    induced_sel = torch.nan_to_num(induced_sel, nan=0.0, posinf=0.0, neginf=0.0)
    invalid = (flow_occs[:, :-1, 2:3] < 0.5) | bad_teacher

    flow_error = F.l1_loss(induced_sel, target_flow, reduction="none")
    flow_error = flow_error.mean(dim=2, keepdim=True).float()   # [B, N-1, 1, H, W]
    flow_loss_raw = torch.nan_to_num(flow_error.detach().clone(), nan=0.0)
    flow_loss_raw[invalid.expand_as(flow_loss_raw)] = 0
    flow_loss_raw = flow_loss_raw.mean()

    u = uncert[:, :-1, 0, :1].float().clamp(min=0.01, max=10.0)  # [B, N-1, 1, H, W]
    err = flow_error * (2 ** 0.5) / (u + EPS) + (u + EPS).log()
    err = err.clamp(max=10.0)
    err[invalid.expand_as(err)] = 0
    err[torch.isinf(err) | torch.isnan(err)] = 0

    loss = err.mean()
    return {"loss": loss, "flow_loss_raw": flow_loss_raw}


@torch.no_grad()
def identity_baseline_loss(data: Dict) -> torch.Tensor:
    """flow_loss_raw a zero-motion predictor would get (induced flow = 0)."""
    flows_fwd = data["flows_fwd"]
    occs = data["occs_fwd"]
    H, W = flows_fwd.shape[-2:]
    tgt = torch.cat([flows_fwd[:, :, 0:1] * 2.0 / W, flows_fwd[:, :, 1:2] * 2.0 / H], dim=2)
    err = tgt.abs().mean(dim=2, keepdim=True)
    err = err * (occs >= 0.5)
    return err.mean()


def pose_distill_loss(out, pseudo_poses, trans_weight: float = 5.0):
    """FVO-style pose loss against teacher pseudo-poses, with learned per-pair
    heteroscedastic confidences: L = L_rot*e^{-cR} + cR + w*(L_t*e^{-ct} + ct).

    pseudo_poses: [B, N-1, 4, 4] teacher relative poses (same convention as ours:
    P maps cam_i points to cam_{i+1} — both come from the same induce_flow geometry).
    """
    # SUPERSEDED (2026-08-11): AnyCam FF proved to be a useless teacher on this data
    # (only 4% better than zero-motion), so distillation was abandoned in favour of the
    # epipolar term below. Kept for reference; requires a model exposing conf_logits.
    if "conf_logits" not in out:
        raise RuntimeError("pose_distill_loss needs conf_logits; the pose head was "
                           "reverted to 7 outputs when distillation was abandoned.")
    P = out["poses"][:, :-1, 0]                    # [B, N-1, 4, 4]
    C = out["conf_logits"]                          # [B, N-1, 2]
    n_pairs = min(P.shape[1], pseudo_poses.shape[1])
    P, T, C = P[:, :n_pairs], pseudo_poses[:, :n_pairs], C[:, :n_pairs]

    # mask out non-finite teachers (rare corrupt files)
    finite = torch.isfinite(T).flatten(2).all(-1)   # [B, n]
    R_rel = torch.matmul(P[..., :3, :3].transpose(-1, -2), T[..., :3, :3])
    cos = ((R_rel[..., 0, 0] + R_rel[..., 1, 1] + R_rel[..., 2, 2]) - 1.0) / 2.0
    rot_err = torch.acos(cos.clamp(-1 + 1e-6, 1 - 1e-6))          # [B, n] radians
    trans_err = (P[..., :3, 3] - T[..., :3, 3]).abs().sum(-1)     # [B, n] L1

    cR, cT = C[..., 0].clamp(-5, 5), C[..., 1].clamp(-5, 5)
    per_pair = rot_err * torch.exp(-cR) + cR + trans_weight * (trans_err * torch.exp(-cT) + cT)
    per_pair = torch.where(finite, per_pair, torch.zeros_like(per_pair))
    denom = finite.float().sum().clamp(min=1.0)
    return {
        "pose_distill": per_pair.sum() / denom,
        "rot_err_rad": (rot_err * finite).sum().detach() / denom,
        "trans_err_l1": (trans_err * finite).sum().detach() / denom,
    }


def epipolar_sampson_loss(out, data, stride: int = 8, parallax_gate: bool = True,
                          trunc: float = 1e-6):
    """Scale-invariant epipolar consistency (Sampson distance) on cached flow.

    Why: the flow-reprojection loss recovers translation only through depth, and at
    small baselines with noisy monocular depth the translation is weakly observable —
    rotation explains almost all the flow, so the DIRECTION of translation is left
    nearly unconstrained (measured: ~90 deg error on Sintel/TUM, but fine on KITTI
    where the camera actually translates). The epipolar constraint

        x'^T E x = 0,   E = [t]_x R

    constrains rotation and translation direction with NO depth, and is invariant to
    the scale of t — exactly the missing signal. Validated on Sintel GT: recovers the
    true direction to ~2.9 deg, and still to ~4-5 deg under realistic degradation
    (dynamic objects + 20% focal error + noisy flow).

    Design notes, both learned the hard way:
      * REDESCENDING loss (truncate at `trunc`) rather than log1p. The ablation showed
        truncation is more accurate under dynamic content (2.85 vs 4.16 deg) because
        moving objects stop contributing entirely instead of being averaged in. It also
        BOUNDS the gradient, which log1p did not: an unbounded epipolar gradient
        crowded the flow term out of the clip_grad_norm budget and wrecked
        out-of-domain performance (KITTI tdir 7.8 -> 67 deg).
      * translation is normalized with a floored, detached norm. Truly normalizing a
        near-zero t makes the Jacobian blow up as 1/|t|, and near-zero t is exactly the
        degenerate solution this term is meant to cure.
      * pairs with no parallax carry no direction information and are down-weighted.
    """
    poses = out["poses"][:, :-1, 0]          # [B, P, 4, 4]  cam_i -> cam_{i+1}
    B, P = poses.shape[0], poses.shape[1]
    flows = data["flows_fwd"]                 # [B, P, 2, H, W] pixels
    occs = data["occs_fwd"]                   # [B, P, 1, H, W]
    calib = data["calibs"].mean(dim=1)        # [B, 4] fx fy cx cy
    H, W = flows.shape[-2:]
    dev = flows.device

    ys = torch.arange(0, H, stride, device=dev)
    xs = torch.arange(0, W, stride, device=dev)
    vv, uu = torch.meshgrid(ys, xs, indexing="ij")
    u = uu.reshape(1, 1, -1).float()
    v = vv.reshape(1, 1, -1).float()

    fl = flows[:, :, :, ::stride, ::stride].flatten(3)          # [B,P,2,S]
    oc = occs[:, :, :, ::stride, ::stride].flatten(3)[:, :, 0]  # [B,P,S]
    u2, v2 = u + fl[:, :, 0], v + fl[:, :, 1]

    fx = calib[:, 0].view(B, 1, 1); fy = calib[:, 1].view(B, 1, 1)
    cx = calib[:, 2].view(B, 1, 1); cy = calib[:, 3].view(B, 1, 1)
    ones = torch.ones_like(u2)
    x1 = torch.stack([(u - cx) / fx + 0 * ones, (v - cy) / fy + 0 * ones, ones], -1)
    x2 = torch.stack([(u2 - cx) / fx, (v2 - cy) / fy, ones], -1)   # [B,P,S,3]

    R = poses[..., :3, :3]                                   # [B,P,3,3]
    t = poses[..., :3, 3]
    # floored + detached norm: direction for meaningful motion, gracefully shrinking
    # (rather than exploding) as |t| -> 0
    tn = t / t.norm(dim=-1, keepdim=True).detach().clamp(min=2e-2)
    tx = torch.zeros(B, P, 3, 3, device=dev, dtype=R.dtype)
    tx[..., 0, 1] = -tn[..., 2]; tx[..., 0, 2] = tn[..., 1]
    tx[..., 1, 0] = tn[..., 2];  tx[..., 1, 2] = -tn[..., 0]
    tx[..., 2, 0] = -tn[..., 1]; tx[..., 2, 1] = tn[..., 0]
    E = tx @ R                                               # [B,P,3,3]

    Ex1 = torch.einsum("bpij,bpsj->bpsi", E, x1)
    Etx2 = torch.einsum("bpji,bpsj->bpsi", E, x2)
    num = (x2 * Ex1).sum(-1) ** 2
    den = Ex1[..., 0] ** 2 + Ex1[..., 1] ** 2 + Etx2[..., 0] ** 2 + Etx2[..., 1] ** 2
    sampson = num / (den + 1e-8)                             # [B,P,S]

    valid = (oc >= 0.5) & torch.isfinite(sampson)
    sampson = torch.nan_to_num(sampson, nan=0.0, posinf=0.0)

    # redescending: inliers contribute linearly, gross outliers (movers) saturate
    per_point = torch.clamp(sampson / trunc, max=1.0)

    if parallax_gate:
        with torch.no_grad():
            xr = torch.einsum("bpij,bpsj->bpsi", R, x1)
            xr = xr[..., :2] / xr[..., 2:].clamp(min=1e-6)
            ur = xr[..., 0] * fx + cx
            vr = xr[..., 1] * fy + cy
            parallax = ((u2 - ur) ** 2 + (v2 - vr) ** 2).sqrt()
            gate = (parallax.median(dim=-1).values / 0.5).clamp(0, 1)  # [B,P]
        per_point = per_point * gate.unsqueeze(-1)

    per_point = torch.where(valid, per_point, torch.zeros_like(per_point))
    denom = valid.float().sum().clamp(min=1.0)
    return {
        "epipolar": per_point.sum() / denom,
        "sampson_raw": (sampson * valid).sum().detach() / denom,
    }


def calib_distill_loss(out: Dict, data: Dict) -> Dict:
    """Distil the calibration head from the cached AnyCalib per-frame intrinsics (px at the
    training resolution). Regress log(f/W) (scale-free) and the principal-point offsets.
    Returns the loss and the per-frame relative focal error for monitoring."""
    calibs = data["calibs"]                     # [B, N, 4] fx fy cx cy (px)
    H, W = data["images"].shape[-2:]
    c = out["calib_raw"]                        # [B, N, 3]
    tgt_logf = torch.log(0.5 * (calibs[..., 0] + calibs[..., 1]) / W)
    pred_logf = c[..., 0] + out.get("logf_prior", 0.30)
    l_f = (pred_logf - tgt_logf).abs().mean()
    tgt_cx = calibs[..., 2] / W - 0.5
    tgt_cy = calibs[..., 3] / H - 0.5
    l_c = ((c[..., 1] - tgt_cx).abs() + (c[..., 2] - tgt_cy).abs()).mean()
    with torch.no_grad():
        f_pred = out["calib"][..., 0]
        f_tgt = 0.5 * (calibs[..., 0] + calibs[..., 1])
        rel = ((f_pred - f_tgt).abs() / f_tgt).mean()
    return {"calib_distill": l_f + 0.5 * l_c, "calib_rel_f_err": rel}
