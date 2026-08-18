"""Calibration metrics beyond focal APE: field-of-view error in degrees (the metric
GeoCalib / AnyCalib report), aggregate focal APE, and principal-point error.

All inputs are [fx, fy, cx, cy] at the SAME resolution (h, w) as the GT.
"""

import numpy as np


def hfov_deg(fx, w):
    return float(np.rad2deg(2 * np.arctan(w / (2.0 * fx))))


def vfov_deg(fy, h):
    return float(np.rad2deg(2 * np.arctan(h / (2.0 * fy))))


def calib_metrics_full(intr, gt_intr, h, w):
    fx, fy, cx, cy = [float(v) for v in intr[:4]]
    gfx, gfy, gcx, gcy = [float(v) for v in gt_intr[:4]]
    fx_ape = abs(fx - gfx) / abs(gfx) * 100.0
    fy_ape = abs(fy - gfy) / abs(gfy) * 100.0
    return {
        "pred_fx": fx, "pred_fy": fy, "pred_cx": cx, "pred_cy": cy,
        "gt_fx": gfx, "gt_fy": gfy, "gt_cx": gcx, "gt_cy": gcy,
        "fx_ape_pct": fx_ape, "fy_ape_pct": fy_ape, "f_ape_pct": 0.5 * (fx_ape + fy_ape),
        "gt_hfov_deg": hfov_deg(gfx, w), "gt_vfov_deg": vfov_deg(gfy, h),
        "hfov_err_deg": abs(hfov_deg(fx, w) - hfov_deg(gfx, w)),
        "vfov_err_deg": abs(vfov_deg(fy, h) - vfov_deg(gfy, h)),
        "pp_err_px": float(np.hypot(cx - gcx, cy - gcy)),
        "pp_err_rel": float(np.hypot((cx - gcx) / w, (cy - gcy) / h)),
        "input_h": int(h), "input_w": int(w), "aspect": float(w / h),
    }
