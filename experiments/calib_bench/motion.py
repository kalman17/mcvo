"""Motion-observability labelling of an evaluation window from GT poses.

Grounded in the critical-motion results for unknown focal length:
  * Kahl, Triggs, Astrom (JMIV 2000): critical motions are (i) translations with
    rotation only about the optical axis, (ii) rotations about at most two centres,
    (iii) forward-looking motions along an ellipse/hyperbola (orbit-like).
  * Sturm (IVC 2002 / CVIU 2005), two views, equal unknown f: unsolvable when the
    optical axes are parallel, or intersect at a point equidistant from both centres.

We reduce a window to per-pair quantities and then to one label:
  theta      rotation angle between consecutive frames (deg)
  rho        parallax ratio |t| / Z_med  (dimensionless; needs a scene-depth scale)
  phi        angle between translation and the optical axis of the first frame (deg)
  axis_gap   closest-approach distance of the two optical axes, / Z_med
  axis_ratio min(d1,d2)/max(d1,d2) where d1,d2 are the depths of the closest-approach
             points along each axis (1 = equidistant), only if both in front

Labels (thresholds are stated in the paper; robustness sweep in the supplement):
  static        rho < 0.005 and theta < 0.3
  rotation      rotational flow dominates: theta_rad > R_DOM * rho and theta >= 0.3
  forward       translation dominates, phi < 20  (parallel axes, forward-looking)
  lateral       translation dominates, phi > 70  (parallel axes, sideways)
  orbit         axes intersect in front (axis_gap < 0.15) with axis_ratio > 0.6 and
                both translation and rotation non-trivial
  general       everything else
Z_med: GT depth median if the loader provides it, else a supplied estimate (e.g. from
UniDepth on the first frame), else the axis-intersection depth when the axes do
intersect in front, else None -> label 'unknown_scale' (rotation/translation split
impossible without scale, though phi and axis geometry are still reported).
"""

import numpy as np

R_DOM = 3.0        # rotational-flow / translational-flow ratio for 'rotation'
T_DOM = 1.0 / 3.0  # ... for translation-dominated classes


def _rot_angle_deg(R):
    return float(np.rad2deg(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def _axis_geometry(P0, P1):
    """Closest approach of the two optical axes (world frame). Returns (gap, d0, d1)."""
    c0, z0 = P0[:3, 3], P0[:3, 2] / np.linalg.norm(P0[:3, 2])
    c1, z1 = P1[:3, 3], P1[:3, 2] / np.linalg.norm(P1[:3, 2])
    A = np.array([[z0 @ z0, -z0 @ z1], [z0 @ z1, -z1 @ z1]])
    b = np.array([(c1 - c0) @ z0, (c1 - c0) @ z1])
    det = np.linalg.det(A)
    if abs(det) < 1e-9:  # parallel axes
        return float("inf"), None, None
    d0, d1 = np.linalg.solve(A, b)
    gap = float(np.linalg.norm((c0 + d0 * z0) - (c1 + d1 * z1)))
    return gap, float(d0), float(d1)


def window_motion(poses, depth_med=None):
    """poses: [N,4,4] camera-to-world. Returns dict of per-window statistics + label."""
    P = np.asarray(poses, dtype=np.float64)
    n = P.shape[0]
    thetas, tnorms, phis = [], [], []
    for i in range(n - 1):
        rel = np.linalg.inv(P[i]) @ P[i + 1]        # cam_{i+1} expressed in cam_i
        thetas.append(_rot_angle_deg(rel[:3, :3]))
        t = rel[:3, 3]
        tn = float(np.linalg.norm(t))
        tnorms.append(tn)
        # angle between translation direction and the optical axis of cam_i (z)
        phis.append(float(np.rad2deg(np.arccos(abs(t[2]) / tn))) if tn > 1e-9 else float("nan"))
    theta = float(np.median(thetas))
    tnorm = float(np.median(tnorms))
    phi = float(np.nanmedian(phis)) if np.isfinite(phis).any() else float("nan")

    # axis geometry between the first and last frame of the window
    gap, d0, d1 = _axis_geometry(P[0], P[-1])
    in_front = d0 is not None and d0 > 0 and d1 > 0
    axis_ratio = float(min(d0, d1) / max(d0, d1)) if in_front else 0.0

    # scale
    scale_src = None
    Z = None
    if depth_med is not None and np.isfinite(depth_med) and depth_med > 0:
        Z, scale_src = float(depth_med), "gt_depth"
    elif in_front and gap < 0.25 * max(d0, d1):
        Z, scale_src = float(0.5 * (d0 + d1)), "axis_intersection"

    out = {
        "theta_deg": theta, "t_norm": tnorm, "phi_deg": phi,
        "axis_gap": gap if np.isfinite(gap) else None,
        "axis_d0": d0, "axis_d1": d1, "axis_ratio": axis_ratio,
        "Z_med": Z, "scale_src": scale_src, "n_pairs": n - 1,
    }
    if Z is None:
        # no scale available. Pure rotation is still identifiable without one:
        # translation exactly zero (e.g. panorama-synthesised windows) and rotation present.
        if tnorm < 1e-9 and theta >= 0.3:
            out.update({"rho": 0.0, "label": "rotation"})
        elif tnorm < 1e-9 and theta < 0.3:
            out.update({"rho": 0.0, "label": "static"})
        else:
            out.update({"rho": None, "label": "unknown_scale"})
        return out

    rho = tnorm / Z
    theta_rad = np.deg2rad(theta)
    out["rho"] = float(rho)
    out["axis_gap_rel"] = float(gap / Z) if np.isfinite(gap) else None
    total_theta = _rot_angle_deg((np.linalg.inv(P[0]) @ P[-1])[:3, :3])
    total_rho = float(np.linalg.norm((np.linalg.inv(P[0]) @ P[-1])[:3, 3]) / Z)

    if rho < 0.005 and theta < 0.3:
        label = "static"
    elif theta >= 0.3 and theta_rad > R_DOM * rho:
        label = "rotation"
    elif (in_front and np.isfinite(gap) and gap / Z < 0.15 and axis_ratio > 0.6
          and total_rho > 0.02 and total_theta > 1.0):
        label = "orbit"
    elif theta_rad < T_DOM * rho and np.isfinite(phi) and phi < 20:
        label = "forward"
    elif theta_rad < T_DOM * rho and np.isfinite(phi) and phi > 70:
        label = "lateral"
    else:
        label = "general"
    out["label"] = label
    return out


def anycam_candidate_stats(probs, dist_landscape=None, candidates=None):
    """Summaries of AnyCam's 32-way focal evidence.
    probs: [nc] learned distribution (softmax head). dist_landscape: [nc] mean flow
    distance per candidate (lower = better fit). Returns entropies (normalised to [0,1]),
    peakedness, and the argmin/argmax candidates."""
    out = {}
    if probs is not None:
        p = np.clip(np.asarray(probs, dtype=np.float64).ravel(), 1e-12, 1)
        p = p / p.sum()
        H = float(-(p * np.log(p)).sum() / np.log(len(p)))
        out.update({"probs_entropy": H, "probs_max": float(p.max()),
                    "probs_argmax": int(p.argmax())})
    if dist_landscape is not None:
        d = np.asarray(dist_landscape, dtype=np.float64).ravel()
        if np.isfinite(d).all() and d.size > 1:
            # softmin landscape sharpness: relative range and normalised entropy of exp(-d/T)
            rng = float((d.max() - d.min()) / (abs(d.min()) + 1e-9))
            w = np.exp(-(d - d.min()) / (np.std(d) + 1e-9))
            w = w / w.sum()
            Hd = float(-(w * np.log(w + 1e-12)).sum() / np.log(len(w)))
            out.update({"dist_rel_range": rng, "dist_entropy": Hd,
                        "dist_argmin": int(d.argmin())})
    if candidates is not None:
        c = np.asarray(candidates, dtype=np.float64).ravel()
        if "probs_argmax" in out:
            out["probs_argmax_focal"] = float(c[out["probs_argmax"]])
        if "dist_argmin" in out:
            out["dist_argmin_focal"] = float(c[out["dist_argmin"]])
    return out
