#!/usr/bin/env python3
import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import open3d as o3d
import torch
import cv2  # For triangulation
import os
import requests
import math
import torch.nn.functional as F

# Add debug visualization imports
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Add project root and anycalib to Python path
import sys
script_dir = Path(__file__).resolve().parent  # experiments/
project_root = script_dir.parent  # anycam/
sys.path.insert(0, str(project_root))  # Add project root to Python path
sys.path.append('/home/kalman/TUM/thesis/anycam/anycalib')

try:
    from anycalib.model.anycalib_pretrained import AnyCalib
except ImportError:
    # Fallback for different path structure
    from anycalib.anycalib.model.anycalib_pretrained import AnyCalib

# Improve CUDA memory handling to reduce fragmentation
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Local experiment helpers
from experiments.common.data_loader import ExperimentDataManager
from experiments.common.anycam_inference import create_inference_engine

# Reuse UniDepth wrapper via factory and image loader
from anycam.models import make_depth_predictor


def rgb_to_tensor(rgb: np.ndarray) -> torch.Tensor:
    if rgb.dtype != np.float32:
        rgb = rgb.astype(np.float32) / 255.0
    return torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)


def depth_to_point_cloud(rgb_image: np.ndarray, depth_map: np.ndarray, focal_length: float, cx: float, cy: float) -> o3d.geometry.PointCloud:
    h, w = depth_map.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    valid = np.isfinite(depth_map) & (depth_map > 0)

    z = depth_map[valid]
    x = (u[valid] - cx) * z / focal_length
    y = (v[valid] - cy) * z / focal_length
    points = np.stack([x, y, z], axis=-1)

    colors = (rgb_image.astype(np.float64) / 255.0)
    colors = colors.reshape(-1, 3)[valid.reshape(-1)]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def align_and_merge_pcds(pcd1: o3d.geometry.PointCloud, pcd2: o3d.geometry.PointCloud, T_0_to_1: np.ndarray) -> o3d.geometry.PointCloud:
    # We want both clouds in frame 0. If T_0_to_1 maps points from frame 0 to frame 1,
    # then points in frame 1 should be transformed by inv(T_0_to_1) to frame 0.
    T_1_to_0 = np.linalg.inv(T_0_to_1)
    pcd2_aligned = o3d.geometry.PointCloud(pcd2)  # copy
    pcd2_aligned.transform(T_1_to_0)
    return pcd1 + pcd2_aligned


def compute_chamfer_distance(pcd_ref: o3d.geometry.PointCloud, pcd_target: o3d.geometry.PointCloud) -> float:
    """Compute average Chamfer Distance (symmetric)."""
    dist1 = pcd_ref.compute_point_cloud_distance(pcd_target)
    dist2 = pcd_target.compute_point_cloud_distance(pcd_ref)
    chamfer = (np.mean(dist1) + np.mean(dist2)) / 2.0
    return float(chamfer)


def compute_hausdorff_distance(pcd_ref: o3d.geometry.PointCloud, pcd_target: o3d.geometry.PointCloud) -> float:
    """Compute Hausdorff Distance (max outlier)."""
    hausdorff = max(np.max(pcd_ref.compute_point_cloud_distance(pcd_target)),
                    np.max(pcd_target.compute_point_cloud_distance(pcd_ref)))
    return float(hausdorff)


# --- New: Optical flow with UniMatch and triangulation helpers ---

def run_optical_flow_unimatch(frames: List[np.ndarray], device: torch.device, ckpt_path: Path | None) -> Tuple[np.ndarray, np.ndarray]:
    """Run UniMatch to get forward and backward flow as (H, W, 2).
    EXACTLY matches AnyCam's FlowOcclusionProcessor.flow_unimatch() method.
    Returns: (flow_fwd, flow_bwd) where flow_fwd is frame0->frame1, flow_bwd is frame1->frame0
    """
    print(f"[FLOW DEBUG] Processing {len(frames)} frames in run_optical_flow_unimatch")
    print(f"[FLOW DEBUG] Frame 0 shape: {frames[0].shape}, range: [{frames[0].min():.3f}, {frames[0].max():.3f}]")
    print(f"[FLOW DEBUG] Frame 1 shape: {frames[1].shape}, range: [{frames[1].min():.3f}, {frames[1].max():.3f}]")
    try:
        from unimatch.unimatch import UniMatch  # type: ignore
    except ModuleNotFoundError:
        import sys
        repo_root = Path(__file__).resolve().parents[1]
        unimatch_root = repo_root / "unimatch"
        if str(unimatch_root) not in sys.path:
            sys.path.insert(0, str(unimatch_root))
        from unimatch.unimatch import UniMatch  # type: ignore

    if ckpt_path is None:
        ckpt_path = Path(os.environ.get("HOME", str(Path.home()))) / \
            ".cache/torch/checkpoints/gmflow-scale2-regrefine6-mixdata-train320x576-4e7b215d.pth"
        if not ckpt_path.exists():
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            url = "https://s3.eu-central-1.amazonaws.com/avg-projects/unimatch/pretrained/gmflow-scale2-regrefine6-mixdata-train320x576-4e7b215d.pth"
            print(f"[UniMatch] Downloading pretrained weights to {ckpt_path} ...")
            r = requests.get(url)
            r.raise_for_status()
            with open(ckpt_path, 'wb') as f:
                f.write(r.content)
            print(f"[UniMatch] Downloaded.")

    model = UniMatch(
        feature_channels=128,
        num_scales=2,
        upsample_factor=4,
        ffn_dim_expansion=4,
        num_transformer_layers=6,
        reg_refine=True,
        task='flow',
    ).to(device).eval()

    state = torch.load(str(ckpt_path), map_location=device)
    model.load_state_dict(state['model'], strict=False)

    # Convert to proper tensor format: AnyCam expects [0, 1] range
    img0 = frames[0].astype(np.float32)
    img1 = frames[1].astype(np.float32)
    if img0.max() > 1.0:
        img0 = img0 / 255.0
        img1 = img1 / 255.0

    # Convert to tensors with batch dimension
    img0 = torch.from_numpy(img0).permute(2, 0, 1).unsqueeze(0).to(device)  # [1, 3, H, W]
    img1 = torch.from_numpy(img1).permute(2, 0, 1).unsqueeze(0).to(device)  # [1, 3, H, W]

    # EXACTLY match AnyCam's flow_unimatch preprocessing
    n, c, h, w = img0.shape
    max_size = 320
    smaller = min(h, w)

    if smaller > max_size:
        scale_factor = max_size / smaller
        target_h = h * scale_factor
        target_w = w * scale_factor
    else:
        target_h = h
        target_w = w
    
    target_h = math.ceil(target_h / 32) * 32
    target_w = math.ceil(target_w / 32) * 32

    if target_h != h or target_w != w:
        img0 = F.interpolate(img0, (target_h, target_w), mode='bilinear', align_corners=True)
        img1 = F.interpolate(img1, (target_h, target_w), mode='bilinear', align_corners=True)

    # CRITICAL: AnyCam transforms [0,1] to [-0.5, 0.5] then to [0, 255] for UniMatch
    img0 = (img0 * 0.5 + 0.5) * 255
    img1 = (img1 * 0.5 + 0.5) * 255

    attn_type = 'swin'
    attn_splits_list = [2, 8]
    corr_radius_list = [-1, 4]
    prop_radius_list = [-1, 1]
    num_reg_refine = 6

    # Handle portrait orientation (H > W) like AnyCam
    if target_h > target_w:
        img0 = img0.permute(0, 1, 3, 2)
        img1 = img1.permute(0, 1, 3, 2)

    with torch.no_grad():
        results_dict = model(img0, img1,
                            attn_type=attn_type,
                            attn_splits_list=attn_splits_list,
                            corr_radius_list=corr_radius_list,
                            prop_radius_list=prop_radius_list,
                            num_reg_refine=num_reg_refine,
                            task="flow",
                            pred_bidir_flow=True,
                            )
    
    flows = results_dict['flow_preds'][-1]

    # Undo portrait orientation transform
    if target_h > target_w:
        flows = flows.permute(0, 1, 3, 2)
        flows = flows[:, [1, 0], :, :]  # Swap flow components

    # Resize back to original resolution
    if target_h != h or target_w != w:
        flows = F.interpolate(flows, (h, w), mode='bilinear', align_corners=True)

    flow_fwd = flows[:n]  # first n are forward
    flow_bwd = flows[n:]   # next n are backward
    
    flow_fwd_np = flow_fwd[0].permute(1, 2, 0).detach().cpu().numpy()  # (H, W, 2)
    flow_bwd_np = flow_bwd[0].permute(1, 2, 0).detach().cpu().numpy()  # (H, W, 2)
    
    return flow_fwd_np, flow_bwd_np


def triangulate_point_cloud(rgb_image: np.ndarray, flow_fwd: np.ndarray, flow_bwd: np.ndarray, focal_length: float, cx: float, cy: float, T_0_to_1: np.ndarray, step: int = 2, cycle_thresh: float = 1.0, debug_dir: Path = None, visualize_correspondences: bool = False, frame2: np.ndarray = None) -> o3d.geometry.PointCloud:
    """Triangulate 3D points from optical flow correspondences.
    
    Fixed major issues:
    1. cv2.triangulatePoints expects projection matrices WITH intrinsics, not normalized coordinates
    2. T_0_to_1 should transform from cam1 to cam0 for OpenCV convention
    3. Proper handling of homogeneous coordinates and depth filtering
    """
    h, w = flow_fwd.shape[:2]
    u, v = np.meshgrid(np.arange(0, w, step), np.arange(0, h, step))
    u_flat, v_flat = u.flatten(), v.flatten()
    
    # line 223, 227, visualize the point correspondences and visually check if they make sense, one after the other, 20 or 30 of them.
    #also it;s flipped to behind the camerra
    flow_u = flow_fwd[v, u, 0].flatten()
    flow_v = flow_fwd[v, u, 1].flatten()
    u2_flat = u_flat + flow_u
    v2_flat = v_flat + flow_v

    print(f"[TRIANG] Initial correspondences: {len(u_flat)}")
    print(f"[TRIANG] Flow range: u=({flow_u.min():.2f}, {flow_u.max():.2f}), v=({flow_v.min():.2f}, {flow_v.max():.2f})")

    # Filter in-bounds correspondences
    valid_bound = (u2_flat >= 0) & (u2_flat < w) & (v2_flat >= 0) & (v2_flat < h)
    print(f"[TRIANG] Valid bounds: {valid_bound.sum()}/{len(u_flat)}")
    if not np.any(valid_bound):
        raise ValueError("No valid correspondences after bounds filtering")

    # Cycle consistency filtering
    u2_int = np.round(u2_flat[valid_bound]).astype(int)
    v2_int = np.round(v2_flat[valid_bound]).astype(int)
    u2_int = np.clip(u2_int, 0, w - 1)
    v2_int = np.clip(v2_int, 0, h - 1)
    
    back_u = flow_bwd[v2_int, u2_int, 0]
    back_v = flow_bwd[v2_int, u2_int, 1]
    u_back = u2_flat[valid_bound] + back_u
    v_back = v2_flat[valid_bound] + back_v
    cycle_error = np.sqrt((u_back - u_flat[valid_bound])**2 + (v_back - v_flat[valid_bound])**2)
    valid_cycle = cycle_error < cycle_thresh
    
    print(f"[TRIANG] Cycle consistency (thresh={cycle_thresh}): {valid_cycle.sum()}/{len(valid_cycle)}")
    print(f"[TRIANG] Cycle error range: {cycle_error.min():.2f} - {cycle_error.max():.2f}")
    
    final_valid_indices = np.where(valid_bound)[0][valid_cycle]
    
    if len(final_valid_indices) == 0:
        print(f"[TRIANG] ERROR: No correspondences survived cycle filtering. Try increasing --cycle_thresh (current: {cycle_thresh})")
        # Fallback: try with relaxed cycle consistency
        fallback_thresh = min(cycle_thresh * 3, 5.0)  # Try 3x threshold, max 5 pixels
        print(f"[TRIANG] Trying fallback cycle threshold: {fallback_thresh}")
        valid_cycle_fallback = cycle_error < fallback_thresh
        final_valid_indices = np.where(valid_bound)[0][valid_cycle_fallback]
        
        if len(final_valid_indices) == 0:
            print(f"[TRIANG] Still no correspondences with relaxed threshold. Flow might be poor quality.")
            raise ValueError("No valid correspondences after cycle filtering")
        else:
            print(f"[TRIANG] Fallback successful: {len(final_valid_indices)} correspondences")

    # Get final pixel correspondences
    pts1_px = np.stack([u_flat[final_valid_indices], v_flat[final_valid_indices]], axis=1).astype(np.float32)
    pts2_px = np.stack([u2_flat[final_valid_indices], v2_flat[final_valid_indices]], axis=1).astype(np.float32)

    print(f"[TRIANG] Final correspondences for triangulation: {len(pts1_px)}")

    # DEBUG: Save correspondences visualization
    if debug_dir:
        save_correspondences_debug(pts1_px, pts2_px, rgb_image, "triangulation", debug_dir, max_points=500)
    
    # Interactive correspondence visualization
    if visualize_correspondences and frame2 is not None:
        print(f"[VIZ] Showing correspondences for focal length {focal_length:.1f}")
        should_continue = visualize_correspondences_interactive(rgb_image, frame2, pts1_px, pts2_px, max_points=50)
        if not should_continue:
            # User requested quit, return empty point cloud
            return o3d.geometry.PointCloud()

    # Build intrinsic matrix and projection matrices
    K = np.array([[focal_length, 0, cx], 
                  [0, focal_length, cy], 
                  [0, 0, 1]], dtype=np.float32)
    
    # OpenCV triangulation expects:
    # P1 = K [I | 0] for first camera
    # P2 = K [R | t] for second camera
    # where [R|t] transforms points from world to camera2
    R = T_0_to_1[:3, :3].astype(np.float32)
    t = T_0_to_1[:3, 3].astype(np.float32)
    # maybe using wrong poses, perhaps swap P1 and P2? could be confusing the direction of the poses.
    P1 = K @ np.hstack((np.eye(3, dtype=np.float32), np.zeros((3, 1), dtype=np.float32)))
    P2 = K @ np.hstack((R, t.reshape(3, 1)))

    print(f"[TRIANG] Focal length: {focal_length:.1f}, Principal point: ({cx:.1f}, {cy:.1f})")
    print(f"[TRIANG] Relative pose T_0_to_1 translation: {t}")

    # Triangulate using pixel coordinates directly (OpenCV handles the rest)
    points_hom = cv2.triangulatePoints(P1, P2, pts1_px.T, pts2_px.T)  # (4, N)
    
    # Convert from homogeneous coordinates
    points_3d = points_hom[:3] / np.clip(points_hom[3:4], 1e-9, None)  # (3, N)
    points_3d = points_3d.T  # (N, 3)

    print(f"[TRIANG] 3D points before filtering: {len(points_3d)}")
    print(f"[TRIANG] Z depth range: {points_3d[:, 2].min():.3f} - {points_3d[:, 2].max():.3f}")

    # COORDINATE SYSTEM FIX: Negate Z to fix coordinate convention
    points_3d[:, 2] = -points_3d[:, 2]
    
    # Filter points with positive depth in camera 0 (less aggressive)
    valid_depth = points_3d[:, 2] > 0.01  # Very permissive: 1cm away
    points_3d = points_3d[valid_depth]
    final_color_indices = final_valid_indices[valid_depth]

    print(f"[TRIANG] 3D points after depth filtering: {len(points_3d)}")

    if len(points_3d) == 0:
        print(f"[TRIANG] ERROR: All points have negative/zero depth. Check relative pose or correspondences.")
        print(f"[TRIANG] Pose determinant: {np.linalg.det(T_0_to_1[:3, :3]):.3f} (should be ~1)")
        raise ValueError("No valid 3D points after depth filtering")

    # Filter outliers based on reasonable depth range
    median_depth = np.median(points_3d[:, 2])
    depth_std = np.std(points_3d[:, 2])
    depth_thresh = median_depth + 3 * depth_std  # Remove extreme outliers
    reasonable_depth = (points_3d[:, 2] > 0.1) & (points_3d[:, 2] < depth_thresh)
    
    print(f"[TRIANG] Median depth: {median_depth:.2f}m, std: {depth_std:.2f}m")
    print(f"[TRIANG] After outlier removal: {reasonable_depth.sum()}/{len(points_3d)}")
    
    if reasonable_depth.sum() == 0:
        print(f"[TRIANG] WARNING: All points are outliers, keeping original points")
        reasonable_depth = np.ones(len(points_3d), dtype=bool)
    
    points_3d = points_3d[reasonable_depth]
    final_color_indices = final_color_indices[reasonable_depth]

    # Extract colors from original pixel locations
    colors = (rgb_image.astype(np.float64) / 255.0)
    colors = colors[v_flat[final_color_indices], u_flat[final_color_indices]]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_3d)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    print(f"[TRIANG] Final point cloud: {len(points_3d)} points")
    
    # DEBUG: Save point cloud stats
    if debug_dir:
        save_point_cloud_debug(pcd, f"triangulated_f{focal_length:.0f}", debug_dir)
    
    return pcd


def load_gt_focal_from_intrinsics(gt_dir: Path, video_path: Path, frame_idx: int = 0) -> float:
    # Read per-sequence JSON and average fx/fy from first two intrinsics
    gt_file = resolve_gt_json_path(gt_dir, video_path)
    with open(gt_file, "r") as f:
        data = json.load(f)
    # Prefer 'intrinsics_per_frame', fallback to 'intrinsics'
    if "intrinsics_per_frame" in data:
        intr = data["intrinsics_per_frame"]
    elif "intrinsics" in data:
        intr = data["intrinsics"]
    else:
        raise KeyError(f"GT JSON missing 'intrinsics_per_frame'/'intrinsics' key: {gt_file}")
    if len(intr) <= frame_idx:
        raise ValueError(f"GT intrinsics has fewer than {frame_idx + 1} entries: {gt_file}")
    K = np.array(intr[frame_idx], dtype=np.float64).reshape(3, 3)
    f = (float(K[0, 0]) + float(K[1, 1])) / 2.0
    return float(f)


def extract_anycam_focal(projection: np.ndarray) -> float:
    # projection is 3x3; average fx, fy
    fx = float(projection[0, 0])
    fy = float(projection[1, 1])
    return (fx + fy) / 2.0


def run_anycalib_per_frame_average_focal(frames: List[np.ndarray], device: torch.device) -> float:
    # AnyCalib is vendored under repo_root/anycalib/anycalib, but the importable package is 'anycalib'.
    # Ensure the parent folder (repo_root/anycalib) is on sys.path, then import 'anycalib.model'.
    try:
        from anycalib.model.anycalib_pretrained import AnyCalib  # type: ignore
    except ModuleNotFoundError:
        import sys
        repo_root = Path(__file__).resolve().parents[1]
        anycalib_parent = repo_root / "anycalib"
        if str(anycalib_parent) not in sys.path:
            sys.path.insert(0, str(anycalib_parent))
        from anycalib.model.anycalib_pretrained import AnyCalib  # type: ignore
    # Use pinhole by default; model_id can be parameterized later
    model = AnyCalib(model_id="anycalib_pinhole").to(device).eval()

    # Batch the two frames for a single predict call
    tensors: List[torch.Tensor] = []
    for frame in frames:
        img = frame.astype(np.float32) / 255.0
        tensors.append(torch.from_numpy(img).permute(2, 0, 1))
    batch = torch.stack(tensors, dim=0).to(device)

    focals: List[float] = []
    with torch.no_grad():
        pred = model.predict(batch, cam_id="pinhole")
        intrinsics_list = pred["intrinsics"]
        # Normalize to iterable of numpy arrays
        if isinstance(intrinsics_list, torch.Tensor):
            intrinsics_list = [intrinsics_list]
        for intr in intrinsics_list:
            if torch.is_tensor(intr):
                arr = intr.detach().cpu().numpy()
            else:
                arr = np.array(intr)
            fx = float(arr[0])
            fy = float(arr[1])
            focals.append((fx + fy) / 2.0)

    return float(np.mean(focals))


def run_anycalib_single_frame(frame: np.ndarray, device: torch.device = None) -> float:
    """Run AnyCalib on a single frame to get its focal length."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    try:
        from anycalib.model.anycalib_pretrained import AnyCalib  # type: ignore
    except ModuleNotFoundError:
        import sys
        repo_root = Path(__file__).resolve().parents[1]
        anycalib_parent = repo_root / "anycalib"
        if str(anycalib_parent) not in sys.path:
            sys.path.insert(0, str(anycalib_parent))
        from anycalib.model.anycalib_pretrained import AnyCalib  # type: ignore

    model = AnyCalib(model_id="anycalib_pinhole").to(device).eval()

    img = frame.astype(np.float32) / 255.0
    tensor = torch.from_numpy(img).permute(2, 0, 1).to(device)

    with torch.no_grad():
        pred = model.predict(tensor, cam_id="pinhole")
        intrinsics_list = pred["intrinsics"]
        if isinstance(intrinsics_list, torch.Tensor):
            intrinsics_list = [intrinsics_list]
        for intr in intrinsics_list:
            if torch.is_tensor(intr):
                arr = intr.detach().cpu().numpy()
            else:
                arr = np.array(intr)
            fx = float(arr[0])
            fy = float(arr[1])
            return (fx + fy) / 2.0


def load_gt_relative_pose_from_dir(gt_dir: Path, video_path: Path, frame_indices: Tuple[int, int] = (0, 1)) -> np.ndarray:
    # Load matching JSON named like the video stem
    try:
        gt_file = resolve_gt_json_path(gt_dir, video_path)
    except FileNotFoundError as e:
        print(f"[WARN] {e}; using identity for alignment")
        return np.eye(4, dtype=np.float64)
    with open(gt_file, "r") as f:
        data = json.load(f)
    if "poses" not in data:
        print(f"[WARN] GT JSON missing 'poses' key: {gt_file}; using identity for alignment")
        return np.eye(4, dtype=np.float64)
    poses_flat = data["poses"]
    frame0_idx, frame1_idx = frame_indices
    if len(poses_flat) <= max(frame0_idx, frame1_idx):
        print(f"[WARN] GT JSON has insufficient poses for frames {frame0_idx},{frame1_idx}: {gt_file}; using identity for alignment")
        return np.eye(4, dtype=np.float64)
    P0 = np.array(poses_flat[frame0_idx], dtype=np.float64).reshape(4, 4)
    P1 = np.array(poses_flat[frame1_idx], dtype=np.float64).reshape(4, 4)
    # c2w poses; transform from cam0 to cam1 is T_0_to_1 = inv(c2w0) @ c2w1
    T_0_to_1 = np.linalg.inv(P0) @ P1
    return T_0_to_1


def pick_random_unprocessed_video(default_videos_dir: Path, default_output_base: Path) -> Path:
    # Gather videos
    candidates = []
    for ext in ("*.mp4", "*.avi", "*.mov", "*.mkv"):
        candidates.extend(default_videos_dir.glob(ext))
        candidates.extend(default_videos_dir.glob(ext.upper()))
    candidates = sorted(candidates)
    if not candidates:
        raise SystemExit(f"No videos found in {default_videos_dir}")

    # Filter those not yet processed (no output folder with expected name)
    unprocessed: List[Path] = []
    for vid in candidates:
        exp_name = f"{vid.stem}_frames-1-2"
        out_dir = default_output_base / exp_name
        # Consider processed if directory exists and contains at least one merged PLY
        if not out_dir.exists() or not any(out_dir.glob(f"{vid.stem}_*_merged.ply")):
            unprocessed.append(vid)

    if not unprocessed:
        raise SystemExit("All videos in default directory appear processed.")

    return random.choice(unprocessed)


def resolve_gt_json_path(gt_dir: Path, video_path: Path) -> Path:
    """Resolve GT JSON path from a video by accommodating '*_video' suffix in stems.

    Tries '<stem_without__video>.json' first, then '<stem>.json'.
    """
    stem = video_path.stem
    candidates: List[str] = []
    if stem.endswith("_video"):
        candidates.append(stem[:-6])
    candidates.append(stem)
    tried = []
    for s in candidates:
        p = gt_dir / f"{s}.json"
        tried.append(str(p))
        if p.exists():
            return p
    raise FileNotFoundError(f"GT JSON not found; tried: {', '.join(tried)}")


def save_debug_images(frames, flow_fwd, flow_bwd, prefix, out_dir):
    """Save debug images and flow visualizations."""
    debug_dir = out_dir / "debug"
    debug_dir.mkdir(exist_ok=True)
    
    # Save input frames
    for i, frame in enumerate(frames):
        frame_path = debug_dir / f"{prefix}_frame{i}.png"
        # Convert from float [0,1] to uint8 [0,255]
        frame_uint8 = (frame * 255).astype(np.uint8)
        cv2.imwrite(str(frame_path), cv2.cvtColor(frame_uint8, cv2.COLOR_RGB2BGR))
        print(f"[DEBUG] Saved frame {i}: {frame_path}")
    
    # Visualize flows
    def flow_to_color(flow):
        """Convert optical flow to color image for visualization."""
        h, w = flow.shape[:2]
        fx, fy = flow[:,:,0], flow[:,:,1]
        
        ang = np.arctan2(fy, fx) + np.pi
        v = np.sqrt(fx*fx + fy*fy)
        
        hsv = np.zeros((h, w, 3), dtype=np.uint8)
        hsv[..., 0] = ang * (180 / np.pi / 2)
        hsv[..., 1] = 255
        hsv[..., 2] = np.minimum(v * 4, 255)
        
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        return rgb
    
    # Save flow visualizations
    flow_fwd_vis = flow_to_color(flow_fwd)
    flow_bwd_vis = flow_to_color(flow_bwd)
    
    cv2.imwrite(str(debug_dir / f"{prefix}_flow_fwd.png"), cv2.cvtColor(flow_fwd_vis, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(debug_dir / f"{prefix}_flow_bwd.png"), cv2.cvtColor(flow_bwd_vis, cv2.COLOR_RGB2BGR))
    
    # Save flow magnitude plots
    flow_mag_fwd = np.sqrt(flow_fwd[:,:,0]**2 + flow_fwd[:,:,1]**2)
    flow_mag_bwd = np.sqrt(flow_bwd[:,:,0]**2 + flow_bwd[:,:,1]**2)
    
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(flow_mag_fwd, cmap='hot')
    plt.title(f'Forward Flow Magnitude\nRange: {flow_mag_fwd.min():.2f} - {flow_mag_fwd.max():.2f}')
    plt.colorbar()
    
    plt.subplot(1, 3, 2)
    plt.imshow(flow_mag_bwd, cmap='hot')
    plt.title(f'Backward Flow Magnitude\nRange: {flow_mag_bwd.min():.2f} - {flow_mag_bwd.max():.2f}')
    plt.colorbar()
    
    plt.subplot(1, 3, 3)
    plt.hist(flow_mag_fwd.flatten(), bins=50, alpha=0.7, label='Forward', density=True)
    plt.hist(flow_mag_bwd.flatten(), bins=50, alpha=0.7, label='Backward', density=True)
    plt.xlabel('Flow Magnitude (pixels)')
    plt.ylabel('Density')
    plt.title('Flow Magnitude Distribution')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(debug_dir / f"{prefix}_flow_analysis.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"[DEBUG] Flow stats - Forward: mean={flow_mag_fwd.mean():.2f}, max={flow_mag_fwd.max():.2f}")
    print(f"[DEBUG] Flow stats - Backward: mean={flow_mag_bwd.mean():.2f}, max={flow_mag_bwd.max():.2f}")
    print(f"[DEBUG] Flow range - Fwd X: [{flow_fwd[:,:,0].min():.2f}, {flow_fwd[:,:,0].max():.2f}]")
    print(f"[DEBUG] Flow range - Fwd Y: [{flow_fwd[:,:,1].min():.2f}, {flow_fwd[:,:,1].max():.2f}]")

def save_correspondences_debug(pts1, pts2, rgb_image, prefix, out_dir, max_points=1000):
    """Visualize correspondences on the image."""
    debug_dir = out_dir / "debug"
    debug_dir.mkdir(exist_ok=True)
    
    h, w = rgb_image.shape[:2]
    
    # Subsample for visualization
    if len(pts1) > max_points:
        indices = np.random.choice(len(pts1), max_points, replace=False)
        pts1_sub = pts1[indices]
        pts2_sub = pts2[indices]
    else:
        pts1_sub = pts1
        pts2_sub = pts2
    
    # Create visualization
    img_vis = (rgb_image * 255).astype(np.uint8).copy()
    
    # Draw correspondences
    for (x1, y1), (x2, y2) in zip(pts1_sub, pts2_sub):
        # Draw point in frame 1
        cv2.circle(img_vis, (int(x1), int(y1)), 2, (0, 255, 0), -1)
        # Draw arrow to frame 2 correspondence
        cv2.arrowedLine(img_vis, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 1, tipLength=0.3)
    
    cv2.imwrite(str(debug_dir / f"{prefix}_correspondences.png"), cv2.cvtColor(img_vis, cv2.COLOR_RGB2BGR))
    print(f"[DEBUG] Saved correspondences visualization: {len(pts1_sub)} points")


def visualize_correspondences_interactive(frame1, frame2, pts1, pts2, max_points=100):
    """Display interactive side-by-side visualization of point correspondences."""
    print(f"[VIZ] Showing interactive correspondence visualization with {len(pts1)} total correspondences")
    print("[VIZ] Press any key to continue, 'q' to quit, 's' to save current view")
    
    h1, w1 = frame1.shape[:2]
    h2, w2 = frame2.shape[:2]
    
    # Ensure both frames have same height for side-by-side display
    if h1 != h2:
        target_h = min(h1, h2)
        frame1 = cv2.resize(frame1, (int(w1 * target_h / h1), target_h))
        frame2 = cv2.resize(frame2, (int(w2 * target_h / h2), target_h))
        # Scale correspondence points accordingly
        scale1_x, scale1_y = w1 * target_h / h1 / w1, target_h / h1
        scale2_x, scale2_y = w2 * target_h / h2 / w2, target_h / h2
        pts1 = pts1 * [scale1_x, scale1_y]
        pts2 = pts2 * [scale2_x, scale2_y]
        h1, w1 = frame1.shape[:2]
        h2, w2 = frame2.shape[:2]
    
    # Subsample correspondences for cleaner visualization
    if len(pts1) > max_points:
        indices = np.random.choice(len(pts1), max_points, replace=False)
        pts1_sub = pts1[indices].astype(int)
        pts2_sub = pts2[indices].astype(int)
    else:
        pts1_sub = pts1.astype(int)
        pts2_sub = pts2.astype(int)
    
    # Create side-by-side image
    combined_w = w1 + w2
    combined = np.zeros((h1, combined_w, 3), dtype=np.uint8)
    
    # Convert frames to uint8 if needed
    if frame1.dtype == np.float32 or frame1.dtype == np.float64:
        frame1_uint8 = (np.clip(frame1, 0, 1) * 255).astype(np.uint8)
    else:
        frame1_uint8 = frame1
    if frame2.dtype == np.float32 or frame2.dtype == np.float64:
        frame2_uint8 = (np.clip(frame2, 0, 1) * 255).astype(np.uint8)
    else:
        frame2_uint8 = frame2
    
    # Place frames side by side
    combined[:, :w1] = frame1_uint8
    combined[:, w1:] = frame2_uint8
    
    # Adjust pts2 coordinates for right side placement
    pts2_adjusted = pts2_sub.copy()
    pts2_adjusted[:, 0] += w1
    
    # Generate random colors for each correspondence
    colors = []
    for i in range(len(pts1_sub)):
        color = (
            int(np.random.randint(50, 255)),
            int(np.random.randint(50, 255)), 
            int(np.random.randint(50, 255))
        )
        colors.append(color)
    
    # Draw correspondences
    for i, ((x1, y1), (x2, y2)) in enumerate(zip(pts1_sub, pts2_adjusted)):
        color = colors[i]
        # Draw points
        cv2.circle(combined, (x1, y1), 3, color, -1)
        cv2.circle(combined, (x2, y2), 3, color, -1)
        # Draw connecting line
        cv2.line(combined, (x1, y1), (x2, y2), color, 1)
    
    # Add text overlay
    cv2.putText(combined, f"Frame 0 -> Frame 1 ({len(pts1_sub)} correspondences)", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(combined, "Press any key to continue, 'q' to quit", 
                (10, h1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Display
    cv2.imshow("Point Correspondences", combined)
    key = cv2.waitKey(0) & 0xFF
    
    if key == ord('s'):
        timestamp = np.random.randint(1000, 9999)
        save_path = f"correspondences_viz_{timestamp}.png"
        cv2.imwrite(save_path, combined)
        print(f"[VIZ] Saved visualization to {save_path}")
    
    cv2.destroyAllWindows()
    
    if key == ord('q'):
        print("[VIZ] User requested quit")
        return False
    
    return True

def save_point_cloud_debug(pcd, prefix, out_dir):
    """Save point cloud statistics and visualization."""
    debug_dir = out_dir / "debug"
    debug_dir.mkdir(exist_ok=True)
    
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    
    stats = {
        "num_points": len(points),
        "x_range": [float(points[:, 0].min()), float(points[:, 0].max())],
        "y_range": [float(points[:, 1].min()), float(points[:, 1].max())],
        "z_range": [float(points[:, 2].min()), float(points[:, 2].max())],
        "x_mean": float(points[:, 0].mean()),
        "y_mean": float(points[:, 1].mean()),
        "z_mean": float(points[:, 2].mean()),
        "x_std": float(points[:, 0].std()),
        "y_std": float(points[:, 1].std()),
        "z_std": float(points[:, 2].std()),
    }
    
    # Save stats
    with open(debug_dir / f"{prefix}_pointcloud_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    
    print(f"[DEBUG] Point cloud {prefix}: {len(points)} points")
    print(f"[DEBUG] X: [{stats['x_range'][0]:.3f}, {stats['x_range'][1]:.3f}], mean={stats['x_mean']:.3f}")
    print(f"[DEBUG] Y: [{stats['y_range'][0]:.3f}, {stats['y_range'][1]:.3f}], mean={stats['y_mean']:.3f}")
    print(f"[DEBUG] Z: [{stats['z_range'][0]:.3f}, {stats['z_range'][1]:.3f}], mean={stats['z_mean']:.3f}")
    
    return stats


def main():
    parser = argparse.ArgumentParser(description="Generate 3D point clouds using UniDepth or flow triangulation with different focal lengths (GT/AnyCam/AnyCalib).")
    parser.add_argument("--video_path", type=str, default=None, help="Path to video or image folder; if omitted, a random unprocessed video from --default_videos_dir is selected")
    parser.add_argument("--output_dir", type=str, default=None, help="Base directory to save PLYs (defaults to experiments/point_clouds/<video>_frames-1-2)")
    parser.add_argument("--gt_dir", type=str, default="/home/kalman/TUM/thesis/Objectron/processed_gt/",
                        help="Directory containing GT poses and intrinsics JSON files")
    parser.add_argument("--frames", type=int, nargs=2, default=[0, 1],
                        help="Which two frame indices to use (default: 0 1)")
    parser.add_argument("--default_videos_dir", type=str, default="/home/kalman/TUM/thesis/Objectron/videos/",
                        help="Default directory to sample videos when --video_path is omitted")
    parser.add_argument("--unidepth_version", type=str, default="v2", help="UniDepth version")
    parser.add_argument("--unidepth_backbone", type=str, default="vits14")
    parser.add_argument("--unidepth_scaling", type=float, default=0.1)
    parser.add_argument("--model_path", type=str, default="pretrained_models/anycam_seq8")
    parser.add_argument("--method", type=str, default="depth", choices=["depth", "triang"], help="depth (UniDepth) or triang (UniMatch+triangulation)")
    parser.add_argument("--unimatch_ckpt", type=str, default="", help="Optional: path to UniMatch flow checkpoint (.pth). If omitted, use AnyCam cached path and auto-download if missing")
    parser.add_argument("--triang_step", type=int, default=2, help="Flow sampling step for triangulation (lower = denser)")
    parser.add_argument("--cycle_thresh", type=float, default=1.0, help="Cycle consistency threshold for flow filtering (pixels)")
    parser.add_argument("--visualize_correspondences", action="store_true", help="Show interactive visualization of point correspondences between frames")
    args = parser.parse_args()

    # Resolve default base output dir under experiments/point_clouds
    script_dir = Path(__file__).parent  # experiments/
    default_output_base = script_dir / "point_clouds"

    # Determine input video
    if args.video_path is None:
        video_path = pick_random_unprocessed_video(Path(args.default_videos_dir), default_output_base)
        print(f"[AUTO] Selected random unprocessed video: {video_path}")
    else:
        video_path = Path(args.video_path)

    # Compute default output directory under experiments/point_clouds/<video>_frames-X-Y
    if args.output_dir is None:
        exp_name = f"{video_path.stem}_frames-{args.frames[0]}-{args.frames[1]}"
        out_dir = default_output_base / exp_name
    else:
        out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load frames (both methods need frame pair for AnyCam focal estimation)
    data_mgr = ExperimentDataManager()
    frame0, _ = data_mgr.load_experiment_data(
        input_path=str(video_path), num_frames=1, start_frame=args.frames[0], skip_frames=1, gt_dir=None
    )
    frame1, _ = data_mgr.load_experiment_data(
        input_path=str(video_path), num_frames=1, start_frame=args.frames[1], skip_frames=1, gt_dir=None
    )
    frames = frame0 + frame1
    print(f"[DEBUG] Loaded frame {args.frames[0]} and frame {args.frames[1]} from video")
    print(f"[DEBUG] Frame 0 shape: {frames[0].shape}, Frame 1 shape: {frames[1].shape}")
    
    if len(frames) < 2:
        raise SystemExit("Failed to load two frames from input")

    h, w = frames[0].shape[:2]
    cx, cy = w / 2.0, h / 2.0

    # GT focal from intrinsics (use first frame index)
    gt_focal = load_gt_focal_from_intrinsics(Path(args.gt_dir), video_path, frame_idx=args.frames[0])

    # AnyCam: run on the two frames and extract focal only
    anycam_engine = create_inference_engine(model_path=args.model_path)
    print(f"[DEBUG] Running AnyCam on frames {args.frames[0]} and {args.frames[1]}")
    result = anycam_engine.run_inference_on_pair([frames[0], frames[1]], pair_name=video_path.stem, ba_refinement=False)
    if result is None:
        raise SystemExit("AnyCam inference failed")
    anycam_focal = extract_anycam_focal(result["projection"])
    anycam_engine.clear_cache()

    # AnyCalib: run on first frame only for single-frame comparison
    anycalib_focal = run_anycalib_single_frame(frames[0], device)

    focals = {
        "gt": gt_focal,
        "anycam": anycam_focal,
        "anycalib": anycalib_focal,
    }

    if args.method == "depth":
        # Depth method: UniDepth on first frame only for fair single-frame comparison
        print(f"[DEBUG] Depth method: running UniDepth on single frame {args.frames[0]}")
        
        from anycam.models import make_depth_predictor
        conf = {
            "type": "unidepth",
            "version": args.unidepth_version,
            "backbone": args.unidepth_backbone,
            "scaling": args.unidepth_scaling,
        }
        depth_predictor = make_depth_predictor(conf).to(device).eval()

        # Process only the first frame with UniDepth
        with torch.no_grad():
            rgb = frames[0].astype(np.float32) / 255.0
            tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(device)
            inv_depth_list = depth_predictor(tensor)
            inv_depth = inv_depth_list[0]  # Bx1xHxW
            depth = (1.0 / inv_depth.clamp_min(1e-6)).squeeze(0).squeeze(0).detach().cpu().numpy()

        # Generate point clouds using only the single frame
        for tag, f in focals.items():
            pcd = depth_to_point_cloud(frames[0], depth, f, cx, cy)
            ply_path = out_dir / f"{video_path.stem}_{tag}_depth_single.ply"
            o3d.io.write_point_cloud(str(ply_path), pcd)
            print(f"Saved single-frame depth {ply_path}")
            
    else:
        # Triangulation method: use both frames
        # Update AnyCalib to use both frames for averaging (override single-frame version)
        anycalib_focal = run_anycalib_per_frame_average_focal([frames[0], frames[1]], device)
        focals["anycalib"] = anycalib_focal

        # Relative pose from GT JSON directory (Objectron c2w): T_0_to_1 = inv(P0) @ P1
        gt_dir = Path(args.gt_dir)
        T_01 = load_gt_relative_pose_from_dir(gt_dir, video_path, tuple(args.frames))

        # Triangulation via UniMatch flow (fwd and bwd for filtering)
        ckpt = Path(args.unimatch_ckpt) if args.unimatch_ckpt else None
        print(f"[DEBUG] Running UniMatch optical flow on frames {args.frames[0]} and {args.frames[1]}")
        print(f"[DEBUG] Frame 0 range: [{frames[0].min():.3f}, {frames[0].max():.3f}]")
        print(f"[DEBUG] Frame 1 range: [{frames[1].min():.3f}, {frames[1].max():.3f}]")
        flow_fwd, flow_bwd = run_optical_flow_unimatch(frames, device, ckpt)
        
        # DEBUG: Save input frames and flow visualizations
        save_debug_images(frames, flow_fwd, flow_bwd, "unimatch", out_dir)
        
        for tag, f in focals.items():
            pcd = triangulate_point_cloud(frames[0], flow_fwd, flow_bwd, f, cx, cy, T_01, step=args.triang_step, cycle_thresh=args.cycle_thresh, debug_dir=out_dir, visualize_correspondences=args.visualize_correspondences, frame2=frames[1])
            ply_path = out_dir / f"{video_path.stem}_{tag}_triang.ply"
            o3d.io.write_point_cloud(str(ply_path), pcd)
            print(f"Saved {ply_path}")

    # Scale triangulated clouds to match median Z from GT depth cloud if method=triang
    if args.method == "triang":
        # Compute median Z from GT depth on frame 0
        gt_pcd = o3d.io.read_point_cloud(str(out_dir / f"{video_path.stem}_gt_triang.ply"))
        gt_points = np.asarray(gt_pcd.points)
        median_z_gt = np.median(gt_points[:, 2]) if len(gt_points) > 0 else 1.0

        for tag in focals:
            ply_path = out_dir / f"{video_path.stem}_{tag}_triang.ply"
            pcd = o3d.io.read_point_cloud(str(ply_path))
            points = np.asarray(pcd.points)
            if len(points) == 0:
                continue
            median_z = np.median(points[:, 2])
            if median_z > 0:
                scale = median_z_gt / median_z
                points *= scale
                pcd.points = o3d.utility.Vector3dVector(points)
                o3d.io.write_point_cloud(str(ply_path), pcd)  # overwrite with scaled
                print(f"Scaled {tag} triang cloud by {scale:.3f} to match GT median Z={median_z_gt:.3f}")

    # Compute quantitative metrics comparing to GT
    metrics: Dict[str, Dict[str, float]] = {}
    
    if args.method == "depth":
        # Single-frame depth comparison
        gt_ply_path = out_dir / f"{video_path.stem}_gt_depth_single.ply"
        file_suffix = "_depth_single.ply"
        comparison_tags = ["anycam", "anycalib"]
    else:
        # Triangulation comparison  
        gt_ply_path = out_dir / f"{video_path.stem}_gt_triang.ply"
        file_suffix = "_triang.ply"
        comparison_tags = ["anycam", "anycalib"]
    
    try:
        gt_pcd = o3d.io.read_point_cloud(str(gt_ply_path))
        if len(gt_pcd.points) == 0:
            print(f"[WARN] GT point cloud is empty: {gt_ply_path}")
        else:
            for tag in comparison_tags:
                target_ply_path = out_dir / f"{video_path.stem}_{tag}{file_suffix}"
                if target_ply_path.exists():
                    target_pcd = o3d.io.read_point_cloud(str(target_ply_path))
                    if len(target_pcd.points) > 0:
                        chamfer = compute_chamfer_distance(gt_pcd, target_pcd)
                        hausdorff = compute_hausdorff_distance(gt_pcd, target_pcd)
                        metrics[tag] = {"chamfer": chamfer, "hausdorff": hausdorff}
                        print(f"{tag.capitalize()} vs. GT: Chamfer={chamfer:.4f}m, Hausdorff={hausdorff:.4f}m")
                    else:
                        print(f"[WARN] {tag} point cloud is empty: {target_ply_path}")
                else:
                    print(f"[WARN] {tag} point cloud file not found: {target_ply_path}")
    except Exception as e:
        print(f"[WARN] Failed to compute metrics: {e}")
    
    # Save metrics to JSON
    if metrics:
        with open(out_dir / f"{video_path.stem}_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

    meta = {
        "method": args.method,
        "gt_focal": float(gt_focal),
        "anycam_focal": float(anycam_focal),
        "anycalib_focal": float(anycalib_focal),
        "metrics": metrics,
    }
    with open(out_dir / f"{video_path.stem}_focals.json", "w") as f:
        json.dump(meta, f, indent=2)
    print("Done.")


if __name__ == "__main__":
    main() 