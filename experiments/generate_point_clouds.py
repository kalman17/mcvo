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


def triangulate_point_cloud(rgb_image: np.ndarray, flow_fwd: np.ndarray, flow_bwd: np.ndarray, focal_length: float, cx: float, cy: float, T_0_to_1: np.ndarray, step: int = 2, cycle_thresh: float = 1.0) -> o3d.geometry.PointCloud:
    """Triangulate 3D points from optical flow correspondences.
    
    Fixed major issues:
    1. cv2.triangulatePoints expects projection matrices WITH intrinsics, not normalized coordinates
    2. T_0_to_1 should transform from cam1 to cam0 for OpenCV convention
    3. Proper handling of homogeneous coordinates and depth filtering
    """
    h, w = flow_fwd.shape[:2]
    u, v = np.meshgrid(np.arange(0, w, step), np.arange(0, h, step))
    u_flat, v_flat = u.flatten(), v.flatten()

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
    return pcd


def load_gt_focal_from_intrinsics(gt_dir: Path, video_path: Path) -> float:
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
    if len(intr) < 2:
        raise ValueError(f"GT intrinsics has fewer than 2 entries: {gt_file}")
    K0 = np.array(intr[0], dtype=np.float64).reshape(3, 3)
    K1 = np.array(intr[1], dtype=np.float64).reshape(3, 3)
    f0 = (float(K0[0, 0]) + float(K0[1, 1])) / 2.0
    f1 = (float(K1[0, 0]) + float(K1[1, 1])) / 2.0
    return (f0 + f1) / 2.0


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


def load_gt_relative_pose_from_dir(gt_dir: Path, video_path: Path) -> np.ndarray:
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
    if len(poses_flat) < 2:
        print(f"[WARN] GT JSON has fewer than 2 poses: {gt_file}; using identity for alignment")
        return np.eye(4, dtype=np.float64)
    P0 = np.array(poses_flat[0], dtype=np.float64).reshape(4, 4)
    P1 = np.array(poses_flat[1], dtype=np.float64).reshape(4, 4)
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


def main():
    parser = argparse.ArgumentParser(description="Generate 3D point clouds using UniDepth or flow triangulation with different focal lengths (GT/AnyCam/AnyCalib).")
    parser.add_argument("--video_path", type=str, default=None, help="Path to video or image folder; if omitted, a random unprocessed video from --default_videos_dir is selected")
    parser.add_argument("--output_dir", type=str, default=None, help="Base directory to save PLYs (defaults to experiments/point_clouds/<video>_frames-1-2)")
    parser.add_argument("--gt_dir", type=str, default="/home/kalman/TUM/thesis/Objectron/processed_gt/", help="Directory containing per-sequence GT JSONs with 'poses' and 'intrinsics'")
    parser.add_argument("--default_videos_dir", type=str, default="/home/kalman/TUM/thesis/Objectron/videos/", help="Default directory to sample videos when --video_path is omitted")
    parser.add_argument("--unidepth_version", type=str, default="v2")
    parser.add_argument("--unidepth_backbone", type=str, default="vits14")
    parser.add_argument("--unidepth_scaling", type=float, default=0.1)
    parser.add_argument("--model_path", type=str, default="pretrained_models/anycam_seq8")
    parser.add_argument("--method", type=str, default="depth", choices=["depth", "triang"], help="depth (UniDepth) or triang (UniMatch+triangulation)")
    parser.add_argument("--unimatch_ckpt", type=str, default="", help="Optional: path to UniMatch flow checkpoint (.pth). If omitted, use AnyCam cached path and auto-download if missing")
    parser.add_argument("--triang_step", type=int, default=2, help="Flow sampling step for triangulation (lower = denser)")
    parser.add_argument("--cycle_thresh", type=float, default=1.0, help="Cycle consistency threshold for flow filtering (pixels)")
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

    # Compute default output directory under experiments/point_clouds/<video>_frames-1-2
    if args.output_dir is None:
        exp_name = f"{video_path.stem}_frames-1-2"
        out_dir = default_output_base / exp_name
    else:
        out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load first two frames
    data_mgr = ExperimentDataManager()
    frames, _ = data_mgr.load_experiment_data(
        input_path=str(video_path), num_frames=2, start_frame=0, skip_frames=1, gt_dir=None
    )
    if len(frames) < 2:
        raise SystemExit("Failed to load two frames from input")

    h, w = frames[0].shape[:2]
    cx, cy = w / 2.0, h / 2.0

    # GT focal from intrinsics in per-sequence JSON
    gt_focal = load_gt_focal_from_intrinsics(Path(args.gt_dir), video_path)

    # AnyCam: run on the two frames and extract focal only (do not use AnyCam poses)
    anycam_engine = create_inference_engine(model_path=args.model_path)
    result = anycam_engine.run_inference_on_pair([frames[0], frames[1]], pair_name=video_path.stem, ba_refinement=False)
    if result is None:
        raise SystemExit("AnyCam inference failed")
    proj = result["projection"]  # numpy 3x3
    anycam_focal = extract_anycam_focal(proj)

    # AnyCalib focal (batch)
    anycalib_focal = run_anycalib_per_frame_average_focal([frames[0], frames[1]], device)

    # Free AnyCam model from GPU to reduce VRAM before running UniMatch
    try:
        anycam_engine.clear_cache()
    except Exception:
        pass
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Relative pose from GT JSON directory (Objectron c2w): T_0_to_1 = inv(P0) @ P1
    gt_dir = Path(args.gt_dir)
    T_01 = load_gt_relative_pose_from_dir(gt_dir, video_path)

    focals = {
        "gt": gt_focal,
        "anycam": anycam_focal,
        "anycalib": anycalib_focal,
    }

    metrics: Dict[str, Dict[str, float]] = {}

    if args.method == "depth":
        # UniDepth: run depths for both frames
        conf = {
            "type": "unidepth",
            "version": args.unidepth_version,
            "backbone": args.unidepth_backbone,
            "scaling": args.unidepth_scaling,
        }
        depth_predictor = make_depth_predictor(conf).to(device).eval()

        depths: List[np.ndarray] = []
        with torch.no_grad():
            for frame in [frames[0], frames[1]]:
                rgb = frame.astype(np.float32) / 255.0
                tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(device)
                inv_depth_list = depth_predictor(tensor)
                inv_depth = inv_depth_list[0]  # Bx1xHxW
                depth = (1.0 / inv_depth.clamp_min(1e-6)).squeeze(0).squeeze(0).detach().cpu().numpy()
                depths.append(depth)

        # Generate and save merged PLYs
        for tag, f in focals.items():
            pcd0 = depth_to_point_cloud(frames[0], depths[0], f, cx, cy)
            pcd1 = depth_to_point_cloud(frames[1], depths[1], f, cx, cy)
            merged = align_and_merge_pcds(pcd0, pcd1, T_01)
            ply_path = out_dir / f"{video_path.stem}_{tag}_merged.ply"
            o3d.io.write_point_cloud(str(ply_path), merged)
            print(f"Saved {ply_path}")
    else:
        # Triangulation via UniMatch flow (fwd and bwd for filtering)
        ckpt = Path(args.unimatch_ckpt) if args.unimatch_ckpt else None
        flow_fwd, flow_bwd = run_optical_flow_unimatch(frames, device, ckpt)
        for tag, f in focals.items():
            pcd = triangulate_point_cloud(frames[0], flow_fwd, flow_bwd, f, cx, cy, T_01, step=args.triang_step, cycle_thresh=args.cycle_thresh)
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

    # Metrics: Compare AnyCam/AnyCalib to GT cloud
    gt_ply_path = out_dir / f"{video_path.stem}_gt_{'merged' if args.method == 'depth' else 'triang'}.ply"
    gt_pcd = o3d.io.read_point_cloud(str(gt_ply_path))
    metrics: Dict[str, Dict[str, float]] = {}
    for tag in ["anycam", "anycalib"]:
        target_ply_path = out_dir / f"{video_path.stem}_{tag}_{'merged' if args.method == 'depth' else 'triang'}.ply"
        target_pcd = o3d.io.read_point_cloud(str(target_ply_path))
        chamfer = compute_chamfer_distance(gt_pcd, target_pcd)
        hausdorff = compute_hausdorff_distance(gt_pcd, target_pcd)
        metrics[tag] = {"chamfer": chamfer, "hausdorff": hausdorff}
        print(f"{tag.capitalize()} vs. GT: Chamfer={chamfer:.4f}m, Hausdorff={hausdorff:.4f}m")

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