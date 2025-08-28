#!/usr/bin/env python3
import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import open3d as o3d
import torch

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
    # Assuming c2w convention: T_0_to_1 = inv(P0) @ P1
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
    parser = argparse.ArgumentParser(description="Generate 3D point clouds using UniDepth depths and different focal lengths (GT/AnyCam/AnyCalib).")
    parser.add_argument("--video_path", type=str, default=None, help="Path to video or image folder; if omitted, a random unprocessed video from --default_videos_dir is selected")
    parser.add_argument("--output_dir", type=str, default=None, help="Base directory to save PLYs (defaults to experiments/point_clouds/<video>_frames-1-2)")
    parser.add_argument("--gt_dir", type=str, default="/home/kalman/TUM/thesis/Objectron/processed_gt/", help="Directory containing per-sequence GT JSONs with 'poses' and 'intrinsics'")
    parser.add_argument("--default_videos_dir", type=str, default="/home/kalman/TUM/thesis/Objectron/videos/", help="Default directory to sample videos when --video_path is omitted")
    parser.add_argument("--unidepth_version", type=str, default="v2")
    parser.add_argument("--unidepth_backbone", type=str, default="vits14")
    parser.add_argument("--unidepth_scaling", type=float, default=0.1)
    parser.add_argument("--model_path", type=str, default="pretrained_models/anycam_seq8")
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

    # AnyCalib: per-frame, average focals
    anycalib_focal = run_anycalib_per_frame_average_focal([frames[0], frames[1]], device)

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

    # Relative pose from GT JSON directory (Objectron c2w): T_0_to_1 = inv(P0) @ P1
    gt_dir = Path(args.gt_dir)
    T_01 = load_gt_relative_pose_from_dir(gt_dir, video_path)

    focals = {
        "gt": gt_focal,
        "anycam": anycam_focal,
        "anycalib": anycalib_focal,
    }

    # Generate and save merged PLYs
    for tag, f in focals.items():
        pcd0 = depth_to_point_cloud(frames[0], depths[0], f, cx, cy)
        pcd1 = depth_to_point_cloud(frames[1], depths[1], f, cx, cy)
        merged = align_and_merge_pcds(pcd0, pcd1, T_01)
        ply_path = out_dir / f"{video_path.stem}_{tag}_merged.ply"
        o3d.io.write_point_cloud(str(ply_path), merged)
        print(f"Saved {ply_path}")

    # Load merged PLYs and compute metrics vs GT
    metrics: Dict[str, Dict[str, float]] = {}
    gt_ply_path = out_dir / f"{video_path.stem}_gt_merged.ply"
    gt_pcd = o3d.io.read_point_cloud(str(gt_ply_path))
    for tag in ["anycam", "anycalib"]:
        target_ply_path = out_dir / f"{video_path.stem}_{tag}_merged.ply"
        target_pcd = o3d.io.read_point_cloud(str(target_ply_path))
        chamfer = compute_chamfer_distance(gt_pcd, target_pcd)
        hausdorff = compute_hausdorff_distance(gt_pcd, target_pcd)
        metrics[tag] = {"chamfer": chamfer, "hausdorff": hausdorff}
        print(f"{tag.capitalize()} vs. GT: Chamfer={chamfer:.4f}m, Hausdorff={hausdorff:.4f}m")

    # Save JSON with focals and metrics
    meta = {
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