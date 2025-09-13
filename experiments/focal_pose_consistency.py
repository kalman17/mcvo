#!/usr/bin/env python3
import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import open3d as o3d
import torch
import cv2  # For triangulation and fundamental matrix
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
from anycam.trainer import make_proj_from_focal_length

def rgb_to_tensor(rgb: np.ndarray) -> torch.Tensor:
    if rgb.dtype != np.float32:
        rgb = rgb.astype(np.float32) / 255.0
    return torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)

# --- Optical flow with UniMatch ---
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
    flow_bwd = flows[n:]  # next n are backward
    flow_fwd_np = flow_fwd[0].permute(1, 2, 0).detach().cpu().numpy()  # (H, W, 2)
    flow_bwd_np = flow_bwd[0].permute(1, 2, 0).detach().cpu().numpy()  # (H, W, 2)
    return flow_fwd_np, flow_bwd_np

# --- Get point correspondences from flow ---
def get_point_correspondences(flow_fwd: np.ndarray, flow_bwd: np.ndarray, step: int = 2, cycle_thresh: float = 1.0, occ_fwd: np.ndarray | None = None, occ_bwd: np.ndarray | None = None) -> Tuple[np.ndarray, np.ndarray]:
    """Extract filtered point correspondences from optical flow.
    If occ_fwd/occ_bwd provided, they should be (H, W) masks where 0 means valid and 1 means occluded.
    """
    h, w = flow_fwd.shape[:2]
    u, v = np.meshgrid(np.arange(0, w, step), np.arange(0, h, step))
    u_flat, v_flat = u.flatten(), v.flatten()
    flow_u = flow_fwd[v, u, 0].flatten()
    flow_v = flow_fwd[v, u, 1].flatten()
    u2_flat = u_flat + flow_u
    v2_flat = v_flat + flow_v
    print(f"[CORR] Initial correspondences: {len(u_flat)}")
    print(f"[CORR] Flow range: u=({flow_u.min():.2f}, {flow_u.max():.2f}), v=({flow_v.min():.2f}, {flow_v.max():.2f})")
    # Filter in-bounds correspondences
    valid_bound = (u2_flat >= 0) & (u2_flat < w) & (v2_flat >= 0) & (v2_flat < h)
    # Occlusion filtering (0 valid, 1 occluded)
    if occ_fwd is not None and occ_bwd is not None:
        occ_src = occ_fwd[v, u].flatten()
        occ_src_valid = occ_src > 0.5  # 1 means valid in AnyCam masks
        u2_int_tmp = np.clip(np.round(u2_flat).astype(int), 0, w - 1)
        v2_int_tmp = np.clip(np.round(v2_flat).astype(int), 0, h - 1)
        occ_dst = occ_bwd[v2_int_tmp, u2_int_tmp]
        occ_dst_valid = occ_dst > 0.5
        valid_bound = valid_bound & occ_src_valid & occ_dst_valid
    print(f"[CORR] Valid bounds: {valid_bound.sum()}/{len(u_flat)}")
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
    print(f"[CORR] Cycle consistency (thresh={cycle_thresh}): {valid_cycle.sum()}/{len(valid_cycle)}")
    print(f"[CORR] Cycle error range: {cycle_error.min():.2f} - {cycle_error.max():.2f}")
    final_valid_indices = np.where(valid_bound)[0][valid_cycle]
    if len(final_valid_indices) == 0:
        print(f"[CORR] ERROR: No correspondences survived cycle filtering. Try increasing --cycle_thresh (current: {cycle_thresh})")
        # Fallback: try with relaxed cycle consistency
        fallback_thresh = min(cycle_thresh * 3, 5.0)  # Try 3x threshold, max 5 pixels
        print(f"[CORR] Trying fallback cycle threshold: {fallback_thresh}")
        valid_cycle_fallback = cycle_error < fallback_thresh
        final_valid_indices = np.where(valid_bound)[0][valid_cycle_fallback]
        if len(final_valid_indices) == 0:
            print(f"[CORR] Still no correspondences with relaxed threshold. Flow might be poor quality.")
            raise ValueError("No valid correspondences after cycle filtering")
        else:
            print(f"[CORR] Fallback successful: {len(final_valid_indices)} correspondences")
    # Get final pixel correspondences
    pts1_px = np.stack([u_flat[final_valid_indices], v_flat[final_valid_indices]], axis=1).astype(np.float32)
    pts2_px = np.stack([u2_flat[final_valid_indices], v2_flat[final_valid_indices]], axis=1).astype(np.float32)
    print(f"[CORR] Final correspondences: {len(pts1_px)}")
    return pts1_px, pts2_px

def compute_fundamental_from_pose_and_focal(f: float, cx: float, cy: float, T: np.ndarray) -> np.ndarray:
    """Compute fundamental matrix from pose and focal length."""
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float32)
    invK = np.linalg.inv(K)
    R = T[:3, :3]
    t = T[:3, 3]
    skew_t = np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]], dtype=np.float32)
    E = skew_t @ R
    F = invK.T @ E @ invK
    return F

def distF(F1: np.ndarray, F2: np.ndarray) -> float:
    """Compute distance between two fundamental matrices."""
    F1 = F1 / np.linalg.norm(F1)
    F2 = F2 / np.linalg.norm(F2)
    d1 = np.linalg.norm(F1 - F2)
    d2 = np.linalg.norm(F1 + F2)
    return min(d1, d2)

def extract_anycam_focal(projection: np.ndarray) -> float:
    # projection is 3x3; average fx, fy
    fx = float(projection[0, 0])
    fy = float(projection[1, 1])
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

def main():
    parser = argparse.ArgumentParser(description="Test consistency of AnyCam's candidate predictions with fundamental matrix from optical flow.")
    parser.add_argument("--video_path", type=str, default=None, help="Path to video or image folder; if omitted, a random unprocessed video from --default_videos_dir is selected")
    parser.add_argument("--output_dir", type=str, default=None, help="Base directory to save results (defaults to experiments/consistency_tests/<video>_frames-1-2)")
    parser.add_argument("--gt_dir", type=str, default="/home/kalman/TUM/thesis/Objectron/processed_gt/",
                        help="Directory containing GT poses and intrinsics JSON files")
    parser.add_argument("--frames", type=int, nargs=2, default=[0, 1],
                        help="Which two frame indices to use (default: 0 1)")
    parser.add_argument("--default_videos_dir", type=str, default="/home/kalman/TUM/thesis/Objectron/videos/",
                        help="Default directory to sample videos when --video_path is omitted")
    parser.add_argument("--model_path", type=str, default="pretrained_models/anycam_seq8")
    parser.add_argument("--unimatch_ckpt", type=str, default="", help="Optional: path to UniMatch flow checkpoint (.pth). If omitted, use AnyCam cached path and auto-download if missing")
    parser.add_argument("--triang_step", type=int, default=2, help="Flow sampling step for correspondences (lower = denser)")
    parser.add_argument("--cycle_thresh", type=float, default=1.0, help="Cycle consistency threshold for flow filtering (pixels)")
    parser.add_argument("--dist_thresh", type=float, default=0.05, help="Threshold for considering a fundamental matrix distance 'good' (default: 5%)")
    args = parser.parse_args()

    # Resolve default base output dir under experiments/consistency_tests
    script_dir = Path(__file__).parent  # experiments/
    default_output_base = script_dir / "consistency_tests"
    # Determine input video
    if args.video_path is None:
        video_path = pick_random_unprocessed_video(Path(args.default_videos_dir), default_output_base)
        print(f"[AUTO] Selected random unprocessed video: {video_path}")
    else:
        video_path = Path(args.video_path)
    # Compute default output directory under experiments/consistency_tests/<video>_frames-X-Y
    if args.output_dir is None:
        exp_name = f"{video_path.stem}_frames-{args.frames[0]}-{args.frames[1]}"
        out_dir = default_output_base / exp_name
    else:
        out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load frames
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

    # Relative pose from GT JSON directory (Objectron c2w): T_0_to_1 = inv(P0) @ P1
    gt_dir = Path(args.gt_dir)
    T_01_gt = load_gt_relative_pose_from_dir(gt_dir, video_path, tuple(args.frames))

    # Run AnyCam on the pair
    # NOTE: Assume you have modified experiments.common.anycam_inference.create_inference_engine and run_inference_on_pair
    # to support return_candidates=True, which returns a dict with "best": {...}, "candidates": list of 32 dicts each with "projection" and "pose" (or "trajectory" for relative pose)
    # For example, in the inference code, collect all hypotheses before selecting the best based on likelihood.
    # Since the terminal shows "Best candidate: 16", likely there are 32 candidates.
    anycam_engine = create_inference_engine(model_path=args.model_path)
    print(f"[DEBUG] Running AnyCam on frames {args.frames[0]} and {args.frames[1]}")
    result = anycam_engine.run_inference_on_pair([frames[0], frames[1]], pair_name=video_path.stem, ba_refinement=False)
    if result is None:
        raise SystemExit("AnyCam inference failed")
    # Extract best for reference
    anycam_best_focal = extract_anycam_focal(result["projection"])
    anycam_best_pose = anycam_engine.extract_relative_pose(result["trajectory"])  # relative pose T_0_to_1 from best trajectory
    # Get all candidates from extras_dict if available
    extras = result.get("extras_dict", {}) if isinstance(result, dict) else {}
    anycam_focals = []
    anycam_poses = []
    best_candidate_index = None
    # Determine internal processing resolution from AnyCam (work entirely at this scale)
    h_int, w_int = h, w
    if extras and "images" in extras:
        imgs_t = extras["images"]
        if isinstance(imgs_t, torch.Tensor):
            # imgs_t shape: [N, C, H, W]
            h_int = int(imgs_t.shape[-2])
            w_int = int(imgs_t.shape[-1])
        else:
            # Fallback to original size
            pass
    cx, cy = w_int / 2.0, h_int / 2.0
    if extras and "focal_length_candidates" in extras and "candidate_trajectories" in extras:
        fl_cand = extras["focal_length_candidates"]  # Tensor [1, N]
        if isinstance(fl_cand, torch.Tensor):
            fl_cand = fl_cand.detach().cpu()
        num_cand = int(fl_cand.shape[-1])
        # Recreate projection matrices for each candidate in internal pixel units (AnyCam processing scale)
        projs = make_proj_from_focal_length(fl_cand[0:1, :], w_int / h_int)[0]  # [N, 3, 3] normalized
        # Map to pixel coordinates at (w_int, h_int), matching fit_video normalization logic
        projs[:, 0, 0] = (projs[:, 0, 0] * 0.5) * w_int
        projs[:, 1, 1] = (projs[:, 1, 1] * 0.5) * h_int
        projs[:, 0, 2] = (projs[:, 0, 2] * 0.5 + 0.5) * w_int
        projs[:, 1, 2] = (projs[:, 1, 2] * 0.5 + 0.5) * h_int
        anycam_focals = projs[:, 0, 0].detach().cpu().numpy().tolist()
        # Candidate relative poses from candidate_trajectories (list of length >= 2, each [N,4,4])
        cand_traj = extras["candidate_trajectories"]
        T0 = cand_traj[0]
        T1 = cand_traj[1]
        if isinstance(T0, torch.Tensor):
            T0 = T0.detach().cpu().numpy()
        if isinstance(T1, torch.Tensor):
            T1 = T1.detach().cpu().numpy()
        anycam_poses = [T1[i] @ np.linalg.inv(T0[i]) for i in range(T0.shape[0])]
        best_candidate_index = int(extras.get("best_candidate_index", 0))
        if len(anycam_focals) != 32:
            print(f"[WARN] Expected 32 candidates, got {len(anycam_focals)}")
    else:
        print("[WARN] Extras missing candidates; analyzing best prediction only")
        anycam_focals = [anycam_best_focal]
        anycam_poses = [anycam_best_pose]
        best_candidate_index = 0
    # Use AnyCam-provided optical flow at internal processing scale
    if not extras or "seq_flow_occs_fwd" not in extras or "seq_flow_occs_bwd" not in extras:
        raise SystemExit("AnyCam extras missing flow; cannot proceed without consistent internal flow")
    flow_fwd_stack = extras["seq_flow_occs_fwd"]  # [T, 3, H, W]
    flow_bwd_stack = extras["seq_flow_occs_bwd"]  # [T, 3, H, W]
    # Choose time indices: forward at 0, backward at 1 if available (first is often zero-padded)
    fwd_idx = 0
    bwd_idx = 1 if (isinstance(flow_bwd_stack, torch.Tensor) and flow_bwd_stack.shape[0] > 1) or (isinstance(flow_bwd_stack, (list, tuple)) and len(flow_bwd_stack) > 1) else 0
    flow_fwd_t = flow_fwd_stack[fwd_idx]
    flow_bwd_t = flow_bwd_stack[bwd_idx]
    if isinstance(flow_fwd_t, torch.Tensor):
        flow_fwd_t = flow_fwd_t.detach().cpu().numpy()
    if isinstance(flow_bwd_t, torch.Tensor):
        flow_bwd_t = flow_bwd_t.detach().cpu().numpy()
    # First two channels: normalized flow (x,y) in [-1,1] domain scaled by 2/w, 2/h; convert to pixels
    flow_fwd = np.stack([
        flow_fwd_t[0] * (w_int / 2.0),
        flow_fwd_t[1] * (h_int / 2.0)
    ], axis=-1)  # (H, W, 2)
    flow_bwd = np.stack([
        flow_bwd_t[0] * (w_int / 2.0),
        flow_bwd_t[1] * (h_int / 2.0)
    ], axis=-1)  # (H, W, 2)
    # Occlusion masks (3rd channel): 0 valid, 1 occluded
    occ_fwd = (flow_fwd_t[2]).astype(np.float32)
    occ_bwd = (flow_bwd_t[2]).astype(np.float32)

    # Get point correspondences
    pts1_px, pts2_px = get_point_correspondences(flow_fwd, flow_bwd, step=args.triang_step, cycle_thresh=args.cycle_thresh, occ_fwd=occ_fwd, occ_bwd=occ_bwd)

    # Compute fundamental matrix from flow correspondences
    F_flow, mask = cv2.findFundamentalMat(pts1_px, pts2_px, cv2.FM_RANSAC, ransacReprojThreshold=1.0)
    if F_flow is None or mask is None or int(np.sum(mask)) < 8:
        print(f"[FUND] RANSAC failed or insufficient inliers ({0 if mask is None else int(np.sum(mask))}/{len(pts1_px)}). Trying 8-point without RANSAC...")
        if len(pts1_px) >= 8:
            F_flow = cv2.findFundamentalMat(pts1_px, pts2_px, cv2.FM_8POINT)[0]
        else:
            F_flow = None
    if F_flow is None:
        raise SystemExit("Failed to estimate fundamental matrix: not enough high-quality correspondences. Try lowering --triang_step or increasing --cycle_thresh.")
    inliers = int(np.sum(mask)) if mask is not None else len(pts1_px)
    print(f"[FUND] Computed F_flow from {inliers} inliers out of {len(pts1_px)} correspondences")
    F_flow = F_flow.astype(np.float64)
    F_flow /= np.linalg.norm(F_flow)

    # For each AnyCam candidate, compute F_pred and distance to F_flow
    dists = []
    for i, (f, T) in enumerate(zip(anycam_focals, anycam_poses)):
        F_pred = compute_fundamental_from_pose_and_focal(f, cx, cy, T)
        F_pred /= np.linalg.norm(F_pred)
        d = distF(F_pred, F_flow)
        dists.append(d)
        print(f"[DIST] Candidate {i}: focal={f:.1f}, dist={d:.6f}")

    # Analyze results
    min_dist = np.min(dists)
    mean_dist = np.mean(dists)
    num_good = np.sum(np.array(dists) < args.dist_thresh)
    best_idx = np.argmin(dists)
    print(f"\n[ANALYSIS] Min distance: {min_dist:.6f} (candidate {best_idx})")
    print(f"[ANALYSIS] Mean distance: {mean_dist:.6f}")
    total_cands = len(dists)
    print(f"[ANALYSIS] Number of good candidates (dist < {args.dist_thresh}): {num_good}/{total_cands}")
    if num_good == total_cands:
        print("[SCENARIO] All predictions are consistent with F_flow")
    elif num_good > 0:
        print(f"[SCENARIO] Some predictions ({num_good}/{total_cands}) are consistent with F_flow")
    else:
        print("[SCENARIO] None of the predictions are consistent with F_flow")
    # Check if the selected best matches the one with min dist
    selected_idx = best_candidate_index if best_candidate_index is not None else 0
    anycam_best_dist = dists[selected_idx]
    if anycam_best_dist > min_dist:
        print(f"[INSIGHT] AnyCam selected a candidate with dist={anycam_best_dist:.6f}, but better exists (min={min_dist:.6f}), suggesting it discards good focals/poses")
    else:
        print(f"[INSIGHT] AnyCam selected the best consistent candidate (dist={anycam_best_dist:.6f})")

    # Save results
    metrics = {
        "dists": dists,
        "min_dist": float(min_dist),
        "mean_dist": float(mean_dist),
        "num_good": int(num_good),
        "best_idx": int(best_idx),
        "anycam_best_dist": float(anycam_best_dist),
        "anycam_focals": anycam_focals,
    }
    with open(out_dir / f"{video_path.stem}_consistency.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("Done.")

if __name__ == "__main__":
    main()