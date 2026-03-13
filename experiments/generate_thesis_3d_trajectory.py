"""
Generate 3D bird's-eye trajectory visualization with camera frustums.

Shows GT, Ours, and AnyCam camera paths + orientations in 3D space.
Optionally adds a sparse point cloud from Sintel GT depth for scene context.

Usage:
    python experiments/generate_thesis_3d_trajectory.py \
        --checkpoint /path/to/epoch_0005.pt \
        --dataset sintel --sequence market_6 \
        --output_dir thesis_results/figures/viz_3d \
        --device cuda:0
"""

import argparse
import struct
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib import rcParams

# Publication-quality settings
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 11
rcParams['axes.labelsize'] = 12
rcParams['axes.titlesize'] = 13
rcParams['legend.fontsize'] = 10
rcParams['figure.dpi'] = 300
rcParams['savefig.dpi'] = 300
rcParams['savefig.bbox'] = 'tight'
rcParams['savefig.pad_inches'] = 0.05

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments.benchmark_phase_c_checkpoints import (
    create_inference_model,
    create_baseline_model,
    load_phase_c_checkpoint,
    load_sintel_dataset,
    load_tumrgbd_dataset,
    extract_gt_from_sample,
)

C_OURS = '#2171b5'
C_ANYCAM = '#e6550d'
C_GT = '#333333'


def align_trajectory_sim3(pred_pos, gt_pos):
    """Sim(3) alignment (Umeyama)."""
    pred = pred_pos.copy()
    gt = gt_pos.copy()
    pred_mean = pred.mean(axis=0)
    gt_mean = gt.mean(axis=0)
    pred_c = pred - pred_mean
    gt_c = gt - gt_mean
    pred_var = np.sum(pred_c ** 2)
    if pred_var < 1e-12:
        return pred
    scale = np.sqrt(np.sum(gt_c ** 2) / pred_var)
    H = pred_c.T @ gt_c
    U, S, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    D = np.diag([1, 1, np.sign(d)])
    R = Vt.T @ D @ U.T
    aligned = scale * (R @ pred.T).T + gt_mean - scale * R @ pred_mean
    return aligned


def align_trajectory_sim3_full(pred_poses_4x4, gt_poses_4x4):
    """Sim(3) align full 4x4 poses (positions + rotations).

    Returns aligned positions and rotations.
    """
    pred_pos = np.array([T[:3, 3] for T in pred_poses_4x4])
    gt_pos = np.array([T[:3, 3] for T in gt_poses_4x4])

    pred = pred_pos.copy()
    gt = gt_pos.copy()
    pred_mean = pred.mean(axis=0)
    gt_mean = gt.mean(axis=0)
    pred_c = pred - pred_mean
    gt_c = gt - gt_mean
    pred_var = np.sum(pred_c ** 2)
    if pred_var < 1e-12:
        return pred_pos, np.array([T[:3, :3] for T in pred_poses_4x4])

    scale = np.sqrt(np.sum(gt_c ** 2) / pred_var)
    H = pred_c.T @ gt_c
    U, S, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    D = np.diag([1, 1, np.sign(d)])
    R_align = Vt.T @ D @ U.T

    aligned_pos = scale * (R_align @ pred.T).T + gt_mean - scale * R_align @ pred_mean
    aligned_rots = np.array([R_align @ T[:3, :3] for T in pred_poses_4x4])

    return aligned_pos, aligned_rots


def run_model_get_poses(model, imgs, device, is_baseline=False, sample=None):
    """Run model and return 4x4 poses [N, 4, 4]."""
    data = {'imgs': imgs.unsqueeze(0).to(device)}
    if is_baseline and sample is not None and 'projs' in sample:
        projs = sample['projs']
        if isinstance(projs, np.ndarray):
            projs = torch.from_numpy(projs).float()
        data['projs'] = projs.unsqueeze(0).to(device)

    with torch.no_grad():
        if is_baseline:
            output = model(data)
            poses = output['proc_poses'][0].cpu().numpy()
        else:
            output = model.forward_with_calibration_info(data)
            poses = output['pose_result']['poses']
            if poses.dim() == 5:
                poses = poses[:, :, 0]
            poses = poses[0].cpu().numpy()
    return poses


def build_trajectory_with_rotations(dataset, indices, model, device, is_baseline=False):
    """Build global trajectory returning both positions and full 4x4 poses."""
    rel_poses_pred = []
    rel_poses_gt = []
    input_frames = []

    for i, idx in enumerate(indices):
        sample = dataset[idx]
        gt_poses_raw = sample.get('poses')
        if gt_poses_raw is None:
            continue

        if isinstance(gt_poses_raw, torch.Tensor):
            gt_poses = gt_poses_raw.numpy()
        else:
            gt_poses = np.array(gt_poses_raw, dtype=np.float32)
        if gt_poses.ndim == 2:
            gt_poses = gt_poses.reshape(-1, 4, 4)

        imgs = sample['imgs']
        if isinstance(imgs, torch.Tensor):
            imgs_t = imgs
        else:
            imgs_t = torch.from_numpy(imgs).float()

        # Store keyframe image
        if isinstance(imgs, np.ndarray):
            frame = imgs[0]
            if frame.shape[0] in (1, 3):
                frame = np.transpose(frame, (1, 2, 0))
            if frame.max() <= 1.0:
                frame = (frame * 255).astype(np.uint8)
            else:
                frame = frame.astype(np.uint8)
        else:
            frame = imgs[0].permute(1, 2, 0).numpy()
            if frame.max() <= 1.0:
                frame = (frame * 255).astype(np.uint8)
        input_frames.append(frame)

        pred_poses = run_model_get_poses(model, imgs_t, device,
                                          is_baseline=is_baseline, sample=sample)

        if pred_poses.shape[0] >= 3:
            pred_rel = np.linalg.inv(pred_poses[0]) @ pred_poses[2]
            rel_poses_pred.append(pred_rel)

        if gt_poses.shape[0] >= 3:
            gt_rel = np.linalg.inv(gt_poses[0]) @ gt_poses[2]
            rel_poses_gt.append(gt_rel)

        torch.cuda.empty_cache()
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(indices)}] trajectory windows processed")

    n = min(len(rel_poses_pred), len(rel_poses_gt))

    pred_trajectory = [np.eye(4)]
    gt_trajectory = [np.eye(4)]

    for i in range(n):
        pred_trajectory.append(pred_trajectory[-1] @ rel_poses_pred[i])
        gt_trajectory.append(gt_trajectory[-1] @ rel_poses_gt[i])

    return pred_trajectory, gt_trajectory, input_frames


def draw_camera_frustum(ax, position, rotation, color, scale=0.08, alpha=0.3):
    """Draw a small camera frustum in 3D.

    The frustum is a wireframe pyramid showing camera orientation.
    """
    # Camera frustum corners in camera space (looking down -Z)
    hw = scale * 0.6  # half-width
    hh = scale * 0.4  # half-height
    d = scale          # depth

    corners_cam = np.array([
        [0, 0, 0],         # apex (camera center)
        [-hw, -hh, d],     # bottom-left
        [hw, -hh, d],      # bottom-right
        [hw, hh, d],       # top-right
        [-hw, hh, d],      # top-left
    ])

    # Transform to world space
    corners_world = (rotation @ corners_cam.T).T + position

    # Draw edges
    apex = corners_world[0]
    for i in range(1, 5):
        ax.plot3D(*zip(apex, corners_world[i]), color=color, linewidth=0.6, alpha=alpha)

    # Draw rectangle (image plane)
    rect = corners_world[1:5]
    rect_closed = np.vstack([rect, rect[0:1]])
    ax.plot3D(rect_closed[:, 0], rect_closed[:, 1], rect_closed[:, 2],
              color=color, linewidth=0.6, alpha=alpha)


def read_sintel_depth(path):
    """Read Sintel .dpt depth file (float32, TAG_FLOAT = 202021.25)."""
    with open(path, 'rb') as f:
        tag = struct.unpack('f', f.read(4))[0]
        assert abs(tag - 202021.25) < 1.0, f"Invalid .dpt tag: {tag}"
        w = struct.unpack('i', f.read(4))[0]
        h = struct.unpack('i', f.read(4))[0]
        data = np.frombuffer(f.read(w * h * 4), dtype=np.float32)
        return data.reshape(h, w)


def load_sintel_pointcloud(sintel_root, sequence, cam_data_dir, depth_dir,
                            rgb_dir, rigidity_dir=None,
                            every_nth_frame=5, pixel_stride=8,
                            max_depth=50.0):
    """Load a sparse colored point cloud from Sintel GT depth.

    Args:
        sintel_root: Path to Sintel training/ directory
        sequence: e.g. 'market_6'
        every_nth_frame: subsample frames
        pixel_stride: subsample pixels (every Nth pixel in x and y)
        max_depth: clip depth beyond this
    """
    import imageio

    depth_path = Path(depth_dir)
    rgb_path = Path(rgb_dir)
    cam_path = Path(cam_data_dir)
    rig_path = Path(rigidity_dir) if rigidity_dir else None

    depth_files = sorted(depth_path.glob('frame_*.dpt'))
    if not depth_files:
        print(f"  No depth files found at {depth_path}")
        return None, None

    all_points = []
    all_colors = []

    for i, dpt_file in enumerate(depth_files):
        if i % every_nth_frame != 0:
            continue

        frame_num = int(dpt_file.stem.split('_')[1])
        rgb_file = rgb_path / f'frame_{frame_num:04d}.png'
        cam_file = cam_path / f'frame_{frame_num:04d}.cam'

        if not rgb_file.exists() or not cam_file.exists():
            continue

        # Read depth
        depth = read_sintel_depth(str(dpt_file))
        h, w = depth.shape

        # Read RGB
        rgb = imageio.imread(str(rgb_file)).astype(np.float32) / 255.0

        # Read camera params (Sintel .cam format from SDK)
        # Binary: float32 tag, then 9 float64 (3x3 intrinsic M), then 12 float64 (3x4 extrinsic N)
        # x = M * N * X (image = intrinsic * extrinsic * world)
        with open(str(cam_file), 'rb') as cf:
            tag = np.fromfile(cf, dtype=np.float32, count=1)[0]
            M = np.fromfile(cf, dtype=np.float64, count=9).reshape(3, 3)  # intrinsic
            N = np.fromfile(cf, dtype=np.float64, count=12).reshape(3, 4)  # extrinsic [R|t]

        fx = M[0, 0]
        fy = M[1, 1]
        cx = M[0, 2]
        cy = M[1, 2]

        # Build 4x4 extrinsic (world-to-cam)
        extrinsic = np.eye(4)
        extrinsic[:3, :] = N

        # Camera-to-world
        try:
            cam_to_world = np.linalg.inv(extrinsic)
        except np.linalg.LinAlgError:
            continue

        # Rigidity mask (optional — filter dynamic objects)
        if rig_path:
            rig_file = rig_path / f'frame_{frame_num:04d}.png'
            if rig_file.exists():
                rigidity = imageio.imread(str(rig_file))
                if rigidity.ndim == 3:
                    rigidity = rigidity[:, :, 0]
                static_mask = rigidity > 128
            else:
                static_mask = np.ones((h, w), dtype=bool)
        else:
            static_mask = np.ones((h, w), dtype=bool)

        # Unproject with subsampling
        ys = np.arange(0, h, pixel_stride)
        xs = np.arange(0, w, pixel_stride)
        yy, xx = np.meshgrid(ys, xs, indexing='ij')

        d = depth[yy, xx]
        mask = (d > 0) & (d < max_depth) & static_mask[yy, xx]

        u = xx[mask].astype(np.float64)
        v = yy[mask].astype(np.float64)
        z = d[mask].astype(np.float64)

        # Unproject to camera coords
        x_cam = (u - cx) * z / fx
        y_cam = (v - cy) * z / fy
        z_cam = z

        pts_cam = np.stack([x_cam, y_cam, z_cam, np.ones_like(z_cam)], axis=1)
        pts_world = (cam_to_world @ pts_cam.T).T[:, :3]

        colors = rgb[yy[mask], xx[mask]]

        all_points.append(pts_world)
        all_colors.append(colors)

        print(f"  Frame {frame_num}: {len(pts_world)} points")

    if not all_points:
        return None, None

    points = np.concatenate(all_points, axis=0)
    colors = np.concatenate(all_colors, axis=0)
    print(f"  Total point cloud: {len(points)} points")
    return points, colors


def generate_3d_plot(our_poses, anycam_poses, gt_poses, seq_name, output_dir,
                     points=None, colors=None, our_ate=None, anycam_ate=None,
                     frustum_every=5, elev=60, azim=-60):
    """Generate publication-quality 3D trajectory plot."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract positions and rotations
    gt_pos_raw = np.array([T[:3, 3] for T in gt_poses])
    gt_rots = np.array([T[:3, :3] for T in gt_poses])
    our_pos_raw = np.array([T[:3, 3] for T in our_poses])
    our_rots_raw = np.array([T[:3, :3] for T in our_poses])
    anycam_pos_raw = np.array([T[:3, 3] for T in anycam_poses])
    anycam_rots_raw = np.array([T[:3, :3] for T in anycam_poses])

    # Truncate to same length
    n = min(len(gt_pos_raw), len(our_pos_raw), len(anycam_pos_raw))
    gt_pos = gt_pos_raw[:n]
    gt_rots = gt_rots[:n]
    our_rots_raw = our_rots_raw[:n]
    anycam_rots_raw = anycam_rots_raw[:n]

    # Position-only Sim(3) alignment (standard for ATE)
    our_pos = align_trajectory_sim3(our_pos_raw[:n], gt_pos)
    anycam_pos = align_trajectory_sim3(anycam_pos_raw[:n], gt_pos)

    # For frustum orientations, apply the same rotation from Sim(3)
    # Re-derive the alignment rotation for frustum drawing
    def get_sim3_rotation(pred_pos, gt_pos):
        pred_c = pred_pos - pred_pos.mean(axis=0)
        gt_c = gt_pos - gt_pos.mean(axis=0)
        H = pred_c.T @ gt_c
        U, S, Vt = np.linalg.svd(H)
        d = np.linalg.det(Vt.T @ U.T)
        D = np.diag([1, 1, np.sign(d)])
        return Vt.T @ D @ U.T

    R_our = get_sim3_rotation(our_pos_raw[:n], gt_pos)
    R_anycam = get_sim3_rotation(anycam_pos_raw[:n], gt_pos)
    our_rots = np.array([R_our @ r for r in our_rots_raw])
    anycam_rots = np.array([R_anycam @ r for r in anycam_rots_raw])

    our_ate = np.sqrt(np.mean(np.sum((our_pos - gt_pos) ** 2, axis=1)))
    anycam_ate = np.sqrt(np.mean(np.sum((anycam_pos - gt_pos) ** 2, axis=1)))
    print(f"[3D] ATE — Ours: {our_ate:.4f}m, AnyCam: {anycam_ate:.4f}m")

    # Compute frustum scale from trajectory extent
    all_pos = np.vstack([gt_pos, our_pos, anycam_pos])
    extent = max(np.ptp(all_pos[:, 0]), np.ptp(all_pos[:, 1]), np.ptp(all_pos[:, 2]))
    frustum_scale = extent * 0.04  # 4% of trajectory extent

    # Create figure
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Point cloud (if available)
    if points is not None and colors is not None:
        # Subsample for plotting performance
        max_plot_pts = 50000
        if len(points) > max_plot_pts:
            idx = np.random.choice(len(points), max_plot_pts, replace=False)
            points = points[idx]
            colors = colors[idx]
        ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                   c=colors, s=0.1, alpha=0.15, rasterized=True)

    # Trajectory lines
    our_label = f'Ours (MCT) — ATE={our_ate:.3f}m'
    anycam_label = f'AnyCam — ATE={anycam_ate:.3f}m'

    ax.plot3D(gt_pos[:, 0], gt_pos[:, 1], gt_pos[:, 2],
              '-', color=C_GT, linewidth=2.5, label='Ground Truth', zorder=10)
    ax.plot3D(our_pos[:, 0], our_pos[:, 1], our_pos[:, 2],
              '-', color=C_OURS, linewidth=1.8, label=our_label, zorder=9)
    ax.plot3D(anycam_pos[:, 0], anycam_pos[:, 1], anycam_pos[:, 2],
              '-', color=C_ANYCAM, linewidth=1.8, label=anycam_label, zorder=8)

    # Camera frustums at intervals
    for i in range(0, n, frustum_every):
        draw_camera_frustum(ax, gt_pos[i], gt_rots[i], C_GT,
                            scale=frustum_scale, alpha=0.5)
        draw_camera_frustum(ax, our_pos[i], our_rots[i], C_OURS,
                            scale=frustum_scale, alpha=0.4)
        draw_camera_frustum(ax, anycam_pos[i], anycam_rots[i], C_ANYCAM,
                            scale=frustum_scale, alpha=0.4)

    # Start/end markers
    ax.scatter(*gt_pos[0], color=C_GT, s=60, marker='o', zorder=11)
    ax.scatter(*gt_pos[-1], color=C_GT, s=60, marker='s', zorder=11)

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(f'3D Camera Trajectory — {seq_name}')
    ax.legend(loc='upper left', framealpha=0.9)
    ax.view_init(elev=elev, azim=azim)

    # Equal aspect ratio — tight around trajectory data
    max_range = extent * 0.55
    mid = all_pos.mean(axis=0)
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)

    # Reduce whitespace
    ax.dist = 8  # zoom in (default is ~10)

    fig.tight_layout()

    # Save from multiple angles
    out1 = output_dir / 'trajectory_3d_birdseye.png'
    fig.savefig(out1)
    print(f"[FIGURE] Saved {out1}")

    # Side view
    ax.view_init(elev=20, azim=-60)
    out2 = output_dir / 'trajectory_3d_side.png'
    fig.savefig(out2)
    print(f"[FIGURE] Saved {out2}")

    # Front view
    ax.view_init(elev=10, azim=0)
    out3 = output_dir / 'trajectory_3d_front.png'
    fig.savefig(out3)
    print(f"[FIGURE] Saved {out3}")

    plt.close(fig)

    return our_pos, anycam_pos, gt_pos, our_rots, anycam_rots, gt_rots, \
           our_ate, anycam_ate, frustum_scale, all_pos, extent


def generate_3d_video(our_pos, anycam_pos, gt_pos, our_rots, anycam_rots, gt_rots,
                      seq_name, output_dir, points=None, colors=None,
                      our_ate=None, anycam_ate=None,
                      frustum_scale=0.08, extent=1.0, all_pos=None,
                      elev=60, azim=-60, fps=8):
    """Generate a video with camera frustums building up frame by frame in 3D."""
    import subprocess
    import shutil

    output_dir = Path(output_dir)
    frames_dir = output_dir / '_3d_video_frames'
    frames_dir.mkdir(parents=True, exist_ok=True)

    n = len(gt_pos)

    # Axis limits
    if all_pos is None:
        all_pos = np.vstack([gt_pos, our_pos, anycam_pos])
    max_range = max(np.ptp(all_pos[:, 0]), np.ptp(all_pos[:, 1]), np.ptp(all_pos[:, 2])) * 0.55
    mid = all_pos.mean(axis=0)

    # Subsample point cloud for video frames (speed)
    pts_plot, col_plot = None, None
    if points is not None and colors is not None:
        max_pts = 30000
        if len(points) > max_pts:
            idx = np.random.choice(len(points), max_pts, replace=False)
            pts_plot = points[idx]
            col_plot = colors[idx]
        else:
            pts_plot, col_plot = points, colors

    our_label = f'Ours (MCT) — ATE={our_ate:.3f}m' if our_ate else 'Ours (MCT)'
    anycam_label = f'AnyCam — ATE={anycam_ate:.3f}m' if anycam_ate else 'AnyCam'

    print(f"[VIDEO] Rendering {n} 3D frames...")

    for i in range(n):
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Point cloud background
        if pts_plot is not None:
            ax.scatter(pts_plot[:, 0], pts_plot[:, 1], pts_plot[:, 2],
                       c=col_plot, s=0.1, alpha=0.12, rasterized=True)

        # Full trajectories as faint reference
        ax.plot3D(gt_pos[:, 0], gt_pos[:, 1], gt_pos[:, 2],
                  '-', color=C_GT, linewidth=1.0, alpha=0.2)
        ax.plot3D(our_pos[:, 0], our_pos[:, 1], our_pos[:, 2],
                  '-', color=C_OURS, linewidth=0.8, alpha=0.15)
        ax.plot3D(anycam_pos[:, 0], anycam_pos[:, 1], anycam_pos[:, 2],
                  '-', color=C_ANYCAM, linewidth=0.8, alpha=0.15)

        # Progressive trajectories
        ax.plot3D(gt_pos[:i+1, 0], gt_pos[:i+1, 1], gt_pos[:i+1, 2],
                  '-', color=C_GT, linewidth=2.5, label='Ground Truth', zorder=10)
        ax.plot3D(our_pos[:i+1, 0], our_pos[:i+1, 1], our_pos[:i+1, 2],
                  '-', color=C_OURS, linewidth=1.8, label=our_label, zorder=9)
        ax.plot3D(anycam_pos[:i+1, 0], anycam_pos[:i+1, 1], anycam_pos[:i+1, 2],
                  '-', color=C_ANYCAM, linewidth=1.8, label=anycam_label, zorder=8)

        # Current camera frustums (latest position)
        draw_camera_frustum(ax, gt_pos[i], gt_rots[i], C_GT,
                            scale=frustum_scale, alpha=0.7)
        draw_camera_frustum(ax, our_pos[i], our_rots[i], C_OURS,
                            scale=frustum_scale, alpha=0.6)
        draw_camera_frustum(ax, anycam_pos[i], anycam_rots[i], C_ANYCAM,
                            scale=frustum_scale, alpha=0.6)

        # Trail frustums (every 8th previous frame, faded)
        for j in range(0, i, 8):
            draw_camera_frustum(ax, gt_pos[j], gt_rots[j], C_GT,
                                scale=frustum_scale * 0.6, alpha=0.15)
            draw_camera_frustum(ax, our_pos[j], our_rots[j], C_OURS,
                                scale=frustum_scale * 0.6, alpha=0.12)
            draw_camera_frustum(ax, anycam_pos[j], anycam_rots[j], C_ANYCAM,
                                scale=frustum_scale * 0.6, alpha=0.12)

        # Start marker
        ax.scatter(*gt_pos[0], color=C_GT, s=50, marker='o', zorder=11)

        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title(f'3D Camera Trajectory — {seq_name} (frame {i})')
        ax.legend(loc='upper left', framealpha=0.9, fontsize=9)
        ax.view_init(elev=elev, azim=azim)

        ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
        ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
        ax.set_zlim(mid[2] - max_range, mid[2] + max_range)
        ax.dist = 8

        fig.tight_layout()
        fig.savefig(frames_dir / f'frame_{i:04d}.png', dpi=150)
        plt.close(fig)

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{n}] 3D video frames rendered")

    # Encode with ffmpeg
    video_path = output_dir / 'trajectory_3d_video.mp4'
    cmd = [
        'ffmpeg', '-y', '-framerate', str(fps),
        '-i', str(frames_dir / 'frame_%04d.png'),
        '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2',
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
        '-crf', '18', '-preset', 'medium',
        str(video_path)
    ]
    print(f"[VIDEO] Encoding with ffmpeg...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[WARN] ffmpeg failed: {result.stderr}")
    else:
        print(f"[VIDEO] Saved {video_path}")
        shutil.rmtree(frames_dir)


def main():
    parser = argparse.ArgumentParser(description='Generate 3D trajectory visualization')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--anycam_config', type=str,
                        default='pretrained_models/anycam_seq8/training_config.yaml')
    parser.add_argument('--pretrained_anycam', type=str,
                        default='pretrained_models/anycam_seq8/training_checkpoint_247500.pt')
    parser.add_argument('--data_root', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default='thesis_results/figures/viz_3d')
    parser.add_argument('--dataset', type=str, default='sintel',
                        choices=['sintel', 'tumrgbd'])
    parser.add_argument('--sequence', type=str, default='market_6')
    parser.add_argument('--max_windows', type=int, default=None)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--image_size', type=int, default=336)
    parser.add_argument('--no_pointcloud', action='store_true',
                        help='Skip point cloud even for Sintel')
    parser.add_argument('--no_video', action='store_true',
                        help='Skip video generation (static plots only)')
    parser.add_argument('--elev', type=float, default=60,
                        help='Elevation angle for bird\'s eye view')
    parser.add_argument('--azim', type=float, default=-60,
                        help='Azimuth angle for bird\'s eye view')
    parser.add_argument('--frustum_every', type=int, default=5,
                        help='Draw camera frustum every N frames')
    parser.add_argument('--video_fps', type=int, default=8,
                        help='Video frame rate')
    parser.add_argument('--save_trajectories', type=str, default=None,
                        help='Save trajectory data to .npz for reuse')
    parser.add_argument('--load_trajectories', type=str, default=None,
                        help='Load trajectory data from .npz (skip inference)')
    args = parser.parse_args()

    device = args.device

    # Auto-detect data root
    data_root = args.data_root
    if data_root is None:
        for candidate in ['/storage/user/maka/eval_datasets', '/data/thesis',
                          '/home/kalman/TUM/thesis']:
            if Path(candidate).exists():
                data_root = candidate
                break
    if data_root is None:
        print("ERROR: Could not auto-detect data_root.")
        sys.exit(1)

    # Load dataset
    if args.dataset == 'tumrgbd':
        print(f"[LOAD] Loading TUM-RGBD dataset (dilation=10)...")
        dataset = load_tumrgbd_dataset(data_root, num_samples=9999,
                                        image_size=args.image_size, frame_count=4, dilation=10)
    else:
        print(f"[LOAD] Loading Sintel dataset (dilation=1)...")
        dataset = load_sintel_dataset(data_root, num_samples=9999,
                                       image_size=args.image_size, frame_count=4, dilation=1)

    # Group by sequence
    from collections import defaultdict
    groups = defaultdict(list)
    for i in range(len(dataset)):
        seq_name, _ = dataset._datapoints[i]
        groups[seq_name].append(i)

    seq_name = args.sequence
    if seq_name not in groups:
        print(f"ERROR: Sequence '{seq_name}' not found. Available: {sorted(groups.keys())}")
        sys.exit(1)

    indices = groups[seq_name]
    print(f"[LOAD] Sequence: {seq_name} ({len(indices)} windows)")

    if args.max_windows and len(indices) > args.max_windows:
        step = max(1, len(indices) // args.max_windows)
        indices = indices[::step][:args.max_windows]
        print(f"[LOAD] Subsampled to {len(indices)} windows")

    if args.load_trajectories:
        # Load pre-computed trajectories (deterministic reuse)
        print(f"\n[LOAD] Loading trajectories from {args.load_trajectories}")
        data = np.load(args.load_trajectories, allow_pickle=True)
        our_poses = [data['our_poses'][i] for i in range(len(data['our_poses']))]
        anycam_poses = [data['anycam_poses'][i] for i in range(len(data['anycam_poses']))]
        gt_poses = [data['gt_poses'][i] for i in range(len(data['gt_poses']))]
        print(f"[LOAD] Lengths: ours={len(our_poses)}, anycam={len(anycam_poses)}, GT={len(gt_poses)}")
    else:
        # Create models and run inference
        print("\n[LOAD] Creating our FAT model...")
        our_model = create_inference_model(args.anycam_config, device)
        load_phase_c_checkpoint(our_model, args.checkpoint, device)
        our_model = our_model.to(device)
        our_model.eval()

        print("[LOAD] Creating vanilla AnyCam baseline...")
        baseline_model = create_baseline_model(args.anycam_config, args.pretrained_anycam, device)

        # Build trajectories (full 4x4 poses)
        print("\n[TRAJ] Building our trajectory...")
        our_poses, gt_poses, frames = build_trajectory_with_rotations(
            dataset, indices, our_model, device, is_baseline=False)

        print("[TRAJ] Building AnyCam trajectory...")
        anycam_poses, _, _ = build_trajectory_with_rotations(
            dataset, indices, baseline_model, device, is_baseline=True)

        print(f"[TRAJ] Lengths: ours={len(our_poses)}, anycam={len(anycam_poses)}, GT={len(gt_poses)}")

        # Save trajectories for deterministic reuse
        if args.save_trajectories:
            save_path = args.save_trajectories
            np.savez(save_path,
                     our_poses=np.array(our_poses),
                     anycam_poses=np.array(anycam_poses),
                     gt_poses=np.array(gt_poses))
            print(f"[SAVE] Trajectories saved to {save_path}")

    # Load point cloud (Sintel only, unless --no_pointcloud)
    points, colors = None, None
    if not args.no_pointcloud and args.dataset == 'sintel':
        print("\n[PCLOUD] Loading Sintel GT depth for point cloud...")
        sintel_base = Path(data_root) / 'Sintel' / 'training'
        # Try common Sintel locations
        for base in [sintel_base, Path(data_root) / 'sintel' / 'training',
                     Path('/storage/group/dataset_mirrors/01_incoming/Sintel/training')]:
            depth_dir = base / 'depth' / seq_name
            if depth_dir.exists():
                points, colors = load_sintel_pointcloud(
                    sintel_root=str(base),
                    sequence=seq_name,
                    cam_data_dir=str(base / 'camdata_left' / seq_name),
                    depth_dir=str(depth_dir),
                    rgb_dir=str(base / 'clean' / seq_name),
                    rigidity_dir=str(base / 'rigidity' / seq_name),
                    every_nth_frame=3,
                    pixel_stride=6,
                    max_depth=40.0,
                )
                break
        if points is None:
            print("  [WARN] Could not load point cloud, proceeding without it.")

    # Generate 3D plot
    print("\n[FIGURE] Generating 3D trajectory plot...")
    result = generate_3d_plot(our_poses, anycam_poses, gt_poses, seq_name,
                               args.output_dir, points=points, colors=colors,
                               frustum_every=args.frustum_every,
                               elev=args.elev, azim=args.azim)
    our_pos, anycam_pos, gt_pos, our_rots, anycam_rots, gt_rots, \
        our_ate, anycam_ate, frustum_scale, all_pos, extent = result

    # Generate video
    if not args.no_video:
        print("\n[VIDEO] Generating 3D trajectory video...")
        generate_3d_video(our_pos, anycam_pos, gt_pos,
                          our_rots, anycam_rots, gt_rots,
                          seq_name, args.output_dir,
                          points=points, colors=colors,
                          our_ate=our_ate, anycam_ate=anycam_ate,
                          frustum_scale=frustum_scale, extent=extent,
                          all_pos=all_pos, elev=args.elev, azim=args.azim,
                          fps=args.video_fps if hasattr(args, 'video_fps') else 8)

    print("\n[DONE] All 3D visualizations complete.")


if __name__ == '__main__':
    main()
