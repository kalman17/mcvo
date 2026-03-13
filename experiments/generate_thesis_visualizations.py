"""
Generate thesis trajectory visualizations for a cherry-picked sequence.

Produces:
  1. Bird's-eye trajectory plot (publication quality PNG)
  2. Side-by-side video: input frames + trajectory building up over time

Uses alley_1 from Sintel (proven best sequence from histogram analysis).

Usage:
    python experiments/generate_thesis_visualizations.py \
        --checkpoint /path/to/phase_C_v3_h100_epoch_0005.pt \
        --output_dir thesis_results/figures \
        --device cuda:0
"""

import argparse
import sys
import subprocess
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams

# Publication-quality settings
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 11
rcParams['axes.labelsize'] = 12
rcParams['axes.titlesize'] = 13
rcParams['xtick.labelsize'] = 10
rcParams['ytick.labelsize'] = 10
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
from experiments.pose_metrics import rotation_error_degrees

C_OURS = '#2171b5'
C_ANYCAM = '#e6550d'
C_GT = '#333333'


def align_trajectory_sim3(pred_pos, gt_pos):
    """Sim(3) alignment (Umeyama) — standard for monocular VO evaluation.

    Aligns predicted trajectory to GT using scale + rotation + translation.
    """
    pred = pred_pos.copy()
    gt = gt_pos.copy()

    # Center
    pred_mean = pred.mean(axis=0)
    gt_mean = gt.mean(axis=0)
    pred_c = pred - pred_mean
    gt_c = gt - gt_mean

    # Scale
    pred_var = np.sum(pred_c ** 2)
    if pred_var < 1e-12:
        return pred  # degenerate
    scale = np.sqrt(np.sum(gt_c ** 2) / pred_var)

    # Rotation (Procrustes)
    H = pred_c.T @ gt_c
    U, S, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    D = np.diag([1, 1, np.sign(d)])
    R = Vt.T @ D @ U.T

    aligned = scale * (R @ pred.T).T + gt_mean - scale * R @ pred_mean
    return aligned


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


def build_trajectory(dataset, indices, model, device, is_baseline=False):
    """
    Build a global camera trajectory by chaining per-window relative poses.

    Each window covers frames [keyframe, keyframe-1, keyframe+1, keyframe+2].
    We extract the relative pose keyframe -> keyframe+1 (indices 0 -> 2 in the
    window's frame ordering) and chain them.

    Returns:
        positions: [N+1, 3] array of camera positions in global coords
        gt_positions: [N+1, 3] array of GT camera positions
        frames: list of input images (first frame from each window + last window's frames)
    """
    # First pass: collect all relative poses and GT
    rel_poses_pred = []  # predicted relative pose frame k -> k+1
    rel_poses_gt = []
    input_frames = []
    gt_absolute = []

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

        # Store the keyframe image (index 0 = keyframe)
        if isinstance(imgs, np.ndarray):
            # imgs shape: [4, C, H, W] or [4, H, W, C]
            frame = imgs[0]
            if frame.shape[0] in (1, 3):  # CHW -> HWC
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

        # Run model
        pred_poses = run_model_get_poses(model, imgs_t, device,
                                          is_baseline=is_baseline, sample=sample)

        # Relative pose: keyframe (index 0) -> keyframe+1 (index 2)
        # In the window frame ordering: [keyframe, keyframe-1, keyframe+1, keyframe+2]
        if pred_poses.shape[0] >= 3:
            pred_rel = np.linalg.inv(pred_poses[0]) @ pred_poses[2]
            rel_poses_pred.append(pred_rel)

        if gt_poses.shape[0] >= 3:
            gt_rel = np.linalg.inv(gt_poses[0]) @ gt_poses[2]
            rel_poses_gt.append(gt_rel)

        # Store first GT absolute pose
        if i == 0:
            gt_absolute.append(gt_poses[0])

        torch.cuda.empty_cache()

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(indices)}] trajectory windows processed")

    # Chain relative poses into global trajectory
    n = min(len(rel_poses_pred), len(rel_poses_gt))

    pred_trajectory = [np.eye(4)]  # Start at identity
    gt_trajectory = [np.eye(4)]

    for i in range(n):
        pred_trajectory.append(pred_trajectory[-1] @ rel_poses_pred[i])
        gt_trajectory.append(gt_trajectory[-1] @ rel_poses_gt[i])

    pred_positions = np.array([T[:3, 3] for T in pred_trajectory])
    gt_positions = np.array([T[:3, 3] for T in gt_trajectory])

    return pred_positions, gt_positions, input_frames


def generate_trajectory_plot(our_pos, anycam_pos, gt_pos, seq_name, output_dir,
                              our_ate=None, anycam_ate=None):
    """Generate publication-quality bird's-eye trajectory comparison."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 5))

    # Labels with ATE if available
    our_label = 'Ours (MCT)'
    anycam_label = 'AnyCam'
    if our_ate is not None:
        our_label += f' (ATE={our_ate:.4f}m)'
    if anycam_ate is not None:
        anycam_label += f' (ATE={anycam_ate:.4f}m)'

    # Plot trajectories (X-Z plane = bird's eye view)
    ax.plot(gt_pos[:, 0], gt_pos[:, 2], '-', color=C_GT, linewidth=2.0,
            label='Ground Truth', zorder=3)
    ax.plot(our_pos[:, 0], our_pos[:, 2], '-', color=C_OURS, linewidth=1.5,
            label=our_label, zorder=2, alpha=0.9)
    ax.plot(anycam_pos[:, 0], anycam_pos[:, 2], '-', color=C_ANYCAM, linewidth=1.5,
            label=anycam_label, zorder=1, alpha=0.8)

    # Mark start/end
    ax.plot(gt_pos[0, 0], gt_pos[0, 2], 'o', color=C_GT, markersize=8, zorder=4)
    ax.plot(gt_pos[-1, 0], gt_pos[-1, 2], 's', color=C_GT, markersize=8, zorder=4)

    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Z (meters)')
    ax.set_title(f'Camera Trajectory — {seq_name}')
    ax.legend(loc='upper left', framealpha=0.9, bbox_to_anchor=(0.02, 0.98))
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path = output_dir / 'trajectory_comparison.png'
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[FIGURE] Saved {out_path}")


def generate_trajectory_video(our_pos, anycam_pos, gt_pos, input_frames,
                               seq_name, output_dir, fps=10):
    """Generate side-by-side video: input frame + trajectory building up."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frames_dir = output_dir / '_video_frames'
    frames_dir.mkdir(parents=True, exist_ok=True)

    n_frames = min(len(input_frames), len(our_pos))
    print(f"[VIDEO] Rendering {n_frames} frames...")

    # Compute axis limits from all trajectories
    all_x = np.concatenate([gt_pos[:, 0], our_pos[:, 0], anycam_pos[:, 0]])
    all_z = np.concatenate([gt_pos[:, 2], our_pos[:, 2], anycam_pos[:, 2]])
    margin = max(np.ptp(all_x), np.ptp(all_z)) * 0.15
    x_min, x_max = all_x.min() - margin, all_x.max() + margin
    z_min, z_max = all_z.min() - margin, all_z.max() + margin

    # Make square aspect ratio
    x_range = x_max - x_min
    z_range = z_max - z_min
    if x_range > z_range:
        diff = (x_range - z_range) / 2
        z_min -= diff
        z_max += diff
    else:
        diff = (z_range - x_range) / 2
        x_min -= diff
        x_max += diff

    for i in range(n_frames):
        fig, (ax_img, ax_traj) = plt.subplots(1, 2, figsize=(12, 4.5),
                                                gridspec_kw={'width_ratios': [1, 1.2]})

        # Left: input frame
        ax_img.imshow(input_frames[i])
        ax_img.set_title(f'Frame {i}', fontsize=11)
        ax_img.axis('off')

        # Right: trajectory up to current frame
        # Full GT as faint reference
        ax_traj.plot(gt_pos[:, 0], gt_pos[:, 2], '-', color=C_GT, linewidth=1.0,
                     alpha=0.3, zorder=1)

        # Progressive trajectories
        ax_traj.plot(gt_pos[:i+1, 0], gt_pos[:i+1, 2], '-', color=C_GT,
                     linewidth=2.0, label='Ground Truth', zorder=3)
        ax_traj.plot(our_pos[:i+1, 0], our_pos[:i+1, 2], '-', color=C_OURS,
                     linewidth=1.8, label='Ours (MCT)', zorder=2)
        ax_traj.plot(anycam_pos[:i+1, 0], anycam_pos[:i+1, 2], '-', color=C_ANYCAM,
                     linewidth=1.8, label='AnyCam', zorder=2)

        # Current position markers
        ax_traj.plot(gt_pos[i, 0], gt_pos[i, 2], 'o', color=C_GT, markersize=6, zorder=4)
        ax_traj.plot(our_pos[i, 0], our_pos[i, 2], 'o', color=C_OURS, markersize=6, zorder=4)
        ax_traj.plot(anycam_pos[i, 0], anycam_pos[i, 2], 'o', color=C_ANYCAM, markersize=6, zorder=4)

        # Start marker
        ax_traj.plot(gt_pos[0, 0], gt_pos[0, 2], '*', color=C_GT, markersize=10, zorder=5)

        ax_traj.set_xlim(x_min, x_max)
        ax_traj.set_ylim(z_min, z_max)
        ax_traj.set_xlabel('X')
        ax_traj.set_ylabel('Z')
        ax_traj.set_title(f'Camera Trajectory — {seq_name}', fontsize=11)
        ax_traj.set_aspect('equal')
        ax_traj.grid(True, alpha=0.3)
        if i == 0:
            ax_traj.legend(loc='upper right', framealpha=0.9, fontsize=9)

        fig.tight_layout()
        fig.savefig(frames_dir / f'frame_{i:04d}.png', dpi=150)
        plt.close(fig)

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{n_frames}] video frames rendered")

    # Stitch with ffmpeg
    video_path = output_dir / 'trajectory_video.mp4'
    cmd = [
        'ffmpeg', '-y', '-framerate', str(fps),
        '-i', str(frames_dir / 'frame_%04d.png'),
        '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2',  # ensure even dimensions for libx264
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
        '-crf', '18', '-preset', 'medium',
        str(video_path)
    ]
    print(f"[VIDEO] Encoding with ffmpeg...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[WARN] ffmpeg failed: {result.stderr}")
        print(f"  Frames saved at {frames_dir}/")
    else:
        print(f"[VIDEO] Saved {video_path}")
        # Clean up frame images
        import shutil
        shutil.rmtree(frames_dir)

    return video_path


def main():
    parser = argparse.ArgumentParser(description='Generate thesis trajectory visualizations')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--anycam_config', type=str,
                        default='pretrained_models/anycam_seq8/training_config.yaml')
    parser.add_argument('--pretrained_anycam', type=str,
                        default='pretrained_models/anycam_seq8/training_checkpoint_247500.pt')
    parser.add_argument('--data_root', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default='thesis_results/figures')
    parser.add_argument('--dataset', type=str, default='sintel',
                        choices=['sintel', 'tumrgbd'],
                        help='Dataset to use')
    parser.add_argument('--sequence', type=str, default=None,
                        help='Sequence name (auto-picks best if not specified)')
    parser.add_argument('--max_windows', type=int, default=None,
                        help='Max windows to use (None = all)')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--image_size', type=int, default=336)
    parser.add_argument('--video_fps', type=int, default=8)
    parser.add_argument('--no_ate_in_legend', action='store_true',
                        help='Omit ATE numbers from trajectory plot legend')
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

    for name, idxs in sorted(groups.items(), key=lambda x: -len(x[1])):
        print(f"  {name}: {len(idxs)} windows")

    if args.sequence:
        seq_name = args.sequence
        if seq_name not in groups:
            print(f"ERROR: Sequence '{seq_name}' not found. Available: {list(groups.keys())}")
            sys.exit(1)
    else:
        # Auto-pick: prefer sequences with lots of GT motion where we beat AnyCam
        seq_name = None  # will be set after cherry-picking below

    if seq_name:
        indices = groups[seq_name]
    else:
        indices = None  # defer until after model loading

    if indices is not None:
        print(f"[LOAD] Sequence: {seq_name} ({len(indices)} windows)")

    # Create models
    print("\n[LOAD] Creating our FAT model...")
    our_model = create_inference_model(args.anycam_config, device)
    load_phase_c_checkpoint(our_model, args.checkpoint, device)
    our_model = our_model.to(device)
    our_model.eval()

    print("[LOAD] Creating vanilla AnyCam baseline...")
    baseline_model = create_baseline_model(args.anycam_config, args.pretrained_anycam, device)

    # Cherry-pick if no sequence specified — find where AnyCam has worst ATE
    if indices is None:
        print("\n[PICK] Scanning ALL sequences for ATE comparison...")
        print("  Strategy: build full trajectory per sequence, Sim(3) align, compare ATE")
        results = []

        for sname, sidxs in sorted(groups.items()):
            if len(sidxs) < 5:
                print(f"  SKIP {sname}: too few windows ({len(sidxs)})")
                continue

            try:
                # Build full trajectories for both models
                our_pos_c, gt_pos_c, _ = build_trajectory(
                    dataset, sidxs, our_model, device, is_baseline=False)
                anycam_pos_c, _, _ = build_trajectory(
                    dataset, sidxs, baseline_model, device, is_baseline=True)

                n_c = min(len(our_pos_c), len(anycam_pos_c), len(gt_pos_c))
                if n_c < 5:
                    print(f"  SKIP {sname}: trajectory too short ({n_c})")
                    continue

                our_pos_c = our_pos_c[:n_c]
                anycam_pos_c = anycam_pos_c[:n_c]
                gt_pos_c = gt_pos_c[:n_c]

                # Sim(3) align
                our_aligned = align_trajectory_sim3(our_pos_c, gt_pos_c)
                anycam_aligned = align_trajectory_sim3(anycam_pos_c, gt_pos_c)

                our_ate_c = np.sqrt(np.mean(np.sum((our_aligned - gt_pos_c) ** 2, axis=1)))
                anycam_ate_c = np.sqrt(np.mean(np.sum((anycam_aligned - gt_pos_c) ** 2, axis=1)))

                # Score: we want sequences where we beat AnyCam on ATE
                # Higher = better for us. Positive means we win.
                ate_ratio = anycam_ate_c / max(our_ate_c, 1e-8)  # >1 means we win
                we_win = our_ate_c < anycam_ate_c

                marker = "***" if we_win else "   "
                print(f"  {marker} {sname}: ATE ours={our_ate_c:.4f}m "
                      f"anycam={anycam_ate_c:.4f}m "
                      f"ratio={ate_ratio:.2f}x "
                      f"({'WIN' if we_win else 'lose'})")

                results.append((sname, our_ate_c, anycam_ate_c, ate_ratio, we_win))

                torch.cuda.empty_cache()
            except Exception as e:
                print(f"  ERROR {sname}: {e}")
                continue

        # Summary
        print(f"\n[PICK] Summary: {len(results)} sequences evaluated")
        wins = [(s, o, a, r, w) for s, o, a, r, w in results if w]
        losses = [(s, o, a, r, w) for s, o, a, r, w in results if not w]
        print(f"  We WIN on ATE: {len(wins)} sequences")
        print(f"  We LOSE on ATE: {len(losses)} sequences")

        if wins:
            # Pick the win with best ATE ratio (biggest margin)
            wins.sort(key=lambda x: -x[3])
            print(f"\n  Top wins by ATE ratio:")
            for s, o, a, r, _ in wins[:5]:
                print(f"    {s}: ours={o:.4f}m anycam={a:.4f}m ratio={r:.2f}x")
            best_seq = wins[0][0]
        else:
            # No wins — pick where AnyCam does worst (highest ATE)
            results.sort(key=lambda x: -x[2])
            print(f"\n  No ATE wins. Sequences where AnyCam is worst:")
            for s, o, a, r, _ in results[:5]:
                print(f"    {s}: ours={o:.4f}m anycam={a:.4f}m")
            best_seq = results[0][0]

        seq_name = best_seq
        indices = groups[seq_name]
        print(f"\n[PICK] Selected: {seq_name} ({len(indices)} windows)")

    # Optionally limit windows
    if args.max_windows and len(indices) > args.max_windows:
        step = max(1, len(indices) // args.max_windows)
        indices = indices[::step][:args.max_windows]
        print(f"[LOAD] Subsampled to {len(indices)} windows")

    # Build trajectories
    print("\n[TRAJ] Building our trajectory...")
    our_pos, gt_pos, input_frames = build_trajectory(
        dataset, indices, our_model, device, is_baseline=False)

    print("[TRAJ] Building AnyCam trajectory...")
    anycam_pos, gt_pos2, _ = build_trajectory(
        dataset, indices, baseline_model, device, is_baseline=True)

    print(f"[TRAJ] Trajectory lengths: ours={len(our_pos)}, "
          f"anycam={len(anycam_pos)}, GT={len(gt_pos)}")

    # Truncate to same length
    n = min(len(our_pos), len(anycam_pos), len(gt_pos))
    our_pos = our_pos[:n]
    anycam_pos = anycam_pos[:n]
    gt_pos = gt_pos[:n]
    input_frames = input_frames[:n]

    # Sim(3) alignment — standard for monocular VO (predictions are up to scale)
    print("\n[ALIGN] Sim(3) aligning predicted trajectories to GT...")
    our_pos = align_trajectory_sim3(our_pos, gt_pos)
    anycam_pos = align_trajectory_sim3(anycam_pos, gt_pos)

    # Compute ATE after alignment
    our_ate = np.sqrt(np.mean(np.sum((our_pos - gt_pos) ** 2, axis=1)))
    anycam_ate = np.sqrt(np.mean(np.sum((anycam_pos - gt_pos) ** 2, axis=1)))
    print(f"[ALIGN] ATE — Ours: {our_ate:.4f}m, AnyCam: {anycam_ate:.4f}m")

    # Generate static trajectory plot
    print("\n[FIGURE] Generating trajectory plot...")
    ate_ours = None if args.no_ate_in_legend else our_ate
    ate_anycam = None if args.no_ate_in_legend else anycam_ate
    generate_trajectory_plot(our_pos, anycam_pos, gt_pos, seq_name, args.output_dir,
                              our_ate=ate_ours, anycam_ate=ate_anycam)

    # Generate video
    print("[VIDEO] Generating trajectory video...")
    generate_trajectory_video(our_pos, anycam_pos, gt_pos, input_frames,
                               seq_name, args.output_dir, fps=args.video_fps)

    print("\n[DONE] All visualizations complete.")


if __name__ == '__main__':
    main()
