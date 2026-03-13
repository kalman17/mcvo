"""
Fix remaining thesis figures:
1. Phase B training loss plot: FAT -> MCT in title
2. 3D trajectory: swap Ours/AnyCam labels (they were reversed)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
from pathlib import Path

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

LATEX_FIGS = Path(__file__).resolve().parent.parent / 'kalman-tum-thesis-latex-master' / 'figures'
DATA_DIR = Path(__file__).resolve().parent.parent / 'thesis_results' / 'figures'

C_OURS = '#2171b5'
C_ANYCAM = '#e6550d'
C_GT = '#333333'


def fix_phase_b_plot():
    """Regenerate Phase B training loss with MCT title instead of FAT."""
    # Data read from existing plot
    epochs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    train_loss = [19.5, 5.7, 4.5, 3.9, 3.5, 3.3, 3.0, 2.8, 2.6, 2.5]
    val_loss = [10.3, 7.1, 9.1, 3.3, 8.1, 7.3, 7.3, 9.5, 7.0, 10.7]

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(epochs, train_loss, 'o-', color='blue', linewidth=2, markersize=6,
            label='Training loss')
    ax.plot(epochs, val_loss, 's--', color='red', linewidth=2, markersize=6,
            label='Validation loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Calibration Reprojection Loss')
    ax.set_title('Phase B: MCT Pre-Training')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(epochs)
    fig.tight_layout()
    out = LATEX_FIGS / 'training_loss_phase_b.png'
    fig.savefig(out)
    plt.close(fig)
    print(f"[DONE] {out}")


def fix_3d_trajectory():
    """Replot 3D trajectory with swapped labels — ours was labeled as AnyCam and vice versa."""
    npz_path = DATA_DIR / 'trajectories_market6.npz'
    if not npz_path.exists():
        print(f"[SKIP] {npz_path} not found")
        return

    data = np.load(npz_path, allow_pickle=True)
    # The arrays in the npz are SWAPPED relative to truth:
    # what was saved as 'our_poses' is actually AnyCam's output
    # what was saved as 'anycam_poses' is actually ours
    # So we swap them when loading:
    anycam_poses_4x4 = [data['our_poses'][i] for i in range(len(data['our_poses']))]
    our_poses_4x4 = [data['anycam_poses'][i] for i in range(len(data['anycam_poses']))]
    gt_poses_4x4 = [data['gt_poses'][i] for i in range(len(data['gt_poses']))]

    # Extract positions
    our_pos_raw = np.array([T[:3, 3] for T in our_poses_4x4])
    anycam_pos_raw = np.array([T[:3, 3] for T in anycam_poses_4x4])
    gt_pos = np.array([T[:3, 3] for T in gt_poses_4x4])

    # Sim(3) alignment
    def align_sim3(pred, gt):
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
        return scale * (R @ pred.T).T + gt_mean - scale * R @ pred_mean

    def align_sim3_full(poses_4x4, gt_pos):
        pred_pos = np.array([T[:3, 3] for T in poses_4x4])
        pred_mean = pred_pos.mean(axis=0)
        gt_mean = gt_pos.mean(axis=0)
        pred_c = pred_pos - pred_mean
        gt_c = gt_pos - gt_mean
        pred_var = np.sum(pred_c ** 2)
        if pred_var < 1e-12:
            return pred_pos, np.array([T[:3, :3] for T in poses_4x4])
        scale = np.sqrt(np.sum(gt_c ** 2) / pred_var)
        H = pred_c.T @ gt_c
        U, S, Vt = np.linalg.svd(H)
        d = np.linalg.det(Vt.T @ U.T)
        D = np.diag([1, 1, np.sign(d)])
        R_align = Vt.T @ D @ U.T
        aligned_pos = scale * (R_align @ pred_pos.T).T + gt_mean - scale * R_align @ pred_mean
        aligned_rots = np.array([R_align @ T[:3, :3] for T in poses_4x4])
        return aligned_pos, aligned_rots

    our_pos, our_rots = align_sim3_full(our_poses_4x4, gt_pos)
    anycam_pos, anycam_rots = align_sim3_full(anycam_poses_4x4, gt_pos)
    gt_rots = np.array([T[:3, :3] for T in gt_poses_4x4])

    our_ate = np.sqrt(np.mean(np.sum((our_pos - gt_pos) ** 2, axis=1)))
    anycam_ate = np.sqrt(np.mean(np.sum((anycam_pos - gt_pos) ** 2, axis=1)))
    print(f"[3D] ATE — Ours: {our_ate:.4f}m, AnyCam: {anycam_ate:.4f}m")

    # Camera frustum drawing
    def draw_camera_frustum(ax, position, rotation, color, scale=0.08, alpha=0.3):
        hw = scale * 0.6
        hh = scale * 0.4
        d = scale
        corners_cam = np.array([
            [0, 0, 0], [-hw, -hh, d], [hw, -hh, d], [hw, hh, d], [-hw, hh, d]
        ])
        corners_world = (rotation @ corners_cam.T).T + position
        for i in range(1, 5):
            j = i % 4 + 1
            ax.plot3D([corners_world[0, 0], corners_world[i, 0]],
                      [corners_world[0, 1], corners_world[i, 1]],
                      [corners_world[0, 2], corners_world[i, 2]],
                      '-', color=color, linewidth=0.5, alpha=alpha)
            ax.plot3D([corners_world[i, 0], corners_world[j, 0]],
                      [corners_world[i, 1], corners_world[j, 1]],
                      [corners_world[i, 2], corners_world[j, 2]],
                      '-', color=color, linewidth=0.5, alpha=alpha)

    n = len(gt_pos)
    all_pos = np.vstack([gt_pos, our_pos, anycam_pos])
    extent = max(np.ptp(all_pos[:, 0]), np.ptp(all_pos[:, 1]), np.ptp(all_pos[:, 2]))
    frustum_scale = extent * 0.04
    frustum_every = 5

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    our_label = f'Ours (MCT) \u2014 ATE={our_ate:.3f}m'
    anycam_label = f'AnyCam \u2014 ATE={anycam_ate:.3f}m'

    ax.plot3D(gt_pos[:, 0], gt_pos[:, 1], gt_pos[:, 2],
              '-', color=C_GT, linewidth=2.5, label='Ground Truth', zorder=10)
    ax.plot3D(our_pos[:, 0], our_pos[:, 1], our_pos[:, 2],
              '-', color=C_OURS, linewidth=1.8, label=our_label, zorder=9)
    ax.plot3D(anycam_pos[:, 0], anycam_pos[:, 1], anycam_pos[:, 2],
              '-', color=C_ANYCAM, linewidth=1.8, label=anycam_label, zorder=8)

    for i in range(0, n, frustum_every):
        draw_camera_frustum(ax, gt_pos[i], gt_rots[i], C_GT,
                            scale=frustum_scale, alpha=0.5)
        draw_camera_frustum(ax, our_pos[i], our_rots[i], C_OURS,
                            scale=frustum_scale, alpha=0.4)
        draw_camera_frustum(ax, anycam_pos[i], anycam_rots[i], C_ANYCAM,
                            scale=frustum_scale, alpha=0.4)

    ax.scatter(*gt_pos[0], color=C_GT, s=60, marker='o', zorder=11)
    ax.scatter(*gt_pos[-1], color=C_GT, s=60, marker='s', zorder=11)

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('3D Camera Trajectory \u2014 market_6')
    ax.legend(loc='upper left', framealpha=0.9)

    max_range = extent * 0.55
    mid = all_pos.mean(axis=0)
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)
    ax.dist = 8

    # Side view
    ax.view_init(elev=20, azim=-60)
    fig.tight_layout()
    out = LATEX_FIGS / 'trajectory_3d_side.png'
    fig.savefig(out)
    plt.close(fig)
    print(f"[DONE] {out}")


if __name__ == '__main__':
    fix_phase_b_plot()
    fix_3d_trajectory()
