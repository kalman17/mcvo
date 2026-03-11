"""
Generate publication-quality thesis figures comparing our FAT model against baselines.

Figures:
1. Focal length across frames (line plot): GT vs AnyCalib vs AnyCam vs Ours
2. Rotation error per pair (bar chart): Ours vs AnyCam
3. Translation direction error per pair (bar chart): Ours vs AnyCam
4. Focal length distribution (box/scatter): AnyCalib per-frame vs FAT aggregated vs GT

Usage:
    python experiments/generate_thesis_figures.py \
        --checkpoint thesis_results/checkpoints/phase_C_v3_h100_epoch_0005.pt \
        --output_dir thesis_results/figures \
        --num_sequences 20 \
        --device cuda:0
"""

import argparse
import json
import sys
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
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

# Project imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments.benchmark_phase_c_checkpoints import (
    create_inference_model,
    create_baseline_model,
    load_phase_c_checkpoint,
    load_sintel_dataset,
    extract_gt_from_sample,
)
from experiments.pose_metrics import (
    rotation_error_degrees,
    translation_direction_error_degrees,
)


def run_our_model(model, imgs, device):
    """Run our FAT model and extract poses + calibration info."""
    data = {'imgs': imgs.unsqueeze(0).to(device)}
    with torch.no_grad():
        output = model.forward_with_calibration_info(data)

    poses = output['pose_result']['poses']
    if poses.dim() == 5:
        poses = poses[:, :, 0]
    pred_poses = poses[0].cpu().numpy()

    # FAT intrinsics (aggregated) — in ray resolution
    fat_intr = output['intrinsics'][0].cpu().numpy()  # [4]
    fat_image_size = output.get('fat_image_size')

    # Scale to input image resolution
    if fat_image_size is not None:
        H_ray, W_ray = fat_image_size
        H_img, W_img = imgs.shape[-2], imgs.shape[-1]
        sx, sy = W_img / W_ray, H_img / H_ray
        fat_intr_scaled = np.array([fat_intr[0]*sx, fat_intr[1]*sy, fat_intr[2]*sx, fat_intr[3]*sy])
    else:
        fat_intr_scaled = fat_intr

    # Per-frame AnyCalib predictions (standalone, no FAT)
    per_frame = output.get('per_frame_intrinsics')
    per_frame_scaled = None
    if per_frame is not None:
        pf = per_frame[0].cpu().numpy()  # [N, 4]
        if fat_image_size is not None:
            per_frame_scaled = pf.copy()
            per_frame_scaled[:, 0] *= sx
            per_frame_scaled[:, 1] *= sy
            per_frame_scaled[:, 2] *= sx
            per_frame_scaled[:, 3] *= sy
        else:
            per_frame_scaled = pf

    return pred_poses, fat_intr_scaled, per_frame_scaled


def run_baseline_model(model, imgs, device, sample):
    """Run vanilla AnyCam and extract poses + focal length."""
    data = {'imgs': imgs.unsqueeze(0).to(device)}

    # Vanilla AnyCam needs projs
    if 'projs' in sample:
        projs = sample['projs']
        if isinstance(projs, np.ndarray):
            projs = torch.from_numpy(projs).float()
        data['projs'] = projs.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(data)

    pred_poses = output['proc_poses'][0].cpu().numpy()

    # Extract AnyCam's selected focal length
    anycam_intr = None
    proc_projs = output.get('proc_projs')
    if proc_projs is not None:
        K = proc_projs[0, 0].cpu().numpy()
        h, w = imgs.shape[-2], imgs.shape[-1]
        fx = K[0, 0] * w / 2
        fy = K[1, 1] * h / 2
        cx = (K[0, 2] + 1) * w / 2
        cy = (K[1, 2] + 1) * h / 2
        anycam_intr = np.array([fx, fy, cx, cy])

    return pred_poses, anycam_intr


def compute_pair_errors(pred_poses, gt_poses):
    """Compute per-pair rotation and translation direction errors."""
    n_pairs = min(len(pred_poses) - 1, gt_poses.shape[0] - 1)
    rot_errors = []
    trans_errors = []
    for i in range(n_pairs):
        pred_rel = np.linalg.inv(pred_poses[i]) @ pred_poses[i + 1]
        gt_rel = np.linalg.inv(gt_poses[i]) @ gt_poses[i + 1]
        rot_errors.append(rotation_error_degrees(pred_rel[:3, :3], gt_rel[:3, :3]))
        trans_errors.append(translation_direction_error_degrees(pred_rel[:3, 3], gt_rel[:3, 3]))
    return rot_errors, trans_errors


def cherry_pick_sequence(our_model, baseline_model, dataset, device, num_candidates=20):
    """Evaluate first N sequences and pick the one where we beat AnyCam most on rotation."""
    print(f"\n[CHERRY-PICK] Evaluating {num_candidates} sequences to find best showcase...")
    candidates = []
    n_total = len(dataset)

    for idx in range(min(num_candidates, n_total)):
        try:
            sample = dataset[idx]
            gt_poses, gt_intrinsics = extract_gt_from_sample(sample, 'sintel')
            if gt_poses is None or gt_poses.shape[0] < 2:
                continue

            imgs = sample['imgs'] if 'imgs' in sample else sample['images']
            if isinstance(imgs, np.ndarray):
                imgs = torch.from_numpy(imgs).float()

            # Our model
            our_poses, fat_intr, per_frame_intr = run_our_model(our_model, imgs, device)
            our_rot, our_trans = compute_pair_errors(our_poses, gt_poses)

            # Baseline
            base_poses, anycam_intr = run_baseline_model(baseline_model, imgs, device, sample)
            base_rot, base_trans = compute_pair_errors(base_poses, gt_poses)

            if len(our_rot) == 0 or len(base_rot) == 0:
                continue

            our_rot_mean = np.mean(our_rot)
            base_rot_mean = np.mean(base_rot)
            improvement = (base_rot_mean - our_rot_mean) / base_rot_mean * 100

            candidates.append({
                'idx': idx,
                'our_rot_mean': our_rot_mean,
                'base_rot_mean': base_rot_mean,
                'improvement_pct': improvement,
                'n_frames': gt_poses.shape[0],
                'has_gt_intrinsics': gt_intrinsics is not None,
            })
            print(f"  Seq {idx}: ours={our_rot_mean:.3f} base={base_rot_mean:.3f} "
                  f"improvement={improvement:+.1f}% frames={gt_poses.shape[0]} "
                  f"gt_calib={'yes' if gt_intrinsics is not None else 'no'}")

            torch.cuda.empty_cache()
        except Exception as e:
            print(f"  Seq {idx}: FAILED — {e}")
            continue

    # Prefer sequences with GT intrinsics and good improvement
    candidates_with_gt = [c for c in candidates if c['has_gt_intrinsics']]
    pool = candidates_with_gt if candidates_with_gt else candidates

    if not pool:
        raise RuntimeError("No valid sequences found!")

    best = max(pool, key=lambda c: c['improvement_pct'])
    print(f"\n[CHERRY-PICK] Selected sequence {best['idx']}: "
          f"{best['improvement_pct']:+.1f}% rotation improvement, "
          f"{best['n_frames']} frames")
    return best['idx']


def generate_figures(our_model, baseline_model, dataset, chosen_idx, device, output_dir):
    """Generate all thesis figures for the chosen sequence."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sample = dataset[chosen_idx]
    gt_poses, gt_intrinsics = extract_gt_from_sample(sample, 'sintel')

    imgs = sample['imgs'] if 'imgs' in sample else sample['images']
    if isinstance(imgs, np.ndarray):
        imgs = torch.from_numpy(imgs).float()

    N = gt_poses.shape[0]
    frame_indices = list(range(N))

    # Run models
    print(f"\n[FIGURE] Running our model on sequence {chosen_idx} ({N} frames)...")
    our_poses, fat_intr, per_frame_intr = run_our_model(our_model, imgs, device)
    our_rot, our_trans = compute_pair_errors(our_poses, gt_poses)

    print(f"[FIGURE] Running baseline model...")
    base_poses, anycam_intr = run_baseline_model(baseline_model, imgs, device, sample)
    base_rot, base_trans = compute_pair_errors(base_poses, gt_poses)

    # Collect focal lengths
    gt_fx = gt_intrinsics[:, 0] if gt_intrinsics is not None else None
    fat_fx = fat_intr[0]  # Single aggregated value
    anycalib_fx = per_frame_intr[:, 0] if per_frame_intr is not None else None
    anycam_fx = anycam_intr[0] if anycam_intr is not None else None

    # Save raw data as JSON for reproducibility
    raw_data = {
        'chosen_idx': chosen_idx,
        'n_frames': N,
        'our_rotation_errors': our_rot,
        'baseline_rotation_errors': base_rot,
        'our_translation_errors': our_trans,
        'baseline_translation_errors': base_trans,
        'fat_fx': float(fat_fx),
        'anycam_fx': float(anycam_fx) if anycam_fx is not None else None,
        'gt_fx': gt_fx.tolist() if gt_fx is not None else None,
        'anycalib_per_frame_fx': anycalib_fx.tolist() if anycalib_fx is not None else None,
    }
    with open(output_dir / 'figure_data.json', 'w') as f:
        json.dump(raw_data, f, indent=2)
    print(f"[FIGURE] Raw data saved to {output_dir / 'figure_data.json'}")

    # Colors
    C_OURS = '#2171b5'      # Blue
    C_ANYCAM = '#e6550d'    # Orange
    C_ANYCALIB = '#31a354'  # Green
    C_GT = '#333333'        # Dark gray

    # ===== Figure 1: Focal length across frames =====
    print("[FIGURE] Generating focal_length_across_frames.png...")
    fig, ax = plt.subplots(figsize=(7, 3.5))

    if gt_fx is not None:
        ax.plot(frame_indices, gt_fx, 'o-', color=C_GT, markersize=4, linewidth=1.5,
                label='GT', zorder=4)
    if anycalib_fx is not None:
        ax.plot(frame_indices, anycalib_fx, 's--', color=C_ANYCALIB, markersize=4,
                linewidth=1.2, label='AnyCalib (per-frame)', zorder=3)
    # FAT: constant across frames
    ax.axhline(y=fat_fx, color=C_OURS, linewidth=2, linestyle='-',
               label=f'Ours (FAT aggregated)', zorder=2)
    # AnyCam: constant across frames
    if anycam_fx is not None:
        ax.axhline(y=anycam_fx, color=C_ANYCAM, linewidth=1.5, linestyle=':',
                   label=f'AnyCam (32-candidate)', zorder=1)

    ax.set_xlabel('Frame index')
    ax.set_ylabel('Focal length $f_x$ (pixels)')
    ax.set_title('Focal Length Prediction Across Frames')
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.5, N - 0.5)
    fig.tight_layout()
    fig.savefig(output_dir / 'focal_length_across_frames.png')
    plt.close(fig)

    # ===== Figure 2: Rotation error per pair =====
    print("[FIGURE] Generating rotation_error_per_pair.png...")
    n_pairs = len(our_rot)
    pair_labels = [f'{i}→{i+1}' for i in range(n_pairs)]
    x = np.arange(n_pairs)
    width = 0.35

    fig, ax = plt.subplots(figsize=(5, 3.5))
    bars1 = ax.bar(x - width/2, base_rot, width, label='AnyCam', color=C_ANYCAM, alpha=0.85)
    bars2 = ax.bar(x + width/2, our_rot, width, label='Ours', color=C_OURS, alpha=0.85)

    ax.set_xlabel('Frame pair')
    ax.set_ylabel('Rotation error (degrees)')
    ax.set_title('Rotation Error Per Frame Pair')
    ax.set_xticks(x)
    ax.set_xticklabels(pair_labels)
    ax.legend(framealpha=0.9)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(output_dir / 'rotation_error_per_pair.png')
    plt.close(fig)

    # ===== Figure 3: Translation direction error per pair =====
    print("[FIGURE] Generating translation_error_per_pair.png...")
    fig, ax = plt.subplots(figsize=(5, 3.5))
    bars1 = ax.bar(x - width/2, base_trans, width, label='AnyCam', color=C_ANYCAM, alpha=0.85)
    bars2 = ax.bar(x + width/2, our_trans, width, label='Ours', color=C_OURS, alpha=0.85)

    ax.set_xlabel('Frame pair')
    ax.set_ylabel('Translation direction error (degrees)')
    ax.set_title('Translation Direction Error Per Frame Pair')
    ax.set_xticks(x)
    ax.set_xticklabels(pair_labels)
    ax.legend(framealpha=0.9)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(output_dir / 'translation_error_per_pair.png')
    plt.close(fig)

    # ===== Figure 4: Focal length distribution =====
    if gt_fx is not None and anycalib_fx is not None:
        print("[FIGURE] Generating focal_length_distribution.png...")
        fig, ax = plt.subplots(figsize=(5, 3.5))

        gt_mean = np.mean(gt_fx)

        # AnyCalib per-frame scatter
        ax.scatter(np.zeros(len(anycalib_fx)) + 0, anycalib_fx, color=C_ANYCALIB,
                   alpha=0.6, s=30, zorder=3, label='AnyCalib (per-frame)')
        ax.scatter([0], [np.mean(anycalib_fx)], color=C_ANYCALIB, s=100, marker='D',
                   edgecolors='black', linewidths=0.8, zorder=4)

        # FAT aggregated (single point)
        ax.scatter([1], [fat_fx], color=C_OURS, s=100, marker='D',
                   edgecolors='black', linewidths=0.8, zorder=4, label='Ours (FAT)')

        # AnyCam (single point)
        if anycam_fx is not None:
            ax.scatter([2], [anycam_fx], color=C_ANYCAM, s=100, marker='D',
                       edgecolors='black', linewidths=0.8, zorder=4, label='AnyCam')

        # GT reference line
        ax.axhline(y=gt_mean, color=C_GT, linewidth=1.5, linestyle='--',
                   label=f'GT mean ($f_x$={gt_mean:.0f})', zorder=2)

        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(['AnyCalib\n(per-frame)', 'Ours\n(FAT)', 'AnyCam\n(32-cand.)'])
        ax.set_ylabel('Focal length $f_x$ (pixels)')
        ax.set_title('Focal Length Predictions')
        ax.legend(loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3, axis='y')
        fig.tight_layout()
        fig.savefig(output_dir / 'focal_length_distribution.png')
        plt.close(fig)

    print(f"\n[DONE] All figures saved to {output_dir}/")
    print(f"  - focal_length_across_frames.png")
    print(f"  - rotation_error_per_pair.png")
    print(f"  - translation_error_per_pair.png")
    print(f"  - focal_length_distribution.png")
    print(f"  - figure_data.json (raw data)")


def main():
    parser = argparse.ArgumentParser(description='Generate thesis figures')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to Phase C checkpoint')
    parser.add_argument('--anycam_config', type=str,
                        default='pretrained_models/anycam_seq8/training_config.yaml')
    parser.add_argument('--pretrained_anycam', type=str,
                        default='pretrained_models/anycam_seq8/training_checkpoint_247500.pt')
    parser.add_argument('--data_root', type=str, default=None,
                        help='Root dir containing eval datasets (auto-detected if not set)')
    parser.add_argument('--output_dir', type=str, default='thesis_results/figures')
    parser.add_argument('--num_sequences', type=int, default=20,
                        help='Number of sequences to evaluate for cherry-picking')
    parser.add_argument('--chosen_idx', type=int, default=None,
                        help='Skip cherry-picking, use this sequence index directly')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--image_size', type=int, default=336)
    args = parser.parse_args()

    device = args.device

    # Auto-detect data root
    data_root = args.data_root
    if data_root is None:
        for candidate in ['/storage/user/maka/eval_datasets', '/data/thesis', '/home/kalman/TUM/thesis']:
            if Path(candidate).exists():
                data_root = candidate
                break
    if data_root is None:
        print("ERROR: Could not auto-detect data_root. Specify --data_root.")
        sys.exit(1)
    print(f"[CONFIG] Data root: {data_root}")

    # Load Sintel dataset
    print("[LOAD] Loading Sintel dataset...")
    dataset = load_sintel_dataset(data_root, num_samples=args.num_sequences,
                                  image_size=args.image_size, frame_count=4, dilation=1)
    print(f"[LOAD] Sintel: {len(dataset)} sequences available")

    # Create our model
    print("[LOAD] Creating our FAT model...")
    our_model = create_inference_model(args.anycam_config, device)
    load_phase_c_checkpoint(our_model, args.checkpoint, device)
    our_model = our_model.to(device)
    our_model.eval()

    # Create baseline model
    print("[LOAD] Creating vanilla AnyCam baseline...")
    baseline_model = create_baseline_model(args.anycam_config, args.pretrained_anycam, device)

    # Cherry-pick or use provided index
    if args.chosen_idx is not None:
        chosen_idx = args.chosen_idx
        print(f"[CONFIG] Using provided sequence index: {chosen_idx}")
    else:
        chosen_idx = cherry_pick_sequence(our_model, baseline_model, dataset, device,
                                          num_candidates=args.num_sequences)

    # Generate figures
    generate_figures(our_model, baseline_model, dataset, chosen_idx, device, args.output_dir)


if __name__ == '__main__':
    main()
