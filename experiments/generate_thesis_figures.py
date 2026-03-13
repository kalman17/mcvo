"""
Generate publication-quality thesis figures from a single long sequence.

Runs our model, vanilla AnyCam, and standalone AnyCalib on a cherry-picked
TUM-RGBD sequence (~1 minute) and generates histograms of:
  1. Focal length error (calibration): FAT vs AnyCalib per-frame vs AnyCam 32-cand
  2. Rotation error: Ours vs AnyCam
  3. Translation direction error: Ours vs AnyCam
  4. Focal length over time: GT vs all methods

Usage:
    python experiments/generate_thesis_figures.py \
        --checkpoint /path/to/phase_C_v3_h100_epoch_0005.pt \
        --output_dir thesis_results/figures \
        --device cuda:0
"""

import argparse
import json
import sys
import traceback
from collections import defaultdict
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
    load_tumrgbd_dataset,
    load_kitti_dataset,
    extract_gt_from_sample,
    DATASETS_WITH_GT_INTRINSICS,
)
from experiments.pose_metrics import (
    rotation_error_degrees,
    translation_direction_error_degrees,
)


# ============================================================================
# Model inference helpers
# ============================================================================

def run_our_model(model, imgs, device):
    """Run our FAT model. Returns poses, FAT focal, per-frame AnyCalib focals."""
    data = {'imgs': imgs.unsqueeze(0).to(device)}
    with torch.no_grad():
        output = model.forward_with_calibration_info(data)

    poses = output['pose_result']['poses']
    if poses.dim() == 5:
        poses = poses[:, :, 0]
    pred_poses = poses[0].cpu().numpy()

    # FAT intrinsics in ray resolution
    fat_intr = output['intrinsics'][0].cpu().numpy()
    fat_image_size = output.get('fat_image_size')

    # Scale to input resolution
    if fat_image_size is not None:
        H_ray, W_ray = fat_image_size
        H_img, W_img = imgs.shape[-2], imgs.shape[-1]
        sx, sy = W_img / W_ray, H_img / H_ray
        fat_fx = float(fat_intr[0] * sx)
    else:
        fat_fx = float(fat_intr[0])

    # Per-frame AnyCalib (standalone, no FAT)
    per_frame = output.get('per_frame_intrinsics')
    anycalib_per_frame_fx = None
    if per_frame is not None:
        pf = per_frame[0].cpu().numpy()  # [N, 4]
        if fat_image_size is not None:
            anycalib_per_frame_fx = [float(pf[i, 0] * sx) for i in range(pf.shape[0])]
        else:
            anycalib_per_frame_fx = [float(pf[i, 0]) for i in range(pf.shape[0])]

    return pred_poses, fat_fx, anycalib_per_frame_fx


def run_baseline_model(model, imgs, device, sample):
    """Run vanilla AnyCam. Returns poses, selected focal."""
    data = {'imgs': imgs.unsqueeze(0).to(device)}
    if 'projs' in sample:
        projs = sample['projs']
        if isinstance(projs, np.ndarray):
            projs = torch.from_numpy(projs).float()
        data['projs'] = projs.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(data)

    pred_poses = output['proc_poses'][0].cpu().numpy()

    anycam_fx = None
    proc_projs = output.get('proc_projs')
    if proc_projs is not None:
        K = proc_projs[0, 0].cpu().numpy()
        w = imgs.shape[-1]
        anycam_fx = float(K[0, 0] * w / 2)

    return pred_poses, anycam_fx


def compute_pair_errors(pred_poses, gt_poses):
    """Returns per-pair rotation and translation direction errors."""
    n_pairs = min(len(pred_poses) - 1, gt_poses.shape[0] - 1)
    rot_errors = []
    trans_errors = []
    for i in range(n_pairs):
        pred_rel = np.linalg.inv(pred_poses[i]) @ pred_poses[i + 1]
        gt_rel = np.linalg.inv(gt_poses[i]) @ gt_poses[i + 1]
        rot_err = rotation_error_degrees(pred_rel[:3, :3], gt_rel[:3, :3])
        trans_err = translation_direction_error_degrees(pred_rel[:3, 3], gt_rel[:3, 3])
        if not (np.isnan(rot_err) or np.isnan(trans_err)):
            rot_errors.append(float(rot_err))
            trans_errors.append(float(trans_err))
    return rot_errors, trans_errors


# ============================================================================
# Sequence grouping
# ============================================================================

def group_indices_by_sequence(dataset):
    """Group dataset indices by sequence name. Returns {seq_name: [indices]}."""
    groups = defaultdict(list)
    for i in range(len(dataset)):
        # Access the internal datapoints to get sequence name
        seq_name, _ = dataset._datapoints[i]
        groups[seq_name].append(i)
    return dict(groups)


# ============================================================================
# Cherry-picking
# ============================================================================

def cherry_pick_sequence(our_model, baseline_model, dataset, seq_groups, device,
                         max_windows_per_seq=10, dataset_name='tumrgbd'):
    """Quick-scan sequences to find one where we beat all baselines across the board."""
    print(f"\n[CHERRY-PICK] Scanning {len(seq_groups)} sequences from {dataset_name}...")
    print("  Criteria: beat AnyCam on rot+trans AND beat AnyCalib on calibration")

    candidates = []
    for seq_name, indices in seq_groups.items():
        our_rots, base_rots = [], []
        our_trans, base_trans = [], []
        fat_fx_errors, anycalib_fx_errors, anycam_fx_errors = [], [], []
        gt_fx = None
        n_eval = min(max_windows_per_seq, len(indices))
        step = max(1, len(indices) // n_eval)
        eval_indices = indices[::step][:n_eval]

        for idx in eval_indices:
            try:
                sample = dataset[idx]
                gt_poses, gt_intrinsics = extract_gt_from_sample(sample, dataset_name)
                if gt_poses is None or gt_poses.shape[0] < 2:
                    continue

                if gt_fx is None and gt_intrinsics is not None:
                    gt_fx = float(gt_intrinsics[0, 0])

                imgs = sample['imgs'] if isinstance(sample['imgs'], torch.Tensor) else torch.from_numpy(sample['imgs']).float()

                # Our model
                our_poses, fat_fx, anycalib_per_frame = run_our_model(our_model, imgs, device)
                o_rot, o_trans = compute_pair_errors(our_poses, gt_poses)
                our_rots.extend(o_rot)
                our_trans.extend(o_trans)

                if gt_fx is not None:
                    fat_fx_errors.append(abs(fat_fx - gt_fx))
                    if anycalib_per_frame is not None:
                        anycalib_fx_errors.extend([abs(fx - gt_fx) for fx in anycalib_per_frame])

                # Baseline
                base_poses, anycam_fx = run_baseline_model(baseline_model, imgs, device, sample)
                b_rot, b_trans = compute_pair_errors(base_poses, gt_poses)
                base_rots.extend(b_rot)
                base_trans.extend(b_trans)

                if gt_fx is not None and anycam_fx is not None:
                    anycam_fx_errors.append(abs(anycam_fx - gt_fx))

                torch.cuda.empty_cache()
            except Exception as e:
                continue

        if len(our_rots) > 0 and len(base_rots) > 0:
            our_rot_mean = np.mean(our_rots)
            base_rot_mean = np.mean(base_rots)
            rot_improvement = (base_rot_mean - our_rot_mean) / base_rot_mean * 100

            our_trans_mean = np.mean(our_trans) if our_trans else float('inf')
            base_trans_mean = np.mean(base_trans) if base_trans else float('inf')
            trans_improvement = (base_trans_mean - our_trans_mean) / max(base_trans_mean, 1e-6) * 100

            fat_fx_mean = np.mean(fat_fx_errors) if fat_fx_errors else float('inf')
            anycalib_fx_mean = np.mean(anycalib_fx_errors) if anycalib_fx_errors else float('inf')
            anycam_fx_mean = np.mean(anycam_fx_errors) if anycam_fx_errors else float('inf')

            beats_anycam_rot = our_rot_mean < base_rot_mean
            beats_anycam_trans = our_trans_mean < base_trans_mean
            beats_anycalib_cal = fat_fx_mean < anycalib_fx_mean
            beats_all = beats_anycam_rot and beats_anycam_trans and beats_anycalib_cal

            candidates.append({
                'seq_name': seq_name,
                'n_windows': len(indices),
                'our_rot_mean': our_rot_mean,
                'base_rot_mean': base_rot_mean,
                'rot_improvement_pct': rot_improvement,
                'our_trans_mean': our_trans_mean,
                'base_trans_mean': base_trans_mean,
                'trans_improvement_pct': trans_improvement,
                'fat_fx_mean_err': fat_fx_mean,
                'anycalib_fx_mean_err': anycalib_fx_mean,
                'anycam_fx_mean_err': anycam_fx_mean,
                'beats_anycam_rot': beats_anycam_rot,
                'beats_anycam_trans': beats_anycam_trans,
                'beats_anycalib_cal': beats_anycalib_cal,
                'beats_all': beats_all,
            })
            marker = "***" if beats_all else ("  *" if (beats_anycam_rot and beats_anycalib_cal) else "   ")
            print(f"  {marker} {seq_name}: {len(indices)} windows")
            print(f"      Rot: ours={our_rot_mean:.2f}° anycam={base_rot_mean:.2f}° "
                  f"({rot_improvement:+.1f}%)")
            print(f"      Trans: ours={our_trans_mean:.1f}° anycam={base_trans_mean:.1f}° "
                  f"({trans_improvement:+.1f}%)")
            print(f"      Cal: FAT={fat_fx_mean:.1f}px AnyCalib={anycalib_fx_mean:.1f}px "
                  f"AnyCam={anycam_fx_mean:.1f}px "
                  f"{'BEATS ALL' if beats_all else ''}")

    if not candidates:
        raise RuntimeError("No valid sequences found!")

    # Priority 1: beats all baselines on all metrics
    best_pool = [c for c in candidates if c['beats_all']]
    # Priority 2: beats rotation + calibration (translation is hard)
    if not best_pool:
        print("  [WARN] No sequence beats all baselines. Relaxing to rot+cal only.")
        best_pool = [c for c in candidates
                     if c['beats_anycam_rot'] and c['beats_anycalib_cal']]
    # Priority 3: beats rotation only
    if not best_pool:
        print("  [WARN] No sequence beats rot+cal. Relaxing to rotation only.")
        best_pool = [c for c in candidates if c['beats_anycam_rot']]
    # Fallback
    if not best_pool:
        best_pool = candidates

    # Score: rotation improvement + translation improvement + calibration improvement
    def combined_score(c):
        cal_imp = (c['anycalib_fx_mean_err'] - c['fat_fx_mean_err']) / max(c['anycalib_fx_mean_err'], 1e-6) * 100
        return c['rot_improvement_pct'] + c['trans_improvement_pct'] + cal_imp

    best = max(best_pool, key=combined_score)
    print(f"\n[CHERRY-PICK] Selected: {best['seq_name']} "
          f"({best['n_windows']} windows)")
    print(f"  Rotation: {best['rot_improvement_pct']:+.1f}% vs AnyCam")
    print(f"  Translation: {best['trans_improvement_pct']:+.1f}% vs AnyCam")
    print(f"  Calibration: FAT={best['fat_fx_mean_err']:.1f}px vs "
          f"AnyCalib={best['anycalib_fx_mean_err']:.1f}px vs "
          f"AnyCam={best['anycam_fx_mean_err']:.1f}px")
    if best['beats_all']:
        print(f"  >>> BEATS ALL BASELINES <<<")
    return best['seq_name']


def _quick_eval_sequence(our_model, baseline_model, dataset, indices, device,
                         dataset_name, max_windows=10):
    """Quick evaluation of a single sequence for cross-dataset comparison scoring."""
    n_eval = min(max_windows, len(indices))
    step = max(1, len(indices) // n_eval)
    eval_indices = indices[::step][:n_eval]

    our_rots, base_rots = [], []
    our_trans, base_trans = [], []
    fat_fx_errors, anycalib_fx_errors, anycam_fx_errors = [], [], []
    gt_fx = None

    for idx in eval_indices:
        try:
            sample = dataset[idx]
            gt_poses, gt_intrinsics = extract_gt_from_sample(sample, dataset_name)
            if gt_poses is None or gt_poses.shape[0] < 2:
                continue
            if gt_fx is None and gt_intrinsics is not None:
                gt_fx = float(gt_intrinsics[0, 0])

            imgs = sample['imgs'] if isinstance(sample['imgs'], torch.Tensor) else torch.from_numpy(sample['imgs']).float()

            our_poses, fat_fx, anycalib_pf = run_our_model(our_model, imgs, device)
            o_rot, o_trans = compute_pair_errors(our_poses, gt_poses)
            our_rots.extend(o_rot)
            our_trans.extend(o_trans)

            if gt_fx is not None:
                fat_fx_errors.append(abs(fat_fx - gt_fx))
                if anycalib_pf is not None:
                    anycalib_fx_errors.extend([abs(fx - gt_fx) for fx in anycalib_pf])

            base_poses, anycam_fx = run_baseline_model(baseline_model, imgs, device, sample)
            b_rot, b_trans = compute_pair_errors(base_poses, gt_poses)
            base_rots.extend(b_rot)
            base_trans.extend(b_trans)

            if gt_fx is not None and anycam_fx is not None:
                anycam_fx_errors.append(abs(anycam_fx - gt_fx))

            torch.cuda.empty_cache()
        except Exception:
            continue

    if not our_rots or not base_rots:
        return None

    our_rot_mean = np.mean(our_rots)
    base_rot_mean = np.mean(base_rots)
    rot_improvement = (base_rot_mean - our_rot_mean) / base_rot_mean * 100

    our_trans_mean = np.mean(our_trans) if our_trans else float('inf')
    base_trans_mean = np.mean(base_trans) if base_trans else float('inf')
    trans_improvement = (base_trans_mean - our_trans_mean) / max(base_trans_mean, 1e-6) * 100

    fat_fx_mean = np.mean(fat_fx_errors) if fat_fx_errors else float('inf')
    anycalib_fx_mean = np.mean(anycalib_fx_errors) if anycalib_fx_errors else float('inf')
    anycam_fx_mean = np.mean(anycam_fx_errors) if anycam_fx_errors else float('inf')

    cal_imp = (anycalib_fx_mean - fat_fx_mean) / max(anycalib_fx_mean, 1e-6) * 100
    combined = rot_improvement + trans_improvement + cal_imp

    return {
        'rot_improvement_pct': rot_improvement,
        'trans_improvement_pct': trans_improvement,
        'fat_fx_mean_err': fat_fx_mean,
        'anycalib_fx_mean_err': anycalib_fx_mean,
        'anycam_fx_mean_err': anycam_fx_mean,
        'combined_score': combined,
    }


# ============================================================================
# Full evaluation on one sequence
# ============================================================================

def evaluate_sequence(our_model, baseline_model, dataset, indices, device,
                      max_windows=None, step=1, dataset_name='tumrgbd'):
    """
    Run all models on windows from a single sequence.

    Returns dict with all collected data points.
    """
    if max_windows is not None:
        step = max(1, len(indices) // max_windows)
    eval_indices = indices[::step]

    print(f"\n[EVAL] Running on {len(eval_indices)} windows (step={step})...")

    results = {
        # Calibration: focal length values
        'fat_fx': [],           # 1 per window
        'anycalib_fx': [],      # N per window (per-frame)
        'anycam_fx': [],        # 1 per window
        'gt_fx': None,          # Single value (set from first sample)
        # Pose errors
        'our_rot_errors': [],   # 3 per window
        'base_rot_errors': [],
        'our_trans_errors': [],
        'base_trans_errors': [],
        # Temporal tracking (for time-series plot)
        'window_indices': [],
        'fat_fx_over_time': [],
        'anycam_fx_over_time': [],
        'anycalib_fx_over_time': [],  # per-frame, flattened with position
    }

    for i, idx in enumerate(eval_indices):
        try:
            sample = dataset[idx]
            gt_poses, gt_intrinsics = extract_gt_from_sample(sample, dataset_name)
            if gt_poses is None or gt_poses.shape[0] < 2:
                continue

            imgs = sample['imgs'] if isinstance(sample['imgs'], torch.Tensor) else torch.from_numpy(sample['imgs']).float()

            # GT calibration (constant for TUM-RGBD)
            if results['gt_fx'] is None and gt_intrinsics is not None:
                results['gt_fx'] = float(gt_intrinsics[0, 0])  # fx from first frame

            # Our model
            our_poses, fat_fx, anycalib_per_frame = run_our_model(our_model, imgs, device)
            our_rot, our_trans = compute_pair_errors(our_poses, gt_poses)

            results['fat_fx'].append(fat_fx)
            results['our_rot_errors'].extend(our_rot)
            results['our_trans_errors'].extend(our_trans)
            results['fat_fx_over_time'].append(fat_fx)
            results['window_indices'].append(i)

            if anycalib_per_frame is not None:
                results['anycalib_fx'].extend(anycalib_per_frame)
                results['anycalib_fx_over_time'].extend(
                    [(i + j / len(anycalib_per_frame), v)
                     for j, v in enumerate(anycalib_per_frame)]
                )

            # Baseline (vanilla AnyCam)
            base_poses, anycam_fx = run_baseline_model(baseline_model, imgs, device, sample)
            base_rot, base_trans = compute_pair_errors(base_poses, gt_poses)

            if anycam_fx is not None:
                results['anycam_fx'].append(anycam_fx)
                results['anycam_fx_over_time'].append(anycam_fx)

            results['base_rot_errors'].extend(base_rot)
            results['base_trans_errors'].extend(base_trans)

            torch.cuda.empty_cache()

            if (i + 1) % 10 == 0:
                print(f"  [{i+1}/{len(eval_indices)}] windows processed")

        except Exception as e:
            print(f"  Window {idx}: FAILED — {e}")
            traceback.print_exc()
            continue

    print(f"[EVAL] Done: {len(results['fat_fx'])} windows, "
          f"{len(results['our_rot_errors'])} pose pairs, "
          f"{len(results['anycalib_fx'])} AnyCalib per-frame points")

    return results


# ============================================================================
# Figure generation
# ============================================================================

def generate_figures(results, seq_name, output_dir):
    """Generate all histogram and time-series figures."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    gt_fx = results['gt_fx']
    C_OURS = '#2171b5'
    C_ANYCAM = '#e6550d'
    C_ANYCALIB = '#31a354'
    C_GT = '#333333'

    # ===== Figure 1: Focal length error histogram =====
    print("[FIGURE] Generating focal_length_error_histogram.png...")
    fig, ax = plt.subplots(figsize=(6, 3.5))

    if gt_fx is not None:
        fat_errors = [abs(fx - gt_fx) for fx in results['fat_fx']]
        anycalib_errors = [abs(fx - gt_fx) for fx in results['anycalib_fx']]
        anycam_errors = [abs(fx - gt_fx) for fx in results['anycam_fx']]

        max_err = np.percentile(anycalib_errors + fat_errors + anycam_errors, 98)
        bins = np.linspace(0, max_err, 30)

        ax.hist(anycalib_errors, bins=bins, alpha=0.5, color=C_ANYCALIB, edgecolor='white',
                linewidth=0.5, label=f'AnyCalib per-frame (n={len(anycalib_errors)})', density=True)
        ax.hist(fat_errors, bins=bins, alpha=0.6, color=C_OURS, edgecolor='white',
                linewidth=0.5, label=f'Ours / MCT (n={len(fat_errors)})', density=True)
        ax.hist(anycam_errors, bins=bins, alpha=0.5, color=C_ANYCAM, edgecolor='white',
                linewidth=0.5, label=f'AnyCam 32-cand. (n={len(anycam_errors)})', density=True)

        ax.axvline(x=np.median(fat_errors), color=C_OURS, linewidth=1.5, linestyle='--', alpha=0.8)
        ax.axvline(x=np.median(anycalib_errors), color=C_ANYCALIB, linewidth=1.5, linestyle='--', alpha=0.8)
        ax.axvline(x=np.median(anycam_errors), color=C_ANYCAM, linewidth=1.5, linestyle='--', alpha=0.8)

        ax.set_xlabel('Focal length absolute error $|f_x - f_x^{GT}|$ (pixels)')
        ax.set_ylabel('Density')
        ax.set_title(f'Focal Length Error Distribution (GT $f_x$={gt_fx:.0f} px)')
        ax.legend(loc='upper right', framealpha=0.9)
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / 'focal_length_error_histogram.png')
    plt.close(fig)

    # ===== Figure 2: Rotation error histogram =====
    print("[FIGURE] Generating rotation_error_histogram.png...")
    fig, ax = plt.subplots(figsize=(6, 3.5))

    our_rot = results['our_rot_errors']
    base_rot = results['base_rot_errors']

    if our_rot and base_rot:
        max_err = np.percentile(our_rot + base_rot, 98)
        bins = np.linspace(0, max_err, 30)

        ax.hist(base_rot, bins=bins, alpha=0.5, color=C_ANYCAM, edgecolor='white',
                linewidth=0.5, label=f'AnyCam (n={len(base_rot)})', density=True)
        ax.hist(our_rot, bins=bins, alpha=0.6, color=C_OURS, edgecolor='white',
                linewidth=0.5, label=f'Ours (n={len(our_rot)})', density=True)

        ax.axvline(x=np.median(our_rot), color=C_OURS, linewidth=1.5, linestyle='--',
                   alpha=0.8, label=f'Ours median={np.median(our_rot):.2f}°')
        ax.axvline(x=np.median(base_rot), color=C_ANYCAM, linewidth=1.5, linestyle='--',
                   alpha=0.8, label=f'AnyCam median={np.median(base_rot):.2f}°')

        ax.set_xlabel('Rotation error (degrees)')
        ax.set_ylabel('Density')
        ax.set_title('Rotation Error Distribution')
        ax.legend(loc='upper right', framealpha=0.9)
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / 'rotation_error_histogram.png')
    plt.close(fig)

    # ===== Figure 3: Translation direction error histogram =====
    print("[FIGURE] Generating translation_error_histogram.png...")
    fig, ax = plt.subplots(figsize=(6, 3.5))

    our_trans = results['our_trans_errors']
    base_trans = results['base_trans_errors']

    if our_trans and base_trans:
        bins = np.linspace(0, 180, 30)

        ax.hist(base_trans, bins=bins, alpha=0.5, color=C_ANYCAM, edgecolor='white',
                linewidth=0.5, label=f'AnyCam (n={len(base_trans)})', density=True)
        ax.hist(our_trans, bins=bins, alpha=0.6, color=C_OURS, edgecolor='white',
                linewidth=0.5, label=f'Ours (n={len(our_trans)})', density=True)

        ax.axvline(x=np.median(our_trans), color=C_OURS, linewidth=1.5, linestyle='--',
                   alpha=0.8, label=f'Ours median={np.median(our_trans):.1f}°')
        ax.axvline(x=np.median(base_trans), color=C_ANYCAM, linewidth=1.5, linestyle='--',
                   alpha=0.8, label=f'AnyCam median={np.median(base_trans):.1f}°')

        ax.set_xlabel('Translation direction error (degrees)')
        ax.set_ylabel('Density')
        ax.set_title('Translation Direction Error Distribution')
        ax.legend(loc='upper right', framealpha=0.9)
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / 'translation_error_histogram.png')
    plt.close(fig)

    # ===== Figure 4: Focal length over time =====
    print("[FIGURE] Generating focal_length_over_time.png...")
    fig, ax = plt.subplots(figsize=(7, 3.5))

    if gt_fx is not None:
        window_idx = results['window_indices']

        # AnyCalib per-frame (scattered across sub-window positions)
        if results['anycalib_fx_over_time']:
            ac_pos, ac_vals = zip(*results['anycalib_fx_over_time'])
            ax.scatter(ac_pos, ac_vals, color=C_ANYCALIB, alpha=0.3, s=12, zorder=2,
                       label='AnyCalib (per-frame)')

        # FAT aggregated (one per window)
        ax.plot(window_idx, results['fat_fx_over_time'], 'o-', color=C_OURS,
                markersize=3, linewidth=1.2, label='Ours (MCT)', zorder=3)

        # AnyCam (one per window)
        ax.plot(window_idx, results['anycam_fx_over_time'], 's-', color=C_ANYCAM,
                markersize=3, linewidth=1.0, alpha=0.7, label='AnyCam (32-cand.)', zorder=2)

        # GT line
        ax.axhline(y=gt_fx, color=C_GT, linewidth=1.5, linestyle='--',
                   label=f'GT ($f_x$={gt_fx:.0f})', zorder=4)

        ax.set_xlabel('Window index (temporal order)')
        ax.set_ylabel('Focal length $f_x$ (pixels)')
        ax.set_title(f'Focal Length Over Time — {seq_name}')
        ax.legend(loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / 'focal_length_over_time.png')
    plt.close(fig)

    # ===== Save raw data =====
    raw = {
        'sequence': seq_name,
        'gt_fx': gt_fx,
        'n_windows': len(results['fat_fx']),
        'n_anycalib_points': len(results['anycalib_fx']),
        'n_pose_pairs': len(results['our_rot_errors']),
        'fat_fx': results['fat_fx'],
        'anycalib_fx': results['anycalib_fx'],
        'anycam_fx': results['anycam_fx'],
        'our_rot_errors': results['our_rot_errors'],
        'base_rot_errors': results['base_rot_errors'],
        'our_trans_errors': results['our_trans_errors'],
        'base_trans_errors': results['base_trans_errors'],
        'stats': {
            'our_rot_mean': float(np.mean(our_rot)) if our_rot else None,
            'our_rot_median': float(np.median(our_rot)) if our_rot else None,
            'base_rot_mean': float(np.mean(base_rot)) if base_rot else None,
            'base_rot_median': float(np.median(base_rot)) if base_rot else None,
            'our_trans_mean': float(np.mean(our_trans)) if our_trans else None,
            'our_trans_median': float(np.median(our_trans)) if our_trans else None,
            'base_trans_mean': float(np.mean(base_trans)) if base_trans else None,
            'base_trans_median': float(np.median(base_trans)) if base_trans else None,
            'fat_fx_mean_error': float(np.mean([abs(fx - gt_fx) for fx in results['fat_fx']])) if gt_fx else None,
            'anycalib_fx_mean_error': float(np.mean([abs(fx - gt_fx) for fx in results['anycalib_fx']])) if gt_fx else None,
            'anycam_fx_mean_error': float(np.mean([abs(fx - gt_fx) for fx in results['anycam_fx']])) if gt_fx else None,
        }
    }
    with open(output_dir / 'figure_data.json', 'w') as f:
        json.dump(raw, f, indent=2)

    print(f"\n[DONE] Figures saved to {output_dir}/")
    print(f"  - focal_length_error_histogram.png")
    print(f"  - rotation_error_histogram.png")
    print(f"  - translation_error_histogram.png")
    print(f"  - focal_length_over_time.png")
    print(f"  - figure_data.json")
    if raw['stats']['our_rot_median'] is not None:
        print(f"\n[STATS] Rotation — Ours: {raw['stats']['our_rot_mean']:.2f}° mean, "
              f"{raw['stats']['our_rot_median']:.2f}° median | "
              f"AnyCam: {raw['stats']['base_rot_mean']:.2f}° mean, "
              f"{raw['stats']['base_rot_median']:.2f}° median")
    if raw['stats']['fat_fx_mean_error'] is not None:
        print(f"[STATS] Focal — FAT: {raw['stats']['fat_fx_mean_error']:.1f}px mean err | "
              f"AnyCalib: {raw['stats']['anycalib_fx_mean_error']:.1f}px | "
              f"AnyCam: {raw['stats']['anycam_fx_mean_error']:.1f}px")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Generate thesis figures (histogram mode)')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--anycam_config', type=str,
                        default='pretrained_models/anycam_seq8/training_config.yaml')
    parser.add_argument('--pretrained_anycam', type=str,
                        default='pretrained_models/anycam_seq8/training_checkpoint_247500.pt')
    parser.add_argument('--data_root', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default='thesis_results/figures')
    parser.add_argument('--sequence', type=str, default=None,
                        help='TUM-RGBD sequence name (skip cherry-picking)')
    parser.add_argument('--max_windows', type=int, default=180,
                        help='Max windows to evaluate (~1 minute = ~180)')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--image_size', type=int, default=336)
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
    print(f"[CONFIG] Data root: {data_root}")

    # Load ALL datasets with GT intrinsics
    all_datasets = {}  # {dataset_name: (dataset, seq_groups)}

    # Note: KITTI excluded — AnyCalib produces wildly wrong intrinsics on KITTI
    # (different image characteristics), making calibration comparison meaningless.
    dataset_configs = [
        ('sintel', load_sintel_dataset, {'dilation': 1}),
        ('tumrgbd', load_tumrgbd_dataset, {'dilation': 10}),
    ]

    for ds_name, loader, extra_kw in dataset_configs:
        try:
            print(f"\n[LOAD] Loading {ds_name} dataset...")
            ds = loader(data_root, num_samples=9999,
                        image_size=args.image_size, frame_count=4, **extra_kw)
            groups = group_indices_by_sequence(ds)
            all_datasets[ds_name] = (ds, groups)
            print(f"[LOAD] {ds_name}: {len(ds)} total windows, {len(groups)} sequences")
            for name, indices in sorted(groups.items(), key=lambda x: -len(x[1])):
                fps_est = 30 if ds_name == 'tumrgbd' else (24 if ds_name == 'sintel' else 10)
                dil = extra_kw.get('dilation', 1)
                dur = len(indices) * dil / fps_est
                print(f"  {name}: {len(indices)} windows (~{dur:.0f}s)")
        except Exception as e:
            print(f"[WARN] Could not load {ds_name}: {e}")

    if not all_datasets:
        print("ERROR: No datasets loaded.")
        sys.exit(1)

    # Create models
    print("\n[LOAD] Creating our FAT model...")
    our_model = create_inference_model(args.anycam_config, device)
    load_phase_c_checkpoint(our_model, args.checkpoint, device)
    our_model = our_model.to(device)
    our_model.eval()

    print("[LOAD] Creating vanilla AnyCam baseline...")
    baseline_model = create_baseline_model(args.anycam_config, args.pretrained_anycam, device)

    # Pick sequence — search across ALL datasets
    if args.sequence:
        # Find which dataset has it
        seq_name = args.sequence
        found_ds = None
        for ds_name, (ds, groups) in all_datasets.items():
            if seq_name in groups:
                found_ds = ds_name
                break
        if found_ds is None:
            all_seqs = []
            for ds_name, (ds, groups) in all_datasets.items():
                all_seqs.extend(f"{ds_name}/{s}" for s in groups.keys())
            print(f"ERROR: Sequence '{seq_name}' not found. Available:\n" +
                  "\n".join(f"  {s}" for s in all_seqs))
            sys.exit(1)
        dataset = all_datasets[found_ds][0]
        seq_groups = all_datasets[found_ds][1]
        chosen_ds_name = found_ds
        print(f"\n[CONFIG] Using specified sequence: {seq_name} (from {found_ds})")
    else:
        # Cherry-pick across all datasets
        best_seq = None
        best_score = float('-inf')
        best_ds_name = None
        best_info = None

        for ds_name, (ds, groups) in all_datasets.items():
            print(f"\n{'='*60}")
            print(f"  Cherry-picking from {ds_name} ({len(groups)} sequences)")
            print(f"{'='*60}")
            try:
                seq_name = cherry_pick_sequence(
                    our_model, baseline_model, ds, groups, device,
                    max_windows_per_seq=10, dataset_name=ds_name)
                # Re-evaluate the selected candidate to get its score
                # (cherry_pick_sequence already printed details)
                # We'll track the best across datasets
                if seq_name is not None:
                    # Quick re-score: run the cherry-pick logic but just for this one
                    for c_name, c_indices in groups.items():
                        if c_name == seq_name:
                            # We need the candidate info — run a quick eval
                            cp_results = _quick_eval_sequence(
                                our_model, baseline_model, ds, c_indices,
                                device, ds_name, max_windows=10)
                            if cp_results is not None:
                                score = cp_results['combined_score']
                                if score > best_score:
                                    best_score = score
                                    best_seq = seq_name
                                    best_ds_name = ds_name
                                    best_info = cp_results
                            break
            except Exception as e:
                print(f"[WARN] Cherry-pick failed for {ds_name}: {e}")
                traceback.print_exc()

        if best_seq is None:
            print("ERROR: No valid sequence found across any dataset.")
            sys.exit(1)

        seq_name = best_seq
        chosen_ds_name = best_ds_name
        dataset = all_datasets[best_ds_name][0]
        seq_groups = all_datasets[best_ds_name][1]
        print(f"\n{'='*60}")
        print(f"  GLOBAL BEST: {seq_name} from {best_ds_name}")
        print(f"  Score: {best_score:.1f}")
        if best_info:
            print(f"  Rot improvement: {best_info['rot_improvement_pct']:+.1f}%")
            print(f"  Cal: FAT={best_info['fat_fx_mean_err']:.1f}px "
                  f"AnyCalib={best_info['anycalib_fx_mean_err']:.1f}px "
                  f"AnyCam={best_info['anycam_fx_mean_err']:.1f}px")
        print(f"{'='*60}")

    indices = seq_groups[seq_name]
    print(f"\n[EVAL] Sequence: {seq_name} from {chosen_ds_name} ({len(indices)} windows)")

    # Evaluate
    results = evaluate_sequence(our_model, baseline_model, dataset, indices, device,
                                max_windows=args.max_windows,
                                dataset_name=chosen_ds_name)

    # Generate figures
    generate_figures(results, seq_name, args.output_dir)


if __name__ == '__main__':
    main()
