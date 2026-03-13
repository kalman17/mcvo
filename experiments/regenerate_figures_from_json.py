"""
Regenerate thesis figures from saved figure_data.json with MCT labels.
No GPU needed — reads pre-computed data only.
"""

import json
import sys
from pathlib import Path

import numpy as np
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


def main():
    data_path = Path(__file__).resolve().parent.parent / 'thesis_results' / 'figures' / 'figure_data.json'
    output_dir = Path(__file__).resolve().parent.parent / 'thesis_results' / 'figures'
    latex_dir = Path(__file__).resolve().parent.parent / 'kalman-tum-thesis-latex-master' / 'figures'

    with open(data_path) as f:
        results = json.load(f)

    seq_name = results['sequence']
    gt_fx = results['gt_fx']

    C_OURS = '#2171b5'
    C_ANYCAM = '#e6550d'
    C_ANYCALIB = '#31a354'
    C_GT = '#333333'

    # ===== Figure 1: Focal length error histogram =====
    print("[FIGURE] Generating focal_length_error_histogram.png...")
    fig, ax = plt.subplots(figsize=(6, 3.5))

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

    n_windows = len(results['fat_fx'])
    window_idx = list(range(n_windows))

    # AnyCalib per-frame: 4 per window, scatter across sub-window positions
    n_per_window = len(results['anycalib_fx']) // n_windows  # should be 4
    ac_pos = []
    ac_vals = results['anycalib_fx']
    for i in range(n_windows):
        for j in range(n_per_window):
            ac_pos.append(i + j / n_per_window)

    ax.scatter(ac_pos, ac_vals, color=C_ANYCALIB, alpha=0.3, s=12, zorder=2,
               label='AnyCalib (per-frame)')

    # MCT aggregated (one per window)
    ax.plot(window_idx, results['fat_fx'], 'o-', color=C_OURS,
            markersize=3, linewidth=1.2, label='Ours (MCT)', zorder=3)

    # AnyCam (one per window)
    ax.plot(window_idx, results['anycam_fx'], 's-', color=C_ANYCAM,
            markersize=3, linewidth=1.0, alpha=0.7, label='AnyCam (32-cand.)', zorder=2)

    # GT line
    ax.axhline(y=gt_fx, color=C_GT, linewidth=1.5, linestyle='--',
               label=f'GT ($f_x$={gt_fx:.0f})', zorder=4)

    ax.set_xlabel('Window index (temporal order)')
    ax.set_ylabel('Focal length $f_x$ (pixels)')
    ax.set_title(f'Focal Length Over Time \u2014 {seq_name}')
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / 'focal_length_over_time.png')
    plt.close(fig)

    # Copy to latex figures directory
    import shutil
    for fname in ['focal_length_error_histogram.png', 'rotation_error_histogram.png',
                  'translation_error_histogram.png', 'focal_length_over_time.png']:
        src = output_dir / fname
        dst = latex_dir / fname
        shutil.copy2(src, dst)
        print(f"  Copied to {dst}")

    print(f"\n[DONE] 4 figures regenerated with MCT labels.")
    stats = results.get('stats', {})
    if stats:
        print(f"  Rot: Ours {stats.get('our_rot_mean', 0):.2f}° mean, "
              f"{stats.get('our_rot_median', 0):.2f}° median | "
              f"AnyCam {stats.get('base_rot_mean', 0):.2f}° mean, "
              f"{stats.get('base_rot_median', 0):.2f}° median")


if __name__ == '__main__':
    main()
