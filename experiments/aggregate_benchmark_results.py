"""
Aggregate per-epoch Phase C benchmark results into evolution plots and summary CSV.

Reads results.json files from per-epoch subdirectories and generates:
  - metrics_vs_epoch.png: rotation, translation, calibration error vs epoch
  - improvement_vs_epoch.png: percentage improvement over AnyCam baseline
  - benchmark_summary.csv: one row per epoch with all metrics

Usage:
    python experiments/aggregate_benchmark_results.py \
        --results_dir /storage/user/maka/train/phase_C/benchmark_results
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def load_all_results(results_dir: Path) -> List[Dict]:
    """Load results.json from each epoch_XXXX/ subdirectory, sorted by epoch."""
    results = []
    for epoch_dir in sorted(results_dir.glob('epoch_*')):
        results_path = epoch_dir / 'results.json'
        if results_path.exists():
            with open(results_path, 'r') as f:
                results.append(json.load(f))
    results.sort(key=lambda r: r.get('epoch', 0))
    return results


def load_baseline(results_dir: Path) -> Dict:
    """Load cached baseline results."""
    cache_path = results_dir / 'baseline_cache.json'
    if cache_path.exists():
        with open(cache_path, 'r') as f:
            return json.load(f)
    return {}


def plot_metrics_vs_epoch(results: List[Dict], baseline: Dict, output_dir: Path):
    """Multi-panel plot: metrics vs epoch for each dataset."""
    if not results:
        return

    epochs = [r['epoch'] for r in results]
    ds_names = list(results[0].get('datasets', {}).keys())
    if not ds_names:
        return

    n_ds = len(ds_names)
    metrics_to_plot = [
        ('rotation_deg_mean', 'Rotation Error (degrees)', 'blue'),
        ('translation_direction_deg_mean', 'Translation Dir Error (degrees)', 'green'),
        ('se3_distance_mean', 'SE(3) Distance', 'purple'),
    ]

    fig, axes = plt.subplots(n_ds, len(metrics_to_plot), figsize=(6 * len(metrics_to_plot), 5 * n_ds), squeeze=False)

    for row, ds_name in enumerate(ds_names):
        for col, (metric_key, ylabel, color) in enumerate(metrics_to_plot):
            ax = axes[row, col]

            values = [
                r['datasets'].get(ds_name, {}).get('ours', {}).get(metric_key, float('nan'))
                for r in results
            ]
            median_key = metric_key.replace('_mean', '_median')
            medians = [
                r['datasets'].get(ds_name, {}).get('ours', {}).get(median_key, float('nan'))
                for r in results
            ]

            ax.plot(epochs, values, 'o-', label='Ours (mean)', linewidth=2, color=color)
            ax.plot(epochs, medians, 's--', label='Ours (median)', linewidth=2, color=color, alpha=0.6)

            if ds_name in baseline and metric_key in baseline[ds_name]:
                bl_val = baseline[ds_name][metric_key]
                ax.axhline(y=bl_val, color='red', linestyle=':', linewidth=2,
                           label=f'Baseline ({bl_val:.2f})')

            ax.set_xlabel('Epoch')
            ax.set_ylabel(ylabel)
            ax.set_title(f'{ds_name}: {ylabel}')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = output_dir / 'metrics_vs_epoch.png'
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[PLOT] Saved: {out_path}")


def plot_improvement_vs_epoch(results: List[Dict], baseline: Dict, output_dir: Path):
    """Plot percentage improvement over baseline vs epoch."""
    if not results or not baseline:
        return

    epochs = [r['epoch'] for r in results]
    ds_names = list(results[0].get('datasets', {}).keys())
    improvement_metrics = ['rotation_deg_mean', 'translation_direction_deg_mean']

    fig, axes = plt.subplots(1, len(improvement_metrics), figsize=(7 * len(improvement_metrics), 5))
    if len(improvement_metrics) == 1:
        axes = [axes]

    colors = plt.cm.Set1(np.linspace(0, 1, max(len(ds_names), 3)))

    for col, metric_key in enumerate(improvement_metrics):
        ax = axes[col]

        for ds_idx, ds_name in enumerate(ds_names):
            if ds_name not in baseline or metric_key not in baseline[ds_name]:
                continue
            bl_val = baseline[ds_name][metric_key]
            if bl_val == 0:
                continue

            improvements = []
            for r in results:
                our_val = r['datasets'].get(ds_name, {}).get('ours', {}).get(metric_key, bl_val)
                pct = (bl_val - our_val) / abs(bl_val) * 100
                improvements.append(pct)

            ax.plot(epochs, improvements, 'o-', label=ds_name, linewidth=2, color=colors[ds_idx])

        ax.axhline(y=0, color='gray', linestyle='--', linewidth=1)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Improvement over Baseline (%)')
        label = metric_key.replace('_mean', '').replace('_', ' ').title()
        ax.set_title(f'{label}: Improvement')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = output_dir / 'improvement_vs_epoch.png'
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[PLOT] Saved: {out_path}")


def write_summary_csv(results: List[Dict], baseline: Dict, output_dir: Path):
    """Write one row per epoch per dataset with all metrics to CSV."""
    if not results:
        return

    ds_names = list(results[0].get('datasets', {}).keys())
    metric_keys = [
        'rotation_deg_mean', 'rotation_deg_median',
        'translation_direction_deg_mean', 'translation_direction_deg_median',
        'se3_distance_mean', 'se3_distance_median',
        'translation_magnitude_mean', 'translation_magnitude_median',
        'f_mape_mean', 'f_mape_median',
    ]

    out_path = output_dir / 'benchmark_summary.csv'
    with open(out_path, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['epoch', 'dataset'] + metric_keys + ['num_sequences']
        writer.writerow(header)

        # Write baseline row for each dataset
        for ds_name in ds_names:
            if ds_name in baseline:
                row = ['baseline', ds_name]
                for key in metric_keys:
                    row.append(baseline[ds_name].get(key, ''))
                row.append(baseline[ds_name].get('num_sequences', ''))
                writer.writerow(row)

        # Write per-epoch rows
        for r in results:
            epoch = r['epoch']
            for ds_name in ds_names:
                ds = r['datasets'].get(ds_name, {})
                ours = ds.get('ours', {})
                row = [epoch, ds_name]
                for key in metric_keys:
                    row.append(ours.get(key, ''))
                row.append(ds.get('num_sequences', ''))
                writer.writerow(row)

    print(f"[CSV] Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description='Aggregate Phase C benchmark results')
    parser.add_argument('--results_dir', type=str, required=True,
                        help='Directory containing epoch_XXXX/results.json files and baseline_cache.json')
    args = parser.parse_args()

    results_dir = Path(args.results_dir)

    results = load_all_results(results_dir)
    baseline = load_baseline(results_dir)

    print(f"[INFO] Loaded {len(results)} epoch results, baseline for {len(baseline)} datasets")

    if not results:
        print("[ERROR] No results found")
        return

    plot_metrics_vs_epoch(results, baseline, results_dir)
    plot_improvement_vs_epoch(results, baseline, results_dir)
    write_summary_csv(results, baseline, results_dir)

    print(f"\n[DONE] Aggregated results saved to {results_dir}")


if __name__ == '__main__':
    main()
