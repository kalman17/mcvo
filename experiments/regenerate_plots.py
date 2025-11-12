#!/usr/bin/env python3
"""
Regenerate benchmark plots from existing JSON results with cleaned labels.

This script reads benchmark_results.json and regenerates:
- benchmark_comparison.png (with cleaned labels)
- maxahead_comparison.png (with "Look Ahead" terminology)
"""

import json
import re
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats


def extract_look_ahead(model_name):
    """Extract look ahead value from model name."""
    match = re.search(r'max_ahead=(\d+)', model_name)
    if match:
        return int(match.group(1))
    return None


def create_clean_label(model_name):
    """Create clean label from model name."""
    look_ahead = extract_look_ahead(model_name)
    if look_ahead is not None:
        return f"Look Ahead {look_ahead}"
    return model_name.replace("Experiment 2", "").strip()


def approximate_distribution_from_stats(stats_dict, n_samples=1000):
    """Create approximate distribution from statistics using normal approximation."""
    mean = stats_dict['mean']
    std = stats_dict['std']
    # Use truncated normal to respect min/max bounds
    a = (stats_dict['min'] - mean) / std if std > 0 else 0
    b = (stats_dict['max'] - mean) / std if std > 0 else 0
    samples = stats.truncnorm.rvs(a, b, loc=mean, scale=std, size=n_samples)
    return samples


def plot_comparison_from_json(json_path, save_dir):
    """Regenerate benchmark_comparison.png from JSON results."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Extract model data (exclude metadata)
    models = {k: v for k, v in data.items() if k != 'metadata'}
    
    # Create clean labels
    clean_labels = {k: create_clean_label(k) for k in models.keys()}
    
    # Extract look ahead values for sorting
    look_ahead_values = {}
    for model_name in models.keys():
        la = extract_look_ahead(model_name)
        if la is not None:
            look_ahead_values[model_name] = la
    
    # Sort models by look ahead value
    sorted_models = sorted(models.items(), key=lambda x: look_ahead_values.get(x[0], 999))
    
    # Prepare data
    colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink']
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Rotation error histogram (approximate from stats)
    for i, (model_name, model_data) in enumerate(sorted_models):
        rot_stats = model_data['rotation']
        samples = approximate_distribution_from_stats(rot_stats)
        axes[0, 0].hist(samples, bins=50, alpha=0.7, 
                       label=clean_labels[model_name], color=colors[i % len(colors)])
    axes[0, 0].set_xlabel('Rotation Error (degrees)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Rotation Error Distribution')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Translation error histogram
    for i, (model_name, model_data) in enumerate(sorted_models):
        trans_stats = model_data['translation']
        samples = approximate_distribution_from_stats(trans_stats)
        axes[0, 1].hist(samples, bins=50, alpha=0.7, 
                       label=clean_labels[model_name], color=colors[i % len(colors)])
    axes[0, 1].set_xlabel('Translation Direction Error (degrees)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Translation Error Distribution')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # CDF for rotation
    for i, (model_name, model_data) in enumerate(sorted_models):
        rot_stats = model_data['rotation']
        samples = approximate_distribution_from_stats(rot_stats)
        axes[0, 2].hist(samples, bins=100, cumulative=True, density=True, 
                        histtype='step', linewidth=2, label=clean_labels[model_name], 
                        color=colors[i % len(colors)])
    axes[0, 2].set_xlabel('Rotation Error (degrees)')
    axes[0, 2].set_ylabel('Cumulative Probability')
    axes[0, 2].set_title('Rotation Error CDF')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # CDF for translation
    for i, (model_name, model_data) in enumerate(sorted_models):
        trans_stats = model_data['translation']
        samples = approximate_distribution_from_stats(trans_stats)
        axes[1, 0].hist(samples, bins=100, cumulative=True, density=True,
                        histtype='step', linewidth=2, label=clean_labels[model_name], 
                        color=colors[i % len(colors)])
    axes[1, 0].set_xlabel('Translation Direction Error (degrees)')
    axes[1, 0].set_ylabel('Cumulative Probability')
    axes[1, 0].set_title('Translation Error CDF')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Box plots for rotation - use short labels
    # Create approximate distributions for box plots
    rot_box_data = []
    rot_labels = []
    for model_name, model_data in sorted_models:
        rot_stats = model_data['rotation']
        # Create approximate distribution for box plot
        samples = approximate_distribution_from_stats(rot_stats, n_samples=500)
        rot_box_data.append(samples)
        la = extract_look_ahead(model_name)
        rot_labels.append(str(la) if la is not None else clean_labels[model_name])
    
    # Create box plot
    bp = axes[1, 1].boxplot(rot_box_data, labels=rot_labels, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
        patch.set_alpha(0.7)
    
    axes[1, 1].set_xlabel('Look Ahead', fontsize=12)
    axes[1, 1].set_ylabel('Rotation Error (degrees)')
    axes[1, 1].set_title('Rotation Error Box Plot')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    # Box plots for translation
    trans_box_data = []
    trans_labels = []
    for model_name, model_data in sorted_models:
        trans_stats = model_data['translation']
        # Create approximate distribution for box plot
        samples = approximate_distribution_from_stats(trans_stats, n_samples=500)
        trans_box_data.append(samples)
        la = extract_look_ahead(model_name)
        trans_labels.append(str(la) if la is not None else clean_labels[model_name])
    
    bp = axes[1, 2].boxplot(trans_box_data, labels=trans_labels, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightgreen')
        patch.set_alpha(0.7)
    
    axes[1, 2].set_xlabel('Look Ahead', fontsize=12)
    axes[1, 2].set_ylabel('Translation Direction Error (degrees)')
    axes[1, 2].set_title('Translation Error Box Plot')
    axes[1, 2].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_path = save_dir / 'benchmark_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"[SAVE] Regenerated benchmark_comparison.png at {output_path}")
    plt.close()


def plot_maxahead_comparison_from_json(json_path, save_dir):
    """Regenerate maxahead_comparison.png from JSON results."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Extract exp2 models with look ahead values
    exp2_models = {}
    for model_name, model_data in data.items():
        if model_name == 'metadata':
            continue
        la = extract_look_ahead(model_name)
        if la is not None:
            exp2_models[la] = model_data
    
    if len(exp2_models) < 2:
        print("[WARN] Not enough models with look ahead values")
        return
    
    # Sort by look ahead
    sorted_look_aheads = sorted(exp2_models.keys())
    
    # Extract metrics
    rot_means = [exp2_models[la]['rotation']['mean'] for la in sorted_look_aheads]
    rot_medians = [exp2_models[la]['rotation']['median'] for la in sorted_look_aheads]
    trans_means = [exp2_models[la]['translation']['mean'] for la in sorted_look_aheads]
    trans_medians = [exp2_models[la]['translation']['median'] for la in sorted_look_aheads]
    
    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Rotation error vs look ahead
    axes[0, 0].plot(sorted_look_aheads, rot_means, 'o-', linewidth=2, markersize=8, label='Mean', color='blue')
    axes[0, 0].plot(sorted_look_aheads, rot_medians, 's--', linewidth=2, markersize=8, label='Median', color='green')
    axes[0, 0].set_xlabel('Look Ahead', fontsize=12)
    axes[0, 0].set_ylabel('Rotation Error (degrees)')
    axes[0, 0].set_title('Rotation Error vs Look Ahead')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_xticks(sorted_look_aheads)
    
    # Translation error vs look ahead
    axes[0, 1].plot(sorted_look_aheads, trans_means, 'o-', linewidth=2, markersize=8, label='Mean', color='blue')
    axes[0, 1].plot(sorted_look_aheads, trans_medians, 's--', linewidth=2, markersize=8, label='Median', color='green')
    axes[0, 1].set_xlabel('Look Ahead', fontsize=12)
    axes[0, 1].set_ylabel('Translation Error (degrees)')
    axes[0, 1].set_title('Translation Error vs Look Ahead')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_xticks(sorted_look_aheads)
    
    # Bar chart: Rotation error comparison
    x_pos = np.arange(len(sorted_look_aheads))
    width = 0.35
    axes[1, 0].bar(x_pos - width/2, rot_means, width, label='Mean', color='blue', alpha=0.7)
    axes[1, 0].bar(x_pos + width/2, rot_medians, width, label='Median', color='green', alpha=0.7)
    axes[1, 0].set_xlabel('Look Ahead', fontsize=12)
    axes[1, 0].set_ylabel('Rotation Error (degrees)')
    axes[1, 0].set_title('Rotation Error by Look Ahead')
    axes[1, 0].set_xticks(x_pos)
    axes[1, 0].set_xticklabels([str(la) for la in sorted_look_aheads])
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Bar chart: Translation error comparison
    axes[1, 1].bar(x_pos - width/2, trans_means, width, label='Mean', color='blue', alpha=0.7)
    axes[1, 1].bar(x_pos + width/2, trans_medians, width, label='Median', color='green', alpha=0.7)
    axes[1, 1].set_xlabel('Look Ahead', fontsize=12)
    axes[1, 1].set_ylabel('Translation Error (degrees)')
    axes[1, 1].set_title('Translation Error by Look Ahead')
    axes[1, 1].set_xticks(x_pos)
    axes[1, 1].set_xticklabels([str(la) for la in sorted_look_aheads])
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_path = save_dir / 'maxahead_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"[SAVE] Regenerated maxahead_comparison.png at {output_path}")
    plt.close()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Regenerate plots from JSON results")
    parser.add_argument("--json_path", type=str, 
                       default="experiments/pose_head_experiment_results/benchmark_results_maxahead_comparison/benchmark_results.json",
                       help="Path to benchmark_results.json")
    parser.add_argument("--save_dir", type=str, default=None,
                       help="Directory to save plots (default: same as JSON directory)")
    
    args = parser.parse_args()
    
    json_path = Path(args.json_path)
    if not json_path.exists():
        print(f"[ERROR] JSON file not found: {json_path}")
        return
    
    if args.save_dir is None:
        save_dir = json_path.parent
    else:
        save_dir = Path(args.save_dir)
    
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[INFO] Reading results from: {json_path}")
    print(f"[INFO] Saving plots to: {save_dir}")
    
    # Regenerate both plots
    plot_comparison_from_json(json_path, save_dir)
    plot_maxahead_comparison_from_json(json_path, save_dir)
    
    print("[DONE] All plots regenerated with cleaned labels!")


if __name__ == "__main__":
    main()

