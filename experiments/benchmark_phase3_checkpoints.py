"""
Comprehensive benchmark script for Phase 3 checkpoints.

Evaluates pose and calibration accuracy on all saved checkpoints,
comparing:
1. Pose prediction vs GT
2. FAT calibration vs GT mean
3. GT mean calibration (baseline)
4. AnyCalib individual frame average vs GT mean

Usage:
    python experiments/benchmark_phase3_checkpoints.py \
        --checkpoint_dir experiments/fat_integration/phase3_training_v2/checkpoints \
        --objectron_videos /data/thesis/Objectron/videos \
        --objectron_gt /data/thesis/Objectron/processed_gt \
        --anycam_config pretrained_models/anycam_seq8/training_config.yaml \
        --num_samples 50 \
        --output_dir experiments/fat_integration/phase3_training_v2/benchmark_results
"""

import argparse
import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import cv2

# Import necessary components
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.train_fat_calibration import (
    ObjectronFATDataset,
    load_dataset_split,
)
from experiments.pose_metrics import (
    se3_distance,
    rotation_error_degrees,
    translation_magnitude_error,
    translation_direction_error_degrees,
)


def parse_args():
    parser = argparse.ArgumentParser(description='Benchmark Phase 3 checkpoints')
    parser.add_argument('--checkpoint_dir', type=str, required=True,
                        help='Directory containing checkpoint_epoch_*.pt files')
    parser.add_argument('--objectron_videos', type=str, required=True,
                        help='Path to Objectron videos directory')
    parser.add_argument('--objectron_gt', type=str, required=True,
                        help='Path to Objectron GT directory')
    parser.add_argument('--anycam_config', type=str, required=True,
                        help='Path to AnyCam config YAML')
    parser.add_argument('--split_file', type=str,
                        default='experiments/objectron_split.json',
                        help='Path to dataset split JSON')
    parser.add_argument('--num_samples', type=int, default=50,
                        help='Number of sequences to benchmark per checkpoint')
    parser.add_argument('--max_ahead', type=int, default=3,
                        help='Max ahead parameter (frames per sequence)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for results')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to run on')
    return parser.parse_args()


def load_gt_data(gt_path: Path) -> Tuple[List, List]:
    """
    Load ground truth poses and intrinsics from JSON file.

    Returns:
        poses_per_frame: List of 4x4 numpy arrays
        intrinsics_per_frame: List of 4-element arrays [fx, fy, cx, cy]
    """
    with open(gt_path, 'r') as f:
        data = json.load(f)

    # Load poses (4x4 matrices)
    poses = []
    if 'poses' in data:
        for pose_flat in data['poses']:
            pose_4x4 = np.array(pose_flat, dtype=np.float32).reshape(4, 4)
            poses.append(pose_4x4)

    # Load intrinsics (3x3 matrices flattened as [fx, 0, cx, 0, fy, cy, 0, 0, 1])
    intrinsics = []
    if 'intrinsics_per_frame' in data:
        for K_flat in data['intrinsics_per_frame']:
            fx = K_flat[0]
            fy = K_flat[4]
            cx = K_flat[2]
            cy = K_flat[5]
            intrinsics.append(np.array([fx, fy, cx, cy], dtype=np.float32))

    return poses, intrinsics


def compute_intrinsics_error(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    """
    Compute intrinsics errors (MAE and MAPE).

    Args:
        pred: [fx, fy, cx, cy]
        gt: [fx, fy, cx, cy]

    Returns:
        Dictionary with fx_mae, fy_mae, cx_mae, cy_mae, fx_mape, fy_mape
    """
    mae = np.abs(pred - gt)
    mape = np.abs(pred - gt) / (np.abs(gt) + 1e-8) * 100

    return {
        'fx_mae': float(mae[0]),
        'fy_mae': float(mae[1]),
        'cx_mae': float(mae[2]),
        'cy_mae': float(mae[3]),
        'fx_mape': float(mape[0]),
        'fy_mape': float(mape[1]),
        'f_mape': float((mape[0] + mape[1]) / 2),  # Average focal length MAPE
    }


def get_gt_path_from_video(video_path: Path, gt_dir: Path) -> Path:
    """
    Map video file to GT JSON file.
    Video: batch-X_Y_video.MOV -> GT: batch-X_Y.json
    """
    stem = video_path.stem  # e.g., "batch-10_0_video"
    gt_name = stem.replace('_video', '') + '.json'  # e.g., "batch-10_0.json"
    return gt_dir / gt_name


def create_phase3_model(anycam_config_path: str, device: str):
    """
    Create AnyCamWrapperWithFATCalibration model for benchmarking.

    Args:
        anycam_config_path: Path to AnyCam config YAML
        device: Device string ('cuda:0' or 'cpu')

    Returns:
        AnyCamWrapperWithFATCalibration model
    """
    import yaml
    from experiments.models.anycam_wrapper_fat import AnyCamWrapperWithFATCalibration
    from experiments.models.anycalib_with_fat import AnyCalibWithFAT

    # Load AnyCam config
    with open(anycam_config_path, 'r') as f:
        full_config = yaml.safe_load(f)

    pose_predictor_config = full_config['model']['pose_predictor']
    depth_predictor_config = full_config['model']['depth_predictor']

    # Create FAT model (AnyCalibWithFAT)
    fat_config = {
        "embed_dim": 1024,  # DINOv2 vitL dimension
        "num_heads": 8,
        "num_layers": 2,
        "dropout": 0.1,
        "use_learnable_agg_token": False,
        "use_visual_conditioning": True,
        "visual_token_dim": 384,  # DINOv2-small
        "num_scales": 4,
    }

    fat_model = AnyCalibWithFAT(
        model_id="anycalib_pinhole",
        use_fat=True,
        fat_config=fat_config,
        use_dinov2_small=True,
        use_dinov2_full=False,
        freeze_backbone=True,
        freeze_decoder=True,
        freeze_calibrator=True,
    )
    fat_model = fat_model.to(device)

    # Create wrapper
    model = AnyCamWrapperWithFATCalibration(
        fat_model=fat_model,
        pose_predictor_config=pose_predictor_config,
        depth_predictor_config=depth_predictor_config,
        use_provided_depth=False,
        use_provided_flow=False,
    )

    return model


def benchmark_checkpoint(
    checkpoint_path: Path,
    test_dataset: ObjectronFATDataset,
    gt_dir: Path,
    anycam_config_path: Path,
    indices: np.ndarray,
    device: torch.device,
) -> Dict:
    """
    Benchmark a single checkpoint on pose and calibration accuracy.

    Returns:
        Dictionary with benchmark results
    """
    print(f"\n{'='*70}")
    print(f"[BENCHMARK] Evaluating checkpoint: {checkpoint_path.name}")
    print(f"{'='*70}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    epoch = checkpoint['epoch']

    # Create model
    model = create_phase3_model(
        anycam_config_path=str(anycam_config_path),
        device=str(device),
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model.to(device)

    # Use provided indices (same for all checkpoints)
    num_samples = len(indices)

    # Results storage
    pose_errors = []
    intrinsics_errors_fat = []
    intrinsics_errors_gt_mean = []
    intrinsics_errors_anycalib_mean = []

    print(f"[BENCHMARK] Evaluating {num_samples} sequences...")

    with torch.no_grad():
        for idx in tqdm(indices, desc=f"Epoch {epoch}"):
            try:
                batch = test_dataset[idx]

                # Get video path from dataset sequence
                seq = test_dataset.sequences[idx]
                video_path = Path(seq['video_path'])
                frame_indices = seq['frame_indices']

                # Get corresponding GT path
                gt_path = get_gt_path_from_video(video_path, gt_dir)
                if not gt_path.exists():
                    continue

                # Load GT poses and intrinsics for the specific frame indices
                gt_poses, gt_intrinsics = load_gt_data(gt_path)

                # Select GT for the frame indices used in this sequence
                if len(gt_poses) == 0 or len(gt_intrinsics) == 0:
                    continue

                # Get GT for selected frames
                gt_poses_seq = [gt_poses[i] for i in frame_indices if i < len(gt_poses)]
                gt_intrinsics_seq = [gt_intrinsics[i] for i in frame_indices if i < len(gt_intrinsics)]

                if len(gt_poses_seq) < 2 or len(gt_intrinsics_seq) < 2:
                    continue

                # Prepare input
                imgs = batch['imgs'].unsqueeze(0).to(device)  # [1, N, 3, H, W]
                data = {'imgs': imgs}

                # Forward pass
                output = model.forward_with_calibration_info(data)

                # Extract predictions
                pred_poses = output['pose_result']['poses']  # [1, N-1, 1, 4, 4] or [1, N-1, 4, 4]
                if pred_poses.dim() == 5:  # [B, F, C, 4, 4]
                    pred_poses = pred_poses[:, :, 0]  # Take first candidate [B, F, 4, 4]

                # Extract intrinsics from output
                batch_intrinsics = output['intrinsics']  # [1, 4] = [fx, fy, cx, cy]
                per_frame_intrinsics_anycalib = output['per_frame_intrinsics']  # [1, N, 4]

                # Convert to numpy
                pred_poses = pred_poses[0].cpu().numpy()  # [N-1, 4, 4]
                batch_intrinsics_np = batch_intrinsics[0].cpu().numpy()  # [4] = [fx, fy, cx, cy]
                per_frame_anycalib = per_frame_intrinsics_anycalib[0].cpu().numpy()  # [N, 4]

                # FAT prediction: [fx, fy, cx, cy]
                fat_pred = batch_intrinsics_np  # Already in correct format

                # GT mean intrinsics (average over sequence)
                gt_intrinsics_array = np.array(gt_intrinsics_seq, dtype=np.float32)  # [N, 4]
                gt_mean = gt_intrinsics_array.mean(axis=0)  # [4]

                # AnyCalib mean (already computed per-frame by model)
                anycalib_mean = per_frame_anycalib.mean(axis=0)  # [4]

                # Compute intrinsics errors
                # 1) FAT vs GT mean
                fat_error = compute_intrinsics_error(fat_pred, gt_mean)
                intrinsics_errors_fat.append(fat_error)

                # 2) GT mean vs GT mean (baseline = 0, but we compute variance)
                gt_variance = gt_intrinsics_array.std(axis=0)
                intrinsics_errors_gt_mean.append({
                    'fx_std': float(gt_variance[0]),
                    'fy_std': float(gt_variance[1]),
                    'cx_std': float(gt_variance[2]),
                    'cy_std': float(gt_variance[3]),
                })

                # 3) AnyCalib mean vs GT mean
                anycalib_error = compute_intrinsics_error(anycalib_mean, gt_mean)
                intrinsics_errors_anycalib_mean.append(anycalib_error)

                # Compute pose errors (for consecutive pairs)
                # Model predicts relative pose between consecutive frames
                for i in range(min(len(pred_poses), len(gt_poses_seq) - 1)):
                    pred_pose = pred_poses[i]  # [4, 4] - relative pose from frame i to frame i+1

                    # GT relative pose: T_i^(-1) * T_(i+1)
                    T_i = gt_poses_seq[i]
                    T_i_plus_1 = gt_poses_seq[i + 1]
                    gt_relative_pose = np.linalg.inv(T_i) @ T_i_plus_1

                    # Compute all 4 pose metrics
                    se3_dist = se3_distance(pred_pose, gt_relative_pose)
                    rot_error = rotation_error_degrees(pred_pose[:3, :3], gt_relative_pose[:3, :3])
                    trans_mag = translation_magnitude_error(pred_pose[:3, 3], gt_relative_pose[:3, 3])
                    trans_dir = translation_direction_error_degrees(pred_pose[:3, 3], gt_relative_pose[:3, 3])

                    pose_errors.append({
                        'se3_distance': float(se3_dist),
                        'rotation_deg': float(rot_error),
                        'translation_magnitude': float(trans_mag),
                        'translation_direction_deg': float(trans_dir),
                    })

            except Exception as e:
                print(f"[WARN] Sequence {idx} failed: {e}")
                import traceback
                traceback.print_exc()
                continue

    # Aggregate results
    def aggregate_metrics(errors: List[Dict], keys: List[str]) -> Dict:
        """Compute mean and median for each key."""
        if not errors:
            return {f'{k}_mean': 0.0 for k in keys} | {f'{k}_median': 0.0 for k in keys}

        result = {}
        for key in keys:
            values = [e[key] for e in errors if key in e]
            if values:
                result[f'{key}_mean'] = float(np.mean(values))
                result[f'{key}_median'] = float(np.median(values))
        return result

    # Pose metrics (all 4)
    pose_metric_keys = ['se3_distance', 'rotation_deg', 'translation_magnitude', 'translation_direction_deg']
    pose_metrics = aggregate_metrics(pose_errors, pose_metric_keys)

    # Intrinsics metrics
    intrinsics_keys = ['fx_mae', 'fy_mae', 'cx_mae', 'cy_mae', 'fx_mape', 'fy_mape', 'f_mape']
    fat_metrics = aggregate_metrics(intrinsics_errors_fat, intrinsics_keys)
    anycalib_metrics = aggregate_metrics(intrinsics_errors_anycalib_mean, intrinsics_keys)

    gt_variance_keys = ['fx_std', 'fy_std', 'cx_std', 'cy_std']
    gt_variance_metrics = aggregate_metrics(intrinsics_errors_gt_mean, gt_variance_keys)

    results = {
        'epoch': epoch,
        'num_samples': len(indices),
        'num_pose_errors': len(pose_errors),
        'pose_metrics': pose_metrics,
        'fat_calibration': fat_metrics,
        'anycalib_calibration': anycalib_metrics,
        'gt_variance': gt_variance_metrics,
    }

    # Print summary
    print(f"\n[RESULTS] Epoch {epoch} Summary:")
    print(f"  Pose Metrics:")
    print(f"    SE(3) distance:         {pose_metrics.get('se3_distance_mean', 0):.4f} (mean), {pose_metrics.get('se3_distance_median', 0):.4f} (median)")
    print(f"    Rotation error:         {pose_metrics.get('rotation_deg_mean', 0):.4f}° (mean), {pose_metrics.get('rotation_deg_median', 0):.4f}° (median)")
    print(f"    Translation magnitude:  {pose_metrics.get('translation_magnitude_mean', 0):.4f} (mean), {pose_metrics.get('translation_magnitude_median', 0):.4f} (median)")
    print(f"    Translation direction:  {pose_metrics.get('translation_direction_deg_mean', 0):.4f}° (mean), {pose_metrics.get('translation_direction_deg_median', 0):.4f}° (median)")
    print(f"  Calibration Metrics:")
    print(f"    FAT Focal MAPE:              {fat_metrics.get('f_mape_mean', 0):.2f}% (mean), {fat_metrics.get('f_mape_median', 0):.2f}% (median)")
    print(f"    AnyCalib Focal MAPE:         {anycalib_metrics.get('f_mape_mean', 0):.2f}% (mean), {anycalib_metrics.get('f_mape_median', 0):.2f}% (median)")
    print(f"  GT Variance (reference only):")
    print(f"    Focal std: {gt_variance_metrics.get('fx_std_mean', 0):.2f} (fx), {gt_variance_metrics.get('fy_std_mean', 0):.2f} (fy)")

    return results


def plot_benchmark_results(all_results: List[Dict], output_dir: Path):
    """
    Plot benchmark results across epochs.
    """
    epochs = [r['epoch'] for r in all_results]

    # Pose errors (all 4 metrics)
    se3_mean = [r['pose_metrics'].get('se3_distance_mean', 0) for r in all_results]
    se3_median = [r['pose_metrics'].get('se3_distance_median', 0) for r in all_results]
    rot_mean = [r['pose_metrics'].get('rotation_deg_mean', 0) for r in all_results]
    rot_median = [r['pose_metrics'].get('rotation_deg_median', 0) for r in all_results]
    trans_mag_mean = [r['pose_metrics'].get('translation_magnitude_mean', 0) for r in all_results]
    trans_mag_median = [r['pose_metrics'].get('translation_magnitude_median', 0) for r in all_results]
    trans_dir_mean = [r['pose_metrics'].get('translation_direction_deg_mean', 0) for r in all_results]
    trans_dir_median = [r['pose_metrics'].get('translation_direction_deg_median', 0) for r in all_results]

    # Calibration errors
    fat_mape_mean = [r['fat_calibration'].get('f_mape_mean', 0) for r in all_results]
    fat_mape_median = [r['fat_calibration'].get('f_mape_median', 0) for r in all_results]
    anycalib_mape_mean = [r['anycalib_calibration'].get('f_mape_mean', 0) for r in all_results]
    anycalib_mape_median = [r['anycalib_calibration'].get('f_mape_median', 0) for r in all_results]

    # Create plots (3x2 grid)
    fig, axes = plt.subplots(3, 2, figsize=(14, 15))

    # Plot 1: SE(3) distance
    axes[0, 0].plot(epochs, se3_mean, 'o-', label='Mean', linewidth=2, color='purple')
    axes[0, 0].plot(epochs, se3_median, 's-', label='Median', linewidth=2, color='purple', alpha=0.7)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('SE(3) Distance')
    axes[0, 0].set_title('Pose: SE(3) Distance (Frobenius Norm)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Rotation error
    axes[0, 1].plot(epochs, rot_mean, 'o-', label='Mean', linewidth=2, color='blue')
    axes[0, 1].plot(epochs, rot_median, 's-', label='Median', linewidth=2, color='blue', alpha=0.7)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Rotation Error (degrees)')
    axes[0, 1].set_title('Pose: Rotation Error (SO(3) Geodesic)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Translation magnitude
    axes[1, 0].plot(epochs, trans_mag_mean, 'o-', label='Mean', linewidth=2, color='green')
    axes[1, 0].plot(epochs, trans_mag_median, 's-', label='Median', linewidth=2, color='green', alpha=0.7)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Translation Magnitude')
    axes[1, 0].set_title('Pose: Translation Magnitude (Euclidean)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Translation direction
    axes[1, 1].plot(epochs, trans_dir_mean, 'o-', label='Mean', linewidth=2, color='red')
    axes[1, 1].plot(epochs, trans_dir_median, 's-', label='Median', linewidth=2, color='red', alpha=0.7)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Translation Direction Error (degrees)')
    axes[1, 1].set_title('Pose: Translation Direction Error')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    # Plot 5: Focal length MAPE (mean)
    axes[2, 0].plot(epochs, fat_mape_mean, 'o-', label='FAT', linewidth=2, color='darkblue')
    axes[2, 0].plot(epochs, anycalib_mape_mean, 's-', label='AnyCalib (per-frame avg)', linewidth=2, color='darkorange')
    axes[2, 0].set_xlabel('Epoch')
    axes[2, 0].set_ylabel('Focal Length MAPE (%)')
    axes[2, 0].set_title('Calibration: Focal Length MAPE (Mean)')
    axes[2, 0].legend()
    axes[2, 0].grid(True, alpha=0.3)

    # Plot 6: Focal length MAPE (median)
    axes[2, 1].plot(epochs, fat_mape_median, 'o-', label='FAT', linewidth=2, color='darkblue')
    axes[2, 1].plot(epochs, anycalib_mape_median, 's-', label='AnyCalib (per-frame avg)', linewidth=2, color='darkorange')
    axes[2, 1].set_xlabel('Epoch')
    axes[2, 1].set_ylabel('Focal Length MAPE (%)')
    axes[2, 1].set_title('Calibration: Focal Length MAPE (Median)')
    axes[2, 1].legend()
    axes[2, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'benchmark_across_epochs.png', dpi=150)
    plt.close()

    print(f"[PLOT] Saved benchmark plot: {output_dir / 'benchmark_across_epochs.png'}")


def main():
    args = parse_args()

    # Setup
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    checkpoint_dir = Path(args.checkpoint_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Benchmark configuration:")
    print(f"  Checkpoint dir: {checkpoint_dir}")
    print(f"  Num samples: {args.num_samples}")
    print(f"  Device: {device}")
    print(f"  Output dir: {output_dir}")

    # Find all checkpoint files (exclude latest_checkpoint.pt)
    checkpoint_files = sorted([
        f for f in checkpoint_dir.glob('checkpoint_epoch_*.pt')
    ], key=lambda x: int(x.stem.split('_')[-1]))

    if not checkpoint_files:
        print(f"[ERROR] No checkpoint files found in {checkpoint_dir}")
        return

    print(f"\n[INFO] Found {len(checkpoint_files)} checkpoints:")
    for ckpt in checkpoint_files:
        print(f"  - {ckpt.name}")

    # Load dataset split
    split = load_dataset_split(args.split_file)

    # Create test dataset
    test_dataset = ObjectronFATDataset(
        video_dir=args.objectron_videos,
        gt_dir=args.objectron_gt,
        video_indices=split['test'],
        max_ahead=args.max_ahead,
        phase=3,  # Phase 3 for pose evaluation
    )

    print(f"[DATASET] Test dataset: {len(test_dataset)} sequences")

    # Sample sequences ONCE (same for all checkpoints)
    num_samples = min(args.num_samples, len(test_dataset))
    np.random.seed(42)  # Fixed seed for reproducibility
    indices = np.random.choice(len(test_dataset), num_samples, replace=False)
    print(f"[BENCHMARK] Using fixed sample of {num_samples} sequences (seed=42) for all epochs")

    # Benchmark each checkpoint
    all_results = []
    gt_dir = Path(args.objectron_gt)

    for checkpoint_path in checkpoint_files:
        results = benchmark_checkpoint(
            checkpoint_path=checkpoint_path,
            test_dataset=test_dataset,
            gt_dir=gt_dir,
            anycam_config_path=Path(args.anycam_config),
            indices=indices,
            device=device,
        )
        all_results.append(results)

    # Save results
    results_path = output_dir / 'benchmark_results.json'
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n[SAVE] Results saved to: {results_path}")

    # Plot results
    plot_benchmark_results(all_results, output_dir)

    # Print summary table
    print(f"\n{'='*120}")
    print(f"[SUMMARY] Benchmark Results Across Epochs (Median Values)")
    print(f"{'='*120}")
    print(f"{'Epoch':<8} {'SE(3)':<10} {'Rot(°)':<10} {'TransMag':<12} {'TransDir(°)':<12} {'FAT MAPE(%)':<15} {'AnyCalib MAPE(%)':<15}")
    print(f"{'-'*120}")
    for r in all_results:
        epoch = r['epoch']
        se3 = r['pose_metrics'].get('se3_distance_median', 0)
        rot = r['pose_metrics'].get('rotation_deg_median', 0)
        trans_mag = r['pose_metrics'].get('translation_magnitude_median', 0)
        trans_dir = r['pose_metrics'].get('translation_direction_deg_median', 0)
        fat = r['fat_calibration'].get('f_mape_median', 0)
        anycalib = r['anycalib_calibration'].get('f_mape_median', 0)
        print(f"{epoch:<8} {se3:<10.4f} {rot:<10.4f} {trans_mag:<12.4f} {trans_dir:<12.4f} {fat:<15.2f} {anycalib:<15.2f}")
    print(f"{'='*120}\n")


if __name__ == '__main__':
    main()
