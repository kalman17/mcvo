#!/usr/bin/env python3
"""
=============================================================================
DA3 Calibration Head Evaluation Script
=============================================================================

Purpose: Evaluate DA3 calibration head performance on calibration accuracy.

Metrics:
- Focal length error: |pred_fx - gt_fx| / gt_fx
- Principal point error: |pred_cx - gt_cx| / image_width
- Relative error to target mean (for Stage 1)

Author: AI Assistant for Kalman's Master's Thesis
Date: December 2025
"""

import sys
import argparse
import json
from pathlib import Path
from typing import Dict, List
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project paths
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "anycalib"))

# DA3 imports
from experiments.models.da3_calibration_head import DA3CalibrationHead
from experiments.train_calibration_head_da3_stage1 import DA3Stage1Dataset
from experiments.dataset_paths import get_objectron_videos, get_objectron_gt
from experiments.train_pose_head_anycalib import AnyCaLibBatchInference, load_dataset_split

print("[INIT] Imports successful")


def evaluate_calibration_accuracy(
    model: DA3CalibrationHead,
    dataloader: DataLoader,
    device: torch.device,
    use_visual_conditioning: bool = True,
) -> Dict:
    """
    Evaluate calibration accuracy.
    
    Returns dictionary with error metrics.
    """
    model.eval()
    
    all_errors = {
        'fx': [],
        'fy': [],
        'cx': [],
        'cy': [],
    }
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch_data in tqdm(dataloader, desc="Evaluating"):
            anycalib_preds = batch_data['anycalib_predictions'].to(device)  # [B, N, 4]
            gt_mean = batch_data['gt_mean_calibration'].to(device)  # [B, 1, 4]
            image_sizes = batch_data['image_size']
            
            B, N, _ = anycalib_preds.shape
            
            if use_visual_conditioning:
                # Need visual tokens (for Stage 2/3)
                if 'visual_tokens' in batch_data:
                    visual_tokens = batch_data['visual_tokens'].to(device)  # [B, N, D_vis]
                else:
                    # Create dummy visual tokens
                    visual_tokens = torch.zeros(B, N, model.vis_dim, device=device)
            else:
                # Stage 1: dummy visual tokens
                visual_tokens = torch.zeros(B, N, model.vis_dim, device=device)
            
            # DataLoader collates tuples as (tensor([H1,H2,...]), tensor([W1,W2,...]))
            H = image_sizes[0][0].item() if isinstance(image_sizes[0], torch.Tensor) else image_sizes[0]
            W = image_sizes[1][0].item() if isinstance(image_sizes[1], torch.Tensor) else image_sizes[1]
            
            pred_calibration = model(
                visual_tokens=visual_tokens,
                anycalib_predictions=anycalib_preds,
                image_size=(H, W),
                use_visual_conditioning=use_visual_conditioning
            )  # [B, 1, 4]
            
            # Compute errors
            pred_np = pred_calibration.cpu().numpy()  # [B, 1, 4]
            gt_np = gt_mean.cpu().numpy()  # [B, 1, 4]
            
            for b in range(B):
                pred = pred_np[b, 0]  # [4]
                gt = gt_np[b, 0]  # [4]
                
                # Relative errors
                rel_err_fx = abs(pred[0] - gt[0]) / (gt[0] + 1e-8)
                rel_err_fy = abs(pred[1] - gt[1]) / (gt[1] + 1e-8)
                rel_err_cx = abs(pred[2] - gt[2]) / (W + 1e-8)
                rel_err_cy = abs(pred[3] - gt[3]) / (H + 1e-8)
                
                all_errors['fx'].append(float(rel_err_fx))
                all_errors['fy'].append(float(rel_err_fy))
                all_errors['cx'].append(float(rel_err_cx))
                all_errors['cy'].append(float(rel_err_cy))
                
                all_predictions.append(pred.tolist())
                all_targets.append(gt.tolist())
    
    # Compute statistics
    stats = {}
    for key, errors in all_errors.items():
        errors = np.array(errors)
        stats[key] = {
            'mean': float(np.mean(errors)),
            'median': float(np.median(errors)),
            'std': float(np.std(errors)),
            'p90': float(np.percentile(errors, 90)),
        }
    
    # Overall relative error
    all_errors_flat = np.concatenate([np.array(errors) for errors in all_errors.values()])
    overall_error = {
        'mean': float(np.mean(all_errors_flat)),
        'median': float(np.median(all_errors_flat)),
    }
    
    results = {
        'focal_length_fx': stats['fx'],
        'focal_length_fy': stats['fy'],
        'principal_point_cx': stats['cx'],
        'principal_point_cy': stats['cy'],
        'overall_relative_error': overall_error,
        'predictions': all_predictions,
        'targets': all_targets,
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate DA3 Calibration Head")
    
    # Model arguments
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--stage", type=int, choices=[1, 2], default=1,
                       help="Training stage (1 or 2)")
    
    # Dataset arguments
    parser.add_argument("--objectron_videos", type=str, default=get_objectron_videos(),
                       help="Objectron videos directory")
    parser.add_argument("--objectron_gt", type=str, default=get_objectron_gt(),
                       help="Objectron GT directory")
    parser.add_argument("--split_file", type=str, default="experiments/objectron_split.json",
                       help="Dataset split file")
    parser.add_argument("--num_frames", type=int, default=2,
                       help="Number of frames per sequence")
    
    # Model arguments
    parser.add_argument("--vis_dim", type=int, default=768,
                       help="Visual token dimension")
    parser.add_argument("--cam_dim", type=int, default=256,
                       help="Camera token dimension")
    parser.add_argument("--hidden_dim", type=int, default=128,
                       help="Hidden layer dimension")
    
    # Evaluation arguments
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Batch size")
    parser.add_argument("--device", type=str, default="cuda:0",
                       help="Device to use")
    parser.add_argument("--output_file", type=str, default=None,
                       help="Output JSON file path")
    
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[INIT] Using device: {device}")
    
    # Load dataset split
    if Path(args.split_file).exists():
        split_data = load_dataset_split(args.split_file)
        test_indices = split_data.get('test', split_data.get('test_indices', []))
    else:
        raise FileNotFoundError(f"Split file not found: {args.split_file}")
    
    # Initialize AnyCalib
    anycalib_inference = AnyCaLibBatchInference(device=device)
    
    # Create dataset
    if args.stage == 1:
        from experiments.train_calibration_head_da3_stage1 import DA3Stage1Dataset
        dataset = DA3Stage1Dataset(
            videos_dir=args.objectron_videos,
            gt_dir=args.objectron_gt,
            anycalib_model=anycalib_inference,
            num_frames=args.num_frames,
            video_indices=test_indices,
            require_gt=True,
        )
    else:
        # Stage 2 needs visual tokens - would need different dataset
        raise NotImplementedError("Stage 2 evaluation dataset not yet implemented")
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    
    print(f"[DATASET] Loaded {len(dataset)} test sequences")
    
    # Create model
    model = DA3CalibrationHead(
        vis_dim=args.vis_dim,
        cam_dim=args.cam_dim,
        hidden_dim=args.hidden_dim,
        num_mixing_layers=2,
    ).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
    
    print(f"[LOAD] Model loaded from {args.checkpoint}")
    
    # Evaluate
    print(f"\n[EVAL] Evaluating calibration accuracy...")
    results = evaluate_calibration_accuracy(
        model=model,
        dataloader=dataloader,
        device=device,
        use_visual_conditioning=(args.stage == 2),
    )
    
    # Print results
    print(f"\n[RESULTS] Calibration Accuracy:")
    print(f"  Focal Length fx: mean={results['focal_length_fx']['mean']:.4f}, median={results['focal_length_fx']['median']:.4f}")
    print(f"  Focal Length fy: mean={results['focal_length_fy']['mean']:.4f}, median={results['focal_length_fy']['median']:.4f}")
    print(f"  Principal Point cx: mean={results['principal_point_cx']['mean']:.4f}, median={results['principal_point_cx']['median']:.4f}")
    print(f"  Principal Point cy: mean={results['principal_point_cy']['mean']:.4f}, median={results['principal_point_cy']['median']:.4f}")
    print(f"  Overall Relative Error: mean={results['overall_relative_error']['mean']:.4f}, median={results['overall_relative_error']['median']:.4f}")
    
    # Save results
    if args.output_file:
        output_path = Path(args.output_file)
    else:
        checkpoint_dir = Path(args.checkpoint).parent
        output_path = checkpoint_dir / "calibration_accuracy.json"
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n[SAVE] Results saved to {output_path}")


if __name__ == "__main__":
    main()

