"""Quick debug script to check predicted vs GT intrinsics on a KITTI sample."""
import sys
import os
import yaml
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.kitti_dataset import KITTIOdometryDataset
from experiments.benchmark_phase_c_checkpoints import (
    create_inference_model, load_phase_c_checkpoint
)


def main():
    data_root = "/storage/user/maka/eval_datasets"
    anycam_config = "/storage/user/maka/anycam/pretrained_models/anycam_seq8/training_config.yaml"
    phase_c_ckpt = "/storage/user/maka/train/phase_C_v3_h100/checkpoints/epoch_0005.pt"

    device = torch.device("cuda:0")

    # Load KITTI dataset
    kitti_path = os.path.join(data_root, "kitti_odom_color")
    dataset = KITTIOdometryDataset(
        data_path=kitti_path, image_size=336, frame_count=4, dilation=5
    )
    print(f"KITTI dataset: {len(dataset)} samples")

    # Get a sample
    sample = dataset[0]
    print(f"\nSample 0:")
    print(f"  imgs shape: {sample['imgs'].shape}")
    print(f"  projs shape: {sample['projs'].shape}")

    # GT intrinsics from projs
    projs = sample['projs']  # [N, 3, 3]
    gt_intrinsics = np.array([
        [projs[i, 0, 0], projs[i, 1, 1], projs[i, 0, 2], projs[i, 1, 2]]
        for i in range(projs.shape[0])
    ])
    gt_mean = gt_intrinsics.mean(axis=0)
    print(f"\n  GT intrinsics (per frame, at 336x336):")
    for i in range(gt_intrinsics.shape[0]):
        print(f"    Frame {i}: fx={gt_intrinsics[i,0]:.2f}, fy={gt_intrinsics[i,1]:.2f}, "
              f"cx={gt_intrinsics[i,2]:.2f}, cy={gt_intrinsics[i,3]:.2f}")
    print(f"  GT mean: fx={gt_mean[0]:.2f}, fy={gt_mean[1]:.2f}, cx={gt_mean[2]:.2f}, cy={gt_mean[3]:.2f}")
    print(f"  GT FoV: {2*np.degrees(np.arctan(336/(2*gt_mean[0]))):.1f} deg horizontal")

    # Create model using the benchmark's own factory function
    print("\nLoading model...")
    model = create_inference_model(anycam_config, device)
    load_phase_c_checkpoint(model, phase_c_ckpt, device)
    model.eval()

    # Prepare data
    data = {
        'imgs': torch.from_numpy(sample['imgs']).unsqueeze(0).to(device),
    }
    print(f"  Input imgs: {data['imgs'].shape}, range [{data['imgs'].min():.3f}, {data['imgs'].max():.3f}]")

    # Run forward
    print("\nRunning forward pass...")
    with torch.no_grad():
        output = model.forward_with_calibration_info(data)

    # Get predicted intrinsics
    batch_intr = output.get('intrinsics')
    fat_image_size = output.get('fat_image_size')
    print(f"\n  FAT image_size (H_ray, W_ray): {fat_image_size}")

    if batch_intr is not None:
        intr = batch_intr[0].cpu().numpy()
        print(f"  Raw FAT intrinsics (at ray resolution): fx={intr[0]:.2f}, fy={intr[1]:.2f}, cx={intr[2]:.2f}, cy={intr[3]:.2f}")

        if fat_image_size is not None:
            H_ray, W_ray = fat_image_size
            H_img, W_img = data['imgs'].shape[-2], data['imgs'].shape[-1]
            sx, sy = W_img / W_ray, H_img / H_ray
            scaled_intr = np.array([intr[0]*sx, intr[1]*sy, intr[2]*sx, intr[3]*sy])
            print(f"  Scale factors: sx={sx:.4f}, sy={sy:.4f}")
            print(f"  Scaled intrinsics (at img resolution): fx={scaled_intr[0]:.2f}, fy={scaled_intr[1]:.2f}, "
                  f"cx={scaled_intr[2]:.2f}, cy={scaled_intr[3]:.2f}")
            print(f"  Predicted FoV: {2*np.degrees(np.arctan(336/(2*scaled_intr[0]))):.1f} deg horizontal")

            # Compute MAPE
            mae = np.abs(scaled_intr - gt_mean)
            mape = np.abs(scaled_intr - gt_mean) / (np.abs(gt_mean) + 1e-8) * 100
            print(f"\n  fx MAE: {mae[0]:.2f} (pred={scaled_intr[0]:.2f} vs gt={gt_mean[0]:.2f})")
            print(f"  fy MAE: {mae[1]:.2f} (pred={scaled_intr[1]:.2f} vs gt={gt_mean[1]:.2f})")
            print(f"  fx MAPE: {mape[0]:.2f}%, fy MAPE: {mape[1]:.2f}%")
            print(f"  f_MAPE (avg): {(mape[0]+mape[1])/2:.2f}%")

    # Also check per-frame intrinsics
    per_frame = output.get('per_frame_intrinsics')
    if per_frame is not None:
        pf = per_frame[0].cpu().numpy()  # [N, 4]
        print(f"\n  Per-frame AnyCalib intrinsics (at ray resolution):")
        for i in range(pf.shape[0]):
            print(f"    Frame {i}: fx={pf[i,0]:.2f}, fy={pf[i,1]:.2f}, cx={pf[i,2]:.2f}, cy={pf[i,3]:.2f}")
            if fat_image_size is not None:
                H_ray, W_ray = fat_image_size
                H_img, W_img = 336, 336
                sx = W_img / W_ray
                print(f"             scaled to img: fx={pf[i,0]*sx:.2f}")

    # Also try standalone AnyCalib for comparison
    print(f"\n--- Standalone AnyCalib ---")
    from anycalib import AnyCalib as StandaloneAnyCalib
    anycalib = StandaloneAnyCalib(device=device)
    imgs = data['imgs'][0]  # [N, 3, H, W]
    focal = anycalib.predict_focal_length(imgs)
    f_px = float(focal[0].cpu())
    h, w = imgs.shape[-2], imgs.shape[-1]
    pred_standalone = np.array([f_px, f_px, w/2.0, h/2.0])
    print(f"  AnyCalib prediction: fx={pred_standalone[0]:.2f}")
    print(f"  AnyCalib FoV: {2*np.degrees(np.arctan(336/(2*pred_standalone[0]))):.1f} deg")
    mae_ac = np.abs(pred_standalone - gt_mean)
    mape_ac = np.abs(pred_standalone - gt_mean) / (np.abs(gt_mean) + 1e-8) * 100
    print(f"  f_MAPE: {(mape_ac[0]+mape_ac[1])/2:.2f}%")

    # Also check AnyCam baseline 32-candidate system
    print(f"\n--- AnyCam Baseline (32-candidate) ---")
    from experiments.benchmark_phase_c_checkpoints import create_baseline_model, _run_model_forward
    pretrained_path = "/storage/user/maka/anycam/pretrained_models/anycam_seq8/training_checkpoint_247500.pt"
    baseline_model = create_baseline_model(anycam_config, pretrained_path, device)
    baseline_model.eval()
    with torch.no_grad():
        bl_output = _run_model_forward(baseline_model, data, is_fat_model=False)
    bl_intr = bl_output.get('model_intrinsics')
    if bl_intr is not None:
        print(f"  AnyCam 32-candidate: fx={bl_intr[0]:.2f}, fy={bl_intr[1]:.2f}, cx={bl_intr[2]:.2f}, cy={bl_intr[3]:.2f}")
        bl_mape = np.abs(bl_intr - gt_mean) / (np.abs(gt_mean) + 1e-8) * 100
        print(f"  f_MAPE: {(bl_mape[0]+bl_mape[1])/2:.2f}%")
    else:
        print(f"  No intrinsics returned")


if __name__ == "__main__":
    main()
