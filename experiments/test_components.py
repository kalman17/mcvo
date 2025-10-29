#!/usr/bin/env python3
"""
Test script to understand component input/output shapes individually.
"""

import torch
import sys
import os
sys.path.append('/home/kalman/TUM/thesis/anycam')

from experiments.train_pose_head_anycalib import AnyCamWrapperWithAnyCaLib, AnyCaLibBatchInference
from anycam.models import make_pose_predictor, make_depth_predictor
from anycam.common.image_processor import make_image_processor
from anycam.trainer import normalize_proj, make_proj_from_focal_length
import yaml

def test_component_shapes():
    """Test each component individually to understand shapes."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[TEST] Using device: {device}")
    
    # 1. Load config
    model_path = Path("pretrained_models/anycam_seq8")
    config_file = model_path / "training_config.yaml"
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    # 2. Test AnyCaLib
    print("\n[TEST 1] Testing AnyCaLib...")
    anycalib = AnyCaLibBatchInference(device)
    
    # Create dummy batch
    batch_size = 1
    num_frames = 4
    h, w = 480, 640
    dummy_images = torch.rand(batch_size, num_frames, 3, h, w, device=device)
    
    focal_lengths = anycalib.predict_focal_length(dummy_images)
    print(f"AnyCaLib output shape: {focal_lengths.shape}")
    print(f"AnyCaLib focal lengths: {focal_lengths}")
    
    # 3. Test pose predictor
    print("\n[TEST 2] Testing pose predictor...")
    pose_predictor = make_pose_predictor(config['model']['pose_predictor'])
    pose_predictor = pose_predictor.to(device)
    
    # Test with dummy data
    dummy_flow_occs = torch.rand(batch_size, num_frames, 3, h, w, device=device)
    dummy_depths = torch.rand(batch_size, num_frames, 1, h, w, device=device)
    
    pose_result = pose_predictor(
        images=dummy_images,
        flow_occs=dummy_flow_occs,
        depths=dummy_depths,
    )
    
    print(f"Pose result keys: {pose_result.keys()}")
    for key, value in pose_result.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")
        else:
            print(f"  {key}: {type(value)}")
    
    # 4. Test image processor
    print("\n[TEST 3] Testing image processor...")
    image_processor = make_image_processor(
        {"type": "flow_occlusion"}, 
        flow_model="unimatch",
        use_provided_flow=False,
        pair_mode="sequential"
    )
    
    images_ip_fwd, images_ip_bwd = image_processor(dummy_images * 2 - 1, data={})
    print(f"Image processor forward output: {images_ip_fwd.shape}")
    print(f"Image processor backward output: {images_ip_bwd.shape}")
    
    # 5. Test projection creation
    print("\n[TEST 4] Testing projection creation...")
    focal_length_normalized = focal_lengths / w
    proj_candidates = make_proj_from_focal_length(
        focal_length_normalized.unsqueeze(1),  # [B, 1]
        aspect_ratio=h/w
    )
    print(f"Projection candidates shape: {proj_candidates.shape}")
    
    # 6. Test flow induction
    print("\n[TEST 5] Testing flow induction...")
    from anycam.trainer import induce_flow_dist
    
    aligned_depths = dummy_depths.view(batch_size, num_frames, 1, 1, h, w)
    flow_occs_in = images_ip_fwd[:, :, 3:6]
    
    induced_flow, dist = induce_flow_dist(
        aligned_depths, 
        proj_candidates, 
        pose_result["poses"], 
        flow_occs_in[..., :2, :, :]
    )
    
    print(f"Induced flow shape: {induced_flow.shape}")
    print(f"Distance shape: {dist.shape}")
    
    # 7. Test loss function
    print("\n[TEST 6] Testing loss function...")
    from anycam.loss import make_loss
    
    loss_config = config['loss'][0].copy()
    loss_config['lambda_fwd_bwd_consistency'] = 0
    criterion = make_loss(loss_config)
    
    # Prepare pose_result for loss
    pose_result["flow_occs_in"] = flow_occs_in
    pose_result["aligned_depths"] = aligned_depths
    pose_result["induced_flow"] = induced_flow
    pose_result["dist"] = dist
    pose_result["proj_candidates"] = proj_candidates
    
    # Add focal_length_probs
    batch_size = induced_flow.shape[0]
    pose_result["focal_length_probs"] = torch.ones(batch_size, 1, device=device)
    
    print(f"Pose result for loss:")
    for key, value in pose_result.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")
    
    # Test loss computation
    try:
        loss, losses, extra_data = criterion({"pose_result": pose_result})
        print(f"Loss computation successful!")
        print(f"Total loss: {loss}")
        print(f"Loss components: {losses}")
    except Exception as e:
        print(f"Loss computation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    from pathlib import Path
    test_component_shapes()
