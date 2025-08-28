import argparse
import numpy as np
import torch
from PIL import Image
import os
import cv2

# Import AnyCalib from the anycalib module
from anycalib import AnyCalib

def extract_frames(video_path, num_frames=20):
    """Extract the first num_frames from the video."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file {video_path}")
    
    frames = []
    count = 0
    while cap.isOpened() and count < num_frames:
        ret, frame = cap.read()
        if not ret:
            break
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
        count += 1
    cap.release()
    print(f"Extracted {len(frames)} frames from {video_path}")
    return frames

def main():
    parser = argparse.ArgumentParser(description='Run AnyCalib on the first 20 frames of a video.')
    parser.add_argument('--video_path', type=str, required=True, help='Path to the input video')
    parser.add_argument('--model_id', type=str, default='anycalib_pinhole', 
                        choices=['anycalib_pinhole', 'anycalib_gen', 'anycalib_dist', 'anycalib_edit'],
                        help='Model ID to use for calibration')
    parser.add_argument('--cam_id', type=str, default='pinhole', 
                        help='Camera model ID for prediction (e.g., pinhole, radial:1, kb:4)')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'],
                        help='Device to run the model on')
    parser.add_argument('--num_frames', type=int, default=20, 
                        help='Number of frames to process from the video')
    args = parser.parse_args()

    # Set device
    dev = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {dev}')

    # Check if video file exists
    if not os.path.exists(args.video_path):
        raise FileNotFoundError(f'Video not found at {args.video_path}')

    # Extract frames from video
    frames = extract_frames(args.video_path, args.num_frames)
    if not frames:
        raise ValueError('No frames extracted from video')

    # Convert frames to batch tensor (B, 3, H, W)
    frame_tensors = []
    for frame in frames:
        frame_tensor = torch.tensor(frame, dtype=torch.float32, device=dev).permute(2, 0, 1) / 255
        frame_tensors.append(frame_tensor)
    batch_images = torch.stack(frame_tensors, dim=0)
    print(f'Prepared batch of {batch_images.shape[0]} frames with shape {batch_images.shape}')

    # Instantiate AnyCalib model
    model = AnyCalib(model_id=args.model_id).to(dev)
    print(f'Initialized AnyCalib with model_id: {args.model_id}')

    # Predict camera parameters for the batch
    output = model.predict(batch_images, cam_id=args.cam_id)
    print(f'Prediction completed for camera model: {args.cam_id} on {batch_images.shape[0]} frames')

    # Display focal length results for each frame
    intrinsics_list = output['intrinsics']
    print(f'Estimated Focal Lengths for each frame (using {args.cam_id} model):')
    for i, intrinsics in enumerate(intrinsics_list):
        if args.cam_id.startswith('simple_'):
            focal_length = intrinsics[0].item()  # Single focal length for simple models
            print(f'Frame {i+1:2d}: Focal Length = {focal_length:.2f}')
        else:
            fx, fy = intrinsics[0].item(), intrinsics[1].item()  # Separate fx, fy for other models
            print(f'Frame {i+1:2d}: Focal Length (fx, fy) = ({fx:.2f}, {fy:.2f})')

if __name__ == '__main__':
    main()
