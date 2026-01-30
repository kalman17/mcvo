#!/usr/bin/env python3
"""
Pose Estimation Metrics for Camera Pose Evaluation

This module implements metrics for evaluating camera pose predictions:
- Rotation error (SO3 geodesic distance via Lie group)
- Translation direction error (angular error between translation vectors)

Author: AI Assistant for Kalman's Master's Thesis
Date: October 16, 2025
"""

import numpy as np
from typing import Union, List, Tuple
import torch


def rotation_matrix_from_pose(pose: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """
    Extract 3x3 rotation matrix from 4x4 pose matrix.
    
    Args:
        pose: 4x4 pose matrix (numpy or torch)
        
    Returns:
        3x3 rotation matrix
    """
    if isinstance(pose, torch.Tensor):
        return pose[:3, :3]
    else:
        return pose[:3, :3]


def translation_vector_from_pose(pose: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """
    Extract translation vector from 4x4 pose matrix.
    
    Args:
        pose: 4x4 pose matrix (numpy or torch)
        
    Returns:
        3D translation vector
    """
    if isinstance(pose, torch.Tensor):
        return pose[:3, 3]
    else:
        return pose[:3, 3]


def rotation_error_degrees(R_pred: np.ndarray, R_gt: np.ndarray) -> float:
    """
    Compute rotation error in degrees using SO(3) geodesic distance.
    
    This computes the angle of rotation needed to align R_pred with R_gt
    using the Lie group distance metric.
    
    Args:
        R_pred: 3x3 predicted rotation matrix
        R_gt: 3x3 ground truth rotation matrix
        
    Returns:
        Rotation error in degrees
        
    Formula:
        R_diff = R_gt^T @ R_pred
        angle = arccos((trace(R_diff) - 1) / 2)
    """
    # Compute relative rotation
    R_diff = R_gt.T @ R_pred
    
    # Compute trace
    trace = np.trace(R_diff)
    
    # Compute angle via Lie algebra
    # Clip to handle numerical errors
    cos_angle = (trace - 1.0) / 2.0
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    
    angle_rad = np.arccos(cos_angle)
    angle_deg = np.degrees(angle_rad)
    
    return angle_deg


def translation_direction_error_degrees(t_pred: np.ndarray, t_gt: np.ndarray, 
                                       eps: float = 1e-8) -> float:
    """
    Compute angular error between translation direction vectors.
    
    This measures the angle between the predicted and ground truth translation
    directions, ignoring the magnitude.
    
    Args:
        t_pred: 3D predicted translation vector
        t_gt: 3D ground truth translation vector
        eps: Small value to avoid division by zero
        
    Returns:
        Angular error in degrees
        
    Formula:
        angle = arccos(dot(t_pred_normalized, t_gt_normalized))
    """
    # Compute norms
    norm_pred = np.linalg.norm(t_pred)
    norm_gt = np.linalg.norm(t_gt)
    
    # Handle zero translation (no movement)
    if norm_pred < eps or norm_gt < eps:
        return 0.0
    
    # Normalize vectors
    t_pred_norm = t_pred / norm_pred
    t_gt_norm = t_gt / norm_gt
    
    # Compute cosine of angle
    cos_angle = np.dot(t_pred_norm, t_gt_norm)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    
    # Compute angle
    angle_rad = np.arccos(cos_angle)
    angle_deg = np.degrees(angle_rad)
    
    return angle_deg


def translation_magnitude_error(t_pred: np.ndarray, t_gt: np.ndarray) -> float:
    """
    Compute Euclidean distance between translation vectors.

    This measures the magnitude difference between predicted and ground truth
    translation vectors, regardless of direction.

    Args:
        t_pred: 3D predicted translation vector
        t_gt: 3D ground truth translation vector

    Returns:
        Euclidean distance (magnitude error) in same units as input vectors
    """
    return np.linalg.norm(t_pred - t_gt)


def se3_distance(pose_pred: np.ndarray, pose_gt: np.ndarray) -> float:
    """
    Compute SE(3) distance between two pose matrices using Frobenius norm.

    This measures the overall difference between two pose matrices by computing
    the Frobenius norm of their difference, which accounts for both rotation
    and translation differences in a single metric.

    Args:
        pose_pred: 4x4 predicted pose matrix
        pose_gt: 4x4 ground truth pose matrix

    Returns:
        Frobenius norm ||T_pred - T_gt||_F

    Formula:
        dist = sqrt(sum((T_pred - T_gt)^2))
    """
    return np.linalg.norm(pose_pred - pose_gt, ord='fro')


def pose_error(pose_pred: np.ndarray, pose_gt: np.ndarray) -> Tuple[float, float, float, float]:
    """
    Compute comprehensive pose errors.

    Args:
        pose_pred: 4x4 predicted pose matrix
        pose_gt: 4x4 ground truth pose matrix

    Returns:
        Tuple of (se3_dist, rotation_error_deg, translation_magnitude, translation_direction_deg)
    """
    # Extract rotation and translation
    R_pred = rotation_matrix_from_pose(pose_pred)
    R_gt = rotation_matrix_from_pose(pose_gt)
    t_pred = translation_vector_from_pose(pose_pred)
    t_gt = translation_vector_from_pose(pose_gt)

    # Compute all errors
    se3_dist = se3_distance(pose_pred, pose_gt)
    rot_err = rotation_error_degrees(R_pred, R_gt)
    trans_mag = translation_magnitude_error(t_pred, t_gt)
    trans_dir = translation_direction_error_degrees(t_pred, t_gt)

    return se3_dist, rot_err, trans_mag, trans_dir


def batch_pose_errors(poses_pred: np.ndarray, poses_gt: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute pose errors for a batch of poses.

    Args:
        poses_pred: (N, 4, 4) array of predicted poses
        poses_gt: (N, 4, 4) array of ground truth poses

    Returns:
        Tuple of (se3_distances, rotation_errors, translation_magnitudes, translation_directions)
        as numpy arrays of shape (N,)
    """
    n = poses_pred.shape[0]
    se3_dists = np.zeros(n)
    rot_errors = np.zeros(n)
    trans_mags = np.zeros(n)
    trans_dirs = np.zeros(n)

    for i in range(n):
        se3_dists[i], rot_errors[i], trans_mags[i], trans_dirs[i] = pose_error(poses_pred[i], poses_gt[i])

    return se3_dists, rot_errors, trans_mags, trans_dirs


def compute_error_statistics(errors: np.ndarray) -> dict:
    """
    Compute statistics for error values.
    
    Args:
        errors: Array of error values
        
    Returns:
        Dictionary with statistics (mean, median, std, min, max, percentiles)
    """
    return {
        'mean': float(np.mean(errors)),
        'median': float(np.median(errors)),
        'std': float(np.std(errors)),
        'min': float(np.min(errors)),
        'max': float(np.max(errors)),
        'p25': float(np.percentile(errors, 25)),
        'p75': float(np.percentile(errors, 75)),
        'p90': float(np.percentile(errors, 90)),
        'p95': float(np.percentile(errors, 95)),
    }


def relative_pose_from_absolute(pose1: np.ndarray, pose2: np.ndarray) -> np.ndarray:
    """
    Compute relative pose from pose1 to pose2.
    
    Args:
        pose1: 4x4 camera-to-world pose at frame 1
        pose2: 4x4 camera-to-world pose at frame 2
        
    Returns:
        4x4 relative pose (transformation from camera1 to camera2)
        
    Formula:
        relative_pose = inv(pose2) @ pose1
    """
    return np.linalg.inv(pose2) @ pose1


def accumulate_trajectory(relative_poses: np.ndarray, start_pose: np.ndarray = None) -> np.ndarray:
    """
    Accumulate relative poses to build a full trajectory.
    
    Given a sequence of relative poses (e.g., 0→1, 1→2, 2→3), this function
    accumulates them to get absolute poses for all frames.
    
    Args:
        relative_poses: Array of shape (N, 4, 4) containing relative poses
                       relative_poses[i] is the transformation from frame i to frame i+1
        start_pose: Optional 4x4 starting pose (default: identity)
        
    Returns:
        Array of shape (N+1, 4, 4) containing absolute poses
        trajectory[0] is the starting pose
        trajectory[i+1] = trajectory[i] @ inv(relative_poses[i])
    """
    if start_pose is None:
        start_pose = np.eye(4)
    
    num_poses = relative_poses.shape[0]
    trajectory = np.zeros((num_poses + 1, 4, 4), dtype=relative_poses.dtype)
    trajectory[0] = start_pose
    
    for i in range(num_poses):
        # Accumulate: next_pose = current_pose @ inv(relative_pose)
        trajectory[i + 1] = trajectory[i] @ np.linalg.inv(relative_poses[i])
    
    return trajectory


if __name__ == "__main__":
    # Simple unit tests
    print("Testing pose metrics...")
    
    # Test 1: Identity rotation should give 0 error
    R_identity = np.eye(3)
    err = rotation_error_degrees(R_identity, R_identity)
    print(f"Test 1 - Identity rotation error: {err:.6f} degrees (expected: 0.0)")
    assert abs(err) < 1e-6, "Identity rotation should have 0 error"
    
    # Test 2: 90 degree rotation around Z axis
    R_90z = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
    err = rotation_error_degrees(R_90z, R_identity)
    print(f"Test 2 - 90° rotation error: {err:.6f} degrees (expected: ~90.0)")
    assert abs(err - 90.0) < 0.1, "90 degree rotation should give ~90 degrees error"
    
    # Test 3: 180 degree rotation around Z axis
    R_180z = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=float)
    err = rotation_error_degrees(R_180z, R_identity)
    print(f"Test 3 - 180° rotation error: {err:.6f} degrees (expected: ~180.0)")
    assert abs(err - 180.0) < 0.1, "180 degree rotation should give ~180 degrees error"
    
    # Test 4: Parallel translation vectors
    t1 = np.array([1.0, 0.0, 0.0])
    t2 = np.array([2.0, 0.0, 0.0])
    err = translation_direction_error_degrees(t1, t2)
    print(f"Test 4 - Parallel translation error: {err:.6f} degrees (expected: 0.0)")
    assert abs(err) < 1e-6, "Parallel translations should have 0 direction error"
    
    # Test 5: Perpendicular translation vectors
    t1 = np.array([1.0, 0.0, 0.0])
    t2 = np.array([0.0, 1.0, 0.0])
    err = translation_direction_error_degrees(t1, t2)
    print(f"Test 5 - Perpendicular translation error: {err:.6f} degrees (expected: ~90.0)")
    assert abs(err - 90.0) < 0.1, "Perpendicular translations should give ~90 degrees"
    
    # Test 6: Opposite translation vectors
    t1 = np.array([1.0, 0.0, 0.0])
    t2 = np.array([-1.0, 0.0, 0.0])
    err = translation_direction_error_degrees(t1, t2)
    print(f"Test 6 - Opposite translation error: {err:.6f} degrees (expected: ~180.0)")
    assert abs(err - 180.0) < 0.1, "Opposite translations should give ~180 degrees"
    
    # Test 7: Translation magnitude error
    t1 = np.array([1.0, 0.0, 0.0])
    t2 = np.array([2.0, 0.0, 0.0])
    err = translation_magnitude_error(t1, t2)
    print(f"Test 7 - Translation magnitude error: {err:.6f} (expected: 1.0)")
    assert abs(err - 1.0) < 1e-6, "Translation magnitude error should be 1.0"
    
    # Test 8: Translation magnitude error with different directions
    t1 = np.array([1.0, 0.0, 0.0])
    t2 = np.array([0.0, 1.0, 0.0])
    err = translation_magnitude_error(t1, t2)
    print(f"Test 8 - Translation magnitude error (perpendicular): {err:.6f} (expected: ~1.414)")
    assert abs(err - np.sqrt(2.0)) < 1e-6, "Translation magnitude error should be sqrt(2)"
    
    # Test 9: Trajectory accumulation
    rel_poses = np.array([
        np.eye(4),  # Identity (no movement)
        np.eye(4),  # Identity (no movement)
    ])
    traj = accumulate_trajectory(rel_poses)
    print(f"Test 9 - Trajectory accumulation: shape {traj.shape} (expected: (3, 4, 4))")
    assert traj.shape == (3, 4, 4), "Trajectory should have shape (3, 4, 4)"
    assert np.allclose(traj[0], np.eye(4)), "First pose should be identity"
    assert np.allclose(traj[1], np.eye(4)), "Second pose should be identity"
    assert np.allclose(traj[2], np.eye(4)), "Third pose should be identity"
    
    print("\n✓ All tests passed!")

