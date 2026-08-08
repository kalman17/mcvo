"""
Calibration Head Components

This module contains all components for:
1. DA3 Calibration Head - Works on AnyCalib's final scalar outputs
2. Feature Aggregation Transformer (FAT) - Works on AnyCalib's intermediate features
"""

# DA3 Components
from .camera_encoder import CameraEncoder
from .visual_camera_mixing import VisualCameraMixing
from .sequence_aggregation import SequenceCameraAggregation
from .camera_decoder import CameraDecoder
from .da3_calibration_head import DA3CalibrationHead

# FAT Components
from .feature_aggregation_transformer import (
    MultiframeCalibrationTransformer,
    MultiframeCalibrationTransformerV2,
    create_fat,
)
from .anycalib_with_fat import AnyCalibWithMCT, AnyCamWrapperWithFAT

__all__ = [
    # DA3
    'CameraEncoder',
    'VisualCameraMixing',
    'SequenceCameraAggregation',
    'CameraDecoder',
    'DA3CalibrationHead',
    # FAT
    'MultiframeCalibrationTransformer',
    'MultiframeCalibrationTransformerV2',
    'create_fat',
    'AnyCalibWithMCT',
    'AnyCamWrapperWithFAT',
]

