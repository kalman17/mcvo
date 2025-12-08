"""
Centralized dataset path configuration.

This module provides a single location to configure dataset paths for all experiments.
Paths can be overridden using environment variables for easy machine-specific configuration.

Usage:
    # Set environment variable (recommended):
    export DATASETS_ROOT=/home/kalmanm/Documents/thesis
    
    # Or set individual paths:
    export OBJECTRON_ROOT=/path/to/Objectron
    export LIGHTSPEED_ROOT=/path/to/lightspeed
    
    # In Python scripts:
    from experiments.dataset_paths import OBJECTRON_VIDEOS, LIGHTSPEED_ROOT
"""

from pathlib import Path
import os

# Base datasets root directory
# Can be overridden with DATASETS_ROOT environment variable
DEFAULT_DATASETS_ROOT = Path(
    os.environ.get("DATASETS_ROOT", "/home/kalmanm/Documents/thesis")
)

# Objectron dataset paths
# Can be overridden with OBJECTRON_ROOT environment variable
OBJECTRON_ROOT = Path(
    os.environ.get("OBJECTRON_ROOT", DEFAULT_DATASETS_ROOT / "Objectron")
)

OBJECTRON_VIDEOS = OBJECTRON_ROOT / "videos"
OBJECTRON_GT = OBJECTRON_ROOT / "processed_gt"
OBJECTRON_ANNOTATIONS = OBJECTRON_ROOT / "annotations"

# LightSpeed dataset path
# Can be overridden with LIGHTSPEED_ROOT environment variable
LIGHTSPEED_ROOT = Path(
    os.environ.get("LIGHTSPEED_ROOT", DEFAULT_DATASETS_ROOT / "dynpose-100k" / "lightspeed")
)

# AnyCam source root (for sys.path modifications)
# Can be overridden with ANYCAM_SRC_ROOT environment variable
ANYCAM_SRC_ROOT = Path(
    os.environ.get("ANYCAM_SRC_ROOT", "/home/kalmanm/git/masters/anycam-extension")
)

# Helper function to get paths as strings (for argparse defaults)
def get_objectron_videos() -> str:
    """Get Objectron videos directory as string."""
    return str(OBJECTRON_VIDEOS)

def get_objectron_gt() -> str:
    """Get Objectron ground truth directory as string."""
    return str(OBJECTRON_GT)

def get_objectron_annotations() -> str:
    """Get Objectron annotations directory as string."""
    return str(OBJECTRON_ANNOTATIONS)

def get_lightspeed_root() -> str:
    """Get LightSpeed dataset root directory as string."""
    return str(LIGHTSPEED_ROOT)

def get_anycam_src_root() -> str:
    """Get AnyCam source root directory as string."""
    return str(ANYCAM_SRC_ROOT)

