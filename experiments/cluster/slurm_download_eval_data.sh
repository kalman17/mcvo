#!/bin/bash
#SBATCH --job-name=dl_eval
#SBATCH --output=/storage/user/maka/logs/download_eval_%j.out
#SBATCH --error=/storage/user/maka/logs/download_eval_%j.err
#SBATCH --partition=NORMAL
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00

set -euo pipefail

DATA_ROOT="/storage/user/maka/eval_datasets"
REPO="/storage/user/maka/anycam"

echo "============================================"
echo "  Download Evaluation Datasets"
echo "  Host: $(hostname)"
echo "  Date: $(date)"
echo "============================================"

mkdir -p "$DATA_ROOT"
cd "$DATA_ROOT"

# ============================================================
# 1. MPI Sintel — GT poses + GT intrinsics
# ============================================================
SINTEL_DIR="$DATA_ROOT/Sintel"
if [ ! -d "$SINTEL_DIR/training/final" ]; then
    echo ""
    echo "=== Downloading MPI Sintel (training set) ==="
    mkdir -p "$SINTEL_DIR"
    cd "$SINTEL_DIR"

    # Training images (final pass)
    if [ ! -f "MPI-Sintel-training_images.zip" ]; then
        wget -q --show-progress http://files.is.tue.mpg.de/sintel/MPI-Sintel-training_images.zip
    fi
    unzip -q -n MPI-Sintel-training_images.zip
    rm -f MPI-Sintel-training_images.zip

    # Camera data (poses + intrinsics)
    if [ ! -f "MPI-Sintel-training_extras.zip" ]; then
        wget -q --show-progress http://files.is.tue.mpg.de/sintel/MPI-Sintel-training_extras.zip
    fi
    unzip -q -n MPI-Sintel-training_extras.zip
    rm -f MPI-Sintel-training_extras.zip

    echo "[OK] Sintel downloaded to $SINTEL_DIR"
else
    echo "[SKIP] Sintel already exists at $SINTEL_DIR"
fi

# ============================================================
# 2. TUM RGB-D — GT poses + fixed intrinsics
#    Download a small subset of sequences (3 sequences)
# ============================================================
TUM_DIR="$DATA_ROOT/TUM_RGBD"
if [ ! -d "$TUM_DIR/rgbd_dataset_freiburg1_desk" ]; then
    echo ""
    echo "=== Downloading TUM RGB-D (3 sequences) ==="
    mkdir -p "$TUM_DIR"
    cd "$TUM_DIR"

    # fr1/desk — office scene
    if [ ! -d "rgbd_dataset_freiburg1_desk" ]; then
        wget -q --show-progress https://cvg.cit.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_desk.tgz
        tar xzf rgbd_dataset_freiburg1_desk.tgz
        rm -f rgbd_dataset_freiburg1_desk.tgz
    fi

    # fr2/desk — larger office scene
    if [ ! -d "rgbd_dataset_freiburg2_desk" ]; then
        wget -q --show-progress https://cvg.cit.tum.de/rgbd/dataset/freiburg2/rgbd_dataset_freiburg2_desk.tgz
        tar xzf rgbd_dataset_freiburg2_desk.tgz
        rm -f rgbd_dataset_freiburg2_desk.tgz
    fi

    # fr3/long_office_household
    if [ ! -d "rgbd_dataset_freiburg3_long_office_household" ]; then
        wget -q --show-progress https://cvg.cit.tum.de/rgbd/dataset/freiburg3/rgbd_dataset_freiburg3_long_office_household.tgz
        tar xzf rgbd_dataset_freiburg3_long_office_household.tgz
        rm -f rgbd_dataset_freiburg3_long_office_household.tgz
    fi

    echo "[OK] TUM RGB-D downloaded to $TUM_DIR"
else
    echo "[SKIP] TUM RGB-D already exists at $TUM_DIR"
fi

# ============================================================
# 3. LightSpeed / DynPose-100k — GT poses only
#    Check if already available on cluster
# ============================================================
LIGHTSPEED_DIR="$DATA_ROOT/LightSpeed"
echo ""
echo "=== LightSpeed Dataset ==="
if [ -d "$LIGHTSPEED_DIR" ] && [ -f "$LIGHTSPEED_DIR/poses.pkl" ]; then
    echo "[SKIP] LightSpeed already exists at $LIGHTSPEED_DIR"
else
    echo "[WARN] LightSpeed dataset not found at $LIGHTSPEED_DIR"
    echo "       This dataset requires manual setup (poses.pkl + frames-24fps/)"
    echo "       Check if it exists elsewhere on the cluster and symlink it."
fi

# ============================================================
# 4. Precompute optical flows for Sintel (needed by AnyCam loader)
# ============================================================
if [ -d "$SINTEL_DIR/training/final" ]; then
    SINTEL_FLOW_DIR="$SINTEL_DIR/training/flow"
    if [ ! -d "$SINTEL_FLOW_DIR" ] || [ -z "$(ls -A "$SINTEL_FLOW_DIR" 2>/dev/null)" ]; then
        echo ""
        echo "=== Precomputing Sintel optical flows ==="
        echo "    (This may take a while without GPU)"
        echo "    NOTE: Sintel ships with GT flow — check if already present"

        # Sintel GT flow is included in the training extras download
        if [ -d "$SINTEL_DIR/training/flow" ] && [ -n "$(ls -A "$SINTEL_DIR/training/flow" 2>/dev/null)" ]; then
            echo "[OK] Sintel GT flow already present"
        else
            echo "[INFO] Sintel GT flow should be in training/flow/ from the extras zip."
            echo "       If missing, run: python $REPO/anycam/datasets/preprocess_flow.py"
        fi
    else
        echo "[SKIP] Sintel flow data already exists"
    fi
fi

echo ""
echo "============================================"
echo "  Download Complete"
echo "  Date: $(date)"
echo "============================================"
echo ""
echo "Dataset locations:"
echo "  Sintel:     $SINTEL_DIR"
echo "  TUM RGB-D:  $TUM_DIR"
echo "  LightSpeed: $LIGHTSPEED_DIR"
