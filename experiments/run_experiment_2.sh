#!/bin/bash
# =============================================================================
# Experiment 2 Runner Script
# =============================================================================
# 
# This script runs Experiment 2 (Multi-Frame Pose Prediction) with different
# configurations and automatically benchmarks the results against Experiment 1
# and the AnyCam baseline.
#
# Usage:
#   bash experiments/run_experiment_2.sh [mode]
#
# Modes:
#   test          - Quick test on 1-2 sequences, 10 epochs, max_ahead=3
#   small         - Train on 20 sequences, 30 epochs, max_ahead=3  
#   full          - Train on all sequences, 50 epochs, max_ahead=3
#   full_extended - Train on all sequences, 50 epochs, max_ahead=6
#   comprehensive - Train on ALL frame sequences, 50 epochs, max_ahead=4
#   optimal       - Optimal settings: max_ahead=4, 50 epochs, batch_size=3, all frames
#                   Uses composed flows, validation 2x/epoch on both test datasets
#
# Author: AI Assistant
# Date: October 23, 2025
# =============================================================================

set -e  # Exit on any error

# Get the mode from command line argument
# Default to "optimal" mode (max_ahead=4, 50 epochs, all frames)
MODE=${1:-"optimal"}

# Set default parameters based on mode
case $MODE in
    "test")
        echo "🧪 Running Experiment 2 in TEST mode"
        MAX_SEQUENCES=2
        NUM_EPOCHS=10
        MAX_AHEAD=3
        BATCH_SIZE=1
        LR=1e-4
        MAX_SAMPLES_EVAL=10
        ;;
    "small")
        echo "🔬 Running Experiment 2 in SMALL mode"
        MAX_SEQUENCES=20
        NUM_EPOCHS=30
        MAX_AHEAD=3
        BATCH_SIZE=3
        LR=1e-4
        MAX_SAMPLES_EVAL=50
        ;;
    "full")
        echo "🚀 Running Experiment 2 in FULL mode"
        MAX_SEQUENCES=""
        NUM_EPOCHS=20
        MAX_AHEAD=3
        BATCH_SIZE=3
        LR=1e-4
        MAX_SAMPLES_EVAL=100
        ;;
    "full_extended")
        echo "🌟 Running Experiment 2 in FULL EXTENDED mode"
        MAX_SEQUENCES=""
        NUM_EPOCHS=50
        MAX_AHEAD=6
        BATCH_SIZE=1
        LR=1e-4
        MAX_SAMPLES_EVAL=100
        ;;
    "comprehensive")
        echo "🔥 Running Experiment 2 in COMPREHENSIVE mode (all sequences, long training)"
        MAX_SEQUENCES=""
        NUM_EPOCHS=50
        MAX_AHEAD=4
        BATCH_SIZE=3
        LR=5e-5
        MAX_SAMPLES_EVAL=200
        ;;
    "optimal")
        echo "🎯 Running Experiment 2 in OPTIMAL mode (max_ahead=4, all frames, 50 epochs)"
        MAX_SEQUENCES=""
        NUM_EPOCHS=50
        MAX_AHEAD=4
        BATCH_SIZE=3
        LR=5e-5
        MAX_SAMPLES_EVAL=200
        ;;
    *)
        echo "❌ Unknown mode: $MODE"
        echo "Available modes: test, small, full, full_extended, comprehensive, optimal"
        exit 1
        ;;
esac

# Display configuration
echo "=========================================="
echo "Experiment 2 Configuration:"
echo "=========================================="
echo "Mode: $MODE"
echo "Max sequences: ${MAX_SEQUENCES:-"all"}"
echo "Epochs: $NUM_EPOCHS"
echo "Max ahead: $MAX_AHEAD (frames: 1,2,3,$(seq -s, 4 $((MAX_AHEAD+1))))"
echo "Batch size: $BATCH_SIZE"
echo "Learning rate: $LR"
echo "Flow composition: Composed flows (default, using consecutive UniMatch flows)"
echo "Validation: Twice per epoch on Objectron test + LightSpeed"
echo "Evaluation samples: $MAX_SAMPLES_EVAL"
echo "=========================================="

# Set up paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Create results directory
RESULTS_DIR="experiments/pose_head_experiment_results/exp2_${MODE}_run"
mkdir -p "$RESULTS_DIR"

echo "📁 Results will be saved to: $RESULTS_DIR"

# =============================================================================
# Step 1: Train Experiment 2 Model
# =============================================================================
echo ""
echo "🎯 Step 1: Training Experiment 2 Model"
echo "======================================"

# Build training command
TRAIN_CMD="python experiments/train_pose_head_anycalib_exp2.py"
TRAIN_CMD="$TRAIN_CMD --num_epochs $NUM_EPOCHS"
TRAIN_CMD="$TRAIN_CMD --batch_size $BATCH_SIZE"
TRAIN_CMD="$TRAIN_CMD --lr $LR"
TRAIN_CMD="$TRAIN_CMD --max_ahead $MAX_AHEAD"
# Note: Composed flows are default (no --use_direct_flow flag means use_composed_flow=True)
TRAIN_CMD="$TRAIN_CMD --save_dir $RESULTS_DIR"
# Validation is automatically enabled during training (Objectron test + LightSpeed, 2x per epoch)

if [ -n "$MAX_SEQUENCES" ]; then
    TRAIN_CMD="$TRAIN_CMD --max_sequences $MAX_SEQUENCES"
fi

echo "Running: $TRAIN_CMD"
echo ""

# Run training
eval $TRAIN_CMD

# Check if training completed successfully
if [ ! -f "$RESULTS_DIR/final_model.pt" ]; then
    echo "❌ Training failed - no final model found!"
    exit 1
fi

echo "✅ Training completed successfully!"

# =============================================================================
# Step 2: Run Multi-Model Benchmarking (on both datasets)
# =============================================================================
echo ""
echo "📊 Step 2: Running Multi-Model Benchmarking"
echo "==========================================="

# Set up model paths
EXP1_MODEL="experiments/pose_head_experiment_results/full_run_eval/final_model.pt"
EXP2_MODEL="$RESULTS_DIR/final_model.pt"
BASELINE_MODEL="pretrained_models/anycam_seq8/training_checkpoint_247500.pt"

# Check if Experiment 1 model exists
if [ ! -f "$EXP1_MODEL" ]; then
    echo "⚠️  Experiment 1 model not found at $EXP1_MODEL"
    echo "   Will benchmark only Experiment 2 vs Baseline"
    EXP1_MODEL=""
fi

# -----------------------------------------------------------------------------
# Benchmark on Objectron Test Split
# -----------------------------------------------------------------------------
echo ""
echo "🔬 Benchmarking on Objectron Test Split..."
echo "----------------------------------------"

BENCHMARK_OBJECTRON_CMD="python experiments/benchmark_against_anycam.py"
BENCHMARK_OBJECTRON_CMD="$BENCHMARK_OBJECTRON_CMD --dataset objectron"
BENCHMARK_OBJECTRON_CMD="$BENCHMARK_OBJECTRON_CMD --max_samples $MAX_SAMPLES_EVAL"
BENCHMARK_OBJECTRON_CMD="$BENCHMARK_OBJECTRON_CMD --save_dir $RESULTS_DIR/benchmark_results_objectron"
BENCHMARK_OBJECTRON_CMD="$BENCHMARK_OBJECTRON_CMD --objectron_videos /home/kalman/TUM/thesis/Objectron/videos/"
BENCHMARK_OBJECTRON_CMD="$BENCHMARK_OBJECTRON_CMD --objectron_gt /home/kalman/TUM/thesis/Objectron/processed_gt/"
BENCHMARK_OBJECTRON_CMD="$BENCHMARK_OBJECTRON_CMD --split_file experiments/objectron_split.json"

if [ -n "$EXP1_MODEL" ]; then
    BENCHMARK_OBJECTRON_CMD="$BENCHMARK_OBJECTRON_CMD --exp1_model $EXP1_MODEL"
fi

BENCHMARK_OBJECTRON_CMD="$BENCHMARK_OBJECTRON_CMD --exp2_model $EXP2_MODEL"
BENCHMARK_OBJECTRON_CMD="$BENCHMARK_OBJECTRON_CMD --baseline_checkpoint $BASELINE_MODEL"

echo "Running: $BENCHMARK_OBJECTRON_CMD"
echo ""

# Run Objectron benchmarking
eval $BENCHMARK_OBJECTRON_CMD

# Check if Objectron benchmarking completed
if [ -f "$RESULTS_DIR/benchmark_results_objectron/benchmark_results.json" ]; then
    echo "✅ Objectron benchmarking completed successfully!"
else
    echo "⚠️  Objectron benchmarking may have failed - results not found"
fi

# -----------------------------------------------------------------------------
# Benchmark on LightSpeed Dataset
# -----------------------------------------------------------------------------
echo ""
echo "🚀 Benchmarking on LightSpeed Dataset..."
echo "--------------------------------------"

BENCHMARK_LIGHTSPEED_CMD="python experiments/benchmark_against_anycam.py"
BENCHMARK_LIGHTSPEED_CMD="$BENCHMARK_LIGHTSPEED_CMD --dataset lightspeed"
BENCHMARK_LIGHTSPEED_CMD="$BENCHMARK_LIGHTSPEED_CMD --max_samples $MAX_SAMPLES_EVAL"
BENCHMARK_LIGHTSPEED_CMD="$BENCHMARK_LIGHTSPEED_CMD --save_dir $RESULTS_DIR/benchmark_results_lightspeed"
BENCHMARK_LIGHTSPEED_CMD="$BENCHMARK_LIGHTSPEED_CMD --lightspeed_dir /home/kalman/TUM/thesis/dynpose-100k/lightspeed/"

if [ -n "$EXP1_MODEL" ]; then
    BENCHMARK_LIGHTSPEED_CMD="$BENCHMARK_LIGHTSPEED_CMD --exp1_model $EXP1_MODEL"
fi

BENCHMARK_LIGHTSPEED_CMD="$BENCHMARK_LIGHTSPEED_CMD --exp2_model $EXP2_MODEL"
BENCHMARK_LIGHTSPEED_CMD="$BENCHMARK_LIGHTSPEED_CMD --baseline_checkpoint $BASELINE_MODEL"

echo "Running: $BENCHMARK_LIGHTSPEED_CMD"
echo ""

# Run LightSpeed benchmarking
eval $BENCHMARK_LIGHTSPEED_CMD

# Check if LightSpeed benchmarking completed
if [ -f "$RESULTS_DIR/benchmark_results_lightspeed/benchmark_results.json" ]; then
    echo "✅ LightSpeed benchmarking completed successfully!"
else
    echo "⚠️  LightSpeed benchmarking may have failed - results not found"
fi

echo ""
echo "✅ All benchmarking completed!"

# =============================================================================
# Step 3: Display Results Summary
# =============================================================================
echo ""
echo "📈 Step 3: Results Summary"
echo "=========================="

# Display benchmark results
if [ -f "$RESULTS_DIR/benchmark_results_objectron/benchmark_report.txt" ]; then
    echo ""
    echo "📋 Objectron Benchmark Report:"
    echo "-------------------"
    cat "$RESULTS_DIR/benchmark_results_objectron/benchmark_report.txt"
fi

if [ -f "$RESULTS_DIR/benchmark_results_lightspeed/benchmark_report.txt" ]; then
    echo ""
    echo "📋 LightSpeed Benchmark Report:"
    echo "-------------------"
    cat "$RESULTS_DIR/benchmark_results_lightspeed/benchmark_report.txt"
fi

# Display training summary
if [ -f "$RESULTS_DIR/training_summary.txt" ]; then
    echo ""
    echo "📊 Training Summary:"
    echo "-------------------"
    cat "$RESULTS_DIR/training_summary.txt"
fi

# =============================================================================
# Step 4: Generate Final Summary
# =============================================================================
echo ""
echo "🎉 Experiment 2 ($MODE mode) completed successfully!"
echo "=================================================="
echo ""
echo "📁 Results saved to: $RESULTS_DIR"
echo "   - final_model.pt (trained model)"
echo "   - loss_curve.png (training + validation visualization)"
echo "   - training_summary.txt (training statistics)"
echo "   - benchmark_results_objectron/ (Objectron test evaluation)"
echo "     - benchmark_results.json (detailed metrics)"
echo "     - benchmark_comparison.png (visualization)"
echo "     - benchmark_report.txt (text report)"
echo "   - benchmark_results_lightspeed/ (LightSpeed evaluation)"
echo "     - benchmark_results.json (detailed metrics)"
echo "     - benchmark_comparison.png (visualization)"
echo "     - benchmark_report.txt (text report)"
echo ""

# Show key metrics if available
for dataset in "objectron" "lightspeed"; do
    results_file="$RESULTS_DIR/benchmark_results_${dataset}/benchmark_results.json"
    if [ -f "$results_file" ]; then
        echo ""
        echo "🔍 Key Results (${dataset}):"
        echo "---------------"
        
        # Extract and display key metrics using Python
        python3 -c "
import json
import sys

try:
    with open('$results_file', 'r') as f:
        data = json.load(f)
    
    print('Rotation Error (degrees):')
    for model_name, results in data.items():
        if model_name != 'metadata' and 'rotation' in results:
            mean_rot = results['rotation']['mean']
            print(f'  {model_name:<20}: {mean_rot:.4f}°')
    
    print()
    print('Translation Error (degrees):')
    for model_name, results in data.items():
        if model_name != 'metadata' and 'translation' in results:
            mean_trans = results['translation']['mean']
            print(f'  {model_name:<20}: {mean_trans:.4f}°')
            
except Exception as e:
    print(f'Could not parse results: {e}')
    sys.exit(0)
"
    fi
done

echo ""
echo "✨ Experiment 2 ($MODE mode) finished!"
echo "======================================"
