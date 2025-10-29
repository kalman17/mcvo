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
#   test     - Quick test on 1-2 sequences, 10 epochs, max_ahead=3
#   small    - Train on 20 sequences, 30 epochs, max_ahead=3  
#   full     - Train on all sequences, 50 epochs, max_ahead=3
#   full_extended - Train on all sequences, 50 epochs, max_ahead=6
#   comprehensive - Train on ALL frame sequences, 100 epochs, max_ahead=3
#
# Author: AI Assistant
# Date: October 23, 2025
# =============================================================================

set -e  # Exit on any error

# Get the mode from command line argument
MODE=${1:-"test"}

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
        NUM_EPOCHS=100
        MAX_AHEAD=3
        BATCH_SIZE=2
        LR=5e-5
        MAX_SAMPLES_EVAL=200
        ;;
    *)
        echo "❌ Unknown mode: $MODE"
        echo "Available modes: test, small, full, full_extended, comprehensive"
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
echo "Direct flow: UniMatch (1->3, 1->4)"
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
TRAIN_CMD="$TRAIN_CMD --use_direct_flow"  # Enable UniMatch direct flow
TRAIN_CMD="$TRAIN_CMD --save_dir $RESULTS_DIR"
TRAIN_CMD="$TRAIN_CMD --run_evaluation"
TRAIN_CMD="$TRAIN_CMD --eval_dataset lightspeed"

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
# Step 2: Run Multi-Model Benchmarking
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

# Build benchmarking command
BENCHMARK_CMD="python experiments/benchmark_against_anycam.py"
BENCHMARK_CMD="$BENCHMARK_CMD --dataset lightspeed"
BENCHMARK_CMD="$BENCHMARK_CMD --max_samples $MAX_SAMPLES_EVAL"
BENCHMARK_CMD="$BENCHMARK_CMD --save_dir $RESULTS_DIR/benchmark_results"

if [ -n "$EXP1_MODEL" ]; then
    BENCHMARK_CMD="$BENCHMARK_CMD --exp1_model $EXP1_MODEL"
fi

BENCHMARK_CMD="$BENCHMARK_CMD --exp2_model $EXP2_MODEL"
BENCHMARK_CMD="$BENCHMARK_CMD --baseline_checkpoint $BASELINE_MODEL"

echo "Running: $BENCHMARK_CMD"
echo ""

# Run benchmarking
eval $BENCHMARK_CMD

# Check if benchmarking completed successfully
if [ ! -f "$RESULTS_DIR/benchmark_results/benchmark_results.json" ]; then
    echo "❌ Benchmarking failed - no results found!"
    exit 1
fi

echo "✅ Benchmarking completed successfully!"

# =============================================================================
# Step 3: Display Results Summary
# =============================================================================
echo ""
echo "📈 Step 3: Results Summary"
echo "=========================="

# Display benchmark results
if [ -f "$RESULTS_DIR/benchmark_results/benchmark_report.txt" ]; then
    echo ""
    echo "📋 Benchmark Report:"
    echo "-------------------"
    cat "$RESULTS_DIR/benchmark_results/benchmark_report.txt"
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
echo "   - loss_curve.png (training visualization)"
echo "   - training_summary.txt (training statistics)"
echo "   - benchmark_results/ (evaluation results)"
echo "     - benchmark_results.json (detailed metrics)"
echo "     - benchmark_comparison.png (visualization)"
echo "     - benchmark_report.txt (text report)"
echo ""

# Show key metrics if available
if [ -f "$RESULTS_DIR/benchmark_results/benchmark_results.json" ]; then
    echo "🔍 Key Results:"
    echo "---------------"
    
    # Extract and display key metrics using Python
    python3 -c "
import json
import sys

try:
    with open('$RESULTS_DIR/benchmark_results/benchmark_results.json', 'r') as f:
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

echo ""
echo "✨ Experiment 2 ($MODE mode) finished!"
echo "======================================"
