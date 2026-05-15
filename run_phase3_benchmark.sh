#!/bin/bash
# Wrapper script to run Phase 3 benchmarking from Docker or host

# Check if running in Docker or host
if [ -d "/data/thesis" ]; then
    # Docker environment
    VIDEOS_DIR="/data/thesis/Objectron/videos"
    GT_DIR="/data/thesis/Objectron/processed_gt"
else
    # Host environment
    VIDEOS_DIR="/home/kalmanm/git/masters/Objectron/videos"
    GT_DIR="/home/kalmanm/git/masters/Objectron/processed_gt"
fi

# Run benchmark
python experiments/benchmark_phase3_checkpoints.py \
    --checkpoint_dir experiments/final_training_phases/phase3_training_v2/checkpoints \
    --objectron_videos "$VIDEOS_DIR" \
    --objectron_gt "$GT_DIR" \
    --anycam_config pretrained_models/anycam_seq8/training_config.yaml \
    --num_samples 50 \
    --max_ahead 3 \
    --output_dir experiments/final_training_phases/phase3_training_v2/benchmark_results \
    --device cuda:0
