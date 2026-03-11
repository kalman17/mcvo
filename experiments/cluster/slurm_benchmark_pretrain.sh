#!/bin/bash
#SBATCH --job-name=bench_pre
#SBATCH --output=/storage/user/maka/logs/bench_pretrain_%j.out
#SBATCH --error=/storage/user/maka/logs/bench_pretrain_%j.err
#SBATCH --partition=NORMAL
#SBATCH --comment="Masters thesis deadline 2026-03-31, benchmark pre-training baseline"
#SBATCH --gres=gpu:1,VRAM:24G
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0-04:00:00

set -euo pipefail

REPO="/storage/user/maka/anycam"
DATA_ROOT="/storage/user/maka/eval_datasets"
OUTPUT_DIR="/storage/user/maka/train/phase_C_v3/benchmark_results/epoch_0000"

PHASE_A_CKPT="/storage/user/maka/train/phase_A_v2/checkpoints/latest.pt"
PHASE_B1_CKPT="/storage/user/maka/train/phase_B1/checkpoints/latest.pt"
PRETRAINED="/storage/user/maka/anycam/pretrained_models/anycam_seq8/training_checkpoint_247500.pt"
ANYCAM_CFG="/storage/user/maka/anycam/pretrained_models/anycam_seq8/training_config.yaml"
COMBINED_CKPT="$OUTPUT_DIR/combined_epoch0000.pt"

export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

eval "$(/storage/user/maka/miniconda3/bin/conda shell.bash hook)"
conda activate anycam
cd /tmp

echo "============================================"
echo "  Pre-training Benchmark (Phase A + B1)"
echo "  Host: $(hostname)"
echo "  GPU:  $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo unknown)"
echo "  Date: $(date)"
echo "============================================"

mkdir -p "$OUTPUT_DIR"

# Step 1: Create combined checkpoint (Phase C model with A + B1 weights, no training)
echo ""
echo "=== Creating combined checkpoint ==="
python3 -c "
import os, sys, torch
os.environ['XFORMERS_DISABLED'] = '1'
sys.path.insert(0, '$REPO')

from experiments.models.unified_wrapper import UnifiedTrainingWrapper

model = UnifiedTrainingWrapper(phase='C', anycam_config_path='$ANYCAM_CFG')
model.load_pretrained_pose_predictor('$PRETRAINED')
model.load_phase_checkpoint('$PHASE_A_CKPT', source_phase='A')
model.load_phase_checkpoint('$PHASE_B1_CKPT', source_phase='B1')

checkpoint = {
    'phase': 'C',
    'epoch': -1,
    'model_state_dict': model.state_dict(),
    'loss_history': [],
    'config': {'note': 'Pre-training combined Phase A + B1, no Phase C training'},
}
torch.save(checkpoint, '$COMBINED_CKPT')
print(f'Saved combined checkpoint to $COMBINED_CKPT')
"

# Step 2: Benchmark the combined checkpoint
echo ""
echo "=== Benchmarking combined model ==="
python3 "$REPO/experiments/benchmark_phase_c_checkpoints.py" \
    --single_checkpoint "$COMBINED_CKPT" \
    --anycam_config "$ANYCAM_CFG" \
    --pretrained_anycam "$PRETRAINED" \
    --data_root "$DATA_ROOT" \
    --mode quick \
    --image_size 336 \
    --output_dir "$OUTPUT_DIR" \
    2>&1

echo ""
echo "=== Pre-training Benchmark COMPLETE ==="
echo "Date: $(date)"
