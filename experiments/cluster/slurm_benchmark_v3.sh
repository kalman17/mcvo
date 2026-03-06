#!/bin/bash
#SBATCH --job-name=bench_v3
#SBATCH --output=/storage/user/maka/logs/bench_v3_%j.out
#SBATCH --error=/storage/user/maka/logs/bench_v3_%j.err
#SBATCH --partition=NORMAL
#SBATCH --comment="Masters thesis deadline 2026-03-31, benchmark A_v3 + B1_v2"
#SBATCH --gres=gpu:1,VRAM:24G
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0-04:00:00

set -euo pipefail

REPO="/storage/user/maka/anycam"
DATA_ROOT="/storage/user/maka/eval_datasets"
OUTPUT_DIR="/storage/user/maka/train/benchmark_v3"

PHASE_A_CKPT="/storage/user/maka/train/phase_A_v3/checkpoints/best.pt"
PHASE_B1_CKPT="/storage/user/maka/train/phase_B1_v2/checkpoints/best.pt"
PRETRAINED="/storage/user/maka/anycam/pretrained_models/anycam_seq8/training_checkpoint_247500.pt"
ANYCAM_CFG="/storage/user/maka/anycam/pretrained_models/anycam_seq8/training_config.yaml"
COMBINED_CKPT="$OUTPUT_DIR/combined_A_v3_B1_v2.pt"

export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

eval "$(/storage/user/maka/miniconda3/bin/conda shell.bash hook)"
conda activate anycam
cd /tmp

echo "============================================"
echo "  Benchmark v3 (Phase A v3 + B1 v2)"
echo "  Host: $(hostname)"
echo "  GPU:  $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo unknown)"
echo "  Date: $(date)"
echo "============================================"

mkdir -p "$OUTPUT_DIR"

# Step 1: Create combined checkpoint
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
    'config': {'note': 'Combined Phase A v3 (best, epoch 4) + Phase B1 v2 (best, epoch 4)'},
}
torch.save(checkpoint, '$COMBINED_CKPT')
print(f'Saved combined checkpoint to $COMBINED_CKPT')
"

# Step 2: Benchmark
echo ""
echo "=== Benchmarking combined model (quick mode) ==="
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
echo "=== Benchmark v3 COMPLETE ==="
echo "Date: $(date)"
