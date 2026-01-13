# Stage 3.1 Training Setup: max_ahead=3, No Alternating

## Configuration

- **max_ahead**: 3 (loads 4 frames per sequence: [0,1,2,3])
- **Alternating Training**: Disabled
- **Benchmark Samples**: 100 fixed samples (no cycling)
- **Starting Checkpoint**: Stage 2 `final_model.pt`

## Training Command

```bash
python experiments/train_calibration_head_da3_stage3.py \
    --objectron_videos /data/thesis/Objectron/videos \
    --objectron_gt /data/thesis/Objectron/processed_gt \
    --stage2_checkpoint experiments/da3_integration/stage2_training/checkpoints/final_model.pt \
    --num_epochs 50 \
    --batch_size 4 \
    --learning_rate 1e-5 \
    --max_ahead 3 \
    --benchmark_samples 100 \
    --benchmark_no_cycle \
    --save_dir experiments/da3_integration/stage3_1_maxahead3
```

## Key Differences from Stage 3

1. **Multi-frame Input**: Uses 4 frames per sequence instead of 2-frame pairs
2. **Fixed Benchmark**: Uses same 100 samples every epoch (no cycling)
3. **Dataset**: Uses `ObjectronVideoDatasetMultiFrame` for multi-frame sequences
4. **Training Strategy**: Standard training (only calibration head is trainable)

## Expected Behavior

- Model processes sequences of 4 frames at once
- DA3 calibration head outputs single focal length for entire sequence
- Pose predictor handles multi-frame input natively
- Benchmark uses fixed set of 100 samples for consistent evaluation across epochs

