# Stage 3.1 Training Setup: max_ahead=4, With Alternating Training

## Configuration

- **max_ahead**: 4 (loads 5 frames per sequence: [0,1,2,3,4])
- **Alternating Training**: Enabled
  - Even epochs: Train calibration head, freeze pose head
  - Odd epochs: Train pose head, freeze calibration head
- **Benchmark Samples**: 100 fixed samples (no cycling)
- **Starting Checkpoint**: Stage 2 `final_model.pt`

## Training Command

```bash
python experiments/train_calibration_head_da3_stage3.py \
    --objectron_videos /data/thesis/Objectron/videos \
    --objectron_gt /data/thesis/Objectron/processed_gt \
    --stage2_checkpoint experiments/da3_integration/stage2_training/checkpoints/final_model.pt \
    --num_epochs 50 \
    --batch_size 2 \
    --learning_rate 1e-5 \
    --max_ahead 4 \
    --alternating_training \
    --benchmark_samples 100 \
    --benchmark_no_cycle \
    --save_dir experiments/da3_integration/stage3_1_maxahead4_alternating
```

## Key Differences from Stage 3

1. **Multi-frame Input**: Uses 5 frames per sequence instead of 2-frame pairs
2. **Fixed Benchmark**: Uses same 100 samples every epoch (no cycling)
3. **Dataset**: Uses `ObjectronVideoDatasetMultiFrame` for multi-frame sequences
4. **Alternating Training Strategy**: 
   - Epoch 0, 2, 4, ...: Train calibration head (pose head frozen)
   - Epoch 1, 3, 5, ...: Train pose head (calibration head frozen)
   - Optimizer is recreated each epoch with appropriate parameters

## Expected Behavior

- Model processes sequences of 5 frames at once
- Training alternates between calibration head and pose head each epoch
- Optimizer is recreated each epoch to match trainable parameters
- Benchmark uses fixed set of 100 samples for consistent evaluation across epochs

