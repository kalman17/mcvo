<!-- 2d7d9592-22a2-4d4c-bef4-e9c56c71c9cb ddb7284b-170c-40e0-b7ea-9175a93a0d22 -->
# Experiment 2 Enhancements Plan

## Overview
This plan adds validation loss tracking, switches to composed flows by default, uses all video frames with reduced epochs, and creates a hyperparameter sweep script for max_ahead values 2-7.

## 1. Add Validation Loss Tracking During Training

### Files to Modify:
- `experiments/train_pose_head_anycalib_exp2.py`

### Changes:
1. **Add validation datasets setup** (after train dataset creation, ~line 1100):
   - Create ObjectronVideoDatasetMultiFrame for test split with same max_ahead
   - Create LightSpeedDataset (always num_frames=2 for evaluation)
   - Both use same batch_size=1 for consistency

2. **Add validation function** (new function, ~line 850):
   ```python
   def evaluate_model_multiframe(model, dataloaders, criterion, device):
       """Evaluate model on multiple validation datasets."""
       model.eval()
       val_losses = {}
       
       with torch.no_grad():
           for dataset_name, dataloader in dataloaders.items():
               total_loss = 0.0
               num_batches = 0
               
               for batch_data in dataloader:
                   # Move to device and evaluate
                   # Handle 2-frame vs multi-frame datasets
                   # Compute loss using criterion
                   # Accumulate
               
               val_losses[dataset_name] = total_loss / num_batches
       
       model.train()
       return val_losses
   ```

3. **Modify training loop** (in `train_pose_head_multiframe`, ~line 850):
   - After each epoch, compute validation losses (2x per epoch: mid-epoch and end-epoch)
   - Store validation losses in loss_history alongside training loss
   - Handle different image sizes by ensuring datasets use same preprocessing

4. **Update loss curve plotting** (modify `plot_loss_curve` call or create new function):
   - Plot training loss + objectron_test loss + lightspeed loss
   - Use different line styles/colors for each curve
   - Save to same `loss_curve.png` file

## 2. Change Default to Composed Flows

### Files to Modify:
- `experiments/train_pose_head_anycalib_exp2.py`

### Changes:
1. **Update argument parser default** (~line 1019-1022):
   - Change `--use_direct_flow` default to `False`
   - Change `--use_composed_flow` default to `True` (add `action="store_true", default=True`)
   - Update help text to reflect new defaults

2. **Update loss initialization logic** (~line 1172-1178):
   - Simplify: if `args.use_composed_flow` is True (default), set `use_direct_flow=False`
   - Remove redundant check since composed_flow takes precedence

## 3. Use All Available Frames with step_size=1

### Files to Modify:
- `experiments/train_pose_head_anycalib_exp2.py`

### Changes:
1. **Modify `_build_pair_index` method** in `ObjectronVideoDatasetMultiFrame` (~line 290):
   - Change `step_size = 10` to `step_size = 1`
   - This creates overlapping sequences: [0,1,2,3], [1,2,3,4], [2,3,4,5], ...
   - Ensure formula: `for start_frame in range(0, safe_total_frames - self.max_ahead, 1)`
   - This will use all frames except the last `max_ahead` frames (which can't form complete sequences)

2. **Reduce epochs in argument parser** (~line 1009):
   - Change default `--num_epochs` from 50 to 10 (or make it configurable)
   - Update help text to note reduced epochs due to more training data

## 4. Create Hyperparameter Sweep Script

### New File:
- `experiments/hyperparameter_sweep_exp2.py`

### Features:
1. **Loop through max_ahead values [2, 3, 4, 5, 6, 7]**
2. **For each max_ahead**:
   - Create unique save directory: `exp2_maxahead_{max_ahead}`
   - Run training with fixed settings:
     - `--num_epochs 10`
     - `--batch_size 2`
     - `--use_composed_flow` (default True)
     - `--max_ahead {current_value}`
     - `--save_dir experiments/pose_head_experiment_results/exp2_maxahead_{max_ahead}`
   - Wait for training to complete before starting next

3. **After all trainings complete**:
   - Run benchmarking script with all trained models:
     - Collect all model paths: `exp2_maxahead_2/final_model.pt`, `exp2_maxahead_3/final_model.pt`, etc.
     - Run benchmark on both datasets:
       - Objectron test split
       - LightSpeed dataset
     - Compare all max_ahead models + AnyCam baseline
   - Generate comprehensive comparison plots and reports

4. **Script structure**:
   ```python
   def run_training_sweep(max_ahead_values, ...):
       trained_models = {}
       for max_ahead in max_ahead_values:
           save_dir = f"experiments/pose_head_experiment_results/exp2_maxahead_{max_ahead}"
           # Run training
           trained_models[max_ahead] = save_dir
       
       # Run benchmarking
       run_comparison_benchmark(trained_models, ...)
   ```

## 5. Update Benchmarking Script

### Files to Modify:
- `experiments/benchmark_against_anycam.py`

### Changes:
1. **Add support for multiple models comparison**:
   - Accept multiple `--exp2_model` paths or a directory pattern
   - Auto-detect all `exp2_maxahead_*` directories if path is parent directory

2. **Run benchmarking twice**:
   - Once on Objectron test split
   - Once on LightSpeed dataset
   - Generate separate comparison plots for each dataset
   - Generate combined report comparing all models on both datasets

3. **Enhanced visualization**:
   - Bar plots comparing all max_ahead values
   - Line plots showing performance vs max_ahead
   - Separate figures for rotation error and translation error

## Implementation Details

### Validation Loss Tracking:
- Evaluation happens 2x per epoch (at 50% progress and at epoch end)
- Handle LightSpeed's 2-frame constraint: use `super().forward()` fallback in wrapper
- Objectron test uses same max_ahead as training
- Store validation losses in `loss_history` as: `{'epoch': ..., 'loss': ..., 'val_objectron': ..., 'val_lightspeed': ...}`

### Dataset Frame Utilization:
- With `step_size=1`, for max_ahead=3 and video with 100 frames:
  - Sequences: [0,1,2,3], [1,2,3,4], ..., [96,97,98,99] = 97 sequences
  - Uses frames 0-99 (all except last 3 incomplete sequences)
- This dramatically increases training data size, hence epoch reduction

### Hyperparameter Sweep:
- Sequential execution (not parallel) to avoid GPU memory issues
- Each model trained from scratch (fresh pose head initialization)
- All models use same base configuration except max_ahead
- Benchmarking runs after all trainings complete

## Testing Strategy:
1. Test validation loss tracking on small subset (1 epoch, 2 sequences)
2. Verify composed flows work with step_size=1
3. Test hyperparameter sweep with max_ahead=[2,3] only first
4. Verify benchmarking script handles multiple models correctly

### To-dos

- [ ] Add validation loss tracking during training (Objectron test + LightSpeed, 2x per epoch)
- [ ] Change default to use_composed_flow=True, use_direct_flow=False
- [ ] Change step_size to 1 in dataset to use all available frames, reduce default epochs to 10
- [ ] Create hyperparameter_sweep_exp2.py to train max_ahead values 2-7 sequentially
- [ ] Update benchmark_against_anycam.py to support multiple models and both datasets