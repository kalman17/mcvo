# Experiment 2: Composed Flow Implementation

## Summary

I've successfully implemented flow composition functionality in Experiment 2, allowing the model to use composed flows instead of direct UniMatch flows for long-range pose evaluation.

## Key Features Added

### 1. Flow Composition Function (`compose_flows`)
- **Purpose**: Composes consecutive flows by warping through intermediate frames
- **Input**: List of consecutive flows [B, 2, H, W] and occlusion masks [B, H, W]
- **Output**: Composed long-range flow [B, 2, H, W] and occlusion mask [B, H, W]
- **Method**: Uses bilinear interpolation to warp previous flows through current flow

### 2. Enhanced Multi-Frame Wrapper
- **Flow Storage**: Extracts flows from consecutive pairs (1→2, 2→3, 3→4)
- **Flow Composition**: Creates composed flows for long-range pairs (1→3, 1→4)
- **Result Structure**: Returns both consecutive and composed flows for loss computation

### 3. Updated Loss Function
- **Two Modes**: 
  - `use_direct_flow=True`: Uses UniMatch for direct GT flow computation
  - `use_direct_flow=False`: Uses composed flows from consecutive pairs
- **Composed Loss**: Computes reprojection loss for both consecutive and composed poses
- **Weighting**: Composed losses are weighted (0.1x) to balance with consecutive losses

### 4. New Command Line Arguments
- `--use_composed_flow`: Enable composed flow mode (overrides `--use_direct_flow`)
- `--disable_composed_loss`: Disable composed pose losses (only consecutive pairs)

## Usage Examples

### Training with Composed Flows
```bash
python experiments/train_pose_head_anycalib_exp2.py \
    --num_epochs 20 \
    --batch_size 1 \
    --lr 1e-4 \
    --max_ahead 3 \
    --use_composed_flow \
    --save_dir experiments/pose_head_experiment_results/exp2_composed_flow
```

### Training with Direct UniMatch Flows (default)
```bash
python experiments/train_pose_head_anycalib_exp2.py \
    --num_epochs 20 \
    --batch_size 1 \
    --lr 1e-4 \
    --max_ahead 3 \
    --use_direct_flow \
    --save_dir experiments/pose_head_experiment_results/exp2_direct_flow
```

## Technical Details

### Flow Composition Algorithm
1. **Start with first flow**: `composed_flow = flow_1to2`
2. **For each subsequent flow**:
   - Warp coordinates by current flow: `warped_coords = coords + current_flow`
   - Interpolate previous flow at warped coordinates
   - Add current flow to warped previous flow
   - Update occlusion mask (AND operation)

### Loss Computation
- **Consecutive Losses**: 3 losses for pairs (1→2, 2→3, 3→4)
- **Composed Losses**: 2 losses for composed poses (1→3, 1→4)
- **Total Loss**: Average of all losses with composed losses weighted by 0.1

## Benefits of Composed Flows

1. **Computational Efficiency**: No need to run UniMatch for long-range pairs
2. **Consistency**: Uses the same flow computation method as consecutive pairs
3. **Robustness**: Avoids potential issues with UniMatch on large temporal gaps
4. **Scalability**: Can handle arbitrary frame ranges by composing flows

## Testing Results

The implementation has been tested and verified:
- ✅ Flow composition function works correctly
- ✅ Multi-frame wrapper extracts and composes flows
- ✅ Loss function computes both consecutive and composed losses
- ✅ Training runs successfully with composed flows
- ✅ Model saves and loads correctly

## Next Steps

To run a full experiment with composed flows:

1. **Quick Test** (already working):
   ```bash
   ./experiments/test_composed_flow_training.sh
   ```

2. **Full Training**:
   ```bash
   python experiments/train_pose_head_anycalib_exp2.py \
       --num_epochs 50 \
       --batch_size 1 \
       --lr 1e-4 \
       --max_ahead 3 \
       --use_composed_flow \
       --save_dir experiments/pose_head_experiment_results/exp2_composed_flow_full
   ```

3. **Comparison Study**:
   - Train one model with `--use_direct_flow`
   - Train another with `--use_composed_flow`
   - Compare results using the benchmarking script

The composed flow implementation is now ready for production use and should provide an alternative approach to long-range pose evaluation that may be more computationally efficient and potentially more robust than direct UniMatch flows.
