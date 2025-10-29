<!-- 2d7d9592-22a2-4d4c-bef4-e9c56c71c9cb d034df3c-1bd0-4588-8478-5f3eeb022011 -->
# Experiment 1 Sanity Check + Experiment 2: Multi-Frame Pose Prediction

## Phase 1: Sanity Check of Experiment 1

### 1.1 Verify AnyCaLib Integration

- **File**: `experiments/train_pose_head_anycalib.py` (lines 475-556)
- **Check**: AnyCaLib runs on first frame only (line 523), assumes constant focal length
- **Verify**: `predict_focal_length()` returns `[batch]` tensor in pixels, correctly extracted as `K[0]` from intrinsics
- **Status**: ✓ CORRECT - using single frame approach as intended

### 1.2 Verify Pose Head Replacement

- **File**: `experiments/train_pose_head_anycalib.py` (lines 643-670)
- **Check**: `reinitialize_pose_head()` deletes old head, creates fresh `AnyCamPoseTokenHead` with random weights
- **Verify**: Only pose_head is trainable (lines 612-641), all other components frozen
- **Status**: ✓ CORRECT - pose head properly reinitialized and isolated

### 1.3 Verify Focal Length Injection

- **File**: `experiments/train_pose_head_anycalib.py` (lines 688-703)
- **Check**: AnyCaLib focal length replaces 32-candidate system
- **Verify**: `proj_candidates` created with single focal value `[B, 1]` instead of 32 candidates
- **Verify**: Poses filtered to first candidate only (lines 737-742)
- **Status**: ✓ CORRECT - single focal length properly injected, no candidate system

### 1.4 Verify Loss Function

- **File**: `experiments/train_pose_head_anycalib.py` (lines 1207-1212)
- **Check**: Uses AnyCam's original `PoseLoss` from config
- **Verify**: `lambda_fwd_bwd_consistency = 0` (disabled for forward-only training)
- **Verify**: Flow reprojection loss computed via `induce_flow_dist()` (line 748)
- **Status**: ✓ CORRECT - unsupervised flow reprojection loss, properly configured

### 1.5 Verify Benchmarking Script

- **File**: `experiments/benchmark_against_anycam.py`
- **Check Trained Model** (lines 400-419):
  - Loads from `experiments/pose_head_experiment_results/full_run_eval/final_model.pt`
  - Uses `AnyCamWrapperWithAnyCaLib` with same architecture as training
  - Correctly loads `model_state_dict` from checkpoint
- **Check Baseline Model** (lines 424-448):
  - Loads from `pretrained_models/anycam_seq8/training_checkpoint_247500.pt`
  - Uses same wrapper but loads pretrained weights
  - Both models use AnyCaLib for consistency (not original candidate system)
- **Check Evaluation** (lines 108-143):
  - Uses `proc_poses` (the actual selected poses) for comparison
  - Correctly computes relative GT poses as `inv(pose1) @ pose2`
  - Compares rotation (3x3) and translation (3D vector) separately
- **Status**: ✓ CORRECT - proper model loading and fair comparison

### 1.6 Sanity Check Conclusion

**Experiment 1 is correctly implemented:**

- ✓ AnyCaLib provides single focal length per sequence (first frame)
- ✓ Pose head is reinitialized and isolated for training
- ✓ No 32-candidate system, direct focal length injection
- ✓ Unsupervised flow reprojection loss (same as original AnyCam)
- ✓ Benchmarking compares correct models on LightSpeed validation set
- ✓ Results showing 37% translation improvement are legitimate

---

## Phase 2: Implement Experiment 2

### 2.1 Create New Training Script

**File**: `experiments/train_pose_head_anycalib_exp2.py` (new file)

**Key modifications from Experiment 1:**

1. **Multi-Frame Dataset** (modify `ObjectronVideoDataset`):

   - Add `max_ahead` parameter (default: 3, meaning frames 1,2,3,4)
   - Modify `_load_frames_from_video()` to load `max_ahead + 1` frames instead of 2
   - Keep same split logic but load longer sequences

2. **Pose Composition Function** (new):
   ```python
   def compose_poses(pose_list: List[torch.Tensor]) -> torch.Tensor:
       """Compose consecutive poses: pose_1->3 = pose_1->2 @ pose_2->3"""
       composed = pose_list[0]
       for pose in pose_list[1:]:
           composed = composed @ pose
       return composed
   ```

3. **Multi-Frame Forward Pass Wrapper** (new class):
   ```python
   class AnyCamWrapperMultiFrame(AnyCamWrapperWithAnyCaLib):
       def __init__(self, ..., max_ahead=3):
           super().__init__(...)
           self.max_ahead = max_ahead
       
       def forward(self, data):
           # Run consecutive pairs (1->2, 2->3, 3->4)
           consecutive_poses = []
           for i in range(self.max_ahead):
               pair_data = extract_pair(data, i, i+1)
               result = super().forward(pair_data)
               consecutive_poses.append(result['proc_poses'])
           
           # Compose long-range poses (1->3, 1->4)
           composed_poses = []
           for ahead in range(2, self.max_ahead + 1):
               composed = compose_poses(consecutive_poses[:ahead])
               composed_poses.append(composed)
           
           return consecutive_poses, composed_poses
   ```

4. **Multi-Frame Loss Function** (new):
   ```python
   class MultiFramePoseLoss(nn.Module):
       def __init__(self, base_loss_config, max_ahead=3):
           self.base_loss = make_loss(base_loss_config)
           self.max_ahead = max_ahead
       
       def forward(self, data, consecutive_results, composed_poses):
           total_loss = 0
           
           # Loss for consecutive pairs (1->2, 2->3, 3->4)
           for i, result in enumerate(consecutive_results):
               loss, _, _ = self.base_loss(result)
               total_loss += loss
           
           # Loss for composed long-range poses (1->3, 1->4)
           for ahead, composed_pose in enumerate(composed_poses, start=2):
               # Reproject from frame 0 to frame 'ahead' using composed pose
               reprojection_data = prepare_reprojection(data, 0, ahead, composed_pose)
               loss, _, _ = self.base_loss(reprojection_data)
               total_loss += loss
           
           return total_loss / (self.max_ahead + len(composed_poses))
   ```

5. **CLI Arguments**:

   - `--max_ahead`: Number of frames ahead to predict (default: 3)
   - All other args same as Experiment 1

### 2.2 Modify Benchmarking Script

**File**: `experiments/benchmark_against_anycam.py`

**Add new comparison modes:**

1. **Dataset Selection**:

   - Extend `--dataset` choices: `['objectron', 'lightspeed']`
   - For `objectron`: Load from `objectron_split.json` test indices

2. **Multi-Model Comparison**:

   - Add `--exp1_model` argument: path to Experiment 1 model
   - Add `--exp2_model` argument: path to Experiment 2 model (the one being trained)
   - Keep `--baseline_checkpoint` for AnyCam baseline
   - Run evaluation on all three models
   - Generate comparison plots with 3 distributions

3. **Report Format**:
   ```
   Model                    | Rot Mean | Rot Median | Trans Mean | Trans Median
   -------------------------|----------|------------|------------|-------------
   AnyCam Baseline          | X.XX°    | X.XX°      | XX.XX°     | XX.XX°
   Experiment 1 (2-frame)   | X.XX°    | X.XX°      | XX.XX°     | XX.XX°
   Experiment 2 (multi-frame)| X.XX°   | X.XX°      | XX.XX°     | XX.XX°
   ```


### 2.3 Create Experiment Runner Script

**File**: `experiments/run_experiment_2.sh`

**Modes:**

- `test`: Quick test on 1-2 sequences, 10 epochs, `max_ahead=3`
- `small`: Train on 20 sequences, 30 epochs, `max_ahead=3`
- `full`: Train on all sequences, 50 epochs, `max_ahead=3`
- `full_extended`: Train on all sequences, 50 epochs, `max_ahead=6`

**Auto-benchmark**: After training, automatically run benchmark comparing to Experiment 1 and baseline

### 2.4 Expected Files Structure

```
experiments/
├── train_pose_head_anycalib.py          # Experiment 1 (existing)
├── train_pose_head_anycalib_exp2.py     # Experiment 2 (NEW)
├── benchmark_against_anycam.py           # Updated for 3-model comparison
├── run_experiment_2.sh                   # NEW runner script
└── pose_head_experiment_results/
    ├── full_run_eval/                    # Experiment 1 results
    │   └── final_model.pt
    └── exp2_full_run/                    # Experiment 2 results (NEW)
        ├── final_model.pt
        ├── loss_curve.png
        └── benchmark_results/
            ├── comparison_3models.png
            └── benchmark_report.txt
```

### 2.5 Key Technical Details

**Pose Composition Mathematics:**

```python
# Consecutive poses from AnyCam predictions
T_1to2 = model(frames[0:2])  # 4x4 transformation
T_2to3 = model(frames[1:3])  # 4x4 transformation
T_3to4 = model(frames[2:4])  # 4x4 transformation

# Composed long-range poses
T_1to3 = T_1to2 @ T_2to3      # Matrix multiplication
T_1to4 = T_1to2 @ T_2to3 @ T_3to4
```

**Flow Reprojection for Long-Range:**

```python
# For composed pose T_1to4, we need:
# 1. Depth from frame 1 (anchor)
# 2. Focal length from AnyCaLib (frame 1)
# 3. Composed pose T_1to4
# 4. Optical flow 1->4 (precomputed by UniMatch or composed)

induced_flow_1to4 = induce_flow_dist(
    depths=depth_frame1,
    projs=anycalib_focal,
    rel_poses=T_1to4,
    flow=None  # Will be computed
)

# Compare to observed flow (UniMatch)
loss_1to4 = L1(induced_flow_1to4, unimatch_flow_1to4)
```

**Data Loading Strategy:**

- For `max_ahead=3`: Load frames [i, i+1, i+2, i+3] as one sample
- Slide window: samples 0-3, 4-7, 8-11, ... (non-overlapping for full training)
- Initial test: samples 0-3, 1-4, 2-5, ... (overlapping for faster iteration)

### 2.6 Implementation Steps

1. Copy `train_pose_head_anycalib.py` to `train_pose_head_anycalib_exp2.py`
2. Modify dataset to load `max_ahead + 1` frames
3. Implement `compose_poses()` utility function
4. Create `AnyCamWrapperMultiFrame` class
5. Implement `MultiFramePoseLoss` class
6. Update training loop to handle multi-frame forward pass
7. Modify `benchmark_against_anycam.py` for 3-model comparison
8. Add Objectron test split as benchmark dataset option
9. Create `run_experiment_2.sh` with auto-benchmark
10. Test on small subset, then scale to full training

---

## Phase 3: Validation & Results

### 3.1 Initial Test Run

```bash
bash experiments/run_experiment_2.sh test
```

Expected: Loss decreases, no errors, produces checkpoint

### 3.2 Full Training Run

```bash
bash experiments/run_experiment_2.sh full
```

Expected: Converges in ~50 epochs, auto-runs benchmark

### 3.3 Success Criteria

- ✓ Training loss converges (similar or better than Exp 1)
- ✓ Rotation error: similar or better than Exp 1
- ✓ Translation error: better than Exp 1 (37% improvement was baseline)
- ✓ Benchmark shows clear comparison across 3 models

### 3.4 Expected Improvements

Based on hypothesis, Experiment 2 should show:

- Faster convergence (more constraints)
- Better translation accuracy (long-range consistency)
- Similar or better rotation accuracy

### To-dos

- [ ] Complete sanity check of Experiment 1 (AnyCaLib integration, pose head, loss, benchmarking)
- [ ] Create train_pose_head_anycalib_exp2.py with multi-frame dataset and wrapper
- [ ] Implement compose_poses() function and AnyCamWrapperMultiFrame class
- [ ] Implement MultiFramePoseLoss with consecutive + composed reprojection losses
- [ ] Modify benchmark_against_anycam.py for 3-model comparison (Exp2 vs Exp1 vs Baseline) and add Objectron test split option
- [ ] Create run_experiment_2.sh with test/small/full modes and auto-benchmarking
- [ ] Run test mode on 1-2 sequences to verify implementation
- [ ] Run full training with all sequences and auto-benchmark