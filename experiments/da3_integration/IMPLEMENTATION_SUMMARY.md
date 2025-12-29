# DA3 Integration Implementation Summary

**Date**: December 2025  
**Status**: Implementation Complete

## Overview

This document summarizes the complete implementation of the Depth Anything 3 (DA3) calibration head integration into AnyCam. All components have been implemented according to the plan in `DEPTH_ANYTHING_3_INTEGRATION_PLAN.md`.

## Files Created

### Phase 1: Core Components
✅ **All model components created in `experiments/models/`**:
- `__init__.py` - Module exports
- `camera_encoder.py` - Camera parameter encoder
- `visual_camera_mixing.py` - Visual-camera mixing (reverse conditioning)
- `sequence_aggregation.py` - Sequence-level aggregation
- `camera_decoder.py` - Camera parameter decoder
- `da3_calibration_head.py` - Complete calibration head

### Phase 2: Integration
✅ **Modified files**:
- `anycam/models/anycam.py` - Added DA3 calibration head support (optional, backward compatible)
- `anycam/models/__init__.py` - Updated `make_pose_predictor` to pass `use_da3_calibration`
- `experiments/train_pose_head_anycalib.py` - Added `AnyCamWrapperWithDA3Calibration` wrapper class

### Phase 3: Training Scripts
✅ **All training scripts created**:
- `train_calibration_head_da3_stage1.py` - Stage 1: Mean calibration learning
- `train_calibration_head_da3_stage2.py` - Stage 2: Visual-conditioned calibration
- `train_calibration_head_da3_stage3.py` - Stage 3: End-to-end flow reprojection

### Phase 4: Evaluation
✅ **Evaluation scripts created**:
- `evaluate_da3_calibration.py` - Calibration accuracy evaluation
- `benchmark_against_anycam.py` - Updated to support DA3 models

## Key Features Implemented

### 1. DA3 Calibration Head Architecture
- **Camera Encoder**: Normalizes and encodes camera parameters to tokens (DA3-style)
- **Visual-Camera Mixing**: Self-attention on concatenated tokens (supervisor's guidance)
  - Concatenates camera tokens + visual tokens → all_tokens
  - Runs self-attention on all_tokens
  - Extracts only camera tokens (visual token updates discarded, DA3-style)
- **Sequence Aggregation**: Mean pooling with attention mask support (supervisor's suggestion, default)
  - Supports variable-length sequences via attention masks
  - Masked mean pooling for proper aggregation
- **Camera Decoder**: Decodes tokens back to camera parameters (DA3-style)

### 2. Staged Training
- **Stage 1**: Train encoder/aggregation/decoder to output mean calibration (MSE loss)
  - Validation: Enabled with train/val split
  - Loss curves: Both training and validation losses plotted
  - Overfitting detection: Automatic warning if val_loss > 1.5 * train_loss
- **Stage 2**: Unfreeze visual-camera mixing, train with visual tokens (MSE loss)
  - Validation: Enabled with train/val split
  - Loss curves: Both training and validation losses plotted
- **Stage 3**: End-to-end training with flow reprojection loss (self-supervised)
  - Validation: Enabled with train/val split
  - Loss curves: Both training and validation losses plotted

### 3. Integration Points
- **AnyCam Model**: Optional DA3 support via `use_da3_calibration` flag
- **Training Wrapper**: `AnyCamWrapperWithDA3Calibration` extends existing wrapper
- **Backward Compatible**: Original 32-candidate system remains functional

### 4. Evaluation and Benchmarking
- **Calibration Evaluation**: Relative error metrics for Stage 1 & 2
- **Pose Benchmarking**: Full integration with existing benchmark script
- **Results Organization**: Clear folder structure for easy thesis reference

### 5. Technical Implementation Details
- **Variable-Length Sequences**: Custom collate function handles different frame counts per video
  - Pads sequences to max length in batch
  - Creates attention masks (1 for valid, 0 for padding)
  - Masked mean pooling in sequence aggregation
- **Validation Support**: All training stages include validation
  - Train/val split from `experiments/objectron_split.json`
  - Validation loss tracked and plotted
  - Overfitting detection in training summary
- **Frame Loading**: Stage 1 & 2 load all available frames (no artificial limits)
  - `num_frames` parameter removed from Stage 1 & 2 datasets
  - Only Stage 3 uses `num_frames` (for pose prediction batch size)

## Training Data Configuration

- **Dataset**: Objectron with all available frames from all available sequences
- **Stage 1 & 2**: Loads ALL frames from each video (no `num_frames` limit)
  - Runs AnyCalib on every frame to get per-frame predictions
  - Computes GT mean calibration from all frames per sequence
  - Uses custom collate function with attention masks for variable-length sequences
- **Stage 3**: Uses frame pairs for pose prediction
  - `extract_all_pairs=True`, `max_pairs_per_video=None` (unlimited)
  - Frame Pairs: All consecutive pairs (0-1, 1-2, 2-3, ...) from each video
  - Self-Supervised: Uses flow reprojection loss (no GT calibration needed)
- **Validation**: All stages support train/val splits (15 validation sequences from Objectron)

## Loss Functions

### Stage 1 & 2
```python
loss = F.mse_loss(predicted_mean_calibration, gt_mean_calibration)
```

### Stage 3
```python
# Flow reprojection loss (self-supervised)
from anycam.loss import make_loss
criterion = make_loss(loss_config)
loss, loss_dict = criterion.compute_pose_loss(pose_result)
```

## Hyperparameters

### Stage 1
- Learning rate: 1e-4
- Batch size: 8-16
- Epochs: 50
- Optimizer: Adam

### Stage 2
- Learning rate: 5e-5 (lower than Stage 1)
- Batch size: 4-8 (smaller due to visual tokens)
- Epochs: 50
- Optimizer: Adam

### Stage 3
- Learning rate: 1e-5 (very low for fine-tuning)
- Batch size: 2-4 (memory intensive)
- Epochs: 50-100
- Optimizer: Adam

## Usage Examples

### Stage 1 Training (Docker Container)
**Note**: Stage 1 loads ALL frames from each video (no `--num_frames` argument needed)
```bash
python experiments/train_calibration_head_da3_stage1.py \
    --objectron_videos /data/thesis/Objectron/videos \
    --objectron_gt /data/thesis/Objectron/processed_gt \
    --num_epochs 50 \
    --batch_size 8 \
    --learning_rate 1e-4 \
    --save_dir experiments/da3_integration/stage1_training
```

### Stage 2 Training (Docker Container)
**Note**: Stage 2 loads ALL frames from each video (no `--num_frames` argument needed)
```bash
python experiments/train_calibration_head_da3_stage2.py \
    --objectron_videos /data/thesis/Objectron/videos \
    --objectron_gt /data/thesis/Objectron/processed_gt \
    --stage1_checkpoint experiments/da3_integration/stage1_training/checkpoints/final_model.pt \
    --num_epochs 50 \
    --batch_size 4 \
    --learning_rate 5e-5 \
    --save_dir experiments/da3_integration/stage2_training
```

### Stage 3 Training (Docker Container)
**Note**: Stage 3 uses `--num_frames` to control how many frames go into pose predictor
```bash
python experiments/train_calibration_head_da3_stage3.py \
    --objectron_videos /data/thesis/Objectron/videos \
    --objectron_gt /data/thesis/Objectron/processed_gt \
    --stage2_checkpoint experiments/da3_integration/stage2_training/checkpoints/final_model.pt \
    --num_epochs 50 \
    --batch_size 2 \
    --learning_rate 1e-5 \
    --save_dir experiments/da3_integration/stage3_training
```

### Benchmarking
```bash
python experiments/benchmark_against_anycam.py \
    --da3_stage3_model experiments/da3_integration/stage3_training/checkpoints/final_model.pt \
    --save_dir experiments/da3_integration/benchmark_results/stage3_vs_baseline \
    --dataset lightspeed
```

## Results Organization

All results are organized in `experiments/da3_integration/`:
- Training results: `stage1_training/`, `stage2_training/`, `stage3_training/`
- Benchmark results: `benchmark_results/stage*_vs_baseline/`
- Clear naming: "DA3 Stage 1", "DA3 Stage 2", "DA3 Stage 3" (differentiated from Exp1, Exp2)

## Evaluation Metrics

### Stage 1 & 2 Evaluation
Training includes evaluation of relative error to target mean calibration:
- `calibration_accuracy.json` (training set) contains:
  - Focal length errors (fx, fy)
  - Principal point errors (cx, cy)
  - Overall relative error
  - Predictions and targets for analysis
- `val_calibration_accuracy.json` (validation set) contains same metrics

### Training Outputs
All stages generate:
- `training_log.txt` - Detailed epoch-by-epoch logs
- `loss_history.json` - Training and validation loss history
- `loss_curve.png` - Plot with both train and validation curves
- `training_summary.txt` - Summary statistics with overfitting detection
- `checkpoints/` - Model checkpoints every 10 epochs + final model

## Implementation Status

✅ **Phase 1**: Core Components - COMPLETE  
✅ **Phase 2**: Integration - COMPLETE  
✅ **Phase 3**: Training Scripts - COMPLETE  
✅ **Phase 4**: Evaluation - COMPLETE  

## Next Steps

1. Run Stage 1 training to verify calibration head learns mean calibration
2. Run Stage 2 training to verify visual tokens improve accuracy
3. Run Stage 3 training for end-to-end optimization
4. Benchmark all stages against AnyCam baseline
5. Analyze results and document findings for thesis

## Detailed Implementation: Bugs, Fixes, and Workarounds

This section documents all implementation challenges, bugs encountered, and their solutions during the development process.

### Stage 2 Implementation: Complete Data Flow and Dimensions

#### **Stage 2 Overview**
Stage 2 trains the DA3 calibration head with visual tokens to improve calibration accuracy. The key difference from Stage 1 is that visual features from DINOv2 are now used to condition the camera parameter predictions.

#### **Complete Data Flow (Stage 2)**

**Step 1: Dataset Preprocessing**
```
Video File → Load ALL frames → [N frames, H, W, 3]
    ↓
Chunked Processing (chunk_size=16 to avoid OOM):
    For each chunk:
        ↓
    AnyCalib Inference → [chunk_size, 4] (fx, fy, cx, cy per frame)
        ↓
    DINOv2 Visual Token Extraction → [chunk_size, 384] (CLS tokens)
        ↓
    Move to CPU, clear GPU cache
    ↓
Concatenate all chunks → [N, 4] + [N, 384]
    ↓
Compute GT Mean Calibration → [1, 4] (mean of all N frames)
    ↓
Store in self.sequences:
    - visual_tokens: [1, N, 384] (numpy array, CPU)
    - anycalib_predictions: [N, 4] (numpy array, CPU)
    - gt_mean_calibration: [1, 4] (numpy array, CPU)
    - image_size: (H, W) tuple
```

**Step 2: DataLoader with Custom Collate Function**
```
__getitem__(idx) returns:
    - visual_tokens: [N, 384] (torch.Tensor, variable N per sequence)
    - anycalib_predictions: [N, 4] (torch.Tensor, variable N)
    - gt_mean_calibration: [1, 4] (torch.Tensor)
    - image_size: (H, W) tuple

collate_variable_length_stage2(batch) processes:
    Input: List of Dicts (each with variable N)
    ↓
    Find max_N = max(N_i for all sequences in batch)
    ↓
    Pad all sequences to max_N:
        - visual_tokens: [B, max_N, 384] (zeros for padding)
        - anycalib_predictions: [B, max_N, 4] (zeros for padding)
        - attention_mask: [B, max_N] (1.0 for valid, 0.0 for padding)
    ↓
    Stack GT means: [B, 1, 4]
    ↓
    Return batched dict with attention_mask
```

**Step 3: Forward Pass Through Model**
```
Input to DA3CalibrationHead.forward():
    - visual_tokens: [B, max_N, 384]
    - anycalib_predictions: [B, max_N, 4]
    - image_size: (H, W)
    - attention_mask: [B, max_N]
    - use_visual_conditioning: True
    ↓
Camera Encoder:
    anycalib_predictions [B, max_N, 4] → camera_tokens [B, max_N, 256]
    ↓
Visual-Camera Mixing:
    visual_tokens [B, max_N, 384] → projected [B, max_N, 256]
    camera_tokens [B, max_N, 256]
    ↓
    Concatenate: [camera_tokens, visual_tokens] → [B, 2*max_N, 256]
    ↓
    Self-attention on all tokens → [B, 2*max_N, 256]
    ↓
    Extract only camera tokens (first max_N) → [B, max_N, 256]
    ↓
Sequence Aggregation (with attention_mask):
    per_frame_tokens [B, max_N, 256] + attention_mask [B, max_N]
    ↓
    Masked mean pooling (only valid frames) → [B, 1, 256]
    ↓
Camera Decoder:
    aggregated_token [B, 1, 256] → camera_params [B, 1, 4]
    ↓
Output: [B, 1, 4] (fx, fy, cx, cy)
```

**Step 4: Loss Computation**
```
pred_calibration: [B, 1, 4]
gt_mean_calibration: [B, 1, 4]
    ↓
loss = F.mse_loss(pred_calibration, gt_mean_calibration)
```

### Critical Bugs and Fixes

#### **Bug 1: xFormers GPU Compatibility (RTX 5090)**
**Problem**: 
- Original implementation used Facebook's `torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')`
- This version uses xFormers for memory-efficient attention
- xFormers only supports GPU compute capability ≤ 9.0
- RTX 5090 has compute capability 12.0 → **incompatible**
- Error: `No operator found for memory_efficient_attention_forward with inputs: ... requires device with capability <= (9, 0) but your GPU has capability (12, 0)`

**Initial Attempt (Failed)**:
- Tried mocking `xformers` module at import level
- Created `_MockXFormers` class to block xFormers import
- **Problem**: Mock was too aggressive, broke PyTorch's internal inspection
- Error: `ImportError: xFormers disabled for GPU compatibility` during `torch.__init__`

**Solution**:
- Switched to **HuggingFace's transformers DINOv2** (`facebook/dinov2-small`)
- HuggingFace version uses native PyTorch attention (SDPA), not xFormers
- Compatible with all GPUs including RTX 5090
- Code:
  ```python
  from transformers import AutoModel, AutoImageProcessor
  self.visual_backbone = AutoModel.from_pretrained('facebook/dinov2-small').to(device).eval()
  self.vis_dim = 384  # DINOv2-S output dimension
  ```
- Also set environment variables: `os.environ["XFORMERS_DISABLED"] = "1"` (for other libraries)

**Data Flow Change**:
- Visual token extraction now uses HuggingFace API:
  ```python
  outputs = self.visual_backbone(inputs)  # HuggingFace forward
  cls_tokens = outputs.last_hidden_state[:, 0, :]  # Extract CLS token
  ```
- Input normalization: ImageNet stats (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
- Input size: Resized to 224x224 (HuggingFace DINOv2 requirement)

#### **Bug 2: Variable-Length Sequence Collation**
**Problem**:
- Different videos have different numbers of frames (e.g., 263 vs 453 frames)
- PyTorch's default `DataLoader` tries to `torch.stack()` tensors of different sizes
- Error: `RuntimeError: stack expects each tensor to be equal size, but got [218, 384] at entry 0 and [187, 384] at entry 1`

**Solution**:
- Created custom `collate_variable_length_stage2()` function
- Pads all sequences to the maximum length in the batch
- Creates `attention_mask` to mark valid vs padded positions
- Implementation:
  ```python
  def collate_variable_length_stage2(batch: List[Dict]) -> Dict:
      max_len = max(item['visual_tokens'].shape[0] for item in batch)
      batch_size = len(batch)
      vis_dim = batch[0]['visual_tokens'].shape[1]  # 384
      
      # Pad to max_len
      padded_visual = torch.zeros(batch_size, max_len, vis_dim)
      padded_preds = torch.zeros(batch_size, max_len, 4)
      attention_mask = torch.zeros(batch_size, max_len)
      
      for i, item in enumerate(batch):
          n_frames = item['visual_tokens'].shape[0]
          padded_visual[i, :n_frames, :] = item['visual_tokens']
          padded_preds[i, :n_frames, :] = item['anycalib_predictions']
          attention_mask[i, :n_frames] = 1.0  # Valid positions
      
      return {
          'visual_tokens': padded_visual,  # [B, max_N, 384]
          'anycalib_predictions': padded_preds,  # [B, max_N, 4]
          'attention_mask': attention_mask,  # [B, max_N]
          'gt_mean_calibration': torch.stack([item['gt_mean_calibration'] for item in batch]),  # [B, 1, 4]
          'image_size': (torch.tensor([item['image_size'][0] for item in batch]),
                        torch.tensor([item['image_size'][1] for item in batch])),
      }
  ```
- Updated `DataLoader` to use custom collate:
  ```python
  train_dataloader = DataLoader(..., collate_fn=collate_variable_length_stage2)
  ```

#### **Bug 3: Attention Mask Propagation**
**Problem**:
- After adding padding and attention masks, the mask wasn't being passed to the model
- Sequence aggregation was averaging over padded positions, corrupting the output

**Solution**:
- Updated `DA3CalibrationHead.forward()` to accept `attention_mask` parameter
- Updated `SequenceCameraAggregation.forward()` to use masked mean pooling:
  ```python
  def forward(self, per_frame_tokens, attention_mask=None):
      if attention_mask is not None:
          mask_expanded = attention_mask.unsqueeze(-1).float()  # [B, N, 1]
          masked_tokens = per_frame_tokens * mask_expanded
          sum_tokens = masked_tokens.sum(dim=1, keepdim=True)  # [B, 1, D_cam]
          num_valid = mask_expanded.sum(dim=1, keepdim=True)  # [B, 1, 1]
          aggregated = sum_tokens / (num_valid + 1e-8)  # Avoid division by zero
          return aggregated
  ```
- Updated training and validation loops to pass `attention_mask`:
  ```python
  pred_calibration = model(
      visual_tokens=visual_tokens,
      anycalib_predictions=anycalib_preds,
      image_size=(H, W),
      use_visual_conditioning=True,
      attention_mask=attention_mask,  # Pass mask
  )
  ```

#### **Bug 4: Checkpoint Loading Dimension Mismatch**
**Problem**:
- Stage 1 was trained with `vis_dim=768` (default, unused since visual mixing was frozen)
- Stage 2 uses `vis_dim=384` (DINOv2-S output dimension)
- Loading Stage 1 checkpoint failed: `RuntimeError: size mismatch for visual_camera_mixing.visual_projection.weight: copying a param with shape torch.Size([256, 768]) from checkpoint, the shape in current model is torch.Size([256, 384])`

**Solution**:
- Implemented smart checkpoint loading that filters mismatched keys
- Since `visual_camera_mixing` was frozen in Stage 1, its weights weren't trained anyway
- Safe to skip and initialize randomly for Stage 2
- Implementation:
  ```python
  state_dict = checkpoint.get('model_state_dict', checkpoint)
  model_state = model.state_dict()
  filtered_state_dict = {}
  skipped_keys = []
  
  for k, v in state_dict.items():
      if k in model_state:
          if v.shape == model_state[k].shape:
              filtered_state_dict[k] = v  # Load matching weights
          else:
              skipped_keys.append(f"{k}: checkpoint {v.shape} vs model {model_state[k].shape}")
      else:
          skipped_keys.append(f"{k}: not in model")
  
  model.load_state_dict(filtered_state_dict, strict=False)
  ```
- Result: Loads `camera_encoder`, `sequence_aggregation`, `camera_decoder` from Stage 1
- Skips `visual_camera_mixing` (initialized randomly, then trained in Stage 2)

#### **Bug 5: Out of Memory (OOM) During Preprocessing**
**Problem**:
- Stage 2 preprocessing loads all frames from a video (potentially 100-500 frames)
- Running DINOv2 and AnyCalib on all frames at once → OOM on 24GB VRAM card

**Solution**:
- Implemented **chunked processing** with `chunk_size=16`
- Process frames in small batches, move results to CPU, clear GPU cache
- Implementation:
  ```python
  self.chunk_size = 16  # Process 16 frames at a time
  
  for chunk_start in range(0, num_frames, self.chunk_size):
      chunk_end = min(chunk_start + self.chunk_size, num_frames)
      chunk_frames = frames[chunk_start:chunk_end]
      
      # Process chunk
      chunk_images = ...  # [1, chunk_size, 3, H, W]
      chunk_visual = self._extract_visual_tokens(chunk_images)  # [1, chunk_size, 384]
      chunk_anycalib = self.anycalib_model.predict_intrinsics(chunk_images)  # [1, chunk_size, 4]
      
      # Move to CPU immediately
      chunk_visual = chunk_visual.cpu().numpy()
      chunk_anycalib = chunk_anycalib.cpu().numpy()
      
      # Clear GPU cache
      torch.cuda.empty_cache()
      
      # Accumulate
      all_visual.append(chunk_visual)
      all_anycalib.append(chunk_anycalib)
  ```

#### **Bug 6: Removed Unused Depth Predictor**
**Problem**:
- Initial implementation loaded AnyCam backbone + depth predictor for visual tokens
- Depth predictor consumed significant VRAM but wasn't used
- AnyCam backbone expects 6-channel input (RGB + depth), but we only have RGB

**Solution**:
- Removed depth predictor entirely
- Switched to standalone DINOv2 (3-channel RGB input)
- Simplified preprocessing pipeline
- Reduced VRAM usage significantly

### Visual-Camera Mixing Implementation Fix

**Original (Incorrect) Approach**:
- Used cross-attention where camera tokens query visual tokens
- Implementation: `camera_tokens = cross_attention(camera_tokens, visual_tokens)`

**Supervisor's Guidance**:
- "Put all tokens to the same set and run regular attention. Then forget about updates of visual tokens. Like the depth anything 3 style."

**Corrected Implementation**:
```python
class VisualCameraMixing(nn.Module):
    def forward(self, visual_tokens, camera_tokens):
        # Project visual tokens to camera dimension
        projected_visual = self.vis_proj(visual_tokens)  # [B, N, D_cam]
        
        # Concatenate: [camera_tokens, visual_tokens]
        all_tokens = torch.cat([camera_tokens, projected_visual], dim=1)  # [B, 2*N, D_cam]
        
        # Self-attention on all tokens
        for norm1, attn, norm2, ffn in self.layers:
            all_tokens = all_tokens + attn(norm1(all_tokens), norm1(all_tokens), norm1(all_tokens))[0]
            all_tokens = all_tokens + ffn(norm2(all_tokens))
        
        # Extract only camera tokens (discard visual token updates)
        updated_camera_tokens = all_tokens[:, :N_cam, :]  # [B, N, D_cam]
        return updated_camera_tokens
```

**Key Points**:
- All tokens (camera + visual) are in the same set
- Self-attention updates all tokens
- Only camera token updates are kept (visual updates discarded)
- Matches DA3's camera conditioning mechanism

### Frame Loading Fix

**Original Problem**:
- Stage 1 & 2 datasets had `num_frames` parameter limiting frame loading
- `_load_frames_from_video()` loop: `while len(frames) < self.num_frames:`
- This limited training to only 2 frames per sequence (default `num_frames=2`)
- Mean calibration was computed from only 2 AnyCalib predictions instead of all frames

**Fix**:
- Removed `num_frames` parameter from Stage 1 & 2 datasets
- Changed loop to: `while True:` (load until video ends)
- Now loads ALL available frames from each video
- Better mean calibration estimates using all frames (potentially 100-500 frames per video)

**Note**: `num_frames` is still used in Stage 3, but only to control how many frames go into the pose predictor at once (for pose prediction, not for training data loading).

### Image Size Extraction Fix

**Problem**:
- DataLoader collates tuples as `(tensor([H1, H2, ...]), tensor([W1, W2, ...]))`
- Not as `[(H1, W1), (H2, W2), ...]`
- Code tried: `H, W = image_sizes[0]` → failed

**Fix**:
```python
# Correct extraction from collated tuples
H = image_sizes[0][0].item() if isinstance(image_sizes[0], torch.Tensor) else image_sizes[0]
W = image_sizes[1][0].item() if isinstance(image_sizes[1], torch.Tensor) else image_sizes[1]
```

### Unbound Variable Fix

**Problem**:
- `plot_loss_curve()` had `val_epochs` and `val_losses` only defined inside conditional
- If `val_loss_history` was empty, variables were unbound → `UnboundLocalError`

**Fix**:
- Initialize `val_epochs = []` and `val_losses = []` before conditional block
- Ensures variables are always defined

### Summary of All Dimension Changes

**Stage 2 Data Dimensions**:
- Visual tokens (DINOv2-S): `[B, max_N, 384]` (was 768 in original plan)
- AnyCalib predictions: `[B, max_N, 4]`
- Camera tokens: `[B, max_N, 256]`
- Aggregated token: `[B, 1, 256]`
- Output calibration: `[B, 1, 4]`
- Attention mask: `[B, max_N]` (1.0 for valid, 0.0 for padding)

**Key Changes from Original Plan**:
1. `vis_dim`: 768 → 384 (DINOv2-S instead of AnyCam backbone)
2. Visual token source: AnyCam backbone → HuggingFace DINOv2
3. Input channels: 6 (RGB+depth) → 3 (RGB only)
4. Variable-length handling: Added padding + attention masks
5. Checkpoint loading: Smart filtering for dimension mismatches

## Notes

- All implementations are backward compatible
- DA3 is opt-in via `use_da3_calibration=False` (default)
- Original experiments (Exp1, Exp2) remain unchanged
- Clear separation between DA3 experiments and previous work
- Container-ready: All paths updated for Docker container usage (`/data/thesis/...`)

