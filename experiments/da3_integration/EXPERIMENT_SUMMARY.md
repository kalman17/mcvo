# DA3 Calibration Head Experiment - Complete Summary

**Date**: December 2025  
**Status**: Implementation Complete - Validation Phase  
**Dataset**: Objectron (small-scale validation experiment)

## Overview

The DA3 (Depth Anything 3) calibration head is a multi-frame camera conditioning architecture that learns to predict camera intrinsics (fx, fy, cx, cy) from sequences of frames. The implementation uses a **three-stage training approach** to gradually introduce complexity:

1. **Stage 1**: Learn mean aggregation (no visual tokens)
2. **Stage 2**: Add visual conditioning (DINOv2-small features)
3. **Stage 3**: End-to-end training with flow reprojection loss

## Architecture Components

### Token Definitions

Before diving into the architecture, it's important to understand the two types of tokens used:

- **Visual Tokens** (`[B, N, 384]`): Rich feature representations extracted from images using DINOv2-small vision transformer. Each frame produces a 384-dimensional CLS (classification) token that captures semantic and geometric information about the scene content (objects, structures, depth cues, etc.). These tokens represent "what the camera sees" in the scene.

- **Camera Tokens** (`[B, N, 256]`): Embedding representations of camera parameters (fx, fy, cx, cy). These are learned representations in a 256-dimensional space that allow the model to capture relationships between camera parameters. These tokens represent "how the camera is configured" (intrinsics).

The architecture combines these two types of information: visual tokens provide scene context, while camera tokens represent the camera configuration. The goal is to predict the optimal camera parameters for the sequence.

### DA3CalibrationHead Structure

```
Input: Visual Tokens [B, N, 384] + AnyCalib Predictions [B, N, 4]
    ↓
Camera Encoder: [B, N, 4] → [B, N, 256]
    ↓
Visual-Camera Mixing (if enabled): [B, N, 256] + [B, N, 384] → [B, N, 256]
    ↓
Sequence Aggregation: [B, N, 256] → [B, 1, 256]
    ↓
Camera Decoder: [B, 1, 256] → [B, 1, 4]
    ↓
Output: Camera Parameters [B, 1, 4] (fx, fy, cx, cy)
```

**Key Dimensions**:
- `vis_dim = 384`: DINOv2-small CLS token dimension
- `cam_dim = 256`: Camera token dimension (internal representation)
- `hidden_dim = 128`: Hidden layer dimension in encoder/decoder
- `num_mixing_layers = 2`: Self-attention layers in visual-camera mixing

### Component Details

1. **Camera Encoder** (`CameraEncoder`)
   - **Purpose**: Maps raw camera parameters to a rich embedding space
   - **Input**: `[B, N, 4]` - Raw camera parameters (fx, fy, cx, cy) in pixels
   - **Process**: 
     - Normalizes parameters by image size (fx, fy by max(H,W)/2, cx by W, cy by H)

     # TODO: better to all be  (normalized) by the same one, eg max(H,W)/2, OR W OR H

     - Projects to embedding space via MLP: `Linear(4 → 128) → ReLU → Linear(128 → 256)`
   - **Output**: `[B, N, 256]` - Camera tokens (embedding representations)
   - **Model Type**: MLP (Multi-Layer Perceptron)
   - **Why increase dimensions?**: Similar to word embeddings in NLP, increasing from 4 to 256 dimensions creates a richer representation space where:
     - The model can learn complex relationships between camera parameters
     - Attention mechanisms can operate effectively
     - Non-linear interactions can be captured

2. **Visual-Camera Mixing** (`VisualCameraMixing`)
   - **Purpose**: Conditions camera tokens with visual information using DA3-style self-attention
   - **Input**: 
     - Visual tokens: `[B, N, 384]` - Scene features from DINOv2-small
     - Camera tokens: `[B, N, 256]` - Camera parameter embeddings
   - **Process**:
     1. Project visual tokens to match camera dimension: `[B, N, 384] → [B, N, 256]`
     2. Concatenate: `[camera_tokens, visual_tokens] → [B, 2*N, 256]`
     3. Self-attention: All tokens attend to each other (transformer mechanism)
     4. Extract only camera tokens: `[B, 2*N, 256] → [B, N, 256]` (discard visual updates)
   - **Output**: `[B, N, 256]` - Visual-conditioned camera tokens
   - **Model Type**: Transformer (MultiheadAttention with 8 heads, 2 layers)
   - **Why this design?**: 
     - Visual tokens provide context about scene content and geometry
     - Camera tokens get updated based on visual information (reverse conditioning)
     - Only camera token updates are kept (visual tokens are auxiliary, discarded after mixing)
     - Matches DA3's camera conditioning mechanism

3. **Sequence Aggregation** (`SequenceCameraAggregation`)
   - **Purpose**: Reduces per-frame tokens to a single sequence-level representation
   - **Input**: `[B, N, 256]` - Per-frame camera tokens (one token per frame)
   - **Process**:
     - **Default (Mean Pooling)**: Simple average across frames
       - `[B, N, 256] → mean(dim=1) → [B, 1, 256]`
       - Supports masking for variable-length sequences (ignores padding)
     - **Optional (Learnable Token)**: Cross-attention with learnable query token
       - Learnable token queries all frame tokens via cross-attention
   - **Output**: `[B, 1, 256]` - Sequence-level camera token
   - **Model Type**: Mean pooling (simple operation) or Cross-attention (if learnable token used)
   - **Why aggregate?**: 
     - Combines information from all frames into a single sequence-level prediction
     - For fixed cameras, we want one calibration per sequence, not per frame
     - Mean pooling is simple and effective (supervisor's recommendation)

     TODO: add a switch true flase flag to add or not learnable token for sequence aggregation.

4. **Camera Decoder** (`CameraDecoder`)
   - **Purpose**: Maps sequence-level embedding back to camera parameters
   - **Input**: `[B, 1, 256]` - Sequence-level camera token
   - **Process**:
     - Projects via MLP: `Linear(256 → 128) → ReLU → Linear(128 → 4)`
     - Denormalizes by image size to get absolute parameters in pixels
   - **Output**: `[B, 1, 4]` - Camera parameters (fx, fy, cx, cy) in pixels
   - **Model Type**: MLP (Multi-Layer Perceptron)
   - **Why decrease dimensions?**: Maps the rich embedding representation back to the 4 camera parameters needed for camera calibration

### Model Architecture Summary

| Component | Model Type | Architecture | Purpose |
|-----------|-----------|--------------|---------|
| **Camera Encoder** | MLP | `Linear(4→128) → ReLU → Linear(128→256)` | Embed raw parameters into rich representation space |
| **Visual-Camera Mixing** | Transformer | `MultiheadAttention` (8 heads, 2 layers) | Condition camera tokens with visual information |
| **Sequence Aggregation** | Mean Pooling | Simple average across frames | Combine per-frame info into sequence-level representation |
| **Camera Decoder** | MLP | `Linear(256→128) → ReLU → Linear(128→4)` | Map embedding back to camera parameters |

## Complete Data Flow by Stage

### Stage 1: Mean Calibration Learning

**Objective**: Learn to aggregate per-frame AnyCalib predictions into sequence-level mean.

**Training Data Flow**:
```
Video File → Load ALL frames → [N frames, H, W, 3]
    ↓
AnyCalib Inference (per frame) → [N, 4] (fx, fy, cx, cy per frame)
    ↓
Compute GT Mean → [1, 4] (mean of all N frames)
    ↓
DataLoader (batch_size=B):
    - anycalib_predictions: [B, N, 4] (variable N per sequence)
    - gt_mean_calibration: [B, 1, 4]
    ↓
Custom Collate (padding to max_N):
    - anycalib_predictions: [B, max_N, 4]
    - attention_mask: [B, max_N] (1.0 for valid, 0.0 for padding)
    ↓
DA3CalibrationHead.forward():
    - visual_tokens: None (not used in Stage 1)
    - anycalib_predictions: [B, max_N, 4]
    - use_visual_conditioning: False
    ↓
Camera Encoder: [B, max_N, 4] → [B, max_N, 256]
    ↓
(Skip Visual-Camera Mixing)
    ↓
Sequence Aggregation (masked): [B, max_N, 256] → [B, 1, 256]
    ↓
Camera Decoder: [B, 1, 256] → [B, 1, 4]
    ↓
Loss: MSE(predicted_mean [B, 1, 4], gt_mean [B, 1, 4])
```

**Key Features**:
- Loads ALL frames from each video (no limit)
- No visual tokens (visual_camera_mixing frozen)
- Only encoder, aggregation, and decoder trained
- Loss: MSE against GT mean calibration

**Hyperparameters**:
- Learning rate: `1e-4`
- Batch size: `8-16`
- Epochs: `50`
- Optimizer: Adam

### Stage 2: Visual-Conditioned Calibration

**Objective**: Learn to leverage visual features for improved calibration accuracy.

**Training Data Flow**:
```
Video File → Load ALL frames → [N frames, H, W, 3]
    ↓
Chunked Processing (chunk_size=16 to avoid OOM):
    For each chunk:
        ↓
    AnyCalib Inference → [chunk_size, 4]
        ↓
    DINOv2-small (HuggingFace) → CLS tokens → [chunk_size, 384]
        ↓
    Move to CPU, clear GPU cache
    ↓
Concatenate all chunks → [N, 4] + [N, 384]
    ↓
Compute GT Mean → [1, 4]
    ↓
DataLoader (batch_size=B):
    - visual_tokens: [B, N, 384] (variable N)
    - anycalib_predictions: [B, N, 4]
    - gt_mean_calibration: [B, 1, 4]
    ↓
Custom Collate (padding to max_N):
    - visual_tokens: [B, max_N, 384]
    - anycalib_predictions: [B, max_N, 4]
    - attention_mask: [B, max_N]
    ↓
DA3CalibrationHead.forward():
    - visual_tokens: [B, max_N, 384]
    - anycalib_predictions: [B, max_N, 4]
    - use_visual_conditioning: True
    ↓
Camera Encoder: [B, max_N, 4] → [B, max_N, 256]
    ↓
Visual-Camera Mixing:
    - Project visual: [B, max_N, 384] → [B, max_N, 256]
    - Concatenate: [camera_tokens, visual_tokens] → [B, 2*max_N, 256]
    - Self-attention → [B, 2*max_N, 256]
    - Extract camera tokens: [B, max_N, 256]
    ↓
Sequence Aggregation (masked): [B, max_N, 256] → [B, 1, 256]
    ↓
Camera Decoder: [B, 1, 256] → [B, 1, 4]
    ↓
Loss: MSE(predicted_mean [B, 1, 4], gt_mean [B, 1, 4])
```

**Key Features**:
- Loads ALL frames from each video
- Visual tokens from DINOv2-small (HuggingFace, vis_dim=384)
- Visual-camera mixing now trained (unfrozen)
- Chunked processing to avoid OOM (16 frames at a time)
- Loss: MSE against GT mean calibration

**Hyperparameters**:
- Learning rate: `5e-5` (lower than Stage 1)
- Batch size: `4-8` (smaller due to visual tokens)
- Epochs: `50`
- Optimizer: Adam

**Checkpoint Loading**:
- Loads Stage 1 weights for encoder, aggregation, decoder
- Skips visual_camera_mixing (dimension mismatch: Stage 1 had vis_dim=768 unused)
- Initializes visual_camera_mixing randomly, then trains it

### Stage 3: End-to-End Flow Reprojection

**Objective**: Integrate into full AnyCam pipeline and train with self-supervised flow reprojection loss.

**Training Data Flow**:
```
Frame Pairs (extract_all_pairs=True, unlimited):
    Video → All consecutive pairs (0-1, 1-2, 2-3, ...)
    ↓
DataLoader (batch_size=B, num_frames=2):
    - images: [B, 2, 3, H, W] (frame pairs)
    ↓
AnyCamWrapperWithDA3Calibration.forward():
    ↓
STEP 1: Extract Visual Tokens
    DINOv2-small (standalone) → [B, 2, 384]
    ↓
STEP 2: Run AnyCalib
    AnyCalib (per frame) → [B, 2, 4]
    ↓
STEP 3: Standalone DA3CalibrationHead
    Input: visual_tokens [B, 2, 384], anycalib_predictions [B, 2, 4]
    Output: camera_params [B, 1, 4]
    Extract: focal_length = camera_params[:, 0, 0]  # [B] - fx
    ↓
STEP 4: Depth Prediction (frozen)
    Depth Predictor → depths [B, 2, 1, H, W]
    ↓
STEP 5: Flow and Occlusion (frozen)
    Image Processor → flow_occs [B, 2, 3, H, W] (flow + occlusion)
    ↓
STEP 6: Pose Prediction (frozen)
    Pose Predictor → poses [B, 2, 4, 4], uncert [B, 2, ?, H, W]
    Override: pose_result["focal_length"] = focal_length (from DA3)
    ↓
STEP 7: Flow Reprojection Loss
    Induce flow from poses + depths + focal_length
    Compare with observed flow
    Loss: Uncertainty-aware flow reprojection loss
```

**Key Features**:
- Standalone DA3CalibrationHead with own DINOv2-small (vis_dim=384)
- Matches Stage 2 dimensions for proper checkpoint loading
- Only DA3CalibrationHead trained (all other components frozen)
- Self-supervised: No GT calibration or poses needed
- Per-epoch pose benchmarking (optional, requires GT poses)

**Hyperparameters**:
- Learning rate: `1e-5` (very low for fine-tuning)
- Batch size: `2-4` (memory intensive)
- Epochs: `50-100`
- Optimizer: Adam

**Memory Management**:
- Periodic GPU cache clearing (`torch.cuda.empty_cache()`)
- Mixed precision training (autocast with fp16)
- Designed for 24GB VRAM

**Per-Epoch Pose Benchmarking** (if GT available):
- Evaluates 20 test samples per epoch (Stage 3)
- Evaluates 100 fixed samples per epoch (Stage 3.1, no cycling)
- Compares DA3 Stage 3 vs AnyCam baseline (32 candidates)
- Logs rotation and translation errors
- Generates `pose_benchmark_curve.png` and `pose_benchmark_log.txt`

### Stage 3.1: Multi-Frame Variants with Optional Alternating Training

**Objective**: Extend Stage 3 with multi-frame sequences (max_ahead=3 or 4) and optional alternating training strategy.

**Four Training Variants**:
1. **max_ahead=3, no alternating**: 4-frame sequences, standard training
2. **max_ahead=4, no alternating**: 5-frame sequences, standard training
3. **max_ahead=3, alternating**: 4-frame sequences, alternating training strategy
4. **max_ahead=4, alternating**: 5-frame sequences, alternating training strategy

**Training Data Flow (Stage 3.1, max_ahead=3)**:
```
DataLoader (batch_size=B, num_frames=4):
    - images: [B, 4, 3, H, W] (4-frame sequences)
    ↓
AnyCamWrapperWithDA3Calibration.forward():
    ↓
STEP 1: Extract Visual Tokens
    DINOv2-small (standalone) → [B, 4, 384]
    ↓
STEP 2: Run AnyCalib
    AnyCalib (per frame) → [B, 4, 4]
    ↓
STEP 3: Standalone DA3CalibrationHead
    Input: visual_tokens [B, 4, 384], anycalib_predictions [B, 4, 4]
    Output: camera_params [B, 1, 4]  # Single focal length for entire sequence
    Extract: focal_length = camera_params[:, 0, 0]  # [B] - fx
    ↓
STEP 4-7: Same as Stage 3 (depth, flow, pose, loss)
```

**Key Differences from Stage 3**:
- **Multi-frame Input**: Uses `max_ahead+1` frames per sequence (4 for max_ahead=3, 5 for max_ahead=4)
- **Dataset**: Uses `ObjectronVideoDatasetMultiFrame` for multi-frame sequences
- **Fixed Benchmark**: Uses same 100 samples every epoch (no cycling) for consistent evaluation
- **Alternating Training** (optional): 
  - Even epochs: Train calibration head, freeze pose head
  - Odd epochs: Train pose head, freeze calibration head
  - Optimizer recreated each epoch with appropriate parameters

**Hyperparameters** (same as Stage 3):
- Learning rate: `1e-5`
- Batch size: `2-4`
- Epochs: `50-100`
- Optimizer: Adam (recreated each epoch for alternating training)

## Models and Components Used

### Visual Token Extraction

**All Stages Use**: HuggingFace DINOv2-small (`facebook/dinov2-small`)
- **Dimension**: `vis_dim = 384` (CLS token)
- **Input**: RGB images `[B, N, 3, H, W]` normalized with ImageNet stats
- **Preprocessing**: Resize to 224×224 (HuggingFace requirement)
- **Output**: CLS token `[B, N, 384]`
- **Why HuggingFace**: Native PyTorch attention (SDPA), compatible with RTX 5090 (compute capability 12.0)
- **Alternative Rejected**: Facebook's torch.hub DINOv2 uses xFormers (incompatible with RTX 5090)

### AnyCalib Integration

**All Stages Use**: AnyCalib pretrained model
- **Input**: RGB images `[B, N, 3, H, W]` in range [0, 1]
- **Output**: Camera intrinsics `[B, N, 4]` (fx, fy, cx, cy) in pixels
- **Mode**: Single frame inference (runs on each frame independently)
- **Wrapper**: `AnyCaLibBatchInference` for batch processing

### AnyCam Pipeline (Stage 3 Only)

**Components** (all frozen during Stage 3 training):
1. **Depth Predictor**: UniDepth v2 → depths `[B, N, 1, H, W]`
2. **Image Processor**: Flow + occlusion estimation → `[B, N, 3, H, W]` (flow + occlusion)
3. **Pose Predictor**: AnyCam pose head → poses `[B, N, 4, 4]`, uncertainties

**Integration Point**:
- DA3CalibrationHead predicts focal length `fx`
- Overrides `pose_result["focal_length"]` before flow reprojection
- Original 32-candidate system bypassed

## Loss Functions

### Stage 1 & 2
```python
loss = F.mse_loss(
    predicted_mean_calibration,  # [B, 1, 4]
    gt_mean_calibration          # [B, 1, 4]
)
```

### Stage 3
```python
from anycam.loss import make_loss

# Flow reprojection loss (self-supervised)
criterion = make_loss(loss_config)  # From training_config.yaml
loss_dict = criterion(output_data)
loss = loss_dict.get('loss', loss_dict.get('total_loss', sum(loss_dict.values())))
```

**Loss Components** (from config):
- Flow loss: L1 distance between induced and observed flow
- Uncertainty weighting: `flow_error * √2 / (uncertainty + ε) + log(uncertainty + ε)`
- Distance loss: Disabled (`lambda_dist: 0`)

## Training Configuration

### Dataset: Objectron (Validation Experiment)

**Important Note**: This is a **small-scale validation experiment**. All three stages are trained exclusively on Objectron to verify the architecture and training methodology. Dataset-specific overfitting is expected and acceptable at this stage.

**Dataset Details**:
- **Videos**: `/data/thesis/Objectron/videos` (70 train, 15 val, 15 test)
- **GT Calibration**: `/data/thesis/Objectron/processed_gt` (for Stage 1 & 2)
- **GT Poses**: Same directory (for pose benchmarking)
- **Frame Loading**:
  - Stage 1 & 2: ALL frames from each video (no limit)
  - Stage 3: All consecutive frame pairs (0-1, 1-2, 2-3, ...)
  - Stage 3.1: Multi-frame sequences (4 frames for max_ahead=3, 5 frames for max_ahead=4)

**Split File**: `experiments/objectron_split.json`
- Train: 70 videos
- Validation: 15 videos
- Test: 15 videos

### Variable-Length Sequence Handling

**Problem**: Different videos have different numbers of frames (e.g., 100-500 frames).

**Solution**: Custom collate function with padding:
```python
def collate_variable_length(batch):
    max_N = max(item['visual_tokens'].shape[0] for item in batch)
    # Pad all sequences to max_N
    # Create attention_mask: 1.0 for valid, 0.0 for padding
    return {
        'visual_tokens': [B, max_N, 384],  # Padded
        'anycalib_predictions': [B, max_N, 4],  # Padded
        'attention_mask': [B, max_N],  # Mask
        'gt_mean_calibration': [B, 1, 4],
    }
```

**Sequence Aggregation**: Uses masked mean pooling to ignore padded positions.

## Key Implementation Details

### GPU Compatibility Fix

**Problem**: Facebook's DINOv2 uses xFormers (compute capability ≤ 9.0), incompatible with RTX 5090 (capability 12.0).

**Solution**: Switched to HuggingFace DINOv2-small:
- Uses native PyTorch attention (SDPA)
- Compatible with all GPUs
- Same architecture, different implementation

### Stage 3 Architecture Fix

**Problem**: Stage 2 uses DINOv2-small (vis_dim=384), but Stage 3 originally used AnyCam backbone (vis_dim=128) → dimension mismatch.

**Solution**: Stage 3 uses **standalone** DA3CalibrationHead with its own DINOv2-small:
- Matches Stage 2 dimensions (vis_dim=384)
- Checkpoint loading works correctly
- Consistent visual features across stages

### Memory Management

**Stage 2 Preprocessing**:
- Chunked processing (16 frames at a time)
- Move results to CPU immediately
- Clear GPU cache after each chunk

**Stage 3 Training**:
- Periodic cache clearing (`torch.cuda.empty_cache()` every 50 batches)
- Mixed precision (autocast with fp16)
- Batch size limited to 2-4

## Training Commands

### Stage 1
```bash
python experiments/train_calibration_head_da3_stage1.py \
    --objectron_videos /data/thesis/Objectron/videos \
    --objectron_gt /data/thesis/Objectron/processed_gt \
    --num_epochs 50 \
    --batch_size 8 \
    --learning_rate 1e-4 \
    --save_dir experiments/da3_integration/stage1_training
```

### Stage 2
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

### Stage 3
```bash
python experiments/train_calibration_head_da3_stage3.py \
    --objectron_videos /data/thesis/Objectron/videos \
    --objectron_gt /data/thesis/Objectron/processed_gt \
    --stage2_checkpoint experiments/da3_integration/stage2_training/checkpoints/final_model.pt \
    --num_epochs 50 \
    --batch_size 4 \
    --learning_rate 1e-5 \
    --benchmark_samples 20 \
    --save_dir experiments/da3_integration/stage3_training_pose
```

**Note**: `--objectron_gt` is optional for Stage 3 (only needed for pose benchmarking). Training is self-supervised.

## Output Files

### Training Outputs (All Stages)
- `training_log.txt`: Detailed epoch-by-epoch logs
- `loss_history.json`: Training and validation loss history
- `loss_curve.png`: Plot with train/val curves
- `training_summary.txt`: Summary statistics
- `checkpoints/`: Model checkpoints (every 10 epochs + final)

### Stage 3 Additional Outputs
- `pose_benchmark_curve.png`: Pose accuracy over epochs (if benchmarking enabled)
- `pose_benchmark_log.txt`: Pose benchmark results
- `pose_benchmark_history.json`: Detailed benchmark data

### Evaluation Outputs (Stage 1 & 2)
- `calibration_accuracy.json`: Training set accuracy metrics
- `val_calibration_accuracy.json`: Validation set accuracy metrics

## Results Organization

```
experiments/da3_integration/
├── stage1_training/          # Stage 1 results
├── stage2_training/          # Stage 2 results
├── stage3_training/          # Stage 3 results (without pose benchmarking)
├── stage3_training_pose/     # Stage 3 results (with pose benchmarking)
├── benchmark_results/         # All benchmark results
│   ├── stage1_calibration/
│   ├── stage2_calibration/
│   ├── stage3_calibration/
│   └── stages_comparison/    # Inter-stage comparison (scientifically valid)
└── IMPLEMENTATION_SUMMARY.md # Detailed implementation notes
```

## Evaluation Considerations

### Validation Experiment Status

**Current Setup**: All stages trained exclusively on Objectron (small-scale validation).

**Valid Comparisons**:
- ✅ **Inter-stage comparison** (Stage 1 vs 2 vs 3): Scientifically valid (same dataset, same protocol)
- ✅ **Training progression**: Verify each stage improves upon previous

**Comparison Reliability**:
- ✅ **DA3 Stage 3 vs AnyCam baseline**: **Somewhat reliable** because DINOv2 from AnyCam encoder is frozen. The visual transformer part that learns dataset features is fixed, and only the calibration head was trained. This makes the comparison more fair as both methods use the same frozen visual features, isolating the comparison to the calibration approach.
- ⚠️ **DA3 vs AnyCalib**: Should be interpreted with caution due to different training datasets (overfitting to Objectron), but the frozen visual backbone provides some reliability.
- ⚠️ **DA3 Stage 3 vs AnyCalib-AnyCam hybrid**: Different training datasets, but frozen visual backbone provides some reliability.

**Future Work**: Retrain on large-scale, diverse datasets (hundreds of thousands of sequences) for full generalization. The frozen visual backbone in the current implementation provides a fairer comparison framework than if the entire model was retrained.

### Benchmarking

**Calibration Accuracy** (Objectron only - has GT calibration):
- Quick benchmarks for individual stages
- Inter-stage comparison (primary validation benchmark)

**Pose Estimation** (Objectron or LightSpeed - both have GT poses):
- Stage 3 vs baseline comparison (for future use after large-scale training)
- Per-epoch benchmarking during Stage 3 training (optional)

## Key Takeaways

1. **Staged Training**: Three stages allow gradual complexity introduction
2. **Consistent Visual Features**: DINOv2-small (vis_dim=384) used across all stages
3. **Self-Supervised Stage 3**: No GT calibration needed, uses flow reprojection loss
4. **GPU Compatibility**: HuggingFace DINOv2 uses native attention (no xFormers)
5. **Memory Efficient**: Designed for 24GB VRAM with chunked processing and cache clearing
6. **Validation Phase**: Current results demonstrate architecture validity, not production performance

## References

- **Implementation Details**: `IMPLEMENTATION_SUMMARY.md`
- **Benchmark Usage**: `BENCHMARK_USAGE.md`
- **Quick Start**: `README.md`
- **Architecture Plan**: `DEPTH_ANYTHING_3_INTEGRATION_PLAN.md`

# notes: 

- next training: fix benchmark dataset, dnot cycle it. add look ahead 3 and 4. see results.
- next training 2: same thing as above, then, adopt a train in alternating cycle: one epoch fix calibration, train anycam (pose head), then next epoch fix anycam and train calibration. 