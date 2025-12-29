# Depth Anything 3 Multi-Frame Camera Conditioning Integration Plan

**Author:** Kalman Mahlich  
**Supervisor:** Daniil Sinitsyn  
**Institution:** Technical University of Munich (TUM)  
**Date:** December 2025  
**Status:** Planning Phase

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Background and Context](#background-and-context)
3. [Current Architecture Analysis](#current-architecture-analysis)
4. [Depth Anything 3 Camera Conditioning](#depth-anything-3-camera-conditioning)
5. [Proposed Architecture](#proposed-architecture)
6. [Implementation Plan](#implementation-plan)
7. [Training Strategy](#training-strategy)
8. [Integration Points](#integration-points)
9. [Evaluation Metrics](#evaluation-metrics)
10. [Timeline and Milestones](#timeline-and-milestones)

---

## Executive Summary

This document outlines a comprehensive plan to integrate **Depth Anything 3 (DA3)** camera conditioning architecture into the AnyCam pipeline, specifically replacing AnyCam's 32-candidate focal length prediction system with a learned multi-frame calibration head inspired by DA3's camera conditioning mechanism. The integration leverages AnyCalib's single-view predictions as initialization and extends them to multi-frame sequences using transformer-based attention mechanisms.

**Key Architectural Principle: Reverse Conditioning**
Unlike DA3 which conditions depth prediction on camera parameters, we condition **camera prediction on visual features**. This means:
- Camera tokens (from AnyCalib initialization) are **updated** with visual information
- Visual features refine and improve the camera calibration prediction
- The updated camera tokens flow through aggregation and decoding to final parameters

**Key Objectives:**
- Replace expensive 32-candidate focal length system with direct learned predictions
- Implement multi-frame camera conditioning using DA3-inspired architecture (reverse direction)
- Train calibration head to leverage visual features for improved calibration
- Integrate seamlessly into existing AnyCam training pipeline

**Expected Benefits:**
- Reduced computational cost (no 32-candidate evaluation during training)
- Improved calibration accuracy through multi-frame information aggregation
- Better generalization through learned visual-conditioned calibration
- Per-frame calibration capability (vs. sequence-level in current system)

---

## Background and Context

### AnyCam Architecture Overview

**AnyCam** (CVPR 2025) is a self-supervised learning framework for recovering camera poses and intrinsics from casual videos without ground truth. The architecture consists of:

1. **Depth Predictor** (frozen): Pretrained UniDepth model for depth estimation
2. **Flow Estimator**: UniMatch for optical flow computation between frame pairs
3. **Pose Predictor**: Transformer-based model predicting:
   - Camera poses (rotation + translation) per frame pair
   - Focal length via **32-candidate system** (computationally expensive)
   - Uncertainty maps for pose predictions

**Current Focal Length Prediction System:**

```
Input Images (N frames)
    ↓
Backbone (DINOv2/CroCo) → Visual Features [N, D]
    ↓
Sequence Token Attention → Sequence Token [1, D_seq]
    ↓
sequence_info_head (MLP) → focal_enc [1, 32] (logits)
    ↓
Softmax → focal_length_probs [1, 32]
    ↓
32 Candidates [0.1, 0.2, ..., 4.0] → Weighted Average
    ↓
Single Focal Length (sequence-level)
```

**Problem:** During training, all 32 candidates are evaluated via flow reprojection loss, selecting the best candidate. This is computationally expensive and may not capture true focal length accurately.

**Architecture Location:**
- Main model: `anycam/models/anycam.py` (lines 379-392)
- Focal encoding: `anycam/models/anycam.py` (lines 444-488)
- Training evaluation: `anycam/trainer.py` (line 405)

### AnyCalib Overview

**AnyCalib** is a model-agnostic single-view camera calibration method that predicts camera intrinsics from a single image by analyzing visual cues. Key characteristics:

- **Input**: Single RGB image
- **Output**: Camera intrinsics (fx, fy, cx, cy) for pinhole camera model
- **Architecture**: 
  - DINOv2 backbone for feature extraction
  - DPT decoder for dense prediction
  - Ray-based representation (3D rays from camera center)
  - On-manifold optimization for calibration parameters

**AnyCalib Pipeline:**

```
Single Image [H, W, 3]
    ↓
DINOv2 Backbone → Visual Features [H*W, D]
    ↓
DPT Decoder → Ray Predictions [H*W, 3]
    ↓
On-Manifold Optimization → Intrinsics (fx, fy, cx, cy)
```

**Current Integration:** In Experiment 1 and 2, AnyCalib predictions are injected directly:
- Run AnyCalib on first frame (or all frames)
- Extract focal length: `fx = intrinsics[0, 0]`
- Use directly in projection matrix computation
- **Limitation**: No learning, no multi-frame aggregation, no visual conditioning

**Architecture Location:**
- Model: `anycalib/anycalib/model/anycalib_pretrained.py`
- Integration: `experiments/train_pose_head_anycalib.py` (lines 506-588)

### Depth Anything 3 Camera Conditioning

**Depth Anything 3 (DA3)** introduces camera conditioning techniques for depth estimation that condition the model on known camera parameters. The architecture uses:

1. **Camera Encoder**: MLP mapping camera parameters (fx, fy, cx, cy) to tokens
2. **Visual-Camera Mixing**: Attention layers mixing visual tokens with camera tokens
3. **Learnable Camera Token**: Sequence-level camera token for aggregation
4. **Camera Decoder**: MLP decoding tokens back to camera parameters

**DA3 Camera Conditioning Architecture:**

```
Camera Parameters (fx, fy, cx, cy) [4]
    ↓
Camera Encoder (MLP) → Camera Tokens [N_cam, D_cam]
    ↓
Visual Features [N_vis, D_vis] ──┐
    ↓                            │
Visual-Camera Attention ──────────┘
    ↓
Mixed Tokens [N_vis, D_mixed]
    ↓
Learnable Camera Token [1, D_cam] ──┐
    ↓                                │
Camera Token Attention ─────────────┘
    ↓
Aggregated Camera Token [1, D_cam]
    ↓
Camera Decoder (MLP) → Camera Parameters (fx, fy, cx, cy) [4]
```

**Key Innovation:** DA3 conditions depth prediction on camera parameters, allowing the model to adapt its predictions based on known calibration. We reverse this: predict camera parameters conditioned on visual features.

**Relevant Components:**
- Camera encoder: Maps intrinsics to embedding space
- Visual-camera attention: Mixes visual and camera information
- Camera decoder: Maps embeddings back to intrinsics
- Learnable aggregation: Sequence-level camera token

---

## Current Architecture Analysis

### AnyCam Forward Pass (Focal Length Prediction)

**Current Flow:**

```
1. Image Input: [B, N, 3, H, W]
   ↓
2. Backbone Feature Extraction:
   - DINOv2/CroCo processes each frame
   - Output: pose_tokens [B, N, D_pose]
   ↓
3. Inter-Frame Attention:
   - Self-attention across frames: pose_token [B, N, D_pose]
   - Sequence token attention: seq_token [B, 1, D_seq]
   ↓
4. Sequence Info Head (MLP):
   - Input: seq_token [B, 1, D_seq]
   - Output: seq_enc [B, 1, D_focal + D_scale]
   - Extract: focal_enc [B, 1, 32]
   ↓
5. Focal Length Decoding:
   - Softmax: focal_probs [B, 1, 32]
   - Candidates: [0.1, 0.2, ..., 4.0] (32 values)
   - Weighted sum: focal_length [B] (single value per batch)
   ↓
6. Projection Matrix:
   - make_proj_from_focal_length(focal_length, w/h)
   - Output: K [B, 3, 3]
```

**Code Locations:**
- Feature extraction: `anycam/models/anycam.py` lines 327-361
- Sequence info head: `anycam/models/anycam.py` line 379
- Focal decoding: `anycam/models/anycam.py` lines 444-488
- Projection: `anycam/trainer.py` line 379

### Current Integration (Experiment 1 & 2)

**AnyCalib Injection:**

```
1. Load frames: [B, N, 3, H, W]
   ↓
2. Run AnyCalib (per frame or first frame):
   - AnyCalib.infer(frame_i) → intrinsics_i [fx, fy, cx, cy]
   - Extract: focal_px = intrinsics_i[0, 0]
   ↓
3. Convert to normalized focal length:
   - focal_norm = focal_px / (max(H, W) / 2)
   ↓
4. Bypass sequence_info_head:
   - Skip focal_enc computation
   - Use AnyCalib focal directly
   ↓
5. Create dummy candidates:
   - focal_candidates = focal_norm.unsqueeze(1) [B, 1]
   - focal_probs = ones [B, 1]
   ↓
6. Continue with projection matrix computation
```

**Limitations:**
- No learning: AnyCalib predictions are fixed
- No multi-frame aggregation: Uses single frame or simple average
- No visual conditioning: Doesn't leverage visual features
- No refinement: Cannot improve during training

---

## Depth Anything 3 Camera Conditioning

### Architecture Components

#### 1. Camera Encoder (`cam_enc.py`)

**Purpose:** Map camera parameters to embedding space for attention mechanisms.

**Note:** Supervisor suggested using DA3's actual `cam_enc.py` and `cam_dec.py` modules if available. If not available, implement similar architecture.

**Architecture (DA3-Inspired):**

```
Input: Camera Parameters [B, N, 4] where 4 = [fx, fy, cx, cy]
    ↓
Normalization Layer:
    - fx, fy: Normalize by image size → [0, 1] range
    - cx, cy: Normalize by image size → [0, 1] range
    ↓
MLP Encoder:
    Linear(4 → D_hidden)
    ReLU()
    Linear(D_hidden → D_cam)
    ↓
Output: Camera Tokens [B, N, D_cam]
```

**Design Rationale:**
- Normalization ensures consistent scale across different image resolutions
- MLP learns meaningful representations of camera parameters
- D_cam typically smaller than visual token dimension (256 vs 768) to reduce parameters
- **If DA3's encoder available**: Use directly or copy exact implementation

#### 2. Visual-Camera Mixing (Reverse Conditioning)

**Purpose:** Update camera tokens with visual information to create visual-conditioned camera predictions. This is the **reverse** of DA3: instead of conditioning depth on camera parameters, we condition camera prediction on visual features.

**Architecture (Cross-Attention Approach - Recommended):**

```
Visual Tokens [B, N, D_vis]  Camera Tokens [B, N, D_cam]
    │                                │
    └────────────────────────────────┘
                    │
    Cross-Attention Layer:
        Query: Camera Tokens [B, N, D_cam]  ← Camera tokens query visual info
        Key: Visual Tokens [B, N, D_vis]    ← Visual tokens provide keys
        Value: Visual Tokens [B, N, D_vis]  ← Visual tokens provide values
        ↓
    Visual Information [B, N, D_vis]
        ↓
    Project to Camera Dimension: Linear(D_vis → D_cam)
        ↓
    Visual-Conditioned Update [B, N, D_cam]
        ↓
    Residual Connection: camera_tokens + update
        ↓
    Updated Camera Tokens [B, N, D_cam]  ← These flow to aggregation
```

**Alternative: Concatenation + MLP Approach:**

```
Visual Tokens [B, N, D_vis]  Camera Tokens [B, N, D_cam]
    │                                │
    └────────────────────────────────┘
                    │
            Concatenate
                    │
    Concatenated [B, N, D_vis + D_cam]
                    │
    MLP: Linear(D_vis + D_cam → D_cam)
        ↓
    Visual-Conditioned Update [B, N, D_cam]
        ↓
    Residual Connection: camera_tokens + update
        ↓
    Updated Camera Tokens [B, N, D_cam]
```

**Design Rationale:**
- **Reverse conditioning**: Camera tokens are updated with visual information (not visual updated with camera)
- **Residual connection**: Preserves AnyCalib initialization while allowing visual refinement
- **Cross-attention preferred**: Allows selective information extraction from visual features
- **Per-frame updates**: Each frame's camera token is conditioned on its corresponding visual features

**Key Difference from DA3:**
- DA3: `depth = f(visual_features, camera_params)` - conditions depth on camera
- Our approach: `camera_params = f(visual_features, anycalib_init)` - conditions camera on visual

#### 3. Sequence-Level Aggregation

**Purpose:** Aggregate per-frame visual-conditioned camera tokens into a sequence-level representation.

**Architecture (Learnable Token with Attention - Recommended):**

```
Learnable Parameter: camera_token [1, D_cam]
    ↓
Expand to Batch: [B, 1, D_cam]
    ↓
Updated Camera Tokens [B, N, D_cam] (from visual-conditioning)
    ↓
Cross-Attention:
    Query: Learnable Token [B, 1, D_cam]
    Key: Updated Camera Tokens [B, N, D_cam]
    Value: Updated Camera Tokens [B, N, D_cam]
    ↓
Aggregated Camera Token [B, 1, D_cam]
```

**Alternative: Simple Average Pooling (Supervisor's Suggestion):**

```
Updated Camera Tokens [B, N, D_cam]
    ↓
Mean Pooling: mean(dim=1, keepdim=True)
    ↓
Aggregated Camera Token [B, 1, D_cam]
```

**Design Rationale:**
- **Learnable token approach**: Allows adaptive weighting of frames (better for sequences with varying quality)
- **Simple averaging**: Matches supervisor's suggestion, simpler, faster, good baseline
- **Recommendation**: Start with simple averaging for Stage 1, optionally upgrade to learnable token for Stage 2/3
- Sequence-level representation suitable for sequence-level calibration

#### 4. Camera Decoder (`cam_dec.py`)

**Purpose:** Map aggregated camera token back to camera parameters.

**Note:** Supervisor suggested using DA3's actual `cam_dec.py` module if available.

**Architecture (DA3-Inspired):**

```
Aggregated Camera Token [B, 1, D_cam]
    ↓
MLP Decoder:
    Linear(D_cam → D_hidden)
    ReLU()
    Linear(D_hidden → 4)
    ↓
Output: Camera Parameters [B, 1, 4] where 4 = [fx, fy, cx, cy]
    ↓
Denormalization:
    - fx, fy: Denormalize by image size
    - cx, cy: Denormalize by image size
    ↓
Final Camera Parameters [B, 1, 4]
```

**Design Rationale:**
- Symmetric to encoder (encode-decode structure)
- MLP learns inverse mapping from embedding to parameters
- Denormalization ensures correct scale for projection matrices
- **If DA3's decoder available**: Use directly or copy exact implementation

### Complete DA3-Inspired Calibration Head

**Full Architecture Diagram:**

```
┌─────────────────────────────────────────────────────────────────┐
│                    INPUT LAYER                                  │
├─────────────────────────────────────────────────────────────────┤
│  Visual Tokens [B, N, N_vis, D_vis]                            │
│  AnyCalib Initial Predictions [B, N, 4] (fx, fy, cx, cy)      │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│              CAMERA ENCODER (Stage 1)                           │
├─────────────────────────────────────────────────────────────────┤
│  AnyCalib Predictions [B, N, 4]                                 │
│      ↓                                                           │
│  Normalize (fx, fy, cx, cy) → [0, 1]                           │
│      ↓                                                           │
│  MLP: Linear(4 → 128) → ReLU → Linear(128 → D_cam)            │
│      ↓                                                           │
│  Camera Tokens [B, N, D_cam]                                   │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│         VISUAL-CAMERA MIXING (Stage 2) - REVERSE CONDITIONING  │
├─────────────────────────────────────────────────────────────────┤
│  Visual Tokens [B, N, D_vis]                                    │
│  Camera Tokens [B, N, D_cam] (from encoder)                     │
│      ↓                                                           │
│  Cross-Attention:                                                │
│      Query: Camera Tokens [B, N, D_cam]  ← Camera queries visual│
│      Key: Visual Tokens [B, N, D_vis]    ← Visual provides info │
│      Value: Visual Tokens [B, N, D_vis]                         │
│      ↓                                                           │
│  Visual Information [B, N, D_vis]                               │
│      ↓                                                           │
│  Project: Linear(D_vis → D_cam)                                │
│      ↓                                                           │
│  Visual Update [B, N, D_cam]                                    │
│      ↓                                                           │
│  Residual: camera_tokens + visual_update                        │
│      ↓                                                           │
│  Updated Camera Tokens [B, N, D_cam]  ← Visual-conditioned!   │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│         SEQUENCE-LEVEL AGGREGATION (Stage 3)                   │
├─────────────────────────────────────────────────────────────────┤
│  Updated Camera Tokens [B, N, D_cam]  ← Uses visual-conditioned│
│  Learnable Camera Token [1, D_cam]                              │
│      ↓                                                           │
│  Expand Learnable Token: [B, 1, D_cam]                         │
│      ↓                                                           │
│  Cross-Attention:                                              │
│      Query: Learnable Token [B, 1, D_cam]                      │
│      Key/Value: Updated Camera Tokens [B, N, D_cam]           │
│      ↓                                                           │
│  Aggregated Camera Token [B, 1, D_cam]  ← Contains visual info │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│              CAMERA DECODER (Stage 4)                           │
├─────────────────────────────────────────────────────────────────┤
│  Aggregated Camera Token [B, 1, D_cam]                         │
│      ↓                                                           │
│  MLP: Linear(D_cam → 128) → ReLU → Linear(128 → 4)            │
│      ↓                                                           │
│  Normalized Parameters [B, 1, 4]                                │
│      ↓                                                           │
│  Denormalize: [0, 1] → pixel coordinates                        │
│      ↓                                                           │
│  Final Camera Parameters [B, 1, 4] (fx, fy, cx, cy)            │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│                    OUTPUT                                       │
├─────────────────────────────────────────────────────────────────┤
│  Camera Intrinsics: [B, 1, 4]                                   │
│  - fx: Focal length (x-axis)                                    │
│  - fy: Focal length (y-axis)                                    │
│  - cx: Principal point (x-axis)                                 │
│  - cy: Principal point (y-axis)                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Proposed Architecture

### Integration into AnyCam Pipeline

**Modified AnyCam Forward Pass:**

```
┌─────────────────────────────────────────────────────────────────┐
│                    EXISTING ANYCAM PIPELINE                    │
├─────────────────────────────────────────────────────────────────┤
│  1. Image Input [B, N, 3, H, W]                                │
│  2. Backbone Feature Extraction → Visual Tokens [B, N, D_vis]  │
│  3. Inter-Frame Attention → pose_tokens [B, N, D_pose]        │
│  4. Sequence Token Attention → seq_token [B, 1, D_seq]          │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│              NEW: DA3 CALIBRATION HEAD                          │
├─────────────────────────────────────────────────────────────────┤
│  Inputs:                                                        │
│    - Visual Tokens [B, N, D_vis] (from backbone)              │
│    - AnyCalib Predictions [B, N, 4] (per-frame initialization) │
│      ↓                                                           │
│  DA3 Calibration Head:                                         │
│    Stage 1: Camera Encoder                                     │
│    Stage 2: Visual-Camera Mixing                                │
│    Stage 3: Sequence Aggregation                                │
│    Stage 4: Camera Decoder                                      │
│      ↓                                                           │
│  Output: Camera Parameters [B, 1, 4] (fx, fy, cx, cy)         │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│              EXISTING ANYCAM PIPELINE (CONTINUED)               │
├─────────────────────────────────────────────────────────────────┤
│  5. Pose Head: pose_tokens → poses [B, N, 4, 4]                │
│  6. Projection Matrix: K = make_proj_from_focal_length(...)    │
│  7. Flow Reprojection Loss: Compare predicted vs. observed flow │
└─────────────────────────────────────────────────────────────────┘
```

### Key Modifications

**1. Replace `sequence_info_head` with DA3 Calibration Head**

**Current:**
```python
# anycam/models/anycam.py line 379
seq_enc = self.sequence_info_head(seq_token.to(torch.float32))
focal_enc = seq_enc[..., :self.focal_enc_dim]
focal_length, focal_length_probs, focal_candidates = self.enc_embed_to_focal(focal_enc)
```

**New:**
```python
# New calibration head
camera_params = self.da3_calibration_head(
    visual_tokens=pose_tokens,  # [B, N, D_vis]
    anycalib_init=anycalib_predictions  # [B, N, 4]
)
# camera_params: [B, 1, 4] -> (fx, fy, cx, cy)
focal_length = camera_params[:, 0, 0]  # Extract fx
```

**2. Add AnyCalib Initialization**

**Before forward pass:**
```python
# Run AnyCalib on all frames
anycalib_predictions = []
for i in range(n_frames):
    frame = images[:, i]  # [B, 3, H, W]
    intrinsics = self.anycalib_model.infer(frame)  # [B, 4]
    anycalib_predictions.append(intrinsics)
anycalib_predictions = torch.stack(anycalib_predictions, dim=1)  # [B, N, 4]
```

**3. Remove Candidate System**

**Remove:**
- `focal_length_probs` computation
- `focal_candidates` generation
- Candidate evaluation loop in training

**Keep:**
- Direct focal length usage in projection matrix
- Flow reprojection loss (unchanged)

### Architecture Details

#### Visual Token Extraction

**Source:** Use pose tokens from AnyCam backbone

**Current Flow:**
```
pose_tokens = self.pose_reassemble_stage(pose_tokens)  # [B, N, D_pose]
pose_tokens = self.pose_feature_fusion_stage(pose_tokens)
pose_token = pose_tokens[-1]  # [B, N, D_pose]
```

**Modification:** Extract visual features before pose-specific processing

**Option A: Use pose tokens directly**
- Pros: Simple, no architecture changes
- Cons: Pose tokens may be optimized for pose, not calibration

**Option B: Extract from backbone before pose processing**
- Pros: Cleaner separation, visual features not pose-biased
- Cons: Requires architecture modification

**Recommendation:** Start with Option A (use pose tokens), evaluate performance, then consider Option B if needed.

#### Dimensionality Matching

**Current Dimensions:**
- `pose_token`: [B, N, D_pose] where D_pose ≈ 768 (DINOv2) or 1024 (CroCo)
- `seq_token`: [B, 1, D_seq] where D_seq ≈ 768 or 1024

**DA3 Calibration Head Dimensions:**
- `D_cam`: Camera token dimension (recommend: 256 or 512)
- `D_mixed`: Mixed visual-camera dimension (recommend: D_vis)
- `D_hidden`: Hidden layer dimension (recommend: 128 or 256)

**Design Choice:** Use smaller dimensions for camera tokens to reduce parameters while maintaining expressiveness.

---

## Implementation Plan

### Phase 1: Core Components (Week 1-2)

#### Step 1.1: Camera Encoder Implementation

**File:** `experiments/models/camera_encoder.py`

**Implementation:**

```python
class CameraEncoder(nn.Module):
    """
    Encodes camera parameters (fx, fy, cx, cy) to embedding space.
    
    Architecture:
        Input: [B, N, 4] (fx, fy, cx, cy)
        Normalize → MLP → Output: [B, N, D_cam]
    """
    def __init__(self, cam_dim=256, hidden_dim=128):
        super().__init__()
        self.cam_dim = cam_dim
        self.hidden_dim = hidden_dim
        
        # Normalization: assumes image size provided
        self.normalize = True
        
        # MLP Encoder
        self.encoder = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, cam_dim)
        )
    
    def forward(self, camera_params, image_size):
        """
        Args:
            camera_params: [B, N, 4] (fx, fy, cx, cy) in pixels
            image_size: (H, W) tuple for normalization
        
        Returns:
            camera_tokens: [B, N, cam_dim]
        """
        B, N, _ = camera_params.shape
        H, W = image_size
        
        # Normalize camera parameters
        if self.normalize:
            fx_norm = camera_params[:, :, 0] / (max(H, W) / 2)  # Normalize fx
            fy_norm = camera_params[:, :, 1] / (max(H, W) / 2)  # Normalize fy
            cx_norm = camera_params[:, :, 2] / W  # Normalize cx
            cy_norm = camera_params[:, :, 3] / H  # Normalize cy
            
            normalized = torch.stack([fx_norm, fy_norm, cx_norm, cy_norm], dim=-1)
        else:
            normalized = camera_params
        
        # Encode
        camera_tokens = self.encoder(normalized)  # [B, N, cam_dim]
        
        return camera_tokens
```

**Testing:**
- Unit test: Verify normalization correctness
- Unit test: Verify output dimensions
- Integration test: Test with AnyCalib predictions

#### Step 1.2: Visual-Camera Mixing Implementation

**File:** `experiments/models/visual_camera_mixing.py`

**Implementation:**

```python
class VisualCameraMixing(nn.Module):
    """
    Updates camera tokens with visual information (reverse conditioning).
    
    Architecture:
        Camera Tokens [B, N, D_cam] + Visual Tokens [B, N, D_vis]
        → Cross-Attention (camera queries visual) → Project → Residual
        → Updated Camera Tokens [B, N, D_cam]
    """
    def __init__(self, vis_dim, cam_dim, num_layers=1):
        super().__init__()
        self.vis_dim = vis_dim
        self.cam_dim = cam_dim
        self.num_layers = num_layers
        
        # Cross-attention: camera tokens query visual tokens
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=cam_dim,
            kdim=vis_dim,
            vdim=vis_dim,
            num_heads=8,
            batch_first=True
        )
        
        # Project visual information to camera dimension
        self.visual_projection = nn.Linear(vis_dim, cam_dim)
        
        # Optional: Additional self-attention layers on updated camera tokens
        self.self_attention_layers = nn.ModuleList([
            nn.MultiheadAttention(cam_dim, num_heads=8, batch_first=True)
            for _ in range(num_layers - 1)
        ])
        
        # Layer norms
        self.norm1 = nn.LayerNorm(cam_dim)
        self.norm2 = nn.LayerNorm(cam_dim) if num_layers > 1 else None
    
    def forward(self, visual_tokens, camera_tokens):
        """
        Args:
            visual_tokens: [B, N, D_vis] - Visual features from backbone
            camera_tokens: [B, N, D_cam] - Camera tokens from encoder
        
        Returns:
            updated_camera_tokens: [B, N, D_cam] - Visual-conditioned camera tokens
        """
        # Cross-attention: camera tokens query visual tokens
        # Query = camera_tokens, Key/Value = visual_tokens
        # This extracts visual information relevant to each camera token
        visual_info, _ = self.cross_attention(
            query=camera_tokens,  # [B, N, D_cam] - camera queries visual
            key=visual_tokens,    # [B, N, D_vis] - visual provides keys
            value=visual_tokens   # [B, N, D_vis] - visual provides values
        )  # [B, N, D_cam] - output dimension matches query (embed_dim=cam_dim)
        
        # Project visual tokens to camera dimension for residual connection
        # This provides a direct visual signal that can be added
        visual_proj = self.visual_projection(visual_tokens)  # [B, N, D_cam]
        
        # Combine: cross-attention output (selective visual info) + direct projection
        # Residual connection preserves AnyCalib initialization
        updated_camera = camera_tokens + visual_info + visual_proj  # [B, N, D_cam]
        updated_camera = self.norm1(updated_camera)
        
        # Optional: Additional self-attention on updated camera tokens
        for self_attn in self.self_attention_layers:
            residual = updated_camera
            updated_camera, _ = self_attn(updated_camera, updated_camera, updated_camera)
            updated_camera = self.norm2(updated_camera + residual)
        
        return updated_camera  # [B, N, D_cam] - visual-conditioned camera tokens
```

**Testing:**
- Unit test: Verify cross-attention updates camera tokens (not visual tokens)
- Unit test: Verify residual connection preserves initialization
- Unit test: Verify output dimensions match input camera_tokens
- Integration test: Test that visual information flows through to final prediction

#### Step 1.3: Sequence Aggregation Implementation

**File:** `experiments/models/sequence_aggregation.py`

**Implementation:**

```python
class SequenceCameraAggregation(nn.Module):
    """
    Aggregates per-frame camera tokens into sequence-level representation.
    
    Architecture:
        Per-Frame Tokens [B, N, D_cam] + Learnable Token [1, D_cam] (optional)
        → Cross-Attention or Mean Pooling → Aggregated Token [B, 1, D_cam]
    """
    def __init__(self, cam_dim, use_learnable_token=True):
        super().__init__()
        self.cam_dim = cam_dim
        self.use_learnable_token = use_learnable_token
        
        if use_learnable_token:
            # Learnable sequence-level camera token
            self.learnable_token = nn.Parameter(torch.randn(1, 1, cam_dim))
            
            # Cross-attention for aggregation
            self.attention = nn.MultiheadAttention(
                cam_dim, num_heads=8, batch_first=True
            )
            
            # Layer norm
            self.norm = nn.LayerNorm(cam_dim)
        # If use_learnable_token=False, just use mean pooling (simpler, supervisor's suggestion)
    
    def forward(self, per_frame_tokens):
        """
        Args:
            per_frame_tokens: [B, N, D_cam] - Per-frame camera tokens (visual-conditioned)
        
        Returns:
            aggregated_token: [B, 1, D_cam] - Sequence-level camera token
        """
        if self.use_learnable_token:
            B, N, _ = per_frame_tokens.shape
            
            # Expand learnable token to batch size
            learnable = self.learnable_token.expand(B, -1, -1)  # [B, 1, D_cam]
            
            # Cross-attention: query from learnable token, key/value from per-frame tokens
            aggregated, _ = self.attention(
                query=learnable,  # [B, 1, D_cam]
                key=per_frame_tokens,  # [B, N, D_cam]
                value=per_frame_tokens  # [B, N, D_cam]
            )  # [B, 1, D_cam]
            
            # Layer norm
            aggregated = self.norm(aggregated)
            
            return aggregated
        else:
            # Simple mean pooling (supervisor's suggestion)
            return per_frame_tokens.mean(dim=1, keepdim=True)  # [B, 1, D_cam]
```

**Testing:**
- Unit test: Verify learnable token initialization
- Unit test: Verify attention output dimensions
- Integration test: Test aggregation with multiple frames

#### Step 1.4: Camera Decoder Implementation

**File:** `experiments/models/camera_decoder.py`

**Implementation:**

```python
class CameraDecoder(nn.Module):
    """
    Decodes aggregated camera token back to camera parameters.
    
    Architecture:
        Aggregated Token [B, 1, D_cam] → MLP → Parameters [B, 1, 4]
    """
    def __init__(self, cam_dim, hidden_dim=128):
        super().__init__()
        self.cam_dim = cam_dim
        self.hidden_dim = hidden_dim
        
        # MLP Decoder
        self.decoder = nn.Sequential(
            nn.Linear(cam_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4)  # Output: fx, fy, cx, cy
        )
    
    def forward(self, aggregated_token, image_size):
        """
        Args:
            aggregated_token: [B, 1, D_cam] - Sequence-level camera token
            image_size: (H, W) tuple for denormalization
        
        Returns:
            camera_params: [B, 1, 4] (fx, fy, cx, cy) in pixels
        """
        # Decode
        normalized_params = self.decoder(aggregated_token)  # [B, 1, 4]
        
        # Denormalize
        H, W = image_size
        fx = normalized_params[:, :, 0] * (max(H, W) / 2)
        fy = normalized_params[:, :, 1] * (max(H, W) / 2)
        cx = normalized_params[:, :, 2] * W
        cy = normalized_params[:, :, 3] * H
        
        camera_params = torch.stack([fx, fy, cx, cy], dim=-1)  # [B, 1, 4]
        
        return camera_params
```

**Testing:**
- Unit test: Verify decoder output dimensions
- Unit test: Verify denormalization correctness
- Integration test: Test full encode-decode cycle

#### Step 1.5: Complete DA3 Calibration Head

**File:** `experiments/models/da3_calibration_head.py`

**Implementation:**

```python
class DA3CalibrationHead(nn.Module):
    """
    Complete DA3-inspired calibration head for multi-frame camera conditioning.
    
    Architecture:
        Visual Tokens + AnyCalib Init → Camera Encoder → Visual-Camera Mixing
        → Sequence Aggregation → Camera Decoder → Camera Parameters
    """
    def __init__(
        self,
        vis_dim=768,  # Visual token dimension (from backbone)
        cam_dim=256,  # Camera token dimension
        hidden_dim=128,  # Hidden layer dimension
        num_mixing_layers=2,  # Number of attention layers in mixing
    ):
        super().__init__()
        
        self.vis_dim = vis_dim
        self.cam_dim = cam_dim
        self.hidden_dim = hidden_dim
        
        # Components
        self.camera_encoder = CameraEncoder(cam_dim, hidden_dim)
        self.visual_camera_mixing = VisualCameraMixing(
            vis_dim, cam_dim, num_layers=num_mixing_layers
        )
        self.sequence_aggregation = SequenceCameraAggregation(cam_dim)
        self.camera_decoder = CameraDecoder(cam_dim, hidden_dim)
    
    def forward(self, visual_tokens, anycalib_predictions, image_size, use_visual_conditioning=True):
        """
        Args:
            visual_tokens: [B, N, D_vis] - Visual features from backbone
            anycalib_predictions: [B, N, 4] - AnyCalib predictions (fx, fy, cx, cy)
            image_size: (H, W) tuple
            use_visual_conditioning: If False, skip visual mixing (Stage 1 training)
        
        Returns:
            camera_params: [B, 1, 4] - Predicted camera parameters (fx, fy, cx, cy)
        """
        # Stage 1: Encode camera parameters
        camera_tokens = self.camera_encoder(anycalib_predictions, image_size)  # [B, N, D_cam]
        
        # Stage 2: Update camera tokens with visual information (reverse conditioning)
        if use_visual_conditioning:
            # Visual-conditioned camera tokens
            updated_camera_tokens = self.visual_camera_mixing(visual_tokens, camera_tokens)  # [B, N, D_cam]
        else:
            # Stage 1 training: skip visual conditioning
            updated_camera_tokens = camera_tokens  # [B, N, D_cam]
        
        # Stage 3: Aggregate to sequence level
        # Aggregate the visual-conditioned camera tokens
        aggregated_camera = self.sequence_aggregation(updated_camera_tokens)  # [B, 1, D_cam]
        
        # Stage 4: Decode to camera parameters
        camera_params = self.camera_decoder(aggregated_camera, image_size)  # [B, 1, 4]
        
        return camera_params
```

**Testing:**
- Integration test: Full forward pass
- Unit test: Verify all stage outputs
- End-to-end test: Test with real AnyCam inputs

### Phase 2: Integration into AnyCam (Week 3)

#### Step 2.1: Modify AnyCam Model

**File:** `anycam/models/anycam.py`

**Modifications:**

1. **Add DA3 Calibration Head Import:**
```python
from experiments.models.da3_calibration_head import DA3CalibrationHead
```

2. **Add Calibration Head to `__init__`:**
```python
def __init__(self, ..., use_da3_calibration=False):
    # ... existing initialization ...
    
    # DA3 Calibration Head (optional)
    self.use_da3_calibration = use_da3_calibration
    if use_da3_calibration:
        vis_dim = self.config.hidden_size  # From backbone
        self.da3_calibration_head = DA3CalibrationHead(
            vis_dim=vis_dim,
            cam_dim=256,
            hidden_dim=128,
            num_mixing_layers=2
        )
    else:
        self.da3_calibration_head = None
```

3. **Modify Forward Pass:**
```python
def forward(self, images, depths=None, flow_occs=None, anycalib_predictions=None):
    # ... existing feature extraction ...
    
    # Extract visual tokens (before pose-specific processing)
    visual_tokens = pose_tokens  # [B, N, D_vis]
    
    # ... existing pose processing ...
    
    # Calibration prediction
    if self.use_da3_calibration and anycalib_predictions is not None:
        # Use DA3 calibration head
        B, N, _, H, W = images.shape
        camera_params = self.da3_calibration_head(
            visual_tokens=visual_tokens,
            anycalib_predictions=anycalib_predictions,  # [B, N, 4]
            image_size=(H, W)
        )  # [B, 1, 4]
        
        # Extract focal length
        focal_length = camera_params[:, 0, 0]  # [B] - fx
        focal_length_probs = None
        focal_candidates = None
    else:
        # Original candidate system
        seq_enc = self.sequence_info_head(seq_token.to(torch.float32))
        focal_enc = seq_enc[..., :self.focal_enc_dim]
        focal_length, focal_length_probs, focal_candidates = self.enc_embed_to_focal(focal_enc)
    
    # ... rest of forward pass ...
```

#### Step 2.2: Modify Trainer Wrapper

**File:** `anycam/trainer.py` or `experiments/train_pose_head_anycalib.py`

**Modifications:**

1. **Add AnyCalib Model Loading:**
```python
class AnyCamWrapperWithDA3Calibration(AnyCamWrapper):
    def __init__(self, ..., use_da3_calibration=False):
        super().__init__(...)
        
        self.use_da3_calibration = use_da3_calibration
        if use_da3_calibration:
            # Load AnyCalib model
            from anycalib.model.anycalib_pretrained import AnyCalib
            self.anycalib_model = AnyCalib(model_id="anycalib_pinhole").to(self.device).eval()
            
            # Freeze AnyCalib
            for param in self.anycalib_model.parameters():
                param.requires_grad = False
```

2. **Add AnyCalib Inference in Forward:**
```python
def forward(self, images, depths=None, flow_occs=None):
    B, N, C, H, W = images.shape
    
    # Run AnyCalib on all frames
    if self.use_da3_calibration:
        anycalib_predictions = []
        for i in range(N):
            frame = images[:, i]  # [B, C, H, W]
            # Convert to numpy format expected by AnyCalib
            frame_np = frame.cpu().permute(0, 2, 3, 1).numpy()  # [B, H, W, C]
            
            # Run AnyCalib (batch inference)
            batch_intrinsics = []
            for b in range(B):
                pred = self.anycalib_model.predict(frame_np[b], cam_id="pinhole")
                intrinsics = pred["intrinsics"][0]  # [4] - fx, fy, cx, cy
                batch_intrinsics.append(intrinsics)
            
            batch_intrinsics = torch.tensor(batch_intrinsics, device=self.device)  # [B, 4]
            anycalib_predictions.append(batch_intrinsics)
        
        anycalib_predictions = torch.stack(anycalib_predictions, dim=1)  # [B, N, 4]
    else:
        anycalib_predictions = None
    
    # Forward pass with AnyCalib predictions
    pose_result = self.pose_predictor(
        images, depths, flow_occs, anycalib_predictions=anycalib_predictions
    )
    
    return pose_result
```

#### Step 2.3: Update Loss Computation

**File:** `anycam/trainer.py`

**Modifications:**

The loss computation should remain unchanged since we're still outputting `focal_length` in the same format. However, we need to ensure compatibility:

```python
# Current loss computation (should work as-is)
focal_length = pose_result["focal_length"]  # [B]
proj_candidates = make_proj_from_focal_length(focal_length, w/h)  # [B, 3, 3]

# If focal_length_probs is None, we skip candidate evaluation
# This is already handled in the trainer code
```

**Verification:**
- Test that loss computation works with single focal length (no candidates)
- Verify projection matrix computation
- Test flow reprojection loss

### Phase 3: Training Scripts (Week 4)

#### Step 3.1: Stage 1 Training Script

**File:** `experiments/train_calibration_head_da3_stage1.py`

**Objective:** Train calibration head to output mean calibration (without visual tokens)

**Architecture:**

```
AnyCalib Predictions [B, N, 4]
    ↓
Camera Encoder → Camera Tokens [B, N, D_cam]
    ↓
Sequence Aggregation → Aggregated Token [B, 1, D_cam]
    ↓
Camera Decoder → Mean Calibration [B, 1, 4]
    ↓
Loss: MSE(mean_calibration, gt_mean_calibration)
```

**Implementation:**

```python
def train_stage1():
    """
    Stage 1: Train calibration head to output mean calibration.
    
    This stage trains the encoder, aggregation, and decoder to learn
    how to aggregate per-frame AnyCalib predictions into a sequence-level
    mean calibration, without using visual features.
    """
    # Load model
    model = DA3CalibrationHead(...)
    
    # Freeze visual-camera mixing (not used in stage 1)
    for param in model.visual_camera_mixing.parameters():
        param.requires_grad = False
    
    # Training loop
    for batch in dataloader:
        anycalib_preds = batch["anycalib_predictions"]  # [B, N, 4]
        gt_mean_calibration = batch["gt_mean_calibration"]  # [B, 1, 4]
        image_size = batch["image_size"]
        
        # Forward (without visual tokens - Stage 1)
        # Pass dummy visual_tokens (won't be used)
        dummy_visual = torch.zeros(B, N, model.vis_dim, device=anycalib_preds.device)
        pred_calibration = model(
            visual_tokens=dummy_visual,
            anycalib_predictions=anycalib_preds,
            image_size=image_size,
            use_visual_conditioning=False  # Skip visual mixing
        )
        
        # Loss
        loss = F.mse_loss(pred_calibration, gt_mean_calibration)
        
        # Backward
        loss.backward()
        optimizer.step()
```

**Dataset:**
- Use Objectron or Lightspeed dataset
- Extract AnyCalib predictions for all frames
- Compute GT mean calibration per sequence
- Loss: MSE between predicted and GT mean calibration

#### Step 3.2: Stage 2 Training Script

**File:** `experiments/train_calibration_head_da3_stage2.py`

**Objective:** Train calibration head with visual tokens to output mean calibration

**Architecture:**

```
Visual Tokens [B, N, D_vis] + AnyCalib Predictions [B, N, 4]
    ↓
Camera Encoder → Camera Tokens [B, N, D_cam]
    ↓
Visual-Camera Mixing → Mixed Tokens [B, N, D_mixed]
    ↓
Sequence Aggregation → Aggregated Token [B, 1, D_cam]
    ↓
Camera Decoder → Mean Calibration [B, 1, 4]
    ↓
Loss: MSE(mean_calibration, gt_mean_calibration)
```

**Implementation:**

```python
def train_stage2():
    """
    Stage 2: Train calibration head with visual tokens.
    
    This stage unfreezes visual-camera mixing and trains the full
    calibration head to leverage visual features for better calibration.
    """
    # Load Stage 1 checkpoint
    model = DA3CalibrationHead(...)
    model.load_state_dict(torch.load("stage1_checkpoint.pt"))
    
    # Unfreeze visual-camera mixing
    for param in model.visual_camera_mixing.parameters():
        param.requires_grad = True
    
    # Training loop
    for batch in dataloader:
        visual_tokens = batch["visual_tokens"]  # [B, N, D_vis]
        anycalib_preds = batch["anycalib_predictions"]  # [B, N, 4]
        gt_mean_calibration = batch["gt_mean_calibration"]  # [B, 1, 4]
        image_size = batch["image_size"]
        
        # Forward (with visual tokens - Stage 2)
        pred_calibration = model(
            visual_tokens=visual_tokens,
            anycalib_predictions=anycalib_preds,
            image_size=image_size,
            use_visual_conditioning=True  # Enable visual conditioning
        )
        
        # Loss
        loss = F.mse_loss(pred_calibration, gt_mean_calibration)
        
        # Backward
        loss.backward()
        optimizer.step()
```

**Dataset:**
- Same as Stage 1, but also extract visual tokens from AnyCam backbone
- Visual tokens: Extract from `pose_tokens` before pose-specific processing

#### Step 3.3: Stage 3 Training Script

**File:** `experiments/train_calibration_head_da3_stage3.py`

**Objective:** Integrate into full AnyCam training pipeline

**Architecture:**

```
Full AnyCam Pipeline:
    Images → Depth → Flow → Visual Tokens
    Images → AnyCalib → AnyCalib Predictions
        ↓
    DA3 Calibration Head → Camera Parameters
        ↓
    Pose Head → Poses
        ↓
    Flow Reprojection Loss
```

**Implementation:**

```python
def train_stage3():
    """
    Stage 3: End-to-end training with flow reprojection loss.
    
    This stage integrates the DA3 calibration head into the full
    AnyCam pipeline and trains using the flow reprojection loss.
    """
    # Load Stage 2 checkpoint
    model = AnyCamWrapperWithDA3Calibration(use_da3_calibration=True)
    # Load calibration head weights from Stage 2
    model.pose_predictor.da3_calibration_head.load_state_dict(
        torch.load("stage2_checkpoint.pt")
    )
    
    # Freeze everything except calibration head (or use LoRA)
    # Option A: Freeze all except calibration head
    for param in model.parameters():
        param.requires_grad = False
    for param in model.pose_predictor.da3_calibration_head.parameters():
        param.requires_grad = True
    
    # Option B: Use LoRA for efficient fine-tuning (recommended)
    # Apply LoRA to calibration head components
    
    # Training loop
    for batch in dataloader:
        images = batch["images"]  # [B, N, 3, H, W]
        flows = batch["flows"]  # [B, N-1, 2, H, W]
        
        # Forward
        pose_result = model(images)
        focal_length = pose_result["focal_length"]  # [B]
        poses = pose_result["poses"]  # [B, N, 4, 4]
        
        # Flow reprojection loss (existing AnyCam loss)
        loss = compute_flow_reprojection_loss(
            poses, focal_length, flows, depths
        )
        
        # Backward
        loss.backward()
        optimizer.step()
```

**Dataset:**
- Use full training dataset (Objectron, Lightspeed, etc.)
- Loss: Flow reprojection loss (unsupervised)
- No GT calibration needed (self-supervised)

---

## Training Strategy

### Three-Stage Training Approach

**Rationale:** Staged training allows gradual introduction of complexity, ensuring each component learns effectively before integration.

#### Stage 1: Mean Calibration Learning

**Objective:** Learn to aggregate per-frame AnyCalib predictions into sequence-level mean calibration.

**Architecture:**
- Camera Encoder: ✓ Trainable
- Visual-Camera Mixing: ✗ Frozen (not used)
- Sequence Aggregation: ✓ Trainable
- Camera Decoder: ✓ Trainable

**Loss Function:**
```
L_stage1 = MSE(predicted_mean_calibration, gt_mean_calibration)
```

**Dataset:**
- Sequences with GT calibration
- Extract AnyCalib predictions for all frames
- Compute GT mean: `mean_calibration = mean(gt_calibrations_per_frame)`

**Training Details:**
- Epochs: 20-50
- Learning Rate: 1e-4
- Optimizer: Adam
- Batch Size: 8-16

**Success Criteria:**
- Loss decreases consistently
- Predicted mean calibration close to GT mean
- Visual inspection: Calibration parameters reasonable

#### Stage 2: Visual-Conditioned Calibration

**Objective:** Learn to leverage visual features for improved calibration.

**Architecture:**
- Camera Encoder: ✓ Trainable (fine-tune)
- Visual-Camera Mixing: ✓ Trainable (unfreeze)
- Sequence Aggregation: ✓ Trainable (fine-tune)
- Camera Decoder: ✓ Trainable (fine-tune)

**Loss Function:**
```
L_stage2 = MSE(predicted_mean_calibration, gt_mean_calibration)
```

**Dataset:**
- Same as Stage 1
- Additionally extract visual tokens from AnyCam backbone

**Training Details:**
- Epochs: 20-50
- Learning Rate: 5e-5 (lower than Stage 1)
- Optimizer: Adam
- Batch Size: 4-8 (larger due to visual tokens)

**Success Criteria:**
- Loss lower than Stage 1
- Visual features improve calibration accuracy
- Ablation: Compare with/without visual tokens

#### Stage 3: End-to-End Flow Reprojection

**Objective:** Integrate into AnyCam pipeline and train with flow reprojection loss.

**Architecture:**
- Full AnyCam pipeline
- DA3 Calibration Head integrated
- Option: Use LoRA for efficient fine-tuning

**Loss Function:**
```
L_stage3 = Flow_Reprojection_Loss(poses, focal_length, flows, depths)
```

**Dataset:**
- Full training dataset (no GT calibration needed)
- Self-supervised training

**Training Details:**
- Epochs: 50-100
- Learning Rate: 1e-5 (very low for fine-tuning)
- Optimizer: Adam
- Batch Size: 2-4 (memory intensive)

**Freezing Strategy:**
- Option A: Freeze all except calibration head
- Option B: Use LoRA on calibration head
- Option C: Fine-tune calibration head + pose head together

**Success Criteria:**
- Flow reprojection loss decreases
- Calibration accuracy maintained or improved
- Pose accuracy maintained or improved

### Low-Rank Adaptation (LoRA) Option

**Rationale:** LoRA allows efficient fine-tuning with minimal parameter updates.

**Implementation:**

```python
from peft import LoraConfig, get_peft_model

# Apply LoRA to calibration head
lora_config = LoraConfig(
    r=16,  # Rank
    lora_alpha=32,
    target_modules=["encoder", "decoder", "attention"],  # Target modules
    lora_dropout=0.1
)

model = get_peft_model(model, lora_config)
```

**Benefits:**
- Reduced memory usage
- Faster training
- Fewer parameters to update
- Maintains pretrained knowledge

---

## Integration Points

### Integration into AnyCam Forward Pass

**Current Flow (AnyCam):**

```
1. Images [B, N, 3, H, W]
   ↓
2. Backbone → Visual Features
   ↓
3. Pose Processing → pose_tokens [B, N, D_pose]
   ↓
4. Sequence Token → seq_token [B, 1, D_seq]
   ↓
5. sequence_info_head → focal_enc [B, 1, 32]
   ↓
6. Focal Decoding → focal_length [B]
   ↓
7. Projection Matrix → K [B, 3, 3]
```

**New Flow (With DA3):**

```
1. Images [B, N, 3, H, W]
   ↓
2. Backbone → Visual Features
   ↓
3. Pose Processing → pose_tokens [B, N, D_pose]
   ↓
4. AnyCalib Inference → anycalib_preds [B, N, 4]
   ↓
5. DA3 Calibration Head:
   - Camera Encoder → camera_tokens [B, N, D_cam]
   - Visual-Camera Mixing → mixed_tokens [B, N, D_mixed]
   - Sequence Aggregation → aggregated_token [B, 1, D_cam]
   - Camera Decoder → camera_params [B, 1, 4]
   ↓
6. Extract focal_length = camera_params[:, 0, 0] [B]
   ↓
7. Projection Matrix → K [B, 3, 3]
```

### Key Integration Points

**1. Visual Token Extraction**
- **Location:** `anycam/models/anycam.py` line 327
- **Extract:** `pose_tokens` before pose-specific processing
- **Format:** [B, N, D_vis]

**2. AnyCalib Inference**
- **Location:** `anycam/trainer.py` or wrapper
- **Timing:** Before forward pass
- **Format:** [B, N, 4] (fx, fy, cx, cy)

**3. Calibration Head Call**
- **Location:** `anycam/models/anycam.py` line 379 (replace sequence_info_head)
- **Inputs:** visual_tokens, anycalib_predictions, image_size
- **Output:** camera_params [B, 1, 4]

**4. Focal Length Extraction**
- **Location:** `anycam/models/anycam.py` line 390
- **Extract:** `focal_length = camera_params[:, 0, 0]`
- **Format:** [B]

**5. Projection Matrix**
- **Location:** `anycam/trainer.py` line 379
- **Unchanged:** Uses focal_length as before

### Backward Compatibility

**Flag-Based Integration:**
```python
# Enable DA3 calibration
model = AnyCam(use_da3_calibration=True)

# Use original system
model = AnyCam(use_da3_calibration=False)
```

**Benefits:**
- Easy A/B testing
- Gradual rollout
- Fallback option

---

## Evaluation Metrics

### Calibration Accuracy Metrics

**1. Focal Length Error:**
```
Error_fx = |predicted_fx - gt_fx| / gt_fx
Error_fy = |predicted_fy - gt_fy| / gt_fy
```

**2. Principal Point Error:**
```
Error_cx = |predicted_cx - gt_cx| / image_width
Error_cy = |predicted_cy - gt_cy| / image_height
```

**3. Reprojection Error:**
```
Reprojection_Error = mean(||projected_point - observed_point||)
```

### Pose Accuracy Metrics

**1. Rotation Error:**
```
Rotation_Error = geodesic_distance(R_pred, R_gt) [degrees]
```

**2. Translation Direction Error:**
```
Translation_Error = angle(t_pred, t_gt) [degrees]
```

### Comparison Metrics

**Baseline Comparisons:**
- AnyCam (32-candidate system)
- AnyCalib (single-frame, no learning)
- Experiment 1 (AnyCalib injection, no learning)
- Experiment 2 (AnyCalib injection + multi-frame consistency)
- DA3 Calibration Head (proposed)

**Metrics to Report:**
- Mean and median errors
- Error distributions (histograms)
- Cumulative distribution functions (CDFs)
- Per-sequence breakdown

### Ablation Studies

**Components to Ablate:**
1. **Visual-Camera Mixing:**
   - With visual tokens vs. without
   - Number of attention layers
   - Attention vs. concatenation

2. **Sequence Aggregation:**
   - Learnable token vs. mean pooling
   - Attention vs. simple aggregation

3. **AnyCalib Initialization:**
   - Per-frame vs. first-frame only
   - With vs. without AnyCalib initialization

4. **Training Stages:**
   - Stage 1 only vs. Stage 2 vs. Stage 3
   - Impact of staged training

---

## Timeline and Milestones

### Week 1-2: Core Components

**Deliverables:**
- [ ] Camera Encoder implementation and tests
- [ ] Visual-Camera Mixing implementation and tests
- [ ] Sequence Aggregation implementation and tests
- [ ] Camera Decoder implementation and tests
- [ ] Complete DA3 Calibration Head integration

**Success Criteria:**
- All components implemented
- Unit tests passing
- Integration tests passing

### Week 3: AnyCam Integration

**Deliverables:**
- [ ] Modified AnyCam model with DA3 calibration head
- [ ] Trainer wrapper with AnyCalib inference
- [ ] Loss computation compatibility verified
- [ ] End-to-end forward pass working

**Success Criteria:**
- Model loads without errors
- Forward pass completes
- Output format compatible with existing code

### Week 4: Stage 1 Training

**Deliverables:**
- [ ] Stage 1 training script
- [ ] Dataset preparation (GT calibration extraction)
- [ ] Trained Stage 1 checkpoint
- [ ] Evaluation on validation set

**Success Criteria:**
- Loss decreases consistently
- Mean calibration error < 5%
- Checkpoint saved

### Week 5: Stage 2 Training

**Deliverables:**
- [ ] Stage 2 training script
- [ ] Visual token extraction pipeline
- [ ] Trained Stage 2 checkpoint
- [ ] Ablation study: with/without visual tokens

**Success Criteria:**
- Loss lower than Stage 1
- Visual tokens improve accuracy
- Checkpoint saved

### Week 6: Stage 3 Training

**Deliverables:**
- [ ] Stage 3 training script
- [ ] Full pipeline integration
- [ ] Trained Stage 3 checkpoint
- [ ] Evaluation on test set

**Success Criteria:**
- Flow reprojection loss decreases
- Calibration accuracy maintained
- Pose accuracy maintained or improved

### Week 7: Evaluation and Analysis

**Deliverables:**
- [ ] Comprehensive evaluation on multiple datasets
- [ ] Comparison with baselines
- [ ] Ablation studies
- [ ] Visualization and analysis

**Success Criteria:**
- All metrics computed
- Comparison plots generated
- Ablation results documented

### Week 8: Documentation and Thesis Writing

**Deliverables:**
- [ ] Updated thesis document
- [ ] Method section written
- [ ] Results section written
- [ ] Code documentation

**Success Criteria:**
- Thesis sections complete
- Code well-documented
- Results clearly presented

---

## Appendix: Code Structure

### New Files to Create

```
experiments/
├── models/
│   ├── __init__.py
│   ├── camera_encoder.py          # Camera parameter encoder
│   ├── visual_camera_mixing.py    # Visual-camera mixing module
│   ├── sequence_aggregation.py    # Sequence-level aggregation
│   ├── camera_decoder.py          # Camera parameter decoder
│   └── da3_calibration_head.py    # Complete calibration head
├── train_calibration_head_da3_stage1.py  # Stage 1 training
├── train_calibration_head_da3_stage2.py  # Stage 2 training
├── train_calibration_head_da3_stage3.py  # Stage 3 training
└── evaluate_da3_calibration.py           # Evaluation script
```

### Modified Files

```
anycam/
├── models/
│   └── anycam.py                  # Add DA3 calibration head option
└── trainer.py                    # Add AnyCalib inference wrapper

experiments/
└── train_pose_head_anycalib.py   # Update for DA3 integration
```

### Dependencies

**New Dependencies:**
- `peft` (for LoRA, optional): `pip install peft`

**Existing Dependencies:**
- PyTorch
- AnyCalib (already integrated)
- AnyCam (existing)

---

## References

1. **AnyCam Paper:** Wimbauer et al., "AnyCam: Learning to Recover Camera Poses and Intrinsics from Casual Videos", CVPR 2025
2. **AnyCalib Repository:** https://github.com/javrtg/AnyCalib
3. **Depth Anything 3 Repository:** https://github.com/ByteDance-Seed/Depth-Anything-3
4. **AnyCam Repository:** https://github.com/Brummi/anycam

---

## Architectural Alignment with Supervisor's Approach

### Key Corrections Made

This plan has been updated to properly align with the supervisor's suggested approach. The main corrections address:

#### 1. **Reverse Conditioning Direction (Critical Fix)**

**Previous Issue:** Visual-conditioned mixing was computed but never used in final prediction.

**Fixed Architecture:**
- Camera tokens are **updated** with visual information (not visual updated with camera)
- Updated camera tokens flow through aggregation and decoding
- Visual information directly influences final camera parameter prediction

**Flow:**
```
AnyCalib Init → Encode → Update with Visual (cross-attention) → Aggregate → Decode → Camera Params
```

#### 2. **Simpler Aggregation Option**

**Supervisor's Suggestion:** Simple average pooling of camera tokens.

**Implementation:**
- Added `use_learnable_token=False` option for simple mean pooling
- Learnable token with attention remains as optional enhancement
- Start with simple averaging, optionally upgrade later

#### 3. **DA3 Encoder/Decoder Usage**

**Supervisor's Suggestion:** Use DA3's actual `cam_enc.py` and `cam_dec.py` if available.

**Implementation:**
- Documented preference for using DA3's modules directly
- Provided DA3-inspired implementation as fallback
- Architecture matches DA3's normalization/denormalization approach

#### 4. **Staged Training Alignment**

**Supervisor's Approach:**
1. Stage 1: Train without visual tokens (mean calibration learning)
2. Stage 2: Add visual tokens (visual-conditioned calibration)
3. Stage 3: Full pipeline integration

**Implementation:**
- `use_visual_conditioning` flag allows skipping visual mixing in Stage 1
- Training scripts properly handle staged progression
- Loss functions match supervisor's suggestions

### Architecture Summary

**Correct Flow (After Fixes):**

```
Input: Visual Tokens [B, N, D_vis] + AnyCalib Predictions [B, N, 4]
    ↓
1. Camera Encoder: AnyCalib → Camera Tokens [B, N, D_cam]
    ↓
2. Visual-Camera Mixing: Camera Tokens query Visual Tokens
    → Updated Camera Tokens [B, N, D_cam] (visual-conditioned)
    ↓
3. Sequence Aggregation: Mean pool or learnable attention
    → Aggregated Token [B, 1, D_cam]
    ↓
4. Camera Decoder: Token → Camera Parameters [B, 1, 4]
    ↓
Output: fx, fy, cx, cy
```

**Key Principle:** Visual features **refine** camera predictions, not the other way around.

---

**Document Status:** Planning Phase (Architecturally Corrected)  
**Last Updated:** December 2025  
**Key Fixes:** Reverse conditioning direction, visual-conditioned camera tokens flow to output  
**Next Review:** After Phase 1 completion

