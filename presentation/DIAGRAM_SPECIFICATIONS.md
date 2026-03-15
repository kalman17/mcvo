# Diagram Specifications for Graphic Designer

6 diagrams for thesis paper and defense presentation. Clean scientific style, vector graphics.

**Color convention:**
- **Blue/teal**: frozen / pretrained
- **Orange/warm**: trainable (our contributions)
- **Gray**: non-differentiable
- **Dashed arrows**: gradient flow (Diagram 4 only)

---

# MATCHED TRIPTIC: Diagrams 1, 2, 3

Visual set with identical block names/shapes for shared components. All show the **inference pipeline** only — no loss, no training details. Those belong in Diagram 4.

---

## Diagram 1: AnyCam Pipeline

**Reference**: AnyCam Figure 2 (Wimbauer et al., CVPR 2025).

The 32-candidate system should be the most visually prominent block — it's what we replace.

```
N video frames
  │
  ├───► UniDepth → depth Dⁱ ──────────┐
  │                                    │ concatenated as
  ├───► UniMatch → flow Fⁱ→ʲ ─────────┤ extra input channels
  │                                    │
  ▼                                    ▼
┌──────────────────────────────────────────────┐
│  DINOv2-small (ViT-S/14)                    │
│  Input: [images, depth, flow]                │
│  → CLS tokens [384-dim] from 4 stages       │
│  → spatial feature maps                      │
└───────┬─────────────────────────┬────────────┘
        │                         │
        │ CLS tokens              │ spatial features
        ▼                         ▼
┌────────────────────┐   ┌──────────────────┐
│  Pose Neck         │   │ Uncertainty Head │
│                    │   │ (DPT decoder)    │
│  reassemble        │   │ → per-pixel σ    │
│  fusion            │   └────────┬─────────┘
│  8-layer self-attn │            │
│  seq token attn    │            ▼
│                    │       Uncertainty σ
│  → pose tokens     │
│    [128-dim]       │
│  → sequence token  │
│    [128-dim]       │
└──┬─────────────┬───┘
   │             │
   │ pose        │ seq
   │ tokens      │ token
   │             │
   │             ▼
   │    ┌──────────────────────────────┐
   │    │  Sequence Head               │
   │    │  MLP on seq token → scores   │
   │    │  for all 32 candidates       │
   │    │  → selects best f            │
   │    └──────────────┬───────────────┘
   │                   │ best focal length
   ▼                   ▼
┌────────────────────────────────────────────────────────┐
│  ███ 32-Candidate System ██████████████████████████    │
│                                                        │
│  32 focal guesses {f₁, ..., f₃₂}                     │
│                                                        │
│  For each fₖ: focal embedding [8] + pose token [128]  │
│                                                        │
│  ┌────────┐  ┌────────┐       ┌────────┐              │
│  │Pose    │  │Pose    │       │Pose    │              │
│  │Head    │  │Head    │  ...  │Head    │  ×32         │
│  │[136]→  │  │[136]→  │       │[136]→  │  shared     │
│  │[64]→[7]│  │[64]→[7]│       │[64]→[7]│  weights    │
│  └───┬────┘  └───┬────┘       └───┬────┘              │
│      ▼           ▼               ▼                    │
│    P_f₁        P_f₂           P_f₃₂                  │
│      └───────────┴───────┬───────┘                    │
│                          │ select best (from Seq Head)│
│                                                        │
│  EXPENSIVE: 32× pose head runs per pair               │
│  COARSE: only 32 discrete focal choices               │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
                  Relative Pose (R, t)
                  Focal Length f
                  Uncertainty σ
```

**Arrow summary:**
1. Frames → UniDepth → depth, Frames → UniMatch → flow
2. Images + depth + flow → DINOv2-small (concatenated input)
3. DINOv2 → CLS tokens → Pose Neck
4. DINOv2 → spatial features → Uncertainty Head → σ
5. Pose Neck → pose tokens → 32× Pose Head (inside 32-cand box)
6. Pose Neck → seq token → Sequence Head → selects best candidate
7. Best candidate → output Pose + f + σ

**Visual guidance:**
- **32-Candidate box**: LARGEST block. Bold/shaded border. Shows fan-out to 3 Pose Head copies + "...×32"
- **Sequence Head**: separate block, runs ONCE, feeds selection INTO the 32-cand box
- **Uncertainty Head**: separate block off the spatial features branch
- **UniDepth, UniMatch**: small blocks, arrows clearly FROM frames, outputs INTO backbone

---

## Diagram 2: AnyCalib Pipeline

**Reference**: AnyCalib Figure 2 (Tirado-Garín & Civera, 2025).

```
Single Image
       │
       ▼
┌─────────────────┐
│  DINOv2 ViT-L   │
│  (ViT-L/14)     │
│  [1024-dim]     │
│  4 scales from   │
│  blocks 4,11,17,23│
└────────┬────────┘
         │
         │ ◄── MCT inserted here in Diagram 3
         │
         ▼
┌─────────────────┐
│  Light-DPT      │
│  decoder        │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Convex         │
│  Upsampling     │
└────────┬────────┘
         │
         ▼
┌─────────────────────┐
│  FoV Field          │
│  θ ∈ T_{z₁}S²      │
│  per-pixel angles   │
└────────┬────────────┘
         │ Exp map → unit rays
         ▼
┌─────────────────────┐
│  Per-Pixel Rays     │
│  p ∈ S²            │
└────────┬────────────┘
         │ Closed-form Ax = b
         ▼
┌─────────────────────┐
│  Intrinsics K       │
│  [f_x, f_y, c_x, c_y]│
└─────────────────────┘

⚠ Per-frame only — no cross-frame consistency
```

**Visual guidance:**
- All blocks same style/name as Diagram 3
- Subtle annotation between ViT-L and Light-DPT: "MCT inserted here"
- Per-frame annotation at bottom

---

## Diagram 3: Our Pipeline

**Purpose**: Main method figure. AnyCam's pose path (left) + AnyCalib's calibration path (right), minus 32-candidate system, plus MCT, plus cross-wire K → Pose Head.

```
N video frames
  │
  ├───► UniDepth → depth Dⁱ ──────────┐
  │                                    │ concatenated as
  ├───► UniMatch → flow Fⁱ→ʲ ─────────┤ extra input channels
  │                                    │
  ▼                                    ▼                              ▼
┌──────────────────────────────────────────────┐   ┌─────────────────┐
│  DINOv2-small (ViT-S/14)                    │   │  DINOv2 ViT-L   │
│  Input: [images, depth, flow]                │   │  (ViT-L/14)     │
│  → CLS tokens [384-dim] from 4 stages       │   │  [1024-dim]     │
│  → spatial feature maps                      │   │  4 scales       │
└───────┬─────────────────────────┬────────────┘   └────────┬────────┘
        │                         │                          │
        │ CLS tokens              │ spatial features         │ per frame
        ▼                         ▼                          ▼
┌────────────────────┐   ┌──────────────────┐   ┌────────────────────────┐
│  Pose Neck         │   │ Uncertainty Head │   │      ★ MCT ★           │
│                    │   │ (DPT decoder)    │   │  Multi-Frame Calib.    │
│  reassemble        │   │ → per-pixel σ    │   │  Transformer           │
│  fusion            │   └────────┬─────────┘   │                        │
│  8-layer self-attn │            │              │  Cross-frame attention │
│  seq token attn    │            ▼              │  at 4 scales           │
│                    │       Uncertainty σ       │  N frames → 1 output  │
│  → pose token      │                          │  (~25M params, NEW)    │
│    [128-dim]       │                          └───────────┬────────────┘
└────────┬───────────┘                                      │
         │                                                  │ aggregated features
         │ pose token [128]                                 ▼
         │                                       ┌─────────────────┐
         │                                       │  Light-DPT      │
         │                                       │  decoder        │
         │                                       └────────┬────────┘
         │                                                │
         │                                                ▼
         │                                       ┌─────────────────┐
         │                                       │  Convex         │
         │                                       │  Upsampling     │
         │                                       └────────┬────────┘
         │                                                │
         │                                                ▼
         │                                       ┌───────────────────┐
         │                                       │  FoV Field → Rays │
         │                                       │  → Closed-form K  │
         │                                       │ [f_x,f_y,c_x,c_y]│
         │                                       └────────┬──────────┘
         │                                                │
         │             focal embedding [8]                │
         │          ◄─────────────────────────────────────┘
         │                │
         ▼                ▼
┌──────────────────────────────┐
│  Pose Head                   │
│  [128+8=136]→[64]→[7]       │
│  → quaternion + translation  │
└──────────┬───────────────────┘
           │
           ▼
  Relative Pose (R, t)
  Intrinsics K
  Uncertainty σ
```

**Arrow summary:**
1. Frames → UniDepth → depth, Frames → UniMatch → flow (same as Diag 1)
2. Images + depth + flow → DINOv2-small (same as Diag 1)
3. DINOv2-small → CLS tokens → Pose Neck → pose token (same as Diag 1)
4. DINOv2-small → spatial features → Uncertainty Head → σ (same as Diag 1)
5. Frames → DINOv2 ViT-L → 4-scale features (from Diag 2)
6. ViT-L features → **★ MCT ★** → aggregated features (**NEW**)
7. MCT output → Light-DPT → Convex Ups → FoV → Rays → K (from Diag 2)
8. K → focal embedding [8] → Pose Head (**NEW cross-wire**)
9. Pose token [128] + focal emb [8] → Pose Head → Pose (R,t) (rewired from Diag 1)

**Comparison table:**

| Component | Diag 1 (AnyCam) | Diag 2 (AnyCalib) | Diag 3 (Ours) |
|---|---|---|---|
| UniDepth/UniMatch → backbone | ✓ | — | ✓ same |
| DINOv2-small | ✓ | — | ✓ same |
| Pose Neck | ✓ | — | ✓ same |
| Uncertainty Head | ✓ | — | ✓ same |
| **32-Candidate System** | ✓ **BIG** | — | **GONE** |
| Pose Head | ×32 inside 32-cand | — | **×1 standalone** |
| Sequence Head | ✓ | — | **GONE** |
| DINOv2 ViT-L | — | ✓ | ✓ same |
| Light-DPT | — | ✓ | ✓ same |
| Convex Upsampling | — | ✓ | ✓ same |
| FoV Field → Rays → K | — | ✓ | ✓ same |
| **★ MCT ★** | — | — | **NEW** |
| **K → focal emb → Pose Head** | — | — | **NEW** |

**Visual emphasis:**
- **MCT**: LARGEST block, orange, bold border, ★ marker
- **Pose Head**: standalone (vs buried ×32 in Diag 1)
- **Cross-wire** K → Pose Head: thick/colored arrow
- All other blocks: same style as their Diag 1/2 counterparts

---

## Diagram 4: Training View (Frozen vs Trainable + Loss)

**Purpose**: Same layout as Diagram 3 with color coding + loss functions at bottom. This is the ONLY diagram that shows training components.

**Color coding on Diagram 3 layout:**

**FROZEN (blue/teal):**
- DINOv2-small (~22M), DINOv2 ViT-L (~300M)
- Uncertainty Head, Light-DPT, Convex Upsampling
- FoV/Rays/K fitting (gray, non-differentiable)
- Focal embedding, UniDepth (~87M), UniMatch (~100M+)

**TRAINABLE (orange):**
- **Pose Neck** (~2.5M)
- **Pose Head** (~21K)
- **★ MCT ★** (~25M)
- Total: **~27.5M / ~370M = 7.5%**

**Loss block at bottom:**
```
Pose (R,t)    K    Depth Dⁱ    Flow Fⁱ→ʲ    Uncertainty σ
    │         │       │            │              │
    ▼         ▼       ▼            ▼              ▼
┌──────────────────────────────────────────────────────┐
│  Training Losses (all self-supervised)               │
│                                                      │
│  • Flow reprojection (Laplacian NLL, weighted by σ) │
│  • Pose consistency (forward-backward)              │
│  • Composed flow (multi-frame, 0.1× weight)        │
│  • Calibration anchor (K vs AnyCalib pseudo-GT)     │
└──────────────────────────────────────────────────────┘
```

**Visual:** dashed gradient arrows only through trainable blocks. Parameter counts next to blocks. Legend: blue=frozen, orange=trainable, gray=non-differentiable.

---

## Diagram 5: MCT Architecture (Zoomed In)

**Purpose**: Detailed view of the MCT block from Diagram 3.

```
FROM: DINOv2 ViT-L (frozen)
      │
      │ 4 scales per frame: [N, 1024, h, w]
      │ blocks 4, 11, 17, 23
      │ (h=w=24 for 336×336, N≈4 frames)
      ▼
┌──────────────────────────────────────────────────────┐
│          MCT (~25M params, trainable)                 │
│                                                      │
│  For each of 4 scales (shared weights):              │
│                                                      │
│  [N, 1024, h, w]                                     │
│       │ flatten                                      │
│       ▼                                              │
│  [h×w, N, 1024]  (spatial pos × frames × dim)       │
│       │                                              │
│       │ optional: concat DINOv2-small CLS tokens    │
│       │ [N,384] → project → [h×w, N, 1024]         │
│       │ → [h×w, 2N, 1024]                           │
│       ▼                                              │
│  ┌──────────────────────────────────┐                │
│  │ Transformer Encoder (2 layers)  │                │
│  │                                  │                │
│  │ Layer ×2:                        │                │
│  │   LayerNorm → MHA (8 heads)     │                │
│  │   + Residual                    │                │
│  │   LayerNorm → FFN (1024→4096)   │                │
│  │   + Residual                    │                │
│  │                                  │                │
│  │ Attention across FRAME dim:     │                │
│  │ frame₁ ↔ frame₂ ↔ ... ↔ frameₙ │                │
│  │ (spatial positions independent) │                │
│  └──────────────────────────────────┘                │
│       │                                              │
│       │ discard visual tokens, keep spatial          │
│       │ final LayerNorm                              │
│       │ mean pool across N frames                    │
│       ▼                                              │
│  [1, 1024, h, w]  (N frames → 1 output)             │
│                                                      │
│  ×4 scales (SAME transformer weights)                │
└──────────────────────────────────────────────────────┘
      │
      │ 4 aggregated feature maps [1, 1024, h, w]
      ▼
TO: Light-DPT → Convex Ups → FoV → Rays → K
```

**Key visuals:**
- **Attention inset**: at one pixel, show N=4 dots (frames) with bidirectional arrows → mean → 1 dot
- **Weight sharing**: same transformer for all 4 scales
- **No spatial attention**: pixels are batch items, only frames attend to frames

---

## Diagram 6: Flow Composition

**Purpose**: How composed flow gives multi-frame consistency at O(N) cost.

```
4 frames: I₁, I₂, I₃, I₄

Consecutive (UniMatch):
  I₁ ──F₁₂──► I₂ ──F₂₃──► I₃ ──F₃₄──► I₄

Composed (bilinear warping, no network):
  I₁ ─ ─ F₁₃ ─ ─► I₃        = compose(F₁₂, F₂₃)
  I₁ ─ ─ ─ F₁₄ ─ ─ ─► I₄   = compose(F₁₃, F₃₄)

Per 4-frame window: 3 consecutive + 2 composed = 5 pairs
Cost: O(N) instead of O(N²)
```

**Visual**: 4 frames in a row. Solid arrows = consecutive. Curved dashed = composed.

---

## Summary

| # | Diagram | Slide / Chapter | Priority |
|---|---------|----------------|----------|
| 1 | AnyCam | Slide 4, Ch. 4 | **HIGH** |
| 2 | AnyCalib | Slide 5, Ch. 4 | **HIGH** |
| 3 | Our Pipeline | Slide 6, Ch. 5 | **HIGH** |
| 4 | Training View | Slide 8, Ch. 5 | **HIGH** |
| 5 | MCT Zoomed | Slide 7, Ch. 5 | Medium |
| 6 | Flow Composition | Ch. 5 | Low |

**Triptic story (1→2→3):**
1. AnyCam: good poses, but 32-candidate calibration is expensive/coarse (big box)
2. AnyCalib: excellent calibration, but per-frame only (gap where MCT goes)
3. Ours: AnyCam pose path minus 32-cand + AnyCalib calib path plus MCT + cross-wire K→Pose Head
