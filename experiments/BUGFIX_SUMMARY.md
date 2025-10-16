# Bug Fix: Ground Truth Not Required for Unsupervised Training

## Issue Discovered

When running the experiment, it failed with:
```
[WARN] No GT found for batch-10_0_video.MOV, skipping
...
[DATASET] 0 sequences have valid GT
ValueError: num_samples should be a positive integer value, but got num_samples=0
```

## Root Cause Analysis

The user made an excellent observation:
> "is the issue because it couldnt find any ground truth or because of something else? because it shouldnt need the GT, it is only based on the reprojection loss and is entirely unsupervised isnt it?"

**The user was absolutely correct!** 🎯

### Two Issues Identified:

1. **Naming Mismatch** (surface issue):
   - Videos: `batch-10_0_video.MOV`
   - Expected GT: `batch-10_0_video.json`
   - Actual GT: `batch-10_0.json` (missing `_video` suffix)

2. **Conceptual Issue** (deeper issue):
   - **GT is NOT needed for training!**
   - Training is fully unsupervised using flow reprojection loss
   - GT was included only for optional validation/monitoring

---

## The Fix

### Changes Made to `train_pose_head_anycalib.py`:

#### 1. Made GT Optional in Dataset Constructor
```python
def __init__(
    self, 
    videos_dir: str,
    gt_dir: Optional[str] = None,  # ← Now optional
    num_frames: int = 2,
    max_sequences: Optional[int] = None,
    image_size: Tuple[int, int] = (480, 640),
    require_gt: bool = False,  # ← New parameter
):
```

#### 2. Skip GT Validation by Default
```python
if self.require_gt and self.gt_dir:
    self._validate_dataset()
else:
    print(f"[DATASET] Running in UNSUPERVISED mode (GT not required)")
```

#### 3. Fixed GT File Matching (Two Naming Patterns)
```python
# Pattern 1: batch-10_0_video.MOV -> batch-10_0_video.json
gt_path1 = self.gt_dir / f"{video_path.stem}.json"
# Pattern 2: batch-10_0_video.MOV -> batch-10_0.json (remove "_video")
stem_without_video = video_path.stem.replace("_video", "")
gt_path2 = self.gt_dir / f"{stem_without_video}.json"
```

#### 4. Create Dummy Placeholders if No GT
```python
# If no GT available, create dummy placeholders (not used in unsupervised training)
if projs is None:
    projs = torch.eye(3).unsqueeze(0).repeat(self.num_frames, 1, 1)
if poses is None:
    poses = torch.eye(4).unsqueeze(0).repeat(self.num_frames, 1, 1)
```

#### 5. Updated Main Function
```python
print(f"[INFO] This is UNSUPERVISED training - GT is NOT required!")
print(f"[INFO] Training uses flow reprojection loss only")
dataset = ObjectronVideoDataset(
    videos_dir=args.videos_dir,
    gt_dir=args.gt_dir,  # Can be None
    num_frames=args.num_frames,
    max_sequences=args.max_sequences,
    require_gt=False,  # ← Key change!
)
```

#### 6. Made GT Directory Optional in Args
```python
parser.add_argument("--gt_dir", type=str,
                   default=None,  # ← Changed from path to None
                   help="Directory with Objectron ground truth JSON files (optional, only for validation)")
```

---

## Why This Works

### Unsupervised Training Flow:

```
Images → [Depth Predictor] → Depth Maps
              ↓
Images → [UniMatch] → Optical Flow
              ↓
Images → [AnyCaLib] → Focal Length
              ↓
        [Pose Predictor] → Predicted Poses
              ↓
  Depth + Pose + Focal → Predicted Flow
              ↓
    || Predicted Flow - Observed Flow ||² → Loss
              ↓
         Backprop → Update Pose Head
```

**No ground truth needed at any step!**

### What Each Component Does:

| Component | Input | Output | Supervised? |
|-----------|-------|--------|-------------|
| Depth Predictor | Image | Depth map | No (frozen, pretrained) |
| UniMatch | Image pair | Optical flow | No (unsupervised) |
| AnyCaLib | Image | Focal length | No (direct prediction) |
| Pose Head | Features | Pose | **Learning here!** |
| Flow Loss | Flows | Scalar | No (consistency loss) |

---

## Why GT Was Originally Included

The GT loading code was added for:

1. **Validation** - Compare predicted poses to GT during/after training
2. **Monitoring** - Track pose error metrics
3. **Debugging** - Ensure pipeline produces reasonable results
4. **Convention** - Many datasets come with GT, so it's common to load it

But it was never required for the core training loop!

---

## Impact of the Fix

### Before:
- ❌ Required GT files to exist
- ❌ Failed on naming mismatch
- ❌ Could only train on annotated datasets
- ❌ Crashed with 0 sequences if GT missing

### After:
- ✅ Works without GT files
- ✅ Handles naming mismatches gracefully
- ✅ Can train on any video dataset
- ✅ Loads all available sequences

---

## How to Run Now

```bash
cd /home/kalman/TUM/thesis/anycam

# Run experiment (no GT needed!)
bash experiments/run_experiment.sh

# Or manually
python experiments/train_pose_head_anycalib.py \
    --videos_dir /home/kalman/TUM/thesis/Objectron/videos/ \
    --max_sequences 5 \
    --num_epochs 2
```

Expected output:
```
[STEP 1] Loading Objectron dataset...
[INFO] This is UNSUPERVISED training - GT is NOT required!
[INFO] Training uses flow reprojection loss only
[DATASET] Found 5 video sequences
[DATASET] Running in UNSUPERVISED mode (GT not required)
[STEP 1] Dataset loaded: 5 sequences  ← Success!
```

---

## Optional: Using GT for Validation

If you want to use GT for validation later, you can:

```bash
python experiments/train_pose_head_anycalib.py \
    --videos_dir /home/kalman/TUM/thesis/Objectron/videos/ \
    --gt_dir /home/kalman/TUM/thesis/Objectron/processed_gt/ \
    --max_sequences 5
```

The script will now:
1. Try to load GT files (with both naming patterns)
2. Use GT if available (for potential validation hooks)
3. Use dummy placeholders if not available
4. Train successfully either way

---

## Key Takeaway

**The user's intuition was spot-on!** 

This is a **fully unsupervised** training approach. Ground truth is completely optional and only useful for validation/monitoring, not for the actual training loss.

This fix makes the experiment:
- ✅ More flexible (works with any videos)
- ✅ More robust (handles missing GT gracefully)
- ✅ Conceptually clearer (reflects the true unsupervised nature)
- ✅ Ready to scale (can use web videos, YouTube, etc.)

---

**Status: Fixed and ready to run!** 🚀

The experiment should now work without any ground truth files, which is exactly how unsupervised training should work.

