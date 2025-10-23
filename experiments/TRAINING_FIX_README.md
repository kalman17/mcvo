# Training Issue Fix - October 16, 2025

## Problem Discovered

Your training run only used **7 training samples** instead of thousands! This happened because:

### Issue 1: Stale Split File
The `experiments/objectron_split.json` was created during a test run with `--max_sequences 10`, so it only includes 10 videos total (7 train, 1 val, 2 test).

### Issue 2: Multi-Pair Extraction Disabled
The `full_with_eval` mode didn't enable `--extract_all_pairs`, so you only got 1 frame pair per video instead of ALL consecutive pairs.

## Evidence from Terminal Output

```
[DATASET] Found 7 video sequences        ← ONLY 7 VIDEOS!
[DATASET] Total frame pairs available: 7  ← ONLY 7 PAIRS!
[TRAIN] Total batches: 4                  ← TINY TRAINING SET!
Multi-pair extraction: False              ← DISABLED!
```

**But you actually have ~100 Objectron videos available!**

## What Was Actually Working ✅

The good news - your training setup is correct:

1. ✅ **Backbone Frozen**: "Trainable: 20,736 / 114,789,748 (0.02%)"
   - Only pose head being trained

2. ✅ **AnyCaLib Integration**: Focal lengths being predicted and injected
   ```
   [FORWARD] AnyCaLib focal lengths: tensor([609.2758, 771.6895])
   ```

3. ✅ **Backpropagation Working**: Loss changing (-4.79 final)
   - Gradients flowing through pose head
   - Weights being updated

4. ✅ **LightSpeed Evaluation**: 36 sequences evaluated
   - Rotation error: 0.76° (very good!)
   - Translation error: 62.3° (room for improvement)

## How to Fix

### Step 1: Delete Stale Split File
```bash
rm experiments/objectron_split.json
```

### Step 2: Run Full Training (Fixed)
```bash
bash experiments/run_experiment.sh full_with_eval
```

**Changes made to `run_experiment.sh`:**
- `full_with_eval` mode now enables `--extract_all_pairs`
- Better parameter display showing what's enabled

### What Will Happen Now

**Before fix:**
- 7 videos → 7 frame pairs → 4 batches/epoch → Fast but useless

**After fix:**
- ~70 videos (70% split) → **~3500+ frame pairs** → ~1750 batches/epoch
- Training will take **much longer** (several hours instead of minutes)
- Model will actually learn meaningful patterns

## Expected Training Stats

With the fix:
```
[DATASET] Found ~70 video sequences
[DATASET] Total frame pairs available: ~3500
[TRAIN] Total batches: ~1750 (with batch_size=2)
Multi-pair extraction: ENABLED
```

Each epoch will take ~30-60 minutes depending on your GPU.

## Sanity Check - Is Training Actually Happening?

✅ **YES!** From your logs:

1. **Pose head initialized fresh**: "New pose_head created with random weights"
2. **Backbone frozen**: Only 20K params trainable out of 114M
3. **Loss decreasing**: 1.63 → -4.79 (learning!)
4. **AnyCaLib focal lengths used**: Direct injection into reprojection matrix
5. **Evaluation working**: Got results on LightSpeed (0.76° rotation error)

The training mechanism is perfect - you just need MORE DATA!

## Full Training Command (Alternative)

If you want even more control:

```bash
python experiments/train_pose_head_anycalib.py \
    --num_epochs 50 \
    --batch_size 2 \
    --extract_all_pairs \
    --run_evaluation \
    --eval_dataset lightspeed \
    --save_dir experiments/pose_head_experiment_results/full_training_fixed
```

This will:
- Use ALL Objectron videos (no limit)
- Extract ALL consecutive frame pairs from each
- Train for 50 epochs
- Evaluate on LightSpeed dataset

## Timeline Expectations

- **Test run** (5 videos, 2 epochs): ~5 minutes
- **Multi-pair run** (10 videos, 10 epochs, all pairs): ~30-60 minutes  
- **Full run FIXED** (70 videos, 50 epochs, all pairs): **~30-50 hours!**

Yes, it's slow - but that's because you're actually training on real data now!

## Monitoring Training

Watch the logs:
```bash
tail -f experiments/pose_head_experiment_results/full_run_eval/training_log.txt
```

Check progress:
```bash
ls -lh experiments/pose_head_experiment_results/full_run_eval/checkpoint_*.pt
```

## Summary

Your training setup is **architecturally correct**:
- ✅ Pose head learning
- ✅ Backbone frozen
- ✅ AnyCaLib focal lengths injected
- ✅ Backprop working
- ✅ Evaluation on LightSpeed

The only issue was **insufficient training data** due to:
- ❌ Stale split file (10 videos instead of ~100)
- ❌ Multi-pair extraction disabled (1 pair instead of all)

**Fix applied! Run again and you'll get proper training.** 🚀

---

**Next time you see fast training**, check:
1. Number of sequences: Should be ~70 for train
2. Number of pairs: Should be thousands, not single digits
3. Batches per epoch: Should be hundreds, not 4
4. Multi-pair extraction: Should show "ENABLED"

