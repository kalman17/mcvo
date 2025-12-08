# Quick Start: Multi-Pair Training & Evaluation

## TL;DR - Run This Now

```bash
# Test the new multi-pair extraction (quick test)
bash experiments/run_experiment.sh multi_pair

# Or run full training with evaluation
bash experiments/run_experiment.sh full_with_eval
```

## What's New?

### 1. Multi-Pair Extraction = More Training Data

**Before:** Each video contributed 1 training sample (frames 0-1)  
**Now:** Each video contributes ALL consecutive frame pairs!

Example:
- Video with 100 frames → 50 training pairs: (0,1), (2,3), (4,5), ..., (98,99)
- 100 videos with 100 frames each → 5,000 training samples instead of 100!

### 2. Automatic Train/Val/Test Split

The first time you run training, a split file is created and saved:
```
experiments/objectron_split.json
```

This ensures:
- ✅ Reproducible experiments
- ✅ No data leakage between train/test
- ✅ Consistent evaluation across runs

### 3. Automatic Evaluation After Training

Training now automatically evaluates on test set (when requested) and reports:
- Rotation error (degrees)
- Translation direction error (degrees)
- Statistics: mean, median, std, percentiles

## Run Modes

### Mode 1: Test Run (Sanity Check)
```bash
bash experiments/run_experiment.sh
```
- 5 videos, 2 epochs
- Single pair per video (backward compatible)
- Fast (~5 minutes)
- Use this to verify everything works

### Mode 2: Multi-Pair Training
```bash
bash experiments/run_experiment.sh multi_pair
```
- 10 videos, 10 epochs
- **ALL consecutive pairs extracted**
- Batch size 4 (more data = bigger batches)
- ~30-60 minutes depending on video lengths

### Mode 3: Full Training
```bash
bash experiments/run_experiment.sh full
```
- All videos, 50 epochs
- Single pair per video
- No evaluation
- For baseline comparison

### Mode 4: Full Training + Evaluation
```bash
bash experiments/run_experiment.sh full_with_eval
```
- All videos, 50 epochs
- Single pair per video
- **Automatic evaluation on test set**
- Requires ground truth annotations

## What Evaluation Tells You

After training, you'll see output like:

```
======================================================================
EVALUATION RESULTS
======================================================================
Test Set Size: 250 frame pairs

Rotation Error (degrees):
  Mean:   12.3456
  Median: 10.2345
  Std:    8.7654
  P90:    25.4321

Translation Direction Error (degrees):
  Mean:   15.6789
  Median: 13.4567
  Std:    9.8765
  P90:    28.9012
======================================================================
```

**What this means:**
- **Rotation Error**: How accurately the model predicts camera rotation
  - Lower is better
  - ~10° is pretty good, <5° is excellent
  
- **Translation Direction Error**: How well the model predicts motion direction
  - Ignores distance/scale (we're not predicting absolute distance)
  - Lower is better
  - ~15° is reasonable, <10° is great

## Output Files

After training with evaluation, you'll find:

```
experiments/pose_head_experiment_results/YOUR_RUN/
├── checkpoint_epoch_5.pt
├── checkpoint_epoch_10.pt
├── ...
├── final_model.pt
├── loss_curve.png                 # Training loss visualization
├── training_log.txt               # Detailed training log
├── training_summary.txt           # Training statistics
├── loss_history.json              # Raw loss data
└── evaluation/                    # NEW!
    └── evaluation_results.json    # Pose accuracy metrics
```

## Advanced: Custom Training

### Train with Multi-Pair + Evaluation
```bash
python experiments/train_pose_head_anycalib.py \
    --extract_all_pairs \
    --run_evaluation \
    --num_epochs 50 \
    --batch_size 4 \
    --save_dir experiments/pose_head_experiment_results/my_experiment
```

### Evaluation Only (After Training)
```bash
python experiments/train_pose_head_anycalib.py \
    --eval_only \
    --run_evaluation \
    --save_dir experiments/pose_head_experiment_results/full_run
```

### Full Evaluation with Baseline Comparison
```bash
python experiments/evaluate_pose_model.py \
    --our_model_path experiments/pose_head_experiment_results/full_run/final_model.pt \
    --baseline_model_path pretrained_models/anycam_seq8 \
    --save_dir experiments/evaluation_comparison
```

This generates:
- Error distribution histograms
- CDF curves  
- Bar chart comparisons
- Detailed text report

## Troubleshooting

### Issue: "No GT found" during evaluation
**Solution:** Evaluation requires ground truth annotations. Make sure:
```bash
ls <DATASETS_ROOT>/Objectron/annotations/
```
contains `.json` files matching your videos.

### Issue: Out of memory with multi-pair extraction
**Solution:** Reduce batch size:
```bash
python experiments/train_pose_head_anycalib.py \
    --extract_all_pairs \
    --batch_size 1  # Instead of 4
    --num_epochs 50
```

### Issue: Training is slow
**Solution:** Multi-pair extraction gives you MORE data, so training takes longer. This is expected and desirable (more data = better model). If you want faster iteration:
1. Use fewer videos: `--max_sequences 10`
2. Use fewer epochs: `--num_epochs 10`
3. Don't use multi-pair initially

## Next Experiment: Multi-Frame Reprojection

Once you've validated that multi-pair training works, the next step is:

**Experiment 2: Multi-frame reprojection**
- Predict from frame 1 to frames 2, 3, 4 (not just 1→2)
- Stack optical flow for long-range predictions
- More loss constraints → potentially better poses

This requires:
- Modifying the pose predictor to output multiple frames
- Flow stacking mechanism
- Extended loss function

(To be implemented next)

## Questions?

Check these files for more details:
- `experiments/MULTI_PAIR_IMPLEMENTATION_SUMMARY.md` - Full technical details
- `experiments/ARCHITECTURE_FINDINGS.md` - AnyCam architecture overview
- `experiments/UNSUPERVISED_TRAINING_NOTE.md` - Why we don't need GT for training

---

**Ready to train with more data? Try the multi_pair mode!** 🚀

