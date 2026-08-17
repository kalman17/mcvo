# Changelog

## 2026-08-17 — Repository renamed `anycam-extension` → `mcvo`

The README now leads with the image-only visual-odometry model (MCVO); the thesis calibration
pipeline (MCT + AnyCam) is kept in full as the calibration branch, second half of the README.
Old links redirect.

## 2026-08-17 — Correction: KITTI evaluation input scale

The KITTI loader used by the benchmark harness returned pixel values in 0–255 instead of
0–1 (the Sintel and TUM-RGBD loaders were correct), so every KITTI number published here
before this date was measured on out-of-spec input, for every method alike. Found by
cross-checking two independent loaders on the same frames; fixed in
`experiments/kitti_dataset.py`, and the harness now asserts float 0–1 input from every
loader. Re-measured on identical windows (`honest_benchmarks/kfix_*`) and, for the
native-frame table, re-run for all methods under one protocol (`honest_benchmarks/N_kitti`,
`N_sintel`, `S_kitti`). What changed:

- KITTI square-window calibration: AnyCalib 18.4 % (was 10.4), MCT 20.4 % (was 7.2),
  AnyCam 66.9 % (was 94.8). MCT does not beat AnyCalib on these crops.
- Multi-frame aggregation on KITTI: no gain (MCT 21.9 % vs per-frame averaging 19.8 % at
  8 frames; was 6.0 vs 9.1). The aggregation gain reported for KITTI is withdrawn; this
  checkpoint does not beat averaging on Sintel or TUM-RGBD either.
- KITTI translation direction: AnyCam 28.6° (was 89.8), MCT 28.2° (was 68.8). The
  statement that the baseline sat at chance level was our artefact.
- Native wide frames, one protocol for all methods (16 four-frame windows per sequence,
  own preprocessing): VGGT 11.6 %, AnyCalib 14.2 %, MCT 15.7 %, Pi3 20.4 %, DA3 38.0 %.
  The earlier 3.99 % came from whole-sequence 8-frame multi-crop inference on correctly
  scaled input; it remains reproducible with that script but was not comparable to the
  competitor numbers placed next to it, which came from ad-hoc runs. VGGT and Pi3 do not
  fail on wide frames when given the full frame.
- VGGT / Pi3 / DA3 on KITTI windows: 0.1° rotation and 1–5° translation direction; the
  earlier statement that they collapse on small-baseline driving footage is withdrawn.
- Unaffected: all Sintel and TUM-RGBD numbers.

## 2026-08 — Corrected evaluation and updated results

The numbers previously shown here (and in the thesis document) came from a benchmark
pipeline that turned out to have three bugs. This update replaces them with results
from a rebuilt evaluation, after fixing the bugs and retraining the calibration
module. In chronological order:

**Audit.** A review of the benchmark code found: (1) the "vanilla AnyCam" baseline was
loading the official checkpoint into a modified architecture, silently discarding part
of the pretrained pose head — it effectively ran with random weights in its first
pose-head layer, which is why its translation-direction error sat at chance level in
the old tables; (2) the AnyCalib reference was invoked without its official input
preprocessing, understating it by roughly a third; (3) the evaluation constructed our
own model with an untrained input path enabled, injecting noise at test time. The old
headline claims (−39.5 % translation vs AnyCam, −32.5 % calibration vs AnyCalib, 2.0×
ATE on `market_6`) were artifacts of (1)–(3) and are withdrawn.

**Root cause of the KITTI calibration failure.** The former out-of-distribution
blow-up (focal errors in the thousands of percent) was traced to aliased, incorrectly
resized inputs reaching the ray-based calibration decoder — an input-handling bug, not
an architecture limit. With AnyCalib-spec input normalization, the same trained
weights went from catastrophic to better than the specialist baseline on KITTI.

**Retraining.** The MCT was retrained with the corrected input handling on the full
training set (~74 k windows), warm-started from the phase-C weights; the released
checkpoint is the validation-best epoch, selected before any test evaluation.

**Re-benchmarking.** All results were re-measured with a new harness (fixed public
test splits, deterministic windows, identical inputs per method, failures logged, raw
rows committed under `honest_benchmarks/`), which reproduces the published AnyCam
Sintel numbers to the third decimal as a sanity anchor. VGGT, Pi3 and Depth Anything 3
were run through the same harness for comparison — including, to our knowledge, the
first calibration-accuracy measurements for those models.

The corrected picture in short: calibration is the strength (native wide-frame KITTI
3.99 % vs 15.6 % for the best billion-parameter model; beats AnyCalib on 95 % of
KITTI windows; multi-frame aggregation wins 100 % of KITTI sequences at N ≥ 2);
rotation improves moderately over AnyCam; translation direction improves on KITTI but
is worse than AnyCam on TUM-RGBD; large supervised models lead absolute pose accuracy.
