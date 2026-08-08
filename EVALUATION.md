# Honest Evaluation Protocol

Goal: measure our model (FAT multi-frame calibration + retrained pose head) against
vanilla AnyCam (pose + candidate focal) and per-frame AnyCalib (calibration) with a
protocol that a reviewer would accept, replacing the cherry-picked thesis numbers.

## Principles

1. **Standard test splits, all sequences, no filtering.**
   - Sintel: `particlesfm` split (14 sequences, the MonST3R/ParticleSfM protocol used by AnyCam).
   - TUM-RGBD: `monst3r` split — the 8 freiburg3 sitting/walking dynamic sequences.
   - KITTI odometry: sequences 00–10 (all with GT poses + calib).
2. **Deterministic window selection.** For window-level metrics: K evenly-spaced windows
   per sequence (per-sequence balanced — no global random sampling that over-weights long
   sequences). Window = `frame_count` frames at dataset-specific dilation
   (sintel 1, tumrgbd 10, kitti 1 — the 'anycam' preset used in prior benchmarks).
3. **Same windows for every method; failures recorded, never skipped.** Every
   (window, model) produces a row: metrics or an error string. Aggregates report
   failure counts. No try/except-continue.
4. **Intact baselines.**
   - Vanilla AnyCam loaded strictly into the unmodified architecture (focal_embed_dim=0);
     any shape mismatch is a hard error (fixed in `anycam/scripts/common.py`).
   - AnyCam predicts its own focal (config `use_provided_proj: false` verified); GT
     intrinsics are never given to any method.
   - AnyCalib run through its official `predict()` (own preprocessing).
5. **Checkpoint selection rule:** fixed BEFORE looking at test results — the checkpoint
   with best *training-validation* loss (recorded in cluster metrics.csv). For the v6
   model that is epoch 2 (val=-0.510). No per-dataset checkpoint switching. Figures and
   tables must come from the SAME checkpoint.
6. **Metrics.**
   - Pose (per consecutive pair): geodesic rotation error (deg); translation direction
     angle (deg) recorded together with ‖t_gt‖ so near-static pairs (direction undefined)
     can be handled by a *pre-declared* rule: pairs with ‖t_gt‖ < 1e-4 (unitless Sintel/
     KITTI-scaled) are excluded from direction aggregation for ALL methods equally, and
     their count is reported.
   - Calibration (per window): fx/fy absolute percentage error vs GT of the processed
     image; plus per-sequence *consistency* = std(fx_pred)/mean(fx_pred) over the
     sequence's windows (multi-frame aggregation should reduce this).
   - Aggregation: median and mean with bootstrap 95% CIs (1000 resamples, seed 0), plus
     paired per-window win-rates vs the relevant baseline.
7. **Two image regimes** (calibration is resolution/FOV-sensitive):
   - `square336`: the training regime of our model (short side → 336, center square crop).
     Note this crops KITTI to a ~29° FOV — far outside typical calibration training FOVs.
   - `aspect336`: short side → 336, full aspect preserved (no crop). Closer to the
     AnyCalib paper protocol.
   Results are reported for both, for all methods, with the regime stated.
8. **Full-sequence trajectory metrics** (ATE/RPE after Sim(3) alignment, AnyCam paper
   protocol via `anycam/scripts/evaluate_trajectories.py`) are the pose headline; window
   metrics are secondary/diagnostic. (Stage 2 — pending.)

## Provenance

Every result JSON embeds: git commit, checkpoint path + selection rule, dataset split
files, window list hash, and the exact model configs. Runs live under `honest_benchmarks/`.
