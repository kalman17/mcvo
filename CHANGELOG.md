# Changelog

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
