"""Calibration-benchmark extensions to the honest harness (paper: wide-frame and
motion-observability calibration). Everything here is additive: new dataset loaders,
motion classification from GT poses, FOV metrics, extra model adapters, and the
own-preprocessing ("native") protocol. Nothing in experiments/honest_benchmark.py's
existing behaviour changes.
"""
