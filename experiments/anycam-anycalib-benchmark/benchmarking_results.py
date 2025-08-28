import argparse
import json
import os
import time
from collections import defaultdict, Counter
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np

# Optional plotting
try:
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except Exception:
    _HAS_MPL = False

import csv


def _compute_abs_and_pct_err(pred: float, gt: float) -> Tuple[float, float]:
    abs_err = abs(float(pred) - float(gt))
    pct_err = 100.0 * abs_err / float(gt) if gt != 0 else float("nan")
    return abs_err, pct_err


def _stats(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"mean": None, "median": None, "std": None, "rmse": None, "q25": None, "q75": None, "iqr": None, "min": None, "max": None}
    arr = np.array(values, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr, ddof=0)),
        "rmse": float(np.sqrt(np.mean(arr ** 2))),
        "q25": float(np.percentile(arr, 25)),
        "q75": float(np.percentile(arr, 75)),
        "iqr": float(np.percentile(arr, 75) - np.percentile(arr, 25)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def load_per_sequence_jsons(per_seq_dir: Path) -> List[Dict[str, Any]]:
    files = sorted([p for p in per_seq_dir.iterdir() if p.is_file() and p.suffix.lower() == ".json"]) 
    data: List[Dict[str, Any]] = []
    for fp in files:
        with open(fp, "r") as f:
            try:
                data.append(json.load(f))
            except Exception as e:
                print(f"[WARN] Failed to load {fp}: {e}")
    return data


def analyze_results(per_seq_dir: Path, out_dir: Path, make_plots: bool = True) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    per_seq_jsons = load_per_sequence_jsons(per_seq_dir)
    if not per_seq_jsons:
        raise FileNotFoundError(f"No per-sequence JSONs found in {per_seq_dir}")

    # Aggregations across all sequences and batches
    rows: List[Dict[str, Any]] = []

    # Per-sequence aggregations
    seq_to_abs_errs_cam: Dict[str, List[float]] = defaultdict(list)
    seq_to_abs_errs_calib: Dict[str, List[float]] = defaultdict(list)
    seq_to_pct_errs_cam: Dict[str, List[float]] = defaultdict(list)
    seq_to_pct_errs_calib: Dict[str, List[float]] = defaultdict(list)

    # Batch-level wins
    batch_wins = Counter()

    for seq_json in per_seq_jsons:
        seq_name = seq_json.get("sequence")
        batches = seq_json.get("batches", [])
        for b in batches:
            gt = float(b["gt_focal"])
            acam_pred = float(b["anycam_pred"])
            acalib_pred = float(b["anycalib_pred"])

            acam_abs, acam_pct = _compute_abs_and_pct_err(acam_pred, gt)
            acalib_abs, acalib_pct = _compute_abs_and_pct_err(acalib_pred, gt)

            rows.append({
                "sequence": seq_name,
                "batch": int(b.get("batch", 0)),
                "gt": gt,
                "anycam_pred": acam_pred,
                "anycalib_pred": acalib_pred,
                "anycam_abs": acam_abs,
                "anycalib_abs": acalib_abs,
                "anycam_pct": acam_pct,
                "anycalib_pct": acalib_pct,
            })

            seq_to_abs_errs_cam[seq_name].append(acam_abs)
            seq_to_abs_errs_calib[seq_name].append(acalib_abs)
            seq_to_pct_errs_cam[seq_name].append(acam_pct)
            seq_to_pct_errs_calib[seq_name].append(acalib_pct)

            if acam_abs < acalib_abs:
                batch_wins["anycam"] += 1
            elif acalib_abs < acam_abs:
                batch_wins["anycalib"] += 1
            else:
                batch_wins["tie"] += 1

    # Global metrics across all batches
    anycam_abs_all = [r["anycam_abs"] for r in rows]
    anycalib_abs_all = [r["anycalib_abs"] for r in rows]
    anycam_pct_all = [r["anycam_pct"] for r in rows]
    anycalib_pct_all = [r["anycalib_pct"] for r in rows]

    anycam_abs_stats = _stats(anycam_abs_all)
    anycalib_abs_stats = _stats(anycalib_abs_all)
    anycam_pct_stats = _stats(anycam_pct_all)
    anycalib_pct_stats = _stats(anycalib_pct_all)

    # Threshold consistency (within X% of GT)
    def within_threshold(pcts: List[float], thr: float) -> float:
        arr = np.array(pcts, dtype=float)
        return float(np.mean(arr < thr) * 100.0)

    thresholds = [5.0, 10.0, 20.0]
    thr_metrics = {
        "thresholds_pct": thresholds,
        "anycam_within_pct": [within_threshold(anycam_pct_all, t) for t in thresholds],
        "anycalib_within_pct": [within_threshold(anycalib_pct_all, t) for t in thresholds],
    }

    # Per-sequence metrics and winner per sequence
    per_seq_metrics: List[Dict[str, Any]] = []
    seq_wins = Counter()

    for seq_name in sorted(seq_to_abs_errs_cam.keys()):
        cam_abs = seq_to_abs_errs_cam[seq_name]
        calib_abs = seq_to_abs_errs_calib[seq_name]
        cam_pct = seq_to_pct_errs_cam[seq_name]
        calib_pct = seq_to_pct_errs_calib[seq_name]

        cam_abs_stats = _stats(cam_abs)
        calib_abs_stats = _stats(calib_abs)
        cam_pct_stats = _stats(cam_pct)
        calib_pct_stats = _stats(calib_pct)

        # Winner by lower mean absolute error in the sequence
        winner = "tie"
        if cam_abs_stats["mean"] is not None and calib_abs_stats["mean"] is not None:
            if cam_abs_stats["mean"] < calib_abs_stats["mean"]:
                winner = "anycam"
            elif calib_abs_stats["mean"] < cam_abs_stats["mean"]:
                winner = "anycalib"
        seq_wins[winner] += 1

        per_seq_metrics.append({
            "sequence": seq_name,
            "num_batches": len(cam_abs),
            "anycam_abs_mean": cam_abs_stats["mean"],
            "anycam_abs_median": cam_abs_stats["median"],
            "anycam_abs_std": cam_abs_stats["std"],
            "anycam_abs_rmse": cam_abs_stats["rmse"],
            "anycalib_abs_mean": calib_abs_stats["mean"],
            "anycalib_abs_median": calib_abs_stats["median"],
            "anycalib_abs_std": calib_abs_stats["std"],
            "anycalib_abs_rmse": calib_abs_stats["rmse"],
            "anycam_pct_mean": cam_pct_stats["mean"],
            "anycam_pct_median": cam_pct_stats["median"],
            "anycam_pct_std": cam_pct_stats["std"],
            "anycalib_pct_mean": calib_pct_stats["mean"],
            "anycalib_pct_median": calib_pct_stats["median"],
            "anycalib_pct_std": calib_pct_stats["std"],
            "winner": winner,
        })

    # Save per-sequence CSV
    per_seq_csv_path = out_dir / "per_sequence_metrics.csv"
    with open(per_seq_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_seq_metrics[0].keys()))
        writer.writeheader()
        for row in per_seq_metrics:
            writer.writerow(row)

    # Save aggregate rows CSV (per batch)
    per_batch_csv_path = out_dir / "per_batch_rows.csv"
    with open(per_batch_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    # Save summary JSON
    summary = {
        "num_sequences": int(len(per_seq_jsons)),
        "num_batches": int(len(rows)),
        "batch_wins": dict(batch_wins),
        "sequence_wins": dict(seq_wins),
        "anycam_abs_stats": anycam_abs_stats,
        "anycalib_abs_stats": anycalib_abs_stats,
        "anycam_pct_stats": anycam_pct_stats,
        "anycalib_pct_stats": anycalib_pct_stats,
        "threshold_consistency": thr_metrics,
        "better_model_by_mean_abs_error": (
            "anycam" if anycam_abs_stats["mean"] < anycalib_abs_stats["mean"] else (
                "anycalib" if anycalib_abs_stats["mean"] < anycam_abs_stats["mean"] else "tie"
            )
        ),
        "more_consistent_model_abs_std": (
            "anycam" if anycam_abs_stats["std"] < anycalib_abs_stats["std"] else (
                "anycalib" if anycalib_abs_stats["std"] < anycam_abs_stats["std"] else "tie"
            )
        )
    }

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Plots
    if _HAS_MPL:
        # Histogram of abs errors
        plt.figure(figsize=(8, 5))
        bins = 40
        plt.hist(anycam_abs_all, bins=bins, alpha=0.6, label="AnyCam abs error")
        plt.hist(anycalib_abs_all, bins=bins, alpha=0.6, label="AnyCalib abs error")
        plt.xlabel("Absolute focal error")
        plt.ylabel("Count")
        plt.title("Absolute focal error distribution")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "abs_error_hist.png")
        plt.close()

        # Boxplot of abs errors
        plt.figure(figsize=(6, 5))
        plt.boxplot([anycam_abs_all, anycalib_abs_all], labels=["AnyCam", "AnyCalib"], showfliers=False)
        plt.ylabel("Absolute focal error")
        plt.title("Absolute focal error (boxplot)")
        plt.tight_layout()
        plt.savefig(out_dir / "abs_error_boxplot.png")
        plt.close()

        # CDF of abs errors
        def _cdf(data: List[float]) -> Tuple[np.ndarray, np.ndarray]:
            x = np.sort(np.array(data, dtype=float))
            y = np.linspace(0, 1, len(x), endpoint=False)
            return x, y
        x1, y1 = _cdf(anycam_abs_all)
        x2, y2 = _cdf(anycalib_abs_all)
        plt.figure(figsize=(6, 5))
        plt.plot(x1, y1, label="AnyCam")
        plt.plot(x2, y2, label="AnyCalib")
        plt.xlabel("Absolute focal error")
        plt.ylabel("Empirical CDF")
        plt.title("Absolute error CDF")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / "abs_error_cdf.png")
        plt.close()

        # Per-sequence scatter: mean abs error
        seq_cam_means = [r["anycam_abs_mean"] for r in per_seq_metrics]
        seq_calib_means = [r["anycalib_abs_mean"] for r in per_seq_metrics]
        lo = float(min(seq_cam_means + seq_calib_means))
        hi = float(max(seq_cam_means + seq_calib_means))
        plt.figure(figsize=(6, 6))
        plt.scatter(seq_cam_means, seq_calib_means, s=20, alpha=0.7)
        plt.plot([lo, hi], [lo, hi], "k--", linewidth=1, label="y=x")
        plt.xlabel("AnyCam mean abs error (per sequence)")
        plt.ylabel("AnyCalib mean abs error (per sequence)")
        plt.title("Per-sequence comparison (lower is better)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "per_sequence_mean_abs_scatter.png")
        plt.close()

        # Wins bar chart
        labels = ["AnyCam", "AnyCalib", "Tie"]
        values = [batch_wins.get("anycam", 0), batch_wins.get("anycalib", 0), batch_wins.get("tie", 0)]
        plt.figure(figsize=(6, 5))
        plt.bar(labels, values, color=["C0", "C1", "C2"]) 
        plt.ylabel("Batch wins count")
        plt.title("Batch-level wins")
        plt.tight_layout()
        plt.savefig(out_dir / "batch_wins_bar.png")
        plt.close()

    print(f"Analysis complete. Outputs saved to {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Analyze AnyCam vs AnyCalib benchmarking results.")
    parser.add_argument("--benchmark_dir", type=str, required=True, help="Path to benchmark directory containing per_sequence/")
    parser.add_argument("--per_sequence_subdir", type=str, default="per_sequence", help="Subdirectory name for per-sequence JSONs")
    parser.add_argument("--out_dir", type=str, default=None, help="Directory to write analysis outputs (defaults under benchmark dir)")
    parser.add_argument("--no_plots", action="store_true", help="Disable plot generation")
    args = parser.parse_args()

    bench_dir = Path(args.benchmark_dir)
    per_seq_dir = bench_dir / args.per_sequence_subdir
    if not per_seq_dir.exists():
        raise FileNotFoundError(f"per_sequence directory not found at {per_seq_dir}")

    ts = int(time.time())
    out_dir = Path(args.out_dir) if args.out_dir else (bench_dir / f"analysis_{ts}")

    analyze_results(per_seq_dir, out_dir, make_plots=(not args.no_plots))


if __name__ == "__main__":
    main()
