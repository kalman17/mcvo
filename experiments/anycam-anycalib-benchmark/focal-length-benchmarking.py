import argparse
import json
import os
import time
from pathlib import Path
from typing import List, Tuple, Dict, Any

import cv2
import numpy as np
import torch

# Matplotlib is optional; gracefully degrade if not available
try:
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except Exception:
    _HAS_MPL = False

# AnyCalib import
from anycalib import AnyCalib


def extract_frames(video_path: str, num_frames: int = 20) -> List[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file {video_path}")
    frames: List[np.ndarray] = []
    while len(frames) < num_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    return frames


def run_anycalib_on_frames(frames: List[np.ndarray], model: AnyCalib, device: torch.device, cam_id: str) -> List[float]:
    if len(frames) == 0:
        return []
    frame_tensors: List[torch.Tensor] = []
    for frame in frames:
        t = torch.tensor(frame, dtype=torch.float32, device=device).permute(2, 0, 1) / 255.0
        frame_tensors.append(t)
    batch_images = torch.stack(frame_tensors, dim=0)
    output = model.predict(batch_images, cam_id=cam_id)
    intrinsics_list = output["intrinsics"]

    focals: List[float] = []
    for intr in intrinsics_list:
        if cam_id.startswith("simple_"):
            focals.append(float(intr[0].item()))
        else:
            fx = float(intr[0].item())
            fy = float(intr[1].item())
            focals.append(0.5 * (fx + fy))
    return focals


def aggregate_in_pairs(values: List[float]) -> List[float]:
    agg: List[float] = []
    for i in range(0, len(values), 2):
        pair = values[i:i+2]
        if len(pair) == 2:
            agg.append(float((pair[0] + pair[1]) / 2.0))
        else:
            # If odd count, keep as single (should not happen with 20 frames)
            agg.append(float(pair[0]))
    return agg


def load_anycam_results(results_root: str, sequence_name: str) -> Tuple[List[float], float]:
    seq_dir = os.path.join(results_root, sequence_name)
    results_file = os.path.join(seq_dir, "results.json")
    if not os.path.exists(results_file):
        raise FileNotFoundError(f"AnyCam results not found for sequence '{sequence_name}' at {results_file}")
    with open(results_file, "r") as f:
        data = json.load(f)
    predicted_focals = data.get("predicted_focals", [])
    gt_focal = float(data.get("gt_focal"))
    return predicted_focals, gt_focal


def safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def compute_errors(pred: float, gt: float) -> Tuple[float, float]:
    abs_err = abs(pred - gt)
    pct_err = 100.0 * abs_err / gt if gt != 0 else float("nan")
    return abs_err, pct_err


def main():
    parser = argparse.ArgumentParser(description="Benchmark AnyCalib vs AnyCam focal predictions vs GT over videos.")
    parser.add_argument("--videos_dir", type=str, default="/home/kalman/TUM/thesis/Objectron/videos/", help="Directory with input videos (.MOV)")
    parser.add_argument("--anycam_results_dir", type=str, default="/home/kalman/TUM/thesis/anycam/experiments/focal-length-consistency/results/focal_consistency_rawframes_1755781992/", help="Directory with AnyCam per-sequence results (folders with results.json)")
    parser.add_argument("--out_dir", type=str, default=None, help="Output directory for benchmark results")
    parser.add_argument("--model_id", type=str, default="anycalib_pinhole", choices=["anycalib_pinhole", "anycalib_gen", "anycalib_dist", "anycalib_edit"], help="AnyCalib model ID")
    parser.add_argument("--cam_id", type=str, default="simple_pinhole", help="Camera model ID for AnyCalib (e.g., simple_pinhole, pinhole)")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device for AnyCalib")
    parser.add_argument("--num_frames", type=int, default=20, help="Number of frames per video to use (should be 20)")
    args = parser.parse_args()

    timestamp = int(time.time())
    out_dir = args.out_dir or os.path.join("experiments", "anycam-anycalib-benchmark", "results", f"benchmark_{timestamp}")
    safe_mkdir(out_dir)

    # Prepare device and model
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = AnyCalib(model_id=args.model_id).to(device)
    print(f"Initialized AnyCalib with model_id: {args.model_id}, cam_id: {args.cam_id}")

    videos_dir = Path(args.videos_dir)
    if not videos_dir.exists():
        raise FileNotFoundError(f"Videos directory not found: {videos_dir}")

    # Collect videos (MOV)
    video_files = sorted([p for p in videos_dir.iterdir() if p.is_file() and p.suffix.lower() == ".mov"])
    print(f"Found {len(video_files)} videos in {videos_dir}")

    # Aggregated results across all sequences and batches
    aggregate_rows: List[Dict[str, Any]] = []

    # Per-sequence results directory
    per_seq_dir = os.path.join(out_dir, "per_sequence")
    safe_mkdir(per_seq_dir)

    num_processed = 0
    for vid_path in video_files:
        seq_name = vid_path.stem
        print(f"Processing sequence: {seq_name}")

        # Load AnyCam results and GT
        try:
            anycam_batch_focals, gt_focal = load_anycam_results(args.anycam_results_dir, seq_name)
        except FileNotFoundError as e:
            print(f"[WARN] {e}. Skipping sequence.")
            continue

        # Extract frames and run AnyCalib
        frames = extract_frames(str(vid_path), num_frames=args.num_frames)
        if len(frames) < 2:
            print(f"[WARN] Sequence {seq_name} has fewer than 2 frames. Skipping.")
            continue
        anycalib_frame_focals = run_anycalib_on_frames(frames, model, device, args.cam_id)
        anycalib_batch_focals = aggregate_in_pairs(anycalib_frame_focals)

        # Align batch counts (expect 10)
        n_batches = min(len(anycam_batch_focals), len(anycalib_batch_focals))
        if n_batches == 0:
            print(f"[WARN] No batches for {seq_name}. Skipping.")
            continue

        # Per-sequence summary
        seq_summary: Dict[str, Any] = {
            "sequence": seq_name,
            "gt_focal": gt_focal,
            "anycam_predicted_focals": anycam_batch_focals[:n_batches],
            "anycalib_predicted_focals": anycalib_batch_focals[:n_batches],
            "batches": []
        }

        for b in range(n_batches):
            acam_pred = float(anycam_batch_focals[b])
            acalib_pred = float(anycalib_batch_focals[b])
            acam_abs, acam_pct = compute_errors(acam_pred, gt_focal)
            acalib_abs, acalib_pct = compute_errors(acalib_pred, gt_focal)

            batch_row = {
                "sequence": seq_name,
                "batch": b + 1,
                "gt_focal": gt_focal,
                "anycam_pred": acam_pred,
                "anycalib_pred": acalib_pred,
                "anycam_abs_err": acam_abs,
                "anycalib_abs_err": acalib_abs,
                "anycam_pct_err": acam_pct,
                "anycalib_pct_err": acalib_pct,
            }
            seq_summary["batches"].append(batch_row)
            aggregate_rows.append(batch_row)

        # Save per-sequence JSON
        with open(os.path.join(per_seq_dir, f"{seq_name}.json"), "w") as f:
            json.dump(seq_summary, f, indent=2)

        num_processed += 1

    # Save aggregate CSV/JSON
    agg_json_path = os.path.join(out_dir, "aggregate.json")
    with open(agg_json_path, "w") as f:
        json.dump(aggregate_rows, f, indent=2)

    # Compute and save summary metrics
    def _safe_vals(key: str) -> List[float]:
        vals: List[float] = []
        for r in aggregate_rows:
            v = r.get(key)
            if v is not None and not (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
                vals.append(float(v))
        return vals

    anycam_abs = _safe_vals("anycam_abs_err")
    anycalib_abs = _safe_vals("anycalib_abs_err")
    anycam_pct = _safe_vals("anycam_pct_err")
    anycalib_pct = _safe_vals("anycalib_pct_err")

    summary = {
        "num_sequences_processed": num_processed,
        "num_batches_total": len(aggregate_rows),
        "anycam_abs_err_mean": float(np.mean(anycam_abs)) if anycam_abs else None,
        "anycam_abs_err_median": float(np.median(anycam_abs)) if anycam_abs else None,
        "anycalib_abs_err_mean": float(np.mean(anycalib_abs)) if anycalib_abs else None,
        "anycalib_abs_err_median": float(np.median(anycalib_abs)) if anycalib_abs else None,
        "anycam_pct_err_mean": float(np.mean(anycam_pct)) if anycam_pct else None,
        "anycam_pct_err_median": float(np.median(anycam_pct)) if anycam_pct else None,
        "anycalib_pct_err_mean": float(np.mean(anycalib_pct)) if anycalib_pct else None,
        "anycalib_pct_err_median": float(np.median(anycalib_pct)) if anycalib_pct else None,
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # Plots (optional)
    if _HAS_MPL and aggregate_rows:
        # 1) Histogram of absolute errors
        plt.figure(figsize=(8, 5))
        bins = 30
        if anycam_abs:
            plt.hist(anycam_abs, bins=bins, alpha=0.6, label="AnyCam abs err")
        if anycalib_abs:
            plt.hist(anycalib_abs, bins=bins, alpha=0.6, label="AnyCalib abs err")
        plt.xlabel("Absolute focal error")
        plt.ylabel("Count")
        plt.legend()
        plt.title("Absolute focal error distribution")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "abs_error_hist.png"))
        plt.close()

        # 2) Scatter predicted vs GT
        gt_vals = [r["gt_focal"] for r in aggregate_rows]
        acam_preds = [r["anycam_pred"] for r in aggregate_rows]
        acalib_preds = [r["anycalib_pred"] for r in aggregate_rows]
        lo = float(min(gt_vals + acam_preds + acalib_preds))
        hi = float(max(gt_vals + acam_preds + acalib_preds))

        plt.figure(figsize=(6, 6))
        plt.scatter(gt_vals, acam_preds, s=10, alpha=0.6, label="AnyCam")
        plt.scatter(gt_vals, acalib_preds, s=10, alpha=0.6, label="AnyCalib")
        plt.plot([lo, hi], [lo, hi], "k--", linewidth=1, label="y=x")
        plt.xlabel("Ground truth focal")
        plt.ylabel("Predicted focal")
        plt.legend()
        plt.title("Predicted vs Ground Truth (all batches)")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "pred_vs_gt_scatter.png"))
        plt.close()

    print(f"Done. Processed {num_processed} sequences. Results saved to: {out_dir}")


if __name__ == "__main__":
    main()
