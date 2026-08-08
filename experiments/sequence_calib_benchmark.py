"""Sequence-level calibration: FAT aggregation vs per-frame AnyCalib averaging,
as a function of the number of frames N sampled evenly across the WHOLE sequence.

This isolates the multi-frame-aggregation claim: within short windows frames are
nearly identical (2-3% focal spread) so any method ties; across a sequence the
per-frame spread is 7-10%, so aggregation quality can actually matter.

Both methods see IDENTICAL frames. Calibration-only (no pose/depth/flow).

Usage:
  python experiments/sequence_calib_benchmark.py --run_name seqcalib_square336 \
      --datasets sintel,tumrgbd,kitti --n_frames 1,2,4,8,16
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from experiments.honest_benchmark import build_dataset, image_size_for  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_name", required=True)
    ap.add_argument("--datasets", default="sintel,tumrgbd,kitti")
    ap.add_argument("--n_frames", default="1,2,4,8,16")
    ap.add_argument("--ours_ckpt", default=str(REPO / "thesis_results/checkpoints/phase_Cb_v6_h100_epoch_0002.pt"))
    ap.add_argument("--image_mode", default="square336")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    out_dir = REPO / "honest_benchmarks" / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_f = open(out_dir / "rows.jsonl", "a")

    device = args.device
    ns = [int(x) for x in args.n_frames.split(",")]

    # --- models (calibration-only) ---
    from experiments.models.anycalib_with_fat import AnyCalibWithMCT
    from anycalib.model.anycalib_pretrained import AnyCalib

    fat_cfg = {"embed_dim": 1024, "num_heads": 8, "num_layers": 2, "dropout": 0.1,
               "use_visual_conditioning": False, "num_scales": 4}
    fat = AnyCalibWithMCT(model_id="anycalib_pinhole", use_fat=True, fat_config=fat_cfg,
                          use_dinov2_small=False, use_dinov2_full=False,
                          freeze_backbone=True, freeze_decoder=True, freeze_calibrator=True,
                          input_normalization=True).to(device).eval()
    ck = torch.load(args.ours_ckpt, map_location=device, weights_only=False)
    sd = {k[len("fat_model."):]: v for k, v in ck["model_state_dict"].items() if k.startswith("fat_model.")}
    missing, unexpected = fat.load_state_dict(sd, strict=False)
    print(f"[fat] loaded, missing={len(missing)} unexpected={len(unexpected)}")

    anycalib = AnyCalib(model_id="anycalib_pinhole").to(device)

    for ds_name in args.datasets.split(","):
        image_size = image_size_for(ds_name, args.image_mode)
        # frame_count=2 keeps loaders happy; we only use the FIRST frame of each datapoint
        ds, per_seq = build_dataset(ds_name, image_size, frame_count=2)
        print(f"[{ds_name}] {len(per_seq)} seqs")

        for seq in sorted(per_seq):
            idxs = per_seq[seq]
            n_max = max(ns)
            pos = np.linspace(0, len(idxs) - 1, n_max).round().astype(int)
            grid = sorted(set(pos.tolist()))
            frame_cache = {}
            gt_intr = None
            for p in grid:
                s = ds[idxs[p]]
                frame_cache[p] = torch.from_numpy(s["imgs"][0]).float()
                if gt_intr is None:
                    pr = np.asarray(s["projs"][0], dtype=np.float64)
                    gt_intr = [pr[0, 0], pr[1, 1], pr[0, 2], pr[1, 2]]

            for n in ns:
                gi = np.linspace(0, len(grid) - 1, n).round().astype(int)
                sel = [grid[g] for g in gi]
                frames = torch.stack([frame_cache[p] for p in sel]).to(device)  # [n,3,H,W]

                row = {"dataset": ds_name, "seq": seq, "n_frames": n,
                       "gt_fx": float(gt_intr[0]), "gt_fy": float(gt_intr[1])}
                try:
                    with torch.no_grad():
                        out = fat(frames, cam_id="pinhole")
                        intr = out["intrinsics"][0]
                        intr = np.asarray(intr.detach().float().cpu()).ravel()
                    row["fat_fx"], row["fat_fy"] = float(intr[0]), float(intr[1])
                except Exception as e:
                    row["fat_error"] = str(e)[:200]
                try:
                    pfs = []
                    with torch.no_grad():
                        for i in range(frames.shape[0]):
                            pred = anycalib.predict(frames[i], cam_id="pinhole")
                            intr = pred["intrinsics"]
                            if isinstance(intr, (list, tuple)):
                                intr = intr[0]
                            pfs.append(np.asarray(intr.detach().float().cpu()).ravel()[:4])
                    pfs = np.stack(pfs)
                    row["anycalib_avg_fx"] = float(pfs[:, 0].mean())
                    row["anycalib_avg_fy"] = float(pfs[:, 1].mean())
                    row["anycalib_med_fx"] = float(np.median(pfs[:, 0]))
                    row["anycalib_med_fy"] = float(np.median(pfs[:, 1]))
                except Exception as e:
                    row["anycalib_error"] = str(e)[:200]

                rows_f.write(json.dumps(row) + "\n")
                rows_f.flush()
            print(f"  {seq} done")

    rows_f.close()
    print("[done]", out_dir / "rows.jsonl")


if __name__ == "__main__":
    main()
