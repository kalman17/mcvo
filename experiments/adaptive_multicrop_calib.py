"""Adaptive multi-crop MCT calibration on native frames (KITTI wide + Sintel).

Reproducible version of the E3a/E3b experiments: 3 square crops across the width,
MCT aggregation over N frames per crop, fuse crop focals by median when they agree
(dispersion < tau), else fall back to the single-field prediction. Reports per-regime
and gated results. Works locally or on the cluster (expects REPO/data/eval symlinks).

Usage: PYTHONPATH=$REPO python experiments/adaptive_multicrop_calib.py \
           --ckpt <merged_ckpt.pt> --out honest_benchmarks/<name>.json
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from experiments.models.anycalib_with_fat import AnyCalibWithMCT  # noqa

SIZE = 336
N_FRAMES = 8
CROPS = 3
SINTEL_SEQS = ("alley_2 ambush_4 ambush_5 ambush_6 cave_2 cave_4 market_2 market_5 "
               "market_6 shaman_3 sleeping_1 sleeping_2 temple_2 temple_3").split()


def load_model(ckpt, device):
    fat_cfg = {"embed_dim": 1024, "num_heads": 8, "num_layers": 2, "dropout": 0.1,
               "use_visual_conditioning": False, "num_scales": 4}
    m = AnyCalibWithMCT(model_id="anycalib_pinhole", use_fat=True, fat_config=fat_cfg,
                        use_dinov2_small=False, use_dinov2_full=False,
                        freeze_backbone=True, freeze_decoder=True,
                        freeze_calibrator=True, input_normalization=True).to(device).eval()
    ck = torch.load(ckpt, map_location=device, weights_only=False)
    sd = {k[len("fat_model."):]: v for k, v in ck["model_state_dict"].items()
          if k.startswith("fat_model.")}
    missing, unexpected = m.load_state_dict(sd, strict=False)
    print(f"[model] loaded {ckpt} (missing {len(missing)}, unexpected {len(unexpected)})")
    return m


def frames_resized(paths, idxs):
    frames = []
    for i in idxs:
        im = cv2.cvtColor(cv2.imread(str(paths[i])), cv2.COLOR_BGR2RGB)
        H, W = im.shape[:2]
        scale = SIZE / H
        im = cv2.resize(im, (int(round(W * scale)), SIZE), interpolation=cv2.INTER_AREA)
        frames.append(im.astype(np.float32) / 255.0)
    return frames, scale


def predict_seq(m, frames, scale, device):
    """(multicrop_fx_native, crop_dispersion, singlefield_fx_native, per_crop)"""
    Wr = frames[0].shape[1]
    xs = np.linspace(0, max(Wr - SIZE, 0), CROPS).round().astype(int)
    fxs = []
    for x0 in xs:
        clip = torch.stack([torch.from_numpy(f[:, x0:x0 + SIZE].transpose(2, 0, 1))
                            for f in frames]).to(device)
        with torch.no_grad():
            out = m(clip, cam_id="pinhole")
        fxs.append(float(np.asarray(out["intrinsics"][0].detach().float().cpu()).ravel()[0]) / scale)
    disp = (max(fxs) - min(fxs)) / np.median(fxs)
    clip_full = torch.stack([torch.from_numpy(f.transpose(2, 0, 1)) for f in frames]).to(device)
    with torch.no_grad():
        out = m(clip_full, cam_id="pinhole")
    sf = float(np.asarray(out["intrinsics"][0].detach().float().cpu()).ravel()[0]) / scale
    return float(np.median(fxs)), float(disp), sf, [round(f, 1) for f in fxs]


def kitti_gt(seq):
    for line in open(REPO / "data/eval/kitti_odom/sequences" / seq / "calib.txt"):
        if line.startswith("P2:"):
            v = [float(x) for x in line.split()[1:]]
            return v[0]
    raise ValueError(seq)


def sintel_gt(seq):
    p = REPO / "data/eval/sintel/training/camdata_left" / seq / "frame_0001.cam"
    with open(p, "rb") as f:
        np.fromfile(f, dtype=np.float32, count=1)
        M = np.fromfile(f, dtype="float64", count=9).reshape(3, 3)
    return M[0, 0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tau", type=float, default=0.2)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    m = load_model(args.ckpt, args.device)
    rows = []

    for seq in [f"{i:02d}" for i in range(11)]:
        d = REPO / "data/eval/kitti_odom/sequences" / seq / "image_2"
        files = sorted(d.glob("*.jpg")) or sorted(d.glob("*.png"))
        if not files:
            continue
        idxs = np.linspace(0, len(files) - 1, N_FRAMES).round().astype(int)
        frames, scale = frames_resized(files, idxs)
        mc, disp, sf, per = predict_seq(m, frames, scale, args.device)
        rows.append({"ds": "kitti", "seq": seq, "gt": kitti_gt(seq),
                     "mc": mc, "disp": disp, "sf": sf, "per_crop": per})

    for seq in SINTEL_SEQS:
        files = sorted((REPO / "data/eval/sintel/training/final" / seq).glob("*.png"))
        idxs = np.linspace(0, len(files) - 1, N_FRAMES).round().astype(int)
        frames, scale = frames_resized(files, idxs)
        mc, disp, sf, per = predict_seq(m, frames, scale, args.device)
        rows.append({"ds": "sintel", "seq": seq, "gt": sintel_gt(seq),
                     "mc": mc, "disp": disp, "sf": sf, "per_crop": per})

    out = {"tau": args.tau, "ckpt": args.ckpt, "rows": rows}
    for name, pick in [("multicrop", lambda r: r["mc"]),
                       ("singlefield", lambda r: r["sf"]),
                       ("adaptive", lambda r: r["mc"] if r["disp"] < args.tau else r["sf"])]:
        for ds in ["kitti", "sintel"]:
            errs = [abs(pick(r) - r["gt"]) / r["gt"] * 100 for r in rows if r["ds"] == ds]
            out[f"{name}_{ds}_median"] = float(np.median(errs))
            print(f"{name:12s} {ds:7s}: fx-APE median {np.median(errs):6.2f}%  (n={len(errs)})")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1)
    print("->", args.out)


if __name__ == "__main__":
    main()
