"""
Honest window-level benchmark: ours (FAT) vs vanilla AnyCam vs per-frame AnyCalib.

See HONEST_EVAL_PROTOCOL.md. Key properties:
- standard test splits, per-sequence balanced deterministic windows
- same windows for every model; failures recorded as rows, never skipped
- baselines intact (strict loading; no GT intrinsics to any method)
- per-window JSONL output with full provenance; resumable

Usage:
  python experiments/honest_benchmark.py \
      --run_name v6_square336 \
      --datasets sintel,tumrgbd,kitti \
      --models ours,anycam,anycalib \
      --ours_ckpt thesis_results/checkpoints/phase_Cb_v6_h100_epoch_0002.pt \
      --image_mode square336 --windows_per_seq 16
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from experiments.pose_metrics import rotation_error_degrees, translation_direction_error_degrees  # noqa: E402

SINTEL_TEST_SEQS = [
    "alley_2", "ambush_4", "ambush_5", "ambush_6", "cave_2", "cave_4", "market_2",
    "market_5", "market_6", "shaman_3", "sleeping_1", "sleeping_2", "temple_2", "temple_3",
]
TUM_TEST_SEQS = [
    "rgbd_dataset_freiburg3_sitting_halfsphere", "rgbd_dataset_freiburg3_sitting_rpy",
    "rgbd_dataset_freiburg3_sitting_static", "rgbd_dataset_freiburg3_sitting_xyz",
    "rgbd_dataset_freiburg3_walking_halfsphere", "rgbd_dataset_freiburg3_walking_rpy",
    "rgbd_dataset_freiburg3_walking_static", "rgbd_dataset_freiburg3_walking_xyz",
]
KITTI_SEQS = [f"{i:02d}" for i in range(11)]

# 'anycam' dilation preset used in all prior benchmarks
DILATION = {"sintel": 1, "tumrgbd": 10, "kitti": 1}
NATIVE_SIZE = {"sintel": (436, 1024), "tumrgbd": (480, 640), "kitti": (370, 1226)}


def _mult14(x):
    return int(round(x / 14) * 14)


def image_size_for(dataset: str, mode: str):
    if mode == "square336":
        return 336
    if mode == "aspect336":
        h, w = NATIVE_SIZE[dataset]
        return (336, _mult14(336 * w / h))
    raise ValueError(mode)


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

def build_dataset(name: str, image_size, frame_count: int):
    from anycam.datasets.common import flow_selector_seq

    dil = DILATION[name]
    if name == "sintel":
        from anycam.datasets.sintel.sintel_dataset import SintelDataset
        ds = SintelDataset(
            data_path=str(REPO / "data/eval/sintel/training"), split_path=None,
            image_size=image_size, frame_count=frame_count, dilation=dil,
            return_depth=False, return_flow=False, flow_selector=flow_selector_seq,
        )
        seq_lens = dict(ds._sequences)
        allowed = SINTEL_TEST_SEQS
    elif name == "tumrgbd":
        from anycam.datasets.tum_rgbd.tumrgbd_dataset import TUMRGBDDataset
        ds = TUMRGBDDataset(
            data_path=str(REPO / "data/eval/tum_rgbd"), split_path=None,
            image_size=image_size, frame_count=frame_count, dilation=dil,
            return_depth=False, return_flow=False, flow_selector=flow_selector_seq,
        )
        seq_lens = {s: (l if isinstance(l, int) else len(l)) for s, l in ds._sequences.items()}
        allowed = TUM_TEST_SEQS
    elif name == "kitti":
        from experiments.kitti_dataset import KITTIOdometryDataset
        ds = KITTIOdometryDataset(
            data_path=str(REPO / "data/eval/kitti_odom"),
            image_size=image_size, frame_count=frame_count, dilation=dil,
        )
        seq_lens = {s: len(ds._poses[s]) for s in ds._calibrations}
        allowed = KITTI_SEQS
    else:
        raise ValueError(name)

    span = (frame_count - 1) * dil
    per_seq = {}
    for idx, (seq, start) in enumerate(ds._datapoints):
        if seq not in allowed:
            continue
        if seq in seq_lens and start + span >= seq_lens[seq]:
            continue
        per_seq.setdefault(seq, []).append(idx)
    return ds, per_seq


def select_windows(per_seq: dict, k: int):
    """Evenly-spaced deterministic selection of up to k windows per sequence."""
    sel = {}
    for seq in sorted(per_seq):
        idxs = per_seq[seq]
        if len(idxs) <= k:
            sel[seq] = list(idxs)
        else:
            pos = np.linspace(0, len(idxs) - 1, k).round().astype(int)
            sel[seq] = [idxs[p] for p in sorted(set(pos.tolist()))]
    return sel


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class OursModel:
    name_prefix = "ours"

    def __init__(self, ckpt: str, device: str, input_normalization: bool = True):
        from experiments.benchmark_phase_c_checkpoints import (
            create_inference_model, load_phase_c_checkpoint,
        )
        cfg = str(REPO / "pretrained_models/anycam_seq8/training_config.yaml")
        self.model = create_inference_model(cfg, device, input_normalization=input_normalization)
        load_phase_c_checkpoint(self.model, ckpt, device)
        self.model.eval()
        self.device = device

    def __call__(self, sample):
        from experiments.benchmark_phase_c_checkpoints import _run_model_forward
        imgs = torch.from_numpy(sample["imgs"]).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            out = _run_model_forward(self.model, {"imgs": imgs}, is_fat_model=True)
        return {
            "pred_poses": out["pred_poses"],
            "intr": out["model_intrinsics"],
        }


class AnyCamBaseline:
    name_prefix = "anycam"

    def __init__(self, device: str):
        from omegaconf import OmegaConf
        from anycam.scripts.common import load_model
        cfg = OmegaConf.load(str(REPO / "pretrained_models/anycam_seq8/training_config.yaml"))
        cfg["model"]["use_provided_flow"] = False
        cfg["model"]["train_directions"] = "forward"
        assert cfg["model"].get("use_provided_proj", False) is False
        self.model = load_model(cfg, str(REPO / "pretrained_models/anycam_seq8/training_checkpoint_247500.pt"))
        self.model = self.model.to(device).eval()
        self.device = device

    def __call__(self, sample):
        from experiments.benchmark_phase_c_checkpoints import _run_model_forward
        imgs = torch.from_numpy(sample["imgs"]).float().unsqueeze(0).to(self.device)
        # AnyCamWrapper.forward requires data['projs'] but (use_provided_proj=False)
        # only reads it for loss bookkeeping — predictions never see it. We feed a
        # FIXED dummy K (focal=W px) to guarantee no GT leakage.
        h, w = imgs.shape[-2], imgs.shape[-1]
        K = torch.tensor(
            [[float(w), 0.0, w / 2.0], [0.0, float(w), h / 2.0], [0.0, 0.0, 1.0]],
            device=self.device,
        ).view(1, 1, 3, 3).expand(1, imgs.shape[1], 3, 3).contiguous()
        with torch.no_grad():
            out = _run_model_forward(self.model, {"imgs": imgs, "projs": K}, is_fat_model=False)
        return {
            "pred_poses": out["pred_poses"],
            "intr": out["model_intrinsics"],  # its own selected focal candidate
        }


class AnyCalibPerFrame:
    name_prefix = "anycalib"

    def __init__(self, device: str):
        from anycalib.model.anycalib_pretrained import AnyCalib
        self.model = AnyCalib(model_id="anycalib_pinhole").to(device)
        self.device = device

    def __call__(self, sample):
        imgs = torch.from_numpy(sample["imgs"]).float().to(self.device)  # [N,3,H,W]
        intrs = []
        for i in range(imgs.shape[0]):
            pred = self.model.predict(imgs[i], cam_id="pinhole")
            intr = pred["intrinsics"]  # Tensor [4] for non-batched input
            if isinstance(intr, (list, tuple)):
                intr = intr[0]
            intrs.append(np.asarray(intr.detach().float().cpu()).ravel()[:4])
        intrs = np.stack(intrs)  # [N,4] fx fy cx cy
        return {
            "pred_poses": None,
            "intr": intrs.mean(axis=0),
            "per_frame_intr": intrs.tolist(),
        }




class VGGTModel:
    """VGGT-1B (GT-supervised, Meta). Absolute c2w poses + intrinsics from images."""
    name_prefix = "vggt"

    def __init__(self, device: str):
        sys.path.insert(0, str(REPO / "third_party/vggt"))
        from vggt.models.vggt import VGGT
        self.model = VGGT.from_pretrained("facebook/VGGT-1B").to(device).eval()
        self.device = device

    def __call__(self, sample):
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri
        imgs = torch.from_numpy(sample["imgs"]).float().to(self.device)  # [N,3,H,W]
        H, W = imgs.shape[-2:]
        dt = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        with torch.no_grad(), torch.autocast("cuda", dtype=dt):
            pred = self.model(imgs[None])
        extri, intri = pose_encoding_to_extri_intri(pred["pose_enc"], (H, W))
        extri = extri[0].float().cpu().numpy()  # [N,3,4] world2cam
        c2w = []
        for E in extri:
            T = np.eye(4); T[:3, :4] = E
            c2w.append(np.linalg.inv(T))
        K = intri[0].float().cpu().numpy()  # [N,3,3]
        intr = np.stack([[k[0, 0], k[1, 1], k[0, 2], k[1, 2]] for k in K]).mean(0)
        return {"pred_poses": np.stack(c2w), "intr": intr}


class Pi3Model:
    """Pi3 (GT-supervised, 959M). c2w poses; focal recovered from local point maps."""
    name_prefix = "pi3"

    def __init__(self, device: str):
        sys.path.insert(0, str(REPO / "third_party/Pi3"))
        from pi3.models.pi3 import Pi3
        self.model = Pi3.from_pretrained("yyfz233/Pi3").to(device).eval()
        self.device = device

    def __call__(self, sample):
        imgs = torch.from_numpy(sample["imgs"]).float().to(self.device)
        H, W = imgs.shape[-2:]
        dt = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        with torch.no_grad(), torch.autocast("cuda", dtype=dt):
            res = self.model(imgs[None])
        poses = res["camera_poses"][0].float().cpu().numpy()  # [N,4,4] c2w
        intr = None
        lp = res.get("local_points")
        if lp is not None:
            # robust pinhole focal from camera-frame point maps (pp assumed centered):
            # u - cx = fx * x/z  ->  fx = (u - cx) * z / x  (median over confident pixels)
            pts = lp[0].float().cpu().numpy()  # [N,H,W,3]
            vv, uu = np.meshgrid(np.arange(H) - H / 2.0, np.arange(W) - W / 2.0, indexing="ij")
            fxs, fys = [], []
            for n in range(pts.shape[0]):
                x, y, z = pts[n, ..., 0], pts[n, ..., 1], pts[n, ..., 2]
                valid = z > 1e-6
                mx = valid & (np.abs(x) > 1e-4 * z)
                my = valid & (np.abs(y) > 1e-4 * z)
                if mx.sum() > 100:
                    fxs.append(float(np.median(uu[mx] * z[mx] / x[mx])))
                if my.sum() > 100:
                    fys.append(float(np.median(vv[my] * z[my] / y[my])))
            if fxs and fys:
                intr = np.array([np.median(fxs), np.median(fys), W / 2.0, H / 2.0])
        return {"pred_poses": poses, "intr": intr}





# ---------------------------------------------------------------------------
# Metrics per window
# ---------------------------------------------------------------------------

def pose_rows(pred_poses, gt_poses):
    """Consecutive-pair errors. pred_poses[i] = T_{i->last} convention."""
    n_pairs = min(len(pred_poses) - 1, gt_poses.shape[0] - 1)
    rows = []
    for i in range(n_pairs):
        pred_rel = np.linalg.inv(pred_poses[i]) @ pred_poses[i + 1]
        gt_rel = np.linalg.inv(gt_poses[i]) @ gt_poses[i + 1]
        rows.append({
            "pair": i,
            "rot_err_deg": float(rotation_error_degrees(pred_rel[:3, :3], gt_rel[:3, :3])),
            "tdir_err_deg": float(translation_direction_error_degrees(pred_rel[:3, 3], gt_rel[:3, 3])),
            "gt_t_norm": float(np.linalg.norm(gt_rel[:3, 3])),
            "pred_t_norm": float(np.linalg.norm(pred_rel[:3, 3])),
        })
    return rows


def calib_metrics(intr, gt_intr_mean):
    fx, fy = float(intr[0]), float(intr[1])
    gfx, gfy = float(gt_intr_mean[0]), float(gt_intr_mean[1])
    return {
        "pred_fx": fx, "pred_fy": fy, "gt_fx": gfx, "gt_fy": gfy,
        "fx_ape_pct": abs(fx - gfx) / abs(gfx) * 100.0,
        "fy_ape_pct": abs(fy - gfy) / abs(gfy) * 100.0,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def gpu_temp():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        return int(out.stdout.strip().splitlines()[0])
    except Exception:
        return -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_name", required=True)
    ap.add_argument("--datasets", default="sintel,tumrgbd,kitti")
    ap.add_argument("--models", default="ours,anycam,anycalib")
    ap.add_argument("--ours_ckpt", default=str(REPO / "thesis_results/checkpoints/phase_Cb_v6_h100_epoch_0002.pt"))
    ap.add_argument("--image_mode", default="square336", choices=["square336", "aspect336"])
    ap.add_argument("--windows_per_seq", type=int, default=16)
    ap.add_argument("--frame_count", type=int, default=4)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max_temp", type=int, default=87)
    args = ap.parse_args()

    out_dir = REPO / "honest_benchmarks" / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = out_dir / "rows.jsonl"

    git_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True
    ).stdout.strip()

    meta = {
        "run_name": args.run_name,
        "git_commit": git_commit,
        "args": vars(args),
        "ckpt_selection_rule": "best training-validation loss (phase_Cb_v6_h100 epoch 2, val=-0.510)",
        "dilation": DILATION,
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    done = set()
    if rows_path.exists():
        with open(rows_path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    done.add((r["dataset"], r["seq"], r["start"], r["model"]))
                except Exception:
                    pass
        print(f"[resume] {len(done)} rows already present")

    # Build models
    models = {}
    for m in args.models.split(","):
        t0 = time.time()
        if m == "ours":
            models["ours"] = OursModel(args.ours_ckpt, args.device)
        elif m == "ours_nonorm":
            models["ours_nonorm"] = OursModel(args.ours_ckpt, args.device, input_normalization=False)
        elif m == "anycam":
            models["anycam"] = AnyCamBaseline(args.device)
        elif m == "anycalib":
            models["anycalib"] = AnyCalibPerFrame(args.device)
        elif m == "vggt":
            models["vggt"] = VGGTModel(args.device)
        elif m == "pi3":
            models["pi3"] = Pi3Model(args.device)
        elif m == "da3":
            models["da3"] = DA3Model(args.device)
        else:
            raise ValueError(m)
        print(f"[model] {m} ready in {time.time()-t0:.1f}s")

    meta["models"] = list(models)
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    rows_f = open(rows_path, "a")

    for ds_name in args.datasets.split(","):
        image_size = image_size_for(ds_name, args.image_mode)
        ds, per_seq = build_dataset(ds_name, image_size, args.frame_count)
        sel = select_windows(per_seq, args.windows_per_seq)
        n_total = sum(len(v) for v in sel.values())
        window_hash = hashlib.sha256(json.dumps(sel, sort_keys=True).encode()).hexdigest()[:12]
        print(f"[{ds_name}] {len(sel)} seqs, {n_total} windows, image_size={image_size}, hash={window_hash}")

        for seq in sorted(sel):
            for ds_idx in sel[seq]:
                seq_, start = ds._datapoints[ds_idx]
                assert seq_ == seq
                sample = None
                for model_name, model in models.items():
                    key = (ds_name, seq, int(start), model_name)
                    if key in done:
                        continue
                    row = {
                        "dataset": ds_name, "seq": seq, "start": int(start),
                        "model": model_name, "image_mode": args.image_mode,
                    }
                    try:
                        if sample is None:
                            sample = ds[ds_idx]
                            _im = np.asarray(sample["imgs"])
                            # loaders are part of the measurement: every model expects float
                            # RGB in [0,1] (the 2026-08-17 KITTI fault was uint8 0..255 here)
                            if _im.dtype.kind != "f" or float(_im.max()) > 1.0 + 1e-3 or float(_im.min()) < -1e-3:
                                raise RuntimeError(f"loader returned imgs dtype={_im.dtype} range=[{_im.min()},{_im.max()}], expected float in [0,1]")
                        gt_poses = np.asarray(sample["poses"], dtype=np.float64).reshape(-1, 4, 4)
                        projs = np.asarray(sample["projs"], dtype=np.float64)
                        gt_intr = np.stack([
                            [projs[i, 0, 0], projs[i, 1, 1], projs[i, 0, 2], projs[i, 1, 2]]
                            for i in range(projs.shape[0])
                        ]).mean(axis=0)

                        t0 = time.time()
                        out = model(sample)
                        row["time_s"] = round(time.time() - t0, 3)

                        if out["pred_poses"] is not None:
                            row["pose"] = pose_rows(out["pred_poses"], gt_poses)
                        if out.get("intr") is not None:
                            row["calib"] = calib_metrics(out["intr"], gt_intr)
                        if out.get("per_frame_intr") is not None:
                            row["per_frame_intr"] = out["per_frame_intr"]
                    except Exception as e:
                        row["error"] = f"{type(e).__name__}: {e}"
                        row["traceback"] = traceback.format_exc()[-1500:]
                    rows_f.write(json.dumps(row) + "\n")
                    rows_f.flush()

            t = gpu_temp()
            if t >= args.max_temp:
                print(f"[thermal] GPU {t}C >= {args.max_temp}C — cooling 45s")
                time.sleep(45)

        print(f"[{ds_name}] done")

    rows_f.close()
    print("[all done]", rows_path)


if __name__ == "__main__":
    main()
