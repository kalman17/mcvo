"""Controlled inference-latency benchmark: AnyCam pipeline vs thesis (MCT) pipeline vs MCVO.

Same process, same GPU, same real windows for every model; CUDA-synchronised timing;
warm-up excluded. Reports median / mean / p90 per model and frame count, plus GPU name,
so the "latency" claim on the README / model card is a measurement, not a reading off
incidental harness timings.

Usage (SLURM):
  PYTHONPATH=. python experiments/bench_latency.py --mcvo_ckpt <mcvo_e3.pt> \
      --ours_ckpt <merged_e4.pt> --out honest_benchmarks/latency.json
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


def timeit(fn, sample, n_warm=5, n_rep=1):
    for _ in range(n_warm):
        fn(sample)
    torch.cuda.synchronize()
    ts = []
    for _ in range(n_rep):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn(sample)
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return ts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mcvo_ckpt", required=True)
    ap.add_argument("--ours_ckpt", required=True)
    ap.add_argument("--out", default=str(REPO / "honest_benchmarks/latency.json"))
    ap.add_argument("--n_windows", type=int, default=30, help="per dataset")
    ap.add_argument("--frame_counts", default="4,8")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--models", default="anycam,thesis_pipeline,mcvo",
                    help="subset of anycam,thesis_pipeline,mcvo,vggt,pi3,da3,mct_calib")
    args = ap.parse_args()

    from experiments.honest_benchmark import OursModel, AnyCamBaseline, MCVOModel, VGGTModel, Pi3Model, DA3Model
    from experiments.calib_bench.datasets import build

    dev = args.device
    gpu = torch.cuda.get_device_name(0)
    builders = {
        "anycam": lambda: AnyCamBaseline(dev),
        "thesis_pipeline": lambda: OursModel(args.ours_ckpt, dev),
        "mcvo": lambda: MCVOModel(args.mcvo_ckpt, dev),
        "vggt": lambda: VGGTModel(dev),
        "pi3": lambda: Pi3Model(dev),
        "da3": lambda: DA3Model(dev),
    }
    def _mct():
        from experiments.calib_bench.models_extra import OursCalib
        return OursCalib(dev, ckpt=args.ours_ckpt, mode="field")
    builders["mct_calib"] = _mct
    models = {n: builders[n]() for n in args.models.split(",")}

    def _nparams(m):
        mods = [v for v in vars(m).values() if isinstance(v, torch.nn.Module)]
        return int(sum(p.numel() for mod in mods for p in mod.parameters()))
    results = {"gpu": gpu, "torch": torch.__version__, "per_model": {},
               "params": {n: _nparams(m) for n, m in models.items()},
               "note": "end-to-end per-window call: images in -> poses out, incl. each model's own "
                       "internal preprocessing and (for AnyCam / thesis pipeline) the UniDepth + UniMatch "
                       "forwards; CUDA-synchronised, 5 warm-up calls excluded"}
    # weight footprint as loaded (bytes of all parameters/buffers, in whatever dtype they are)
    def _nbytes(m):
        mods = [v for v in vars(m).values() if isinstance(v, torch.nn.Module)]
        return int(sum(t.numel() * t.element_size() for mod in mods for t in list(mod.parameters()) + list(mod.buffers())))
    results["weight_bytes"] = {n: _nbytes(m) for n, m in models.items()}
    print("params (M):", {n: round(v / 1e6, 1) for n, v in results["params"].items()}, flush=True)
    print("weights (GiB):", {n: round(v / 2**30, 2) for n, v in results["weight_bytes"].items()}, flush=True)
    for fc in [int(x) for x in args.frame_counts.split(",")]:
        for ds_name in ["sintel", "tumrgbd", "kitti"]:
            ds = build(ds_name, 336, fc)
            idx = np.linspace(0, len(ds._datapoints) - 1, args.n_windows).round().astype(int)
            samples = [ds[int(i)] for i in idx]
            for name, m in models.items():
                # warm-up on the first sample, then one timed call per window
                timeit(m, samples[0], n_warm=5, n_rep=0)
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                ts = []
                for s in samples:
                    ts += timeit(m, s, n_warm=0, n_rep=1)
                peak = torch.cuda.max_memory_allocated() / 2**30   # GiB, weights + activations
                key = f"{name}|f{fc}|{ds_name}"
                results["per_model"][key] = {
                    "median_s": float(np.median(ts)), "mean_s": float(np.mean(ts)),
                    "p90_s": float(np.percentile(ts, 90)), "n": len(ts), "peak_mem_gib": float(peak),
                }
                print(f"{key:32s} median {np.median(ts)*1000:7.1f} ms  mean {np.mean(ts)*1000:7.1f} ms  p90 {np.percentile(ts,90)*1000:7.1f} ms  peak {peak:5.2f} GiB (n={len(ts)})", flush=True)
    # summary over datasets per model and frame count
    summ = {}
    mem = {}
    for fc in [int(x) for x in args.frame_counts.split(",")]:
        for name in models:
            pk = [v["peak_mem_gib"] for k, v in results["per_model"].items() if k.startswith(f"{name}|f{fc}|")]
            mem[f"{name}|f{fc}"] = float(max(pk)) if pk else None
    results["peak_mem_gib"] = mem
    for fc in [int(x) for x in args.frame_counts.split(",")]:
        for name in models:
            meds = [v["median_s"] for k, v in results["per_model"].items() if k.startswith(f"{name}|f{fc}|")]
            summ[f"{name}|f{fc}"] = float(np.median(meds))
    results["summary_median_s"] = summ
    for fc in [int(x) for x in args.frame_counts.split(",")]:
        line = f"[{fc} frames] " + " | ".join(f"{n} {summ[f'{n}|f{fc}']*1000:.1f} ms / {mem[f'{n}|f{fc}']:.2f} GiB" for n in models)
        if "mcvo" in models:
            m = summ[f"mcvo|f{fc}"]
            results[f"factor_over_mcvo_f{fc}"] = {n: summ[f"{n}|f{fc}"] / m for n in models if n != "mcvo"}
            line += "  -> x over mcvo: " + ", ".join(f"{n} {v:.1f}x" for n, v in results[f"factor_over_mcvo_f{fc}"].items())
        print("\n" + line + f"  ({gpu})")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(args.out, "w"), indent=1)
    print("->", args.out)


if __name__ == "__main__":
    main()
