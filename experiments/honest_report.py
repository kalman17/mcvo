"""Aggregate honest_benchmark rows.jsonl into a report with paired stats.

Usage: python experiments/honest_report.py honest_benchmarks/<run_name>
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

TDIR_MIN_GT_NORM = 1e-4  # pre-declared rule (HONEST_EVAL_PROTOCOL.md §6)
RNG = np.random.default_rng(0)


def boot_ci(vals, stat=np.median, n=1000):
    vals = np.asarray(vals)
    if len(vals) == 0:
        return (float("nan"), float("nan"))
    idx = RNG.integers(0, len(vals), size=(n, len(vals)))
    s = np.sort([stat(vals[i]) for i in idx])
    return (float(s[int(0.025 * n)]), float(s[int(0.975 * n)]))


def fmt(v, ci=None):
    if ci:
        return f"{v:7.2f} [{ci[0]:.2f},{ci[1]:.2f}]"
    return f"{v:7.2f}"


def main(run_dir):
    run_dir = Path(run_dir)
    rows = [json.loads(l) for l in open(run_dir / "rows.jsonl")]

    datasets = sorted({r["dataset"] for r in rows})
    models = sorted({r["model"] for r in rows})

    print(f"# Honest benchmark report: {run_dir.name}")
    print(f"rows={len(rows)}  models={models}  datasets={datasets}\n")

    # failures
    print("## Failures")
    any_fail = False
    for ds in datasets:
        for m in models:
            errs = [r for r in rows if r["dataset"] == ds and r["model"] == m and "error" in r]
            n_all = len([r for r in rows if r["dataset"] == ds and r["model"] == m])
            if errs:
                any_fail = True
                print(f"  {ds}/{m}: {len(errs)}/{n_all} failed — first: {errs[0]['error'][:100]}")
    if not any_fail:
        print("  none")
    print()

    # window-key -> model -> row  (paired analysis)
    index = defaultdict(dict)
    for r in rows:
        index[(r["dataset"], r["seq"], r["start"])][r["model"]] = r

    for ds in datasets:
        print(f"## {ds}")
        keys = sorted(k for k in index if k[0] == ds)

        # ---------------- pose ----------------
        pose_models = [m for m in models if any("pose" in index[k].get(m, {}) for k in keys)]
        if pose_models:
            print("  Pose (consecutive pairs):")
            static_excluded = 0
            for m in pose_models:
                rot, tdir = [], []
                for k in keys:
                    r = index[k].get(m)
                    if not r or "pose" not in r:
                        continue
                    for p in r["pose"]:
                        if not np.isnan(p["rot_err_deg"]):
                            rot.append(p["rot_err_deg"])
                        if p["gt_t_norm"] >= TDIR_MIN_GT_NORM and not np.isnan(p["tdir_err_deg"]):
                            tdir.append(p["tdir_err_deg"])
                        elif p["gt_t_norm"] < TDIR_MIN_GT_NORM:
                            static_excluded += 1
                print(f"    {m:12s} rot med {fmt(np.median(rot), boot_ci(rot))}  mean {fmt(np.mean(rot))} | "
                      f"tdir med {fmt(np.median(tdir), boot_ci(tdir))}  mean {fmt(np.mean(tdir))}  (n_rot={len(rot)}, n_tdir={len(tdir)})")
            print(f"    [static pairs excluded from tdir: {static_excluded // max(1,len(pose_models))} per model]")

            # paired win-rate vs anycam
            if "anycam" in pose_models:
                for m in pose_models:
                    if m == "anycam":
                        continue
                    wins_r, ties_r, n = 0, 0, 0
                    wins_t, n_t = 0, 0
                    for k in keys:
                        ra, rb = index[k].get("anycam"), index[k].get(m)
                        if not ra or not rb or "pose" not in ra or "pose" not in rb:
                            continue
                        for pa, pb in zip(ra["pose"], rb["pose"]):
                            n += 1
                            if pb["rot_err_deg"] < pa["rot_err_deg"]:
                                wins_r += 1
                            if pa["gt_t_norm"] >= TDIR_MIN_GT_NORM:
                                n_t += 1
                                if pb["tdir_err_deg"] < pa["tdir_err_deg"]:
                                    wins_t += 1
                    if n:
                        print(f"    {m} vs anycam: rot win {100*wins_r/n:.1f}% (n={n}), tdir win {100*wins_t/max(1,n_t):.1f}% (n={n_t})")

        # ---------------- calibration ----------------
        calib_models = [m for m in models if any("calib" in index[k].get(m, {}) for k in keys)]
        if calib_models:
            print("  Calibration (fx APE % vs GT of processed image):")
            for m in calib_models:
                fx = [index[k][m]["calib"]["fx_ape_pct"] for k in keys
                      if m in index[k] and "calib" in index[k][m]]
                fy = [index[k][m]["calib"]["fy_ape_pct"] for k in keys
                      if m in index[k] and "calib" in index[k][m]]
                f_ape = [(a + b) / 2 for a, b in zip(fx, fy)]
                print(f"    {m:12s} f_APE med {fmt(np.median(f_ape), boot_ci(f_ape))}  mean {fmt(np.mean(f_ape))}  (n={len(f_ape)})")

            # per-sequence focal consistency: std/mean of predicted fx across windows
            print("  Focal consistency (per-seq std(fx_pred)/mean(fx_pred), median over seqs):")
            for m in calib_models:
                cons = []
                for seq in sorted({k[1] for k in keys}):
                    fxp = [index[k][m]["calib"]["pred_fx"] for k in keys
                           if k[1] == seq and m in index[k] and "calib" in index[k][m]]
                    if len(fxp) >= 3:
                        cons.append(float(np.std(fxp) / (abs(np.mean(fxp)) + 1e-9)))
                if cons:
                    print(f"    {m:12s} med {100*np.median(cons):6.2f}%  mean {100*np.mean(cons):6.2f}%  (seqs={len(cons)})")

            # paired win-rates vs anycalib
            if "anycalib" in calib_models:
                for m in calib_models:
                    if m == "anycalib":
                        continue
                    wins, n = 0, 0
                    for k in keys:
                        ra, rb = index[k].get("anycalib"), index[k].get(m)
                        if not ra or not rb or "calib" not in ra or "calib" not in rb:
                            continue
                        fa = (ra["calib"]["fx_ape_pct"] + ra["calib"]["fy_ape_pct"]) / 2
                        fb = (rb["calib"]["fx_ape_pct"] + rb["calib"]["fy_ape_pct"]) / 2
                        n += 1
                        wins += fb < fa
                    if n:
                        print(f"    {m} vs anycalib: f_APE win {100*wins/n:.1f}% (n={n})")
        print()


if __name__ == "__main__":
    main(sys.argv[1])
