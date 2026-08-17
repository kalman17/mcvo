"""MCVO full trainer (E2+): self-supervised image-only VO on preprocessed data.

Runs locally or on the cluster. Deterministic 90/10 split (seed 42), per-epoch val,
epoch + intra-epoch (30 min) checkpoints, resume support.

Usage (cluster):
  PYTHONPATH=$REPO python mcvo/train.py --data_dir /path/to/preprocessed \
      --save_dir /path/to/runs/mcvo --backbone facebook/dinov2-base \
      --d_model 512 --depth 8 --batch_size 8 --epochs 6
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from experiments.datasets.preprocessed_dataset import PreprocessedMultiFrameDataset, collate_fn  # noqa
from mcvo.model import MCVO  # noqa
from mcvo.loss import (mcvo_selfsup_loss, identity_baseline_loss,  # noqa
                       pose_distill_loss, epipolar_sampson_loss)


def batch_to_data(batch, device):
    b = {k: v.to(device, non_blocking=True) for k, v in batch.items() if torch.is_tensor(v)}
    return {k: b[k] for k in ("images", "depths", "flows_fwd", "occs_fwd", "calibs")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--save_dir", required=True)
    ap.add_argument("--backbone", default="facebook/dinov2-base")
    ap.add_argument("--d_model", type=int, default=512)
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--max_ahead", type=int, default=3)
    ap.add_argument("--num_workers", type=int, default=5)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--pseudo_pose_dir", default=None,
                    help="dir with cached AnyCam pseudo-poses (enables distillation)")
    ap.add_argument("--lambda_pose", type=float, default=1.0)
    ap.add_argument("--lambda_epi", type=float, default=0.0,
                    help="weight of the epipolar (Sampson) consistency term")
    ap.add_argument("--epi_stride", type=int, default=8)
    ap.add_argument("--init_from", default=None,
                    help="warm start MODEL WEIGHTS ONLY from a checkpoint")
    args = ap.parse_args()

    device = "cuda"
    save_dir = Path(args.save_dir)
    (save_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    # Refuse to silently overwrite another run's checkpoints. A fresh run into a used
    # directory rewrites epoch_000N.pt in place, and any benchmark that recorded only
    # the path then points at a different model than the one it measured (this bit us
    # with mcvo_e4epi: a lambda=0.2 rerun overwrote the lambda=1.0 epoch_0001.pt).
    existing = sorted((save_dir / "checkpoints").glob("epoch_*.pt"))
    if existing and not args.resume:
        raise RuntimeError(
            f"{save_dir}/checkpoints already holds {len(existing)} epoch checkpoint(s) "
            f"(e.g. {existing[0].name}). Pass --resume to continue that run, or use a "
            "new --save_dir. Refusing to overwrite."
        )

    full = PreprocessedMultiFrameDataset(
        data_dir=args.data_dir, max_ahead=args.max_ahead, image_size=336, phase="A",
    )
    if args.pseudo_pose_dir:
        from mcvo.pseudo_pose_dataset import PseudoPoseDataset
        full = PseudoPoseDataset(full, args.pseudo_pose_dir)
    n_val = max(1, int(len(full) * 0.1))
    train_ds, val_ds = random_split(
        full, [len(full) - n_val, n_val], generator=torch.Generator().manual_seed(42),
    )
    dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                    num_workers=args.num_workers, collate_fn=collate_fn,
                    pin_memory=True, drop_last=True)
    vdl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                     num_workers=args.num_workers, collate_fn=collate_fn)
    print(f"[mcvo] dataset {len(full)} -> train {len(train_ds)} / val {len(val_ds)}", flush=True)

    model = MCVO(backbone=args.backbone, d_model=args.d_model,
                 depth=args.depth, heads=args.heads).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    print(f"[mcvo] trainable {sum(p.numel() for p in params)/1e6:.1f}M", flush=True)

    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-5)
    total_steps = (len(dl)) * args.epochs
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps)

    start_epoch = 0
    if args.init_from and Path(args.init_from).exists():
        ck = torch.load(args.init_from, map_location="cpu", weights_only=False)
        missing, unexpected = model.load_state_dict(ck["model_state_dict"], strict=False)
        print(f"[mcvo] warm-started weights from {args.init_from} "
              f"({len(missing)} missing, {len(unexpected)} unexpected)", flush=True)
    if args.resume and Path(args.resume).exists():
        ck = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model_state_dict"])
        opt.load_state_dict(ck["opt_state_dict"])
        sched.load_state_dict(ck["sched_state_dict"])
        start_epoch = ck["epoch"]
        print(f"[mcvo] resumed from {args.resume} at epoch {start_epoch}", flush=True)

    metrics_path = save_dir / "metrics.csv"
    if not metrics_path.exists():
        with open(metrics_path, "w", newline="") as f:
            csv.writer(f).writerow(["epoch", "train_loss", "train_flow_raw",
                                    "val_loss", "val_flow_raw", "val_id_base"])

    def save(epoch, name):
        torch.save({
            "model_state_dict": model.state_dict(),
            "opt_state_dict": opt.state_dict(),
            "sched_state_dict": sched.state_dict(),
            "epoch": epoch, "args": vars(args),
        }, save_dir / "checkpoints" / name)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        if model.backbone is not None:
            model.backbone.eval()
        run_l, run_r, nb, last_save = 0.0, 0.0, 0, time.time()
        for i, batch in enumerate(dl):
            data = batch_to_data(batch, device)
            out = model(images=data["images"])
            losses = mcvo_selfsup_loss(out, data)
            if args.pseudo_pose_dir and "pseudo_poses" in batch:
                d = pose_distill_loss(out, batch["pseudo_poses"].to(device))
                losses["loss"] = losses["loss"] + args.lambda_pose * d["pose_distill"]
                losses["rot_err"] = d["rot_err_rad"]; losses["trans_err"] = d["trans_err_l1"]
            if args.lambda_epi > 0:
                e = epipolar_sampson_loss(out, data, stride=args.epi_stride)
                losses["loss"] = losses["loss"] + args.lambda_epi * e["epipolar"]
                losses["epi"] = e["epipolar"].detach()
            opt.zero_grad(set_to_none=True)
            if not torch.isfinite(losses["loss"]):
                print(f"  [skip] non-finite loss at batch {i+1}", flush=True)
                sched.step(); continue
            losses["loss"].backward()
            # A single corrupt batch with inf grads would make clip_grad_norm scale
            # by 0*inf=NaN and permanently poison the weights — skip such steps.
            gn = torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
            if not torch.isfinite(gn):
                print(f"  [skip] non-finite grad norm at batch {i+1}", flush=True)
                opt.zero_grad(set_to_none=True); sched.step(); continue
            opt.step(); sched.step()
            run_l += float(losses["loss"].detach()); run_r += float(losses["flow_loss_raw"]); nb += 1
            if (i + 1) % 50 == 0:
                extra = f" epi {float(losses.get('epi', 0)):.4f}" if args.lambda_epi > 0 else ""
                print(f"  e{epoch+1} b{i+1}/{len(dl)}: loss {run_l/max(nb,1):.4f} "
                      f"flow_raw {run_r/max(nb,1):.5f}{extra} "
                      f"lr {sched.get_last_lr()[0]:.2e}", flush=True)
            if time.time() - last_save > 1800:
                save(epoch, f"intra_epoch{epoch+1}.pt"); last_save = time.time()

        # validation
        model.eval()
        vl, vr, vb, vid = 0.0, 0.0, 0, 0.0
        with torch.no_grad():
            for batch in vdl:
                data = batch_to_data(batch, device)
                out = model(images=data["images"])
                losses = mcvo_selfsup_loss(out, data)
                if args.pseudo_pose_dir and "pseudo_poses" in batch:
                    d = pose_distill_loss(out, batch["pseudo_poses"].to(device))
                    losses["loss"] = losses["loss"] + args.lambda_pose * d["pose_distill"]
                if args.lambda_epi > 0:
                    e = epipolar_sampson_loss(out, data, stride=args.epi_stride)
                    losses["loss"] = losses["loss"] + args.lambda_epi * e["epipolar"]
                vl += float(losses["loss"]); vr += float(losses["flow_loss_raw"]); vb += 1
                vid += float(identity_baseline_loss(data))
        vl, vr, vid = vl / max(vb, 1), vr / max(vb, 1), vid / max(vb, 1)
        print(f"[mcvo] epoch {epoch+1}: train {run_l/max(nb,1):.4f}/{run_r/max(nb,1):.5f} | "
              f"VAL {vl:.4f}/{vr:.5f} (id_base {vid:.5f})", flush=True)
        with open(metrics_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch + 1, run_l / max(nb,1), run_r / max(nb,1), vl, vr, vid])
        save(epoch + 1, f"epoch_{epoch+1:04d}.pt")
        save(epoch + 1, "latest.pt")

    print("[mcvo] training complete", flush=True)


if __name__ == "__main__":
    main()
