"""Extra model adapters for the calibration benchmark. Same call contract as the
adapters in honest_benchmark.py: __call__(sample) -> dict with at least "intr"
([fx, fy, cx, cy] at the resolution of sample["imgs"]) and optionally "pred_poses",
"per_frame_intr", "extra" (free-form JSON-able diagnostics).

Adapters here:
  OursCalib        standalone MCT calibration (no depth/flow pipeline), modes
                   field | multicrop | adaptive — the paper's "ours"
  AnyCamCandidates AnyCam baseline + its 32-candidate evidence (learned probs and
                   flow-distance landscape) + UniDepth median depth for scale
  GeoCalibModel    GeoCalib (ECCV'24) per frame, mean-aggregated
  UniDepthIntr     UniDepthV2 intrinsics head, per frame, mean-aggregated
  VGGTNative       VGGT with its own load_and_preprocess_images policy (crop mode)
  Pi3Native        Pi3 with its own PIXEL_LIMIT / aspect-preserving resize
(DA3 and AnyCalib already apply their own preprocessing inside the existing adapters,
so on native-aspect inputs those adapters *are* the own-preprocessing protocol.)
"""

import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from experiments.calib_bench.motion import anycam_candidate_stats  # noqa: E402

import os
# The released MCT weights (thesis_final_e4 == huggingface thekman17/anycam-mct).
# Cluster: /path/to/... ; override with MCT_CKPT.
DEFAULT_MCT_CKPT = Path(os.environ.get(
    "MCT_CKPT", str(REPO / "thesis_results/checkpoints/b1_normfix/merged_e4.pt")))
MCT_FAT_CFG = {"embed_dim": 1024, "num_heads": 8, "num_layers": 2, "dropout": 0.1,
               "use_visual_conditioning": False, "num_scales": 4}


def _intr4(intr_tensor):
    return np.asarray(intr_tensor.detach().float().cpu()).ravel()[:4].astype(np.float64)


# --------------------------------------------------------------------------- ours

class OursCalib:
    """MCT calibration branch only. imgs may be any aspect (native protocol) or square."""
    name_prefix = "ours"

    def __init__(self, device, ckpt=None, mode="adaptive", crops=3, tau=0.2, size=336,
                 input_normalization=True):
        from experiments.models.anycalib_with_fat import AnyCalibWithMCT
        ckpt = str(ckpt or DEFAULT_MCT_CKPT)
        self.m = AnyCalibWithMCT(model_id="anycalib_pinhole", use_fat=True, fat_config=MCT_FAT_CFG,
                                 use_dinov2_small=False, use_dinov2_full=False,
                                 freeze_backbone=True, freeze_decoder=True, freeze_calibrator=True,
                                 input_normalization=input_normalization).to(device).eval()
        ck = torch.load(ckpt, map_location=device, weights_only=False)
        sd = {k[len("fat_model."):]: v for k, v in ck["model_state_dict"].items() if k.startswith("fat_model.")}
        if not sd:
            raise RuntimeError(f"no fat_model.* keys in {ckpt}")
        missing, unexpected = self.m.load_state_dict(sd, strict=False)
        mct_missing = [k for k in missing if k.startswith("fat.")]
        if mct_missing or unexpected:
            raise RuntimeError(f"MCT weight mismatch loading {ckpt}: missing {mct_missing[:5]} unexpected {list(unexpected)[:5]}")
        self.device, self.mode, self.crops, self.tau, self.size = device, mode, crops, tau, size
        self.ckpt = ckpt

    def _run(self, clip):
        with torch.no_grad():
            out = self.m(clip.to(self.device), cam_id="pinhole")
        return _intr4(out["intrinsics"][0])

    def __call__(self, sample):
        imgs = torch.from_numpy(sample["imgs"]).float()  # [N,3,H,W]
        N, _, H, W = imgs.shape
        sf = self._run(imgs)                       # single field on the given input
        extra = {"single_field": sf.tolist(), "mode": self.mode}
        if self.mode == "field" or W / H < 1.5:
            return {"pred_poses": None, "intr": sf, "extra": extra}
        # square crops across the width, focal is crop-invariant (only cx shifts)
        S = H
        xs = np.linspace(0, W - S, self.crops).round().astype(int)
        per = []
        for x0 in xs:
            per.append(self._run(imgs[:, :, :, x0:x0 + S]))
        per = np.stack(per)  # [crops,4]
        fx_med, fy_med = float(np.median(per[:, 0])), float(np.median(per[:, 1]))
        disp = float((per[:, 0].max() - per[:, 0].min()) / max(np.median(per[:, 0]), 1e-6))
        mc = np.array([fx_med, fy_med, W / 2.0, H / 2.0])
        extra.update({"per_crop_fx": per[:, 0].round(1).tolist(), "crop_dispersion": disp,
                      "multicrop": mc.tolist()})
        if self.mode == "multicrop" or (self.mode == "adaptive" and disp < self.tau):
            return {"pred_poses": None, "intr": mc, "extra": extra}
        return {"pred_poses": None, "intr": sf, "extra": extra}


# --------------------------------------------------------------------------- AnyCam + candidates

class AnyCamCandidates:
    """Vanilla AnyCam (strict load, no GT leakage) exposing its focal-candidate evidence."""
    name_prefix = "anycam"

    def __init__(self, device):
        from omegaconf import OmegaConf
        from anycam.scripts.common import load_model
        cfg = OmegaConf.load(str(REPO / "pretrained_models/anycam_seq8/training_config.yaml"))
        cfg["model"]["use_provided_flow"] = False
        cfg["model"]["train_directions"] = "forward"
        assert cfg["model"].get("use_provided_proj", False) is False
        self.model = load_model(cfg, str(REPO / "pretrained_models/anycam_seq8/training_checkpoint_247500.pt"))
        self.model = self.model.to(device).eval()
        self.device = device
        self.depth_scaling = 0.1  # UniDepthV2Wrapper default: out = 1 / (metric * 0.1)

    def __call__(self, sample):
        imgs = torch.from_numpy(sample["imgs"]).float().unsqueeze(0).to(self.device)
        h, w = imgs.shape[-2], imgs.shape[-1]
        K = torch.tensor([[float(w), 0.0, w / 2.0], [0.0, float(w), h / 2.0], [0.0, 0.0, 1.0]],
                         device=self.device).view(1, 1, 3, 3).expand(1, imgs.shape[1], 3, 3).contiguous()
        with torch.no_grad():
            out = self.model({"imgs": imgs, "projs": K})
        pr = out["pose_result"]
        # selected projection (argmax of learned probs) -> pixels
        P = out["proc_projs"][0, 0].float().cpu().numpy()
        intr = np.array([P[0, 0] * w / 2, P[1, 1] * h / 2, (P[0, 2] + 1) * w / 2, (P[1, 2] + 1) * h / 2])
        # poses: identical to the existing AnyCamBaseline adapter (proc_poses[0])
        pred_poses = out["proc_poses"][0].float().cpu().numpy()
        extra = {}
        probs = pr.get("focal_length_probs")
        cands = pr.get("focal_length_candidates")
        dist = pr.get("dist")
        probs_np = probs[0, 0].float().cpu().numpy() if probs is not None else None
        cands_np = None
        if cands is not None:
            c = cands[0].float().cpu().numpy().ravel()
            cands_np = c * w / 2.0  # normalised focal -> pixels (fx_n = 2 fx / w)
        land = None
        if dist is not None:
            d = dist[0].float()  # [f, nc, ...] or [f, nc, 1, h, w]
            land = d.mean(dim=tuple(i for i in range(d.dim()) if i != 1)).cpu().numpy() if d.dim() > 1 else None
        st = anycam_candidate_stats(probs_np, land, cands_np)
        extra["candidates"] = st
        if probs_np is not None:
            extra["probs"] = [round(float(x), 5) for x in probs_np]
        if land is not None:
            extra["dist_landscape"] = [round(float(x), 6) for x in land]
        if cands_np is not None:
            extra["candidate_focals_px"] = [round(float(x), 1) for x in cands_np]
        # scene scale from the depth teacher (first frame), metres
        try:
            ad = pr["aligned_depths"][0, 0].float()          # [1,1,h,w] scaled inverse depth
            metric = 1.0 / (ad.clamp_min(1e-6) * self.depth_scaling)
            extra["unidepth_med_m"] = float(metric.median().cpu())
        except Exception:
            pass
        return {"pred_poses": pred_poses, "intr": intr, "extra": extra}


# --------------------------------------------------------------------------- GeoCalib

class GeoCalibModel:
    name_prefix = "geocalib"

    def __init__(self, device, weights="pinhole"):
        from geocalib import GeoCalib
        self.model = GeoCalib(weights=weights).to(device)
        self.device = device

    def __call__(self, sample):
        imgs = torch.from_numpy(sample["imgs"]).float().to(self.device)
        per = []
        for i in range(imgs.shape[0]):
            with torch.no_grad():
                r = self.model.calibrate(imgs[i])
            cam = r["camera"]
            f = cam.f[0].float().cpu().numpy().ravel()
            c = cam.c[0].float().cpu().numpy().ravel()
            per.append([float(f[0]), float(f[1]), float(c[0]), float(c[1])])
        per = np.array(per)
        return {"pred_poses": None, "intr": per.mean(0), "per_frame_intr": per.tolist()}


# --------------------------------------------------------------------------- UniDepth intrinsics

class UniDepthIntr:
    """UniDepthV2 (ViT-L) camera head via its official infer(); per-frame, mean-aggregated."""
    name_prefix = "unidepth"

    def __init__(self, device, backbone="vitl14"):
        from anycam.models.depth_predictor_wrapper import _shim_xformers_components
        _shim_xformers_components()
        self.model = torch.hub.load("Brummi/UniDepth:stable", "UniDepth", version="v2",
                                    backbone=backbone, pretrained=True, trust_repo=True).to(device).eval()
        self.device = device

    def __call__(self, sample):
        imgs = torch.from_numpy(sample["imgs"]).float().to(self.device)
        per, extra = [], {}
        for i in range(imgs.shape[0]):
            with torch.no_grad():
                pred = self.model.infer(imgs[i])
            K = pred["intrinsics"]
            K = K[0] if K.dim() == 3 else K
            K = K.float().cpu().numpy()
            per.append([K[0, 0], K[1, 1], K[0, 2], K[1, 2]])
            if i == 0 and "depth" in pred:   # metric depth of the first frame -> scene scale
                extra["unidepth_med_m"] = float(pred["depth"].float().median().cpu())
        per = np.array(per, dtype=np.float64)
        return {"pred_poses": None, "intr": per.mean(0), "per_frame_intr": per.tolist(), "extra": extra}


# --------------------------------------------------------------------------- VGGT / Pi3 native

def _resize_chw(imgs, th, tw):
    return torch.nn.functional.interpolate(imgs, size=(th, tw), mode="bicubic", align_corners=False, antialias=True).clamp(0, 1)


class VGGTNative:
    """VGGT with its own 'crop' policy: width -> 518, height /14, centre-crop if > 518.
    Focal mapped back to the input resolution of sample['imgs']."""
    name_prefix = "vggt"

    def __init__(self, device, mode="crop"):
        sys.path.insert(0, str(REPO / "third_party/vggt"))
        from vggt.models.vggt import VGGT
        self.model = VGGT.from_pretrained("facebook/VGGT-1B").to(device).eval()
        self.device, self.mode = device, mode

    def __call__(self, sample):
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri
        imgs = torch.from_numpy(sample["imgs"]).float().to(self.device)
        N, _, H, W = imgs.shape
        if self.mode == "crop":
            tw = 518
            th = int(round(H * (518 / W) / 14) * 14)
            x = _resize_chw(imgs, th, tw)
            if th > 518:
                y0 = (th - 518) // 2
                x = x[:, :, y0:y0 + 518]
                th = 518
        else:  # pad
            s = 518 / max(H, W)
            th, tw = int(round(H * s / 14) * 14), int(round(W * s / 14) * 14)
            x = _resize_chw(imgs, th, tw)
            pad_h, pad_w = 518 - th, 518 - tw
            x = torch.nn.functional.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=1.0)
            th, tw = 518, 518
        dt = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        with torch.no_grad(), torch.autocast("cuda", dtype=dt):
            pred = self.model(x[None])
        extri, intri = pose_encoding_to_extri_intri(pred["pose_enc"], (th, tw))
        K = intri[0].float().cpu().numpy()
        # in crop mode, width scale is exact; height scale identical (isotropic)
        s_back = W / 518.0 if self.mode == "crop" else max(H, W) / 518.0
        intr = np.stack([[k[0, 0] * s_back, k[1, 1] * s_back, W / 2.0, H / 2.0] for k in K]).mean(0)
        extri = extri[0].float().cpu().numpy()
        c2w = []
        for E in extri:
            T = np.eye(4); T[:3, :4] = E; c2w.append(np.linalg.inv(T))
        return {"pred_poses": np.stack(c2w), "intr": intr, "extra": {"proc_hw": [th, tw]}}


class Pi3Native:
    """Pi3 with its own resize: aspect kept, area <= PIXEL_LIMIT, dims multiples of 14."""
    name_prefix = "pi3"

    def __init__(self, device, pixel_limit=255000):
        sys.path.insert(0, str(REPO / "third_party/Pi3"))
        from pi3.models.pi3 import Pi3
        self.model = Pi3.from_pretrained("yyfz233/Pi3").to(device).eval()
        self.device, self.pixel_limit = device, pixel_limit

    def __call__(self, sample):
        import math
        imgs = torch.from_numpy(sample["imgs"]).float().to(self.device)
        N, _, H, W = imgs.shape
        scale = math.sqrt(self.pixel_limit / (W * H))
        k, m = round(W * scale / 14), round(H * scale / 14)
        while (k * 14) * (m * 14) > self.pixel_limit:
            if k / m > W / H:
                k -= 1
            else:
                m -= 1
        tw, th = max(1, k) * 14, max(1, m) * 14
        x = _resize_chw(imgs, th, tw)
        dt = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        with torch.no_grad(), torch.autocast("cuda", dtype=dt):
            res = self.model(x[None])
        poses = res["camera_poses"][0].float().cpu().numpy()
        intr = None
        lp = res.get("local_points")
        if lp is not None:
            pts = lp[0].float().cpu().numpy()
            vv, uu = np.meshgrid(np.arange(th) - th / 2.0, np.arange(tw) - tw / 2.0, indexing="ij")
            fxs, fys = [], []
            for n in range(pts.shape[0]):
                px, py, pz = pts[n, ..., 0], pts[n, ..., 1], pts[n, ..., 2]
                valid = pz > 1e-6
                mx = valid & (np.abs(px) > 1e-4 * pz)
                my = valid & (np.abs(py) > 1e-4 * pz)
                if mx.sum() > 100:
                    fxs.append(float(np.median(uu[mx] * pz[mx] / px[mx])))
                if my.sum() > 100:
                    fys.append(float(np.median(vv[my] * pz[my] / py[my])))
            if fxs and fys:
                intr = np.array([np.median(fxs) * W / tw, np.median(fys) * H / th, W / 2.0, H / 2.0])
        return {"pred_poses": poses, "intr": intr, "extra": {"proc_hw": [th, tw]}}


# --------------------------------------------------------------------------- Monodepth2 pose net (SfMLearner family)

class Monodepth2Pose:
    """Monodepth2's self-supervised pose network (ResNet-18 encoder on a frame pair +
    small decoder -> axis-angle, translation), mono_640x192 weights trained on KITTI.
    Representative of the tiny photometric-loss pose regressors. Input pairs are resized
    to its native 640x192 (from our square crops: an aspect stretch, stated in the paper).
    Predicted T maps cam_i points into cam_{i+1} (same convention as MCVO)."""
    name_prefix = "monodepth2"

    def __init__(self, device, root=None):
        root = Path(root or (REPO / "third_party/monodepth2"))
        sys.path.insert(0, str(root))
        import networks
        from layers import transformation_from_parameters
        w = root / "models/mono_640x192"
        self.enc = networks.ResnetEncoder(18, False, 2)
        self.enc.load_state_dict(torch.load(w / "pose_encoder.pth", map_location="cpu"))
        self.dec = networks.PoseDecoder(self.enc.num_ch_enc, 1, 2)
        self.dec.load_state_dict(torch.load(w / "pose.pth", map_location="cpu"))
        self.enc.to(device).eval(); self.dec.to(device).eval()
        self.t_from = transformation_from_parameters
        self.device = device

    def __call__(self, sample):
        imgs = torch.from_numpy(sample["imgs"]).float().to(self.device)  # [N,3,H,W] in [0,1]
        x = torch.nn.functional.interpolate(imgs, size=(192, 640), mode="bilinear", align_corners=False)
        absp = [np.eye(4)]
        with torch.no_grad():
            for i in range(x.shape[0] - 1):
                pair = torch.cat([x[i:i + 1], x[i + 1:i + 2]], dim=1)
                aa, t = self.dec([self.enc(pair)])
                T = self.t_from(aa[:, 0], t[:, 0])[0].float().cpu().numpy()
                absp.append(absp[-1] @ np.linalg.inv(T))
        return {"pred_poses": np.stack(absp), "intr": None}
