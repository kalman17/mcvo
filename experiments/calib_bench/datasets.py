"""Uniform sequence loaders for the calibration benchmark.

Contract (same as the existing harness expects):
    ds._datapoints  : list of (seq_id, start_frame_index)
    ds[idx]         : {"imgs": float32 [N,3,H,W] in [0,1],
                       "projs": float64 [N,3,3] intrinsics AT THE RETURNED RESOLUTION,
                       "poses": float64 [N,4,4] camera-to-world (or None if no GT),
                       "ids": [N] frame indices, "seq": str,
                       "native_hw": (H0, W0) of the raw frame,
                       "depth_med": float or None (median GT depth of first frame, metres,
                                                   used only for motion labelling)}

One resize policy for every dataset: antialiased (INTER_AREA when shrinking), then:
    image_size = int        -> short side to that size, centre-crop to square
    image_size = (h, w)     -> resize to h on the short side keeping aspect, centre-crop
    image_size = None       -> "native": aspect kept, long side capped at MAX_LONG px,
                               dims rounded to multiples of 14 (each model applies its
                               own official preprocessing from there)
The old AnyCam loaders resize with INTER_LINEAR (aliasing on 3-4x shrinks) - the very
input fault found in the audit - which is why the paper runs go through this file only.
"""

import json
import os
import struct
import zlib
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
MAX_LONG = 1024


# --------------------------------------------------------------------------- helpers

def _round14(x):
    return max(14, int(round(x / 14)) * 14)


def resize_and_crop(img, K, image_size):
    """img HxWx3 uint8/float, K 3x3 in native px. Returns (img, K) at target."""
    H0, W0 = img.shape[:2]
    if image_size is None:
        s = min(1.0, MAX_LONG / max(H0, W0))
        th, tw = _round14(H0 * s), _round14(W0 * s)
        crop = None
    elif isinstance(image_size, int):
        s = image_size / min(H0, W0)
        th, tw = int(round(H0 * s)), int(round(W0 * s))
        crop = ((tw - image_size) // 2, (th - image_size) // 2, image_size, image_size)
    else:
        h, w = image_size
        s = max(h / H0, w / W0)
        th, tw = int(round(H0 * s)), int(round(W0 * s))
        crop = ((tw - w) // 2, (th - h) // 2, w, h)
    if (th, tw) != (H0, W0):
        interp = cv2.INTER_AREA if (th < H0 or tw < W0) else cv2.INTER_LINEAR
        img = cv2.resize(img, (tw, th), interpolation=interp)
    K = K.copy().astype(np.float64)
    K[0, :] *= tw / W0
    K[1, :] *= th / H0
    if crop is not None:
        x0, y0, cw, ch = crop
        img = img[y0:y0 + ch, x0:x0 + cw]
        K[0, 2] -= x0
        K[1, 2] -= y0
    return img, K


def _to_chw01(img):
    if img.dtype != np.float32:
        img = img.astype(np.float32) / 255.0
    return np.ascontiguousarray(img.transpose(2, 0, 1))


def _read_rgb(path):
    im = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if im is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)


def _quat_wxyz_to_R(w, x, y, z):
    n = np.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _T(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


# --------------------------------------------------------------------------- base

class SeqDataset:
    """Generic: subclasses fill self.seqs = {seq: dict(frames=[paths], K=[3x3 per frame]
    or single 3x3, poses=[4x4]|None, depth=[paths]|None)} then call _build()."""

    NAME = "seq"

    def __init__(self, image_size, frame_count, dilation, sequences=None, max_frames=None):
        self.image_size = image_size
        self.frame_count = frame_count
        self.dilation = dilation
        self.max_frames = max_frames
        self.seqs = {}
        self._datapoints = []
        self._load(sequences)
        self._build()

    # subclasses implement
    def _load(self, sequences):
        raise NotImplementedError

    def _build(self):
        span = (self.frame_count - 1) * self.dilation
        for seq in sorted(self.seqs):
            n = len(self.seqs[seq]["frames"])
            if self.max_frames:
                n = min(n, self.max_frames)
            for s in range(0, n - span):
                if self._window_ok(seq, s):
                    self._datapoints.append((seq, s))

    def _window_ok(self, seq, start):
        return True

    def _seq_len(self, seq):
        return len(self.seqs[seq]["frames"])

    def _read_frame(self, seq, fid):
        return _read_rgb(self.seqs[seq]["frames"][fid])

    def _K(self, seq, fid):
        K = self.seqs[seq]["K"]
        return np.asarray(K[fid] if isinstance(K, list) else K, dtype=np.float64)

    def _depth_med(self, seq, fid):
        return None

    def __len__(self):
        return len(self._datapoints)

    def __getitem__(self, index):
        seq, start = self._datapoints[index]
        ids = [start + i * self.dilation for i in range(self.frame_count)]
        imgs, projs = [], []
        native_hw = None
        for fid in ids:
            im = self._read_frame(seq, fid)
            native_hw = im.shape[:2]
            im, K = resize_and_crop(im, self._K(seq, fid), self.image_size)
            imgs.append(_to_chw01(im))
            projs.append(K)
        poses = self.seqs[seq]["poses"]
        poses = np.stack([poses[f] for f in ids]).astype(np.float64) if poses is not None else None
        return {
            "imgs": np.stack(imgs), "projs": np.stack(projs), "poses": poses,
            "ids": np.array(ids), "seq": seq, "native_hw": tuple(int(v) for v in native_hw),
            "depth_med": self._depth_med(seq, ids[0]), "data_id": index,
        }


# --------------------------------------------------------------------------- KITTI odometry

class KITTIOdom(SeqDataset):
    NAME = "kitti"
    ROOT = REPO / "data/eval/kitti_odom"

    def _load(self, sequences):
        seqs = sequences or [f"{i:02d}" for i in range(11)]
        for s in seqs:
            sd = self.ROOT / "sequences" / s
            pf = self.ROOT / "poses" / f"{s}.txt"
            if not sd.is_dir() or not pf.is_file():
                continue
            K = None
            for line in open(sd / "calib.txt"):
                if line.startswith("P2:"):
                    K = np.array([float(x) for x in line.split()[1:]]).reshape(3, 4)[:, :3]
            poses = []
            for line in open(pf):
                v = [float(x) for x in line.split()]
                if len(v) == 12:
                    T = np.eye(4); T[:3, :4] = np.array(v).reshape(3, 4); poses.append(T)
            frames = sorted((sd / "image_2").glob("*.png")) or sorted((sd / "image_2").glob("*.jpg"))
            n = min(len(frames), len(poses))
            self.seqs[s] = {"frames": frames[:n], "K": K, "poses": poses[:n]}


# --------------------------------------------------------------------------- KITTI-360 perspective

class KITTI360Persp(SeqDataset):
    """Rectified perspective cam0 (1408x376, 3.74:1). Poses from cam0_to_world.txt; frames
    without a pose (vehicle stationary) are excluded and windows may not span them."""
    NAME = "kitti360"
    ROOT = Path(os.environ.get("KITTI360_ROOT", "/storage/group/dataset_mirrors/01_incoming/kitti_360/KITTI-360"))

    def _load(self, sequences):
        K = None
        for line in open(self.ROOT / "calibration/perspective.txt"):
            if line.startswith("P_rect_00:"):
                K = np.array([float(x) for x in line.split()[1:]]).reshape(3, 4)[:, :3]
        drives = sequences or sorted(p.name for p in (self.ROOT / "data_poses").iterdir())
        for d in drives:
            pf = self.ROOT / "data_poses" / d / "cam0_to_world.txt"
            imgdir = self.ROOT / "data_2d_raw" / d / "image_00/data_rect"
            if not pf.is_file() or not imgdir.is_dir():
                continue
            frames, poses, fids = [], [], []
            for line in open(pf):
                v = line.split()
                if len(v) != 17:
                    continue
                fid = int(v[0])
                p = imgdir / f"{fid:010d}.png"
                if not p.is_file():
                    continue
                frames.append(p); fids.append(fid)
                poses.append(np.array([float(x) for x in v[1:]]).reshape(4, 4))
            self.seqs[d] = {"frames": frames, "K": K, "poses": poses, "fids": fids}

    def _window_ok(self, seq, start):
        # require truly consecutive source frames so dilation means what it says
        fids = self.seqs[seq]["fids"]
        span = (self.frame_count - 1) * self.dilation
        return fids[start + span] - fids[start] == span


# --------------------------------------------------------------------------- Sintel

class Sintel(SeqDataset):
    NAME = "sintel"
    ROOT = REPO / "data/eval/sintel/training"
    TEST = ["alley_2", "ambush_4", "ambush_5", "ambush_6", "cave_2", "cave_4", "market_2",
            "market_5", "market_6", "shaman_3", "sleeping_1", "sleeping_2", "temple_2", "temple_3"]

    @staticmethod
    def _cam(path):
        with open(path, "rb") as f:
            assert f.read(4) == b"PIEH"
            M = np.frombuffer(f.read(72), dtype=np.float64).reshape(3, 3)
            N = np.frombuffer(f.read(96), dtype=np.float64).reshape(3, 4)
        T = np.eye(4); T[:3, :4] = N          # world->cam
        return M.copy(), np.linalg.inv(T)     # K, cam->world

    @staticmethod
    def _dpt(path):
        with open(path, "rb") as f:
            assert f.read(4) == b"PIEH"
            w, h = struct.unpack("ii", f.read(8))
            return np.frombuffer(f.read(4 * w * h), dtype=np.float32).reshape(h, w)

    def _load(self, sequences):
        for s in sequences or self.TEST:
            fdir = self.ROOT / "final" / s
            if not fdir.is_dir():
                continue
            frames = sorted(fdir.glob("frame_*.png"))
            Ks, poses = [], []
            for p in frames:
                K, T = self._cam(self.ROOT / "camdata_left" / s / (p.stem + ".cam"))
                Ks.append(K); poses.append(T)
            depth = [self.ROOT / "depth" / s / (p.stem + ".dpt") for p in frames]
            self.seqs[s] = {"frames": frames, "K": Ks, "poses": poses, "depth": depth}

    def _depth_med(self, seq, fid):
        p = self.seqs[seq]["depth"][fid]
        if not p.is_file():
            return None
        d = self._dpt(p)
        d = d[np.isfinite(d) & (d > 0) & (d < 1e4)]
        return float(np.median(d)) if d.size else None


# --------------------------------------------------------------------------- TUM-RGBD

class TUMRGBD(SeqDataset):
    NAME = "tumrgbd"
    ROOT = REPO / "data/eval/tum_rgbd"
    TEST = ["rgbd_dataset_freiburg3_sitting_halfsphere", "rgbd_dataset_freiburg3_sitting_rpy",
            "rgbd_dataset_freiburg3_sitting_static", "rgbd_dataset_freiburg3_sitting_xyz",
            "rgbd_dataset_freiburg3_walking_halfsphere", "rgbd_dataset_freiburg3_walking_rpy",
            "rgbd_dataset_freiburg3_walking_static", "rgbd_dataset_freiburg3_walking_xyz"]
    K_FR3 = np.array([[535.4, 0, 320.1], [0, 539.2, 247.6], [0, 0, 1.0]])

    def _load(self, sequences):
        for s in sequences or self.TEST:
            sd = self.ROOT / s
            if not sd.is_dir():
                continue
            rgb = [l.split() for l in open(sd / "rgb.txt") if not l.startswith("#")]
            dep = [l.split() for l in open(sd / "depth.txt") if not l.startswith("#")]
            gt = [l.split() for l in open(sd / "groundtruth.txt") if not l.startswith("#")]
            gt_t = np.array([float(g[0]) for g in gt])
            dep_t = np.array([float(d[0]) for d in dep])
            frames, poses, depth = [], [], []
            for r in rgb:
                t = float(r[0])
                j = int(np.argmin(np.abs(gt_t - t)))
                if abs(gt_t[j] - t) > 0.02:
                    continue
                tx, ty, tz, qx, qy, qz, qw = [float(x) for x in gt[j][1:8]]
                k = int(np.argmin(np.abs(dep_t - t)))
                frames.append(sd / r[1]); depth.append(sd / dep[k][1])
                poses.append(_T(_quat_wxyz_to_R(qw, qx, qy, qz), [tx, ty, tz]))
            self.seqs[s] = {"frames": frames, "K": self.K_FR3, "poses": poses, "depth": depth}

    def _depth_med(self, seq, fid):
        d = cv2.imread(str(self.seqs[seq]["depth"][fid]), cv2.IMREAD_UNCHANGED)
        if d is None:
            return None
        d = d.astype(np.float64) / 5000.0
        d = d[d > 0]
        return float(np.median(d)) if d.size else None


# --------------------------------------------------------------------------- EuRoC

class EuRoC(SeqDataset):
    """cam0 undistorted (radtan -> pinhole, alpha=0). GT from state_groundtruth_estimate0
    (body frame) composed with T_BS from sensor.yaml. Camera-to-world."""
    NAME = "euroc"
    ROOT = Path(os.environ.get("EUROC_ROOT", "/storage/group/dataset_mirrors/euroc"))

    def _load(self, sequences):
        import yaml
        seqs = sequences or sorted(p.name for p in self.ROOT.iterdir() if p.name[:2] in ("MH", "V1", "V2"))
        for s in seqs:
            m = self.ROOT / s / "mav0"
            if not (m / "cam0/sensor.yaml").is_file():
                continue
            y = yaml.safe_load(open(m / "cam0/sensor.yaml"))
            fu, fv, cu, cv_ = y["intrinsics"]
            W, H = y["resolution"]
            K0 = np.array([[fu, 0, cu], [0, fv, cv_], [0, 0, 1.0]])
            D = np.array(y["distortion_coefficients"], dtype=np.float64)
            T_BS = np.array(y["T_BS"]["data"]).reshape(4, 4)
            Kopt, _ = cv2.getOptimalNewCameraMatrix(K0, D, (W, H), 0)
            # getOptimalNewCameraMatrix stretches x and y differently (fx != fy). Force
            # square pixels: f = max(fx, fy) zooms in slightly further, so every output
            # pixel still lies inside the valid undistorted region (no black borders).
            f = float(max(Kopt[0, 0], Kopt[1, 1]))
            Knew = np.array([[f, 0, Kopt[0, 2]], [0, f, Kopt[1, 2]], [0, 0, 1.0]])
            m1, m2 = cv2.initUndistortRectifyMap(K0, D, None, Knew, (W, H), cv2.CV_32FC1)
            gt = np.loadtxt(m / "state_groundtruth_estimate0/data.csv", delimiter=",", comments="#")
            gt_t = gt[:, 0]
            frames, poses = [], []
            for p in sorted((m / "cam0/data").glob("*.png")):
                t = int(p.stem)
                j = int(np.argmin(np.abs(gt_t - t)))
                if abs(gt_t[j] - t) > 5e6:  # 5 ms
                    continue
                pos, q = gt[j, 1:4], gt[j, 4:8]  # q = w x y z
                T_WB = _T(_quat_wxyz_to_R(*q), pos)
                frames.append(p); poses.append(T_WB @ T_BS)
            self.seqs[s] = {"frames": frames, "K": Knew, "poses": poses, "maps": (m1, m2)}

    def _read_frame(self, seq, fid):
        im = _read_rgb(self.seqs[seq]["frames"][fid])
        m1, m2 = self.seqs[seq]["maps"]
        return cv2.remap(im, m1, m2, cv2.INTER_LINEAR)


# --------------------------------------------------------------------------- TartanAir

class TartanAir(SeqDataset):
    """image_left, K fixed (320,320,320,240). pose_left.txt: x y z qx qy qz qw in NED camera
    frame (x fwd, y right, z down); converted to CV camera (x right, y down, z fwd)."""
    NAME = "tartanair"
    ROOT = Path(os.environ.get("TARTANAIR_ROOT", "/storage/group/dataset_mirrors/01_incoming/tartanair/dataset"))
    K = np.array([[320.0, 0, 320.0], [0, 320.0, 240.0], [0, 0, 1.0]])
    P_NED_FROM_CV = np.array([[0, 0, 1.0], [1.0, 0, 0], [0, 1.0, 0]])

    def _load(self, sequences):
        trajs = []
        if sequences:
            trajs = [self.ROOT / s for s in sequences]
        else:
            for env in sorted(self.ROOT.iterdir()):
                for diff in ("Easy", "Hard"):
                    for p in sorted((env / diff).glob("P0*")) if (env / diff).is_dir() else []:
                        trajs.append(p)
        for tp in trajs:
            if not (tp / "pose_left.txt").is_file():
                continue
            name = f"{tp.parents[1].name}/{tp.parent.name}/{tp.name}"
            P = np.loadtxt(tp / "pose_left.txt")
            frames = sorted((tp / "image_left").glob("*_left.png"))
            n = min(len(frames), len(P))
            poses = []
            for i in range(n):
                x, y, z, qx, qy, qz, qw = P[i]
                R_ned = _quat_wxyz_to_R(qw, qx, qy, qz)
                poses.append(_T(R_ned @ self.P_NED_FROM_CV, [x, y, z]))
            depth = [tp / "depth_left" / (f.stem + "_depth.npy") for f in frames[:n]]
            self.seqs[name] = {"frames": frames[:n], "K": self.K, "poses": poses, "depth": depth}

    def _depth_med(self, seq, fid):
        p = self.seqs[seq]["depth"][fid]
        if not p.is_file():
            return None
        d = np.load(p)
        d = d[np.isfinite(d) & (d > 0) & (d < 1e4)]
        return float(np.median(d)) if d.size else None


# --------------------------------------------------------------------------- Objectron

class Objectron(SeqDataset):
    """Portrait 1440x1920 phone videos orbiting an object. GT from processed_gt json:
    poses = ARKit camera transforms (x right, y up, z backward), intrinsics for the
    LANDSCAPE sensor -> swapped for the portrait frame (fx<->fy, cx<->cy)."""
    NAME = "objectron"
    ROOT = Path(os.environ.get("OBJECTRON_ROOT", "/home/kalmanm/Documents/thesis/Objectron"))
    FLIP = np.diag([1.0, -1.0, -1.0, 1.0])

    def _load(self, sequences):
        vids = sorted((self.ROOT / "videos").glob("*_video.MOV"))
        if sequences:
            vids = [v for v in vids if v.name.replace("_video.MOV", "") in sequences]
        for v in vids:
            name = v.name.replace("_video.MOV", "")
            gp = self.ROOT / "processed_gt" / f"{name}.json"
            if not gp.is_file():
                continue
            g = json.load(open(gp))
            poses = [np.array(p).reshape(4, 4) @ self.FLIP for p in g["poses"]]
            Ks = []
            for k in g["intrinsics_per_frame"]:
                fx, fy, cx, cy = k[0], k[4], k[2], k[5]
                Ks.append(np.array([[fy, 0, cy], [0, fx, cx], [0, 0, 1.0]]))
            n = min(len(poses), len(Ks))
            self.seqs[name] = {"frames": [v] * n, "K": Ks[:n], "poses": poses[:n], "video": v}
            self._cap = {}

    def _read_frame(self, seq, fid):
        cap = cv2.VideoCapture(str(self.seqs[seq]["video"]))
        cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
        ok, im = cap.read()
        cap.release()
        if not ok:
            raise FileNotFoundError(f"{seq} frame {fid}")
        return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)


# --------------------------------------------------------------------------- Panorama synthesis

class PanoSynth(SeqDataset):
    """Pinhole views rendered from equirectangular panoramas with EXACT intrinsics.
    Each 'sequence' is one panorama × one (hfov, aspect) config; a window is a
    pure-rotation sequence (yaw steps + small pitch/roll jitter) about the centre.
    Frames are rendered on demand at RENDER_H rows and the chosen aspect.
    poses: camera-to-world with zero translation (rotation-only ground truth)."""
    NAME = "panosynth"
    ROOT = Path(os.environ.get("OPENPANO_ROOT", "/storage/group/dataset_mirrors/01_incoming/openpanov2/panoramas"))
    RENDER_H = 480
    # (hfov_deg, aspect w/h) grid — FOV sweep at 4:3 and aspect sweep at ~90° hfov
    CONFIGS = [(40, 4 / 3), (60, 4 / 3), (80, 4 / 3), (100, 4 / 3), (120, 4 / 3),
               (90, 1.0), (90, 16 / 9), (90, 2.4), (90, 3.3)]
    YAW_STEP_DEG = 3.0

    def __init__(self, image_size, frame_count, dilation, sequences=None, max_frames=None,
                 n_panos=60, frames_per_seq=8, seed=0):
        self.n_panos, self.frames_per_seq, self.seed = n_panos, frames_per_seq, seed
        super().__init__(image_size, frame_count, dilation, sequences, max_frames)

    def _load(self, sequences):
        names = [l.strip() for l in open(self.ROOT / "test_panos.txt") if l.strip()]
        rng = np.random.RandomState(self.seed)
        names = list(rng.permutation(names))[: self.n_panos]
        self._pano_cache = {}
        for nm in names:
            for ci, (hfov, ar) in enumerate(self.CONFIGS):
                W = int(round(self.RENDER_H * ar))
                W += W % 2
                f = (W / 2) / np.tan(np.deg2rad(hfov) / 2)
                K = np.array([[f, 0, W / 2], [0, f, self.RENDER_H / 2], [0, 0, 1.0]])
                r = np.random.RandomState((self.seed * 1000 + ci + zlib.crc32(nm.encode())) % (2**32))
                yaw0 = r.uniform(0, 360)
                pitch = r.uniform(-10, 10)
                poses, rots = [], []
                for i in range(self.frames_per_seq):
                    yaw = yaw0 + i * self.YAW_STEP_DEG
                    R = self._rot(np.deg2rad(yaw), np.deg2rad(pitch), 0.0)
                    rots.append(R); poses.append(_T(R, [0, 0, 0]))
                self.seqs[f"{nm}|hfov{hfov}|ar{ar:.2f}"] = {
                    "frames": [nm] * self.frames_per_seq, "K": K, "poses": poses,
                    "rots": rots, "W": W, "pano": nm}

    @staticmethod
    def _rot(yaw, pitch, roll):
        cy, sy, cp, sp, cr, sr = np.cos(yaw), np.sin(yaw), np.cos(pitch), np.sin(pitch), np.cos(roll), np.sin(roll)
        Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        Rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
        Rz = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]])
        return Ry @ Rx @ Rz

    def _pano(self, nm):
        if nm not in self._pano_cache:
            if len(self._pano_cache) > 4:
                self._pano_cache.clear()
            p = self.ROOT / "test" / nm
            im = _read_rgb(p)
            if im.shape[1] > 4096:
                im = cv2.resize(im, (4096, 2048), interpolation=cv2.INTER_AREA)
            self._pano_cache[nm] = im
        return self._pano_cache[nm]

    def _read_frame(self, seq, fid):
        s = self.seqs[seq]
        pano = self._pano(s["pano"])
        Hp, Wp = pano.shape[:2]
        H, W = self.RENDER_H, s["W"]
        K, R = s["K"], s["rots"][fid]
        u, v = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
        d = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u)], -1)
        d = d @ R.T                                        # camera ray -> world
        d /= np.linalg.norm(d, axis=-1, keepdims=True)
        lon = np.arctan2(d[..., 0], d[..., 2])             # yaw around +y
        lat = np.arcsin(np.clip(d[..., 1], -1, 1))         # +y down in CV camera
        mx = ((lon / (2 * np.pi)) + 0.5) * Wp
        my = ((lat / np.pi) + 0.5) * Hp
        return cv2.remap(pano, mx.astype(np.float32), my.astype(np.float32),
                         cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)


# --------------------------------------------------------------------------- registry

REGISTRY = {
    "kitti": KITTIOdom, "kitti360": KITTI360Persp, "sintel": Sintel, "tumrgbd": TUMRGBD,
    "euroc": EuRoC, "tartanair": TartanAir, "objectron": Objectron, "panosynth": PanoSynth,
}

DEFAULT_DILATION = {
    "kitti": 1, "kitti360": 1, "sintel": 1, "tumrgbd": 3, "euroc": 2, "tartanair": 1,
    "objectron": 3, "panosynth": 1,
}


def build(name, image_size, frame_count, dilation=None, **kw):
    cls = REGISTRY[name]
    dil = DEFAULT_DILATION[name] if dilation is None else dilation
    return cls(image_size, frame_count, dil, **kw)
