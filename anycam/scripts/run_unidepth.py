import argparse
import os
from pathlib import Path
from typing import Iterable, List

import numpy as np
from PIL import Image

import torch

# Reuse the existing wrapper from the codebase
from anycam.models import make_depth_predictor


def find_images(input_path: Path, pattern: str | None) -> List[Path]:
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        if pattern is None:
            patterns = ["**/*.png", "**/*.jpg", "**/*.jpeg", "**/*.bmp"]
            files: List[Path] = []
            for p in patterns:
                files.extend(sorted(input_path.rglob(p)))
            return files
        else:
            return sorted(input_path.rglob(pattern))
    # Treat as glob
    return sorted(Path().glob(str(input_path)))


def load_image_to_tensor(image_path: Path) -> torch.Tensor:
    img = Image.open(image_path).convert("RGB")
    np_img = np.asarray(img).astype(np.float32) / 255.0
    tensor = torch.from_numpy(np_img).permute(2, 0, 1).unsqueeze(0)  # 1x3xHxW
    return tensor


def save_depth_npy(depth: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, depth.astype(np.float32))


def save_depth_preview_png(depth: np.ndarray, out_path: Path, percentile: float = 2.0) -> None:
    # Robust min/max via percentiles, then map to 16-bit for visualization
    d = depth.copy()
    finite_mask = np.isfinite(d)
    if finite_mask.any():
        vals = d[finite_mask]
        lo = np.percentile(vals, percentile)
        hi = np.percentile(vals, 100.0 - percentile)
        if hi <= lo:
            lo, hi = float(vals.min()), float(vals.max())
        d = np.clip((d - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    else:
        d = np.zeros_like(d)
    img16 = (d * 65535.0 + 0.5).astype(np.uint16)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img16).save(out_path)


def main():
    parser = argparse.ArgumentParser(description="Run UniDepth (single-image depth) on images.")
    parser.add_argument("--input", required=True, type=str,
                        help="Image file, directory, or glob pattern (absolute path recommended).")
    parser.add_argument("--output", required=True, type=str,
                        help="Output directory to save results.")
    parser.add_argument("--pattern", type=str, default=None,
                        help="Optional glob pattern when input is a directory, e.g. '**/*.jpg'.")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"],
                        help="Device selection.")
    parser.add_argument("--version", type=str, default="v2", help="UniDepth version (default: v2).")
    parser.add_argument("--backbone", type=str, default="vits14", help="UniDepth backbone (default: vits14).")
    parser.add_argument("--scaling", type=float, default=0.1,
                        help="Scaling factor used inside the wrapper (default: 0.1).")
    parser.add_argument("--save-npy", action="store_true", help="Save metric depth as .npy (float32).")
    parser.add_argument("--save-png", action="store_true", help="Save preview PNG (16-bit, normalized).")

    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # Configure and load the UniDepth wrapper
    conf = {
        "type": "unidepth",
        "version": args.version,
        "backbone": args.backbone,
        "scaling": args.scaling,
    }

    depth_predictor = make_depth_predictor(conf)
    depth_predictor.eval()
    depth_predictor.to(device)

    # Gather images
    image_paths = find_images(input_path, args.pattern)
    if not image_paths:
        raise SystemExit(f"No images found for input: {input_path}")

    print(f"Found {len(image_paths)} images. Running UniDepth on {device}...")

    for img_path in image_paths:
        rel = img_path.name if input_path.is_file() else img_path.relative_to(img_path.parents[0]) if img_path.is_file() else img_path.name
        # Preserve directory structure under output using relative to the input root when possible
        if input_path.is_dir():
            try:
                rel = img_path.relative_to(input_path)
            except Exception:
                rel = img_path.name
        out_stem = (output_dir / Path(rel)).with_suffix("")

        rgb = load_image_to_tensor(img_path).to(device)

        with torch.no_grad():
            # Wrapper returns a list with inverse depth per AnyCam convention; convert to metric depth
            inv_depth_list = depth_predictor(rgb)
            inv_depth = inv_depth_list[0]  # Bx1xHxW
            depth = (1.0 / inv_depth.clamp_min(1e-6)).squeeze(0).squeeze(0).detach().cpu().numpy()

        if args.save_npy:
            save_depth_npy(depth, out_stem.with_suffix(".npy"))
        if args.save_png:
            save_depth_preview_png(depth, out_stem.with_suffix("_preview.png"))

        if not args.save_npy and not args.save_png:
            # Default to saving .npy if no option specified
            save_depth_npy(depth, out_stem.with_suffix(".npy"))

    print(f"Done. Results saved to: {output_dir}")


if __name__ == "__main__":
    main() 