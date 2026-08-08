"""Merge fine-tuned FAT weights (phase B1 normfix run) into the Cb_v6 checkpoint.

Produces a full checkpoint usable by the honest window benchmark (pose head from
Cb_v6 epoch 2, FAT from the fine-tune).

Usage: python experiments/merge_finetuned_fat.py <finetuned_b1.pt> <out.pt>
"""
import sys
import torch

BASE = "thesis_results/checkpoints/phase_Cb_v6_h100_epoch_0002.pt"


def main(ft_path, out_path, base_path=None):
    base = torch.load(base_path or BASE, map_location="cpu", weights_only=False)
    ft = torch.load(ft_path, map_location="cpu", weights_only=False)
    bsd = base["model_state_dict"]
    fsd = ft.get("model_state_dict", ft)

    fat_keys = {k: v for k, v in fsd.items() if "fat_model.fat." in k}
    assert fat_keys, f"no FAT keys in {ft_path}: sample={list(fsd)[:5]}"
    replaced = 0
    for k, v in fat_keys.items():
        assert k in bsd and bsd[k].shape == v.shape, k
        bsd[k] = v
        replaced += 1

    torch.save({"model_state_dict": bsd, "epoch": ft.get("epoch", -1),
                "merged_from": {"base": BASE, "fat": ft_path}}, out_path)
    print(f"merged {replaced} FAT keys -> {out_path}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
