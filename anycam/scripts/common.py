from pathlib import Path

from omegaconf import OmegaConf
import torch

from anycam.trainer import AnyCamWrapper


def get_checkpoint_path(model_path: Path):
    if not model_path.is_dir():
        return model_path

    prefix = "training_checkpoint_"
    ckpts = Path(model_path).glob(f"{prefix}*.pt")

    training_steps = [int(ckpt.stem.split(prefix)[1]) for ckpt in ckpts]

    ckpt_path = f"{prefix}{max(training_steps)}.pt"
    ckpt_path = Path(model_path) / ckpt_path

    return ckpt_path


def load_model(config: OmegaConf, checkpoint_path: Path, config_overwrite: dict = None):
    model_conf = config["model"]
    model_conf["train_directions"] = "forward"

    if config_overwrite is not None:
        model_conf = OmegaConf.merge(model_conf, OmegaConf.create(config_overwrite))
    
    model = AnyCamWrapper(model_conf)

    cp = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    model_cp = cp["model"]

    # Shape-mismatched keys indicate the constructed architecture differs from the
    # checkpoint (e.g. focal_embed_dim). Silently dropping them produces randomly
    # initialized layers masquerading as pretrained ones — refuse instead.
    current_state = model.state_dict()
    mismatched = [
        k for k, v in model_cp.items()
        if k in current_state and current_state[k].shape != v.shape
    ]
    if mismatched:
        details = ", ".join(
            f"{k}: ckpt{tuple(model_cp[k].shape)} vs model{tuple(current_state[k].shape)}"
            for k in mismatched
        )
        raise RuntimeError(
            f"Checkpoint/architecture mismatch — refusing to load with random weights: {details}. "
            f"If loading the official anycam checkpoint, ensure focal_embed_dim is not set (vanilla=0)."
        )

    missing, unexpected = model.load_state_dict(model_cp, strict=False)
    missing = [k for k in missing if not k.startswith(("depth_predictor", "flow_predictor"))]
    if missing:
        print(f"[load_model] WARNING missing keys (left at init): {missing}")
    if unexpected:
        print(f"[load_model] WARNING unexpected checkpoint keys (ignored): {unexpected}")

    return model


