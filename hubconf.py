import torch
from torch import nn
from dotdict import dotdict
from omegaconf import OmegaConf
import huggingface_hub


from anycam.trainer import AnyCamWrapper
from anycam.scripts.fit_video import fit_video
from anycam.utils.geometry import se3_ensure_numerical_accuracy


ANYCAM_VERSIONS = {
    ("1.0", "seq2"): "anycam_v1_seq2",
    ("1.0", "seq8"): "anycam_v1_seq8",
}


class AnyCamInference(nn.Module):
    def __init__(self, anycam_model):
        super(AnyCamInference, self).__init__()

        self.anycam_model = anycam_model

    def process_video(self, frames, config=None, ba_refinement=True):
        """
        Process a video by running the AnyCam model on the provided frames.
        
        Args:
            frames: List of frames as numpy arrays with shape (H,W,3) and values in [0,1]
            config: Optional configuration dictionary for processing
            ba_refinement: Whether to perform bundle adjustment (default: True)
        
        Returns:
            Dict containing:
                - trajectory: The estimated camera trajectory
                - projection_matrix: The camera projection matrix
                - depths: Estimated depth maps
                - uncertainties: Uncertainty maps if available
                - extras: Additional information from the fitting process
        """

        
        # Default configuration if none provided
        if config is None:
            default_config = {
                "with_rerun": False,
                "do_ba_refinement": ba_refinement,
                "prediction": {
                    "model_seq_len": 100,
                    "shift": 99,
                    "square_crop": True,
                    "return_all_uncerts": False,
                },
                "ba_refinement": {
                    "with_rerun": False,
                    "max_uncert": 0.05,
                    "lambda_smoothness": 0.1,
                    "long_tracks": True,
                    "n_steps_last_global": 5000,
                },
                "ba_refinement_level": 2,
                "dataset": {
                    "image_size": [336, None]
                }
            }
            config = dotdict(default_config)
        elif not isinstance(config, dotdict):
            config = dotdict(config)
        
        # Ensure the BA refinement setting is applied to the config
        config.do_ba_refinement = ba_refinement
        
        criterion = None

        # Process the video frames
        trajectory, proj, extras_dict, ba_extras = fit_video(
            config,
            self.anycam_model,
            criterion,
            frames,
            return_extras=True,
        )
        
        # Extract depth and uncertainty information
        depths = extras_dict.get("seq_depths", [])
        
        # Get uncertainties based on whether BA refinement was used
        if not ba_refinement:
            best_candidate = extras_dict.get("best_candidate", 0)
            uncertainties = extras_dict.get("uncertainties", [])
            if uncertainties:
                uncertainties = torch.stack(uncertainties)[:, 0, best_candidate, :1, :, :]
        else:
            uncertainties = extras_dict.get("ba_uncertainties", None)
            
        # Ensure numerical accuracy of trajectory
        trajectory = [se3_ensure_numerical_accuracy(torch.tensor(pose)) for pose in trajectory]
        
        return {
            "trajectory": trajectory,
            "projection_matrix": proj,
            "depths": depths,
            "uncertainties": uncertainties,
            "extras": extras_dict,
            "ba_extras": ba_extras
        }


def AnyCam(version="1.0", training_variant="seq8", pretrained=True):
    """
    Load the AnyCam model with the specified version and training variant.

    Args:
        version (str): The version of the AnyCam model to load.
        training_variant (str): The training variant of the AnyCam model to load.
        pretrained (bool): Whether to load pretrained weights.

    Returns:
        nn.Module: The AnyCam model.
    """


    # Check if the version and training variant are valid
    if (version, training_variant) not in ANYCAM_VERSIONS:
        raise ValueError(f"Invalid version or training variant: {version}, {training_variant}")
    
    # Get the model name based on the version and training variant
    model_name = ANYCAM_VERSIONS[(version, training_variant)]

    # Load the model from huggingface using the model name
    config_path = huggingface_hub.hf_hub_download(repo_id=f"fwimbauer/{model_name}", filename="config.yaml", repo_type="model")
    model_path = huggingface_hub.hf_hub_download(repo_id=f"fwimbauer/{model_name}", filename="pytorch_model.bin", repo_type="model")

    # Setup AnyCamWrapper
    # Load the config file

    config = OmegaConf.load(config_path)
    
    # Get model configuration
    model_conf = config["model"]
    model_conf["use_provided_flow"] = False
    model_conf["train_directions"] = "forward"
    
    # Create AnyCamWrapper instance
    model = AnyCamWrapper(model_conf)
    
    if pretrained:
        # Load the pretrained weights
        checkpoint = torch.load(model_path, map_location="cpu")
        model.load_state_dict(checkpoint["model"], strict=False)
    
    # Setup AnyCamInference
    inference_model = AnyCamInference(model)
    
    # Set to eval mode
    model.eval()
    inference_model.eval()
    
    return inference_model
