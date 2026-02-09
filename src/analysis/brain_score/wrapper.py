import torch
from functools import partial
from brainscore_vision.model_helpers.activations.pytorch import (
    PytorchWrapper,
    load_preprocess_images,
)
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from src.modules.models.VGG import VGG


def get_brain_model(model_name, region_layer_map, model=None, ckpt_path=None, config=None, batch_size=64):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Instantiate/Load Model
    if model is None:
        # Ensure config matches training
        if config is None:
            # Default helper logic (though run_brain_score should pass config)
            config = {
                "network": "VGG",
                "vgg_variant": "16",
                "dataset": "imagenet100",
                "use_flex": True,
                "fully_flex": True,
            }
        model = VGG(config)
        model.to(device)
        
        if ckpt_path:
            checkpoint = torch.load(ckpt_path, map_location=device)
            if "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            elif "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            else:
                state_dict = checkpoint

            # Fix 'module.' prefix if present
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            model.load_state_dict(state_dict)
    
    model.eval()

    # 2. Preprocessing
    # load_preprocess_images handles resizing (224) and normalization (ImageNet)
    preprocessing = partial(load_preprocess_images, image_size=224)

    # 3. Activations Wrapper
    activations_model = PytorchWrapper(
        identifier=model_name,
        model=model,
        preprocessing=preprocessing,
        batch_size=batch_size,
    )

    # 4. Region Commitment
    # region_layer_map is now required/passed in
    # expect format: {'V4': 'layer_name', 'IT': 'layer_name'}
    
    brain_model = ModelCommitment(
        identifier=model_name,
        activations_model=activations_model,
        layers=list(region_layer_map.values()),
        region_layer_map=region_layer_map,
    )
    return brain_model
