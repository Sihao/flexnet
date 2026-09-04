from typing import Dict, List
import torch
import torch.nn as nn


def extract_activations(
    model: nn.Module,
    image: torch.Tensor,
    layer_names: List[str],
) -> Dict[str, torch.Tensor]:
    """Run forward pass, capture activations from named layers via forward hooks.

    Args:
        model: pre-loaded model in eval mode. Left untouched; the image is moved to
            the model's device.
        image: (C, H, W) or (N, C, H, W) input.
        layer_names: list of dotted module names matching model.named_modules() output
            (e.g., 'layer1.0.conv1').

    Returns:
        Dict mapping layer name to activation tensor (shape preserved as captured).

    Raises:
        ValueError if any layer_name is not found in the model.
        ValueError if layer_names contains duplicate entries.
    """
    seen = set()
    duplicates = []
    for name in layer_names:
        if name in seen and name not in duplicates:
            duplicates.append(name)
        seen.add(name)
    if duplicates:
        raise ValueError(f"duplicate layer name(s) in layer_names: {duplicates}")

    name_to_module = {name: module for name, module in model.named_modules()}

    missing = [n for n in layer_names if n not in name_to_module]
    if missing:
        raise ValueError(f"Layer names not found in model: {missing}")

    p = next(model.parameters(), None)
    device = p.device if p is not None else image.device
    was_training = model.training

    result: Dict[str, torch.Tensor] = {}
    hooks = []

    try:
        model.eval()

        if image.dim() == 3:
            image = image.unsqueeze(0)
        image = image.to(device)

        for name in layer_names:
            def make_hook(layer_name):
                def hook(module, input, output):
                    result[layer_name] = output.detach().cpu()
                return hook

            hook_handle = name_to_module[name].register_forward_hook(make_hook(name))
            hooks.append(hook_handle)

        with torch.no_grad():
            model(image)
    finally:
        for h in hooks:
            h.remove()
        if was_training:
            model.train()

    return result
