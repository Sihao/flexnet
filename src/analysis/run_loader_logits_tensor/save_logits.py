#!/Users/donyin/miniconda3/envs/imperial/bin/python -m pdb
# from src.training.dataset_select import get_dataset_obj
from src.analysis.run_loader import RunLoader
from src.flex_neurons.models.layers.flex import Flex2D
from pathlib import Path

import torch

"""
- this is a class that saves the logits of a model layerwise / if the mechanism is threshold based

- consider the situtaion where the logits mechanism is 
    - threshold based (using weights)
    - spatial attention or that of cmp (using intermediate output + dont do anything)

- this is made in case we need to save the logits for further analysis, e.g., clustering
"""


class SaveLogitsHandler:
    def __init__(self, run_loader: RunLoader):
        self.run_loader = run_loader

    # ---- main ----
    def save_logits(self, save_dir: Path):
        self.save_dir = save_dir
        self.save_dir.mkdir(exist_ok=True, parents=True)

        logits_mechanism = self.run_loader.config.get("logits_mechanism", None)
        if logits_mechanism == "THRESHOLD":
            return self._get_threshold_weights()
        elif logits_mechanism == "SpatialAttentionWeightedSum":
            self._get_spatial_attention_output()
        else:
            self._get_dummy_output()

    # ---- method 1 ----
    def _get_threshold_weights(self):
        idx = 0
        for name, layer in self.run_loader.model.named_modules():
            if isinstance(layer, Flex2D):
                if hasattr(layer, "threshold"):
                    threshold_tensor = layer.threshold
                    print(
                        f"Layer {name} has threshold tensor with shape: {threshold_tensor.shape}"
                    )
                    save_path = self.save_dir / f"{idx:03d}.pt"
                    torch.save(threshold_tensor, save_path)
                    idx += 1

    # ---- method 2 ----
    def _get_spatial_attention_output(self):
        raise NotImplementedError(
            "_get_spatial_attention_output is not yet implemented. "
            "Feed images through the model and capture spatial attention logits via forward hooks."
        )

    # ---- method 3 ----
    def _get_dummy_output(self):
        raise NotImplementedError("_get_dummy_output is a placeholder; implement as needed.")


if __name__ == "__main__":
    run_loader = RunLoader("/Users/donyin/Desktop/experiment-cifar/000005")
    save_logits_handler = SaveLogitsHandler(run_loader)
    save_dir = Path("__test__")
    save_logits_handler.save_logits(save_dir)
