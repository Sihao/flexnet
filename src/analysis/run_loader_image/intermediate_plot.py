#!/Users/donyin/miniconda3/envs/imperial/bin/python

"""
This script contains utility functions for visualizing tensors.
"""

from pathlib import Path
import matplotlib.pyplot as plt
from src.utils.server import is_on_server
from src.analysis.run_loader import RunLoader
from torch.utils.data import DataLoader
from src.training.dataset_select import get_dataset_obj

import torch

import matplotlib.pyplot as plt
import torch
from natsort import natsorted
from PIL import Image
import os
from pathlib import Path


class IntermediateProcessViz:
    def __init__(self, run_loader):
        self.run_loader = run_loader

    def visualise(self, single_image_batch, chunks, save_dir: Path = Path("./")):
        self.temp_dir = save_dir.parent / f"temp_{save_dir.name.strip('.png')}"
        self.temp_dir.mkdir(exist_ok=True)

        self.register_hook()

        single_image_batch = single_image_batch.to(self.run_loader.device)
        self.run_loader.model(single_image_batch)

        filtered_layer_outputs = {
            k: v for k, v in self.layer_outputs.items() if v.ndim == 4
        }
        self.plot_header(single_image_batch, self.temp_dir / "_header.png")
        self.plot_intermediate_representations(
            filtered_layer_outputs, chunks, save_dir.parent
        )
        self.concate_chunks(save_dir)

    def plot_header(self, single_image_batch, save_path):
        fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(12, 3))
        normalised_first_image = self.normalise_between_0_and_1(
            single_image_batch.squeeze()
        )

        if normalised_first_image.size(0) == 3:
            axes[0].imshow(normalised_first_image.permute(1, 2, 0).cpu().numpy())
            for i, color in enumerate(["R", "G", "B"]):
                axes[i + 1].imshow(normalised_first_image[i].cpu().numpy(), cmap="gray")
                axes[i + 1].set_title(f"{color} channel")
        else:
            axes[0].imshow(normalised_first_image.squeeze().cpu().numpy(), cmap="gray")
            axes[0].set_title("Grayscale Image")
            for ax in axes[1:]:
                ax.axis("off")

        for ax in axes:
            ax.axis("off")
        plt.tight_layout()
        plt.savefig(save_path, dpi=210)
        plt.close()

    def plot_intermediate_representations(
        self, filtered_layer_outputs, chunks, save_dir
    ):
        total_layers = len(filtered_layer_outputs)
        layers_per_chunk = max(1, total_layers // chunks)

        for chunk_index in range(chunks):
            fig, axes = plt.subplots(
                nrows=layers_per_chunk, ncols=3, figsize=(10, 2 * layers_per_chunk)
            )
            start = chunk_index * layers_per_chunk
            end = start + layers_per_chunk
            for i, (key, output) in enumerate(
                list(filtered_layer_outputs.items())[start:end]
            ):
                for j in range(3):
                    channel_image = (
                        self.normalise_between_0_and_1(output[0, j]).cpu().numpy()
                    )
                    ax = axes[i, j]
                    ax.imshow(channel_image, cmap="gray")
                    ax.axis("off")
                    if j == 0:
                        ax.set_title(f"Layer {key}: 1st channel", fontsize=5)
                    elif j == 1:
                        ax.set_title(f"Layer {key}: Middle channel", fontsize=5)
                    else:
                        ax.set_title(f"Layer {key}: Last channel", fontsize=5)

            plt.tight_layout()
            plt.savefig(self.temp_dir / f"_chunk_{chunk_index + 1}.png", dpi=210)
            plt.close()

    def concate_chunks(self, save_dir: Path):
        header_path = self.temp_dir / "_header.png"
        chunk_paths = natsorted(
            [str(path) for path in self.temp_dir.glob("_chunk_*.png")]
        )
        images = [Image.open(header_path)] + [
            Image.open(chunk_path) for chunk_path in chunk_paths
        ]
        total_width = max(image.width for image in images)
        total_height = sum(image.height for image in images)
        concatenated_image = Image.new("RGB", (total_width, total_height))
        concatenated_image.paste((255, 255, 255), (0, 0, total_width, total_height))

        y_offset = 0
        for image in images:
            if image.width < total_width:
                x_offset = (total_width - image.width) // 2
            else:
                x_offset = 0
            concatenated_image.paste(image, (x_offset, y_offset))
            y_offset += image.height
        concatenated_image.save(save_dir)

        # ---- [remove entire temporary dir self.temp_dir] ----
        [os.remove(str(path)) for path in self.temp_dir.glob("*")]
        self.temp_dir.rmdir()

    @staticmethod
    def normalise_between_0_and_1(tensor):
        return (tensor - tensor.min()) / (tensor.max() - tensor.min())

    def register_hook(self):
        self.layer_outputs = {}

        def hook(module, input, output, key):
            if isinstance(output, torch.Tensor):
                self.layer_outputs[key] = output.detach()

        for name, layer in self.run_loader.model.named_modules():
            layer.register_forward_hook(
                lambda module, input, output, key=name: hook(module, input, output, key)
            )


if __name__ == "__main__":
    # dataset_folder = Path("/Users/donyin/Desktop/imagenet100")
    # if is_on_server():
    #     dataset_folder = Path("/rds/general/user/dy723/home/_data/ImageNet100")
    # dataset_imagenet = ImageNet100Dataset(folder=dataset_folder, mode="TRAIN")

    dataset_cifar10 = get_dataset_obj("cifar10", "TRAIN")
    base_folder = Path("/Users/donyin/Desktop/experiment")
    loader_vgg16 = RunLoader(base_folder / "1275fe75", whether_load_checkpoint=True)

    dataloader = DataLoader(dataset_cifar10, batch_size=1, shuffle=False)
    single_image_batch, _ = next(iter(dataloader))

    viz = IntermediateProcessViz(loader_vgg16)
    viz.visualise(
        single_image_batch=single_image_batch,
        chunks=4,
        save_dir=Path("intermediate_representations.png"),
    )
