from torchvision import transforms, datasets
from torchvision.datasets import CIFAR10
from pathlib import Path
import torch
import yaml
from src.training.augmentations import AddGaussianNoise
from src.utils.server import is_on_server


class Cifar10Dataset(CIFAR10):
    def __init__(self, mode="TRAIN", gaussian_noise_std=0.1):
        root_path = Path(__file__).parents[2]
        with open(root_path / "configurations.yml", "r") as f:
            config = yaml.safe_load(f)

        if is_on_server():
            data_root = Path(config["system"]["cifar10_dir"]["server"])
        else:
            data_root = Path(config["system"]["cifar10_dir"]["local"])

        # Ensure parent directory exists
        data_root.mkdir(parents=True, exist_ok=True)

        self.norm_means, self.norm_stds = [0.4914, 0.4822, 0.4465], [
            0.2023,
            0.1994,
            0.2010,
        ]

        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=self.norm_means, std=self.norm_stds),
                AddGaussianNoise(mean=0.0, std=gaussian_noise_std),
            ]
        )

        super().__init__(
            root=str(data_root),
            train=(mode == "TRAIN"),
            transform=transform,
            download=True,
        )


if __name__ == "__main__":

    dataset = Cifar10Dataset(mode="TRAIN")
    print(f"Dataset size: {len(dataset)}")
    img, label = dataset[0]
    print(f"Image shape: {img.shape}, Label: {label}")
