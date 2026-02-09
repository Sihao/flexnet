import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[2]))

import os
import yaml
from ffcv.writer import DatasetWriter
from ffcv.fields import RGBImageField, IntField

# Import existing dataset class
from src.training.dataset_imagenet import ImageNet100Dataset


def write_beton(dataset, output_path, resolution=224):
    """
    Write a PyTorch dataset to FFCV .beton format.
    """
    print(f"Writing dataset to {output_path}...")

    fields = {
        "image": RGBImageField(
            write_mode="jpg", max_resolution=resolution, compress_probability=0.0
        ),
        "label": IntField(),
    }

    # Create writer
    writer = DatasetWriter(str(output_path), fields, num_workers=1)

    # Write dataset
    print(f"Starting writer.from_indexed_dataset with len={len(dataset)}...")
    writer.from_indexed_dataset(dataset)
    print(f"Finished writing {output_path}")


def main():
    # Load configuration to get paths
    config_path = Path("configurations.yml")
    if not config_path.exists():
        raise FileNotFoundError(
            "configurations.yml not found. Please run from project root."
        )

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Determine paths
    # Assuming local execution for now as per user context
    # Adjust logic if server needed, but user said "switch to new env... on baseline vgg16 training imagenet100"
    # We'll use the paths defined in config

    data_root = Path(config["system"]["imagenet_dir"]["local"])
    output_dir = data_root

    # 1. Write Validation Set
    val_dataset = ImageNet100Dataset(folder=data_root, mode="VAL")
    val_output_path = output_dir / "val.beton"
    if not val_output_path.exists():
        write_beton(val_dataset, val_output_path)
    else:
        print(f"Skipping {val_output_path}, already exists.")

    # 2. Write Training Set
    train_dataset = ImageNet100Dataset(folder=data_root, mode="TRAIN")
    train_output_path = output_dir / "train.beton"
    if not train_output_path.exists():
        write_beton(train_dataset, train_output_path)
    else:
        print(f"Skipping {train_output_path}, already exists.")


if __name__ == "__main__":
    main()
