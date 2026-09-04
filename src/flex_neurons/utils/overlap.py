import json
import os
from pathlib import Path


def get_overlapping_classes(labels_path: Path, dataset_path: Path):
    """
    Determines which classes overlap between the keys in labels_path (ImageNet-100)
    and the subdirectories in dataset_path.

    Args:
        labels_path (Path): Path to Labels.json.
        dataset_path (Path): Path to the dataset root directory (containing class folders).

    Returns:
        overlap (set): Set of class keys present in both.
        in100_keys (set): All keys in Labels.json.
        dataset_keys (set): All keys in dataset_path.
    """
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels.json not found at {labels_path}")

    with open(labels_path, "r") as f:
        labels = json.load(f)

    in100_keys = set(labels.keys())

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path not found at {dataset_path}")

    dataset_keys = set(
        d.name for d in dataset_path.iterdir() if d.is_dir() and not d.name.startswith(".")
    )

    overlap = in100_keys.intersection(dataset_keys)
    
    return overlap, in100_keys, dataset_keys
