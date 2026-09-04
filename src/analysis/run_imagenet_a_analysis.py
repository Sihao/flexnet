import json
import torch
import os
import argparse
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader
import sys

# Add project root to sys.path
sys.path.append(os.getcwd())

from src.analysis.run_loader import RunLoader
from src.flex_neurons.data.dataset_imagenet import ImageNetRDataset
from src.flex_neurons.utils.device import select_device
from src.flex_neurons.utils.normalization import (
    Normalize,
    denormalize_batch,
    IMAGENET_MEAN,
    IMAGENET_STD,
)
from src.flex_neurons.utils.overlap import get_overlapping_classes
import yaml


def run_imagenet_a_analysis(
    experiment_id, imagenet_a_path, batch_size=32, device_str=None
):
    """
    Run analysis on ImageNet-A.
    """

    # 1. Setup
    if isinstance(experiment_id, int) or (
        isinstance(experiment_id, str) and experiment_id.isdigit()
    ):
        exp_path = f"__local__/experiment-{experiment_id}/000000"
    else:
        exp_path = experiment_id

    print(f"Loading Experiment from {exp_path}...")

    # Output Setup
    output_dir = Path(exp_path) / "results" / "imagenet_a_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "imagenet_a_results.json"
    print(f"Results will be saved to: {output_file}")

    # Device
    if device_str:
        device = torch.device(device_str)
    else:
        device = select_device()
    print(f"Using device: {device}")

    # 2. Load Model
    try:
        loader = RunLoader(exp_path)
    except Exception as e:
        print(f"Error loading experiment from {exp_path}: {e}")
        return

    model = loader.model
    # Wrap model with Normalize
    # Note: validation/attacks usually operate on [0,1], but model might expect normalized.
    # Here we wrap model to accept [0,1].
    normalization = Normalize(IMAGENET_MEAN, IMAGENET_STD).to(device)
    model = torch.nn.Sequential(normalization, model)
    model.eval()
    model.to(device)

    # 3. Locate Labels.json (for IN-100 filtering)
    root_path = Path(__file__).parents[2]
    config_path = root_path / "configs" / "configurations.yml"

    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        # Check system/server logic
        from src.flex_neurons.utils.server import is_on_server

        if is_on_server():
            in100_path = Path(config["system"]["imagenet_dir"]["server"])
        else:
            in100_path = Path(config["system"]["imagenet_dir"]["local"])

        labels_file = in100_path / "Labels.json"

        if not labels_file.exists():
            print(f"Error: Labels.json not found at {labels_file}")
            return

        print(f"Using Labels.json from: {labels_file}")

    except Exception as e:
        print(f"Error reading configuration to find Labels.json: {e}")
        return

    # 4. Dataset & Overlap
    imagenet_a_path = Path(imagenet_a_path)
    if not imagenet_a_path.exists():
        print(f"Error: ImageNet-A path not found at {imagenet_a_path}")
        return

    print("Checking Class Overlap...")
    overlap, in100_keys, ina_keys = get_overlapping_classes(labels_file, imagenet_a_path)
    print(f"ImageNet-100 Classes: {len(in100_keys)}")
    print(f"ImageNet-A Classes: {len(ina_keys)}")
    print(f"Overlapping Classes: {len(overlap)}")

    if len(overlap) == 0:
        print("Error: No overlapping classes found. Aborting.")
        return

    print("Loading Dataset...")
    try:
        # ImageNetRDataset explicitly filters by labels_file keys and existence in folder
        dataset = ImageNetRDataset(folder=imagenet_a_path, labels_file=labels_file)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    print(f"Dataset loaded with {len(dataset)} images.")

    if len(dataset) == 0:
        print("Error: Dataset is empty.")
        return

    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )

    # 5. Evaluate
    correct = 0
    total = 0

    results = {}

    for images, labels in tqdm(dataloader, desc="Evaluating ImageNet-A"):
        images, labels = images.to(device), labels.to(device)

        # Denormalize because ImageNetRDataset normalizes, but our model wrapper also normalizes
        # We need to pass [0,1] to the model wrapper.
        images = denormalize_batch(images, device)
        images = torch.clamp(images, 0, 1)

        with torch.no_grad():
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    acc = correct / total if total > 0 else 0
    print(f"ImageNet-A Accuracy: {acc:.4f}")

    results["accuracy"] = acc
    results["correct"] = correct
    results["total"] = total
    results["overlap_count"] = len(overlap)

    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)
    
    print(f"Done. Results saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ImageNet-A Analysis")
    parser.add_argument("--experiment", type=str, required=True, help="Experiment ID or path")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument(
        "--data_path", 
        type=str, 
        default="/mnt/bronknas/Sihao/FlexNet/Datasets/imagenet-a/",
        help="Path to ImageNet-A dataset"
    )

    args = parser.parse_args()

    run_imagenet_a_analysis(
        experiment_id=args.experiment,
        imagenet_a_path=args.data_path,
        batch_size=args.batch_size,
        device_str=args.device,
    )
