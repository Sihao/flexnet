import json
import torch
import os
import argparse
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader

from src.analysis.run_loader import RunLoader
from src.training.dataset_imagenet import ImageNetRDataset
from src.utils.device import select_device
from src.utils.normalization import (
    Normalize,
    denormalize_batch,
    IMAGENET_MEAN,
    IMAGENET_STD,
)


def run_perturbation_analysis(
    experiment_id, imagenet_c_path, batch_size=32, device_str=None
):
    """
    Run perturbation analysis on ImageNet-C.
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
    output_dir = Path(exp_path) / "results" / "perturbation_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "perturbation_analysis_results.json"
    print(f"Results will be saved to: {output_file}")

    # Device
    if device_str:
        device = torch.device(device_str)
    else:
        # User requested "default to gpu" if no flag, but select_device usually handles "cuda if available"
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
    normalization = Normalize(IMAGENET_MEAN, IMAGENET_STD).to(device)
    model = torch.nn.Sequential(normalization, model)
    model.eval()
    model.to(device)

    # 3. Locate Labels.json (for IN-100 filtering)
    # Logic adapted from src/training/dataset_select.py
    import yaml

    root_path = Path(__file__).parents[2]
    config_path = root_path / "configurations.yml"

    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        # Check system/server logic
        from src.utils.server import is_on_server

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

    # 4. Perturbation Loop
    imagenet_c_path = Path(imagenet_c_path)
    if not imagenet_c_path.exists():
        print(f"Error: ImageNet-C path not found at {imagenet_c_path}")
        return

    # Scan for types (subdirectories)
    # Exclude non-directory items and hidden files
    perturbation_types = [
        d.name
        for d in imagenet_c_path.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    ]
    perturbation_types.sort()

    print(f"Found {len(perturbation_types)} perturbation types: {perturbation_types}")

    results = {}
    if output_file.exists():
        try:
            with open(output_file, "r") as f:
                results = json.load(f)
            print(f"Loaded existing results from {output_file}")
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Could not load existing results ({e}), starting fresh.")
            results = {}

    severities = [1, 2, 3, 4, 5]

    for p_type in perturbation_types:
        print(f"\nProcessing Perturbation: {p_type}")

        if p_type not in results:
            results[p_type] = {"severity": [], "accuracies": []}

        current_severities = results[p_type]["severity"]
        current_accuracies = results[p_type]["accuracies"]

        # Determine which severities to run (skip if already done)
        severities_to_run = [s for s in severities if s not in current_severities]

        if not severities_to_run:
            print(f"  All severities for {p_type} already processed. Skipping.")
            continue

        for severity in severities_to_run:
            # Construct path: root / type / severity
            # Note: User said "Inside .../{type}/{severity}/ you will find directories for 200 imagenet classes"
            # So path is imagenet_c_path / p_type / str(severity)
            dataset_path = imagenet_c_path / p_type / str(severity)

            if not dataset_path.exists():
                print(
                    f"  Warning: Path {dataset_path} does not exist. Skipping severity {severity}."
                )
                continue

            print(f"  Severity {severity} - Loading Dataset...")

            # Use ImageNetRDataset logic which filters by Labels.json
            try:
                dataset = ImageNetRDataset(folder=dataset_path, labels_file=labels_file)
            except Exception as e:
                print(f"  Error loading dataset for {p_type} s={severity}: {e}")
                continue

            if len(dataset) == 0:
                print(
                    f"  Warning: Dataset is empty for {p_type} s={severity}. Skipping."
                )
                continue

            dataloader = DataLoader(
                dataset, batch_size=batch_size, shuffle=False, num_workers=4
            )

            # Evaluate
            correct = 0
            total = 0

            # Progress bar for batches
            for images, labels in tqdm(
                dataloader, desc=f"  Eval {p_type} s={severity}", leave=False
            ):
                images, labels = images.to(device), labels.to(device)

                # Model wrapper handles normalization (expected input [0,1] normalized by wrapper)
                # ImageNetRDataset reads image (PIL/read_image) -> /255 -> transforms (Resize, Normalize)
                # WAIT. ImageNetRDataset applies transforms.Normalize internally!
                # See src/training/dataset_imagenet.py:
                # transform_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[...]))
                # But our model wrapper ALSO applies normalization.
                # Double normalization is bad.

                # Check run_attack_comparison strategy:
                # Loader loads ImageNet100Dataset -> applies normalization.
                # run_attack_comparison DENORMALIZES batch before attack, then clamps [0,1].
                # Then wrapper re-normalizes.

                # Here we are just evaluating.
                # Option 1: Don't use wrapper model, just use raw model.
                # (Raw model expects normalized input provided by dataset).
                # Option 2: Use wrapper model, but denormalize dataset output.

                # Since I am using `ImageNetRDataset` which enforces normalization in `__init__`,
                # the dataset yields normalized tensors.
                # The model loaded via `RunLoader` expects normalized tensors (normally).
                # BUT, I wrapped `model` with `Normalize` in this script (line ~50).
                # `Normalize` expects [0,1].

                # If I use `ImageNetRDataset` as is, it outputs Normalized data.
                # Passing Normalized data to `Normalize` wrapper -> Wrong.

                # Fix: Denormalize the batch from the dataloader before passing to model wrapper.
                # This aligns with `run_attack_comparison` logic where attacks (and model wrapper) operate on [0,1] concepts
                # but dataset provides normalized.

                images = denormalize_batch(images, device)
                images = torch.clamp(images, 0, 1)  # Ensure [0,1]

                with torch.no_grad():
                    outputs = model(images)
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()

            acc = correct / total if total > 0 else 0
            print(f"  Accuracy: {acc:.4f}")

            # Store
            current_severities.append(severity)
            current_accuracies.append(acc)

            # Sort by severity to keep JSON clean
            # Zip, sort, unzip
            zipped = sorted(zip(current_severities, current_accuracies))
            results[p_type]["severity"] = [x[0] for x in zipped]
            results[p_type]["accuracies"] = [x[1] for x in zipped]

            # Save incremental
            with open(output_file, "w") as f:
                json.dump(results, f, indent=4)

    print(f"Perturbation Analysis Complete. Results saved to {output_file}")


if __name__ == "__main__":
    # Test run
    pass
