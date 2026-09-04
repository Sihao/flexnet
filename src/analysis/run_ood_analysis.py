import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import json
import datetime
import numpy as np
from pathlib import Path
import sys
import os

# Add project root to path
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from src.analysis.run_loader import RunLoader
from src.flex_neurons.data.dataset_select import get_dataset_obj
from src.flex_neurons.utils.device import select_device


def run_ood_analysis(experiment_id, batch_size=32, limit=None, device=None):
    """
    Run OOD Analysis (Accuracy on ImageNet-R) for a given experiment.

    Args:
        experiment_id: Experiment ID.
        batch_size: Batch size.
        limit: Max number of batches to run (for debugging/quick check).
        device: 'cpu' or 'cuda'. If None, auto-select.
    """
    # 1. Setup
    if isinstance(experiment_id, int) or (
        isinstance(experiment_id, str) and experiment_id.isdigit()
    ):
        exp_path = f"__local__/experiment-{experiment_id}/000000"
    else:
        exp_path = experiment_id

    # Timestamped output
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_base_dir = Path(exp_path) / "results" / "ood_analysis" / timestamp
    output_base_dir.mkdir(parents=True, exist_ok=True)

    if device is None:
        device = select_device()
    print(f"[INFO] Using device: {device}")

    # 2. Load Model
    print(f"[INFO] Loading Experiment from {exp_path}...")
    try:
        loader = RunLoader(exp_path, device=device)
    except Exception as e:
        print(f"[ERROR] Error loading experiment: {e}")
        return

    model = loader.model
    model.eval()
    model.to(device)

    # 3. Load Dataset
    print("[INFO] Loading ImageNet-Sketch Dataset...")
    try:
        # Manually constructing dataset object to point to Sketch path while using ImageNet-R class logic
        # This avoids modifying dataset_select.py which assumes imagenet-r
        # Sketch path: /mnt/bronknas/Sihao/FlexNet/Datasets/imagenetsketch/sketch/

        # We need to import ImageNetRDataset here or assume get_dataset_obj handles it.
        # run_ood_analysis imports get_dataset_obj.
        # Let's import the class directly to be safe and flexible.
        from src.flex_neurons.data.dataset_imagenet import ImageNetRDataset

        # Load Labels from Experiment Config (or assume standard ImageNet100)
        # We need the Labels.json path. dataset_select does this.
        # Let's peek at how dataset_select gets the labels path.
        # It uses: config["system"]["imagenet_dir"]["local"] / "Labels.json"

        # Locate Labels.json: try the standard data/ path first, then fall back
        # to reading configs/configurations.yml for the configured imagenet_dir.
        project_root = Path(__file__).resolve().parents[2]
        # parents[0]=analysis, parents[1]=src, parents[2]=repo root

        labels_file = project_root / "data" / "imagenet100" / "Labels.json"

        if not labels_file.exists():
            import yaml

            with open(project_root / "configs" / "configurations.yml", "r") as f:
                config = yaml.safe_load(f)
            labels_file = (
                Path(config["system"]["imagenet_dir"]["local"]) / "Labels.json"
            )

        sketch_folder = Path(
            "/mnt/bronknas/Sihao/FlexNet/Datasets/imagenetsketch/sketch/"
        )

        dataset = ImageNetRDataset(folder=sketch_folder, labels_file=labels_file)

    except Exception as e:
        print(f"[ERROR] Failed to load ImageNet-Sketch: {e}")
        return

    # Handle per-class limit
    if limit is not None:
        print(f"[INFO] Applying limit: {limit} images per class")

        targets = np.array(dataset.targets)
        classes = np.unique(targets)
        selected_indices = []

        for cls in classes:
            cls_indices = np.where(targets == cls)[0]
            if len(cls_indices) > limit:
                # Deterministic selection (first N)
                selected_indices.extend(cls_indices[:limit])
            else:
                selected_indices.extend(cls_indices)

        selected_indices = sorted(selected_indices)
        dataset = Subset(dataset, selected_indices)

    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )  # workers=0 for safe CPU/debugging
    print(f"[INFO] Dataset Size: {len(dataset)} images")

    # 4. Evaluate
    print("[INFO] Starting Evaluation...")
    correct = 0
    correct_top5 = 0
    total = 0
    detailed_predictions = []

    # Track class-wise stats
    class_correct = {}
    class_correct_top5 = {}
    class_total = {}

    with torch.no_grad():
        for i, (images, labels) in enumerate(dataloader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            # Helper for confidence
            probs = torch.softmax(outputs, dim=1)

            # Top-1
            _, predicted = torch.max(outputs.data, 1)

            # Top-5
            top5_probs, top5_preds = torch.topk(probs, 5, dim=1)

            # Collect Batch Statistics
            batch_correct = predicted == labels

            # Top-5 Correct
            # labels: (B), top5_preds: (B, 5)
            # expand labels to (B, 1) then compare
            expanded_labels = labels.view(-1, 1).expand_as(top5_preds)
            batch_correct_top5 = (
                (top5_preds == expanded_labels).sum(dim=1).bool()
            )  # (B)

            total += labels.size(0)
            correct += batch_correct.sum().item()
            correct_top5 += batch_correct_top5.sum().item()

            # Detailed Output & Class Stats

            # Helper to get original index
            def get_original_path(current_idx_in_batch, batch_start_idx):
                idx_in_dataloader = batch_start_idx + current_idx_in_batch
                if isinstance(dataset, Subset):
                    original_idx = dataset.indices[idx_in_dataloader]
                    return dataset.dataset.image_paths[original_idx]
                else:
                    return dataset.image_paths[idx_in_dataloader]

            start_idx = i * batch_size

            for j in range(len(labels)):
                pred_cls = predicted[j].item()
                true_cls = labels[j].item()
                is_correct = bool(batch_correct[j].item())
                is_correct_top5 = bool(batch_correct_top5[j].item())

                # Update class stats
                if true_cls not in class_total:
                    class_total[true_cls] = 0
                    class_correct[true_cls] = 0
                    class_correct_top5[true_cls] = 0
                class_total[true_cls] += 1
                if is_correct:
                    class_correct[true_cls] += 1
                if is_correct_top5:
                    class_correct_top5[true_cls] += 1

                # Get Path (No Type Extraction)
                try:
                    path = str(get_original_path(j, start_idx))
                except Exception as e:
                    path = "N/A"

                # User requested removing type logic
                # img_type = "sketch" # Or ignore field

                entry = {
                    "image_path": path,
                    # "image_type": img_type, # Removed as per request
                    "target_class_idx": true_cls,
                    "predicted_class_idx": pred_cls,
                    "correct": is_correct,
                    "correct_top5": is_correct_top5,
                }

                # Confidence reporting (always)
                pred_conf = probs[j, pred_cls].item()  # Top-1 confidence
                true_conf = probs[j, true_cls].item()  # Confidence of correct class
                entry["confidence"] = true_conf
                entry["top1_confidence"] = pred_conf
                entry["confidence_diff"] = pred_conf - true_conf

                detailed_predictions.append(entry)

            if (i + 1) % 10 == 0:
                print(
                    f"Image {total} - Current Acc: {100 * correct / total:.2f}% (Top-5: {100 * correct_top5 / total:.2f}%)"
                )

    accuracy = 100 * correct / total if total > 0 else 0.0
    accuracy_top5 = 100 * correct_top5 / total if total > 0 else 0.0

    print(
        f"\n[RESULT] Final Accuracy on evaluated samples: {accuracy:.2f}% ({correct}/{total})"
    )
    print(
        f"[RESULT] Final Top-5 Accuracy: {accuracy_top5:.2f}% ({correct_top5}/{total})"
    )

    # Calculate Class-wise Accuracies
    class_accuracies = {}
    class_accuracies_top5 = {}

    for cls in class_total:
        acc = 100 * class_correct[cls] / class_total[cls]
        acc5 = 100 * class_correct_top5[cls] / class_total[cls]
        class_accuracies[int(cls)] = acc  # ensure int key for json
        class_accuracies_top5[int(cls)] = acc5

    # 5. Save Results
    results = {
        "experiment_id": experiment_id,
        "dataset": "imagenet-r",
        "accuracy": accuracy,
        "accuracy_top5": accuracy_top5,
        "class_accuracies": class_accuracies,
        "class_accuracies_top5": class_accuracies_top5,
        "num_samples": total,
        "timestamp": timestamp,
        "predictions": detailed_predictions,
    }

    output_file = output_base_dir / "ood_results_detailed.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)

    print(f"[INFO] Detailed results saved to {output_file}")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Run OOD Analysis (ImageNet-R) for a given experiment."
    )
    parser.add_argument(
        "--experiment-id", type=str, required=True, help="Experiment ID or path"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size (default: 32)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of images per class (default: None)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use 'cpu' or 'cuda' (default: auto-select)",
    )

    args = parser.parse_args()

    run_ood_analysis(args.experiment_id, args.batch_size, args.limit, args.device)


if __name__ == "__main__":
    main()
