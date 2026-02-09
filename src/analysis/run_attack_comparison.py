#!/Users/donyin/miniconda3/envs/imperial/bin/python

"""
Script to compare performance of FGSM, Jitter, PGD, SPSA, OnePixel attacks.
"""

import matplotlib.pyplot as plt
import json
import torch
import torchattacks
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.analysis.run_loader import RunLoader
from src.training.dataset_select import get_dataset_obj
from src.utils.device import select_device
from src.utils.normalization import (
    Normalize,
    denormalize_batch,
    IMAGENET_MEAN,
    IMAGENET_STD,
)


DEFAULT_ATTACK_PARAMS = {
    "APGD": {"steps": 100},
    "Jitter": {"steps": 10, "scale": 10},
    "SPSA": {"nb_iter": 40, "delta": 0.001, "lr": 0.001, "max_batch_size": 2},
    "OnePixel": {"steps": 100, "popsize": 400},
    "FGSM": {} # No internal params besides eps
}

def resolve_attack_params(attack_name, user_params, batch_size):
    """
    Resolves parameters for attacks by merging user_params with defaults.
    """
    defaults = DEFAULT_ATTACK_PARAMS.get(attack_name, {}).copy()
    
    # Update defaults with any user provided params
    # Note: dict.update updates in place
    params = defaults
    params.update(user_params)
    
    if attack_name == "SPSA":
        # SPSA has dynamic default for nb_sample based on batch_size argument
        if "nb_sample" not in params:
             params["nb_sample"] = batch_size
    
    return params


def get_attack(
    model, attack_name, epsilon, device, batch_size=32, attack_params=None
):
    """
    Factory function to get torchattacks object.
    
    Args:
        model: PyTorch model
        attack_name: Name of the attack
        epsilon: perturbation budget
        device: torch device
        batch_size: Batch size for attacks that need it (SPSA, OnePixel)
        attack_params (dict): Optional dictionary of parameter overrides.
    """
    # Merge provided params with defaults
    # Logic: Defaults -> attack_params (overrides)
    final_params = resolve_attack_params(attack_name, attack_params or {}, batch_size)

    if attack_name == "FGSM":
        return torchattacks.FGSM(model, eps=epsilon)
        
    elif attack_name == "APGD":
        return torchattacks.APGD(model, eps=epsilon, steps=final_params["steps"])
        
    elif attack_name == "Jitter":
        return torchattacks.Jitter(
            model, 
            eps=epsilon, 
            alpha=epsilon / final_params["steps"], 
            steps=final_params["steps"], 
            scale=final_params["scale"]
        )
        
    elif attack_name == "SPSA":
        return torchattacks.SPSA(
            model,
            eps=epsilon,
            delta=final_params["delta"],
            lr=final_params["lr"],
            nb_iter=final_params["nb_iter"],
            nb_sample=final_params["nb_sample"],
            max_batch_size=final_params["max_batch_size"],
        )
        
    elif attack_name == "OnePixel":
        pixels = int(max(1, np.floor(epsilon)))
        return torchattacks.OnePixel(
            model, 
            pixels=pixels, 
            steps=final_params["steps"], 
            popsize=final_params["popsize"], 
            inf_batch=batch_size
        )
    else:
        raise NotImplementedError(f"Attack {attack_name} not implemented.")



def get_epsilon_list(attack_name):
    """
    Get the list of epsilons for a given attack.
    Source of truth for consistency across comparison and generation.
    """
    if attack_name == "FGSM":
        # 0 to 0.2
        return np.linspace(0, 0.2, 20).tolist()
    elif attack_name == "APGD":
        # 0 to 0.01
        return np.linspace(0, 0.01, 20).tolist()
    elif attack_name == "Jitter":
        # 0 to 0.02
        return np.linspace(0, 0.02, 20).tolist()
    elif attack_name == "SPSA":
        # 0 to 0.05
        return np.linspace(0, 0.05, 20).tolist()
    elif attack_name == "OnePixel":
        # [1, 42]. Sample integer list.
        return np.linspace(0, 42, 20).tolist()
    else:
        raise ValueError(f"Unknown attack: {attack_name}")


def run_attack_comparison(
    experiment_id,
    batch_size=16,
    attacks=None,
    viz_filter=None,
    resume=False,
    attack_params_dict=None,
    max_samples=None,
    seed=None,
):
    """
    Run adversarial attack comparison for a given experiment.

    Args:
        experiment_id (int or str): Experiment ID.
        batch_size (int): Batch size.
        attacks (list, optional): List of attacks. Defaults to ["FGSM", "Jitter", "PGD", "SPSA", "OnePixel"].
        viz_filter (str, optional): Filter dataset by string.
        resume (bool): If True, skip epsilons already found in output JSON.
        max_samples (int, optional): Limit number of samples.
    """
    # 1. Setup
    import datetime

    if isinstance(experiment_id, int) or (
        isinstance(experiment_id, str) and experiment_id.isdigit()
    ):
        exp_path = f"__local__/experiment-{experiment_id}/000000"
    else:
        exp_path = experiment_id

    # Timestamped output
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_base_dir = Path(exp_path) / "results" / "attack_comparison" / timestamp

    if viz_filter:
        safe_name = Path(viz_filter).stem if "." in viz_filter else viz_filter
        output_base_dir = output_base_dir / f"viz_{safe_name}"

    output_base_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results will be saved to: {output_base_dir}")

    device = select_device()
    print(f"Using device: {device}")

    if attack_params_dict is None:
        attack_params_dict = {}

    # 2. Load Model
    print(f"Loading Experiment from {exp_path}...")
    try:
        loader = RunLoader(exp_path)
    except Exception as e:
        print(f"Error loading experiment from {exp_path}: {e}")
        return

    model = loader.model
    # --- Normalization Wrapper ---
    # Wrap model to accept [0, 1] inputs and normalize internally
    normalization = Normalize(IMAGENET_MEAN, IMAGENET_STD).to(device)
    model = torch.nn.Sequential(normalization, model)

    model.eval()
    model.to(device)

    # 3. Load Dataset
    print(f"Loading Validation Dataset...")
    dataset = get_dataset_obj("imagenet100", "VAL")

    if viz_filter:
        print(f"Filtering dataset for string '{viz_filter}'...")
        found_indices = []
        if hasattr(dataset, "image_paths"):
            for idx, p in enumerate(dataset.image_paths):
                if viz_filter in str(p):
                    found_indices.append(idx)

        if found_indices:
            print(
                f"Found {len(found_indices)} images matching '{viz_filter}'. Using subset."
            )
            from torch.utils.data import Subset

            dataset = Subset(dataset, found_indices)
        else:
            print(
                f"Warning: No images found for filter '{viz_filter}'. Using full dataset."
            )

    if max_samples is not None and len(dataset) > max_samples:
        print(f"Limiting dataset to {max_samples} samples (Seed: {seed}, Balanced: True).")
        from torch.utils.data import Subset
        
        if seed is not None:
             torch.manual_seed(seed)
             # Balanced sampling
             # 1. Get targets
             if hasattr(dataset, "targets") and isinstance(dataset.targets, (list, np.ndarray)):
                 targets = np.array(dataset.targets)
             elif hasattr(dataset, "image_labels_str"):
                 # ImageNet100Dataset specific
                 unique_labels = sorted(list(set(dataset.image_labels_str)))
                 label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
                 targets = np.array([label_to_idx[l] for l in dataset.image_labels_str])
             elif hasattr(dataset, "labels") and isinstance(dataset.labels, (list, np.ndarray)):
                targets = np.array(dataset.labels)
             else:
                 # Try to iterate (slow but safe fallback)
                 print("Warning: Dataset targets not directly accessible. Iterating to find targets...")
                 targets = []
                 for _, label in dataset:
                     targets.append(label)
                 targets = np.array(targets)
             
             unique_classes = np.unique(targets)
             n_classes = len(unique_classes)
             samples_per_class = max_samples // n_classes
             remainder = max_samples % n_classes
             
             selected_indices = []
             
             for cls_idx in unique_classes:
                 cls_indices = np.where(targets == cls_idx)[0]
                 # Shuffle indices for this class
                 shuffled_cls = cls_indices[torch.randperm(len(cls_indices)).numpy()]
                 
                 # Take N samples
                 count = samples_per_class
                 if remainder > 0:
                     count += 1
                     remainder -= 1
                 
                 selected_indices.extend(shuffled_cls[:count])
             
             selected_indices.sort() # Keep order
             dataset = Subset(dataset, selected_indices)
        else:
             # Default simple subset
             dataset = Subset(dataset, range(max_samples))

    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )
    print(f"Dataset Size: {len(dataset)}")

    # 4. Define Attacks and Parameters
    if attacks is None:
        attacks_to_run = ["FGSM", "Jitter", "APGD", "SPSA", "OnePixel"]
    else:
        attacks_to_run = attacks

    results = {}

    # 5. Run Attacks
    output_file = output_base_dir / "attack_comparison_results.json"

    # Load existing results if they exist to allow rudimentary resuming/monitoring without dataloss
    if output_file.exists():
        try:
            with open(output_file, "r") as f:
                results = json.load(f)
            print(f"Loaded existing results from {output_file}")
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Could not load existing results ({e}), starting fresh.")
            print(f"Could not load existing results ({e}), starting fresh.")
            results = {}

    # Update metadata
    if "metadata" not in results:
        results["metadata"] = {}
    results["metadata"]["n_samples"] = len(dataset)
    results["metadata"]["dataset_name"] = "imagenet100"
    results["metadata"]["experiment_id"] = str(experiment_id)
    results["metadata"]["seed"] = seed
    results["metadata"]["max_samples"] = max_samples
    results["metadata"]["batch_size"] = batch_size
    results["metadata"]["viz_filter"] = viz_filter
    results["metadata"]["attacks"] = attacks_to_run
    
    # Collect and log attack parameters upfront
    attack_params_log = {}
    for attack_name in attacks_to_run:
         user_params = attack_params_dict.get(attack_name, {})
         attack_params_log[attack_name] = resolve_attack_params(attack_name, user_params, batch_size)
    results["metadata"]["attack_parameters"] = attack_params_log

    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)

    for attack_name in attacks_to_run:
        print(f"\nRunning Attack: {attack_name}...")

        # Define ranges per attack
        epsilons = get_epsilon_list(attack_name)
        # Initialize result structure if not present
        if attack_name not in results:
            # Resolve parameters for logging (already done for metadata, copy here)
            resolved_params = resolve_attack_params(attack_name, attack_params_dict.get(attack_name, {}), batch_size)
            
            results[attack_name] = {
                "epsilons": [], 
                "accuracies": [],
                "parameters": resolved_params
            }

        current_epsilons = results[attack_name]["epsilons"]
        current_accuracies = results[attack_name]["accuracies"]

        if not resume:
            # If not resuming, we overwrite this attack's results?
            # Or do we want to keep them but re-run?
            # User asked for "continue flag such that it checks... and skips"
            # Implicitly if resume=False (default), it should probably re-run or complain?
            # Standard behavior: overwrite if not resuming.
            current_epsilons = []
            current_accuracies = []

        for eps in tqdm(epsilons, desc=f"{attack_name} Epsilons"):
            # Check if this epsilon is already processed
            # Use approximation for float comparison
            already_done = False
            if resume:
                for existing_eps in current_epsilons:
                    if abs(existing_eps - eps) < 1e-6:
                        already_done = True
                        break

            if already_done:
                # print(f"Skipping eps={eps:.4f} (already done)")
                continue

            if eps == 0:
                # Clean accuracy
                correct = 0
                total = 0
                with torch.no_grad():
                    for images, labels in dataloader:
                        # images from loader are Normalized.
                        # Denormalize to [0,1] for cleaner verification of wrapper model
                        images, labels = images.to(device), labels.to(device)
                        images = denormalize_batch(images, device)
                        images = torch.clamp(images, 0, 1)

                        outputs = model(images)
                        _, predicted = outputs.max(1)
                        total += labels.size(0)
                        correct += predicted.eq(labels).sum().item()
                acc = correct / total
                current_accuracies.append(acc)
                current_epsilons.append(eps)

                # Update and Save
                results[attack_name]["epsilons"] = current_epsilons
                results[attack_name]["accuracies"] = current_accuracies
                with open(output_file, "w") as f:
                    json.dump(results, f, indent=4)
                continue

            attack = get_attack(
                model,
                attack_name,
                eps,
                device,
                batch_size=batch_size,
                attack_params=attack_params_dict.get(attack_name, {}),
            )

            correct = 0
            total = 0

            # Using torchattacks: attack(images, labels) -> adversarial_images
            for images, labels in dataloader:
                images, labels = images.to(device), labels.to(device)

                # Denormalize to [0, 1] before attack
                images = denormalize_batch(images, device)
                images = torch.clamp(images, 0, 1)

                try:
                    adv_images = attack(images, labels)
                    with torch.no_grad():
                        outputs = model(adv_images)
                        _, predicted = outputs.max(1)
                        total += labels.size(0)
                        correct += predicted.eq(labels).sum().item()
                except Exception as e:
                    print(f"Error in batch for {attack_name} eps={eps}: {e}")
                    pass

            acc = correct / total if total > 0 else 0
            current_accuracies.append(acc)
            current_epsilons.append(eps)

            # Update and Save Incremental
            results[attack_name]["epsilons"] = current_epsilons
            results[attack_name]["accuracies"] = current_accuracies
            with open(output_file, "w") as f:
                json.dump(results, f, indent=4)

        # Sort results by epsilon just in case they got out of order due to incomplete resumes (though typically linear)
        # But we loop linearly, so appending should be fine.
        print(
            f"{attack_name} Results: {list(zip(current_epsilons, current_accuracies))}"
        )

    print(f"\nFinal results saved to {output_file}")

    # 7. Plot Comparison (New Row Implementation)
    try:
        from plotting import plot_attack_comparison

        plot_path = output_base_dir / "attack_comparison_plot.png"
        plot_attack_comparison(results, experiment_id, output_path=plot_path)
    except ImportError:
        print(
            "Warning: Could not import plot_attack_comparison from plotting. Is plotting.py in path?"
        )
    except Exception as e:
        print(f"Error plotting results: {e}")


if __name__ == "__main__":
    # Internal logic for standalone run
    run_attack_comparison(15, batch_size=16)


def generate_attack_examples(experiment_id, image_path, output_dir=None, attacks=None):
    """
    Generate adversarial examples for a specific image across all attacks and epsilons.
    """
    from PIL import Image
    from torchvision import transforms
    from torchvision.utils import save_image
    import os

    # 1. Setup
    if isinstance(experiment_id, int) or (
        isinstance(experiment_id, str) and experiment_id.isdigit()
    ):
        exp_path = f"__local__/experiment-{experiment_id}/000000"
    else:
        exp_path = experiment_id

    device = select_device()
    print(f"Using device: {device}")

    # 2. Output Directory
    if output_dir is None:
        import datetime

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = (
            Path(exp_path)
            / "results"
            / "attack_comparison"
            / "attack_examples"
            / timestamp
        )
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving examples to: {output_dir}")

    # 3. Load Model with Normalization
    print(f"Loading Experiment from {exp_path}...")
    loader = RunLoader(exp_path)
    model = loader.model
    normalization = Normalize(IMAGENET_MEAN, IMAGENET_STD).to(device)
    model = torch.nn.Sequential(normalization, model)
    model.eval()
    model.to(device)

    # 4. Load Image
    print(f"Loading Example Image: {image_path}")
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return

    img_pil = Image.open(image_path).convert("RGB")
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224), antialias=True),
            transforms.ToTensor(),  # [0, 1]
        ]
    )
    image_tensor = transform(img_pil).unsqueeze(0).to(device)

    # Save original
    save_image(image_tensor, output_dir / "original.png")

    # 5. Define Attacks
    if attacks is None:
        attacks_to_run = ["FGSM", "Jitter", "APGD", "SPSA", "OnePixel"]
    else:
        attacks_to_run = attacks

    # 6. Run Generation
    for attack_name in attacks_to_run:
        print(f"Generating examples for {attack_name}...")

        # Ranges (Copied from run_attack_comparison to ensure consistency)
        try:
            epsilons = get_epsilon_list(attack_name)
        except ValueError:
            print(f"Warning: Unknown attack {attack_name}, skipping.")
            continue

        att_dir = output_dir / attack_name
        att_dir.mkdir(exist_ok=True)

        for eps in tqdm(epsilons, desc=f"{attack_name}"):
            if eps == 0:
                # Just save original again as eps_0
                save_image(image_tensor, att_dir / f"eps_{eps:.5f}.png")
                continue

            attack = get_attack(model, attack_name, eps, device, batch_size=32)
            try:
                # image_tensor is [0, 1] and model expects it (via wrapper)
                # But get_attack returns an attack that expects [0, 1] input?
                # Yes, torchattacks wrappers usually expect [0,1].
                # Model wrapper handles normalization.

                adv_image = attack(
                    image_tensor, torch.tensor([0]).to(device)
                )  # Dummy label, generic generation?
                # Wait, generic generation needs a target label usually for untargeted attack to move away from.
                # Use model prediction as label
                with torch.no_grad():
                    pred_label = model(image_tensor).max(1)[1]

                adv_image = attack(image_tensor, pred_label)

                save_image(adv_image, att_dir / f"eps_{eps:.5f}.png")
            except Exception as e:
                print(f"Error generating {attack_name} eps {eps}: {e}")
