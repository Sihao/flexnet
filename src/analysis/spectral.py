import matplotlib

matplotlib.use("Agg")
import torch
import numpy as np
import random
from torch.utils.data import DataLoader, Subset

from src.training.dataset_select import get_dataset_obj
from src.utils.device import select_device
from src.utils.spectral_utils import (
    load_experiment_model,
    get_radial_profile,
    compute_slope,
    get_target_layer,
)
from plotting import plot_spectral_results


def analyze_layer_spectral_slope(
    experiment_id,
    layer_idx,
    num_images=100,
    batch_size=100,
    device=None,
    dataset_name="imagenet100",
    split="TRAIN",
    ylim_max=None,
):
    """
    Main functional entry point for the spectral slope analysis.

    Args:
        experiment_id (str/int): Experiment ID or path.
        layer_idx (int): Index of the Conv/Pool layer to analyze.
        num_images (int): Number of images to process.
        batch_size (int): Batch size for DataLoader.
        device (str, optional): Computation device ('cpu', 'cuda').
        dataset_name (str): Name of the dataset to load.
        split (str): Dataset split ('TRAIN', 'VAL').
        ylim_max (int, optional): Manual override for plot Y-limit.

    Returns:
        tuple: (mean_slope, std_slope, all_slopes)
    """
    if device is None:
        device = select_device()
    else:
        # ensure it's a string or torch device
        pass

    print(
        f"Running Spectral Slope Analysis on Experiment {experiment_id}, Layer Index {layer_idx} (Device: {device})..."
    )

    # 1. Load Model
    model, base_output_dir = load_experiment_model(experiment_id, device)

    # 2. Identify Target Layer
    layer, layer_name = get_target_layer(model, layer_idx)
    print(f"Target Layer: {layer_name} ({layer.__class__.__name__})")

    # Create timestamped output directory
    import datetime

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = base_output_dir / f"{layer_name}" / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results will be saved to: {output_dir}")

    # 3. Setup Hook
    activations = []

    def hook(model, input, output):
        activations.append(output.detach())

    handle = layer.register_forward_hook(hook)

    # 4. Prepare Data
    ds = get_dataset_obj(dataset_name, split)
    if len(ds) < num_images:
        num_images = len(ds)
        print(f"Dataset too small, using all {num_images} images.")

    indices = random.sample(range(len(ds)), num_images)
    subset = Subset(ds, indices)
    dl = DataLoader(subset, batch_size=batch_size, shuffle=False)

    all_slopes = []
    all_profiles = []

    print(f"Processing {num_images} images...")

    # 5. Execution Loop

    # Store detailed results
    # List of tuples: (image_path, layer_name, channel_idx, slope)
    detailed_results = []

    current_idx = 0

    with torch.no_grad():
        for b_i, (images, _) in enumerate(dl):
            batch_size_curr = images.shape[0]
            images = images.to(device)
            model(images)

            if not activations:
                current_idx += batch_size_curr
                continue

            # shape: [B, C, H, W]
            batch_acts = activations[-1]
            activations.clear()

            # Identify image paths for this batch
            batch_global_indices = indices[current_idx : current_idx + batch_size_curr]
            batch_paths = []
            for g_idx in batch_global_indices:
                if hasattr(ds, "image_paths"):
                    batch_paths.append(str(ds.image_paths[g_idx]))
                elif hasattr(ds, "samples"):
                    # ImageFolder standard
                    batch_paths.append(str(ds.samples[g_idx][0]))
                else:
                    batch_paths.append(f"img_idx_{g_idx}")

            current_idx += batch_size_curr

            # Iterate per image
            for img_i in range(batch_size_curr):
                img_act = batch_acts[img_i]  # [C, H, W]
                img_path = batch_paths[img_i]

                # Compute FFT for all channels of this image efficiently?
                # or just loop over channels
                # Keeping it simple loop over channels to store metadata
                # Optim: Batched FFT per image [C, H, W] -> [C, H, W]

                fft = torch.fft.fftn(img_act.float(), dim=(-2, -1))
                fft_shifted = torch.fft.fftshift(fft, dim=(-2, -1))
                power = torch.abs(fft_shifted) ** 2  # [C, H, W]

                power_np = power.cpu().numpy()
                h, w = power_np.shape[1], power_np.shape[2]
                nyquist = min(h, w) // 2

                for ch_j in range(power_np.shape[0]):
                    if np.isnan(power_np[ch_j]).any():
                        continue

                    profile = get_radial_profile(power_np[ch_j])
                    all_profiles.append(profile)

                    slope, _, _, _ = compute_slope(profile, nyquist)

                    if not np.isnan(slope):
                        all_slopes.append(slope)
                        detailed_results.append((img_path, layer_name, ch_j, slope))

    handle.remove()

    # 6. Finalize
    if not all_slopes:
        print("No valid slopes calculated (all NaNs).")
        return None, None, []

    mean_slope = np.mean(all_slopes)
    std_slope = np.std(all_slopes)

    print(
        f"Mean Slope: {mean_slope:.2f} +/- {std_slope:.2f} (Computed on {len(all_slopes)} valid samples)"
    )

    # Save slopes to CSV with metadata
    slopes_file = output_dir / f"slopes_{layer_name}_{len(all_slopes)}samples.csv"

    import csv

    with open(slopes_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["input_image", "layer", "channel", "slope"])
        writer.writerows(detailed_results)

    print(f"Saved slopes to {slopes_file}")

    # Save all_profiles
    profiles_file = output_dir / f"spectra_{layer_name}_{len(all_slopes)}samples.npy"
    np.save(profiles_file, np.array(all_profiles))
    print(f"Saved spectra to {profiles_file}")

    # Save metadata
    metadata = {
        "mean_slope": float(mean_slope),
        "std_slope": float(std_slope),
        "nyquist": int(nyquist),
        "num_samples": len(all_slopes),
        "layer_name": layer_name,
    }
    metadata_file = output_dir / f"metadata_{layer_name}_{len(all_slopes)}samples.json"
    import json

    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"Saved metadata to {metadata_file}")

    # Return dictionary of paths
    paths = {
        "slopes": slopes_file,
        "spectra": profiles_file,
        "metadata": metadata_file,
        "output_dir": output_dir,
    }

    return paths, layer_name


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run Spectral Slope Analysis")
    parser.add_argument(
        "--experiment", type=str, required=True, help="Experiment ID or Path"
    )
    parser.add_argument(
        "--layer", type=int, required=True, help="Index of Conv/Pool layer"
    )
    parser.add_argument("--images", type=int, default=128, help="Number of images")
    parser.add_argument("--device", type=str, default=None, help="Device (cpu/cuda)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument(
        "--ylim-max", type=int, default=None, help="Y-axis limit for slope histogram"
    )

    args = parser.parse_args()

    analyze_layer_spectral_slope(
        args.experiment,
        args.layer,
        num_images=args.images,
        batch_size=args.batch_size,
        device=args.device,
        ylim_max=args.ylim_max,
    )


if __name__ == "__main__":
    main()
