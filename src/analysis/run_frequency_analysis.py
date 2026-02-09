import torch
import torch.nn as nn
import torch.fft
import numpy as np
import torch.fft
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from scipy import ndimage
from torch.utils.data import DataLoader, ConcatDataset

from src.analysis.run_loader import RunLoader
from src.training.dataset_select import get_dataset_obj
from src.utils.device import select_device
from src.modules.layers.flex import Flex2D


def get_layer_name(module, idx):
    if isinstance(module, (nn.Conv2d, Flex2D)):
        return f"Layer_{idx}_{module.__class__.__name__}"
    return f"Layer_{idx}"


def run_frequency_analysis(experiment_id, batch_size=32, device=None):
    """
    Computes and plots the average 2D Fourier Transform (Frequency Spectrum)
    of layer activations across Training and Validation sets.
    """
    if device is None:
        device = select_device()
    print(f"Using device: {device}")

    # 1. Load Experiment
    if isinstance(experiment_id, int) or (
        isinstance(experiment_id, str) and experiment_id.isdigit()
    ):
        exp_path = f"__local__/experiment-{experiment_id}/000000"
    else:
        exp_path = experiment_id

    print(f"Loading Experiment from {exp_path}...")
    try:
        loader = RunLoader(exp_path)
    except Exception as e:
        print(f"Error loading experiment: {e}")
        return

    model = loader.model.to(device)
    model.eval()

    # 2. Output Directory
    output_dir = Path(exp_path) / "results" / "frequency_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 3. Register Hooks
    # We want to capture activations from all Conv/Flex layers
    # We will assume 'features' Sequential block for VGG
    activations = {}
    accumulators = {}  # To store sum of magnitudes
    counts = (
        {}
    )  # To store count of samples processed per layer (might vary if we want per-channel avg? No, just N images)

    def get_activation_hook(name):
        def hook(model, input, output):
            # output shape: [N, C, H, W]
            with torch.no_grad():
                # Compute FFT over spatial dims (H, W) -> dims (-2, -1)
                # fftn computes n-dim transform. We want 2D spatial.
                # output is real-valued.

                # Check for NaNs
                if torch.isnan(output).any():
                    return

                # 1. FFT
                fft_out = torch.fft.fftn(output.float(), dim=(-2, -1))
                # 2. Shift zero freq to center
                fft_shifted = torch.fft.fftshift(fft_out, dim=(-2, -1))
                # 3. Magnitude
                magnitude = torch.abs(fft_shifted)
                # 4. Average over Batch (N) and Channels (C) ?
                # The user wants "frequency bias for each layer".
                # Usually this means we average over N and C to get a single 2D heatmap per layer.
                # Or we might want to keep C?
                # "Vectorizing" usually implies aggregating C. Let's aggregate C for a compact summary.

                avg_mag_batch_channel = magnitude.mean(dim=(0, 1))  # [H, W]

                # Accumulate
                # Need to handle varying H, W?
                # Hooks are defined per layer, H/W is constant for a fixed input size.
                if name not in accumulators:
                    accumulators[name] = torch.zeros_like(
                        avg_mag_batch_channel, device=device
                    )
                    counts[name] = 0

                # We are averaging over N inside the batch already.
                # To compute global average:
                # Global_Avg = (Sum of Batch_Avgs * Batch_Size) / Total_Images
                # Wait, sum(Batch_Mean * B) is sum of all single image means.
                # Let's just accumulate the sum over N and C.

                # Sum over batch dim (0) and channel dim (1)
                sum_mag = magnitude.sum(dim=(0, 1))  # [H, W]
                accumulators[name] += sum_mag
                counts[name] += (
                    output.shape[0] * output.shape[1]
                )  # Total samples = N * C

        return hook

    hooks = []
    layer_names = []

    print("Registering hooks on Conv/Flex layers...")
    layer_counter = 0
    if hasattr(model, "features"):
        for module in model.features:
            if isinstance(module, (nn.Conv2d, Flex2D)):
                name = f"Layer_{layer_counter}_{module.__class__.__name__}"
                layer_names.append(name)
                h = module.register_forward_hook(get_activation_hook(name))
                hooks.append(h)
                print(f"Hooked {name}")
                layer_counter += 1
    else:
        # Fallback for models without .features
        print("Model does not have 'features' attribute. Hooking all named modules...")
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, Flex2D)):
                # Use counter instead of module name if desired, or combine
                # User asked for "order of conv/pool count", so let's enforce counter
                name = f"Layer_{layer_counter}_{module.__class__.__name__}"
                layer_names.append(name)
                h = module.register_forward_hook(get_activation_hook(name))
                hooks.append(h)
                print(f"Hooked {name}")
                layer_counter += 1

    if not hooks:
        print("No Conv/Flex layers found to hook.")
        return

    # 4. Load Datasets
    print("Loading Datasets...")
    try:
        # Train and Val
        ds_train = get_dataset_obj("imagenet100", "TRAIN")
        ds_val = get_dataset_obj("imagenet100", "VAL")

        # Combine? Or run sequentially?
        # User said "training+val".
        full_dataset = ConcatDataset([ds_train, ds_val])
        print(f"Total images (Train+Val): {len(full_dataset)}")

        loader = DataLoader(
            full_dataset, batch_size=batch_size, shuffle=False, num_workers=4
        )

    except Exception as e:
        print(f"Error loading datasets: {e}")
        return

    # 5. Run Pass
    print("Running forward pass on dataset...")
    with torch.no_grad():
        for batch_idx, (images, _) in enumerate(tqdm(loader)):
            images = images.to(device)
            model(images)

            # Optional: Clear intermediate memory if needed, but we calculate in hook.
            # Just ensure we don't store the full outputs.

    # 6. Compute Averages and Plot
    print("\nComputing averages and generating plots...")

    # Save aggregated data
    results = {}

    for name in layer_names:
        if name in accumulators:
            total_sum = accumulators[name]
            count = counts[name]

            # Average Magnitude
            avg_spectrum = total_sum / count

            # Move to CPU for plotting/saving
            avg_spectrum_np = avg_spectrum.cpu().numpy()
            results[name] = avg_spectrum_np

            # Log Magnitude for visualization (add small epsilon)
            log_spectrum = np.log(avg_spectrum_np + 1e-8)

            # --- 1. First Quadrant (Positive Frequencies) ---
            h, w = log_spectrum.shape
            cy, cx = h // 2, w // 2
            # Extract bottom-right quadrant (assuming fftshift puts 0,0 at center)
            # Indices [cy:, cx:] correspond to frequencies >= 0
            quadrant = log_spectrum[cy:, cx:]

            fig1, ax1 = plt.subplots(figsize=(6, 6))
            im1 = ax1.imshow(
                quadrant, cmap="inferno", origin="upper"
            )  # upper/lower depends on preference, usually image coordinates
            ax1.set_title(f"{name}: 1st Quadrant (Log Mag)")
            ax1.set_xlabel("Freq X (+)")
            ax1.set_ylabel("Freq Y (+)")
            fig1.colorbar(im1, ax=ax1)

            plot_path_2d = output_dir / f"{name}_freq_quadrant.png"
            plt.savefig(plot_path_2d, bbox_inches="tight", dpi=150)
            plt.close(fig1)

            # --- 2. Radial Profile (Collapse Phases / Sum Power) ---
            # Compute radial average of the Magnitude (Linear scale, then maybe log for plot?)
            # or Radial Sum? User said "Sum to show power". Usually "Power vs Freq".
            # Let's do Radial Mean of Log Magnitude or Log of (Radial Mean of Magnitude).
            # The spectrum is currently Average Magnitude.
            # Let's compute Radial Mean of the Log Spectrum (which we visualized).

            # Create coordinate grid for distance from center
            y, x = np.indices((h, w))
            r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
            r = r.astype(int)

            # Bin by radius
            tbin = np.bincount(r.ravel(), log_spectrum.ravel())
            nr = np.bincount(r.ravel())
            radialprofile = tbin / nr

            # Plot 1D
            fig2, ax2 = plt.subplots(figsize=(8, 4))
            ax2.plot(radialprofile, color="purple", linewidth=2)
            ax2.set_title(f"{name}: Radial Frequency Profile")
            ax2.set_xlabel("Spatial Frequency (Radius)")
            ax2.set_ylabel("Log Magnitude")
            ax2.grid(True, alpha=0.3)

            plot_path_1d = output_dir / f"{name}_freq_radial.png"
            plt.savefig(plot_path_1d, bbox_inches="tight", dpi=150)
            plt.close(fig2)

            print(f"Saved plots to {plot_path_2d} and {plot_path_1d}")

    # Save raw data (NPZ)
    npz_path = output_dir / "frequency_analysis_data.npz"
    np.savez(npz_path, **results)
    print(f"Saved raw data to {npz_path}")

    # Cleanup hooks
    for h in hooks:
        h.remove()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Frequency Analysis")
    parser.add_argument(
        "--experiment-id", type=str, required=True, help="Experiment ID"
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--device", type=str, default=None, help="Device (cpu/cuda)")

    args = parser.parse_args()

    run_frequency_analysis(
        args.experiment_id, batch_size=args.batch_size, device=args.device
    )
