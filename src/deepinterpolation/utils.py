import numpy as np
import tifffile
from pathlib import Path


def compute_pixel_snr(tiff_input, epsilon=1e-8):
    """
    Computes the per-pixel Signal-to-Noise Ratio (SNR) for a given Tiff stack.

    SNR = Mean(time) / Std(time)

    Args:
        tiff_input (str or np.ndarray): Path to tiff file or numpy array (T, H, W).
        epsilon (float): Small constant to avoid division by zero.

    Returns:
        np.ndarray: SNR map of shape (H, W).
    """
    if isinstance(tiff_input, (str, Path)):
        stack = tifffile.imread(str(tiff_input))
    elif isinstance(tiff_input, np.ndarray):
        stack = tiff_input
    else:
        raise TypeError("tiff_input must be a file path or numpy array")

    if stack.ndim != 3:
        raise ValueError(f"Expected 3D stack (T, H, W), got {stack.ndim}D")

    T, H, W = stack.shape

    # Check for torchmetrics (lazy import if possible, but standard import here)
    try:
        import torch
        from torchmetrics.audio import SignalNoiseRatio
    except ImportError:
        raise ImportError("torchmetrics is required. Please install it.")

    # Prepare data for SignalNoiseRatio
    # It typically expects (Batch, Time). We treat each pixel as a batch item.
    # Reshape to (H*W, T)
    stack_flat = stack.reshape(T, -1).transpose(1, 0)  # (H*W, T)

    # Convert to Tensor
    preds = torch.from_numpy(stack_flat).float()  # Raw signal

    # Target: We define the "Signal" as the Mean over time
    mean_flat = np.mean(stack_flat, axis=1, keepdims=True)  # (H*W, 1)
    target = torch.from_numpy(mean_flat).float().repeat(1, T)  # (H*W, T)

    # Instantiate Metric
    # SignalNoiseRatio calculates 10*log10( ||target||^2 / ||preds-target||^2 )
    snr_metric = SignalNoiseRatio()

    # Compute
    # Note: torchmetrics usually reduces. We want per-pixel.
    # SignalNoiseRatio doesn't easily support 'none' reduction in functional API sometimes,
    # but the class updates state.
    # Let's check functional API.
    # functional.signal_noise_ratio(preds, target) returns a tensor of shape (Batch,)
    from torchmetrics.functional.audio import signal_noise_ratio

    snr_vals = signal_noise_ratio(preds, target)  # Result shape: (H*W,)

    # Reshape back to (H, W)
    snr_map = snr_vals.numpy().reshape(H, W)

    return snr_map


def compute_correlation_image(tiff_input, kernel_size=5, epsilon=1e-8):
    """
    Computes the local correlation image.
    For each pixel, computes the average correlation with its neighbors over time.

    Args:
        tiff_input (str or np.ndarray): Path or array (T, H, W).
        kernel_size (int): Size of the neighborhood (odd number, e.g. 3 for 3x3).
        epsilon (float): Small constant.

    Returns:
        np.ndarray: Correlation image (H, W).
    """
    import torch
    import torch.nn.functional as F

    if isinstance(tiff_input, (str, Path)):
        stack = tifffile.imread(str(tiff_input))
    elif isinstance(tiff_input, np.ndarray):
        stack = tiff_input
    else:
        raise TypeError("tiff_input must be a file path or numpy array")

    if stack.ndim != 3:
        raise ValueError(f"Expected 3D stack (T, H, W), got {stack.ndim}D")

    T, H, W = stack.shape

    # Check device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Convert to tensor (Treat T as Batch dimension for 2D conv)
    # Shape: (T, 1, H, W)
    tensor_stack = torch.from_numpy(stack).float().to(device).unsqueeze(1)

    # 1. Z-score normalize over time
    mean = tensor_stack.mean(dim=0, keepdim=True)  # (1, 1, H, W)
    std = tensor_stack.std(dim=0, keepdim=True)  # (1, 1, H, W)

    # Avoid div by zero
    z_stack = (tensor_stack - mean) / (std + epsilon)

    # 2. Define Kernel
    # shape (Out, In, H, W) -> (1, 1, K, K)
    assert kernel_size % 2 == 1, "Kernel size must be odd"
    kernel = torch.ones((1, 1, kernel_size, kernel_size), device=device)
    center = kernel_size // 2
    kernel[0, 0, center, center] = 0.0  # Zero out center

    # Normalize kernel by number of neighbors
    num_neighbors = (kernel_size * kernel_size) - 1
    kernel = kernel / num_neighbors

    # 3. Convolve
    # padding = center to keep size same
    neighbors_avg = F.conv2d(z_stack, kernel, padding=center)

    # 4. Compute Correlation
    # Correlation = Mean_t ( Z_pixel * Neighbors_avg )
    correlation_map = (z_stack * neighbors_avg).mean(dim=0).squeeze()  # (H, W)

    return correlation_map.detach().cpu().numpy()


if __name__ == "__main__":
    # Test script for verification
    import argparse
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description="DeepInterpolation Utils Test")
    parser.add_argument(
        "--test_snr", type=str, help="Path to tiff to test SNR computation"
    )
    parser.add_argument(
        "--test_corr", type=str, help="Path to tiff to test Correlation Image"
    )
    parser.add_argument(
        "--kernel_size", type=int, default=5, help="Kernel size for correlation image"
    )
    parser.add_argument(
        "--output_plot", type=str, default=None, help="Path to save plot"
    )

    args = parser.parse_args()

    if args.test_snr:
        print(f"Computing SNR for {args.test_snr}...")
        snr = compute_pixel_snr(args.test_snr)
        print(f"SNR Map Shape: {snr.shape}")
        print(
            f"SNR Stats: Min={snr.min():.4f}, Max={snr.max():.4f}, Mean={snr.mean():.4f}"
        )

        if args.output_plot:
            plt.figure(figsize=(6, 6))
            plt.imshow(snr, cmap="viridis")
            plt.colorbar(label="SNR")
            plt.title("Pixel-wise SNR (TorchMetrics)")
            plt.savefig(args.output_plot)
            print(f"Plot saved to {args.output_plot}")

    if args.test_corr:
        print(
            f"Computing Correlation Image for {args.test_corr} (Kernel={args.kernel_size})..."
        )
        corr = compute_correlation_image(args.test_corr, kernel_size=args.kernel_size)
        print(f"Corr Map Shape: {corr.shape}")
        print(
            f"Corr Stats: Min={corr.min():.4f}, Max={corr.max():.4f}, Mean={corr.mean():.4f}"
        )

        if args.output_plot:
            plt.figure(figsize=(6, 6))
            plt.imshow(corr, cmap="hot", vmin=0, vmax=1)
            plt.colorbar(label="Correlation")
            plt.title(f"Local Correlation Image (k={args.kernel_size})")
            plt.savefig(args.output_plot)
            print(f"Plot saved to {args.output_plot}")
