import numpy as np
import torch
import torch.nn as nn
import matplotlib

# Use non-interactive backend if not already set, but better to let caller handle backend if possible?
# But utility should probably just plot.
import matplotlib.pyplot as plt
from pathlib import Path
from src.modules.layers.flex import Flex2D
from src.analysis.run_loader import RunLoader


def load_experiment_model(experiment_id, device):
    """
    Loads the experiment model and determines output directory.

    Args:
        experiment_id (str/int): ID or path to experiment.
        device (torch.device): Device to load model on.

    Returns:
        tuple: (model, output_dir_path)
    """
    if isinstance(experiment_id, int) or (
        isinstance(experiment_id, str) and experiment_id.isdigit()
    ):
        exp_path = f"__local__/experiment-{experiment_id}/000000"
    else:
        exp_path = experiment_id

    loader = RunLoader(exp_path)
    model = loader.model.to(device)
    model.eval()

    output_dir = Path(exp_path) / "results" / "spectral_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    return model, output_dir


def get_radial_profile(power_spectrum, normalize=True):
    """
    Computes the radial average of a 2D power spectrum.

    Args:
        power_spectrum (np.ndarray): 2D array of the power spectrum (centered).
        normalize (bool): If True, normalized to the DC power (r=0).

    Returns:
        np.ndarray: 1D radial profile.
    """
    h, w = power_spectrum.shape
    cy, cx = h // 2, w // 2
    y, x = np.indices((h, w))
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2).astype(int)

    # Bin by radius
    tbin = np.bincount(r.ravel(), power_spectrum.ravel())
    nr = np.bincount(r.ravel())

    # Avoid division by zero
    non_zero = nr > 0
    radialprofile = tbin[non_zero] / nr[non_zero]

    if normalize and len(radialprofile) > 0:
        mean_power = radialprofile.mean()
        if mean_power > 1e-12:
            radialprofile /= mean_power

    return radialprofile


def compute_slope(profile, nyquist_limit):
    """
    Computes the slope of the linear fit on the log-log plot of the radial profile.

    Args:
        profile (np.ndarray): 1D radial profile.
        nyquist_limit (int): Nyquist frequency limit (pixels).

    Returns:
        float: Slope of the fit.
        float: Intercept of the fit.
        np.ndarray: x values used for fit.
        np.ndarray: y values used for fit.
    """
    freqs = np.arange(len(profile))
    mask = (freqs > 0) & (freqs <= nyquist_limit)
    x_fit = freqs[mask]
    y_fit = profile[mask]

    # Filter zeros/negative values to avoid log(error)
    valid_mask = y_fit > 1e-12
    x_fit = x_fit[valid_mask]
    y_fit = y_fit[valid_mask]

    if len(x_fit) < 2:
        return np.nan, np.nan, x_fit, y_fit

    log_x = np.log(x_fit)
    log_y = np.log(y_fit)
    slope, intercept = np.polyfit(log_x, log_y, 1)

    return slope, intercept, x_fit, y_fit


def is_conv_pool_layer(module):
    """Checks if a module is a Convolution or Pooling layer."""
    return isinstance(module, (nn.Conv2d, nn.MaxPool2d, nn.AvgPool2d, Flex2D))


def get_target_layer(model, layer_idx):
    """
    Finds the Nth Conv/Pool layer in the model.

    Args:
        model (nn.Module): The model to traverse.
        layer_idx (int): The index of the conv/pool layer to find (0-indexed).

    Returns:
        tuple: (module, layer_name)
    """
    count = 0
    for name, module in model.named_modules():
        if is_conv_pool_layer(module):
            if count == layer_idx:
                return module, name
            count += 1

    raise ValueError(
        f"Layer index {layer_idx} out of range. Found {count} conv/pool layers."
    )
