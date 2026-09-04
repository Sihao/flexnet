import matplotlib.pyplot as plt
import numpy as np


def plot_power_spectrum(spectra_by_model: dict, ax=None, log_scale: bool = True):
    """
    Plot radial power spectra for one or more models on a log-log axes.

    Args:
        spectra_by_model: dict mapping model name (str) to a dict with keys
                          'freq' (np.ndarray) and 'power' (np.ndarray), as
                          returned by compute_power_spectrum.
        ax:               matplotlib Axes to draw on. Created if None.
        log_scale:        If True, use log-log scale (default). If False,
                          use linear scale.

    Returns:
        The matplotlib Axes.
    """
    if ax is None:
        _, ax = plt.subplots()

    for model_name, spectrum in spectra_by_model.items():
        freq = np.asarray(spectrum["freq"])
        power = np.asarray(spectrum["power"])
        if log_scale:
            # Clip to a tiny positive floor so zero/negative power values
            # don't get silently dropped (and warned about) by the log
            # y-scale set below. Preserves the number of plotted points.
            power = np.clip(power, 1e-300, None)
        ax.plot(freq, power, label=model_name, linewidth=1.5)

    if log_scale:
        ax.set_xscale("log")
        ax.set_yscale("log")

    ax.set_xlabel("Spatial frequency (cycles / pixel, normalized)")
    ax.set_ylabel("Power")
    ax.set_title("Radial power spectrum")
    ax.legend()
    ax.grid(True, which="both", linestyle="--", alpha=0.4)

    return ax


def plot_slope_comparison(slopes_by_model: dict, ax=None):
    """
    Bar plot comparing spectral slope across models.

    Args:
        slopes_by_model: dict mapping model name (str) to slope value (float).
        ax:              matplotlib Axes to draw on. Created if None.

    Returns:
        The matplotlib Axes.
    """
    if ax is None:
        _, ax = plt.subplots()

    names = list(slopes_by_model.keys())
    values = [float(slopes_by_model[n]) for n in names]
    x_pos = np.arange(len(names))

    ax.bar(x_pos, values, tick_label=names)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Model")
    ax.set_ylabel("Spectral slope")
    ax.set_title("Spectral slope comparison")

    return ax
