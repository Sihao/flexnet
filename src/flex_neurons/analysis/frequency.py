import numpy as np
import torch
from scipy import stats


def _validate_image(image: torch.Tensor) -> torch.Tensor:
    """
    Validate and return image as a 2D float numpy array or raise ValueError.

    Accepts shapes: (H, W), (C, H, W), (N, C, H, W).
    For multi-channel/multi-image inputs returns the mean over N and C,
    collapsing to (H, W).
    """
    if not isinstance(image, torch.Tensor):
        raise ValueError(f"image must be a torch.Tensor, got {type(image)}")

    if image.dtype not in (torch.float32, torch.float64, torch.float16):
        raise ValueError(
            f"image dtype must be float (float16/32/64), got {image.dtype}"
        )

    rank = image.ndim
    if rank == 2:
        h, w = image.shape
    elif rank == 3:
        _, h, w = image.shape
    elif rank == 4:
        _, _, h, w = image.shape
    else:
        raise ValueError(f"image must have rank 2, 3, or 4; got rank {rank}")

    if h != w:
        raise ValueError(f"image must be square (H == W), got H={h} W={w}")

    # Collapse to (H, W) by averaging over N, C dims.
    if rank == 3:
        img2d = image.float().mean(dim=0)  # (H, W)
    elif rank == 4:
        img2d = image.float().mean(dim=(0, 1))  # (H, W)
    else:
        img2d = image.float()

    return img2d


def compute_power_spectrum(image: torch.Tensor) -> dict:
    """
    Radial power spectrum of a 2D image or batch of images.

    Args:
        image: tensor of shape (H, W) or (C, H, W) or (N, C, H, W).
               For multi-channel/multi-image input, returns mean spectrum.

    Returns:
        dict with keys:
            'freq'  -- 1D np.ndarray, normalized frequencies in (0, sqrt(2)/2]
                   (diagonal FFT corners exceed the axis-aligned Nyquist
                   frequency of 0.5)
            'power' -- 1D np.ndarray, mean radial power (DC bin excluded)
    """
    img2d = _validate_image(image)  # (H, W) float tensor
    n = img2d.shape[0]  # square, so N == H == W

    # 2D FFT; DC sits at index (0, 0) in the unshifted output.
    # We use amplitude (|FFT|) rather than magnitude-squared so that a
    # 1/f-amplitude image yields a radial slope of ~-1 (not ~-2).
    fft2 = torch.fft.fft2(img2d)
    power2d = torch.abs(fft2).cpu().numpy()  # (N, N), amplitude

    # Build integer-radius grid in pixel units.  Bin index i maps to
    # normalized frequency i/n, so Nyquist bin (n//2) -> 0.5.
    px_freq = np.fft.fftfreq(n, d=1.0 / n)  # pixel-unit frequencies
    fx_px, fy_px = np.meshgrid(px_freq, px_freq)
    r_px = np.rint(np.sqrt(fx_px**2 + fy_px**2)).astype(int)

    max_r = r_px.max()
    tbin = np.bincount(r_px.ravel(), weights=power2d.ravel(), minlength=max_r + 1)
    count = np.bincount(r_px.ravel(), minlength=max_r + 1)

    nonzero = count > 0
    radial_power = np.where(nonzero, tbin / np.where(nonzero, count, 1), 0.0)

    # Normalize frequency: bin index i -> i/n gives cycles/pixel, Nyquist = 0.5
    bin_indices = np.arange(len(radial_power))
    freq_norm = bin_indices / n

    # Drop the DC bin (index 0) only. Diagonal FFT corners have radius up
    # to sqrt(2) * Nyquist (normalized ~0.707), which is real high-frequency
    # content -- do not truncate it at the axis-aligned Nyquist of 0.5.
    keep = bin_indices > 0
    freq = freq_norm[keep]
    power = radial_power[keep]

    return {"freq": freq, "power": power}


def fit_slope(
    freq: np.ndarray,
    power: np.ndarray,
    fit_range: tuple = (0.05, 0.4),
) -> dict:
    """
    Log-log linear fit of power vs frequency over a frequency band.

    Args:
        freq:      1D array of normalized frequencies (from compute_power_spectrum).
        power:     1D array of mean radial power (same length as freq).
        fit_range: (low, high) normalized-frequency band for the fit.

    Returns:
        dict with keys:
            'slope'     -- float
            'intercept' -- float
            'r_squared' -- float
            'fit_freq'  -- np.ndarray of frequencies in the fit band
            'fit_power' -- np.ndarray of fitted line evaluated at fit_freq
    """
    low, high = fit_range
    in_band = (freq >= low) & (freq <= high) & (power > 0)

    freq_band = freq[in_band]

    log_freq = np.log10(freq_band)
    log_power = np.log10(power[in_band])

    # Drop any non-finite values (e.g. log of zero that slipped through).
    finite = np.isfinite(log_freq) & np.isfinite(log_power)
    log_freq = log_freq[finite]
    log_power = log_power[finite]
    freq_band = freq_band[finite]

    if len(log_freq) < 5:
        raise ValueError(
            f"Fewer than 5 valid points in fit band [{low}, {high}]; "
            f"got {len(log_freq)}. Widen fit_range or check the spectrum."
        )

    result = stats.linregress(log_freq, log_power)
    slope = float(result.slope)
    intercept = float(result.intercept)
    r_squared = float(result.rvalue ** 2)

    fit_power = 10 ** (intercept + slope * np.log10(freq_band))

    return {
        "slope": slope,
        "intercept": intercept,
        "r_squared": r_squared,
        "fit_freq": freq_band,
        "fit_power": fit_power,
    }
