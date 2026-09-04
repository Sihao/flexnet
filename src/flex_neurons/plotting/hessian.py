import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers 3D projection

from src.third_party.pyhessian.density_plot import density_generate


def plot_eigenvalue_density(
    density: dict, ax=None, sigma_squared: float = 1e-5
):
    """Plot smoothed empirical spectral distribution.

    density: output of compute_eigenvalue_density.
    Returns the Axes used.
    """
    if ax is None:
        _, ax = plt.subplots()

    eigenvalues = density["eigenvalues"]
    weights = density["weights"]

    # density_generate expects shape (num_runs, iter) for both arrays.
    smoothed, grids = density_generate(
        eigenvalues, weights, sigma_squared=sigma_squared
    )

    ax.semilogy(grids, smoothed + 1e-7)
    ax.set_xlabel("Eigenvalue")
    ax.set_ylabel("Density (log scale)")
    ax.set_title("Eigenvalue Spectral Density")
    return ax


def plot_top_eigenvalues_bar(top: dict, ax=None):
    """Bar chart of top eigenvalues.

    top: output of compute_top_eigenvalues.
    Returns the Axes used.
    """
    if ax is None:
        _, ax = plt.subplots()

    eigenvalues = top["eigenvalues"]
    indices = np.arange(1, len(eigenvalues) + 1)

    ax.bar(indices, eigenvalues)
    ax.set_xlabel("Rank")
    ax.set_ylabel("Eigenvalue")
    ax.set_title("Top Hessian Eigenvalues")
    ax.set_xticks(indices)
    return ax


def plot_landscape_contour(landscape: dict, ax=None, levels: int = 20):
    """2D filled contour of the loss landscape.

    landscape: output of compute_loss_landscape.
    Returns the Axes used.
    """
    if ax is None:
        _, ax = plt.subplots()

    losses = landscape["losses"]
    steps = landscape["steps"]
    distance = landscape["distance"]

    coords = np.linspace(-distance, distance, steps)
    X, Y = np.meshgrid(coords, coords)

    cf = ax.contourf(X, Y, losses, levels=levels, cmap="viridis")
    plt.colorbar(cf, ax=ax, label="Loss")
    ax.set_xlabel("Direction 1")
    ax.set_ylabel("Direction 2")
    ax.set_title("Loss Landscape (contour)")
    return ax


def plot_landscape_3d(landscape: dict, ax=None):
    """3D surface of the loss landscape.

    landscape: output of compute_loss_landscape.
    ax must be a 3D Axes (projection='3d'). If None, one is created.
    Returns the Axes used.
    """
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

    if not isinstance(ax, Axes3D):
        raise TypeError(
            "plot_landscape_3d requires a 3D axis (projection='3d'); "
            f"got a 2D Axes ({type(ax).__name__})"
        )

    losses = landscape["losses"]
    steps = landscape["steps"]
    distance = landscape["distance"]

    coords = np.linspace(-distance, distance, steps)
    X, Y = np.meshgrid(coords, coords)

    ax.plot_surface(X, Y, losses, cmap="viridis", edgecolor="none")
    ax.set_xlabel("Direction 1")
    ax.set_ylabel("Direction 2")
    ax.set_zlabel("Loss")
    ax.set_title("Loss Landscape (3D)")
    return ax
