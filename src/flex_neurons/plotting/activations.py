"""Plotting utilities for activation maps."""
import matplotlib.pyplot as plt


def plot_activation_grid(
    activations_by_layer: dict,
    n_channels: int = 8,
    figsize_per_row: tuple = (12, 1.5),
):
    """One row per layer, n_channels columns. Returns matplotlib Figure.

    activations_by_layer: dict[str, torch.Tensor], each tensor shape
        (1, C, H, W) or (C, H, W).
    """
    for name, tensor in activations_by_layer.items():
        if tensor.dim() not in (3, 4):
            raise ValueError(
                f"Layer '{name}' has tensor with {tensor.dim()} dims;"
                " expected 3 or 4."
            )

    n_rows = len(activations_by_layer)
    figsize = (figsize_per_row[0], figsize_per_row[1] * n_rows)
    # squeeze=False guarantees axes is always a 2D ndarray (n_rows x n_channels),
    # regardless of whether n_rows or n_channels is 1.
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_channels, figsize=figsize, squeeze=False)

    for row_idx, (layer_name, tensor) in enumerate(activations_by_layer.items()):
        if tensor.dim() == 4:
            tensor = tensor[0]  # take first sample -> (C, H, W)

        n_available = tensor.shape[0]
        row_axes = axes[row_idx]

        for col_idx in range(n_channels):
            axis = row_axes[col_idx]
            if col_idx >= n_available:
                axis.axis("off")
                continue

            channel = tensor[col_idx].numpy()
            axis.imshow(channel, cmap="gray")
            axis.set_xticks([])
            axis.set_yticks([])

            if col_idx == 0:
                axis.set_ylabel(
                    layer_name, fontsize=7, rotation=0,
                    labelpad=40, va="center",
                )

    plt.tight_layout()
    return fig
