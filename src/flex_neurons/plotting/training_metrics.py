import matplotlib.pyplot as plt
from typing import Optional

from src.flex_neurons.plotting.style import (
    FIGSIZE_DOUBLE,
    FIGSIZE_SINGLE,
    LINE_WIDTH,
    get_color,
)

_LOSS_KEYS = ("train_loss", "val_loss")
_ACC_KEYS = ("train_acc", "val_acc")
_REQUIRED_KEYS = _LOSS_KEYS + _ACC_KEYS

_TRAIN_COLOR = get_color("vanilla")
_VAL_COLOR = get_color("baseline")


def _validate_keys(history: dict, keys: tuple) -> None:
    missing = [k for k in keys if k not in history]
    if missing:
        raise ValueError(f"history is missing required keys: {missing}")

    lengths = {k: len(history[k]) for k in keys}
    unique_lengths = set(lengths.values())
    if len(unique_lengths) != 1:
        raise ValueError(
            f"history values have mismatched lengths: {lengths}"
        )


def _validate_loss_keys(history: dict) -> None:
    _validate_keys(history, _LOSS_KEYS)


def _validate_acc_keys(history: dict) -> None:
    _validate_keys(history, _ACC_KEYS)


def _validate_history(history: dict) -> None:
    _validate_loss_keys(history)
    _validate_acc_keys(history)

    loss_len = len(history[_LOSS_KEYS[0]])
    acc_len = len(history[_ACC_KEYS[0]])
    if loss_len != acc_len:
        lengths = {k: len(history[k]) for k in _REQUIRED_KEYS}
        raise ValueError(
            f"history values have mismatched lengths: {lengths}"
        )


def plot_training_curves(
    history: dict,
    ax=None,
    title: Optional[str] = None,
) -> tuple:
    """Plot training and validation loss/accuracy.

    Args:
        history: dict with keys 'train_loss', 'val_loss', 'train_acc',
                'val_acc'. Each value is a 1D iterable (one entry per epoch).
        ax: optional 2-tuple of Axes (loss_ax, acc_ax). If None, creates
            a new (1, 2) Figure.
        title: optional figure title.

    Returns:
        (fig, (loss_ax, acc_ax)). fig is None if external axes were passed.

    Raises:
        ValueError if any required key is missing or if lengths differ.
    """
    _validate_history(history)

    fig = None
    if ax is None:
        fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_DOUBLE)
        loss_ax, acc_ax = axes
    else:
        loss_ax, acc_ax = ax

    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss panel
    loss_ax.plot(
        epochs, history["train_loss"],
        linestyle="solid", color=_TRAIN_COLOR,
        linewidth=LINE_WIDTH, label="Train",
    )
    loss_ax.plot(
        epochs, history["val_loss"],
        linestyle="dashed", color=_VAL_COLOR,
        linewidth=LINE_WIDTH, label="Val",
    )
    loss_ax.set_xlabel("Epoch")
    loss_ax.set_ylabel("Loss")
    loss_ax.legend(frameon=False)

    # Accuracy panel
    acc_ax.plot(
        epochs, history["train_acc"],
        linestyle="solid", color=_TRAIN_COLOR,
        linewidth=LINE_WIDTH, label="Train",
    )
    acc_ax.plot(
        epochs, history["val_acc"],
        linestyle="dashed", color=_VAL_COLOR,
        linewidth=LINE_WIDTH, label="Val",
    )
    acc_ax.set_xlabel("Epoch")
    acc_ax.set_ylabel("Top-1 accuracy")
    acc_ax.set_ylim(0, 1)
    acc_ax.legend(frameon=False)

    if title is not None and fig is not None:
        fig.suptitle(title)

    return fig, (loss_ax, acc_ax)


def plot_loss_curve(history: dict, ax=None):
    """Single-axis loss-only plot. Used by training_metrics CLI.

    Returns Axes.
    """
    _validate_loss_keys(history)

    if ax is None:
        _, ax = plt.subplots(figsize=FIGSIZE_SINGLE)

    epochs = range(1, len(history["train_loss"]) + 1)

    ax.plot(
        epochs, history["train_loss"],
        linestyle="solid", color=_TRAIN_COLOR,
        linewidth=LINE_WIDTH, label="Train",
    )
    ax.plot(
        epochs, history["val_loss"],
        linestyle="dashed", color=_VAL_COLOR,
        linewidth=LINE_WIDTH, label="Val",
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend(frameon=False)

    return ax


def plot_accuracy_curve(history: dict, ax=None):
    """Single-axis accuracy-only plot. Returns Axes."""
    _validate_acc_keys(history)

    if ax is None:
        _, ax = plt.subplots(figsize=FIGSIZE_SINGLE)

    epochs = range(1, len(history["train_acc"]) + 1)

    ax.plot(
        epochs, history["train_acc"],
        linestyle="solid", color=_TRAIN_COLOR,
        linewidth=LINE_WIDTH, label="Train",
    )
    ax.plot(
        epochs, history["val_acc"],
        linestyle="dashed", color=_VAL_COLOR,
        linewidth=LINE_WIDTH, label="Val",
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Top-1 accuracy")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False)

    return ax
