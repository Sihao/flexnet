import matplotlib.pyplot as plt
import numpy as np


def plot_imagenet_c_heatmap(results: dict, ax=None):
    """Heatmap of ImageNet-C top-1 accuracy.

    Args:
        results: dict[corruption][severity] = top1 float, as returned by evaluate_imagenet_c.
        ax: Optional matplotlib Axes. If None, one is created.

    Returns:
        The Axes object with the heatmap drawn.
    """
    corruptions = list(results.keys())
    if not corruptions:
        raise ValueError("results dict is empty")

    # Collect severities in sorted order from first entry.
    severities = sorted(next(iter(results.values())).keys())

    # Build (n_corruptions, n_severities) matrix.
    data = np.array(
        [[results[c][s] for s in severities] for c in corruptions],
        dtype=float,
    )

    if ax is None:
        fig, ax = plt.subplots(
            figsize=(max(4, len(severities) * 1.2), max(4, len(corruptions) * 0.5))
        )

    im = ax.imshow(data, cmap="viridis", aspect="auto", vmin=0.0, vmax=1.0)

    ax.set_xticks(range(len(severities)))
    ax.set_xticklabels([str(s) for s in severities])
    ax.set_xlabel("Severity")

    ax.set_yticks(range(len(corruptions)))
    ax.set_yticklabels([c.replace("_", " ") for c in corruptions], fontsize=8)
    ax.set_ylabel("Corruption")

    ax.set_title("ImageNet-C Top-1 Accuracy")

    plt.colorbar(im, ax=ax, label="Top-1 Accuracy")

    return ax


def plot_imagenet_a_bar(results_by_model: dict, ax=None):
    """Bar plot of ImageNet-A top-1 accuracy per model.

    Args:
        results_by_model: dict[model_name: str, top1: float].
        ax: Optional matplotlib Axes. If None, one is created.

    Returns:
        The Axes object with the bar plot drawn.
    """
    if not results_by_model:
        raise ValueError("results_by_model dict is empty")

    models = list(results_by_model.keys())
    accuracies = [results_by_model[m] for m in models]

    if ax is None:
        fig, ax = plt.subplots(figsize=(max(4, len(models) * 1.2), 4))

    x = np.arange(len(models))
    bars = ax.bar(x, accuracies, color=plt.cm.viridis(np.linspace(0.2, 0.8, len(models))))

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("Top-1 Accuracy")
    ax.set_title("ImageNet-A Accuracy by Model")
    ax.set_ylim(0, 1)
    ax.legend(bars, models, title="Model", fontsize=8)

    return ax


def plot_imagenet_c_summary_bar(results_by_model: dict, ax=None):
    """Bar plot of mean ImageNet-C top-1 across all corruptions and severities.

    Args:
        results_by_model: dict[model_name: str, dict[corruption][severity] = top1 float].
        ax: Optional matplotlib Axes. If None, one is created.

    Returns:
        The Axes object with the bar plot drawn.
    """
    if not results_by_model:
        raise ValueError("results_by_model dict is empty")

    models = list(results_by_model.keys())
    means = []
    for model_name in models:
        per_model = results_by_model[model_name]
        values = [
            acc
            for corruption_data in per_model.values()
            for acc in corruption_data.values()
        ]
        if not values:
            means.append(0.0)
        elif np.all(np.isnan(values)):
            # Every severity was skipped (e.g. empty ImageNet-C splits). Avoid
            # np.nanmean's "Mean of empty slice" RuntimeWarning and record nan
            # explicitly rather than silently defaulting to 0.0.
            means.append(float("nan"))
        else:
            means.append(float(np.nanmean(values)))

    if ax is None:
        fig, ax = plt.subplots(figsize=(max(4, len(models) * 1.2), 4))

    x = np.arange(len(models))
    bars = ax.bar(x, means, color=plt.cm.viridis(np.linspace(0.2, 0.8, len(models))))

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("Mean Top-1 Accuracy")
    ax.set_title("ImageNet-C Mean Accuracy by Model")
    ax.set_ylim(0, 1)
    ax.legend(bars, models, title="Model", fontsize=8)

    return ax
