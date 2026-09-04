"""Plotting utilities for brain-score neural predictivity results."""
import numpy as np
import matplotlib.pyplot as plt


def plot_brain_score_comparison(scores_by_model: dict, ax=None):
    """Grouped bar chart of brain-score results across benchmarks.

    Args:
        scores_by_model: dict[model_name] -> dict[benchmark_id] ->
            {'center': float, 'error': float}.
        ax: existing Axes to draw on; a new figure is created if None.

    Returns:
        matplotlib Axes.
    """
    if not scores_by_model:
        raise ValueError("scores_by_model must be non-empty.")

    model_names = list(scores_by_model.keys())
    # Collect union of all benchmark ids, preserving insertion order.
    benchmark_ids = list(
        dict.fromkeys(
            bench_id
            for model_scores in scores_by_model.values()
            for bench_id in model_scores
        )
    )

    if not benchmark_ids:
        raise ValueError(
            "No benchmark ids found in scores_by_model values."
        )

    n_benchmarks = len(benchmark_ids)
    n_models = len(model_names)
    x = np.arange(n_benchmarks)
    bar_width = 0.8 / n_models

    if ax is None:
        _, ax = plt.subplots(figsize=(max(6, 2 * n_benchmarks), 5))

    colors = plt.cm.viridis(np.linspace(0.15, 0.85, n_models))

    for i, model_name in enumerate(model_names):
        model_scores = scores_by_model[model_name]
        centers = [
            model_scores.get(b, {}).get("center", 0.0) or 0.0
            for b in benchmark_ids
        ]
        errors = [
            model_scores.get(b, {}).get("error") or 0.0
            for b in benchmark_ids
        ]
        offsets = x + (i - (n_models - 1) / 2.0) * bar_width
        ax.bar(
            offsets,
            centers,
            width=bar_width,
            yerr=errors,
            label=model_name,
            color=colors[i],
            capsize=4,
            error_kw={"elinewidth": 1.2},
        )

    ax.set_xticks(x)
    ax.set_xticklabels(benchmark_ids, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Neural Predictivity (r)")
    ax.set_title("Brain-Score Comparison")
    ax.legend(title="Model")
    ax.set_ylim(bottom=0)
    return ax


def plot_majajhong_heatmap(
    scores_by_model: dict, region: str = "IT", ax=None
):
    """Heatmap of MajajHong layer-by-model scores.

    Args:
        scores_by_model: dict[model_name] -> dict[benchmark_id] ->
            {'center': float, 'error': float, 'raw': list|None}.
            Per-layer raw scores are expected under the 'raw' key.
        region: brain region to plot ('IT' or 'V4').
        ax: existing Axes to draw on; a new figure is created if None.

    Returns:
        matplotlib Axes.
    """
    if not scores_by_model:
        raise ValueError("scores_by_model must be non-empty.")

    model_names = list(scores_by_model.keys())

    # Identify the benchmark id that matches the requested region.
    target_bench = None
    for bench_id in next(iter(scores_by_model.values())):
        if region in bench_id and "MajajHong" in bench_id:
            target_bench = bench_id
            break

    if target_bench is None:
        raise ValueError(
            f"No MajajHong benchmark found for region '{region}' in "
            "scores_by_model keys."
        )

    # Build matrix: rows = layers (from raw folds), cols = models.
    # Raw is expected to be a list of per-layer scores.
    # Fall back to the single center value when raw is unavailable.
    columns = []
    for model_name in model_names:
        bench_data = scores_by_model[model_name].get(target_bench, {})
        raw = bench_data.get("raw")
        if raw is not None and isinstance(raw, (list, np.ndarray)):
            columns.append(np.asarray(raw, dtype=float))
        else:
            columns.append(
                np.array([bench_data.get("center", float("nan"))])
            )

    # Pad columns to the same length.
    max_len = max(len(c) for c in columns)
    matrix = np.full((max_len, len(model_names)), np.nan)
    for col_idx, col in enumerate(columns):
        matrix[: len(col), col_idx] = col

    if ax is None:
        fig_h = max(4, 0.3 * max_len)
        _, ax = plt.subplots(figsize=(max(4, len(model_names) * 1.2), fig_h))

    img = ax.imshow(matrix, aspect="auto", cmap="viridis", interpolation="nearest")
    plt.colorbar(img, ax=ax, label="Neural Predictivity (r)")

    ax.set_xticks(range(len(model_names)))
    ax.set_xticklabels(model_names, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Layer index")
    ax.set_title(f"MajajHong {region} — layer × model heatmap")
    return ax
