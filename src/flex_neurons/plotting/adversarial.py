import matplotlib.pyplot as plt

from flex_neurons.analysis.adversarial import SUPPORTED_ATTACKS


def plot_attack_comparison(
    results_by_model: dict,
    attack_name: str,
    ax=None,
):
    """results_by_model: dict[model_name] -> dict[eps] -> dict (output of run_attack).
    Plots adv_acc vs eps, one line per model. Returns Axes."""
    if not results_by_model:
        raise ValueError("results_by_model dict is empty")

    if ax is None:
        _, ax = plt.subplots()

    for model_name, eps_results in results_by_model.items():
        eps_vals = sorted(eps_results.keys())
        adv_accs = [eps_results[e]["adv_acc"] for e in eps_vals]
        ax.plot(eps_vals, adv_accs, marker="o", label=model_name)

    ax.set_xlabel("Epsilon")
    ax.set_ylabel("Adversarial accuracy")
    ax.set_title(f"Attack: {attack_name}")
    ax.legend()
    return ax


def plot_multi_attack_comparison(
    results_by_model_attack: dict,
    ax=None,
):
    """results_by_model_attack: dict[model_name][attack_name] -> dict[eps] -> run_attack output.
    Multi-panel figure: one subplot per attack. Returns Figure."""
    # Collect the set of attacks that appear across all models
    attack_names = []
    for model_results in results_by_model_attack.values():
        for name in model_results:
            if name not in attack_names:
                attack_names.append(name)

    # Sort in canonical order where possible
    ordered = [a for a in SUPPORTED_ATTACKS if a in attack_names]
    remainder = [a for a in attack_names if a not in ordered]
    attack_names = ordered + remainder

    n = len(attack_names)
    if n == 0:
        fig, _ = plt.subplots()
        return fig

    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), squeeze=False)
    axes = axes[0]  # shape: (n,)

    for col, attack_name in enumerate(attack_names):
        panel_results = {}
        for model_name, model_results in results_by_model_attack.items():
            if attack_name in model_results:
                panel_results[model_name] = model_results[attack_name]
        plot_attack_comparison(panel_results, attack_name, ax=axes[col])

    fig.tight_layout()
    return fig
