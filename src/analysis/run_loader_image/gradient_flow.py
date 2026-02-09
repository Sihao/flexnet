#!/Users/donyin/miniconda3/envs/imperial/bin/python

"""
plot gradient flow
"""

from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
from src.analysis.run_loader import RunLoader
from pathlib import Path
from torch import nn


def plot_grad_flow(run_loader, save_path, set_y_lim=False, full_layer_name=False):
    """
    Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage:
        plug this function in trainer class after loss.backwards() as
        plot_grad_flow(self.model.named_parameters())
    """
    run_loader._do_a_dummy_backward_pass()
    named_parameters = run_loader.model.named_parameters()

    ave_grads, max_grads, layers = [], [], []
    for n, p in named_parameters:
        if (p.requires_grad) and ("bias" not in n):
            layers.append(n)
            try:
                ave_grads.append(p.grad.abs().mean().cpu().item())
            except AttributeError:
                ave_grads.append(0)
            try:
                max_grads.append(p.grad.abs().max().cpu().item())
            except AttributeError:
                max_grads.append(0)
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.45, lw=1, color="darkcyan")
    plt.bar(
        np.arange(len(max_grads)), ave_grads, alpha=0.45, lw=1, color="midnightblue"
    )

    plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
    if full_layer_name:
        plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    else:
        pass
    plt.xlim(left=0, right=len(ave_grads))
    if set_y_lim:
        plt.ylim(bottom=-0.001, top=0.04)  # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("Gradient Value")
    plt.title("Gradient Flow Across Layers")
    plt.grid(True)
    plt.legend(
        [
            Line2D([0], [0], color="darkcyan", lw=4),
            Line2D([0], [0], color="midnightblue", lw=4),
            Line2D([0], [0], color="k", lw=4),
        ],
        ["max-gradient", "mean-gradient", "zero-gradient"],
    )
    plt.tight_layout()
    plt.savefig(save_path)
    plt.clf()


if __name__ == "__main__":
    from src.training.dataset_select import get_dataset_obj
    import torch

    np.random.seed(42)
    torch.manual_seed(42)

    dataset = get_dataset_obj("cifar10", "TEST")
    base_folder = Path("/Users/donyin/Desktop/experiment")
    loader_vgg16 = RunLoader(base_folder / "1275fe75", whether_load_checkpoint=False)
    plot_grad_flow(run_loader=loader_vgg16, save_path="gradient_flow_vgg16.png")
