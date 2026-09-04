import contextlib
import sys
from typing import Callable
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.third_party.pyhessian.pyhessian import hessian as _Hessian
from src.third_party.loss_landscapes.loss_landscapes import random_plane
from src.third_party.loss_landscapes.loss_landscapes.metrics import Loss
import src.third_party.loss_landscapes.loss_landscapes.model_interface.model_parameters as _model_parameters_module

# The vendored hessian.py module (re-exported as the `hessian` class by
# pyhessian/pyhessian/__init__.py). We need the *module*, not the class, so we
# can patch the `select_device` name it calls internally.
_pyhessian_hessian_module = sys.modules[_Hessian.__module__]


@contextlib.contextmanager
def _forced_device(module, device: torch.device):
    """Force a vendored module's `select_device()` to return `device`.

    Both vendored libraries (pyhessian and loss_landscapes) call their own
    imported `select_device()` internally and unconditionally prefer cuda when
    it is available -- ignoring whatever device the caller actually wants.
    This patches that call out for the duration of the block so a
    `cuda=False` request is honored even on a machine with a GPU.
    """
    original_select_device = module.select_device
    module.select_device = lambda *args, **kwargs: device
    try:
        yield
    finally:
        module.select_device = original_select_device


def compute_top_eigenvalues(
    model: nn.Module,
    criterion: Callable,
    loader: DataLoader,
    k: int = 5,
    *,
    cuda: bool = True,
) -> dict:
    """Compute top-k Hessian eigenvalues via power iteration (vendored pyhessian).

    Returns:
        {'eigenvalues': np.ndarray of shape (k,),
         'eigenvectors': list of parameter-shaped tensors}
    """
    if not any(p.requires_grad for p in model.parameters()):
        raise ValueError(
            "compute_top_eigenvalues requires a model with at least one "
            "trainable (requires_grad=True) parameter"
        )
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")

    device = torch.device("cuda" if cuda and torch.cuda.is_available() else "cpu")

    was_training = model.training
    original_device = next(model.parameters()).device
    try:
        model.eval()
        model.to(device)
        try:
            inputs, targets = next(iter(loader))
        except StopIteration:
            raise ValueError(
                "compute_top_eigenvalues received an empty data loader"
            )
        inputs, targets = inputs.to(device), targets.to(device)
        with _forced_device(_pyhessian_hessian_module, device):
            hess = _Hessian(
                model, criterion, data_batch=(inputs, targets)
            )
        eigenvalues, eigenvectors = hess.eigenvalues(top_n=k)
    finally:
        model.to(original_device)
        if was_training:
            model.train()

    return {
        "eigenvalues": np.array(eigenvalues),
        "eigenvectors": eigenvectors,
    }


def compute_eigenvalue_density(
    model: nn.Module,
    criterion: Callable,
    loader: DataLoader,
    *,
    n_iter: int = 100,
    n_v: int = 1,
    cuda: bool = True,
) -> dict:
    """Stochastic Lanczos quadrature density via vendored pyhessian.

    Returns:
        {'eigenvalues': np.ndarray, 'weights': np.ndarray}
    """
    if not any(p.requires_grad for p in model.parameters()):
        raise ValueError(
            "compute_eigenvalue_density requires a model with at least one "
            "trainable (requires_grad=True) parameter"
        )

    device = torch.device("cuda" if cuda and torch.cuda.is_available() else "cpu")

    was_training = model.training
    original_device = next(model.parameters()).device
    try:
        model.eval()
        model.to(device)
        with _forced_device(_pyhessian_hessian_module, device):
            hess = _Hessian(model, criterion, dataloader=loader)
        eigen_list_full, weight_list_full = hess.density(
            iter=n_iter, num_runs=n_v
        )
    finally:
        model.to(original_device)
        if was_training:
            model.train()

    return {
        "eigenvalues": np.array(eigen_list_full),
        "weights": np.array(weight_list_full),
    }


def compute_loss_landscape(
    model: nn.Module,
    criterion: Callable,
    loader: DataLoader,
    *,
    distance: float = 1.0,
    steps: int = 21,
    normalization: str = "filter",
    cuda: bool = True,
) -> dict:
    """2D loss landscape on random orthogonal directions (vendored loss_landscapes).

    Returns:
        {'losses': np.ndarray of shape (steps, steps), 'distance': float,
         'steps': int}
    """
    if next(model.parameters(), None) is None:
        raise ValueError(
            "compute_loss_landscape requires a model with at least one parameter"
        )
    if distance <= 0:
        raise ValueError(f"distance must be > 0, got {distance}")
    if steps < 3:
        raise ValueError(f"steps must be >= 3, got {steps}")

    device = torch.device("cuda" if cuda and torch.cuda.is_available() else "cpu")

    was_training = model.training
    original_device = next(model.parameters()).device
    try:
        model.eval()
        model.to(device)
        try:
            inputs, targets = next(iter(loader))
        except StopIteration:
            raise ValueError(
                "compute_loss_landscape received an empty data loader"
            )
        inputs, targets = inputs.to(device), targets.to(device)
        metric = Loss(criterion, inputs, targets)
        with _forced_device(_model_parameters_module, device):
            loss_data = random_plane(
                model,
                metric,
                distance=distance,
                steps=steps,
                normalization=normalization,
                deepcopy_model=True,
            )
    finally:
        model.to(original_device)
        if was_training:
            model.train()

    return {
        "losses": np.array(loss_data),
        "distance": distance,
        "steps": steps,
    }
