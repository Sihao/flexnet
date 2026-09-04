"""
Loss criteria for training -- plain torch, no project dependencies.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftTargetCrossEntropy(nn.Module):
    """Cross entropy against soft targets (e.g. mixup-mixed one-hot vectors)."""

    def forward(self, logits, soft_targets):
        return torch.mean(torch.sum(-soft_targets * F.log_softmax(logits, dim=1), dim=1))


def build_criterion(config):
    """Build a loss criterion from a config dict.

    config["loss"] selects the criterion (default "ce"):
      - "ce":      nn.CrossEntropyLoss(label_smoothing=config.get("label_smoothing", 0.0))
      - "soft_ce": SoftTargetCrossEntropy() -- smoothing baked into mixup soft labels
      - "bce":     nn.BCEWithLogitsLoss()
    Raises ValueError for any other value.
    """
    loss_name = config.get("loss", "ce")

    if loss_name == "ce":
        return nn.CrossEntropyLoss(label_smoothing=config.get("label_smoothing", 0.0))
    if loss_name == "soft_ce":
        return SoftTargetCrossEntropy()
    if loss_name == "bce":
        return nn.BCEWithLogitsLoss()

    raise ValueError(f"Unknown loss: {loss_name!r}")
