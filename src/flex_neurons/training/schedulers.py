"""
LR scheduler builder -- reads scheduler config and returns a torch
lr_scheduler instance. No DDP coupling; callers step it once per epoch.
"""

import math
from typing import Optional

from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearLR,
    ReduceLROnPlateau,
    SequentialLR,
)


def build_scheduler(optimizer, config, total_epochs):
    """Build an LR scheduler from `config`.

    Reads `scheduler` (default "CosineAnnealingLR"), `warmup_epochs`
    (default 0), and `min_lr` (default 1e-6) from `config`.

    - "ReduceLROnPlateau": plateau scheduler, unaffected by warmup_epochs.
    - "CosineAnnealingLR" (or unset): cosine decay over `total_epochs`,
      preceded by `warmup_epochs` of linear warmup when > 0.
    - anything else: NotImplementedError.
    """
    warmup_epochs = config.get("warmup_epochs", 0)
    min_lr = config.get("min_lr", 1e-6)
    scheduler_name = config.get("scheduler", "CosineAnnealingLR")

    if scheduler_name == "ReduceLROnPlateau":
        return ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=10)

    if scheduler_name == "CosineAnnealingLR":
        if warmup_epochs == 0:
            return CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=min_lr)

        warmup = LinearLR(
            optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs
        )
        cosine = CosineAnnealingLR(
            optimizer, T_max=total_epochs - warmup_epochs, eta_min=min_lr
        )
        return SequentialLR(
            optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs]
        )

    raise NotImplementedError(f"Unknown scheduler: {scheduler_name!r}")


def step_scheduler(scheduler, metric: Optional[float] = None) -> None:
    """Step `scheduler` by one epoch, branching on scheduler type.

    `ReduceLROnPlateau` decides whether to reduce the LR by comparing
    `metric` against its running best, so it must be stepped as
    `scheduler.step(metric)`. Every other torch LR scheduler (e.g.
    CosineAnnealingLR, SequentialLR) tracks its own internal epoch
    counter and is stepped with no argument -- passing a metric to
    those is either ignored or a `TypeError`, depending on the class.
    Callers should go through this helper rather than calling
    `scheduler.step()` directly, so the branch only needs to exist
    once.

    Args:
        scheduler: An LR scheduler instance (as returned by
            `build_scheduler`), or None, in which case this is a no-op.
        metric: The value plateau schedulers should watch (e.g.
            validation loss, since `build_scheduler` constructs
            `ReduceLROnPlateau` with `mode="min"`). Ignored for
            non-plateau schedulers.

    Raises:
        ValueError: If `scheduler` is a `ReduceLROnPlateau` and `metric`
            is None, or is non-finite (NaN/inf). A NaN commonly means no
            validation loader was configured -- see e.g.
            `training/loop.py`, which passes `float("nan")` for
            `epoch_val_loss` when `val_loader is None`. Stepping a
            plateau scheduler with NaN never registers an improvement,
            so `num_bad_epochs` grows every epoch and the LR silently
            decays forever; failing loudly here is far better than that.
    """
    if scheduler is None:
        return

    if isinstance(scheduler, ReduceLROnPlateau):
        if metric is None:
            raise ValueError(
                f"{type(scheduler).__name__} requires a validation metric "
                "but None was passed; pass the per-epoch validation loss."
            )
        if not math.isfinite(metric):
            raise ValueError(
                f"{type(scheduler).__name__} received a non-finite metric "
                f"({metric!r}); is a validation loader configured? A "
                "plateau scheduler cannot run without a real validation "
                "metric."
            )
        scheduler.step(metric)
        return

    scheduler.step()
