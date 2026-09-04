"""
Exponential moving average (EMA) of model weights.

Keeps a shadow copy of a model's parameters/buffers that is updated after
every optimizer step via a decayed running average. Used to evaluate with
smoother, less noisy weights than the raw training weights.
"""

import copy

import torch


class ModelEMA:
    """Maintains an EMA shadow copy of a model's state_dict.

    Args:
        model: The live model to shadow. If wrapped in DDP (has a `.module`
            attribute), the underlying module is copied instead of the wrapper.
        decay: EMA decay rate; higher means the shadow changes more slowly.
    """

    def __init__(self, model, decay: float = 0.9999):
        unwrapped = getattr(model, "module", model)
        self.module = copy.deepcopy(unwrapped)
        self.decay = decay
        for param in self.module.parameters():
            param.requires_grad = False
        self.module.eval()

    def update(self, model):
        """Update the shadow weights toward the live model's current weights."""
        live = getattr(model, "module", model)
        with torch.no_grad():
            for ema_v, live_v in zip(
                self.module.state_dict().values(), live.state_dict().values()
            ):
                if torch.is_floating_point(ema_v):
                    ema_v.mul_(self.decay).add_(live_v, alpha=1 - self.decay)
                else:
                    ema_v.copy_(live_v)

    def state_dict(self):
        return self.module.state_dict()

    def load_state_dict(self, state_dict):
        self.module.load_state_dict(state_dict)
