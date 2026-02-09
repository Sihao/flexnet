"""
NOTE:
- hard sigmoid function makes the model generalises much much better
- it also make this s-shaped curve in dichotomy and conv ratio rather easy to achieve
- see the images in this folder
- the hypothesis is that it smoothens the landscape / gives model more freedom to select the right dichotomy
"""

import torch
from torch import nn


def sigmoid_smoothed(logits, tau=1.0):
    """
    Using annealling trick to make a discrete distribution differentiable.
    In this modified version, only the influence of logits increase as tau gets smaller.
    """
    y = torch.sigmoid(logits / max(tau, 1e-8))  # clamping to prevent division by zero
    return y


class SharpScaledSigmoid(nn.Module):
    def __init__(self, scale=50):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        return torch.sigmoid(self.scale * x)

    def apply(self, x):
        return self.forward(x)


class NoneScaledSigmoid(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.sigmoid(x)

    def apply(self, x):
        return self.forward(x)


class HardsigmoidFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)

        output = torch.zeros_like(input)
        output[input <= -3] = 0
        output[input >= 3] = 1
        output[(input > -3) & (input < 3)] = input[(input > -3) & (input < 3)] / 6 + 0.5
        return output

    @staticmethod
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        grad_input = torch.zeros_like(input)
        grad_input[(input > -3) & (input < 3)] = 1 / 6
        return grad_input * grad_output

    @staticmethod
    def hessian(ctx, grad_output):
        (input,) = ctx.saved_tensors
        return torch.zeros_like(input)


sigmoid_plain = nn.Sigmoid()
