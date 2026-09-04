"""
Note: to add more loss functions, follow the pattern of the existing functions.
- the function should take a tensor as input
- the function should return a single value tensor

"""

import torch
from torch import abs, tensor, clamp


def linear_binary_loss(logits):
    """
    Computes the loss that encourages the logits to be closer to 0 or 1.
    This is a linear version of the peaked at half loss.

    The loss is minimized when the logits are close to 0 or 1 and is
    maximized when the logits are close to 0.5. When logits are outside [0, 1],
    the loss is 0.

    Parameters:
    - logits (torch.Tensor): A tensor containing the logits. Values
      should typically be between 0 and 1 for this loss to make sense, but
      values outside this range are clamped to a loss of 0.
    """
    loss = (1 - abs(2 * logits - 1)).mean()
    return clamp(loss, min=0.0, max=1.0)


def nonlinear_binary_loss(logits, coefficient=3):
    """
    Computes a loss that is maximized when the logits are close to 0.5
    and minimized when the logits move away from 0.5.

    The function peaks at 0.5 and is always positive.

    Parameters:
    - logits (torch.Tensor): A tensor containing the logits. Values
      should typically be between 0 and 1 for this loss to be meaningful.
    - coefficient (float, optional): A coefficient that determines the width of the curve.
      Defaults to 3. Increasing 'a' will make the function narrower.
    """
    return (1 / ((coefficient * (logits - 0.5)) ** 2 + 1)).mean()


def no_effect_loss(logits):
    """
    A dummy loss function that always returns zero.

    This function has no effect when added to other losses.

    Parameters:
    - logits (torch.Tensor): A tensor containing the logits.
      This input has no effect on the outcome of the function.

    Returns:
    - torch.Tensor: A single value tensor containing zero.
    """
    return tensor(0.0, requires_grad=False)


if __name__ == "__main__":
    tensor1 = torch.rand([3, 8, 8], requires_grad=True)

    linear_loss = linear_binary_loss(tensor1)
    nonlinear_loss = nonlinear_binary_loss(tensor1)
    no_effect = no_effect_loss(tensor1)

    # print("tensor1:", tensor1)
    print("linear_loss:", linear_loss)
    print("nonlinear_loss:", nonlinear_loss)
    print("no_effect:", no_effect)
