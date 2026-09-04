"""
This script is about adding auxiliary loss to the model that regularizes the logits to be bimodal / binary
- there are two ways to do this at the time of writing this script:
1. implement hooks in the model structure and collects the logits at the desired layers, then add them to the main loss
2. using the autograd function to add the auxiliary loss to the logits at the desired layers so we dont have to add hooks everywhere

but this script is about the second method, which is more elegant and easier to implement
but it is not working properly at the time of writing this script, so we will use the first method for now
# -------- [about logits regulation] --------
# we are now giving up on the logits regulation as there seems to be clear effect
# or benefits other than lowering the train and test accuracy
# plus, now that we are not fixed on the idea of making the neurons completely binary I feel
# this parameter can be dropped for now
# but it is still unclear whether using a logits regulation will cause the models
# to be more rebust against adversarial attacks
# although there is no reason to believe so (that it will), still worth some test if time allows
# settings("tau", [20])  # temperature for smoothed sigmoid
# settings("desired_final_tau", [1 / 50])  # same as *50

"""

import torch
import torch.nn as nn
import torch.optim as optim
from src.modules.loss.binary import nonlinear_binary_loss


class AddAuxiliaryLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, auxiliary_loss_fn, weight_coef):
        ctx.save_for_backward(logits)
        ctx.auxiliary_loss_fn = auxiliary_loss_fn
        ctx.weight_coef = weight_coef
        return logits.clone()

    @staticmethod
    def backward(ctx, grad_output):
        (logits,) = ctx.saved_tensors
        auxiliary_loss_fn = ctx.auxiliary_loss_fn
        weight_coef = ctx.weight_coef
        auxiliary_loss = auxiliary_loss_fn(logits)
        grad_aux = torch.autograd.grad(
            auxiliary_loss, logits, retain_graph=True, allow_unused=True
        )[0]
        if grad_aux is None:
            print("zero grad_aux")
            grad_aux = torch.zeros_like(logits)
        grad_logits = (
            grad_output + weight_coef * grad_aux
        )  # Multiply grad_aux by the coefficient
        return grad_logits, None, None  # Added an additional None for the weight_coef


def add_auxiliary_loss(
    logits, auxiliary_loss_fn, weight_coef=1.0
):  # Default coefficient is 1
    return AddAuxiliaryLoss.apply(logits, auxiliary_loss_fn, weight_coef)


if __name__ == "__main__":
    # Define a two-layer model with auxiliary losses
    class SimpleModel(nn.Module):
        def __init__(self):
            super(SimpleModel, self).__init__()
            self.fc1 = nn.Linear(5, 5)
            self.fc2 = nn.Linear(5, 1)

        def forward(self, x):
            x = self.fc1(x)
            x = add_auxiliary_loss(
                x, nonlinear_binary_loss
            )  # Apply auxiliary loss after first layer
            x = self.fc2(x)
            x = add_auxiliary_loss(
                x, nonlinear_binary_loss
            )  # Apply auxiliary loss after second layer
            return x

    # Dataset: X is input, Y is target
    X = torch.randn(10, 5)
    Y = torch.randn(10, 1)

    model = SimpleModel()
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)

    # Training loop
    for epoch in range(10000):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, Y)
        loss.backward()
        optimizer.step()

    print("Training Complete!")
