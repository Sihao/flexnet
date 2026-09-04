import torch

"""
for some reasons using BP (plain backpropagation) with STE (Straight-Through Estimator) makes the params / gardient all nan values after training for some epochs (only after training)
"""


class STE_FPBP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return (
            input > 0.5
        ).float()  # The forward pass applies a step function, creating a binary mask

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output  # The backward pass lets gradients pass through unchanged


class STE_FSBP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        input = torch.sigmoid(input)
        return (
            input > 0.5
        ).float()  # The forward pass applies a step function, creating a binary mask

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output  # The backward pass lets gradients pass through unchanged


class STE_FPBS(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return (
            input > 0.5
        ).float()  # The forward pass applies a step function, creating a binary mask

    @staticmethod
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        sigmoid_input = torch.sigmoid(input)
        grad_sigmoid = sigmoid_input * (1 - sigmoid_input)
        return (
            grad_output * grad_sigmoid
        )  # The backward pass uses the gradient of a sigmoid


class STE_FSBS(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        input = torch.sigmoid(input)
        return (input > 0.5).float()

    @staticmethod
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        sigmoid_input = torch.sigmoid(input)
        grad_sigmoid = sigmoid_input * (1 - sigmoid_input)
        return grad_output * grad_sigmoid
