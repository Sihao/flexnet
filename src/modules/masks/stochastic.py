import torch


class StochasticRoundFPBP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        floor, ceil = input.floor(), input.ceil()
        probability_to_ceil = input - floor
        decisions = torch.rand_like(input)
        output = torch.where(decisions < probability_to_ceil, ceil, floor)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class StochasticRoundFPBS(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        floor, ceil = input.floor(), input.ceil()
        probability_to_ceil = input - floor
        decisions = torch.rand_like(input)
        output = torch.where(decisions < probability_to_ceil, ceil, floor)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        sigmoid_input = torch.sigmoid(input)
        grad_sigmoid = sigmoid_input * (1 - sigmoid_input)
        return grad_output * grad_sigmoid


class StochasticRoundFSBP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        input = torch.sigmoid(input)
        ctx.save_for_backward(input)
        floor, ceil = input.floor(), input.ceil()
        probability_to_ceil = input - floor
        decisions = torch.rand_like(input)
        output = torch.where(decisions < probability_to_ceil, ceil, floor)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class StochasticRoundFSBS(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        input = torch.sigmoid(input)
        ctx.save_for_backward(input)
        floor, ceil = input.floor(), input.ceil()
        probability_to_ceil = input - floor
        decisions = torch.rand_like(input)
        output = torch.where(decisions < probability_to_ceil, ceil, floor)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        sigmoid_input = torch.sigmoid(input)
        grad_sigmoid = sigmoid_input * (1 - sigmoid_input)
        return grad_output * grad_sigmoid


class StochasticRoundFCBP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        input = input.clamp(0, 1)  # wow work so much much better than sigmoid damnit
        ctx.save_for_backward(input)
        floor, ceil = input.floor(), input.ceil()
        probability_to_ceil = input - floor
        decisions = torch.rand_like(input)
        output = torch.where(decisions < probability_to_ceil, ceil, floor)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class StochasticRoundFCBS(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        input = input.clamp(0, 1)  # wow work so much much better than sigmoid damnit
        ctx.save_for_backward(input)
        floor, ceil = input.floor(), input.ceil()
        probability_to_ceil = input - floor
        decisions = torch.rand_like(input)
        output = torch.where(decisions < probability_to_ceil, ceil, floor)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        sigmoid_input = torch.sigmoid(input)
        grad_sigmoid = sigmoid_input * (1 - sigmoid_input)
        return grad_output * grad_sigmoid
