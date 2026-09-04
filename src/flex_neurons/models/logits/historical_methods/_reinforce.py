"""
REINFORCE algorithm and variance reduction techniques for making a mask of binary decisions
utilize policy gradient methods like REINFORCE.
also Variance reduction techniques (like using a baseline) can help stabilize training.

[NOTE] Not included in further development as these probably too away for the current project's point of interest.
"""

import torch
from torch import nn
from torch.autograd import Function


## NOTE: need to update the reward externally


class REINFORCEBernoulli(Function):
    @staticmethod
    def forward(ctx, prob):
        sample = torch.bernoulli(prob)
        ctx.save_for_backward(prob, sample)
        return sample

    @staticmethod
    def backward(ctx, grad_output):
        prob, sample = ctx.saved_tensors
        # Placeholder gradient, you need to multiply by (reward - baseline) externally
        return grad_output * (sample - prob)


class REINFORCEDecisionMaker(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # you might want to shape it according to t_flex_pool shape
        # shape of [batch_size, channels, height, width]
        self.prob = nn.Parameter(
            torch.ones(1, 1, 1, 1) * 0.5
        )  # Initialized to 0.5, but you can choose any initialization

    def forward(self, t_flex_pool, t_flex_conv):
        # Just an example, you can define any operation with t_flex_pool and t_flex_conv to determine final prob
        prob = torch.sigmoid(self.prob + t_flex_pool.mean() - t_flex_conv.mean())

        # Applying the custom REINFORCE Bernoulli decision
        return REINFORCEBernoulli.apply(prob)


if __name__ == "__main__":
    model = REINFORCEDecisionMaker()
    t_flex_pool = torch.randn(32, 64, 8, 8)
    t_flex_conv = torch.randn(32, 64, 8, 8)
    output = model(t_flex_pool, t_flex_conv)
