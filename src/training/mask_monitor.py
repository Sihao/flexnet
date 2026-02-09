"""
- measuring bimodality of a tensor.
This is applied on the mask tensor to measure how bimodal it is during training.s

- counting the ratio of convolutions in a tensor.
This is applied on the mask tensor to measure how many convolutions are in the mask.

"""

import torch
from torch import Tensor


def measure_homogeneity(tensor: Tensor):
    """
    [tested]
    Compute the bimodality measure of a tensor.
    Note:
        All 0 or all 1 will return high bimodality.
        This function assumes that no gradient should be computed.
    Returns:
        float: A bimodality measure ranging from 0 to 1, where a larger value
        indicates a more bimodal tensor.
    """
    with torch.no_grad():
        return torch.mean(torch.abs(tensor - 0.5) * 2).item()


def count_conv_ratio(tensor: Tensor):
    """
    [tested]
    value in mask above 0.5 = conv; count the ratio of values above 0.5
    Returns:
        float: Ratio of values in the tensor that are above 0.5.
    """
    num_conv = torch.sum(tensor > 0.5).item()  # not differentiable
    num_total = tensor.numel()
    return num_conv / num_total


def count_conv_ratio_learnable(tensor: Tensor):
    # num_conv = torch.sum(torch.sigmoid(100 * (tensor - 0.5)))
    num_conv = torch.sum(torch.sigmoid(tensor - 0.5))
    num_total = tensor.numel()
    return num_conv / num_total


if __name__ == "__main__":
    # make a dummy tensor of size 10 of random values between 0 and 1
    tensor1 = torch.rand(10)
    tensor2 = torch.ones(10)
    tensor3 = torch.zeros(10)
    tensor4 = torch.cat((tensor2[:5], tensor3[5:]))

    for t in [tensor1, tensor2, tensor3, tensor4]:
        print(measure_homogeneity(t))

    # make a tensor that is 20% 0.7 and rest 0.3
    tensor1 = torch.ones(20)
    tensor2 = torch.zeros(10)
    tensor3 = torch.cat([tensor1, tensor2], dim=0)
    print(count_conv_ratio(tensor3))
