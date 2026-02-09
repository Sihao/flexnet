import torch
from torch.nn import functional as F


def channel_repeat(tensor, out_channels):
    """
    Repeats the channels of the tensor to match the desired out_channels.
    Assumes the input tensor is of shape (B, C, H, W).
    """
    _, in_channels, _, _ = tensor.size()
    repeats, remainder = out_channels // in_channels, out_channels % in_channels
    tensor_repeated = tensor.repeat(1, repeats, 1, 1)
    if remainder:
        tensor_remainder = tensor[:, :remainder, :, :]
        tensor_repeated = torch.cat((tensor_repeated, tensor_remainder), dim=1)
    return tensor_repeated


def channel_interpolate(tensor, out_channels):
    """
    Interpolates a tensor along the channel axis. This is for addressing the mismatch in the number of channels between the output of the maxpool layer and the output of the convolutional layer.
    """
    tensor = tensor.permute(0, 2, 1, 3)
    return tensor


def channel_expand_view(tensor, out_channels):
    """
    Expands the channels of the tensor to match the desired out_channels using views if possible.
    This avoids memory allocation compared to repeat or interpolate.
    Assumes the input tensor is of shape (B, C, H, W).
    """
    B, in_channels, H, W = tensor.size()

    # Perfect divisibility check
    if out_channels % in_channels == 0:
        k = out_channels // in_channels
        # (B, C, H, W) -> (B, C, 1, H, W) -> (B, C, k, H, W) -> (B, C*k, H, W)
        # This returns a view
        return (
            tensor.unsqueeze(2)
            .expand(B, in_channels, k, H, W)
            .reshape(B, out_channels, H, W)
        )
    else:
        # Fallback to copy-based repetition if not perfectly divisible
        return channel_repeat(tensor, out_channels)
