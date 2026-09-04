import torch
from torch import cat


def channel_wise_maxpool(tensor_1, tensor_2):
    """
    Take two tensors of identical shape and return a tensor of the same shape using element-wise max pooling.
    Also returns the ratio of values from tensor_2 to the total.
    """
    assert (
        tensor_1.shape == tensor_2.shape
    ), "tensor_1 and tensor_2 must have the same shape"

    # OLD:
    # joint = cat([tensor_1.unsqueeze(-1), tensor_2.unsqueeze(-1)], dim=-1)
    # pooled, indices = torch.max(joint, dim=-1)

    # NEW:
    # Use element-wise maximum to avoid intermediate allocation of 'joint' (which is 2x size)
    # tensor_1 and tensor_2 might be views (if expanded), but maximum handles them efficiently
    pooled = torch.maximum(tensor_1, tensor_2)

    # count values are from conv (tensor 2)
    with torch.no_grad():
        # indices == 1 was used when we did max(stack([pool, conv])), so index 1 meant conv won.
        # Now we compare: if tensor_2 > tensor_1, then conv won.
        # Note: Tie-breaking behavior might differ slightly (max prefers first index usually?),
        # but for floats equality is rare.
        # Actually torch.max(dim) returns the first index of maximum value.
        # So if pool == conv, index was 0 (pool).
        # So here we want (tensor_2 > tensor_1) which is strict greater.
        # If equal, it is False (0) -> pool. Result matches.

        is_conv = tensor_2 > tensor_1
        conv_ratio = is_conv.sum().item() / tensor_1.numel()
        cp_identity_matrix = is_conv.int()

    return pooled, conv_ratio, cp_identity_matrix


if __name__ == "__main__":
    tensor1 = torch.rand(8, 3, 32, 32)
    tensor2 = torch.rand(8, 3, 32, 32)
    joint, ratio, cp_identity_matrix = channel_wise_maxpool(tensor1, tensor2)

    print(joint.shape)
    print(f"Ratio of values from tensor1 to total: {ratio:.4f}")
