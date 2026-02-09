"""
- VoxelToVector:
    This class is designed to convert a 3D voxel to a vector representation using a 3D convolution layer. It has one method: forward, which takes a tensor of shape (channel, height, width) and transforms it using a convolutional layer.

"""

from torch import nn


class VoxelToVectorConv(nn.Module):
    def __init__(self, additional_dim=768):
        super().__init__()
        self.conv3d_layer = nn.Conv3d(
            in_channels=1, out_channels=additional_dim, kernel_size=(1, 1, 1)
        )

    def forward(self, x):
        """x = x.permute(0, 4, 2, 3, 1); if we have a batch dimension"""
        error = f"Input tensor should be of shape (batch, channel, height, width), but got {x.shape}"
        assert len(x.shape) == 4, error
        x = x.unsqueeze(-1)
        x = x.permute(0, 4, 2, 3, 1)
        x = self.conv3d_layer(x)
        x = x.permute(0, 4, 2, 3, 1)

        return x
