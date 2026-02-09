"""
this is the branch of the sptial attention block that concatenates the pooled and convolved tensors on the channel dimension and then applies a convolutional layer on top of it for the final logits

"""

import torch
from torch import nn
from torch import cat
from monai.networks.nets import UNet
from torch.nn import functional as F
from src.utils.device import select_device


class UNetBlock(nn.Module):
    def __init__(self, channels, out_dimensions, config=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.channels, self.out_dimensions = channels, out_dimensions
        self.unet_channels = config.get("unet_channels", (16, 32, 64))
        self.device = select_device()
        assert len(self.unet_channels) == 3, "unet_channels must be a tuple of length 3"

        # ------- make strides depending on the out dimensions --------
        # if any of the out dimensions are too small, a large stride will cause the output to be smaller than 1
        if min(self.out_dimensions) < 4:
            self.strides = (1, 2)
        else:
            self.strides = (2, 2)

        # -------- init UNet using monai --------
        self.unet = UNet(
            spatial_dims=3,
            in_channels=2,  # pool and conv
            out_channels=1,  # logits
            channels=self.unet_channels,
            strides=self.strides,
            num_res_units=2,
            act="PRELU",
            norm="INSTANCE",
            adn_ordering="NDA",
            dropout=config.get("dropout", 0.2),
        ).to(self.device)

        # -------- init a UNet for when the width and height are too small --------
        # self.unet_

        if channels % 2 != 0:  # if is odd 1d conv to even for unet to work
            self.odd_channels = True
            self.conv_block_expand_pool = self._make_1d_conv(channels, channels + 1)
            self.conv_block_expand_conv = self._make_1d_conv(channels, channels + 1)
            self.conv_block_shrink = self._make_1d_conv(channels + 1, channels)
        else:
            self.odd_channels = False

    def forward(self, t_flex_pool, t_flex_conv):
        # ======= start checks =======
        assert (
            t_flex_pool.shape == t_flex_conv.shape
        ), "pool and conv tensors must have the same shape"
        assert t_flex_pool.dim() in [3, 4], "pool tensor must have 3 or 4 dimensions"
        if self.odd_channels:
            t_flex_pool = self.conv_block_expand_pool(t_flex_pool)
            t_flex_conv = self.conv_block_expand_conv(t_flex_conv)
        # ======= finish checks =======

        # Reshape tensors from [B, C, H, W] to [B, 1, C, H, W]
        t_flex_pool = t_flex_pool.unsqueeze(1)
        t_flex_conv = t_flex_conv.unsqueeze(1)

        # Concatenate tensors along the new channel dimension to get [B, 2, C, H, W]
        concatenated = cat([t_flex_pool, t_flex_conv], dim=1)
        output = self.unet(concatenated).squeeze(1)  # problem line

        if self.odd_channels:
            output = self.conv_block_shrink(output)
        return output

    def _make_1d_conv(self, in_channels, out_channels):
        """
        a spatial attention block takes in a tensor and returns a tensor of the same shape
        """
        modules = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
        ]
        return nn.Sequential(*modules)


if __name__ == "__main__":
    tensor1 = torch.rand(8, 3, 32, 32)
    tensor2 = torch.rand(8, 3, 32, 32)
    block = UNetBlock()
    output = block(tensor1, tensor2)
    print(output.shape)
