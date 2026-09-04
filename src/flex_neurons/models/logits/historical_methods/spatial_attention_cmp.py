"""
channel-wise soft max pool for spatial attention;
conv and pool at fed to the spatial attention block, and the output of the spatial attention block is a tensor of shape (8, 3, 32, 32, 2), where the last dimension is the softmax weights for the pooled and convolved tensors
"""

from torch import nn, cat
from torch.nn import functional as F


class SpatialAttentionBlockChannelMaxPool(nn.Module):
    def __init__(self, num_blocks, in_channels, out_channels, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_blocks = num_blocks
        self.in_channels, self.out_channels = in_channels, out_channels

        # -------- init block for pool --------
        self.blocks_pool = [
            block for _ in range(num_blocks) for block in self.spatial_attention_block()
        ]
        self.blocks_pool.extend(
            [
                nn.Conv2d(
                    self.in_channels,
                    self.out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
                nn.LeakyReLU(inplace=True),
                nn.BatchNorm2d(self.out_channels),
            ]
        )
        self.blocks_pool = nn.Sequential(*self.blocks_pool)

        # -------- init blocks for conv --------
        self.blocks_conv = [
            block for _ in range(num_blocks) for block in self.spatial_attention_block()
        ]
        self.blocks_conv.extend(
            [
                nn.Conv2d(
                    self.in_channels,
                    self.out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
                nn.LeakyReLU(inplace=True),
                nn.BatchNorm2d(self.out_channels),
            ]
        )
        self.blocks_conv = nn.Sequential(*self.blocks_conv)

    def forward(self, t_flex_pool, t_flex_conv):
        t_flex_pool = self.blocks_pool(t_flex_pool)
        t_flex_conv = self.blocks_conv(t_flex_conv)
        concatenated = cat(
            [t_flex_pool.unsqueeze(-1), t_flex_conv.unsqueeze(-1)], dim=-1
        )
        softmax_weights = F.softmax(concatenated, dim=-1)
        softmax_pooled = (softmax_weights * concatenated).sum(
            dim=-1
        )  # shape will be (8, 3, 32, 32)
        return softmax_pooled

    def spatial_attention_block(self):
        """
        a spatial attention block takes in a tensor and returns a tensor of the same shape
        """
        return [
            nn.Conv2d(
                self.in_channels, self.in_channels, kernel_size=3, stride=1, padding=1
            ),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(self.in_channels),
        ]
