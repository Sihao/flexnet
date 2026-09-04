"""
using a gated sigmoid layer to determined how much information is used from each of the conv and pool

"""

from torch import nn


class SpatialAttentionBlockGated(nn.Module):
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

        # -------- gate layer --------
        self.gate_layer = nn.Sequential(
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=1, stride=1),
            nn.Sigmoid(),
        )

    def forward(self, t_flex_pool, t_flex_conv):
        t_flex_pool = self.blocks_pool(t_flex_pool)
        t_flex_conv = self.blocks_conv(t_flex_conv)
        gate_weights = self.gate_layer(t_flex_pool + t_flex_conv)
        combined = gate_weights * t_flex_pool + (1 - gate_weights) * t_flex_conv
        return combined

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
