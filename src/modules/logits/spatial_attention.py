"""
this is the branch of the sptial attention block that concatenates the pooled and convolved tensors on the channel dimension and then applies a convolutional layer on top of it for the final logits

"""

from torch import nn, cat


class SpatialAttentionBlock(nn.Module):
    def __init__(self, num_blocks, in_channels, out_channels, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # -------- init attributes --------
        self.num_blocks, self.in_channels, self.out_channels = (
            num_blocks,
            in_channels,
            out_channels,
        )

        # -------- init layers --------
        self.blocks = [
            block
            for _ in range(num_blocks)
            for block in self._make_spatial_attention_block()
        ]

        self.blocks.extend(
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

        self.blocks = nn.Sequential(*self.blocks)

    def forward(self, x):
        output = self.blocks(x)
        return output

    def _make_spatial_attention_block(self):
        """
        a spatial attention block takes in a tensor and returns a tensor of the same shape
        """
        return [
            nn.Conv2d(
                self.in_channels,
                self.in_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(self.in_channels),
        ]
