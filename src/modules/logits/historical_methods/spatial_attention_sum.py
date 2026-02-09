""" """

import torch
from torch import nn


class SpatialAttentionBlockPreSum(nn.Module):
    def __init__(self, num_blocks, in_channels, out_channels, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_blocks = num_blocks
        self.in_channels, self.out_channels = in_channels, out_channels

        # -------- init layers --------
        self.blocks = []
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

    def forward(self, t_flex_pool, t_flex_conv):
        combined_logits = t_flex_pool + t_flex_conv
        output = self.blocks(combined_logits)
        return output

    def _make_spatial_attention_block(self):
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


if __name__ == "__main__":
    tensor1 = torch.randn(8, 3, 32, 32)
    tensor2 = torch.randn(8, 3, 32, 32)
    spatial_attention_block = SpatialAttentionBlockPreSum(
        num_blocks=2, in_channels=3, out_channels=3
    )
    output = spatial_attention_block(tensor1, tensor2)
    print(output.shape)
