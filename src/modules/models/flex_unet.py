import torch
import torch.nn as nn
import torch.nn.functional as F
from src.modules.layers.flex import Flex2D


class FlexBlock(nn.Module):
    """
    Flex2D -> BN -> ReLU
    """

    def __init__(self, in_channels, out_channels, config):
        super().__init__()
        self.block = nn.Sequential(
            Flex2D(in_channels, out_channels, kernel_size=3, padding=1, config=config),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class StandardBlock(nn.Module):
    """
    Standard Conv2d -> BN -> ReLU
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class FlexUNet(nn.Module):
    def __init__(self, config, in_channels=None):
        super().__init__()
        self.config = config

        if in_channels is None:
            # Default for DeepInterpolation: 2 * N
            N = config.get("frame_window_N", 30)
            in_channels = 2 * N

        self.in_channels = in_channels

        # Determine block type
        # Check for vanilla flag
        is_vanilla = config.get("vanilla", False)
        # Use StandardBlock if vanilla, else FlexBlock
        Block = (
            StandardBlock
            if is_vanilla
            else lambda in_c, out_c, conf: FlexBlock(in_c, out_c, conf)
        )

        # Helper for StandardBlock which doesn't take config
        if is_vanilla:

            def Block(in_c, out_c, conf):
                return StandardBlock(in_c, out_c)

        else:

            def Block(in_c, out_c, conf):
                return FlexBlock(in_c, out_c, conf)

        # --- Encoder (Flex or Vanilla) ---
        # Level 1
        self.inc = Block(in_channels, 64, config)
        self.pool1 = nn.MaxPool2d(2)

        # Level 2
        self.down2_conv = Block(64, 128, config)
        self.pool2 = nn.MaxPool2d(2)

        # Level 3
        self.down3_conv = Block(128, 256, config)
        self.pool3 = nn.MaxPool2d(2)

        # Level 4
        self.down4_conv = Block(256, 512, config)
        self.pool4 = nn.MaxPool2d(2)

        # --- Bridge (Flex or Vanilla) ---
        # 512 -> 1024
        self.bridge = Block(512, 1024, config)

        # --- Decoder (Standard) ---
        # Up 1
        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.up1_conv = StandardBlock(1024, 512)  # 512 from up + 512 from skip

        # Up 2
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up2_conv = StandardBlock(512, 256)

        # Up 3
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up3_conv = StandardBlock(256, 128)

        # Up 4
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up4_conv = StandardBlock(128, 64)

        # --- Output Head ---
        # 1x1 Conv acts as a pixel-wise linear projection to 1 channel
        self.outc = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)  # [B, 64, H, W]
        p1 = self.pool1(x1)  # [B, 64, H/2, W/2]

        x2 = self.down2_conv(p1)  # [B, 128, H/2, W/2]
        p2 = self.pool2(x2)  # [B, 128, H/4, W/4]

        x3 = self.down3_conv(p2)  # [B, 256, H/4, W/4]
        p3 = self.pool3(x3)  # [B, 256, H/8, W/8]

        x4 = self.down4_conv(p3)  # [B, 512, H/8, W/8]
        p4 = self.pool4(x4)  # [B, 512, H/16, W/16]

        # Bridge
        x5 = self.bridge(p4)  # [B, 1024, H/16, W/16]

        # Decoder
        u1 = self.up1(x5)  # [B, 512, H/8, W/8]
        # Pad if necessary? Assuming power of 2 input for now.
        # Concatenate skip connection x4
        d1 = torch.cat([x4, u1], dim=1)
        x6 = self.up1_conv(d1)

        u2 = self.up2(x6)  # [B, 256, H/4, W/4]
        d2 = torch.cat([x3, u2], dim=1)
        x7 = self.up2_conv(d2)

        u3 = self.up3(x7)  # [B, 128, H/2, W/2]
        d3 = torch.cat([x2, u3], dim=1)
        x8 = self.up3_conv(d3)

        u4 = self.up4(x8)  # [B, 64, H, W]
        d4 = torch.cat([x1, u4], dim=1)
        x9 = self.up4_conv(d4)

        out = self.outc(x9)
        return out
