import torch
import torch.nn as nn
import math
from src.modules.layers.flex import Flex2D
from src.modules.models.utils import DimensionTracer
from src.utils.general import apply_kaiming_initialization
from src.modules.layers.flex import MaxPool2d as FlexMaxPool2d

# Note: FlexMaxPool2d in flex.py is just a wrapper for returning indices,
# but we might need dimension tracking wrapper.
# We will define a local wrapper.

# -----------------------------------------------------------------------------
# Wrappers to handle Flex2D + DimensionTracer
# -----------------------------------------------------------------------------


class FlexConvWrapper(nn.Module):
    """
    A wrapper that conditionally acts as Flex2D or nn.Conv2d based on config.
    It integrates with DimensionTracer to calculate/update input/output dimensions.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        config=None,
        tracer=None,
        track_dimension=True,
        bias=False,
    ):
        super().__init__()

        use_flex = config.get("use_flex", True) if config else False

        # Dimensions
        if tracer:
            self.in_dimensions = tracer.calculate_dimension()
            if track_dimension:
                tracer(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                )
                self.out_dimensions = tracer.calculate_dimension()
            else:
                # If not tracking, we just calculate what the output WOULD be locally for initialization
                # but don't update the global tracer state.
                # However, Flex2D .init_dimension_dependent_modules() needs self.out_dimensions
                # So we must calculate it manually without modifying tracer registry.
                c, w, h = self.in_dimensions
                F = kernel_size
                P = padding
                S = stride
                w_out = (w - F + 2 * P) // S + 1
                h_out = (h - F + 2 * P) // S + 1
                self.out_dimensions = (out_channels, w_out, h_out)

        if use_flex:
            self.layer = Flex2D(
                in_channels, out_channels, kernel_size, stride, padding, config=config
            )
            # Initialize threshold if needed
            self.layer.in_dimensions = getattr(self, "in_dimensions", None)
            self.layer.out_dimensions = getattr(self, "out_dimensions", None)
            if hasattr(self, "out_dimensions"):
                self.layer.init_dimension_dependent_modules()
        else:
            self.layer = nn.Conv2d(
                in_channels, out_channels, kernel_size, stride, padding, bias=bias
            )

    def forward(self, x):
        return self.layer(x)


def conv3x3(in_planes, out_planes, stride=1, config=None, tracer=None):
    """3x3 convolution with padding"""
    return FlexConvWrapper(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        config=config,
        tracer=tracer,
        bias=False,
    )


def conv1x1(
    in_planes, out_planes, stride=1, config=None, tracer=None, track_dimension=True
):
    """1x1 convolution"""
    return FlexConvWrapper(
        in_planes,
        out_planes,
        kernel_size=1,
        stride=stride,
        padding=0,
        config=config,
        tracer=tracer,
        track_dimension=track_dimension,
        bias=False,
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self, inplanes, planes, stride=1, downsample=None, config=None, tracer=None
    ):
        super(BasicBlock, self).__init__()

        # Main path: updates tracer
        self.conv1 = conv3x3(inplanes, planes, stride, config, tracer)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, config=config, tracer=tracer)
        self.bn2 = nn.BatchNorm2d(planes)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self, inplanes, planes, stride=1, downsample=None, config=None, tracer=None
    ):
        super(Bottleneck, self).__init__()

        self.conv1 = conv1x1(inplanes, planes, stride=1, config=config, tracer=tracer)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = conv3x3(
            planes, planes, stride=stride, config=config, tracer=tracer
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = conv1x1(
            planes, planes * self.expansion, stride=1, config=config, tracer=tracer
        )
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class FlexResNet(nn.Module):
    def __init__(self, config: dict):
        super(FlexResNet, self).__init__()
        self.config = config
        self.use_flex = config.get("use_flex", True)
        self.__name__ = "FlexResNet"

        # Determine dimensions
        if "cifar" in config.get("dataset", ""):
            self.in_dimensions, self.num_classes = (3, 32, 32), 10
            # CIFAR ResNet typically starts with 16 channels, 6n+2 layers
            # e.g., depth 32 -> n=5
            self.base_channel = 16
        elif "imagenet" in config.get("dataset", ""):
            self.in_dimensions, self.num_classes = (3, 224, 224), 100  # ImageNet100
            self.base_channel = 64  # Standard ResNet uses 64
        else:
            raise ValueError(f"Dataset not supported: {config.get('dataset')}")

        self.tracer = DimensionTracer(self.in_dimensions)

        # Parse depth
        # We look for 'resnet_depth' in config, or infer from somewhere.
        # Default to 50 to mimic ResNet-50 as requested.
        depth = config.get("resnet_depth", 50)

        # Determine block type and count
        if depth == 50:
            block = Bottleneck
            # Standard ResNet-50 counts: [3, 4, 6, 3]
            layers_count = [3, 4, 6, 3]
        elif depth >= 50:
            # Fallback for other deep variants if needed, or stick to CIFAR logic?
            # Let's assume if it's not 50, it triggers the old logic,
            # OR we can just error if it's not a supported standard config.
            # User asked to mimic ResNet-50.
            block = Bottleneck
            n = (depth - 2) // 6
            layers_count = [n, n, n]
        else:
            block = BasicBlock
            assert (depth - 2) % 6 == 0, "depth should be 6n+2"
            n = (depth - 2) // 6
            layers_count = [n, n, n]

        self.inplanes = self.base_channel

        # Initial Conv
        if (
            "imagenet" in config.get("dataset", "") or depth == 50
        ):  # Assume ResNet-50 implies ImageNet-like structure
            # Standard ImageNet ResNet Start
            self.conv1 = FlexConvWrapper(
                self.in_dimensions[0],
                self.inplanes,
                kernel_size=7,
                stride=2,
                padding=3,
                config=config,
                tracer=self.tracer,
                bias=False,
            )
            self.bn1 = nn.BatchNorm2d(self.inplanes)
            self.relu = nn.ReLU(inplace=True)

            # MaxPool for ImageNet - also needs dimension tracking
            self.tracer(kernel_size=3, stride=2, padding=1)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        else:
            # CIFAR Start
            self.conv1 = FlexConvWrapper(
                self.in_dimensions[0],
                self.inplanes,
                kernel_size=3,
                stride=1,
                padding=1,
                config=config,
                tracer=self.tracer,
                bias=False,
            )
            self.bn1 = nn.BatchNorm2d(self.inplanes)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.Identity()

        # Stack Layers
        self.layer1 = self._make_layer(
            block, self.base_channel, layers_count[0], stride=1
        )
        self.layer2 = self._make_layer(
            block, self.base_channel * 2, layers_count[1], stride=2
        )
        self.layer3 = self._make_layer(
            block, self.base_channel * 4, layers_count[2], stride=2
        )

        # Layer 4 (Standard ResNet has 4 stages)
        if len(layers_count) >= 4:
            self.layer4 = self._make_layer(
                block, self.base_channel * 8, layers_count[3], stride=2
            )
        else:
            self.layer4 = nn.Identity()

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Final dimensions
        if len(layers_count) >= 4:
            final_channels = self.base_channel * 8 * block.expansion
        else:
            final_channels = self.base_channel * 4 * block.expansion

        self.fc = nn.Linear(final_channels, self.num_classes)

        apply_kaiming_initialization(self)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            # Downsample path
            # We must NOT update the main tracer here because the Block's main path will do it.
            # But we need to construct it using dimensions.
            # We pass track_dimension=False
            downsample = nn.Sequential(
                conv1x1(
                    self.inplanes,
                    planes * block.expansion,
                    stride,
                    config=self.config,
                    tracer=self.tracer,
                    track_dimension=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                config=self.config,
                tracer=self.tracer,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(self.inplanes, planes, config=self.config, tracer=self.tracer)
            )

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    # -------------------------------------------------------------------------
    # Flex2D introspection methods
    # -------------------------------------------------------------------------

    def check_homogeneity(self):
        binariness_values = []
        for module in self.modules():
            if isinstance(module, Flex2D):
                binariness_values.append(module.homogeneity)
                # If FlexConvWrapper hides Flex2D in .layer?
        # Actually self.modules() recurses, so it should find the Flex2D instances inside FlexConvWrapper
        return binariness_values

    def check_conv_ratio(self):
        conv_ratios = []
        for module in self.modules():
            if isinstance(module, Flex2D):
                conv_ratios.append(module.conv_ratio)
        return conv_ratios

    def _get_cp_id_matrices(self):
        cp_identity_matrices = []
        for module in self.modules():
            if isinstance(module, Flex2D):
                cp_identity_matrix = (module.cp_identity_matrix >= 0.5).float()
                cp_identity_matrices.append(cp_identity_matrix)
        return cp_identity_matrices
