import math, torch
import torch.nn as nn
from torch import stack
import torch.nn.functional as F
from src.modules.layers.flex import Flex2D
from src.modules.models.utils import DimensionTracer
from src.utils.general import apply_kaiming_initialization
from torch.nn import BatchNorm2d


class SimpleFlexNet(nn.Module):
    def __init__(self, config: dict, num_layers: int = 4, use_batch_norm: bool = False):
        super().__init__()
        self.config, self.__name__ = config, config.get("__name__", "SimpleFlexNet")
        self.num_layers, self.use_batch_norm = num_layers, use_batch_norm

        # ---- determine input dimensions and number of classes based on dataset ----
        if "cifar" in config.get("dataset", ""):
            self.in_dimensions, self.num_classes = (3, 32, 32), 10
        elif "imagenet" in config.get("dataset", ""):
            self.in_dimensions, self.num_classes = (3, 224, 224), 100
        else:
            raise ValueError(f"Dataset not supported: {config.get('dataset')}")

        # ---- determine the dimensions of the network ----
        dimension_tracer = DimensionTracer(self.in_dimensions)

        class Conv2d(Flex2D if config.get("use_flex", True) else nn.Conv2d):
            def __init__(self, *args, **kwargs):
                if config.get("use_flex", True):
                    kwargs["config"] = config
                super().__init__(*args, **kwargs)
                self.in_dimensions = dimension_tracer.calculate_dimension()
                dimension_tracer(**kwargs)
                self.out_dimensions = dimension_tracer.calculate_dimension()

                if isinstance(self, Flex2D):
                    self.init_dimension_dependent_modules()

        class MaxPool2d(nn.MaxPool2d):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                # -------- figure out all the in and out dimensions --------
                self.in_dimensions = dimension_tracer.calculate_dimension()
                dimension_tracer(**kwargs)
                self.out_dimensions = dimension_tracer.calculate_dimension()

            def forward(self, x):
                self.output, self.indices = F.max_pool2d(
                    x,
                    self.kernel_size,
                    self.stride,
                    self.padding,
                    self.dilation,
                    self.ceil_mode,
                    return_indices=True,
                )
                return self.output

        # -------- build the features block --------
        layers, in_channels, out_channels = [], 3, 16
        for i in range(self.num_layers):
            layers.append(
                Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    padding=1,
                )
            )
            if self.use_batch_norm:
                layers.append(BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            if (
                i + 1
            ) % 2 == 0 and out_channels < 512:  # MaxPool2d after every two Conv2d layers, stop if out_channels reaches 512
                layers.append(MaxPool2d(kernel_size=2, stride=2))
            in_channels = out_channels
            if out_channels < 512:
                out_channels *= 2
        self.features = nn.Sequential(*layers)

        # -------- build the classifier block --------
        default_dense_size = 128
        feature_flatten_size = math.prod(dimension_tracer.calculate_dimension())
        self.classifier = nn.Sequential(
            nn.Linear(
                in_features=feature_flatten_size, out_features=default_dense_size
            ),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=default_dense_size, out_features=default_dense_size),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=default_dense_size, out_features=self.num_classes),
        )

        apply_kaiming_initialization(self)

    def forward(self, x):
        x = self.features(x)
        x = x.reshape(x.size(0), -1)  # flatten feature tensor
        x = self.classifier(x)
        return x

    def check_homogeneity(self):
        """
        checking how binary each flex2d mask is
        this outputs a list of length num of flex2d used in the network
        each is a float between 0 and 1
        higher the more binary
        """
        binariness_values = []
        for module in self.modules():
            if isinstance(module, Flex2D):
                binariness_values.append(module.homogeneity)
        return binariness_values

    def _get_cp_id_matrices(self):
        assert self.config.get(
            "use_flex"
        ), "This model does not use Flex2D layers for checking conv ratio."
        cp_identity_matrices = []
        for module in self.modules():
            if isinstance(module, Flex2D):
                cp_identity_matrix = (module.cp_identity_matrix >= 0.5).float()
                cp_identity_matrices.append(cp_identity_matrix)
        return cp_identity_matrices

    def check_conv_ratio(self):
        """
        check the ratio of conv in each flex2d mask
        defined as the ratio of values above 0.5
        this outputs a list of length num of flex2d used in the network
        each is a float between 0 and 1
        higher the more conv
        """
        conv_ratios = []
        for module in self.modules():
            if isinstance(module, Flex2D):
                conv_ratios.append(module.conv_ratio)
        return conv_ratios
