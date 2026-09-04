""" """

from torch import nn
from torch.nn.init import kaiming_normal_


def apply_kaiming_initialization(model):
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Conv3d)):
            kaiming_normal_(module.weight, nonlinearity="leaky_relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)


if __name__ == "__main__":
    pass
