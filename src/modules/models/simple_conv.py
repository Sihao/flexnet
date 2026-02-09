import torch
import torch.nn as nn
from src.utils.general import apply_kaiming_initialization
from src.modules import masks


class SimpleConvNet(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(256 * 2 * 2, 128)
        self.fc2 = nn.Linear(128, 10)

        self.activation = getattr(masks, config["custom_activation"])()

        apply_kaiming_initialization(self)

    def forward(self, x):
        x = self.pool(self.activation.apply(self.conv1(x)))
        x = self.pool(self.activation.apply(self.conv2(x)))
        x = self.pool(self.activation.apply(self.conv3(x)))
        x = self.pool(self.activation.apply(self.conv4(x)))
        x = torch.flatten(x, 1)
        x = self.activation.apply(self.fc1(x))
        x = self.fc2(x)
        return x
