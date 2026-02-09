import torch

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class Normalize(torch.nn.Module):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.register_buffer("mean", torch.Tensor(mean).reshape(1, 3, 1, 1))
        self.register_buffer("std", torch.Tensor(std).reshape(1, 3, 1, 1))

    def forward(self, x):
        return (x - self.mean) / self.std


def denormalize_batch(x, device):
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1).to(device)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1).to(device)
    return x * std + mean
