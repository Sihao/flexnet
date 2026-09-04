import warnings

import torch
from torchvision import transforms

try:
    from torchvision.transforms import RandAugment
except ImportError:
    RandAugment = None

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_train_transform(config=None):
    """
    Build the training transform pipeline.

    Reads `aug_recipe` from `config` (default "basic"):
      - "basic": the original ImageNet train transform (RandomResizedCrop +
        RandomHorizontalFlip + ToTensor + Normalize).
      - "modern": adds RandAugment, ColorJitter, and RandomErasing on top of
        the basic pipeline.
    """
    config = config or {}
    aug_recipe = config.get("aug_recipe", "basic")

    if aug_recipe == "basic":
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )
    elif aug_recipe == "modern":
        ops = [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
        ]
        if RandAugment is not None:
            ops.append(RandAugment(num_ops=2, magnitude=9))
        else:
            warnings.warn(
                "torchvision.transforms.RandAugment is unavailable "
                "(requires a newer torchvision); skipping RandAugment "
                "in the 'modern' aug_recipe."
            )
        ops.extend(
            [
                transforms.ColorJitter(0.4, 0.4, 0.4),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                transforms.RandomErasing(p=0.25, value="random"),
            ]
        )
        return transforms.Compose(ops)
    else:
        raise ValueError(f"Unknown aug_recipe: {aug_recipe}")


def build_val_transform(config=None):
    """Build the validation transform pipeline (Resize + CenterCrop + ToTensor + Normalize)."""
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


class AddGaussianNoise(object):
    def __init__(self, mean=0.0, std=1.0):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        noise = torch.randn(tensor.size()) * self.std + self.mean
        return tensor + noise

    def __repr__(self):
        return self.__class__.__name__ + "(mean={0}, std={1})".format(
            self.mean, self.std
        )
