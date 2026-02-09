import torch
from pathlib import Path
from typing import List
from ffcv.loader import Loader, OrderOption
from ffcv.transforms import ToTensor, ToDevice, ToTorchImage, NormalizeImage
from ffcv.fields.decoders import IntDecoder, SimpleRGBImageDecoder

import numpy as np

# Standard ImageNet statistics
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


def get_ffcv_loader(
    beton_path: Path,
    batch_size: int,
    device,
    num_workers: int = 4,
    distributed: bool = False,
    drop_last: bool = True,
):
    """
    Creates an FFCV loader for the given .beton file.

    Args:
        beton_path: Path to the .beton file
        batch_size: Batch size
        device: Torch device (e.g., torch.device('cuda:0'))
        num_workers: Number of workers for data loading
        distributed: Whether to use distributed loading (not typically needed for single GPU)
        drop_last: Whether to drop the last incomplete batch
    """

    # Define pipelines
    # 1. Image pipeline: Decode -> ToTensor -> ToDevice -> ToTorchImage (CHW) -> Normalize
    image_pipeline: List[Operation] = [
        SimpleRGBImageDecoder(),
        ToTensor(),
        ToDevice(device, non_blocking=True),
        ToTorchImage(),
        NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float32),
    ]

    # 2. Label pipeline: Decode -> ToTensor -> ToDevice -> Squeeze
    label_pipeline: List[Operation] = [
        IntDecoder(),
        ToTensor(),
        ToDevice(device, non_blocking=True),
        torch.squeeze,
    ]

    # Order option: Random for training (usually handled by shuffle in standard loader),
    # but FFCV uses OrderOption.RANDOM
    # For validation, usually SEQUENTIAL
    is_train = "train" in beton_path.name.lower()
    order = OrderOption.RANDOM if is_train else OrderOption.SEQUENTIAL

    loader = Loader(
        fname=str(beton_path),
        batch_size=batch_size,
        num_workers=num_workers,
        order=order,
        drop_last=drop_last,
        pipelines={"image": image_pipeline, "label": label_pipeline},
        distributed=distributed,
    )

    return loader
