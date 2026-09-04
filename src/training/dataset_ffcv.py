"""FFCV fast data loading for ImageNet.

The pipelines here reproduce the torchvision transforms used on the JPEG path
(src/training/dataset_select.py:62-77) so that an FFCV run is comparable to a
non-FFCV run:

  * train: RandomResizedCrop(224) + RandomHorizontalFlip + Normalize
  * val:   Resize(256) + CenterCrop(224) + Normalize   (== CenterCrop ratio 224/256)

FFCV's NormalizeImage runs on the raw uint8 [0,255] image, so the ImageNet
mean/std are scaled by 255 to match torchvision's ToTensor()->Normalize()
(which normalizes in the [0,1] range). Output dtype is float32 to feed the
fp32 models (no AMP in this project).
"""
import numpy as np
from pathlib import Path

from ffcv.loader import Loader, OrderOption
from ffcv.transforms import (
    ToTensor,
    ToDevice,
    ToTorchImage,
    NormalizeImage,
    RandomHorizontalFlip,
    Squeeze,
)
from ffcv.fields.decoders import (
    IntDecoder,
    RandomResizedCropRGBImageDecoder,
    CenterCropRGBImageDecoder,
)

# ImageNet statistics scaled to the [0,255] range FFCV's NormalizeImage expects.
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406]) * 255
IMAGENET_STD = np.array([0.229, 0.224, 0.225]) * 255

CROP_SIZE = 224
# torchvision val is Resize(256) + CenterCrop(224); the equivalent FFCV crop ratio.
VAL_CROP_RATIO = 224 / 256


def get_ffcv_loader(
    beton_path: Path,
    batch_size: int,
    device,
    *,
    is_train: bool,
    num_workers: int = 8,
    distributed: bool = False,
    drop_last: bool = None,
):
    """Create an FFCV Loader for a .beton file with transforms matching the
    torchvision JPEG pipeline.

    Args:
        beton_path: Path to the .beton file.
        batch_size: Batch size.
        device: Torch device (e.g. torch.device('cuda:0')).
        is_train: True -> train augmentation + quasi-random order; False -> val.
        num_workers: Data-loading workers.
        distributed: FFCV distributed loading (single-GPU here -> False).
        drop_last: Drop last incomplete batch; defaults to is_train.
    """
    if drop_last is None:
        drop_last = is_train

    # Canonical FFCV ImageNet order: NormalizeImage runs on the GPU (after ToDevice +
    # ToTorchImage) -- this is the well-tested recipe and the layout FFCV's NormalizeImage
    # expects. It uses cupy, which needs a modern driver; the A10 training nodes run
    # driver 570.86 (>= the 525.60 that cupy-cuda12x needs), so this is fine. (FFCV's CPU
    # normalize path can't precede ToDevice -- "Can't be in JIT mode and on the GPU".)
    if is_train:
        image_pipeline = [
            RandomResizedCropRGBImageDecoder((CROP_SIZE, CROP_SIZE)),
            RandomHorizontalFlip(),
            ToTensor(),
            ToDevice(device, non_blocking=True),
            ToTorchImage(),
            NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float32),
        ]
        # QUASI_RANDOM: near-random order without loading the whole (larger-than-RAM)
        # beton into memory -- the disk/Lustre-friendly choice for full ImageNet.
        order = OrderOption.QUASI_RANDOM
    else:
        image_pipeline = [
            CenterCropRGBImageDecoder((CROP_SIZE, CROP_SIZE), ratio=VAL_CROP_RATIO),
            ToTensor(),
            ToDevice(device, non_blocking=True),
            ToTorchImage(),
            NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float32),
        ]
        order = OrderOption.SEQUENTIAL

    label_pipeline = [
        IntDecoder(),
        ToTensor(),
        Squeeze(),
        ToDevice(device, non_blocking=True),
    ]

    loader = Loader(
        fname=str(beton_path),
        batch_size=batch_size,
        num_workers=num_workers,
        order=order,
        os_cache=False,  # beton exceeds RAM; pair with QUASI_RANDOM
        drop_last=drop_last,
        pipelines={"image": image_pipeline, "label": label_pipeline},
        distributed=distributed,
    )
    return loader
