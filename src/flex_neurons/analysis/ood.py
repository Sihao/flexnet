import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from PIL import Image, ImageOps
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

DEFAULT_CORRUPTIONS = (
    "gaussian_noise", "shot_noise", "impulse_noise",
    "defocus_blur", "glass_blur", "motion_blur", "zoom_blur",
    "snow", "frost", "fog", "brightness", "contrast",
    "elastic_transform", "pixelate", "jpeg_compression",
)

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]

_IMAGENET_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
])


def _exif_safe_loader(path):
    """Image loader that honors EXIF orientation tags before converting to RGB.

    PIL's default ImageFolder loader (Image.open) does not apply EXIF
    orientation metadata, so images with such tags (common in ImageNet-C /
    ImageNet-A) can be silently loaded rotated or mirrored.
    """
    with open(path, "rb") as f:
        img = Image.open(f)
        img = ImageOps.exif_transpose(img)
        return img.convert("RGB")


def _load_imagenet_a_indices(data_root: Path) -> List[int]:
    """Load the 200-class ImageNet-A valid logit indices from imagenet_a_indices.json.

    The file is expected at data_root.parent/imagenet_a_indices.json.
    If absent, raise an informative error.
    """
    raw = Path(data_root)
    resolved = raw.resolve()
    candidates = []
    for base in (raw.parent, resolved.parent, raw, resolved):
        candidate = base / "imagenet_a_indices.json"
        if candidate not in candidates:
            candidates.append(candidate)
    for path in candidates:
        if path.exists():
            with open(path) as f:
                indices = json.load(f)
            return indices

    raise FileNotFoundError(
        "imagenet_a_indices.json not found. This file maps the 200 ImageNet-A "
        "classes to their corresponding positions in a full 1000-class logit vector. "
        "Obtain it from https://github.com/hendrycks/natural-adv-examples or a "
        "compatible source, and place it at: "
        f"{candidates[0]}"
    )


def evaluate_imagenet_c(
    model: nn.Module,
    data_root: Path,
    severities: List[int] = [1, 2, 3, 4, 5],
    corruptions: Optional[List[str]] = None,
    batch_size: int = 64,
    num_workers: int = 4,
    device: str = "cuda",
) -> Dict[str, Dict[int, float]]:
    """Evaluate top-1 accuracy per corruption per severity on ImageNet-C.

    Folder structure: <data_root>/<corruption>/<severity>/<class>/

    Args:
        model: PyTorch model. Will be moved to device and set to eval mode.
        data_root: Root directory of ImageNet-C.
        severities: Severity levels to evaluate (1-5).
        corruptions: Corruption types to evaluate. Defaults to DEFAULT_CORRUPTIONS.
        batch_size: DataLoader batch size.
        num_workers: DataLoader worker count.
        device: Torch device string.

    Returns:
        Nested dict: results[corruption][severity] = top1_accuracy (float in [0, 1]).

    Raises:
        FileNotFoundError: if data_root does not exist.
    """
    data_root = Path(data_root)
    if not data_root.exists():
        raise FileNotFoundError(f"ImageNet-C data_root not found: {data_root}")

    if corruptions is None:
        corruptions = list(DEFAULT_CORRUPTIONS)

    dev = torch.device(device)
    model = model.to(dev)
    model.eval()

    results: Dict[str, Dict[int, float]] = {}

    for corruption in corruptions:
        results[corruption] = {}
        for severity in severities:
            split_path = data_root / corruption / str(severity)
            if not split_path.exists():
                raise FileNotFoundError(
                    f"Expected ImageNet-C split not found: {split_path}"
                )

            try:
                dataset = datasets.ImageFolder(
                    str(split_path), transform=_IMAGENET_TRANSFORM, loader=_exif_safe_loader
                )
            except (RuntimeError, FileNotFoundError) as e:
                warnings.warn(
                    f"Skipping empty ImageNet-C split {split_path}: {e}",
                    UserWarning,
                    stacklevel=2,
                )
                results[corruption][severity] = float("nan")
                continue

            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
            )

            correct = 0
            total = 0
            with torch.no_grad():
                for images, labels in loader:
                    images = images.to(dev)
                    labels = labels.to(dev)
                    outputs = model(images)
                    _, predicted = outputs.max(1)
                    correct += predicted.eq(labels).sum().item()
                    total += labels.size(0)

            results[corruption][severity] = correct / total if total > 0 else 0.0

    return results


def evaluate_imagenet_a(
    model: nn.Module,
    data_root: Path,
    batch_size: int = 64,
    num_workers: int = 4,
    device: str = "cuda",
) -> float:
    """Evaluate top-1 accuracy on ImageNet-A.

    ImageNet-A uses a 200-class subset of ImageNet. Model logits are mapped onto
    the 200 valid classes via imagenet_a_indices.json (expected at data_root.parent).

    Args:
        model: PyTorch model producing 1000-class logits. Moved to device, set to eval.
        data_root: Root directory of ImageNet-A (contains class subdirectories).
        batch_size: DataLoader batch size.
        num_workers: DataLoader worker count.
        device: Torch device string.

    Returns:
        Top-1 accuracy as a float in [0, 1].

    Raises:
        FileNotFoundError: if data_root or imagenet_a_indices.json is missing.
        ValueError: if imagenet_a_indices.json holds WordNet ID strings instead
            of integer 1000-class indices (unsupported — see below), or if the
            model's logit vector is smaller than required.

    Note:
        imagenet_a_indices.json must contain a list of 200 integer 1000-class
        logit indices. WordNet ID strings are not supported: resolving a
        WordNet ID to its 1000-class logit column requires an ImageNet-1k
        WordNet-ID ordering table that this module does not have.
    """
    data_root = Path(data_root)
    if not data_root.exists():
        raise FileNotFoundError(f"ImageNet-A data_root not found: {data_root}")

    valid_indices = _load_imagenet_a_indices(data_root)

    if valid_indices and isinstance(valid_indices[0], str):
        raise ValueError(
            "imagenet_a_indices.json contains WordNet ID strings, which are not "
            "supported: mapping a WordNet ID to its 1000-class logit column "
            "requires an ImageNet-1k WordNet-ID ordering table that this module "
            "does not have. Regenerate imagenet_a_indices.json as a list of "
            "integer 1000-class indices instead."
        )

    valid_indices_tensor = torch.tensor(valid_indices, dtype=torch.long)

    dev = torch.device(device)
    model = model.to(dev)
    model.eval()
    valid_indices_tensor = valid_indices_tensor.to(dev)

    dataset = datasets.ImageFolder(
        str(data_root), transform=_IMAGENET_TRANSFORM, loader=_exif_safe_loader
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    # `valid_indices` is a list of 200 integer 1000-class logit indices, in
    # whatever order imagenet_a_indices.json stores them. `dataset.class_to_idx`
    # maps folder name (a WordNet ID string) -> 0..199 in alphabetical folder
    # order. We have no 1000-class-index -> WordNet-ID table here, so we cannot
    # verify that the JSON order matches the dataset's alphabetical folder
    # order; warn and assume it does.
    warnings.warn(
        "imagenet_a_indices.json contains integers (1000-class indices). "
        "Cannot verify that the ordering matches dataset folder ordering. "
        "Assuming alphabetical folder order matches JSON order.",
        UserWarning,
        stacklevel=2,
    )

    # Minimum number of logit columns the model must produce: enough to cover
    # every index referenced by imagenet_a_indices.json. Checked once, against
    # the first batch of outputs, before any indexing happens.
    min_required_logits = int(valid_indices_tensor.max().item()) + 1
    shape_verified = False

    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(dev)
            labels = labels.to(dev)
            outputs = model(images)
            if not shape_verified:
                if outputs.shape[1] < min_required_logits:
                    raise ValueError(
                        "evaluate_imagenet_a requires a model that produces "
                        "1000-class ImageNet logits (outputs.shape[1] >= "
                        f"{min_required_logits}, the highest index referenced by "
                        "imagenet_a_indices.json), but the model produced "
                        f"outputs with shape {tuple(outputs.shape)}."
                    )
                shape_verified = True
            # Restrict logits to the 200 valid ImageNet-A classes.
            # outputs: (N, 1000) -> restricted: (N, 200)
            # Column i of restricted corresponds to valid_indices[i] in 1000-class space.
            restricted = outputs[:, valid_indices_tensor]
            predicted = restricted.argmax(1)  # index 0..199 in valid_indices order
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

    return correct / total if total > 0 else 0.0
