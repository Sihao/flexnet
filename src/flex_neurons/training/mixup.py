"""
Mixup / CutMix batch collator.

Combines images and label-smoothed soft labels for two standard batch-level
augmentations. Meant to be passed as `collate_fn` to a DataLoader so mixing
happens once per batch on already-loaded tensors -- no coupling to any
model or training loop.

Reference for the cutmix bbox sampling: timm.data.mixup.
"""

import numpy as np
import torch


def rand_bbox(W: int, H: int, lam: float):
    """Sample a random bounding box whose area is a `1 - lam` fraction of W*H.

    Returns (bbx1, bby1, bbx2, bby2), clipped to the image bounds.
    """
    cut_w = int(W * np.sqrt(1.0 - lam))
    cut_h = int(H * np.sqrt(1.0 - lam))

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


class MixupCutmixCollator:
    """Collate function that applies Mixup or CutMix to a batch, with label smoothing.

    Usage:
        collator = MixupCutmixCollator(num_classes=1000)
        loader = DataLoader(dataset, batch_size=..., collate_fn=collator)

    __call__ takes a list of (image_tensor, int_label) tuples -- the default
    per-sample format yielded by a Dataset -- and returns (images, soft_labels),
    where images is a stacked (B, C, H, W) tensor and soft_labels is (B, num_classes).
    """

    def __init__(
        self,
        num_classes: int,
        mixup_alpha: float = 0.1,
        cutmix_alpha: float = 1.0,
        mixup_prob: float = 0.5,
        label_smoothing: float = 0.1,
    ):
        self.num_classes = num_classes
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.mixup_prob = mixup_prob
        self.label_smoothing = label_smoothing

    def _smooth_one_hot(self, labels: torch.Tensor) -> torch.Tensor:
        off_value = self.label_smoothing / self.num_classes
        on_value = 1.0 - self.label_smoothing + off_value
        soft = torch.full((labels.size(0), self.num_classes), off_value, dtype=torch.float32)
        soft.scatter_(1, labels.unsqueeze(1), on_value)
        return soft

    def __call__(self, batch):
        images = torch.stack([sample[0] for sample in batch], dim=0)
        labels = torch.tensor([sample[1] for sample in batch], dtype=torch.long)
        soft_labels = self._smooth_one_hot(labels)

        use_mixup = self.mixup_alpha > 0.0
        use_cutmix = self.cutmix_alpha > 0.0

        if not use_mixup and not use_cutmix:
            # Beta(0, 0) is degenerate -- special-case as a no-op (lam == 1).
            return images, soft_labels

        batch_size = images.size(0)
        perm = torch.randperm(batch_size)

        # Only flip a coin when both augmentations are actually enabled;
        # otherwise always take the one enabled branch. This avoids ever
        # drawing from Beta(0, 0), which raises ValueError.
        if use_mixup and use_cutmix:
            do_mixup = np.random.rand() < self.mixup_prob
        else:
            do_mixup = use_mixup

        if do_mixup:
            lam = float(np.random.beta(self.mixup_alpha, self.mixup_alpha))
            images = lam * images + (1.0 - lam) * images[perm]
            soft_labels = lam * soft_labels + (1.0 - lam) * soft_labels[perm]
        else:
            lam = float(np.random.beta(self.cutmix_alpha, self.cutmix_alpha))
            _, _, H, W = images.shape
            bbx1, bby1, bbx2, bby2 = rand_bbox(W, H, lam)
            images[:, :, bby1:bby2, bbx1:bbx2] = images[perm][:, :, bby1:bby2, bbx1:bbx2]
            # Recompute lam from the actual patch area to account for bbox clipping.
            lam = 1.0 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
            soft_labels = lam * soft_labels + (1.0 - lam) * soft_labels[perm]

        return images, soft_labels
