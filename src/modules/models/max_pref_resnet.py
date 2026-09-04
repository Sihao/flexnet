"""
MaxPrefResNet: a static, parameter-free-where-max ResNet-50 whose per-site conv-vs-max
choice is fixed ahead of time by a "max-pref manifest" instead of being learned/gated at
runtime the way Flex2D does (src/modules/layers/flex.py).

Each of the 53 canonical 1x1/3x3/7x7 conv sites in a standard ResNet-50 (stem conv1, and
conv1/conv2/conv3 + downsample.0 for every Bottleneck) is resolved once, in __init__, to
either an nn.Conv2d or a parameter-free MaxOp. The resulting network has a fully static
forward graph -- no branching, no gating, no learned threshold.
"""

import json
from pathlib import Path

import torch
import torch.nn as nn

from src.modules.layers._utils import channel_expand_view
from src.utils.general import apply_kaiming_initialization

DEFAULT_MAX_PREF_THRESHOLD = 0.5


class MaxOp(nn.Module):
    """
    Parameter-free stand-in for a conv site. Mirrors the max branch of Flex2D
    (src/modules/layers/flex.py: self.flex_pool -> channel_expand_view) exactly:
    max-pool with the site's own geometry, then expand channels to match what the
    conv it replaces would have produced.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        pooled = torch.nn.functional.max_pool2d(
            x, self.kernel_size, self.stride, self.padding
        )
        return channel_expand_view(pooled, self.out_channels)


class StableBatchNorm2d(nn.BatchNorm2d):
    """BatchNorm2d hardened against the running-variance underflow that made trained
    MaxPrefResNet checkpoints explode in eval mode (see lessons.md / project_maxpref_resnet).
    Two complementary guards:

      * a larger default ``eps`` (1e-4 vs torch's 1e-5). This is the primary fix: it bounds
        eval-mode amplification even when a channel's running_var has collapsed, turning the
        original ~1e6 loss explosion into a much smaller (chance-level, pre-recalibration)
        output. 1e-4 keeps recalibrated accuracy close to plain-BN while still guarding
        against the collapse (raise to 1e-3 for a stronger raw-checkpoint guard, at ~1-2 pts).
      * a small ``running_var`` floor applied after each *ordinary* training update, so the
        EMA can't decay to the float32 denormal floor for near-constant channels.

    Crucially the floor is DISABLED during BN recalibration (recalibrate_batchnorm sets
    var_floor=0 for the pass): recalibration must reproduce the true, un-clamped variances
    -- clamping them upward corrupts the recomputed stats and sends eval back to ~chance
    (regression caught on job 5691441). The floor default is kept well below any
    signal-bearing variance so it only ever catches genuine underflow.
    """

    def __init__(self, num_features, eps=1e-4, momentum=0.1, var_floor=1e-5, **kwargs):
        super().__init__(num_features, eps=eps, momentum=momentum, **kwargs)
        if var_floor < 0:
            raise ValueError(f"var_floor must be >= 0, got {var_floor}")
        self.var_floor = float(var_floor)

    def forward(self, x):
        out = super().forward(x)
        if (self.training and self.track_running_stats
                and self.running_var is not None and self.var_floor > 0):
            # Apply the variance floor WITHOUT tripping autograd's in-place guard.
            # super().forward() runs native batch_norm, which (in training) updates
            # running_var in place -> version 1 and saves that exact tensor for backward.
            # An in-place clamp_ here bumps the SAME object to version 2, so backward aborts
            # with "a variable needed for gradient computation has been modified by an inplace
            # operation: [FloatTensor [C]] is at version 2; expected version 1" (job 5693201,
            # first backward of the 90ep run; C=2048 == layer4 bn3). torch.no_grad() suppresses
            # graph-building but NOT the version counter, so it does not help. Rebinding the
            # buffer to a freshly clamped tensor leaves the saved object untouched (still
            # version 1) while the floored values carry into the next iteration's EMA.
            with torch.no_grad():
                self.running_var = self.running_var.clamp(min=self.var_floor)
        return out


def make_bn(num_features, *, stable=True, eps=1e-4, var_floor=1e-5):
    """BN factory for MaxPrefResNet: the hardened StableBatchNorm2d by default;
    ``stable=False`` (config['stable_bn']=false) falls back to a plain nn.BatchNorm2d."""
    if stable:
        return StableBatchNorm2d(num_features, eps=eps, var_floor=var_floor)
    return nn.BatchNorm2d(num_features)


@torch.no_grad()
def recalibrate_batchnorm(model, data_loader, num_batches=None, device=None, reset=True):
    """Reset and recompute BatchNorm running statistics from real data.

    A trained MaxPrefResNet must be BN-recalibrated before evaluation: its checkpoints
    ship with collapsed running_var (an eval-mode explosion, ~chance top-1) even though
    the weights are sound. This runs train-mode forward passes (no grad, cumulative
    averaging) to rebuild usable running stats, restoring true accuracy without any
    fine-tuning -- validated at top-1 0.13% -> 16.4% on ImageNet val (job 5691247).

    Restores each BN's original momentum and the model's prior train/eval mode.
    Returns {"bn_layers": n, "images": m}.
    """
    if device is None:
        device = next(model.parameters()).device
    bns = [m for m in model.modules() if isinstance(m, nn.BatchNorm2d)]
    if not bns:
        raise ValueError("recalibrate_batchnorm: model has no BatchNorm2d layers")
    saved_momentum = [m.momentum for m in bns]
    # Disable the StableBatchNorm2d variance floor for the recalibration pass: clamping the
    # freshly recomputed running_var upward corrupts the exact stats we are rebuilding and
    # sends eval back to ~chance (regression on job 5691441). Restored in finally.
    saved_floor = [getattr(m, "var_floor", None) for m in bns]
    for m in bns:
        if reset:
            m.reset_running_stats()
        m.momentum = None  # cumulative moving average over the recalibration batches
        if isinstance(m, StableBatchNorm2d):
            m.var_floor = 0.0
    was_training = model.training
    model.train()
    seen = 0
    try:
        for i, batch in enumerate(data_loader):
            if num_batches is not None and i >= num_batches:
                break
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            model(x.to(device, non_blocking=True))
            seen += int(x.size(0))
    finally:
        for m, mom, floor in zip(bns, saved_momentum, saved_floor):
            m.momentum = mom
            if floor is not None:
                m.var_floor = floor
        model.train(was_training)
    if seen == 0:
        raise ValueError("recalibrate_batchnorm: data_loader yielded no batches")
    return {"bn_layers": len(bns), "images": seen}


def site_op(name, manifest, in_channels, out_channels, kernel_size, stride=1, padding=0):
    """
    Resolve one canonical site to a concrete op using the (already-loaded) manifest.
    """
    op = manifest.get(name)
    if op == "conv":
        return nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False
        )
    if op == "max":
        return MaxOp(in_channels, out_channels, kernel_size, stride, padding)
    raise ValueError(
        f"Unknown max-pref op {op!r} for site {name!r} (expected 'conv' or 'max')"
    )


def _canonicalize_site_name(name: str) -> str:
    """
    The flex conv-ratio JSON (scripts/analyze_flex_conv_ratio_hpc.py) records names from
    FlexResNet.named_modules(), where the Flex2D module sits one level below the
    FlexConvWrapper that owns it (e.g. 'layer1.0.conv1.layer'). MaxPrefResNet's canonical
    site names drop that '.layer' suffix (e.g. 'layer1.0.conv1').
    """
    suffix = ".layer"
    return name[: -len(suffix)] if name.endswith(suffix) else name


def _manifest_from_flex_json(flex_json_path, threshold) -> dict:
    data = json.loads(Path(flex_json_path).read_text())
    layers = data["layers"] if isinstance(data, dict) and "layers" in data else data
    return {
        _canonicalize_site_name(entry["name"]): ("conv" if entry["conv"] >= threshold else "max")
        for entry in layers
    }


def _load_manifest(config: dict) -> dict:
    """
    Resolve the per-site conv/max manifest for MaxPrefResNet.

    - If config["max_pref_manifest"] is given, it is a path to a JSON file in the
      #277 shared-contract format written by scripts/build_max_pref_manifest.py
      (subissue #278):
        {"threshold": <float>, "source_json": <str>, "num_max": <int>,
         "num_conv": <int>,
         "sites": {canonical_name: {"op": "conv"|"max", "conv_ratio": <float>,
                                     "role": <str>, "stride": <int>, "kernel": <int>,
                                     "in_ch": <int>, "out_ch": <int>, "padding": <int>},
                    ...}}   # exactly 53 entries, keys already canonical.
      Only the "op" field is needed here; site geometry is re-derived by this
      module's own construction, which must agree with the manifest's geometry.
    - Otherwise it is rebuilt in memory by thresholding config["flex_conv_ratio_json"]
      (the per-layer conv-ratio JSON produced by scripts/analyze_flex_conv_ratio_hpc.py)
      at config["max_pref_threshold"] (default 0.5): conv_ratio >= threshold -> "conv".
    """
    manifest_path = config.get("max_pref_manifest")
    if manifest_path:
        data = json.loads(Path(manifest_path).read_text())
        if not isinstance(data, dict) or "sites" not in data:
            raise ValueError(
                f"max-pref manifest {manifest_path!r} is not in the expected "
                "contract format: a top-level JSON object with a 'sites' dict "
                "mapping canonical site name -> {'op': 'conv'|'max', ...} "
                "(see scripts/build_max_pref_manifest.py)."
            )
        return {name: site["op"] for name, site in data["sites"].items()}

    flex_json_path = config.get("flex_conv_ratio_json")
    if not flex_json_path:
        raise ValueError(
            "MaxPrefResNet requires either config['max_pref_manifest'] (path to a "
            "resolved per-site conv/max manifest) or config['flex_conv_ratio_json'] "
            "(path to a raw flex conv-ratio JSON to threshold via "
            "config['max_pref_threshold'])."
        )
    threshold = config.get("max_pref_threshold", DEFAULT_MAX_PREF_THRESHOLD)
    return _manifest_from_flex_json(flex_json_path, threshold)


def _expected_site_names(layers_count):
    """The 53 canonical sites for a standard ResNet-50: stem conv1, plus conv1/conv2/conv3
    (and downsample.0 on the first block) of every Bottleneck in every stage."""
    names = ["conv1"]
    for layer_idx, num_blocks in enumerate(layers_count, start=1):
        for block_idx in range(num_blocks):
            prefix = f"layer{layer_idx}.{block_idx}"
            names.append(f"{prefix}.conv1")
            names.append(f"{prefix}.conv2")
            names.append(f"{prefix}.conv3")
            if block_idx == 0:
                names.append(f"{prefix}.downsample.0")
    return names


class Bottleneck(nn.Module):
    """Mirrors src/modules/models/flex_resnet.py Bottleneck exactly, with each conv
    resolved statically via site_op instead of being a learned Flex2D branch."""

    expansion = 4

    def __init__(self, inplanes, planes, manifest, name_prefix, stride=1, downsample=None,
                 bn_kwargs=None):
        super(Bottleneck, self).__init__()
        bn_kwargs = bn_kwargs or {}

        self.conv1 = site_op(
            f"{name_prefix}.conv1", manifest, inplanes, planes, kernel_size=1, stride=1, padding=0
        )
        self.bn1 = make_bn(planes, **bn_kwargs)

        self.conv2 = site_op(
            f"{name_prefix}.conv2", manifest, planes, planes, kernel_size=3, stride=stride, padding=1
        )
        self.bn2 = make_bn(planes, **bn_kwargs)

        self.conv3 = site_op(
            f"{name_prefix}.conv3",
            manifest,
            planes,
            planes * self.expansion,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.bn3 = make_bn(planes * self.expansion, **bn_kwargs)

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


class MaxPrefResNet(nn.Module):
    def __init__(self, config: dict):
        super(MaxPrefResNet, self).__init__()
        self.config = config
        self.__name__ = "MaxPrefResNet"

        # Determine dimensions -- mirrors FlexResNet.__init__ exactly.
        dataset = config.get("dataset", "")
        if "cifar" in dataset:
            self.in_dimensions, self.num_classes = (3, 32, 32), 10
            self.base_channel = 16
        elif dataset == "imagenet":
            self.in_dimensions, self.num_classes = (3, 224, 224), 1000
            self.base_channel = 64
        elif "imagenet" in dataset:
            self.in_dimensions, self.num_classes = (3, 224, 224), 100
            self.base_channel = 64
        else:
            raise ValueError(f"Dataset not supported: {dataset}")

        depth = config.get("resnet_depth", 50)
        if depth != 50:
            raise ValueError(
                f"MaxPrefResNet only supports resnet_depth=50 (Bottleneck/[3,4,6,3]); got {depth}"
            )
        layers_count = [3, 4, 6, 3]

        manifest = _load_manifest(config)
        expected_names = _expected_site_names(layers_count)
        missing = [name for name in expected_names if name not in manifest]
        if missing:
            raise ValueError(
                f"max-pref manifest is missing {len(missing)}/{len(expected_names)} "
                f"required canonical site(s): {missing}"
            )
        self.manifest = manifest

        # BatchNorm construction: hardened StableBatchNorm2d by default (prevents the
        # running_var underflow that made trained checkpoints unusable in eval). Opt out
        # or tune via config: stable_bn / bn_eps / bn_var_floor.
        self.bn_kwargs = {
            "stable": config.get("stable_bn", True),
            "eps": config.get("bn_eps", 1e-4),
            "var_floor": config.get("bn_var_floor", 1e-5),
        }

        self.inplanes = self.base_channel

        # Stem -- always the ImageNet-style ResNet-50 stem (FlexResNet takes this branch
        # unconditionally once depth == 50).
        self.conv1 = site_op(
            "conv1",
            manifest,
            self.in_dimensions[0],
            self.inplanes,
            kernel_size=7,
            stride=2,
            padding=3,
        )
        self.bn1 = make_bn(self.inplanes, **self.bn_kwargs)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Stack layers
        self.layer1 = self._make_layer(manifest, 1, self.base_channel, layers_count[0], stride=1)
        self.layer2 = self._make_layer(manifest, 2, self.base_channel * 2, layers_count[1], stride=2)
        self.layer3 = self._make_layer(manifest, 3, self.base_channel * 4, layers_count[2], stride=2)
        self.layer4 = self._make_layer(manifest, 4, self.base_channel * 8, layers_count[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        final_channels = self.base_channel * 8 * Bottleneck.expansion
        self.fc = nn.Linear(final_channels, self.num_classes)

        apply_kaiming_initialization(self)

    def _make_layer(self, manifest, layer_idx, planes, blocks, stride):
        name_prefix0 = f"layer{layer_idx}.0"

        downsample = None
        if stride != 1 or self.inplanes != planes * Bottleneck.expansion:
            downsample = nn.Sequential(
                site_op(
                    f"{name_prefix0}.downsample.0",
                    manifest,
                    self.inplanes,
                    planes * Bottleneck.expansion,
                    kernel_size=1,
                    stride=stride,
                    padding=0,
                ),
                make_bn(planes * Bottleneck.expansion, **self.bn_kwargs),
            )

        layers = [
            Bottleneck(
                self.inplanes, planes, manifest, name_prefix0, stride=stride,
                downsample=downsample, bn_kwargs=self.bn_kwargs,
            )
        ]
        self.inplanes = planes * Bottleneck.expansion
        for block_idx in range(1, blocks):
            layers.append(
                Bottleneck(self.inplanes, planes, manifest, f"layer{layer_idx}.{block_idx}",
                           bn_kwargs=self.bn_kwargs)
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

    def recalibrate_bn(self, data_loader, num_batches=None, device=None):
        """Rebuild BatchNorm running stats from real data before evaluation.

        Required for correct eval: a trained MaxPrefResNet's saved BN stats are collapsed
        and score ~chance until recalibrated. Thin wrapper over recalibrate_batchnorm.
        """
        return recalibrate_batchnorm(self, data_loader, num_batches=num_batches, device=device)
