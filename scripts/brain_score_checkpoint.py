#!/usr/bin/env python3
"""
Brain-Score a single (Flex)ResNet checkpoint.

Built for the iso-accuracy trajectory comparison. Unlike
``src/analysis/brain_score/run_brain_score.py`` (hardcoded for VGG / Exp2 /
Exp4), this builds the model from the experiment's ``configurations.json`` via
the project model registry, so it works for both FlexResNet (use_flex=true) and
vanilla ResNet (use_flex=false). It reuses the tested ``process_layer`` scoring
loop from ``run_brain_score``.

Runs in the ``brain_score`` conda env (has brainscore_vision + benchmark data
under ``data/brain_score_data``). CPU is fine — the PLS cross-validation is
CPU-bound regardless of GPU.

Idempotent: skips (layer, benchmark) pairs already present in the output JSON.

Example:
  conda run -n brain_score python scripts/brain_score_checkpoint.py \
    --ckpt .../checkpoint_40.pth --config .../configurations.json \
    --exp-name flex_e40 --layers layer4.2.conv3 --benchmarks MajajHong2015.public.IT-pls \
    --output results/brain_score_trajectory/flex_e40.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import torch

# Default Brain-Score data location BEFORE importing run_brain_score (its
# module-level code reads FLEX_DATA_ROOT to set BRAINIO_HOME/BRAINSCORE_HOME).
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))  # make `src` importable when run as a script
os.environ.setdefault(
    "FLEX_DATA_ROOT", str(PROJECT_ROOT / "data" / "brain_score_data")
)

from src.modules import models  # noqa: E402
from src.analysis.brain_score.run_brain_score import (  # noqa: E402
    process_layer,
    load_results,
)

DEFAULT_BENCHMARKS = [
    "FreemanZiemba2013.V1.public-pls",
    "FreemanZiemba2013.V2.public-pls",
    "MajajHong2015.public.V4-pls",
    "MajajHong2015.public.IT-pls",
]


def load_model(config: dict, ckpt_path: Path, device: str):
    network = config.get("network", "FlexResNet")
    model = getattr(models, network)(config=config)
    model.to(device)
    checkpoint = torch.load(ckpt_path, map_location=device)
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[warn] {len(missing)} missing keys on load (e.g. {missing[:3]})")
    if unexpected:
        print(f"[warn] {len(unexpected)} unexpected keys (e.g. {unexpected[:3]})")
    model.eval()
    return model


def resolve_layers(model, requested: list[str]) -> list[str]:
    conv_like = [
        name
        for name, mod in model.named_modules()
        if isinstance(mod, torch.nn.Conv2d)
    ]
    if requested == ["auto"]:
        # representative deep layer (best for V4/IT) — last conv-like module
        return [conv_like[-1]]
    resolved = []
    for r in requested:
        if r in conv_like:
            resolved.append(r)
            continue
        # dotted-boundary match (e.g. "layer4.2.conv3" -> "...conv3.layer.flex_conv"),
        # not a bare substring match: "features.3" must not match "features.30".
        pattern = re.compile(rf"(^|\.){re.escape(r)}(\.|$)")
        matches = [n for n in conv_like if pattern.search(n)]
        if not matches:
            raise ValueError(
                f"layer '{r}' not found; available examples: {conv_like[-5:]}"
            )
        if len(matches) > 1:
            raise ValueError(
                f"layer '{r}' is ambiguous; matching candidates: {matches}"
            )
        resolved.append(matches[0])
    return resolved


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", required=True, type=Path)
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--exp-name", required=True, help="Label, e.g. flex_e40")
    p.add_argument("--layers", nargs="+", default=["auto"],
                   help="Layer names/substrings, or 'auto' for deepest conv.")
    p.add_argument("--benchmarks", nargs="+", default=DEFAULT_BENCHMARKS)
    p.add_argument("--output", required=True, type=Path)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not args.ckpt.is_file():
        print(f"[error] checkpoint not found: {args.ckpt}", file=sys.stderr)
        return 2
    if not args.config.is_file():
        print(f"[error] config not found: {args.config}", file=sys.stderr)
        return 2

    config = json.loads(args.config.read_text())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[info] device={device} network={config.get('network')} "
          f"use_flex={config.get('use_flex')}")

    model = load_model(config, args.ckpt, device)
    layers = resolve_layers(model, args.layers)
    print(f"[info] scoring layers: {layers}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    results_data = load_results(args.output)

    # idempotency: drop benchmarks already scored for each layer
    for layer in layers:
        done = set(results_data.get(args.exp_name, {}).get(layer, {}).keys())
        pending = [b for b in args.benchmarks if b not in done]
        if not pending:
            print(f"[skip] {args.exp_name}/{layer}: all benchmarks present")
            continue
        print(f"[run] {args.exp_name}/{layer}: {pending}")
        process_layer(
            model=model,
            layer=layer,
            exp_name=args.exp_name,
            config=config,
            results_data=results_data,
            output_file=args.output,
            benchmarks=pending,
            early_layers=set(),  # explicit layers: do not skip any benchmark
        )

    print(f"[ok] brain-score results -> {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
