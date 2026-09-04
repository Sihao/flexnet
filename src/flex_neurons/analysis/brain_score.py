"""Brain-score neural predictivity scoring for flex_neurons models."""
from functools import partial
from typing import Callable, Dict, List, Optional

import torch.nn as nn

DEFAULT_BENCHMARKS = (
    "MajajHong2015public.IT-pls",
    "MajajHong2015public.V4-pls",
    "FreemanZiemba2013public.V1-pls",
    "FreemanZiemba2013public.V2-pls",
)


def _clear_hooks(model: nn.Module) -> Dict[nn.Module, tuple]:
    """Remove all forward/backward hooks from model and its submodules.

    Returns a snapshot mapping each module to a copy of its original
    ``(_forward_hooks, _forward_pre_hooks, _backward_hooks)`` OrderedDicts,
    so the caller's hooks can be restored later via ``_restore_hooks``.
    """
    snapshot: Dict[nn.Module, tuple] = {}
    for module in model.modules():
        forward_hooks = getattr(module, "_forward_hooks", None)
        forward_pre_hooks = getattr(module, "_forward_pre_hooks", None)
        backward_hooks = getattr(module, "_backward_hooks", None)
        snapshot[module] = (
            forward_hooks.copy() if forward_hooks is not None else None,
            forward_pre_hooks.copy() if forward_pre_hooks is not None else None,
            backward_hooks.copy() if backward_hooks is not None else None,
        )
        if forward_hooks is not None:
            forward_hooks.clear()  # pylint: disable=protected-access
        if forward_pre_hooks is not None:
            forward_pre_hooks.clear()  # pylint: disable=protected-access
        if backward_hooks is not None:
            backward_hooks.clear()  # pylint: disable=protected-access
    return snapshot


def _restore_hooks(snapshot: Dict[nn.Module, tuple]) -> None:
    """Restore hook containers previously saved by ``_clear_hooks``."""
    for module, (forward_hooks, forward_pre_hooks, backward_hooks) in snapshot.items():
        if forward_hooks is not None:
            module._forward_hooks.clear()  # pylint: disable=protected-access
            module._forward_hooks.update(forward_hooks)  # pylint: disable=protected-access
        if forward_pre_hooks is not None:
            module._forward_pre_hooks.clear()  # pylint: disable=protected-access
            module._forward_pre_hooks.update(forward_pre_hooks)  # pylint: disable=protected-access
        if backward_hooks is not None:
            module._backward_hooks.clear()  # pylint: disable=protected-access
            module._backward_hooks.update(backward_hooks)  # pylint: disable=protected-access


def _load_benchmark(bench_id: str):
    """Instantiate the concrete benchmark object for a given benchmark id."""
    # pylint: disable=import-outside-toplevel
    if "FreemanZiemba" in bench_id:
        import brainscore_vision.benchmarks.freemanziemba2013\
            .benchmarks.public_benchmarks as _fz
        FreemanZiembaV1PublicBenchmark = _fz.FreemanZiembaV1PublicBenchmark
        FreemanZiembaV2PublicBenchmark = _fz.FreemanZiembaV2PublicBenchmark
        if "V1" in bench_id:
            return FreemanZiembaV1PublicBenchmark()
        if "V2" in bench_id:
            return FreemanZiembaV2PublicBenchmark()
        raise ValueError(
            f"Unrecognised FreemanZiemba benchmark: {bench_id}"
        )
    if "MajajHong" in bench_id:
        from brainscore_vision.benchmarks.majajhong2015.benchmark import (
            MajajHongV4PublicBenchmark,
            MajajHongITPublicBenchmark,
        )
        if "V4" in bench_id:
            return MajajHongV4PublicBenchmark()
        if "IT" in bench_id:
            return MajajHongITPublicBenchmark()
        raise ValueError(
            f"Unrecognised MajajHong benchmark: {bench_id}"
        )
    raise ValueError(
        f"Unknown benchmark family for '{bench_id}'. "
        "Supported families: FreemanZiemba2013, MajajHong2015."
    )


def _extract_score(score_assembly) -> dict:
    """Pull center, error, and raw folds out of a brain-score DataAssembly."""
    agg = score_assembly.coords.get("aggregation")

    if agg is not None and "center" in agg.values:
        center = float(score_assembly.sel(aggregation="center").item())
    else:
        center = float(score_assembly.values.item())

    error = None
    if agg is not None and "error" in agg.values:
        try:
            error = float(score_assembly.sel(aggregation="error").item())
        except (TypeError, ValueError):
            error = None  # conversion failed; leave as None

    raw = None
    if "raw" in score_assembly.attrs:
        raw_obj = score_assembly.attrs["raw"]
        if hasattr(raw_obj, "values"):
            try:
                raw = raw_obj.values.tolist()
            except (TypeError, AttributeError):
                raw = raw_obj
        else:
            raw = raw_obj

    return {"center": center, "error": error, "raw": raw}


def score_model(
    model: nn.Module,
    model_id: str,
    layer_mapping: Dict[str, str],
    benchmarks: Optional[List[str]] = None,
    *,
    image_size: int = 224,
    preprocessing: Optional[Callable] = None,
) -> Dict[str, dict]:
    """Score model on brain-score benchmarks.

    Args:
        model: torch model.
        model_id: stable identifier for caching brain-score results.
        layer_mapping: dict mapping benchmark region (e.g., 'V1', 'V4', 'IT')
            to module name in model.
        benchmarks: list of benchmark identifiers; defaults to
            DEFAULT_BENCHMARKS.
        image_size: input resolution.
        preprocessing: optional callable mapping PIL.Image -> torch.Tensor.
            Defaults to ImageNet normalization via load_preprocess_images.

    Returns:
        dict[benchmark_id] -> {'center': float, 'error': float, 'raw': any}

    Raises:
        ImportError: if brainscore_vision is not installed.
        ValueError: if benchmarks list is empty or layer_mapping is empty.
    """
    if not layer_mapping:
        raise ValueError(
            "layer_mapping must be a non-empty dict mapping region to "
            "module name."
        )

    if benchmarks is None:
        benchmarks = list(DEFAULT_BENCHMARKS)

    if not benchmarks:
        raise ValueError("benchmarks list must be non-empty.")

    try:
        # pylint: disable=import-outside-toplevel
        from brainscore_vision.model_helpers.activations.pytorch import (
            PytorchWrapper,
            load_preprocess_images,
        )
        from brainscore_vision.model_helpers.brain_transformation import (
            ModelCommitment,
        )
    except ImportError as import_err:
        raise ImportError(
            "brainscore_vision is required for brain-score scoring. "
            "Install it from the vendored repo at brainscore_vision_repo/ "
            f"or via pip. Original error: {import_err}"
        ) from import_err

    hooks_snapshot = _clear_hooks(model)
    try:
        model.eval()

        if preprocessing is None:
            preprocessing = partial(load_preprocess_images, image_size=image_size)

        activations_model = PytorchWrapper(
            identifier=model_id,
            model=model,
            preprocessing=preprocessing,
        )

        available = dict(model.named_modules())
        missing = [
            name for name in layer_mapping.values() if name not in available
        ]
        if missing:
            examples = list(available.keys())[:10]
            raise ValueError(
                f"layer_mapping references module name(s) not found on model: "
                f"{missing}. Example valid module names: {examples}"
            )

        brain_model = ModelCommitment(
            identifier=model_id,
            activations_model=activations_model,
            layers=list(layer_mapping.values()),
            region_layer_map=layer_mapping,
        )

        results: Dict[str, dict] = {}
        for bench_id in benchmarks:
            benchmark = _load_benchmark(bench_id)
            score_assembly = benchmark(brain_model)
            results[bench_id] = _extract_score(score_assembly)

        return results
    finally:
        _restore_hooks(hooks_snapshot)
