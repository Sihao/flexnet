"""Shared utilities for flex_neurons."""

from .device import (
    select_device,
    is_ddp,
    get_rank,
    is_main_process,
    setup_ddp,
    cleanup_ddp,
    check_cuda_memory_usage,
)
from .general import apply_kaiming_initialization
from .normalization import (
    Normalize,
    IMAGENET_MEAN,
    IMAGENET_STD,
    denormalize_batch,
)
from .overlap import get_overlapping_classes
from .server import is_on_server
from .simple_logger import SimpleLogger
# Alias so callers can use get_device() as the canonical name
get_device = select_device

# spectral_utils is available as src.flex_neurons.utils.spectral_utils but is
# NOT eagerly imported here because it transitively pulls in src.analysis and
# src.modules which require the full project dependency stack (natsort, etc.).

__all__ = [
    "select_device",
    "get_device",
    "is_ddp",
    "get_rank",
    "is_main_process",
    "setup_ddp",
    "cleanup_ddp",
    "check_cuda_memory_usage",
    "apply_kaiming_initialization",
    "Normalize",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "denormalize_batch",
    "get_overlapping_classes",
    "is_on_server",
    "SimpleLogger",
]
