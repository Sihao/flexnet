import os
import torch
import torch.distributed as dist


def select_device(priority=["cuda", "cpu"]):  # mps in the middle
    """
    Selects the device based on the given priority list.
    In DDP mode (LOCAL_RANK env var set), selects the specific GPU for this process.
    Otherwise falls back to priority-based selection.

    Parameters:
        - priority (list): List of strings representing device priorities.

    Returns:
        - torch.device: Device selected based on priority.
    """
    local_rank = os.environ.get("LOCAL_RANK")
    if local_rank is not None and torch.cuda.is_available():
        return torch.device(f"cuda:{local_rank}")

    if "cuda" in priority and torch.cuda.is_available():
        return torch.device("cuda")
    if (
        "mps" in priority
        and torch.backends.mps.is_available()
        and torch.backends.mps.is_built()
    ):
        return torch.device("mps")
    if "cpu" in priority:
        return torch.device("cpu")

    raise ValueError("No valid device found in priority list.")


def is_ddp() -> bool:
    """Return True if running in a DDP context (torchrun / torch.distributed.launch)."""
    return os.environ.get("LOCAL_RANK") is not None


def get_rank() -> int:
    """Global rank. Returns 0 for non-DDP runs."""
    if dist.is_initialized():
        return dist.get_rank()
    return 0


def is_main_process() -> bool:
    """True on rank 0 or non-DDP runs."""
    return get_rank() == 0


def setup_ddp():
    """Initialize the DDP process group if LOCAL_RANK is set. No-op otherwise."""
    if not is_ddp():
        return
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")


def cleanup_ddp():
    """Destroy the DDP process group if initialized."""
    if dist.is_initialized():
        dist.destroy_process_group()


def check_cuda_memory_usage():
    """cuda only"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        current_memory_allocated = torch.cuda.memory_allocated(device=device) / (
            1024**2
        )  # in MB
        max_memory_allocated = torch.cuda.max_memory_allocated(device=device) / (
            1024**2
        )  # in MB
        current_memory_cached = torch.cuda.memory_reserved(device=device) / (
            1024**2
        )  # in MB
        max_memory_cached = torch.cuda.max_memory_reserved(device=device) / (
            1024**2
        )  # in MB
        total_memory = torch.cuda.get_device_properties(device).total_memory / (
            1024**2
        )  # in MB

        print(f"Current memory allocated: {current_memory_allocated:.2f} MB")
        print(f"Max memory allocated during this run: {max_memory_allocated:.2f} MB")
        print(f"Current memory cached (reserved): {current_memory_cached:.2f} MB")
        print(
            f"Max memory cached (reserved) during this run: {max_memory_cached:.2f} MB"
        )
        print(f"Total CUDA memory: {total_memory:.2f} MB")
        return (
            max_memory_allocated / total_memory
        )  # Percentage Max Memory Allocated in Run
    else:
        print("CUDA is NOT available. Unable to show memory usage.")
        return 0
