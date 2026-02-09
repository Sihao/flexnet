import torch


def select_device(priority=["cuda", "cpu"]):  # mps in the middle
    """
    Selects the device based on the given priority list.
    If top priority device is not available, it will try the next device in the list.
    E.g., it will try to select "cuda" first, then "mps", and finally "cpu"
    So we can have the same code on local machine and on the server.
    Just with

    - local: select_device(["cuda", "cpu"])
    - server: select_device(["cuda", "cpu"])

    Parameters:
        - priority (list): List of strings representing device priorities.

    Returns:
        - torch.device: Device selected based on priority.
    """

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
