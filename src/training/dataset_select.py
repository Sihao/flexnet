from torch.utils.data import DataLoader
from src.training.dataset_cifar10 import Cifar10Dataset
from src.training.dataset_imagenet import ImageNet100Dataset, ImageNetRDataset
from src.training.dataset_subset import create_balanced_subset, create_random_subset
from src.utils.server import is_on_server
import os
import yaml
from pathlib import Path
from torchvision.datasets import ImageFolder
from torchvision import transforms


def get_dataset_obj(dataset_name: str, mode: str, config=None):
    """
    This function returns the dataset object based on the dataset name and mode (TRAIN or VAL).
    """
    root_path = Path(__file__).parents[2]
    if config is None:
        with open(root_path / "configurations.yml", "r") as f:
            config = yaml.safe_load(f)

    if dataset_name == "cifar10":
        return Cifar10Dataset(mode)
    elif (
        dataset_name == "cifar10-down-50"
    ):  # [NOTE] 10 classes, 100 images per class / totalling 1000 images / Linnea used
        cifar10_train = Cifar10Dataset(mode)
        return create_balanced_subset(
            cifar10_train, num_classes=10, num_samples_per_class=100
        )
    elif dataset_name == "cifar10-random-small-100":
        cifar10_train = Cifar10Dataset(mode)
        return create_random_subset(cifar10_train, num_samples=100, seed=42)
    elif dataset_name == "cifar10-random-small-500":
        cifar10_train = Cifar10Dataset(mode)
        return create_random_subset(cifar10_train, num_samples=500, seed=42)
    elif dataset_name == "cifar10-random-small-gaussian-noise-0.5":
        cifar10_train = Cifar10Dataset(mode, gaussian_noise_std=0.5)
        return create_random_subset(cifar10_train, num_samples=100, seed=42)
    elif dataset_name == "imagenet100":
        if is_on_server():
            folder_path = Path(config["system"]["imagenet_dir"]["server"])
        else:
            folder_path = Path(config["system"]["imagenet_dir"]["local"])
        return ImageNet100Dataset(folder=folder_path, mode=mode)
    elif dataset_name == "imagenet":
        # Node-local staging override: when training I/O on the network filesystem
        # is unreliable (e.g. the post-cutover fs8 Lustre client wedges the
        # DataLoader's small-random/mmap read pattern in cl_sync_io_wait), the job
        # script stages the dataset to node-local disk and exports
        # IMAGENET_LOCAL_DIR so training reads locally and never touches Lustre.
        local_override = os.environ.get("IMAGENET_LOCAL_DIR")
        if local_override:
            folder_path = Path(local_override)
        elif is_on_server():
            folder_path = Path(config["system"]["imagenet_full_dir"]["server"])
        else:
            folder_path = Path(config["system"]["imagenet_full_dir"]["local"])

        # Standard ImageNet transforms
        if mode == "TRAIN":
            transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            return ImageFolder(str(folder_path / "train"), transform=transform)
        else:
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            return ImageFolder(str(folder_path / "val"), transform=transform)
    elif dataset_name == "imagenet-r":
        # ImageNet-R (OOD)
        # Assumed local path: data/imagenet-r (copied previously)
        folder_path = root_path / "data/imagenet-r"

        # Get labels from ImageNet-100 path (from config)
        if is_on_server():
            in100_path = Path(config["system"]["imagenet_dir"]["server"])
        else:
            in100_path = Path(config["system"]["imagenet_dir"]["local"])

        labels_file = in100_path / "Labels.json"
        return ImageNetRDataset(folder=folder_path, labels_file=labels_file)
    elif dataset_name == "deepinterpolation":
        from src.deepinterpolation.loader import DeepInterpolationDataset

        if config is None:
            raise ValueError(
                "DeepInterpolationDataset requires 'config' to be passed to get_dataset_obj"
            )

        print(f"DEBUG: get_dataset_obj config keys: {list(config.keys())}")
        tiff_path = config.get("tiff_file_path")
        print(f"DEBUG: get_dataset_obj extracted tiff_path: {tiff_path}")

        N = config.get("frame_window_N")
        return DeepInterpolationDataset(tiff_path=tiff_path, frame_window_N=N)
    else:
        raise ValueError(f"Invalid dataset name: {dataset_name}")


def get_dataloader(dataset_name: str, mode: str, batch_size: int, config=None):
    """
    Factory to get the appropriate dataloader.
    Supports standard PyTorch DataLoader and FFCV.
    """
    # Check for FFCV usage (supported for imagenet100 and full imagenet).
    use_ffcv = bool(
        config
        and config.get("use_ffcv", False)
        and dataset_name in ("imagenet100", "imagenet")
    )

    if use_ffcv:
        from src.training.dataset_ffcv import get_ffcv_loader
        from src.utils.device import select_device

        # Resolve the directory holding train.beton / val.beton.
        if dataset_name == "imagenet":
            # Full ImageNet: prefer the node-local staging override
            # (IMAGENET_BETON_DIR, mirroring IMAGENET_LOCAL_DIR for the JPEG
            # path); else the system block's imagenet_beton_dir. The per-run
            # config has no 'system' block, so fall back to the root
            # configurations.yml like get_dataset_obj does.
            beton_override = os.environ.get("IMAGENET_BETON_DIR")
            if beton_override:
                base_path = Path(beton_override)
            else:
                system_cfg = (config or {}).get("system")
                if system_cfg is None:
                    root_cfg = Path(__file__).parents[2] / "configurations.yml"
                    with open(root_cfg, "r") as f:
                        system_cfg = yaml.safe_load(f)["system"]
                key = "server" if is_on_server() else "local"
                base_path = Path(system_cfg["imagenet_beton_dir"][key])
        else:
            key = "server" if is_on_server() else "local"
            base_path = Path(config["system"]["imagenet_dir"][key])

        beton_filename = "train.beton" if mode == "TRAIN" else "val.beton"
        beton_path = base_path / beton_filename

        if not beton_path.exists():
            raise FileNotFoundError(
                f"FFCV .beton file not found at {beton_path}. Generate it with "
                "scripts/hpc/ffcv_write_betons.sh "
                "(writer: src/training/write_ffcv_imagenet_full.py)."
            )

        print(f"[FFCV] Loading {mode} data from {beton_path}")
        device = select_device()
        num_workers = int(config.get("ffcv_num_workers", 8))
        return get_ffcv_loader(
            beton_path,
            batch_size,
            device,
            is_train=(mode == "TRAIN"),
            num_workers=num_workers,
        )

    else:
        # Standard PyTorch DataLoader
        dataset = get_dataset_obj(dataset_name, mode, config=config)
        shuffle = True if mode == "TRAIN" else False
        num_workers = 8 if "imagenet" in dataset_name else 0
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                          num_workers=num_workers, pin_memory=True)
