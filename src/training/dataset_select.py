from torch.utils.data import DataLoader
from src.training.dataset_cifar10 import Cifar10Dataset
from src.training.dataset_imagenet import ImageNet100Dataset, ImageNetRDataset
from src.training.dataset_subset import create_balanced_subset, create_random_subset
from src.utils.server import is_on_server
from src.utils.device import select_device
import yaml
from pathlib import Path


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
    # Check for FFCV usage
    use_ffcv = False
    if config and config.get("use_ffcv", False) and dataset_name == "imagenet100":
        use_ffcv = True

    if use_ffcv:
        from src.training.dataset_ffcv import get_ffcv_loader
        from src.utils.device import select_device

        # Determine .beton path
        # Assuming typical structure: data_dir/imagenet100/train.beton
        if is_on_server():
            base_path = Path(config["system"]["imagenet_dir"]["server"])
        else:
            base_path = Path(config["system"]["imagenet_dir"]["local"])

        beton_filename = "train.beton" if mode == "TRAIN" else "val.beton"
        beton_path = base_path / beton_filename

        if not beton_path.exists():
            raise FileNotFoundError(
                f"FFCV .beton file not found at {beton_path}. Did you run write_ffcv_dataset.py?"
            )

        print(f"[FFCV] Loading {mode} data from {beton_path}")
        device = select_device()
        return get_ffcv_loader(beton_path, batch_size, device)

    else:
        # Standard PyTorch DataLoader
        dataset = get_dataset_obj(dataset_name, mode, config=config)
        shuffle = True if mode == "TRAIN" else False
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
