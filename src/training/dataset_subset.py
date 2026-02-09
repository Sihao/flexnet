#!/Users/donyin/miniconda3/envs/imperial/bin/python

import numpy as np
import torch, random
from tqdm import tqdm
from natsort import natsorted
from torch.utils.data import Subset
from src.training._dataset_check import check_dataset


def create_random_subset(dataset, num_samples, seed=42):
    """
    Creates a random subset of the dataset.
    [IMPORTANT]
        This function does not gurantee a balanced subset.
        I.e. the number of samples per class may not be equal.

    Parameters:
    - dataset: The original dataset from which to create a subset.
    - num_samples: The number of samples to include in the subset.
    - seed: Optional; A seed value for reproducibility of the random selection.

    Returns:
    - A torch.utils.data.Subset containing the randomly selected samples.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    num_samples = min(num_samples, len(dataset))
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    return torch.utils.data.Subset(dataset, indices)


def create_balanced_subset(dataset, num_classes, num_samples_per_class, seed=42):
    """
    Create a balanced subset of the dataset with equal number of samples per class.

    Args:
    - dataset: The dataset to subset.
    - num_classes: cifar10 has 10 classes
    - num_samples_per_class: The number of samples to keep for each class.
    - seed: Random seed for reproducibility.

    Returns:
    - A PyTorch Subset object containing a balanced subset of the original dataset.
    """
    np.random.seed(seed)  # Set the random seed for numpy operations

    # -------- initialize a dictionary to hold indices for each class --------
    class_indices = {i: [] for i in range(num_classes)}

    # -------- populate the dictionary with indices for each class --------
    print("Populating dictionary with indices for each class")

    for idx in tqdm(range(len(dataset))):
        _, label = dataset[idx]
        label = label.item() if isinstance(label, torch.Tensor) else label
        class_indices[label].append(idx)

    # -------- sample indices for each class --------
    balanced_indices = []
    for _, indices in class_indices.items():
        if len(indices) >= num_samples_per_class:
            balanced_indices.extend(
                np.random.choice(indices, num_samples_per_class, replace=False)
            )
        else:
            raise ValueError(
                f"Not enough samples in class {_} to sample {num_samples_per_class}"
            )

    # -------- shuffle the combined indices to mix classes --------
    np.random.shuffle(balanced_indices)

    # -------- create a subset with the balanced indices -------
    balanced_subset = Subset(dataset, balanced_indices)

    return balanced_subset


if __name__ == "__main__":
    from src.training.dataset_select import get_dataset_obj
    from collections import Counter
    from torch.utils.data import DataLoader
    import random

    cifar_train, cifar_test = get_dataset_obj("cifar10", "TRAIN"), get_dataset_obj(
        "cifar10", "TEST"
    )
    # imagenet_train, imagenet_test = get_dataset_obj("imagenet100", "TRAIN"), get_dataset_obj("imagenet100", "TEST")

    # # For CIFAR, to take 100 random indices
    # cifar_indices = random.sample(range(len(cifar_train)), 100)
    # cifar_train_subset = Subset(cifar_train, cifar_indices)

    # # For ImageNet, to take 2000 random indices
    # imagenet_indices = random.sample(range(len(imagenet_train)), 2000)
    # imagenet_train_subset = Subset(imagenet_train, imagenet_indices)

    balanced_cifar10_subset = create_balanced_subset(
        cifar_train, num_classes=10, num_samples_per_class=100, seed=42
    )
    check_dataset(balanced_cifar10_subset)
    # compare_train_test_dataset(cifar_train, cifar_test)
