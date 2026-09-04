#!/Users/donyin/miniconda3/envs/imperial/bin/python

"""
get accuracies on test dataset from trained checkpoints
this has to be called seperately
"""

import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import DataLoader
from src.analysis.run_loader import RunLoader
from src.flex_neurons.training.loop import get_accuracy, get_balanced_accuracy


def get_test_accuracies(run_loader, dataset):
    """
    get runs' accuracy on the test dataset
    """
    print("Getting Test Accuracies")
    model, device = run_loader.model, run_loader.device
    model.eval()

    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0)

    with torch.no_grad():
        accuracies, balanced_accuracies = [], []
        for batch in tqdm(dataloader):
            images, labels = batch
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            accuracy, balanced_accuracy = get_accuracy(
                logits, labels
            ), get_balanced_accuracy(logits, labels)
            accuracies.append(accuracy)
            balanced_accuracies.append(balanced_accuracy)

    return np.mean(accuracies), np.mean(balanced_accuracies)


if __name__ == "__main__":
    run_loader = RunLoader(Path("/Users/donyin/Desktop/experiment/00b67141"))
