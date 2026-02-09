#!/Users/donyin/miniconda3/envs/imperial/bin/python

"""
visual testing whether a dataset is functioning properly
"""

import numpy as np
import seaborn as sns
from tqdm import tqdm
from natsort import natsorted
import matplotlib.pyplot as plt
from collections import Counter
import torchvision.utils as vutils
from torch.utils.data import DataLoader


def check_dataset(dataset):
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    # Create a figure for the combined plots
    fig = plt.figure(figsize=(8, 21))  # Adjust the figure size as needed

    # Add a grid for the image plot
    grid_ax = plt.subplot2grid((2, 1), (0, 0), fig=fig)
    bar_ax = plt.subplot2grid((2, 1), (1, 0), fig=fig)

    # -------- Creating a grid of images --------
    images, labels = next(iter(dataloader))
    grid = vutils.make_grid(images, nrow=4, normalize=True)
    grid_ax.imshow(np.transpose(grid.numpy(), (1, 2, 0)))
    grid_ax.axis("off")
    images_value_range = f"{images.min().item():.2f} - {images.max().item():.2f}"
    grid_ax.set_title(f"Image Shape: {images.shape}\nValue Range: {images_value_range}")

    # -------- Calculating coordinates to place labels --------
    # Adjustments for text placement on the image grid
    num_rows, num_cols = 4, 4
    step_x, step_y = images.shape[3] + 2, images.shape[2] + 2
    start_x, start_y = step_x / 2, step_y / 2
    for idx, label in enumerate(labels):
        x = start_x + (idx % num_cols) * step_x
        y = start_y + (idx // num_cols) * step_y
        try:
            text = list(dataset.classes)[label.item()]
        except AttributeError:
            text = label.item()
        grid_ax.text(
            x,
            y,
            text,
            color="white",
            ha="center",
            va="center",
            fontsize=12,
            transform=grid_ax.transData,
        )

    # -------- Calculating label distribution --------
    label_counts = Counter()
    for _, labels in tqdm(dataloader):
        label_counts.update(labels.cpu().numpy())
    sorted_label_counts = natsorted(label_counts.items(), key=lambda x: x[0])

    # -------- Plotting label distribution as a horizontal bar plot --------
    labels, counts = zip(*sorted_label_counts)  # Unzipping labels and counts
    sns.barplot(y=list(labels), x=list(counts), orient="h", ax=bar_ax)
    bar_ax.set_xlabel("Count")
    bar_ax.set_ylabel("Label")
    bar_ax.set_title("Label Distribution")

    plt.tight_layout()
    plt.savefig("_dataset_distribution.png")


if __name__ == "__main__":
    pass
    """see tests in dataset_subset.py"""
