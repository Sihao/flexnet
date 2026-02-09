#!/Users/donyin/miniconda3/envs/imperial/bin/python

"""https://github.com/marcellodebernardi/loss-landscapes/tree/master"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn
from pathlib import Path
from torch.utils.data import DataLoader
from src.analysis.run_loader import RunLoader
from src.training.dataset_select import get_dataset_obj
from src._loss_surface.loss_landscapes import metrics
from src._loss_surface.loss_landscapes import random_plane, linear_interpolation
from src.utils.device import select_device


class PlotLossSurface:
    def __init__(
        self,
        run_folder,
        steps=100,
        distance=1,
        normalisation: str = "filter",
        criterion=torch.nn.CrossEntropyLoss(),
    ):
        # fmt: off
        torch.manual_seed(42); np.random.seed(42); self.steps = steps; self.distance = distance; self.normalisation = normalisation
        self.run_loader_initial = RunLoader(run_folder, whether_load_checkpoint=False)
        self.run_loader_terminal = RunLoader(run_folder, whether_load_checkpoint=True)
        batch_size = self.run_loader_initial.config.batch_size
        # fmt: on

        # -------- [ make some secondary data ] --------
        self.device = select_device()
        dataset = get_dataset_obj(self.run_loader_initial.config.dataset, "TRAIN")
        self.dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        self.x, self.y = iter(self.dataloader).__next__()
        self.x, self.y = self.x.to(self.device), self.y.to(self.device)
        self.metric = metrics.Loss(criterion, self.x, self.y)

    def prepare_random_plane(self, save_as=None):
        """prepare the random plane data for both 2d and 3d surface plotting"""
        self.loss_data_plane = random_plane(
            self.run_loader_terminal.model,
            self.metric,
            distance=self.distance,
            steps=self.steps,
            normalization=self.normalisation,
            deepcopy_model=False,
        )
        if save_as:
            np.save(save_as, self.loss_data_plane)


if __name__ == "__main__":
    pass
