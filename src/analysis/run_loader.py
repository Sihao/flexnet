#!/Users/donyin/miniconda3/envs/imperial/bin/python
"""
This class:
1. takes the folder and reads the configuration.json file
2. construct the model and load the checkpoint
- this serves as a base class for all the analysis scripts
- in other words, it loads the model and the checkpoint

e.g., loader = RunLoader(some_path)

you get:
        loader.model
        loader.optimizer
        loader.logger
        loader.current_epoch
        loader.current_loss
        loader.criterion

"""

import json, torch, os
from torch import nn

from torch import optim
from torch.optim import lr_scheduler
from pathlib import Path
from natsort import natsorted

from torch.utils.data import DataLoader
from src.utils.device import select_device
from src.utils.general import apply_kaiming_initialization
from src.training.dataset_select import get_dataset_obj
from src.training.dataset_subset import create_random_subset
from src.utils.simple_logger import SimpleLogger
import pandas
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from src.modules import models  # e.g., getattr(models, "SimpleFlexNet") / VGG16


class RunLoader:
    def __init__(
        self,
        run_folder: Path,
        whether_load_checkpoint: bool = True,
        whether_instantiate_model=True,
        device=None,
    ):
        # regular stuff
        self.run_folder = Path(run_folder)
        self.device = select_device() if device is None else device

        # for copying
        self.whether_load_checkpoint = whether_load_checkpoint

        self._load_config()

        if whether_instantiate_model:
            self._init_plain_model()
            self._init_checkpoint()

        if whether_instantiate_model and whether_load_checkpoint:
            print(f"[DEBUG] RunLoader initialized with device: {self.device}")
            print("Loading Checkpoint")
            self._load_model_and_optimizer()

        # Ensure logs directory exists before creating database
        logs_dir = self.run_folder / "logs"
        logs_dir.mkdir(exist_ok=True, parents=True)
        self.logger = SimpleLogger(filename=logs_dir / "metrics.db")

    # ---- loading model ----
    def _load_config(self):
        configurations_dir = self.run_folder / "configurations.json"
        self.config = json.load(configurations_dir.open("r"))
        return self

    def _init_plain_model(self):
        self.model = getattr(models, self.config.get("network"))(config=self.config)
        apply_kaiming_initialization(self.model)
        self.model, learning_rate = self.model.to(self.device), self.config.get(
            "learning_rate"
        )
        optimizer_name = self.config.get("optimizer", "SGD")
        weight_decay = self.config.get("weight_decay", 1e-5)

        if optimizer_name == "SGD":
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=learning_rate,
                momentum=0.9,
                weight_decay=weight_decay,
            )
        elif optimizer_name == "ADAM":
            self.optimizer = optim.Adam(
                self.model.parameters(), lr=learning_rate, weight_decay=weight_decay
            )
        elif optimizer_name == "ADAMW":
            self.optimizer = optim.AdamW(
                self.model.parameters(), lr=learning_rate, weight_decay=weight_decay
            )
        else:
            raise NotImplementedError

        # Scheduler
        scheduler_name = self.config.get("scheduler", None)
        if scheduler_name == "CosineAnnealingLR":
            self.scheduler = lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=200, eta_min=1e-6
            )
        elif scheduler_name == "ReduceLROnPlateau":
            self.scheduler = lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="min", factor=0.1, patience=10
            )
        else:
            self.scheduler = None

        self.current_epoch, self.current_loss = 0, 0
        label_smoothing = self.config.get("label_smoothing", 0.0)
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        return self

    # ---- init ----
    def _do_a_dummy_backward_pass(self):  # [IMPORTANT]: make dataset input
        """this is useful when plotting the gradients as well as debuggin model architecture"""
        torch.manual_seed(42)
        dataset_name = self.config.get("dataset")
        dataset = get_dataset_obj(dataset_name, "TRAIN")
        dataset = create_random_subset(dataset, self.config.get("batch_size"))
        train_loader = DataLoader(
            dataset, batch_size=self.config.get("batch_size"), shuffle=True
        )
        images, labels = next(iter(train_loader))
        images, labels = images.to(self.device), labels.to(self.device)
        outputs = self.model(images)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(outputs, labels)
        loss.backward()
        return self

    def _do_a_dummy_forward_pass(self):
        """[IMPORTANT]: run a dummy forward pass to initialise the model first"""
        torch.manual_seed(42)
        self.model.train()
        self.model(torch.rand(1, *self.config.in_dimensions).to(self.device))
        self.model.eval()
        return self

    # ---- save and load ----
    def _init_checkpoint(self):
        ckpt_dir = self.run_folder / "checkpoints"
        ckpt_dir.mkdir(exist_ok=True, parents=True)
        if not list(ckpt_dir.glob("*.pth")):
            self.save_checkpoint()

    def _load_model_and_optimizer(self):
        ckpt_dir = self.run_folder / "checkpoints"
        ckpt_files = list(ckpt_dir.glob("*.pth"))
        ckpt_files = natsorted(ckpt_files)

        if not len(ckpt_files) == 1:
            print("[WARNING]")
            print(f"Expecting 1 .pth file, found {len(ckpt_files)}.")
            print(f"run_folder: {self.run_folder}")
            assert False, "Check the run folder"

        print(f"[DEBUG] Loading checkpoint with map_location={self.device}")
        save_content = torch.load(ckpt_files[0], map_location=self.device)
        self.model.load_state_dict(save_content["model_state_dict"], strict=False)
        try:
            self.optimizer.load_state_dict(save_content["optimizer_state_dict"])
        except Exception as e:
            print(
                f"[WARNING] Optimizer load failed: {e}. Continue as we only need model for analysis."
            )
        self.current_epoch, self.current_loss = (
            save_content["epoch"],
            save_content["loss"],
        )
        return self

    def save_checkpoint(self):
        """
        save: model (state_dict); optimizer (state_dict); epoch number; loss
        """
        ckpt_dir = self.run_folder / "checkpoints"
        ckpt_dir.mkdir(exist_ok=True, parents=True)

        existing = ckpt_dir.glob("*.pth")
        existing = [f for f in existing if f.name.startswith("checkpoint")]
        [os.remove(f) for f in existing]

        save_name = self.current_epoch
        save_content = {"model_state_dict": self.model.state_dict()}
        save_content.update({"optimizer_state_dict": self.optimizer.state_dict()})
        save_content.update({"epoch": self.current_epoch, "loss": self.current_loss})
        torch.save(save_content, ckpt_dir / f"checkpoint_{save_name}.pth")

    def draw_conv_ratio_and_homogeneity(self, save_as: Path):
        """
        Draw both conv ratio and homogeneity side by side in a single plot
        Also save the data in a json file for later use
        """
        save_as.parent.mkdir(exist_ok=True, parents=True)
        dataframe = self.logger.get_dataframe()

        # Get conv ratio data
        conv_ratio_cols = [
            col for col in dataframe.columns if col.startswith("Conv Ratio")
        ]
        conv_ratio_cols = natsorted(conv_ratio_cols)
        conv_ratios = dataframe[conv_ratio_cols].iloc[-1]
        conv_ratios = pandas.to_numeric(conv_ratios)
        conv_ratios = dict(conv_ratios)

        # Get homogeneity data
        homogeneity_cols = [
            col for col in dataframe.columns if col.startswith("Binariness")
        ]
        homogeneity_cols = natsorted(homogeneity_cols)
        homogeneity = dataframe[homogeneity_cols].iloc[-1]
        homogeneity = pandas.to_numeric(homogeneity)
        homogeneity = dict(homogeneity)

        # Save data to json
        config_with_data = self.config.copy()
        config_with_data["conv_ratios"] = {k: float(v) for k, v in conv_ratios.items()}
        config_with_data["homogeneity"] = {k: float(v) for k, v in homogeneity.items()}
        save_path = save_as.parent / f"{save_as.stem}.json"
        with open(save_path, "w") as f:
            json.dump(config_with_data, f, indent=4)

        # Plot side by side bars
        plt.figure(figsize=(15, 6))
        x = np.arange(len(conv_ratios))
        width = 0.35

        plt.bar(x - width / 2, list(conv_ratios.values()), width, label="Conv Ratio")
        plt.bar(x + width / 2, list(homogeneity.values()), width, label="Homogeneity")

        plt.xlabel("Layer")
        plt.ylabel("Value")
        plt.title("Convolution Ratio and Homogeneity by Layer")
        plt.xticks(x, [f"Layer {i}" for i in range(len(conv_ratios))], rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_as, bbox_inches="tight", dpi=300)
        plt.close()


if __name__ == "__main__":
    pass
