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
import torch.distributed as dist
from src.utils.device import select_device, is_ddp, is_main_process
from src.utils.general import apply_kaiming_initialization
from src.training.dataset_select import get_dataset_obj
from src.training.dataset_subset import create_random_subset
from src.utils.simple_logger import SimpleLogger
from src.utils.checkpoint_retention import prune_checkpoints
import pandas
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from src.modules import models as models  # e.g., getattr(models, "FlexResNet") / VGG16


class RunLoader:
    def __init__(
        self,
        run_folder: Path,
        whether_load_checkpoint: bool = True,
        whether_instantiate_model=True,
        device=None,
        load_epoch: int = None,
        allow_cold_start: bool = False,
    ):
        # regular stuff
        self.run_folder = Path(run_folder)
        self.device = select_device() if device is None else device

        # for copying
        self.whether_load_checkpoint = whether_load_checkpoint
        # When set, load this specific epoch's checkpoint instead of the
        # highest-epoch one. Used by the iso-accuracy trajectory analyses,
        # which target specific matched-accuracy checkpoints.
        self.load_epoch = load_epoch
        # Only a genuine training cold-start may persist an untrained epoch-0
        # checkpoint into an empty checkpoints/ dir (see _init_checkpoint).
        # Analysis loaders keep the default False so a missing trained
        # checkpoint fails loudly instead of being silently fabricated.
        self.allow_cold_start = allow_cold_start

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
                momentum=self.config.get("momentum", 0.9),
                weight_decay=weight_decay,
                nesterov=self.config.get("nesterov", False),
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
                self.optimizer,
                T_max=self.config.get("cosine_t_max", 200),
                eta_min=self.config.get("cosine_eta_min", 1e-6),
            )
        elif scheduler_name == "WarmupCosine":
            # Linear warmup for `warmup_epochs`, then cosine decay to
            # `cosine_eta_min` over the remaining epochs (epoch-granularity:
            # scheduler.step() is called once per epoch in the training loop).
            # Standard, architecture-agnostic recipe applied identically to
            # flex and vanilla so accuracy differences reflect architecture,
            # not tuning. State_dict round-trips for resume across job chains.
            warmup_epochs = int(self.config.get("warmup_epochs", 5))
            total_epochs = int(self.config.get("target_epoch", 90))
            eta_min = self.config.get("cosine_eta_min", 0.0)
            start_factor = self.config.get("warmup_start_factor", 0.1)
            cosine_epochs = max(1, total_epochs - warmup_epochs)
            if warmup_epochs > 0:
                warmup = lr_scheduler.LinearLR(
                    self.optimizer,
                    start_factor=start_factor,
                    end_factor=1.0,
                    total_iters=warmup_epochs,
                )
                cosine = lr_scheduler.CosineAnnealingLR(
                    self.optimizer, T_max=cosine_epochs, eta_min=eta_min
                )
                self.scheduler = lr_scheduler.SequentialLR(
                    self.optimizer,
                    schedulers=[warmup, cosine],
                    milestones=[warmup_epochs],
                )
            else:
                self.scheduler = lr_scheduler.CosineAnnealingLR(
                    self.optimizer, T_max=total_epochs, eta_min=eta_min
                )
        elif scheduler_name == "MultiStepLR":
            self.scheduler = lr_scheduler.MultiStepLR(
                self.optimizer,
                milestones=self.config.get("lr_milestones", [30, 60, 80]),
                gamma=self.config.get("lr_gamma", 0.1),
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
        # Under DDP, only rank 0 writes the initial checkpoint; other ranks
        # wait at a barrier and then see the file on disk. Without this guard,
        # all ranks race on the same .checkpoint_N.pth.tmp path and the losers
        # of os.replace() hit FileNotFoundError.
        #
        # The initial (untrained) checkpoint is persisted ONLY for a genuine
        # training cold-start (allow_cold_start=True). Analysis loaders keep the
        # default allow_cold_start=False: an empty checkpoints/ dir then means
        # the trained checkpoint is missing (e.g. reclaimed to NAS), and
        # _load_model_and_optimizer() raises loudly. Previously this method
        # fabricated an epoch-0 checkpoint unconditionally, which a later
        # whether_load_checkpoint=True loader silently picked up as the "latest"
        # epoch and scored an untrained model -- the shared-probe find failure.
        ckpt_dir = self.run_folder / "checkpoints"
        ckpt_dir.mkdir(exist_ok=True, parents=True)
        if (
            self.allow_cold_start
            and is_main_process()
            and not list(ckpt_dir.glob("*.pth"))
        ):
            self.save_checkpoint()
        if is_ddp() and dist.is_initialized():
            dist.barrier()

    def _load_model_and_optimizer(self):
        ckpt_dir = self.run_folder / "checkpoints"
        # Pick the highest-epoch checkpoint matching checkpoint_<N>.pth.
        # The training loop keeps the most recent file plus optional milestone
        # snapshots (see save_checkpoint), so multiple files are normal.
        candidates = []
        for f in ckpt_dir.glob("checkpoint_*.pth"):
            try:
                epoch = int(f.stem.replace("checkpoint_", ""))
            except ValueError:
                continue
            candidates.append((epoch, f))
        if not candidates:
            raise FileNotFoundError(
                f"No checkpoint_<N>.pth found in {ckpt_dir}"
            )
        candidates.sort(key=lambda x: x[0])
        if self.load_epoch is not None:
            match = [c for c in candidates if c[0] == self.load_epoch]
            if not match:
                available = ", ".join(str(e) for e, _ in candidates)
                raise FileNotFoundError(
                    f"Requested epoch {self.load_epoch} not found in {ckpt_dir} "
                    f"(available: {available})"
                )
            latest_epoch, latest_path = match[0]
        else:
            latest_epoch, latest_path = candidates[-1]

        print(f"[DEBUG] Loading checkpoint with map_location={self.device}")
        print(f"[DEBUG] Loading {latest_path.name} (epoch {latest_epoch})")
        save_content = torch.load(latest_path, map_location=self.device)
        self.model.load_state_dict(save_content["model_state_dict"], strict=False)
        try:
            self.optimizer.load_state_dict(save_content["optimizer_state_dict"])
        except Exception as e:
            print(
                f"[WARNING] Optimizer load failed: {e}. Continue as we only need model for analysis."
            )
        if (
            self.scheduler is not None
            and "scheduler_state_dict" in save_content
        ):
            try:
                self.scheduler.load_state_dict(save_content["scheduler_state_dict"])
            except Exception as e:
                print(f"[WARNING] Scheduler load failed: {e}.")
        self.current_epoch, self.current_loss = (
            save_content["epoch"],
            save_content["loss"],
        )
        # Re-sync scheduler to the absolute resumed epoch. The training loop
        # calls scheduler.step() AFTER save_checkpoint(), so every chain
        # rollover (24h-walltime resubmission) silently drops one step; the lag
        # accumulates and the MultiStepLR milestones (e.g. epoch 30/60/80 LR
        # drops) fire late or never. MultiStepLR/CosineAnnealingLR LR is a
        # closed-form function of last_epoch, so force it to current_epoch and
        # reapply the correct LR. Idempotent and robust to any prior drift.
        if self.scheduler is not None:
            try:
                self.scheduler.last_epoch = self.current_epoch
                closed = self.scheduler._get_closed_form_lr()
                for pg, lr in zip(self.optimizer.param_groups, closed):
                    pg["lr"] = lr
                self.scheduler._last_lr = list(closed)
                print(
                    f"[scheduler] re-synced last_epoch={self.current_epoch}, "
                    f"lr={closed[0]:.5f}"
                )
            except Exception as e:
                print(f"[WARNING] scheduler re-sync failed: {e}")
        return self

    def save_checkpoint(self):
        """
        save: model (state_dict); optimizer (state_dict); epoch number; loss
        Uses atomic write (tmp + rename) to prevent corruption if killed mid-save.
        """
        ckpt_dir = self.run_folder / "checkpoints"
        ckpt_dir.mkdir(exist_ok=True, parents=True)

        save_name = self.current_epoch
        target = ckpt_dir / f"checkpoint_{save_name}.pth"
        tmp = ckpt_dir / f".checkpoint_{save_name}.pth.tmp"

        save_content = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epoch": self.current_epoch,
            "loss": self.current_loss,
        }
        if self.scheduler is not None:
            save_content["scheduler_state_dict"] = self.scheduler.state_dict()
        torch.save(save_content, tmp)
        os.replace(tmp, target)  # atomic on POSIX

        # Clean up old checkpoints only after new one is safely written.
        # Only deletes checkpoints the archive manifest confirms are mirrored
        # elsewhere (see src/utils/checkpoint_retention.py); with no manifest
        # present, nothing is deleted. Also preserves milestone epochs (every
        # Nth) for later analysis.
        prune_checkpoints(
            ckpt_dir,
            self.run_folder,
            keep_latest=self.config.get("keep_latest_checkpoints", 2),
            keep_every_n=self.config.get("keep_every_n_epochs"),
            current_epoch=self.current_epoch,
        )

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
    print("RunLoader is a library module. Import it and call RunLoader(path_to_run_folder).")
