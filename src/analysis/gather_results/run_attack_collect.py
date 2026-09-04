#!/Users/donyin/miniconda3/envs/imperial/bin/python

"""
Gather results from the runs folder after training and analysis is done for making linear regression plots / other analysis

- takes all information
- save it to a single csv file in a defined folder
- all the images that share a common name will be saved in the same folder with their names changed to the run_name

"""

import json
import numpy as np
from pathlib import Path
from natsort import natsorted
import matplotlib.pyplot as plt
from matplotlib import gridspec
from src.analysis.run_batch_processor import RunBatchProcessor
from tqdm import tqdm


class RunAttackResultsCollector(RunBatchProcessor):
    def __init__(
        self, runs_folder: Path, attack_folder_name: str, additional_labels: dict = None
    ):
        super().__init__(runs_folder)
        """attributes: run_names, runs_folder"""
        self.attack_folder_name, self.additional_labels = (
            attack_folder_name,
            additional_labels,
        )

    def process_all_runs(self, skip_existing=True):
        """
        make one attack plot for the input attack folder in the results dir
        """
        # -------- validate file structure --------
        for run_name in tqdm(
            self.run_names, desc=f"Processing Runs {self.attack_folder_name}"
        ):
            plot_path = (
                self.runs_folder
                / run_name
                / "results"
                / f"{self.attack_folder_name}_plot.png"
            )
            if plot_path.exists() and skip_existing:
                print(
                    f"Attack ({self.attack_folder_name}) Plot Already Exists For Run: {run_name}"
                )
                continue
            self.make_plot_one_run(run_name)

    def make_plot_one_run(self, run_name):
        accuracies, epsilons, adversarial_images = self.process_one_run(run_name)
        num_labels = len(self.additional_labels) + 1
        num_epsilons = len(epsilons)

        fig = plt.figure(figsize=(6 + num_labels, 16 + num_epsilons))
        gs = gridspec.GridSpec(2, 1, height_ratios=[1, 3])

        # -------- make accuracies vs epsilons plot --------
        ax1 = plt.subplot(gs[0])
        ax1.plot(epsilons, accuracies, marker="o", label=run_name)
        ax1.set_xlabel("Epsilons")
        ax1.set_ylabel("Accuracies")
        ax1.set_xlim(0, max(epsilons) * 1.01)
        ax1.set_ylim(0, 1.1)
        ax1.set_yticks(np.arange(0, 1.1, 0.1))
        ax1.grid(True, linestyle="--", alpha=0.7)

        for run, label in self.additional_labels.items():
            run_folder = self.runs_folder / run
            accuracies, epsilons, _ = self.process_one_run(run_folder)
            ax1.plot(epsilons, accuracies, marker="o", label=label)
        ax1.legend()

        # -------- make empty placeholder grid subplots --------
        ax2 = plt.subplot(gs[1])
        ax2.set_xticks([])
        ax2.set_yticks([])
        gs2 = gridspec.GridSpecFromSubplotSpec(
            num_epsilons, num_labels, subplot_spec=gs[1]
        )

        for i in range(num_epsilons):
            for j in range(num_labels):
                ax = plt.subplot(gs2[i, j])
                ax.set_xticks([])
                ax.set_yticks([])
                if i == 0:
                    if j == 0:
                        ax.set_title("Current Run", fontsize=6)
                    else:
                        ax.set_title(
                            list(self.additional_labels.values())[j - 1], fontsize=6
                        )

        # -------- add images to the grid --------
        # first add the current run
        for i, (epsilon, image_path) in enumerate(zip(epsilons, adversarial_images)):
            ax = plt.subplot(gs2[i, 0])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.imshow(plt.imread(image_path))
            ax.set_xlabel(f"Epsilon: {epsilon}", fontsize=6)

        # Adjust epsilon labels to the vertical center of the adversarial images
        for i, epsilon in enumerate(epsilons):
            ax = plt.subplot(gs2[i, 0])
            ax.text(
                -0.2,
                0.5,
                f"{epsilon.__round__(3)}",
                fontsize=6,
                transform=ax.transAxes,
                ha="right",
                va="center",
            )

        # then add the other runs
        for j, (run, label) in enumerate(self.additional_labels.items()):
            run_folder = self.runs_folder / run
            _, _, adversarial_images = self.process_one_run(run_folder)
            for i, (epsilon, image_path) in enumerate(
                zip(epsilons, adversarial_images)
            ):
                ax = plt.subplot(gs2[i, j + 1])
                ax.set_xticks([])
                ax.set_yticks([])
                ax.imshow(plt.imread(image_path))

        # -------- save plot --------
        plot_path = (
            self.runs_folder
            / run_name
            / "results"
            / f"{self.attack_folder_name}_plot.png"
        )
        plt.tight_layout()
        plt.savefig(str(plot_path), dpi=300)
        plt.close(fig)

    def process_one_run(self, run_name):
        # -------- validate file structure --------
        results_folder = self.runs_folder / run_name / "results"
        attack_folder = results_folder / self.attack_folder_name
        accuracies_json = [
            f for f in attack_folder.glob("*.json") if "accuracies" in f.stem
        ]
        assert (
            results_folder.exists()
        ), f"Results Folder Does Not Exist For Run: {run_name}"
        assert (
            attack_folder.exists()
        ), f"Attack Folder Does Not Exist For Run: {run_name}"
        assert (
            len(accuracies_json) == 1
        ), f"0 or More Than One Accuracies Json File Found For Run: {run_name}"

        # -------- load accuracies json --------
        accuracies_json = accuracies_json[0]
        accuracies_dict = json.load(accuracies_json.open("r"))
        accuracies, epsilons = (
            accuracies_dict["accuracies"],
            accuracies_dict["epsilons"],
        )
        assert len(accuracies) == len(
            epsilons
        ), f"Accuracies / Epsilons Diff Length For {run_name}"

        # -------- load images --------
        adversarial_images: list[Path] = [
            f for f in attack_folder.glob("*.png") if "adversarial_image" in f.stem
        ]
        adversarial_images: list[Path] = natsorted(
            adversarial_images, key=lambda x: x.stem
        )
        assert len(adversarial_images) == len(
            epsilons
        ), f"Adversarial Images / Epsilons Diff Length For {run_name}"

        return accuracies, epsilons, adversarial_images
