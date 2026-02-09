#!/usr/bin/env python

"""
DEPRECATED: Use src/analysis/run_attack_comparison.py directly which uses torchattacks.

this folder contains the attack class for various adversarial attacks, using torchattacks:
https://github.com/Harry24k/adversarial-attacks-pytorch
usage:

- feed in a model and a dataloader
- define a method
- define where to save the plot and run main

# ======== [main bit] ========
    attack = Attack(attack="PGD", data_loader=test_loader)  # FGSM, PGD
    attack.add_model(model=model, state_dict_path=pretrained_model)
    attack.run_main(save_to=Path("attack.png"))
"""


import numpy as np
from PIL import Image

from pathlib import Path
import matplotlib.pyplot as plt
import torch, torchattacks, json
from scipy.integrate import trapezoid
import torchvision.utils as vutils
from torch.utils.data import DataLoader
from src.utils.device import select_device
from src.analysis.run_loader import RunLoader
from src.training.dataset_select import get_dataset_obj
from src.training.dataset_subset import create_random_subset
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from src.analysis.run_loader_image.intermediate_plot import IntermediateProcessViz


class AttackGaussianNoise:
    def __init__(self, mean=0.0, std=1.0):
        """
        this should make an attack object that can be used to attack a model
        this object has to be used such as:
            dv_images = attack(images, labels)
        where the images are the batch of images to be attacked
        scaled to (0, 1) and the labels are the target labels

        since this is gaussian noise, the labels are not needed
        the init method should take in the parameters for the gaussian noise
        """
        self.mean, self.std = mean, std

    def __call__(self, images, labels=None):
        """
        Add Gaussian noise to a batch of images.

        Parameters
        ----------
        images : torch.Tensor
            Batch of images.
        labels : torch.Tensor, optional
            Labels for the batch. Not used in this attack.

        Returns
        -------
        torch.Tensor
            Batch of images with added Gaussian noise.
        """
        noise = torch.randn_like(images) * self.std + self.mean
        return images + noise


def save_image_batch(batch, filename):
    """[NOTE] Save a batch of images as a single grid image, for debugging purposes ONLY."""
    # ---- convert to grid ----
    scale_to_255 = lambda x: (x - x.min()) / (x.max() - x.min()) * 255
    batch = scale_to_255(batch)
    grid_img = vutils.make_grid(batch, nrow=int(len(batch) ** 0.5))

    # ---- convert to PIL image ----
    ndarr = grid_img.permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    img = Image.fromarray(ndarr)
    img.save(filename)


def attack_batch(batch, target, device, attack):
    """
    [NOTE] The main low level function that takes a batch of images and return a batch of adversarial images.

    Parameters
        ----------
        batch : torch.Tensor
            Batch of images.
        target : torch.Tensor
            Labels for the batch.
        device : torch.device
            Device to run the attack on.
        attack : Attack
            The attack object.
    """
    # first save max and min for entire batch
    v_mins, v_maxs = batch.min(), batch.max()

    # -------- scaling images to (0 - 1) --------
    scale_to_0_1 = lambda x: (x - x.min()) / (x.max() - x.min())
    batch_scaled = scale_to_0_1(batch).to(device).requires_grad_()

    # --------  Create adversarial images
    batch_scaled = attack(batch_scaled, target)

    # ------- scale back to original range ----
    scale_back = lambda x: (x - 0.0) / (0.99999 - 0.0) * (v_maxs - v_mins) + v_mins
    batch_scaled = scale_back(batch_scaled)

    # # --------  inspect what's going on --------
    # save_image_batch(batch_scaled, "test.png")
    # save_image_batch(batch, "test_orig.png")

    return batch_scaled


def top_n_accuracy(adv_outputs, target, n):
    """
    Calculate the top-n accuracy.

    Parameters
    ----------
    adv_outputs : Tensor
        The output predictions from the model.
    target : Tensor
        The true labels.
    n : int
        The number of top predictions to consider for accuracy calculation.

    Returns
    -------
    float
        The top-n accuracy.
    """

    _, pred = adv_outputs.topk(n, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    correct = correct[:n].reshape(-1).float().sum(0, keepdim=True)
    return correct.item()


def area_under_curve(values, dx=1):
    return trapezoid(values, dx=dx)


class Attack:
    def __init__(
        self,
        run_loader,
        attack: str,
        data_loader: DataLoader,
        accuracy_top_n: int,
        save_to: Path,
    ):
        """
        example:
        from tests._utils.models import model, pretrained_model
        from tests._utils.dataloader import test_loader
        attack = Attack(attack="PGD", data_loader=test_loader)  # FGSM, PGD
        attack.add_model(model=model, state_dict_path=pretrained_model)
        attack.run_main(save_to=Path("attack.png"))
        """
        self.device = select_device()
        self.accuracy_top_n = accuracy_top_n
        self.attack_name, self.data_loader = attack, data_loader
        self.run_loader, self.save_to = run_loader, save_to
        self.save_to.mkdir(parents=True, exist_ok=True)

        if self.attack_name == "FGSM":  # [white-box]
            self.epsilons_base = 0
            self.epsilons_max = 0.64
            self.epsilons_num = 23
        elif self.attack_name == "PGD":  # [white-box]
            self.epsilons_base = 0
            self.epsilons_max = 23 / 255
            self.epsilons_num = 23
        elif self.attack_name == "SPGD":  # [white-box] super PGD with stronger epsilon
            self.epsilons_base = 0
            self.epsilons_max = 0.5
            self.epsilons_num = 23
        elif (
            self.attack_name == "SPSA"
        ):  # [black-box] (Simultaneous Perturbation Stochastic Approximation)
            self.epsilons_base = 0
            self.epsilons_max = 0.21
            self.epsilons_num = 23
        elif self.attack_name == "OnePixel":  # [black-box] (One Pixel Attack)
            self.epsilons_base = 1
            self.epsilons_max = 42
            self.epsilons_num = 23
        elif (
            self.attack_name == "JSMA"
        ):  # [white-box] (Jacobian-based Saliency Map Attack)
            self.epsilons_base = 0
            self.epsilons_max = 1.5
            self.epsilons_num = 23
        elif self.attack_name == "Jitter":  # [white-box] (Jitter Attack)
            self.epsilons_base = 0
            self.epsilons_max = 0.3
            self.epsilons_num = 23
        elif self.attack_name == "GaussianNoise":  # [non-attack] (Gaussian Noise)
            self.epsilons_base = 0
            self.epsilons_max = 0.42
            self.epsilons_num = 23
        else:
            raise NotImplementedError

        self.epsilons = np.linspace(
            self.epsilons_base, self.epsilons_max, self.epsilons_num
        )

    # ------ [ top level helpers ] ------
    def set_model(self, model, model_name: str = None):
        model.to(self.device).eval()
        self.model, self.model_name = model, model_name

    def test_save_data(self):
        """
        [NOTE] Main Function. This method saves the test data. It first creates the directory where the data will be saved if it doesn't exist. Then it tests the model and calculates the area under the curve (AUC) of the model's accuracy.
        The AUC and the model's accuracy for different epsilon values are saved as json files.
        Finally, it saves the original and adversarial images for each epsilon value.
        """
        self.save_to.mkdir(parents=True, exist_ok=True)
        accuracies, examples = self._test_model(self.model)
        auc = area_under_curve(accuracies, dx=self.epsilons_max / self.epsilons_num)

        with open(
            self.save_to
            / f"{self.attack_name}_model_auc_score_top_{self.accuracy_top_n}.json",
            "w",
        ) as f:
            json.dump(
                {
                    f"{self.attack_name} Attack Area Under Curve Top {self.accuracy_top_n}": auc
                },
                f,
                indent=4,
            )

        with open(
            self.save_to
            / f"{self.attack_name}_model_accuracies_top_{self.accuracy_top_n}.json",
            "w",
        ) as f:
            json.dump(
                {"epsilons": self.epsilons.tolist(), "accuracies": accuracies},
                f,
                indent=4,
            )

        for _, ex in enumerate(examples):
            epsilon, example_pair = ex["epsilon"], ex["examples"]
            original, adversarial = example_pair
            original, adversarial = original.transpose(1, 2, 0), adversarial.transpose(
                1, 2, 0
            )
            path_original = self.save_to / Path(f"original_image.png")
            path_adversarial = self.save_to / Path(
                f"{self.attack_name}_adversarial_image_{epsilon:.4f}.png"
            )

            (
                plt.imsave(path_original, original, format="png")
                if not path_original.exists()
                else None
            )
            (
                plt.imsave(path_adversarial, adversarial, format="png")
                if not path_adversarial.exists()
                else None
            )

    # ------ [ bottom level helpers ] ------
    def _test_model(self, model):
        """[NOTE] 1 model vs all epsilons"""
        accuracies, examples = zip(
            *[self._test_model_epsilon_pair(model, eps) for eps in self.epsilons]
        )
        return accuracies, examples

    def _test_model_epsilon_pair(self, model, epsilon):
        """[NOTE] 1 model vs 1 epsilon"""
        correct, adv_example = 0, None

        if self.attack_name == "FGSM":
            attack = torchattacks.FGSM(model, eps=epsilon)
        elif self.attack_name == "PGD":
            attack = torchattacks.PGD(model, eps=epsilon, alpha=epsilon / 4, steps=20)
        elif self.attack_name == "SPGD":
            attack = torchattacks.PGD(model, eps=epsilon, alpha=epsilon / 4, steps=20)
        elif self.attack_name == "SPSA":
            attack = torchattacks.SPSA(
                model,
                eps=epsilon,
                delta=0.01,
                lr=0.01,
                nb_iter=int(40 * epsilon),
                nb_sample=128,
                max_batch_size=16,
            )
        elif self.attack_name == "OnePixel":
            attack = torchattacks.OnePixel(
                model,
                pixels=max(1, int(epsilon)),
                steps=int(2 * epsilon),
                popsize=int(epsilon),
                inf_batch=128,
            )
        elif self.attack_name == "JSMA":
            attack = torchattacks.JSMA(model, theta=epsilon, gamma=0.1)
        elif self.attack_name == "Jitter":
            attack = torchattacks.Jitter(
                model, eps=epsilon, alpha=epsilon / 4, steps=20, scale=10
            )
        elif self.attack_name == "GaussianNoise":
            attack = AttackGaussianNoise(std=epsilon)
        else:
            raise NotImplementedError

        for batch, target in self.data_loader:
            batch, target = batch.to(self.device), target.to(self.device)

            # -------- attack images --------
            batch_attacked = attack_batch(batch, target, self.device, attack)

            # -------- predictions after the attack --------
            adv_outputs = model(batch_attacked)

            # -------- calculate top-n accuracy --------
            correct += top_n_accuracy(adv_outputs, target, self.accuracy_top_n)

            # -------- save some pairs for showing --------
            if not adv_example:
                scale_to_0_1 = lambda x: (x - x.min()) / (x.max() - x.min())
                batch_0_1, batch_attacked_0_1 = scale_to_0_1(batch), scale_to_0_1(
                    batch_attacked
                )
                adv_example = (
                    batch_0_1[0].detach().cpu().numpy(),
                    batch_attacked_0_1[0].detach().cpu().numpy(),
                )
                adv_examples = {"epsilon": epsilon, "examples": adv_example}

        final_acc = correct / float(len(self.data_loader.dataset))
        print(
            f"{self.attack_name} Epsilon: {epsilon.__round__(3)} | Test Accuracy = {correct} / {len(self.data_loader.dataset)} = {final_acc.__round__(3)}"
        )
        return (final_acc, adv_examples)


def plot_attack(
    run_loader,
    dataset,
    which_attack: str,
    batch_size=4,
    accuracy_top_n=1,
    save_to=Path("_attack.png"),
    seed=42,
):
    """[NOTE] A wrapper that takes various parameters and runs the attack."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    attach = Attack(
        run_loader=run_loader,
        attack=which_attack,
        data_loader=dataloader,
        accuracy_top_n=accuracy_top_n,
        save_to=save_to,
    )

    attach.set_model(model=run_loader.model, model_name="Current Model")
    attach.test_save_data()


if __name__ == "__main__":
    dataset = get_dataset_obj("cifar10", "TEST")
    dataset = create_random_subset(dataset, 100)
    plot_attack(
        run_loader=RunLoader("__local__/experiment/000011"),
        dataset=dataset,
        which_attack="GaussianNoise",
        accuracy_top_n=1,
        save_to=Path("__local__/tests"),
        seed=42,
    )
