#!/Users/donyin/miniconda3/envs/imperial/bin/python -m pdb

"""
take a torchattack attack and visualize the attacked images to check
"""

import torch
from torch.utils.data import DataLoader
from pathlib import Path
from src.utils.device import select_device
from src.analysis.run_loader_image.attacks import attack_batch, save_image_batch


def visual_check_attack_images(model, dataset, attack, save_dir):
    """
    Visual check of attack images.

    Parameters:
    model (torch.nn.Module): The model to attack.
    dataset (torch.utils.data.Dataset): The dataset to use.
    attack (torchattacks.attack): The attack instance (e.g., torchattacks.SPSA(model, eps=epsilon)).
    save_to (Path): The directory to save the attacked images.
    """
    device = select_device()
    model.to(device).eval()
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    save_dir.mkdir(parents=True, exist_ok=True)

    for batch, target in dataloader:
        batch, target = batch.to(device), target.to(device)
        batch_attacked = attack_batch(batch, target, device, attack)

        # concate original and attacked images
        concated = torch.cat((batch, batch_attacked), 0)
        save_image_batch(concated, save_dir / "concated_images.png")

        break  # stops at one plot


if __name__ == "__main__":
    import torchattacks
    from src.training.dataset_select import get_dataset_obj
    from src.analysis.run_loader import RunLoader
    from src.training.dataset_subset import create_random_subset

    dataset = get_dataset_obj("cifar10", "TEST")
    dataset = create_random_subset(dataset, 100)
    run_loader = RunLoader(
        run_folder=Path(
            "/Users/donyin/Library/CloudStorage/Dropbox/root-dir/flex/layer/__local__/experiment/000011"
        ),
        whether_load_checkpoint=False,
        whether_instantiate_model=True,
    )
    model = run_loader.model

    # ==== example ====
    # Jitter(model, eps=0.03137254901960784, alpha=0.00784313725490196, steps=10, scale=10, std=0.1, random_start=True)
    # [add: white box]
    eps = 0.5
    alpha = eps / 4
    attack = torchattacks.Jitter(model, eps=eps, alpha=alpha, steps=20, scale=10)

    visual_check_attack_images(
        model=model,
        dataset=dataset,
        attack=attack,
        save_dir=Path("__visual_check_spsa__"),
    )
