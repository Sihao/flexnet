#!/Users/donyin/miniconda3/envs/imperial/bin/python

"""
activation sanity checks
- place stochastic rounding in a simple conv net
- compare how quick it converges with ReLU / other methods
"""


import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from src.utils.general import apply_kaiming_initialization
from src.training.dataset_select import get_dataset_obj
from src.training.dataset_subset import create_random_subset
from src.utils.device import select_device

from src.modules.masks.relu import ReLU
from src.modules.masks.stochastic import StochasticRound, StochasticRoundSigmoid
from src.modules.masks.ste import STEFunc, STEFuncSigmoid
from modules.models.VGG import VGG


def compare_configs_on_simple_conv(hyperparams, *configs):
    torch.manual_seed(42)

    num_epochs = hyperparams.get("num_epochs", 30)
    learning_rate = hyperparams.get("learning_rate", 5e-4)
    batch_size = hyperparams.get("batch_size", 8)

    criterion = nn.CrossEntropyLoss()
    device = select_device(priority=["cuda", "mps", "cpu"])
    print(f"Using device: {device}")

    dataset = get_dataset_obj("cifar10", "TRAIN")
    dataset = create_random_subset(dataset, num_samples=100, seed=42)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )

    nets = [SimpleConvNet(config).to(device) for config in configs]

    optimizer_name = hyperparams.get("optimizer", "Adam").lower()
    if optimizer_name == "adam":
        optimizers = [
            optim.Adam(net.parameters(), lr=learning_rate, weight_decay=1e-5)
            for net in nets
        ]
    elif optimizer_name == "sgd":
        optimizers = [
            optim.SGD(
                net.parameters(), lr=learning_rate, weight_decay=1e-5, momentum=0.9
            )
            for net in nets
        ]
    else:
        raise ValueError(f"Optimizer {hyperparams.get('optimizer')} not supported")

    losses = [[] for _ in configs]

    # training loop
    for epoch in range(num_epochs):
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)

            for idx, (net, optimizer) in enumerate(zip(nets, optimizers)):
                optimizer.zero_grad()
                outputs = net(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                losses[idx].append(loss.item())
                print(f"Epoch {epoch+1}, {configs[idx].__name__} Loss: {loss.item()}")

    # plotting
    plt.figure()
    for idx, func_losses in enumerate(losses):
        plt.plot(func_losses, label=configs[idx].__name__)

    plt.xlabel("Training Iteration")
    plt.ylabel("Loss")
    plt.title("Loss vs Training Iteration for Different Activation Functions")
    plt.legend()
    plt.savefig("_loss_compare.png")


if __name__ == "__main__":
    compare_activations_on_simple_conv(
        # STEFunc,
        # STEFuncSigmoid,
        StochasticRound,
        StochasticRoundSigmoid,
        # ReLU,
    )

    # # next - small scale test with flex layers
