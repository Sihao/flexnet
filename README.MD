# Input-dependent flexible computation as an inductive bias for learning robust feature representations

**Sihao Lu¹, Dong Yin¹, Linnea Evanson¹, Maksim Lavrov¹, Iakov Kharitonov¹, Andriy S. Kozlov¹**

¹ Department of Bioengineering, Imperial College London

---

This repository contains the implementation for **Flexible Neurons**, exploring input-dependent flexible computation as an inductive bias to enhance the robustness of learned feature representations.

## Abstract
Object recognition requires balancing selectivity and invariance, a feat achieved in both biological and artificial systems through AND-like and OR-like operations. While biological neurons are capable of intrinsically switching between these operations, artificial units are typically hard-wired for a single computation. We address this discrepancy by introducing a deep learning layer where input strength flexibly governs the choice between selectivity and invariance. This flexible mechanism serves as an inductive bias predisposing the network towards low-frequency encoding, and a more smooth input loss landscape. We show that these properties confer robustness against algorithmic and adversarial corruptions. Furthermore, this approach enhances alignment with biological representations, offering a principled, biomimetic strategy for defensive optimization.

## Installation

```bash
pip install -r requirements.txt
```

## Experiments

This repository includes pre-trained weights for two primary experimental setups:

*   **Experiment 2 (Standard VGG)**: A baseline VGG-16 model trained on ImageNet-100.
    *   Path: `__local__/experiment-2/000000/checkpoints/checkpoint_200.pth`
*   **Experiment 4 (FlexNet)**: A VGG-16 model augmented with Flexible Neuron layers using `CHANNELWISE_MAXPOOL` joint mechanism.
    *   Path: `__local__/experiment-4/000000/checkpoints/checkpoint_200.pth`

## Training Instructions

To train a model or reproduce the experiments, follow these steps:

### 1. Configuration
Modify `configurations.yml` to set your desired hyperparameters, model architecture, and dataset paths.

### 2. Initialize Experiment
Run the initialization script to generate the directory structure and specific configuration files for the experiment grid:
```bash
conda run -n flex-ffcv python step_1_init_experiments.py
```
This will create a new experiment folder in the location specified in `configurations.yml` (default is `__local__`).

### 3. Run Training
Execute the training script. By default, it will attempt to train/continue all runs within the latest experiment directory:
```bash
conda run -n flex-ffcv python step_2_train_or_continue.py
```
You can also specify a particular experiment or run:
```bash
conda run -n flex-ffcv python step_2_train_or_continue.py --experiment_name experiment-4 --run_name 000000
```

## Acknowledgements

This work was conducted at the [Kozlov Lab](https://www.kozlovlab.com/).
Final code by Sihao Lu and Dong Yin, based on earlier implementations by Linnea Evanson, Maksim Lavrov, Iakov Kharitonov.
