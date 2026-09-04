# Input-dependent flexible computation as an inductive bias for learning robust feature representations

**Sihao Lu¹, Dong Yin¹, Linnea Evanson¹, Maksim Lavrov¹, Iakov Kharitonov¹, Andriy S. Kozlov¹**

¹ Department of Bioengineering, Imperial College London

---

This repository contains the implementation for **Flexible Neurons**, the pre-trained
weights of the matched-accuracy checkpoint pair used in every analysis of the manuscript,
the analysis outputs those checkpoints produced, and the scripts that render the
manuscript figures from them.

## Abstract

Object recognition requires balancing selectivity and invariance, a feat achieved in both
biological and artificial systems through AND-like and OR-like operations. While biological
neurons are capable of intrinsically switching between these operations, artificial units
are typically hard-wired for a single computation. We address this discrepancy by
introducing a deep learning layer where input strength flexibly governs the choice between
selectivity and invariance. This flexible mechanism serves as an inductive bias
predisposing the network towards low-frequency encoding, and a more smooth input loss
landscape. We show that these properties confer robustness against algorithmic and
adversarial corruptions. Furthermore, this approach enhances alignment with biological
representations, offering a principled, biomimetic strategy for defensive optimization.

## Repository layout

```
src/                      FlexLayer implementation, model builders, training + analysis code
scripts/                  analysis entry points and the figure-render pipeline
scripts/hpc/              SLURM wrappers used on our cluster (site-specific paths; see below)
configs/                  training configuration (grid template + the two exact run configs)
data/                     analysis outputs for the manuscript's checkpoint pair (see below)
weights/                  pre-trained checkpoints of the matched-accuracy pair (Git LFS)
figures/                  reference renders of manuscript Figures 1-6 (png + pdf)
step_1_init_experiments.py, step_2_train_or_continue.py, cli_tool.py   training entry points
```

## Reproduce the manuscript figures (no GPU needed)

All figure inputs ship in `data/`, so the figures render on any machine from a plain
Python environment — no GPU, no torch, no dataset download:

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install numpy matplotlib scipy pillow
scripts/render_figures.sh
```

This renders Figures 1–6 for the matched-accuracy pair into the repository root as
`manuscript_styled_fig<N>_ISO0.60_flexvggE89_vs_vanillavggE70.{png,svg,pdf}`. The output
is deterministic: it should match `figures/` byte-for-byte. `ONLY=3,4` renders a subset;
`PYTHON=/path/to/python` selects the interpreter.

## The matched-accuracy pair

Every analysis in the manuscript compares the two architectures at equal clean accuracy.
Both networks were trained on ImageNet-1k with identical schedules; checkpoints were then
selected so that top-1 validation accuracy matches:

| model                    | checkpoint             | epoch | ImageNet-1k top-1 |
|--------------------------|------------------------|-------|-------------------|
| Flexible VGG16           | `weights/vgg16-flex-e89.pth`    | 89    | 0.5995 |
| Baseline (vanilla) VGG16 | `weights/vgg16-vanilla-e70.pth` | 70    | 0.6003 |

The selection procedure (`scripts/select_iso_accuracy_checkpoints.py`) and its frozen
result (`data/iso_accuracy_selection.frozen.json`, 12 accuracy levels) are included;
`scripts/iso_pairs_from_selection.py` is the single source of truth every driver reads.
`data/` currently ships the analysis outputs of the highest-accuracy pair — the pair
every figure in the manuscript uses.

### Weights

The checkpoints are stored with Git LFS (`git lfs pull` after cloning; ~1.7 GB each).
Verify integrity with `sha256sum -c weights/CHECKSUMS.sha256`. Each file is a full
training checkpoint:

```python
ckpt = torch.load("weights/vgg16-flex-e89.pth", map_location="cpu")
ckpt.keys()  # model_state_dict, optimizer_state_dict, epoch, loss, scheduler_state_dict
```

## Recompute the analyses (GPU needed)

The numbers under `data/` were produced by the pipeline in `scripts/`. To recompute them,
assemble a run directory from the shipped weights and config, then run the analysis entry
points:

```bash
mkdir -p run-flex/checkpoints
cp weights/vgg16-flex-e89.pth run-flex/checkpoints/checkpoint_89.pth
cp configs/experiment-flex-vgg16-imagenet-dense.json run-flex/configurations.json

# frequency / hessian / adversarial attacks (Figs 2-4)
IMAGENET_C_PATH=/path/to/imagenet-c \
python scripts/analyze_checkpoint_hpc.py --run-dir run-flex --analyses attacks frequency hessian perturbation

# block-1 spectral statistics over 512 training images (Fig 2C-E)
python scripts/analyze_block1_spectra_hpc.py --run-dir run-flex \
  --train-dir /path/to/imagenet/train \
  --probe-image data/iso_analysis_staged/probe/input_29.JPEG

# loss-surface probe selection + surfaces (Fig 4C)
python scripts/analyze_fig4c_best_pair_hpc.py --help

# routing statistics over the full validation set (Fig 6; flexible checkpoint only)
python scripts/analyze_flex_selection_stats_hpc.py --run-dir run-flex \
  --out-npz vgg16-flex-e89.npz --out-json vgg16-flex-e89.json

# Brain-Score benchmarks (Fig 5)
python scripts/brain_score_checkpoint.py --help
python scripts/assemble_7L_brainscore.py --help
```

Dataset locations are taken from the environment (`IMAGENET_LOCAL_DIR`, `IMAGENET_C_PATH`)
or from explicit flags. The wrappers in `scripts/hpc/` are the SLURM jobs we ran on our
cluster; their partition names, conda paths, and staging paths are site-specific — treat
them as a record of the exact settings (walltimes, batch sizes, GPU types), not as portable
executables.

## Train from scratch

```bash
pip install -r requirements.txt
python step_1_init_experiments.py     # builds experiment dirs under __local__/ from configs/configurations.yml
python step_2_train_or_continue.py --experiment_name <name> --run_name 000000
```

The exact configurations of the two manuscript runs are
`configs/experiment-flex-vgg16-imagenet-dense.json` and
`configs/experiment-vanilla-vgg16-imagenet-dense.json` (VGG16, ImageNet-1k, matched
optimizer and cosine schedule; the flexible network replaces all 13 convolutions with
flexible layers using the `CHANNELWISE_MAXPOOL` joint mechanism). Per-epoch training
logs of both runs, including the full 50,000-image validation evaluations, are under
`data/logs/`.

## Acknowledgements

This work was conducted at the [Kozlov Lab](https://www.kozlovlab.com/).
Final code by Sihao Lu and Dong Yin, based on earlier implementations by Linnea Evanson,
Maksim Lavrov, Iakov Kharitonov.
