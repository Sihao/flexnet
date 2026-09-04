#!/bin/bash
# ============================================================================
# Rockefeller HPC: One-time setup for Flexible-Neurons project
# Run this interactively on the login node after cloning the repo
# ============================================================================

set -e

echo "=== Flexible-Neurons HPC Setup ==="

# --- 1. Create project directory structure ---
PROJECT_DIR="$HOME/Flexible-Neurons"
DATA_DIR="$HOME/scratch/datasets"
OUTPUT_DIR="$HOME/scratch/flex_outputs"

mkdir -p "$HOME/scratch"
mkdir -p "$DATA_DIR"
mkdir -p "$OUTPUT_DIR"

echo "Project dir: $PROJECT_DIR"
echo "Data dir:    $DATA_DIR"
echo "Output dir:  $OUTPUT_DIR"

# --- 2. Clone repo if not present ---
if [ ! -d "$PROJECT_DIR" ]; then
    echo "Cloning repository..."
    git clone https://github.com/Sihao/flexnet.git "$PROJECT_DIR"
else
    echo "Repository already exists at $PROJECT_DIR"
    echo "Run 'cd $PROJECT_DIR && git pull' to update"
fi

# --- 3. Create conda environment ---
echo ""
echo "=== Setting up conda environment ==="
if conda env list | grep -q "flexnet"; then
    echo "Conda env 'flexnet' already exists"
else
    echo "Creating conda environment 'flexnet'..."
    conda create -n flexnet python=3.9 -y
fi

echo ""
echo "Activating environment and installing dependencies..."
eval "$(conda shell.bash hook)"
conda activate flexnet

# Install PyTorch with CUDA support (CUDA 12.x for L40S/A100)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install project dependencies
if [ -f "$PROJECT_DIR/requirements.txt" ]; then
    pip install -r "$PROJECT_DIR/requirements.txt"
fi

# Additional packages needed for revision experiments
pip install autoattack scipy scikit-learn pandas matplotlib pyyaml rich natsort

echo ""
echo "=== Verifying GPU access ==="
python -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"

# --- 4. Create HPC-specific configurations.yml ---
cat > "$PROJECT_DIR/configurations_hpc.yml" << 'HPCEOF'
# -------- system (Rockefeller HPC) --------
system:
  experiment_dir:
    server: "${OUTPUT_DIR}"
    local: "__local__"
  imagenet_full_dir:
    server: "${DATA_DIR}/imagenet_full"
    local: "${DATA_DIR}/imagenet_full"

# -------- meta --------
network:
  values: [VGG]
  depends_on: null

vgg_variant:
  values: ["16"]
  depends_on: null

learning_rate:
  values: [0.0001]
  depends_on: null

weight_decay:
  values: [0.0001]
  depends_on: null

batch_size:
  values: [32]
  depends_on: null

validate_every_n_batch:
  values: [100]
  depends_on: null

dataset:
  values: [imagenet100]
  depends_on: null

optimizer:
  values: [ADAMW]
  depends_on: null

scheduler:
  values: [CosineAnnealingLR]
  depends_on: null

use_ffcv:
  values: [false]
  depends_on: { dataset: [imagenet100] }

use_flex:
  values: [false]
  depends_on: null
HPCEOF

# Substitute env vars into the config
sed -i "s|\${OUTPUT_DIR}|${OUTPUT_DIR}|g" "$PROJECT_DIR/configurations_hpc.yml"
sed -i "s|\${DATA_DIR}|${DATA_DIR}|g" "$PROJECT_DIR/configurations_hpc.yml"

echo ""
echo "=== Setup complete ==="
echo ""
echo "Next steps:"
echo "  1. Transfer ImageNet-100 dataset to: $DATA_DIR/imagenet_full/"
echo "     From your local machine: scp -r /path/to/imagenet100 rockefeller-hpc:$DATA_DIR/"
echo "  2. Copy HPC config: cp $PROJECT_DIR/configurations_hpc.yml $PROJECT_DIR/configurations.yml"
echo "  3. Submit a test job: sbatch $PROJECT_DIR/scripts/hpc/train_single_gpu.sh"
echo ""
