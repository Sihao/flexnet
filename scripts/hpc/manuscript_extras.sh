#!/bin/bash
# ============================================================================
# Generate the two manuscript figures that need a live model forward pass and
# are NOT in the standard iso-analysis pipeline: Fig 2E (feature maps) and
# Fig 4C (input loss-surface projection), for one restored checkpoint dir.
#
# Usage:
#   sbatch --partition=hpc_a10_a --gpus=a10:1 \
#          scripts/hpc/manuscript_extras.sh <run_dir_relative_to_project> [class_index]
#
# Reads exactly one ImageNet val image (no ImageFolder full-tree scan), so it is
# light on Lustre; outputs land under <run_dir>/results/manuscript_extras/.
# Idempotent: analyze_manuscript_extras_hpc.py skips if the .done marker exists.
# ============================================================================
#SBATCH --job-name=flexextras
#SBATCH --partition=hpc_a10_a
#SBATCH --gpus=a10:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

set -u
PROJECT_DIR="$HOME/Flexible-Neurons"

RUN_DIR="${1:-}"
CLASS_INDEX="${2:-207}"
if [ -z "$RUN_DIR" ]; then
    echo "ERROR: run_dir argument required" >&2
    echo "Usage: sbatch manuscript_extras.sh <run_dir> [class_index]" >&2
    exit 1
fi

echo "============================================"
echo "Job ID:      ${SLURM_JOB_ID:-<none>}"
echo "Node:        ${SLURM_NODELIST:-<none>}"
echo "Run dir:     $RUN_DIR"
echo "Class index: $CLASS_INDEX"
echo "Start:       $(date)"
echo "============================================"

set +u
source /lustre/fs8/huds_lab/store/slu/anaconda3/etc/profile.d/conda.sh
conda activate flexnet
source /lustre/fs4/ruit/store/ruitsoft/soft/ruit-cudas/switch-cuda.sh 12.4 2>/dev/null || true
set -u

cd "$PROJECT_DIR" || { echo "ERROR: cannot cd to $PROJECT_DIR" >&2; exit 1; }
mkdir -p slurm_logs

if [ ! -f "$RUN_DIR/configurations.json" ]; then
    echo "ERROR: $RUN_DIR/configurations.json not found" >&2
    exit 1
fi

python scripts/analyze_manuscript_extras_hpc.py \
    --run-dir "$RUN_DIR" --class-index "$CLASS_INDEX" \
    --grid-points 51 --range-scale 1.0 --device cuda
rc=$?
echo "End: $(date)  rc=$rc"
exit $rc
