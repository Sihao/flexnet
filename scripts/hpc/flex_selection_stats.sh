#!/bin/bash
# ============================================================================
# Rich conv-vs-max selection statistics for a trained flex network over
# ImageNet val (see scripts/analyze_flex_selection_stats_hpc.py). GPU forward
# pass -> MUST run via sbatch on a compute node, never on the login node.
#
# Usage:
#   sbatch scripts/hpc/flex_selection_stats.sh <run_dir> <exp_name> [num_images]
#   sbatch scripts/hpc/flex_selection_stats.sh \
#       /lustre/fs8/huds_lab/scratch/slu/iso_analysis/vgg16-flex-e89/000000 \
#       vgg16-flex-e89 0            # 0 = FULL 50k val
#
# Output: results/flex_selection_stats/<exp_name>.{npz,json}
# ============================================================================
#SBATCH --job-name=flexsel
#SBATCH --partition=hpc_a10_a
#SBATCH --gpus=a10:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

set -u
PROJECT_DIR="$HOME/Flexible-Neurons"

RUN_DIR="${1:-}"
EXP_NAME="${2:-}"
NUM_IMAGES="${3:-0}"
if [ -z "$RUN_DIR" ] || [ -z "$EXP_NAME" ]; then
    echo "Usage: sbatch flex_selection_stats.sh <run_dir> <exp_name> [num_images]" >&2
    exit 1
fi

echo "============================================"
echo "Job ID:      ${SLURM_JOB_ID:-<none>}   Node: ${SLURM_NODELIST:-<none>}"
echo "Run dir:     $RUN_DIR"
echo "Exp name:    $EXP_NAME   num_images: $NUM_IMAGES (0 = full val)"
echo "Start:       $(date)"
echo "============================================"

set +u
source /lustre/fs8/huds_lab/store/slu/anaconda3/etc/profile.d/conda.sh
conda activate flexnet
source /lustre/fs4/ruit/store/ruitsoft/soft/ruit-cudas/switch-cuda.sh 12.4 2>/dev/null || true
set -u

cd "$PROJECT_DIR" || { echo "ERROR: cannot cd to $PROJECT_DIR" >&2; exit 1; }
mkdir -p slurm_logs results/flex_selection_stats

if [ ! -f "$RUN_DIR/configurations.json" ]; then
    echo "ERROR: $RUN_DIR/configurations.json not found" >&2
    exit 1
fi

python scripts/analyze_flex_selection_stats_hpc.py \
    --run-dir "$RUN_DIR" --exp-name "$EXP_NAME" \
    --num-images "$NUM_IMAGES" --batch-size 128 --num-workers 8 --device cuda \
    --out-npz "results/flex_selection_stats/${EXP_NAME}.npz" \
    --out-json "results/flex_selection_stats/${EXP_NAME}.json"
rc=$?
echo "End: $(date)  rc=$rc"
exit $rc
