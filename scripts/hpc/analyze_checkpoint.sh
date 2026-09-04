#!/bin/bash
# ============================================================================
# Run the GPU/ImageNet downstream analyses (attacks/frequency/hessian) on one
# restored checkpoint dir, for the iso-accuracy trajectory comparison.
#
# Usage:
#   sbatch scripts/hpc/analyze_checkpoint.sh <run_dir_relative_to_project> [analyses...]
#   sbatch scripts/hpc/analyze_checkpoint.sh __local__/iso_analysis/flex-e40/000000
#   sbatch scripts/hpc/analyze_checkpoint.sh __local__/iso_analysis/flex-e40/000000 frequency hessian
#
# Idempotent: analyze_checkpoint_hpc.py skips analyses whose marker exists.
# ============================================================================
#SBATCH --job-name=flexanalyze
#SBATCH --partition=hpc_l40s_b
#SBATCH --gpus=l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

set -u
PROJECT_DIR="$HOME/Flexible-Neurons"

RUN_DIR="${1:-}"
if [ -z "$RUN_DIR" ]; then
    echo "ERROR: run_dir argument required" >&2
    echo "Usage: sbatch analyze_checkpoint.sh <run_dir> [analyses...]" >&2
    exit 1
fi
shift || true
ANALYSES=("$@")   # may be empty -> runner uses its default (all)

echo "============================================"
echo "Job ID:     ${SLURM_JOB_ID:-<none>}"
echo "Node:       ${SLURM_NODELIST:-<none>}"
echo "Run dir:    $RUN_DIR"
echo "Analyses:   ${ANALYSES[*]:-<all>}"
echo "Start:      $(date)"
echo "============================================"

# conda's activate scripts reference unbound vars ($PS1, $_CE_CONDA, ...); with
# `set -u` (nounset) active those abort the shell and kill the job in ~1s before
# any analysis runs. Relax nounset around the conda/CUDA block, then restore it.
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

# --- Stage ImageNet val to node-local disk (bypass post-cutover fs8 Lustre-client wedge) ---
# attacks/frequency/hessian read only the 1000-class val set. The fs8 client wedges the
# PyTorch DataLoader read pattern, so stage val off-Lustre to node-local /tmp (XFS) once per
# node and export IMAGENET_LOCAL_DIR (honoured by the imagenet branch of dataset_select).
# Reuse a training full-stage if this job landed on a node that already has one.
STAGE_SRC="/lustre/fs8/huds_lab/scratch/slu/Data/imagenet_full"
if [ -f "/tmp/imagenet_full/.stage_ok" ] && [ -d "/tmp/imagenet_full/val" ]; then
    export IMAGENET_LOCAL_DIR="/tmp/imagenet_full"
    echo "[stage] reusing full training stage at /tmp/imagenet_full on $(hostname -s)"
else
    STAGE_DST="/tmp/imagenet_val_stage"
    STAGE_OK="${STAGE_DST}/.stage_ok"
    STAGE_LOCK="/tmp/imagenet_val_stage.staging.lock"
    if [ -f "$STAGE_OK" ]; then
        echo "[stage] val already staged at $STAGE_DST on $(hostname -s)"
    elif mkdir "$STAGE_LOCK" 2>/dev/null; then
        echo "[stage] staging ImageNet val -> $STAGE_DST on $(hostname -s) (48-way) ..."
        rm -rf "$STAGE_DST"; mkdir -p "$STAGE_DST/val"
        t0=$(date +%s)
        ls "$STAGE_SRC/val" | xargs -P 48 -I{} cp -r "$STAGE_SRC/val/{}" "$STAGE_DST/val/"
        t1=$(date +%s)
        SRC_N=$(find "$STAGE_SRC/val" -type f 2>/dev/null | wc -l)
        DST_N=$(find "$STAGE_DST/val" -type f 2>/dev/null | wc -l)
        echo "[stage] copied val in $((t1-t0))s ; src_files=$SRC_N dst_files=$DST_N"
        if [ "$SRC_N" -gt 0 ] && [ "$DST_N" -eq "$SRC_N" ]; then
            touch "$STAGE_OK"; echo "[stage] OK: val file counts match"
        else
            echo "[stage] ERROR: count mismatch (src=$SRC_N dst=$DST_N) -> discard, fallback to fs8"; rm -rf "$STAGE_DST"
        fi
        rmdir "$STAGE_LOCK" 2>/dev/null || true
    else
        echo "[stage] another job is staging val on this node; waiting up to 15 min ..."
        for i in $(seq 1 60); do [ -f "$STAGE_OK" ] && break; sleep 15; done
    fi
    if [ -f "$STAGE_OK" ]; then
        export IMAGENET_LOCAL_DIR="$STAGE_DST"
        echo "[stage] analyses read node-local IMAGENET_LOCAL_DIR=$IMAGENET_LOCAL_DIR"
    else
        echo "[stage] WARNING: val staging unavailable; falling back to fs8 dataset path (may wedge)"
    fi
fi

if [ "${#ANALYSES[@]}" -gt 0 ]; then
    python scripts/analyze_checkpoint_hpc.py --run-dir "$RUN_DIR" --device cuda --analyses "${ANALYSES[@]}"
else
    python scripts/analyze_checkpoint_hpc.py --run-dir "$RUN_DIR" --device cuda
fi
RC=$?
echo "Analyses exited with code: $RC"
echo "End: $(date)"
exit $RC
