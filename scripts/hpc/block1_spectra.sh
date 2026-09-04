#!/usr/bin/env bash
# ============================================================================
# Fig2 C/D draft-faithful spectra: 512 random TRAIN images through block1's
# last conv/flex layer -> per-channel radial power profiles + slopes.
#
# Usage:
#   sbatch [overrides] scripts/hpc/block1_spectra.sh <run_dir_relative_to_project>
#   e.g. sbatch -p hpc_a100_a --gpus=a100:1 scripts/hpc/block1_spectra.sh \
#        __local__/iso_analysis/vgg16-flex-e89/000000
#
# Reads train JPEGs directly (512 files + ~500 class listdirs, no tree scan,
# no staging): prefers a node-local training full-stage when present.
# Idempotent via results/block1_spectra/.done (FORCE=1 to recompute).
# ============================================================================
#SBATCH --job-name=vgg16spectra
#SBATCH --partition=hpc_a100_a
#SBATCH --gpus=a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

set -u
PROJECT_DIR="$HOME/Flexible-Neurons"

RUN="${1:-}"
if [ -z "$RUN" ]; then
    echo "ERROR: run_dir argument required" >&2
    exit 1
fi

echo "============================================"
echo "Job ID:     ${SLURM_JOB_ID:-<none>}"
echo "Node:       ${SLURM_NODELIST:-<none>}"
echo "Run dir:    $RUN"
echo "Start:      $(date)"
echo "============================================"

# conda's activate scripts reference unbound vars; relax nounset around them
set +u
source /lustre/fs8/huds_lab/store/slu/anaconda3/etc/profile.d/conda.sh
conda activate flexnet
source /lustre/fs4/ruit/store/ruitsoft/soft/ruit-cudas/switch-cuda.sh 12.4 2>/dev/null || true
set -u

cd "$PROJECT_DIR" || { echo "ERROR: cannot cd to $PROJECT_DIR" >&2; exit 1; }
mkdir -p slurm_logs

if [ ! -f "$RUN/configurations.json" ]; then
    echo "ERROR: $RUN/configurations.json not found" >&2
    exit 1
fi

# Serial in-process listing of ~400 class dirs WEDGED on the fs8 Lustre
# client (jobs 6070519/6070520: 40+ min stuck in cl_sync_io_wait, no forward
# pass ever ran). Stage instead: parallel listing -> parallel copy of only the
# selected files to node-local /tmp, then torch reads local disk exclusively.
TRAIN_SRC="/lustre/fs8/huds_lab/scratch/slu/Data/imagenet_full/train"
PROBE_SRC="/lustre/fs8/huds_lab/scratch/slu/Data/imagenet_full/val/n01632777/ILSVRC2012_val_00034583.JPEG"
if [ -f "/tmp/imagenet_full/.stage_ok" ] && [ -d "/tmp/imagenet_full/train" ]; then
    TRAIN_SRC="/tmp/imagenet_full/train"
    echo "[stage] reusing node-local train stage at $TRAIN_SRC"
fi

N_IMAGES="${N_IMAGES:-512}"
STAGE="/tmp/block1_spectra_stage_${SLURM_JOB_ID:-manual}"
mkdir -p "$STAGE/files"
echo "[stage] listing class root ..."
ls "$TRAIN_SRC" > "$STAGE/classes.txt"
echo "[stage] $(wc -l < "$STAGE/classes.txt") classes; picking $N_IMAGES files (16-way listing) ..."
python -u scripts/pick_train_files.py --train-root "$TRAIN_SRC" --classes "$STAGE/classes.txt" --n "$N_IMAGES" --seed 0 --workers 16 --out "$STAGE/filelist.txt" || exit 1
echo "[stage] copying $N_IMAGES files 16-way -> $STAGE/files ..."
t0=$(date +%s)
(cd "$TRAIN_SRC" && xargs -a "$STAGE/filelist.txt" -P 16 -I{} cp --parents "{}" "$STAGE/files/")
echo "[stage] copy done in $(( $(date +%s) - t0 ))s ($(find "$STAGE/files" -type f | wc -l) files)"
cp "$PROBE_SRC" "$STAGE/probe.JPEG" || echo "[stage] WARNING: probe copy failed; job continues without probe featmap"
PROBE_ARG=""
[ -f "$STAGE/probe.JPEG" ] && PROBE_ARG="--probe-image $STAGE/probe.JPEG"

FORCE_FLAG=""
[ "${FORCE:-0}" = "1" ] && FORCE_FLAG="--force"
python -u scripts/analyze_block1_spectra_hpc.py --run-dir "$RUN" --file-list "$STAGE/filelist.txt" --files-root "$STAGE/files" --num-images "$N_IMAGES" --device cuda $PROBE_ARG $FORCE_FLAG
RC=$?
rm -rf "$STAGE"
echo "block1 spectra exited with code: $RC"
echo "End: $(date)"
exit $RC
