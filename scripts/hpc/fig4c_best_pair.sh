#!/usr/bin/env bash
# ============================================================================
# Draft Fig4C principled probe selection: pool gate-passing validation images,
# rank by baseline/flex top-2 input-Hessian curvature ratio, compute the 51x51
# surfaces for the top-2 images only.
#
# Usage:
#   sbatch scripts/hpc/fig4c_best_pair.sh <flex_run_dir> <vanilla_run_dir>
#
# Writes results/manuscript_extras_best/rank{1,2}/ + summary.json in each run
# dir. Never touches manuscript_extras or manuscript_extras_img2.
# ============================================================================
#SBATCH --job-name=fig4c-best2
#SBATCH --partition=hpc_v100_b
#SBATCH --gpus=v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

set -u
PROJECT_DIR="$HOME/Flexible-Neurons"

FLEX_RUN="${1:-}"
VAN_RUN="${2:-}"
if [ -z "$FLEX_RUN" ] || [ -z "$VAN_RUN" ]; then
    echo "ERROR: flex_run_dir and vanilla_run_dir arguments required" >&2
    exit 1
fi

echo "============================================"
echo "Job ID:     ${SLURM_JOB_ID:-<none>}"
echo "Node:       ${SLURM_NODELIST:-<none>}"
echo "Flex run:   $FLEX_RUN"
echo "Vanilla:    $VAN_RUN"
echo "Start:      $(date)"
echo "============================================"

set +u
source /lustre/fs8/huds_lab/store/slu/anaconda3/etc/profile.d/conda.sh
conda activate flexnet
source /lustre/fs4/ruit/store/ruitsoft/soft/ruit-cudas/switch-cuda.sh 12.4 2>/dev/null || true
set -u

cd "$PROJECT_DIR" || { echo "ERROR: cannot cd to $PROJECT_DIR" >&2; exit 1; }
mkdir -p slurm_logs

for RUN in "$FLEX_RUN" "$VAN_RUN"; do
    if [ ! -f "$RUN/configurations.json" ]; then
        echo "ERROR: $RUN/configurations.json not found" >&2
        exit 1
    fi
done

VAL_SRC="/lustre/fs8/huds_lab/scratch/slu/Data/imagenet_full/val"
N_CANDS="${N_CANDS:-48}"
PICK_SEED="${PICK_SEED:-2}"
POOL_SIZE="${POOL_SIZE:-16}"
STAGE="/tmp/fig4c_best2_stage_${SLURM_JOB_ID:-manual}"
mkdir -p "$STAGE/files/n01632777" "$STAGE/files/n03100240"
echo "[stage] listing class root ..."
ls "$VAL_SRC" > "$STAGE/classes.txt"
echo "[stage] $(wc -l < "$STAGE/classes.txt") classes; picking $N_CANDS candidates (16-way listing) ..."
python -u scripts/pick_train_files.py --train-root "$VAL_SRC" --classes "$STAGE/classes.txt" --n "$N_CANDS" --seed "$PICK_SEED" --workers 16 --out "$STAGE/cands.txt" || exit 1
echo "[stage] copying $N_CANDS files 16-way -> $STAGE/files ..."
(cd "$VAL_SRC" && xargs -a "$STAGE/cands.txt" -P 16 -I{} cp --parents "{}" "$STAGE/files/")
cp "$VAL_SRC/n01632777/ILSVRC2012_val_00034583.JPEG" "$STAGE/files/n01632777/"
cp "$VAL_SRC/n03100240/ILSVRC2012_val_00024103.JPEG" "$STAGE/files/n03100240/"
echo "[stage] copy done ($(find "$STAGE/files" -type f | wc -l) files)"

FORCE_FLAG=""
[ "${FORCE:-0}" = "1" ] && FORCE_FLAG="--force"
python -u scripts/analyze_fig4c_best_pair_hpc.py --flex-run-dir "$FLEX_RUN" --vanilla-run-dir "$VAN_RUN" --candidates "$STAGE/cands.txt" --files-root "$STAGE/files" --classes "$STAGE/classes.txt" --forced "$STAGE/files/n01632777/ILSVRC2012_val_00034583.JPEG:29" --forced "$STAGE/files/n03100240/ILSVRC2012_val_00024103.JPEG:511" --pool-size "$POOL_SIZE" --steps 50 --grid-points 51 --range-scale 10.0 --device cuda $FORCE_FLAG
RC=$?
rm -rf "$STAGE"
echo "fig4c best-pair exited with code: $RC"
echo "End: $(date)"
exit $RC
