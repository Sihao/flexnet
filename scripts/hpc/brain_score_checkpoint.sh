#!/bin/bash
# ============================================================================
# Brain-Score one restored checkpoint on the HPC, as a SLURM job, using the
# dedicated py3.11 `brainscore` env + benchmark data in scratch. Submitting one
# of these per matched checkpoint gives ~18x parallelism vs the serial local
# CPU run.
#
# Usage:
#   sbatch scripts/hpc/brain_score_checkpoint.sh <run_dir> <exp_name> [layers...]
#   sbatch scripts/hpc/brain_score_checkpoint.sh __local__/iso_analysis/flex-e40/000000 flex-e40
#
# brain_score_checkpoint.py is idempotent (skips already-scored layer/benchmark).
# ============================================================================
#SBATCH --job-name=flexbs
#SBATCH --partition=hpc_a10_a
#SBATCH --gpus=a10:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

set -u
PROJECT_DIR="$HOME/Flexible-Neurons"
SCRATCH="/lustre/fs8/huds_lab/scratch/slu/brainscore_hpc"
BS_ENV="$SCRATCH/env"
export FLEX_DATA_ROOT="$SCRATCH/brain_score_data"

# --- result_caching per-job isolation (fixes FreemanZiemba V1/V2 scoring 0) ---
# brainscore's load_assembly()/_place_on_screen()/ceilings are wrapped in
# result_caching's @store, which writes "<key>.filepart" then os.rename()s it to
# "<key>.pkl". The FreemanZiemba load_assembly cache key is (region, access) ONLY
# -- identical across every checkpoint AND every layer -- so concurrent sweep jobs
# sharing the default ~/.result_caching all target the SAME .filepart: the first
# job's rename wins, every other job's rename dies with [Errno 2] No such file,
# the benchmark raises, and process_layer records 0.0. (MajajHong V4/IT mostly won
# their races, which is why only V1/V2 collapsed to 0.) Point each SLURM job at a
# private, node-local cache dir so no two jobs ever share a .filepart. Within-job
# cross-layer caching is preserved, and the many tiny cache files stay off Lustre.
export RESULTCACHING_HOME="/tmp/result_caching_${SLURM_JOB_ID:-$$}"
mkdir -p "$RESULTCACHING_HOME"

RUN_DIR="${1:-}"
EXP_NAME="${2:-}"
if [ -z "$RUN_DIR" ] || [ -z "$EXP_NAME" ]; then
    echo "Usage: sbatch brain_score_checkpoint.sh <run_dir> <exp_name> [layers...]" >&2
    exit 1
fi
shift 2 || true
LAYERS=("$@"); [ "${#LAYERS[@]}" -eq 0 ] && LAYERS=("auto")

# Default to all four standard brain-score benchmarks: FreemanZiemba V1/V2 (public)
# + MajajHong V4/IT. V1/V2 used to score 0 via a result_caching cross-job race on
# the shared ~/.result_caching filepart (fixed above by the per-job
# RESULTCACHING_HOME). Override the set with e.g.
#   BENCHMARKS="FreemanZiemba2013.V1.public-pls FreemanZiemba2013.V2.public-pls" sbatch ...
read -r -a BENCHMARKS <<< "${BENCHMARKS:-FreemanZiemba2013.V1.public-pls FreemanZiemba2013.V2.public-pls MajajHong2015.public.V4-pls MajajHong2015.public.IT-pls}"

echo "============================================"
echo "Job ID:   ${SLURM_JOB_ID:-<none>}   Node: ${SLURM_NODELIST:-<none>}"
echo "Run dir:  $RUN_DIR"
echo "Exp name: $EXP_NAME   Layers: ${LAYERS[*]}"
echo "Data:     $FLEX_DATA_ROOT"
echo "Start:    $(date)"
echo "============================================"

cd "$PROJECT_DIR" || { echo "ERROR: cannot cd to $PROJECT_DIR" >&2; exit 1; }
mkdir -p slurm_logs

CFG="$RUN_DIR/configurations.json"
CKPT=$(ls -1 "$RUN_DIR"/checkpoints/checkpoint_*.pth 2>/dev/null | head -1)
if [ ! -f "$CFG" ] || [ -z "$CKPT" ]; then
    echo "ERROR: missing config ($CFG) or checkpoint in $RUN_DIR/checkpoints" >&2
    exit 1
fi

OUT="$SCRATCH/results/${EXP_NAME}.json"
mkdir -p "$(dirname "$OUT")"

source /lustre/fs8/huds_lab/store/slu/anaconda3/etc/profile.d/conda.sh
conda activate "$BS_ENV"

python scripts/brain_score_checkpoint.py \
    --ckpt "$CKPT" --config "$CFG" --exp-name "$EXP_NAME" \
    --layers "${LAYERS[@]}" --benchmarks "${BENCHMARKS[@]}" --output "$OUT"
RC=$?
echo "brain_score exited with code: $RC"
echo "End: $(date)"
exit $RC
