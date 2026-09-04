#!/bin/bash
# ============================================================================
# Single-GPU training with auto-resubmission for 24h wall time limit
#
# Usage:
#   sbatch train_single_gpu.sh <experiment_name>
#   sbatch train_single_gpu.sh experiment-2
#
# The script will:
#   1. Pre-schedule a follow-up job (afterany dependency) so SLURM always
#      has a successor queued, even if this job is force-killed.
#   2. Run training for up to 23.5 hours (leaving 30min buffer for checkpoint).
#   3. On completion (epoch >= 500), crash, or crash-loop, cancel the queued
#      follow-up. Otherwise let it run to continue training.
# ============================================================================

#SBATCH --job-name=flexnet
#SBATCH --partition=hpc_l40s_b
#SBATCH --gpus=l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=23:30:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --signal=B:USR1@600

# --- Wall-time signal handling ---
WALL_TIME_HIT=0
TRAIN_PID=""
TRAIN_EXIT=""

handle_wall_time() {
    echo ""
    echo "*** USR1 received: SLURM wall time approaching (10 min remaining) ***"
    WALL_TIME_HIT=1
    if [ -n "$TRAIN_PID" ]; then
        echo "Sending SIGTERM to training process (PID $TRAIN_PID)..."
        kill -TERM "$TRAIN_PID" 2>/dev/null
        echo "Waiting for training process to save checkpoint and exit..."
        wait "$TRAIN_PID" 2>/dev/null
        TRAIN_EXIT=$?
    fi
}

trap 'handle_wall_time' USR1

usage() {
    echo "Usage: sbatch train_single_gpu.sh <experiment_name>" >&2
    echo "  experiment_name: name of experiment directory under __local__/" >&2
    echo "" >&2
    echo "Note: this script does NOT take a num_gpus arg. Use train_multi_gpu.sh for multi-GPU." >&2
    exit 1
}

# --- Parse experiment name ---
if [ "$#" -gt 1 ]; then
    echo "ERROR: too many arguments (got $#, expected exactly 1)" >&2
    usage
fi
EXPERIMENT_NAME="${1:-}"
if [ -z "$EXPERIMENT_NAME" ]; then
    echo "ERROR: experiment_name is required" >&2
    usage
fi
PROJECT_DIR="$HOME/Flexible-Neurons"

echo "============================================"
echo "Job ID:        $SLURM_JOB_ID"
echo "Node:          $SLURM_NODELIST"
echo "Partition:     $SLURM_JOB_PARTITION"
echo "GPUs:          ${SLURM_GPUS_ON_NODE:-${SLURM_JOB_GPUS:-1}}"
echo "Experiment:    $EXPERIMENT_NAME"
echo "Start time:    $(date)"
echo "============================================"

# --- Require SLURM environment ---
# This script pre-schedules follow-up jobs via an afterany SLURM dependency.
# Running it outside SLURM would silently skip that logic and mis-report.
if [ -z "$SLURM_JOB_ID" ]; then
    echo "ERROR: not running under SLURM (SLURM_JOB_ID empty). Submit with sbatch."
    exit 1
fi

# --- Resolve canonical script path ---
# Under SLURM batch mode, $0 points at the spool copy
# (/var/spool/slurmd/jobNNN/slurm_script), which SLURM deletes after the job
# ends. Any follow-up job submitted with "$0" would fail to find the script.
# Resolve to the real on-disk path instead.
SCRIPT_PATH="$PROJECT_DIR/scripts/hpc/train_single_gpu.sh"
if [ ! -f "$SCRIPT_PATH" ]; then
    echo "ERROR: could not locate canonical script path: $SCRIPT_PATH"
    exit 1
fi

# --- Pre-schedule follow-up job ---
# SLURM may force-kill this job at wall-time before an end-of-script sbatch
# can fire. Queue the next job up front (with afterany dependency) so it
# always runs; cancel it later if training is complete or crashed.
#
# sbatch --parsable still writes error text to stdout when it fails, so
# validate the captured value before trusting it. On federated/multi-cluster
# SLURM, --parsable returns "jobid;clustername"; strip the cluster suffix.
NEXT_JOB_RAW=$(sbatch --parsable --dependency=afterany:$SLURM_JOB_ID "$SCRIPT_PATH" "$EXPERIMENT_NAME" 2>/dev/null)
NEXT_JOB_ID="${NEXT_JOB_RAW%%;*}"
if ! [[ "$NEXT_JOB_ID" =~ ^[0-9]+$ ]]; then
    echo "WARNING: sbatch pre-schedule failed or returned unexpected output: '$NEXT_JOB_RAW'"
    echo "Continuing without a pre-scheduled follow-up."
    NEXT_JOB_ID=""
else
    echo "Queued follow-up job: $NEXT_JOB_ID"
fi

# --- Setup environment ---
source /lustre/fs8/huds_lab/store/slu/anaconda3/etc/profile.d/conda.sh
# Pick the conda env from the experiment config's optional "conda_env" key
# (default: flexnet). FFCV runs (use_ffcv=true) need flexnet-ffcv, which has the
# ffcv package; plain flexnet does not. Read with grep BEFORE any env is active
# (no python on PATH yet); absolute path so cwd does not matter.
CFG_ENV_JSON="$PROJECT_DIR/__local__/${EXPERIMENT_NAME}/000000/configurations.json"
CONDA_ENV=$(grep -oP '"conda_env"\s*:\s*"\K[^"]+' "$CFG_ENV_JSON" 2>/dev/null | head -1)
if [ -z "$CONDA_ENV" ]; then
    CONDA_ENV=flexnet
fi
echo "Conda env: $CONDA_ENV"
conda activate "$CONDA_ENV"

# Load CUDA
source /lustre/fs4/ruit/store/ruitsoft/soft/ruit-cudas/switch-cuda.sh 12.4 2>/dev/null || true

cd "$PROJECT_DIR"
mkdir -p slurm_logs

# --- Quota preflight: abort before burning GPU time on a full filesystem ---
bash scripts/hpc/quota_preflight.sh
RC=$?
if [ "$RC" -eq 0 ]; then
    : # usage below warn threshold; continue silently
elif [ "$RC" -eq 1 ]; then
    echo "WARNING: quota preflight usage is above the warn threshold. Continuing anyway."
else
    # The preflight fails closed by design: exit 2 means usage is at/above
    # the hard threshold OR the check itself could not verify quota (see
    # its own output above for which). Any OTHER unexpected code (127
    # preflight script missing, 126 not executable, 128+N killed by a
    # signal, ...) must be treated exactly the same way -- the whole point
    # of a preflight is that failing to run it is not "safe to proceed",
    # it is "unverified", and training on an unverified full filesystem is
    # exactly the failure mode this preflight exists to prevent (#303).
    echo "ERROR: quota preflight aborted (exit $RC): usage is at/above the hard threshold, or the quota could not be determined (lfs missing/failed/unparsable) -- see the quota_preflight output above for which. Refusing to stage data / train on an unverified filesystem." >&2
    if [ -n "$NEXT_JOB_ID" ]; then
        scancel "$NEXT_JOB_ID"
        echo "Cancelled pre-scheduled follow-up job $NEXT_JOB_ID."
    fi
    exit 1
fi

# --- Read target_epoch from experiment config (default 500 for backwards compat) ---
CONFIG_JSON="__local__/${EXPERIMENT_NAME}/000000/configurations.json"
TARGET_EPOCH=$(python -c "import json,sys; print(json.load(open('${CONFIG_JSON}')).get('target_epoch', 500))" 2>/dev/null || echo 500)
if ! [[ "$TARGET_EPOCH" =~ ^[0-9]+$ ]]; then
    echo "WARNING: target_epoch read returned non-numeric ('$TARGET_EPOCH'); falling back to 500."
    TARGET_EPOCH=500
fi
echo "Target epoch: $TARGET_EPOCH"

# --- Resubmission counter ---
# Guards against infinite resubmission loops. Represents "how many runs have
# actually been attempted". Check the cap using the existing value *before*
# incrementing, so a capped-out attempt doesn't leave the counter charged for
# a run that never trained. Reset to 0 at end-of-script whenever progress is
# made (see reset-on-progress block below).
COUNTER_FILE="__local__/${EXPERIMENT_NAME}/000000/.resubmit_count"
RESUBMIT_COUNT=0
if [ -f "$COUNTER_FILE" ]; then
    COUNTER_RAW=$(cat "$COUNTER_FILE" 2>/dev/null)
    if [[ "$COUNTER_RAW" =~ ^[0-9]+$ ]]; then
        RESUBMIT_COUNT=$COUNTER_RAW
    else
        echo "WARNING: counter file contents not numeric ('$COUNTER_RAW'); treating as 0."
        RESUBMIT_COUNT=0
    fi
fi

if [ "$RESUBMIT_COUNT" -ge 30 ]; then
    echo "ERROR: resubmission count $RESUBMIT_COUNT has reached max of 30."
    echo "Aborting before training starts — investigate why training has not completed."
    if [ -n "$NEXT_JOB_ID" ]; then
        scancel "$NEXT_JOB_ID"
        echo "Cancelled pre-scheduled follow-up job $NEXT_JOB_ID."
    fi
    exit 1
fi

RESUBMIT_COUNT=$((RESUBMIT_COUNT + 1))
mkdir -p "$(dirname "$COUNTER_FILE")"
echo "$RESUBMIT_COUNT" > "$COUNTER_FILE"
echo "Resubmission count: $RESUBMIT_COUNT of 30 max"

# --- Verify GPU ---
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}'); print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')"

# --- Record pre-run checkpoint epoch so we can detect progress ---
# PRE_EPOCH is a unified "unknown pre-epoch" sentinel:
#   -1  -> no valid prior checkpoint (either none exists, or the filename did
#          not parse). In both cases, any newly produced valid checkpoint
#          after training counts as progress: we went from "no known state"
#          to "known epoch N".
#   N>=0 -> last known epoch on disk before this run.
#
# The find pattern uses -regex to accept ONLY numeric-epoch checkpoints
# (checkpoint_123.pth). A user-dropped file like checkpoint_best.pth or
# checkpoint_42_backup.pth would otherwise be picked as newest-by-mtime and
# bomb the parse check.
CKPT_DIR="__local__/${EXPERIMENT_NAME}/000000/checkpoints"
mkdir -p "$CKPT_DIR"
PRE_EPOCH=-1
PRE_CKPT=$(find "$CKPT_DIR" -maxdepth 1 -type f -regextype posix-extended -regex '.*/checkpoint_[0-9]+\.pth' -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)
if [ -n "$PRE_CKPT" ]; then
    PRE_EPOCH_RAW=$(basename "$PRE_CKPT" | sed 's/checkpoint_//;s/\.pth//')
    if [[ "$PRE_EPOCH_RAW" =~ ^[0-9]+$ ]]; then
        PRE_EPOCH=$PRE_EPOCH_RAW
    else
        echo "WARNING: could not parse epoch from pre-run checkpoint '$PRE_CKPT'; treating as unknown (-1)."
        PRE_EPOCH=-1
    fi
fi
echo "Pre-run checkpoint epoch: $PRE_EPOCH"

# --- Stage dataset to node-local disk (bypass post-cutover fs8 Lustre-client wedge) ---
# The fs8 client wedges the PyTorch DataLoader read pattern (cl_sync_io_wait); bulk
# parallel cp off fs8 works fine (~1700 files/s @48-way), so we stage the dataset to
# node-local /tmp once per node and train from there (off-Lustre). Idempotent via a
# sentinel, race-safe via a lock (flex+vanilla may share a node), verified by file count.
STAGE_SRC="/lustre/fs8/huds_lab/scratch/slu/Data/imagenet_full"
STAGE_DST="/tmp/imagenet_full"
STAGE_OK="${STAGE_DST}/.stage_ok"
STAGE_LOCK="/tmp/imagenet_full.staging.lock"
SRC_COUNT_CACHE="${PROJECT_DIR}/.imagenet_src_filecount"
if [ -f "$STAGE_OK" ]; then
    echo "[stage] dataset already staged at $STAGE_DST on $(hostname -s)"
elif mkdir "$STAGE_LOCK" 2>/dev/null; then
    echo "[stage] staging ImageNet -> $STAGE_DST on $(hostname -s) (48-way) ..."
    rm -rf "$STAGE_DST"; mkdir -p "$STAGE_DST/train" "$STAGE_DST/val"
    t0=$(date +%s)
    ls "$STAGE_SRC/train" | xargs -P 48 -I{} cp -r "$STAGE_SRC/train/{}" "$STAGE_DST/train/"
    ls "$STAGE_SRC/val"   | xargs -P 48 -I{} cp -r "$STAGE_SRC/val/{}"   "$STAGE_DST/val/"
    t1=$(date +%s)
    [ -s "$SRC_COUNT_CACHE" ] || find "$STAGE_SRC/train" "$STAGE_SRC/val" -type f 2>/dev/null | wc -l > "$SRC_COUNT_CACHE"
    SRC_N=$(cat "$SRC_COUNT_CACHE" 2>/dev/null || echo 0)
    DST_N=$(find "$STAGE_DST/train" "$STAGE_DST/val" -type f 2>/dev/null | wc -l)
    echo "[stage] copied in $((t1-t0))s ; src_files=$SRC_N dst_files=$DST_N"
    if [ "$SRC_N" -gt 0 ] && [ "$DST_N" -eq "$SRC_N" ]; then
        touch "$STAGE_OK"; echo "[stage] OK: file counts match"
    else
        echo "[stage] ERROR: count mismatch (src=$SRC_N dst=$DST_N) -> discarding local copy, fallback to fs8"; rm -rf "$STAGE_DST"
    fi
    rmdir "$STAGE_LOCK" 2>/dev/null || true
else
    echo "[stage] another job is staging on this node; waiting up to 30 min ..."
    for i in $(seq 1 120); do [ -f "$STAGE_OK" ] && break; sleep 15; done
fi
if [ -f "$STAGE_OK" ]; then
    export IMAGENET_LOCAL_DIR="$STAGE_DST"
    echo "[stage] training reads node-local IMAGENET_LOCAL_DIR=$IMAGENET_LOCAL_DIR"
else
    echo "[stage] WARNING: staging unavailable; falling back to fs8 dataset path (may wedge)"
fi

# --- Run training ---
echo ""
echo "Starting training for $EXPERIMENT_NAME..."
python step_2_train_or_continue.py --experiment_name "$EXPERIMENT_NAME" --run_name 000000 &
TRAIN_PID=$!
wait "$TRAIN_PID"
RAW_EXIT=$?
echo "DEBUG: raw wait exit code: $RAW_EXIT"
# Only set TRAIN_EXIT if the handler didn't already capture it
if [ -z "$TRAIN_EXIT" ]; then
    TRAIN_EXIT=$RAW_EXIT
fi

# If USR1 arrived but handler couldn't signal (race: PID not yet set)
if [ "$WALL_TIME_HIT" -eq 1 ] && kill -0 "$TRAIN_PID" 2>/dev/null; then
    kill -TERM "$TRAIN_PID" 2>/dev/null
    wait "$TRAIN_PID" 2>/dev/null
    TRAIN_EXIT=$?
fi

echo ""
echo "Training exited with code: $TRAIN_EXIT"
if [ "$WALL_TIME_HIT" -eq 1 ]; then
    echo "Exit was triggered by SLURM wall-time signal."
fi
echo "End time: $(date)"

# --- Check if training is complete, and cancel or let run the pre-scheduled follow-up ---
# Resubmission rules (a follow-up is already queued at script start):
#   1. Training complete (EPOCH >= 500): cancel follow-up.
#   2. Non-zero exit with no wall-time hit (crash): cancel follow-up, exit 1.
#   3. No progress + no wall-time hit (crash loop): cancel follow-up, exit 1.
#   4. Otherwise (progress made OR wall-time hit): let pre-scheduled follow-up run.
# Progress semantics: EPOCH > PRE_EPOCH. Because PRE_EPOCH == -1 is the
# "no known prior state" sentinel, any newly produced valid checkpoint
# (EPOCH >= 0) counts as progress on a first run or after an unparseable
# pre-run checkpoint.
# CKPT_DIR is guaranteed to exist (mkdir -p above), so no outer -d gate needed.
LATEST_CKPT=$(find "$CKPT_DIR" -maxdepth 1 -type f -regextype posix-extended -regex '.*/checkpoint_[0-9]+\.pth' -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)
if [ -n "$LATEST_CKPT" ]; then
    EPOCH=$(basename "$LATEST_CKPT" | sed 's/checkpoint_//;s/\.pth//')
    echo "Latest checkpoint: epoch $EPOCH (was $PRE_EPOCH before run)"

    if ! [[ "$EPOCH" =~ ^[0-9]+$ ]]; then
        echo "ERROR: Could not parse epoch from checkpoint: $LATEST_CKPT"
        echo "Cancelling follow-up — please investigate."
        [ -n "$NEXT_JOB_ID" ] && scancel "$NEXT_JOB_ID"
        exit 1
    fi

    PROGRESS_MADE=0
    [ "$EPOCH" -gt "$PRE_EPOCH" ] && PROGRESS_MADE=1

    # Reset-on-progress: once we've advanced past the pre-run epoch, the job
    # is clearly not stuck in a crash loop, so zero out the resubmission
    # counter. This keeps the counter meaningful across long training runs.
    if [ "$PROGRESS_MADE" -eq 1 ] && [ -f "$COUNTER_FILE" ]; then
        echo "0" > "$COUNTER_FILE"
        echo "Progress made; reset resubmission counter to 0."
    fi

    if [ "$EPOCH" -ge "$TARGET_EPOCH" ]; then
        echo "Training complete at epoch $EPOCH (target $TARGET_EPOCH)!"
        if [ -n "$NEXT_JOB_ID" ]; then
            scancel "$NEXT_JOB_ID"
            echo "Training complete, cancelled follow-up $NEXT_JOB_ID."
        fi
    elif [ "$TRAIN_EXIT" -ne 0 ] && [ "$WALL_TIME_HIT" -ne 1 ]; then
        echo "ERROR: training failed (exit $TRAIN_EXIT) and no wall-time signal."
        echo "Cancelling follow-up — fix the error first. See .err log."
        [ -n "$NEXT_JOB_ID" ] && scancel "$NEXT_JOB_ID"
        exit 1
    elif [ "$PROGRESS_MADE" -eq 0 ] && [ "$WALL_TIME_HIT" -ne 1 ]; then
        echo "ERROR: no progress made (epoch still $EPOCH) and no wall-time signal."
        echo "Cancelling follow-up — likely a crash loop. See .err log."
        [ -n "$NEXT_JOB_ID" ] && scancel "$NEXT_JOB_ID"
        exit 1
    else
        echo "Training not complete (epoch $EPOCH < $TARGET_EPOCH)."
        if [ -n "$NEXT_JOB_ID" ]; then
            echo "Letting queued follow-up job $NEXT_JOB_ID run."
        else
            echo "WARNING: no pre-scheduled follow-up job to rely on."
        fi
    fi
else
    echo "No checkpoint found after training. Cancelling follow-up."
    [ -n "$NEXT_JOB_ID" ] && scancel "$NEXT_JOB_ID"
    exit 1
fi
