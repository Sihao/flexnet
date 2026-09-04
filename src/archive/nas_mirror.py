"""
Mirror HPC training checkpoints to the bronknas NAS.

Runs on the LOCAL WORKSTATION only -- the HPC has no route to the NAS, and
the NAS (/mnt/bronknas) is a CIFS mount only the workstation has. For each
experiment this module:

  1. discovers checkpoints on the HPC (one ssh call per experiment),
  2. transfers any checkpoint missing, size-mismatched, or (mtime-checked)
     silently rewritten since it was last mirrored to the NAS,
  3. verifies each copy by byte size and mtime,
  4. records what is archived in a manifest written back to the HPC, and
  5. deletes the verified HPC copies, always keeping the newest N.

Two-stage transfer is mandatory, not an optimization: the CIFS mount rejects
rsync's mkstemp temp files despite group-write permissions (docs/lessons.md,
Issue #35), so every transfer goes HPC -> local stage_dir via rsync, then
stage_dir -> NAS via `cp` + `mv` (the mount accepts plain cp/mv).

HPC layout:  HPC_ROOT/EXPERIMENT/000000/checkpoints/checkpoint_<epoch>.pth
NAS layout:  NAS_ROOT/EXPERIMENT/checkpoints/checkpoint_<epoch>.pth

The NAS layout deliberately has NO "000000" run-id level; the HPC layout
does. That asymmetry is load-bearing -- scripts/iso_analysis_cron.sh,
scripts/submit_iso_analyses.sh and scripts/hpc/submit_available_iso_pairs.sh
all read the flat NAS layout and must keep working.

Usage:
    python -m src.archive.nas_mirror inventory
    python -m src.archive.nas_mirror sync --experiments NAME1,NAME2
    python -m src.archive.nas_mirror prune --keep-on-hpc 2
    python -m src.archive.nas_mirror watch --keep-on-hpc 2 --interval-seconds 900
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import logging
import logging.handlers
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, replace as _dc_replace
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Iterable, Iterator, NamedTuple, Optional

logger = logging.getLogger(__name__)

TOOL_NAME = "nas_mirror"
TOOL_VERSION = "1.0.0"

# ---------------------------------------------------------------------------
# Ground truth (measured 2026-08-10): see module docstring for the layouts.
# ---------------------------------------------------------------------------
DEFAULT_SSH_HOST = "rockefeller-hpc"
DEFAULT_HPC_ROOT = "/lustre/fs8/home/slu/Flexible-Neurons/__local__"
DEFAULT_NAS_ROOT = "/mnt/bronknas/Sihao/FlexNet/trajectory_checkpoints"
DEFAULT_STAGE_DIR = Path("/tmp/.nas_mirror_stage")
DEFAULT_KEEP_ON_HPC = 2

# Exclusive-lock directory (issue #317): deliberately NOT config.stage_dir --
# see exclusive_lock()'s docstring. A module attribute (like _WRITE_STABILITY_
# DELAY_S) rather than a MirrorConfig field so tests can monkeypatch it to a
# tmp_path and never touch the real /tmp.
_LOCK_DIR = Path("/tmp/.nas_mirror_locks")
# Absolute floor of free space required on stage_dir before staging anything,
# regardless of the file being transferred (see also the 3x-file-size rule
# in _check_stage_space). The workstation root filesystem has ~20 GB free
# and the largest checkpoint (VGG16) is ~1.66 GB, so 2 GiB is a safe floor
# that still leaves headroom for a full-size transfer plus the OS.
DEFAULT_MIN_FREE_STAGE_BYTES = 2 * 1024**3

_RUN_ID = "000000"  # the fixed run-id directory level the HPC layout always has
_MANIFEST_NAME = ".archive_state.json"
_CHECKPOINT_RE = re.compile(r"checkpoint_(\d+)\.pth")  # used with fullmatch()

# Conservative allowlist for an experiment directory name -- used with
# fullmatch(). Deliberately excludes '/' (so a name can't escape hpc_root or
# nas_root into an arbitrary path) and quote/shell metacharacters (so a name
# can't break out of the single-quoting used when it is embedded in a remote
# ssh command string). See validate_experiment_name.
_EXPERIMENT_NAME_RE = re.compile(r"[A-Za-z0-9._-]+")


def validate_experiment_name(experiment: str) -> None:
    """Raise MirrorError unless *experiment* is a safe directory-name component.

    Issue #316: an experiment name reaches several remote ssh command strings
    (see _hpc_experiment_root and friends) that only single-quote it, which
    protects everything except a literal single quote -- a name containing
    one breaks out of the quoting and injects arbitrary shell commands (the
    auditor's proof: a name that turned a scoped `rm -f` into an arbitrary
    `rm -f ...; touch /tmp/PWNED; ...`). A name of '..' or containing '/'
    would separately escape hpc_root/nas_root into an arbitrary path even
    with quoting fixed. shlex.quote() at every interpolation site (see
    CommandRunner call sites throughout this module) closes the quoting
    half; this allowlist closes the path-escape half and is defense in depth
    for the quoting half too -- do both, not one.
    """
    if not experiment or not _EXPERIMENT_NAME_RE.fullmatch(experiment) or experiment in (".", ".."):
        raise MirrorError(
            f"unsafe experiment name {experiment!r}: must be non-empty and match "
            f"{_EXPERIMENT_NAME_RE.pattern!r} (letters, digits, '.', '_', '-' only)"
        )


class CheckpointStat(NamedTuple):
    """Byte size and mtime (whole seconds since the epoch) of one checkpoint file.

    Comparing two CheckpointStats for equality checks BOTH fields. Size alone
    is not enough to prove a NAS copy still reflects the HPC content it was
    verified against (issue #313): the HPC train script auto-resubmits in
    23.5h chunks, so a job that dies mid-epoch can rewrite checkpoint_N.pth
    with new weights at the SAME epoch number and the SAME byte size (same
    model/optimizer shapes). mtime is truncated to whole seconds because the
    two filesystems involved -- Lustre on the HPC, CIFS on the NAS -- do not
    offer the same sub-second timestamp resolution; comparing at a coarser,
    shared resolution avoids a spurious mismatch (and an unnecessary but
    harmless re-transfer) from that alone.
    """

    size: int
    mtime: int


_SSH_TIMEOUT_S = 60
_RSYNC_TIMEOUT_S = 900  # ceiling for a full checkpoint download (VGG16 ~1.66 GB)
_RSYNC_IO_TIMEOUT_S = 180  # rsync's own --timeout (stall detector), matches existing scripts
_CP_TIMEOUT_S = 300
_MV_TIMEOUT_S = 30
_RM_TIMEOUT_S = 30
_MANIFEST_WRITE_TIMEOUT_S = 30

# Standard Linux strerror() text for EDQUOT (errno 122), the specific failure
# the HPC home quota produces. Matched literally against ssh/rm stderr.
_EDQUOT_TEXT = "Disk quota exceeded"


class MirrorError(Exception):
    """Raised when a command fails, a verification fails, or the module is misconfigured."""


class MirrorConnectivityError(MirrorError):
    """Raised when an ssh/rsync command cannot even reach the HPC (exit 255, launch failure, or timeout).

    A distinct subclass so the per-experiment loops in sync()/prune() can let
    it propagate for a CONTRACT-6 exit code of 2, instead of folding it into
    the per-experiment False/error result reserved for verification failures
    (a checkpoint size mismatch, a still-being-written checkpoint, an EDQUOT
    manifest write). "Delete nothing" already holds either way -- only the
    exit code classification differs.
    """


@dataclass(frozen=True)
class MirrorConfig:
    """Immutable configuration for a mirror run.

    All fields carry defaults (required for a frozen dataclass whose first
    field, ssh_host, also has one) -- hpc_root/nas_root default to the
    measured ground-truth paths, stage_dir to the local scratch directory the
    older watcher scripts already used, min_free_stage_bytes to a safe floor
    given the workstation's ~20 GB free root filesystem.
    """

    ssh_host: str = DEFAULT_SSH_HOST
    hpc_root: str = DEFAULT_HPC_ROOT
    nas_root: str = DEFAULT_NAS_ROOT
    stage_dir: Path = DEFAULT_STAGE_DIR
    keep_on_hpc: int = DEFAULT_KEEP_ON_HPC
    dry_run: bool = False
    min_free_stage_bytes: int = DEFAULT_MIN_FREE_STAGE_BYTES
    log_file: Optional[Path] = None
    # Device id of the NAS mount, captured once by main() right after the
    # startup ensure_nas_mounted() call (see nas_device_id/assert_nas_live
    # below). None disables the live re-check -- every test that builds a
    # MirrorConfig directly against a tmp_path (never a real mount point)
    # relies on that, since main() is the only caller that ever sets it.
    nas_dev: Optional[int] = None

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "MirrorConfig":
        """Build a MirrorConfig from parsed CLI arguments.

        Only overrides a field when the corresponding attribute is present
        and not None on *args*, so this also works with a partial/minimal
        Namespace (e.g. in tests) and falls back to the dataclass defaults
        for anything not supplied. ssh_host and min_free_stage_bytes have no
        CLI flag in this tool's argument list, so they always take the
        dataclass default here.
        """
        kwargs: dict = {}
        for field_name in ("hpc_root", "nas_root", "stage_dir", "keep_on_hpc", "log_file"):
            value = getattr(args, field_name, None)
            if value is not None:
                kwargs[field_name] = value
        kwargs["dry_run"] = bool(getattr(args, "dry_run", False))
        return cls(**kwargs)


# ---------------------------------------------------------------------------
# CommandRunner: the seam that makes this module testable without an HPC.
# ---------------------------------------------------------------------------

class CommandRunner:
    """Thin wrapper around subprocess.run so tests can substitute a fake with the same interface.

    Every ssh, rsync, cp, mv and rm invocation in this module goes through an
    instance of this class rather than calling subprocess directly -- always
    a list argv, never shell=True -- so tests never touch a real HPC or NAS.
    """

    def __init__(self, default_timeout: float = _SSH_TIMEOUT_S) -> None:
        self.default_timeout = default_timeout

    def run(
        self,
        argv: list,
        *,
        timeout: Optional[float] = None,
        check: bool = True,
        input_text: Optional[str] = None,
    ) -> subprocess.CompletedProcess:
        """Run *argv* to completion, capturing stdout/stderr as text.

        Raises MirrorError when the process cannot be launched, times out, or
        (when check is True) exits non-zero -- always including the argv and
        the captured stderr so a failure is debuggable from the log alone.

        For ssh/rsync specifically -- the only two commands in this module
        that cross the network -- a launch failure, a timeout, or an exit
        code of 255 (the documented ssh/rsync convention for "the transport
        itself failed", distinct from a remote command's own exit status)
        raises MirrorConnectivityError instead, so callers can tell a
        connectivity/configuration failure apart from an ordinary command
        failure (e.g. `cat` exiting 1 for a missing file). rsync exit 30
        ("timeout in data send/receive") and 35 ("timeout waiting for daemon
        connection") are the same kind of transport failure -- this module
        sets rsync's own --timeout, so a network stall is the most likely
        trigger -- and are classified the same way, but only for an actual
        rsync invocation: those codes carry no such meaning for an arbitrary
        remote command run over ssh, where a plain MirrorError is correct.
        """
        use_timeout = self.default_timeout if timeout is None else timeout
        is_remote = bool(argv) and argv[0] in ("ssh", "rsync")
        try:
            result = subprocess.run(
                argv,
                capture_output=True,
                text=True,
                timeout=use_timeout,
                shell=False,
                input=input_text,
            )
        except subprocess.TimeoutExpired as exc:
            error_cls = MirrorConnectivityError if is_remote else MirrorError
            raise error_cls(f"command timed out after {use_timeout}s: {argv!r}") from exc
        except OSError as exc:
            error_cls = MirrorConnectivityError if is_remote else MirrorError
            raise error_cls(f"failed to launch command {argv!r}: {exc}") from exc

        if check and result.returncode != 0:
            stderr_tail = (result.stderr or "").strip()[-2000:]
            if is_remote and result.returncode == 255:
                raise MirrorConnectivityError(
                    f"ssh/rsync transport failure (exit 255): {argv!r}\nstderr: {stderr_tail}"
                )
            if argv[0] == "rsync" and result.returncode in (30, 35):
                raise MirrorConnectivityError(
                    f"rsync transport timeout (exit {result.returncode}): {argv!r}\nstderr: {stderr_tail}"
                )
            raise MirrorError(f"command exited {result.returncode}: {argv!r}\nstderr: {stderr_tail}")
        return result

    def check_output(self, argv: list, *, timeout: Optional[float] = None) -> str:
        """Run *argv* and return its stdout with trailing newlines stripped. Raises MirrorError on failure."""
        result = self.run(argv, timeout=timeout, check=True)
        return result.stdout.rstrip("\n")


# ---------------------------------------------------------------------------
# Remote/local path helpers -- the ONLY place the HPC-vs-NAS layout
# asymmetry (000000 run-id level present on HPC, absent on NAS) is encoded.
# ---------------------------------------------------------------------------

def _hpc_experiment_root(config: MirrorConfig, experiment: str) -> str:
    """Return HPC_ROOT/EXPERIMENT/000000 as a POSIX path string (the remote is always Linux)."""
    return f"{config.hpc_root}/{experiment}/{_RUN_ID}"


def _hpc_checkpoints_dir(config: MirrorConfig, experiment: str) -> str:
    return f"{_hpc_experiment_root(config, experiment)}/checkpoints"


def _hpc_manifest_path(config: MirrorConfig, experiment: str) -> str:
    return f"{_hpc_experiment_root(config, experiment)}/{_MANIFEST_NAME}"


def _nas_experiment_root(config: MirrorConfig, experiment: str) -> Path:
    """NAS layout has NO run-id level -- do not add one here."""
    return Path(config.nas_root) / experiment


def _nas_checkpoints_dir(config: MirrorConfig, experiment: str) -> Path:
    return _nas_experiment_root(config, experiment) / "checkpoints"


# ---------------------------------------------------------------------------
# Discovery and inventory (STEP 3)
# ---------------------------------------------------------------------------

def list_experiments(runner: CommandRunner, config: MirrorConfig) -> list:
    """Discover every experiment under HPC_ROOT that has a checkpoints/ directory.

    ONE ssh call: find every directory literally named 'checkpoints' at the
    fixed depth EXPERIMENT/000000/checkpoints below HPC_ROOT (three path
    components -- 000000 is depth 2, checkpoints is depth 3). The experiment
    name is the first path component after HPC_ROOT. Never hardcodes an
    experiment list -- a hardcoded list in the old watcher
    (scripts/watch_checkpoints_trajectory.sh) is what silently lost two VGG
    runs that were never added to it.

    Unlike the per-experiment listings below, a missing/unreadable HPC_ROOT
    is treated as a hard configuration error (not "zero experiments"), since
    silently returning an empty list here could mask a mistyped --hpc-root.
    """
    listing = runner.check_output(
        [
            "ssh", "-n", "-o", "BatchMode=yes", config.ssh_host,
            f"find {shlex.quote(config.hpc_root)} -mindepth 3 -maxdepth 3 -type d -name checkpoints",
        ],
        timeout=_SSH_TIMEOUT_S,
    )
    root = PurePosixPath(config.hpc_root)
    experiments = set()
    for line in listing.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rel = PurePosixPath(line).relative_to(root)
        except ValueError:
            logger.warning("find output outside hpc_root, skipping: %r", line)
            continue
        parts = rel.parts  # expected: (EXPERIMENT, '000000', 'checkpoints')
        if len(parts) != 3 or parts[1] != _RUN_ID or parts[2] != "checkpoints":
            logger.warning("unexpected checkpoints path shape, skipping: %r", line)
            continue
        # Issue #316: a directory a caller doesn't control (anything that can
        # write under HPC_ROOT) must not be trusted just because `find`
        # reported it -- validate it the same way a CLI-supplied name is
        # validated in resolve_experiments, and skip (not abort) a bad one so
        # one hostile/malformed directory can't hide every other experiment.
        try:
            validate_experiment_name(parts[0])
        except MirrorError as exc:
            logger.warning("discovered experiment name failed validation, skipping: %s", exc)
            continue
        experiments.add(parts[0])
    return sorted(experiments)


def _remote_dir_listing(runner: CommandRunner, config: MirrorConfig, remote_dir: str) -> dict:
    """Return {filename: size_bytes} for every regular file directly inside remote_dir.

    ONE ssh call. A directory that does not exist yet (e.g. logs/ before the
    first log is written, or a brand-new experiment) is treated as empty, not
    an error -- this is expected during a live, multi-day training watch.

    Uses `if test -d D; then find ...; else true; fi`, NOT `test -d D &&
    find ... || true` (issue #323): the trailing `|| true` swallowed find's
    own exit code too, so a `find` killed partway (e.g. OOM on the shared
    login node) or that hit a permission error after already printing some
    output still exited 0, and the resulting PARTIAL listing was silently
    accepted as complete. The if/else form still treats "directory absent"
    as empty (the else branch), but a `find` that runs and then fails now
    determines the whole command's exit status, so check=True correctly
    raises instead of returning a truncated listing.
    """
    output = runner.check_output(
        [
            "ssh", "-n", "-o", "BatchMode=yes", config.ssh_host,
            f"if test -d {shlex.quote(remote_dir)}; then find {shlex.quote(remote_dir)} "
            f"-maxdepth 1 -type f -printf '%f %s\\n'; else true; fi",
        ],
        timeout=_SSH_TIMEOUT_S,
    )
    result: dict = {}
    for line in output.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            name, size_str = line.rsplit(" ", 1)
            result[name] = int(size_str)
        except ValueError:
            logger.warning("unparseable directory listing line for %s: %r", remote_dir, line)
    return result


def _filter_checkpoint_listing(listing: dict) -> dict:
    """Filter a {filename: value} dict down to real checkpoint_<digits>.pth entries, keyed by epoch.

    Anything that isn't an exact match -- '.checkpoint_5.pth.tmp', backup
    names like 'checkpoint_5.pth.bak', or a non-numeric epoch -- is dropped.
    Value-type agnostic (a plain size, or a CheckpointStat) -- it only ever
    looks at the filename key and copies the value through untouched.

    A name whose digit run has a leading zero (checkpoint_07.pth) is ALSO
    dropped, not kept (issue #322): it parses to the same epoch int as
    checkpoint_7.pth, so if both existed on the same listing, one would
    silently overwrite the other in whatever order the directory happened to
    be read in -- and prune_experiment always reconstructs the CANONICAL
    (non-padded) filename for its delete list, so a verified leading-zero
    file's size could get attributed to deleting an entirely different,
    unverified file. The canonical writer (src/analysis/run_loader.py) never
    produces a leading-zero name -- an f-string on a plain int can't -- so a
    file that has one is unexpected either way, and the safe response is to
    leave it alone (never sync, never prune it) and log it for a human to
    look at, not guess which of two aliased files is the "real" one.
    """
    result: dict = {}
    for name, value in listing.items():
        match = _CHECKPOINT_RE.fullmatch(name)
        if not match:
            continue
        digits = match.group(1)
        epoch = int(digits)
        if str(epoch) != digits:
            logger.warning("non-canonical checkpoint filename %r (leading zero?); ignoring, not managing it", name)
            continue
        result[epoch] = value
    return result


def _remote_checkpoint_listing(runner: CommandRunner, config: MirrorConfig, remote_dir: str) -> dict:
    """Return {filename: CheckpointStat(size, mtime)} for every regular file directly inside remote_dir.

    ONE ssh call, same shape as _remote_dir_listing, but also captures each
    file's mtime (%T@, seconds since the epoch) -- issue #313 needs it to
    tell a checkpoint that was silently rewritten (same epoch number, same
    byte size, different content) apart from one that has genuinely not
    changed since it was last archived. A directory that does not exist yet
    is treated as empty, not an error, exactly like _remote_dir_listing.

    Same if/else form as _remote_dir_listing, not `... && find ... || true`
    (issue #323) -- see that function's docstring for why.
    """
    output = runner.check_output(
        [
            "ssh", "-n", "-o", "BatchMode=yes", config.ssh_host,
            f"if test -d {shlex.quote(remote_dir)}; then find {shlex.quote(remote_dir)} "
            f"-maxdepth 1 -type f -printf '%f %s %T@\\n'; else true; fi",
        ],
        timeout=_SSH_TIMEOUT_S,
    )
    result: dict = {}
    for line in output.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            name, size_str, mtime_str = line.rsplit(" ", 2)
            result[name] = CheckpointStat(int(size_str), int(float(mtime_str)))
        except ValueError:
            logger.warning("unparseable checkpoint listing line for %s: %r", remote_dir, line)
    return result


def remote_checkpoints(runner: CommandRunner, config: MirrorConfig, experiment: str) -> dict:
    """Return {epoch: CheckpointStat(size, mtime)} for every real checkpoint file on the HPC. ONE ssh call."""
    listing = _remote_checkpoint_listing(runner, config, _hpc_checkpoints_dir(config, experiment))
    return _filter_checkpoint_listing(listing)


def local_nas_checkpoints(config: MirrorConfig, experiment: str) -> dict:
    """Return {epoch: CheckpointStat(size, mtime)} for every real checkpoint file already on the NAS.

    Local filesystem read (the NAS is a CIFS mount on this workstation) --
    no ssh, no CommandRunner involved. The mtime half of CheckpointStat only
    means anything here because _two_stage_copy uses `cp -p`, so the mtime
    on the NAS copy is the ORIGINAL HPC mtime at transfer time, not "whenever
    we happened to copy it".

    A stale CIFS file handle (ESTALE) or a transport-level failure (EIO)
    raises a bare OSError from pathlib here -- it only swallows ENOENT,
    ENOTDIR, EBADF and ELOOP internally. Issue #321: left uncaught, that
    OSError would propagate as a raw traceback and exit 1, the code CONTRACT
    6 reserves for a verification failure -- not the exit 2 a dead/stale NAS
    mount actually is. Caught here and re-raised as MirrorError so it takes
    the same path (and the same "delete nothing this round") as every other
    NAS/HPC failure in this module.
    """
    nas_dir = _nas_checkpoints_dir(config, experiment)
    try:
        if not nas_dir.is_dir():
            return {}
        listing = {
            entry.name: CheckpointStat(entry.stat().st_size, int(entry.stat().st_mtime))
            for entry in nas_dir.iterdir() if entry.is_file()
        }
    except OSError as exc:
        raise MirrorError(f"failed to read NAS checkpoints for {experiment} at {nas_dir}: {exc}") from exc
    return _filter_checkpoint_listing(listing)


def _remote_file_size(runner: CommandRunner, config: MirrorConfig, remote_path: str) -> int:
    """Return the byte size of remote_path on the HPC via a single ssh stat call. Raises MirrorError if missing."""
    output = runner.check_output(
        ["ssh", "-n", "-o", "BatchMode=yes", config.ssh_host, f"stat -c%s {shlex.quote(remote_path)}"],
        timeout=_SSH_TIMEOUT_S,
    )
    output = output.strip()
    try:
        return int(output)
    except ValueError as exc:
        raise MirrorError(f"unexpected stat output for {remote_path!r}: {output!r}") from exc


# ---------------------------------------------------------------------------
# Transfer, two-stage, mandatory (STEP 4)
# ---------------------------------------------------------------------------

def _check_stage_space(config: MirrorConfig, remote_size: int) -> None:
    """Raise MirrorError if stage_dir lacks room for a transfer of remote_size bytes.

    Free space must be at least min_free_stage_bytes (an absolute floor
    regardless of file size) AND at least 3x remote_size (room for the
    staged copy plus headroom -- the file only ever exists once at a time in
    stage_dir, but 3x is a deliberately conservative margin given the
    workstation's ~20 GB root filesystem is a real constraint).
    """
    config.stage_dir.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(config.stage_dir)
    threshold = max(config.min_free_stage_bytes, 3 * remote_size)
    if usage.free < threshold:
        raise MirrorError(
            f"insufficient free space on stage dir {config.stage_dir}: "
            f"{usage.free} bytes free, need >= {threshold} bytes "
            f"(min_free_stage_bytes={config.min_free_stage_bytes}, 3x remote_size={3 * remote_size})"
        )


def _two_stage_copy(
    runner: CommandRunner, config: MirrorConfig, remote_path: str, nas_dest: Path, *, stage_name: str
) -> int:
    """Copy remote_path -> nas_dest via the mandatory rsync-then-cp two-stage path.

    rsync cannot write directly to the CIFS-mounted NAS: its mkstemp temp
    file is rejected by the mount's forced permissions (docs/lessons.md,
    Issue #35). So: rsync HPC -> local stage_dir (where rsync is happy), then
    `cp` the staged file to a .tmp name on the NAS, then `mv` onto the final
    name (the mount accepts plain cp/mv). The staged file and any leftover
    .tmp are always removed, success or failure, via best-effort `rm -f`
    calls that log rather than raise, so a cleanup hiccup never masks the
    real error from the rsync/cp/mv chain.

    nas_tmp carries this process's pid so two concurrent runs (issue #317,
    e.g. two different --stage-dir values that don't share a lock) can never
    cp/mv into the same NAS temp path and interleave writes to one file.

    Bookends the whole copy with assert_nas_live (issue #312): once before
    touching the NAS at all (so a dropped mount is caught before mkdir would
    silently recreate the NAS tree on the root filesystem), and once more
    right before the final size is trusted as verified (so a mount that drops
    mid-copy is still caught instead of quietly passing verification).

    `cp -p` (not plain `cp`) preserves the source's mtime, and rsync -a
    already preserves it HPC -> stage_dir, and `mv`/rename preserves it
    stage_dir -> nas_dest -- so nas_dest ends up carrying the ORIGINAL HPC
    mtime all the way through. That is load-bearing for issue #313: it is
    how sync_experiment/prune_experiment can tell a byte-identical-sized but
    silently rewritten checkpoint (a resumed training job replaying an
    epoch number) apart from one that has genuinely not changed since it was
    last archived, without needing to add a field to the manifest.

    Returns the byte size of the file that ends up at nas_dest.
    """
    assert_nas_live(config)
    nas_dest.parent.mkdir(parents=True, exist_ok=True)
    config.stage_dir.mkdir(parents=True, exist_ok=True)
    staged = config.stage_dir / stage_name
    nas_tmp = nas_dest.with_name(f".{nas_dest.name}.{os.getpid()}.tmp")
    try:
        runner.run(
            [
                "rsync", "-a", "--partial", f"--timeout={_RSYNC_IO_TIMEOUT_S}",
                f"{config.ssh_host}:{remote_path}", str(staged),
            ],
            timeout=_RSYNC_TIMEOUT_S,
        )
        runner.run(["cp", "-p", str(staged), str(nas_tmp)], timeout=_CP_TIMEOUT_S)
        runner.run(["mv", str(nas_tmp), str(nas_dest)], timeout=_MV_TIMEOUT_S)
    finally:
        for leftover in (staged, nas_tmp):
            try:
                runner.run(["rm", "-f", str(leftover)], timeout=_RM_TIMEOUT_S)
            except MirrorError as exc:
                logger.warning("cleanup of %s failed: %s", leftover, exc)
    assert_nas_live(config)
    return nas_dest.stat().st_size


def transfer_checkpoint(runner: CommandRunner, config: MirrorConfig, experiment: str, epoch: int) -> bool:
    """Copy one checkpoint from the HPC to the NAS via the mandatory two-stage path.

    Checks free stage space, then transfers and verifies the final NAS size
    against a fresh HPC stat taken right before the transfer starts. On a
    size mismatch the bad NAS file is removed and MirrorError is raised, so
    the next run retries it.

    Does NOT stat the file twice, several seconds apart, to detect a
    checkpoint still being written (issue #324, removed -- it used to be
    right here). The only writer in this codebase, save_checkpoint() in
    src/analysis/run_loader.py, writes to a temp name and os.replace()s it
    into place -- atomic on POSIX, so checkpoint_N.pth becomes visible at
    its FINAL size in one step and two stats taken any distance apart always
    agreed. The check measured nothing real, and it was not free: for the
    ~96 pending dense VGG checkpoints alone it meant that many extra ssh
    connections to a SHARED login node (this project's HPC rules are
    explicit that unnecessary login-node load is a real cost) plus 5+
    seconds of pure sleep each. The one race this cannot rule out -- a
    resumed training job rewriting checkpoint_N.pth after this stat but
    before rsync reads it -- is not a torn/partial read either way:
    os.replace() swaps the directory entry to a new inode while rsync (once
    it has opened the file) keeps reading the OLD one to completion through
    its still-open file descriptor. A rewrite landing between this stat and
    the transfer, not a mid-write torn read, is what issue #313's
    size+mtime check exists to catch.
    """
    remote_path = f"{_hpc_checkpoints_dir(config, experiment)}/checkpoint_{epoch}.pth"
    nas_dest = _nas_checkpoints_dir(config, experiment) / f"checkpoint_{epoch}.pth"

    remote_size = _remote_file_size(runner, config, remote_path)
    if config.dry_run:
        logger.info("[DRY RUN] would archive %s epoch %d (%d bytes)", experiment, epoch, remote_size)
        return True

    _check_stage_space(config, remote_size)

    stage_name = f".{experiment}__checkpoint_{epoch}.pth.stage"
    final_size = _two_stage_copy(runner, config, remote_path, nas_dest, stage_name=stage_name)

    if final_size != remote_size:
        try:
            runner.run(["rm", "-f", str(nas_dest)], timeout=_RM_TIMEOUT_S)
        except MirrorError as exc:
            logger.warning("failed to remove bad NAS copy %s: %s", nas_dest, exc)
        raise MirrorError(
            f"size mismatch after transfer for {experiment} epoch {epoch}: "
            f"remote={remote_size} nas={final_size} bytes; removed bad NAS copy"
        )

    logger.info("archived %s epoch %d -> %d bytes", experiment, epoch, remote_size)
    return True


def _mirror_one_small_file(
    runner: CommandRunner,
    config: MirrorConfig,
    experiment: str,
    remote_path: str,
    dest: Path,
    expected_size: int,
    *,
    stage_name: str,
) -> None:
    """Two-stage copy one small file, logging (not raising) and removing a bad NAS copy on mismatch.

    Unlike checkpoints, a mismatched config/log mirror is not safety-critical
    for HPC pruning, so this is best-effort: log and move on rather than
    raise, letting the rest of the sync pass continue. Every transfer is
    logged -- with the experiment, the filename, and the byte size -- on
    both the success and the mismatch path, matching transfer_checkpoint's
    logging for checkpoints (STEP 7).
    """
    final_size = _two_stage_copy(runner, config, remote_path, dest, stage_name=stage_name)
    if final_size != expected_size:
        try:
            runner.run(["rm", "-f", str(dest)], timeout=_RM_TIMEOUT_S)
        except MirrorError as exc:
            logger.warning("failed to remove bad NAS copy %s: %s", dest, exc)
        logger.warning(
            "size mismatch mirroring %s %s (remote=%d nas=%d bytes); removed bad NAS copy",
            experiment, remote_path, expected_size, final_size,
        )
        return

    logger.info("mirrored %s %s -> %d bytes", experiment, dest.name, final_size)


def mirror_small_files(runner: CommandRunner, config: MirrorConfig, experiment: str) -> None:
    """Mirror configurations.json and the whole logs/ directory to the NAS, two-stage.

    Uses two ssh listing calls total (one for the experiment root, one for
    logs/), not one per file. Missing files/dirs are logged and skipped, not
    treated as errors -- during early training neither may exist yet.
    """
    if config.dry_run:
        logger.info("[DRY RUN] would mirror configurations.json and logs/ for %s", experiment)
        return

    exp_root = _hpc_experiment_root(config, experiment)
    root_listing = _remote_dir_listing(runner, config, exp_root)
    cfg_size = root_listing.get("configurations.json")
    if cfg_size is not None:
        cfg_dest = _nas_experiment_root(config, experiment) / "configurations.json"
        if not (cfg_dest.exists() and cfg_dest.stat().st_size == cfg_size):
            _mirror_one_small_file(
                runner, config, experiment, f"{exp_root}/configurations.json", cfg_dest, cfg_size,
                stage_name=f".{experiment}__configurations.json.stage",
            )
    else:
        logger.debug("no configurations.json yet for %s", experiment)

    logs_listing = _remote_dir_listing(runner, config, f"{exp_root}/logs")
    for name, size in logs_listing.items():
        dest = _nas_experiment_root(config, experiment) / "logs" / name
        if dest.exists() and dest.stat().st_size == size:
            continue
        _mirror_one_small_file(
            runner, config, experiment, f"{exp_root}/logs/{name}", dest, size,
            stage_name=f".{experiment}__logs__{name}.stage",
        )


# ---------------------------------------------------------------------------
# Archive state manifest (STEP 5)
# ---------------------------------------------------------------------------

def _archived_through_epoch(archived_epochs: set, existing_epochs: set) -> int:
    """Largest N such that every epoch in 1..N that exists (HPC or NAS) is archived. 0 if none exist.

    An epoch number that simply does not exist anywhere is not a gap -- only
    an *existing* un-archived epoch breaks the contiguous run. Walks the
    SORTED existing epochs directly rather than range(1, max(existing)+1)
    (issue #318): the checkpoint filename regex accepts any digit run, so a
    single stray file such as a Unix-timestamp-named checkpoint_1691234567.pth
    used to force ~1.7 billion range() iterations -- while holding the
    exclusive lock, blocking every other run including the systemd timer.
    This walk is linear in the number of checkpoints instead, which is what
    actually bounds it in practice.
    """
    if not existing_epochs:
        return 0
    result = 0
    for epoch in sorted(existing_epochs):
        if epoch not in archived_epochs:
            return epoch - 1
        result = epoch
    return result


def _utc_now_iso() -> str:
    """Return the current UTC time as ISO 8601, seconds precision, ending in 'Z'."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def build_manifest(
    archived_epochs: Iterable[int], experiment: str, config: MirrorConfig, *, known_epochs: Optional[set] = None
) -> dict:
    """Build the archive-state manifest dict for *experiment*.

    known_epochs: every epoch number known to exist on the HPC or the NAS,
    used to compute archived_through_epoch per the "no gap in what exists"
    rule (see _archived_through_epoch). This is not one of the three
    positional parameters the issue named (archived_epochs, experiment,
    config) because that rule cannot be evaluated from archived_epochs
    alone -- it needs to know what epochs *exist*, which requires HPC/NAS
    listings the caller already has. Kept keyword-only with a safe default
    (equivalent to "nothing exists that isn't archived") so the 3-positional
    call shape from the issue still works.

    Returns a dict with exactly: schema_version, experiment, archived_epochs,
    archived_through_epoch, nas_root, updated_utc, writer.
    """
    archived_sorted = sorted(set(archived_epochs))
    existing = set(known_epochs) if known_epochs is not None else set(archived_sorted)
    return {
        "schema_version": 1,
        "experiment": experiment,
        "archived_epochs": archived_sorted,
        "archived_through_epoch": _archived_through_epoch(set(archived_sorted), existing),
        "nas_root": config.nas_root,
        "updated_utc": _utc_now_iso(),
        "writer": f"{TOOL_NAME}/{TOOL_VERSION}",
    }


def _empty_manifest(experiment: str, config: MirrorConfig) -> dict:
    """A manifest representing 'nothing archived', used whenever the real one is missing/malformed."""
    return build_manifest([], experiment, config, known_epochs=set())


def read_manifest(runner: CommandRunner, config: MirrorConfig, experiment: str) -> dict:
    """Read HPC_ROOT/EXPERIMENT/000000/.archive_state.json.

    A missing file, malformed JSON, or JSON without a valid archived_epochs
    list are ALL treated as "nothing archived" -- never as "everything
    archived". This is the safety net prune_experiment relies on.

    A MirrorConnectivityError (the HPC itself is unreachable) is NOT folded
    into that safety net (issue #319): silently treating "can't reach the
    HPC" the same as "no manifest yet" would make prune_experiment run with
    an artificially-empty archived_epochs and report a clean, empty result
    instead of surfacing the real connectivity failure as CONTRACT 6's exit
    2. "Delete nothing" still holds either way -- only the exit code and the
    visibility of the failure differ.
    """
    remote_path = _hpc_manifest_path(config, experiment)
    try:
        raw = runner.check_output(
            ["ssh", "-n", "-o", "BatchMode=yes", config.ssh_host, f"cat {shlex.quote(remote_path)}"],
            timeout=_SSH_TIMEOUT_S,
        )
    except MirrorConnectivityError:
        raise
    except MirrorError:
        logger.info("no existing manifest for %s (treating as nothing archived)", experiment)
        return _empty_manifest(experiment, config)

    try:
        manifest = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("malformed manifest for %s; treating as nothing archived", experiment)
        return _empty_manifest(experiment, config)

    if not isinstance(manifest, dict) or not isinstance(manifest.get("archived_epochs"), list):
        logger.warning("manifest for %s missing archived_epochs list; treating as nothing archived", experiment)
        return _empty_manifest(experiment, config)

    return manifest


def write_manifest(runner: CommandRunner, config: MirrorConfig, experiment: str, manifest: dict) -> bool:
    """Write *manifest* to HPC_ROOT/EXPERIMENT/000000/.archive_state.json.

    Writes a .tmp file first, then renames over the target on the remote in
    the SAME ssh call, so a reader never observes a half-written file.

    If the HPC home quota is exceeded, the remote write fails with errno 122
    (EDQUOT, strerror "Disk quota exceeded"). That specific failure is caught,
    logged as a warning, and does NOT raise -- the caller must not delete
    anything on the HPC this run, since the manifest is the only proof of
    what is archived. Returns False in that case, True on a successful write
    (including a dry-run "would write").

    Two precautions on that match (issue #320): the remote command is run
    with LC_ALL=C, since ssh can forward the client's locale (AcceptEnv
    LANG/LC_*) and a non-C locale's strerror text would silently make a
    quota failure fatal instead of the non-fatal path STEP 5 requires; and
    the match looks ONLY at the command's stderr, not the full exception
    message (which also contains the argv repr), since an unrelated failure
    whose stderr happens to merely CONTAIN this substring -- a login banner,
    a bashrc quota warning, both real risks on an HPC that is genuinely over
    quota -- must not be silently reclassified as "quota, not fatal".
    """
    target = _hpc_manifest_path(config, experiment)
    tmp = f"{target}.tmp"
    payload = json.dumps(manifest, indent=2, sort_keys=True)

    if config.dry_run:
        logger.info("[DRY RUN] would write manifest for %s: %s", experiment, payload)
        return True

    try:
        runner.run(
            [
                "ssh", "-o", "BatchMode=yes", config.ssh_host,
                f"LC_ALL=C cat > {shlex.quote(tmp)} && LC_ALL=C mv {shlex.quote(tmp)} {shlex.quote(target)}",
            ],
            timeout=_MANIFEST_WRITE_TIMEOUT_S,
            input_text=payload,
        )
    except MirrorError as exc:
        _, _, stderr_text = str(exc).partition("\nstderr: ")
        if _EDQUOT_TEXT in stderr_text:
            logger.warning(
                "manifest write for %s failed: HPC home quota exceeded (%s). "
                "Not pruning this run -- the stale manifest still protects newly-archived "
                "checkpoints from deletion since prune re-reads it fresh.",
                experiment, _EDQUOT_TEXT,
            )
            return False
        raise

    logger.info("wrote manifest for %s (archived_through_epoch=%d)", experiment, manifest["archived_through_epoch"])
    return True


# ---------------------------------------------------------------------------
# Prune -- the deleting half. Guarded hard. (STEP 6)
# ---------------------------------------------------------------------------

def _validate_keep_on_hpc(keep_on_hpc: int) -> None:
    """Raise ValueError unless keep_on_hpc >= 1 (a run with no local checkpoint cannot resume).

    Shared by prune_experiment (checked once per experiment, cheap either
    way) and watch (issue #325: checked ONCE up front, before the first
    sync -- watch prunes every pass just like the standalone prune
    subcommand, so relying solely on prune_experiment's own check would let
    a bad --keep-on-hpc run a full, possibly multi-hour sync pass before
    ever discovering the misconfiguration).
    """
    if keep_on_hpc < 1:
        raise ValueError(f"keep_on_hpc must be >= 1 (a run with no local checkpoint cannot resume); got {keep_on_hpc}")


def prune_experiment(runner: CommandRunner, config: MirrorConfig, experiment: str) -> tuple[list, int]:
    """Delete HPC checkpoints that are safely archived, always keeping the newest keep_on_hpc.

    An epoch may be deleted only when ALL of:
      (a) it is in archived_epochs of the manifest, read fresh from the HPC now,
      (b) the NAS copy exists and its (size, mtime) matches the HPC copy,
          re-checked now (not trusted from anything computed earlier in the
          run) -- mtime as well as size since issue #313: a resumed training
          job can rewrite a checkpoint at the same epoch number with the same
          byte size, and size alone cannot tell that apart from an unchanged
          file, and
      (c) it is not among the newest keep_on_hpc checkpoints by epoch number.

    Never deletes the newest checkpoint under any configuration -- keep_on_hpc
    >= 1 is enforced below, and the newest epoch is always inside that
    keep-window. keep_on_hpc == 0 raises ValueError: a run with no local
    checkpoint cannot resume.

    Deletions are batched into a single ssh call. Returns
    (deleted_epochs, freed_bytes); (,) is ([], 0) when nothing qualifies.
    """
    _validate_keep_on_hpc(config.keep_on_hpc)

    # Re-check the NAS is still the mount this run started against (#312)
    # BEFORE trusting anything read from it below -- guard (b) is only as
    # good as local_nas_checkpoints() actually reading the NAS.
    assert_nas_live(config)

    hpc = remote_checkpoints(runner, config, experiment)
    if not hpc:
        return [], 0

    manifest = read_manifest(runner, config, experiment)
    archived_epochs = set(manifest.get("archived_epochs", []))

    newest_epochs = set(sorted(hpc)[-config.keep_on_hpc:])
    nas = local_nas_checkpoints(config, experiment)

    to_delete: list = []
    for epoch, hpc_stat in hpc.items():
        if epoch in newest_epochs:
            continue
        if epoch not in archived_epochs:
            continue
        nas_stat = nas.get(epoch)
        if nas_stat != hpc_stat:
            logger.warning(
                "%s epoch %d not pruned: NAS (size, mtime) %r != HPC %r", experiment, epoch, nas_stat, hpc_stat
            )
            continue
        to_delete.append(epoch)

    to_delete.sort()
    if not to_delete:
        return [], 0

    freed_bytes = sum(hpc[e].size for e in to_delete)

    if config.dry_run:
        logger.info("[DRY RUN] would delete %s epochs %s (%d bytes)", experiment, to_delete, freed_bytes)
        return to_delete, freed_bytes

    remote_paths = [f"{_hpc_checkpoints_dir(config, experiment)}/checkpoint_{e}.pth" for e in to_delete]
    quoted = " ".join(shlex.quote(p) for p in remote_paths)
    runner.run(["ssh", "-n", "-o", "BatchMode=yes", config.ssh_host, f"rm -f {quoted}"], timeout=_RM_TIMEOUT_S)
    logger.info("pruned %s epochs %s (%d bytes freed)", experiment, to_delete, freed_bytes)
    return to_delete, freed_bytes


# ---------------------------------------------------------------------------
# Orchestration (STEP 7)
# ---------------------------------------------------------------------------

def resolve_experiments(runner: CommandRunner, config: MirrorConfig, requested: Optional[list]) -> list:
    """Return *requested* if given (and non-empty), else the full discovered experiment list.

    Every requested name is validated (issue #316) before being returned --
    unlike the discovery path in list_experiments, a bad CLI-supplied name is
    a hard error (raises MirrorError) rather than something to skip, since
    the caller explicitly asked for it.
    """
    if requested:
        for experiment in requested:
            validate_experiment_name(experiment)
        return requested
    return list_experiments(runner, config)


def sync_experiment(runner: CommandRunner, config: MirrorConfig, experiment: str) -> bool:
    """Discover, transfer, and record archive state for one experiment.

    Transfers every checkpoint missing or size-mismatched on the NAS, mirrors
    configurations.json and logs/, then writes the archive-state manifest.

    Returns True when the whole pass succeeded cleanly (every wanted
    transfer completed and the manifest was written); False when something
    recoverable failed this round (a per-checkpoint transfer failure, or a
    quota-blocked manifest write). Per-checkpoint transfer failures are
    logged and do not abort the rest of the experiment; a MirrorError only
    propagates out of this function for a failure that is not per-checkpoint
    (e.g. the initial listing calls themselves failing). A
    MirrorConnectivityError always propagates, even from inside the
    per-checkpoint loop or mirror_small_files, so callers can tell a genuine
    HPC connectivity/configuration failure apart from an ordinary
    per-checkpoint verification failure (see MirrorConnectivityError).
    """
    # Re-check the NAS is still the mount this run started against (#312)
    # before reading it as authoritative for "already mirrored" below.
    assert_nas_live(config)

    hpc = remote_checkpoints(runner, config, experiment)
    nas = local_nas_checkpoints(config, experiment)

    ok = True
    # Issue #314: seed archived_epochs with an epoch only when it is either
    # (a) already gone from the HPC (prune already ran, so the NAS copy is
    # the only one left and can only have gotten there via a transfer that
    # was itself verified), or (b) still on the HPC AND its (size, mtime)
    # matches the NAS copy right now -- NOT merely "present on the NAS at
    # some point", which is what let a failed re-transfer's stale NAS leftover
    # get claimed as archived. Matching by (size, mtime), not size alone, is
    # issue #313: a resumed training job can rewrite a checkpoint at the same
    # epoch number with the same byte size.
    archived_epochs = {epoch for epoch in nas if epoch not in hpc or nas[epoch] == hpc[epoch]}

    for epoch, hpc_stat in sorted(hpc.items()):
        if nas.get(epoch) == hpc_stat:
            continue  # already mirrored and verified: size AND mtime both match
        try:
            # transfer_checkpoint() always returns True or raises (issue
            # #324 removed its only False-returning path) -- reaching the
            # line after this try/except IS the success case.
            transfer_checkpoint(runner, config, experiment, epoch)
        except MirrorConnectivityError:
            raise  # connectivity/configuration failure, not a per-checkpoint verification failure
        except MirrorError as exc:
            logger.error("transfer failed for %s epoch %d: %s", experiment, epoch, exc)
            ok = False
            archived_epochs.discard(epoch)  # #314: a failed re-transfer must never leave a stale claim
            continue
        archived_epochs.add(epoch)

    try:
        mirror_small_files(runner, config, experiment)
    except MirrorConnectivityError:
        raise  # connectivity/configuration failure, not a per-file verification failure
    except MirrorError as exc:
        logger.error("failed to mirror small files for %s: %s", experiment, exc)
        ok = False

    known_epochs = set(hpc) | set(nas) | archived_epochs
    manifest = build_manifest(sorted(archived_epochs), experiment, config, known_epochs=known_epochs)
    written = write_manifest(runner, config, experiment, manifest)

    return ok and written


def sync(runner: CommandRunner, config: MirrorConfig, experiments: list) -> dict:
    """Run sync_experiment for each of *experiments*, continuing past per-experiment errors.

    Returns {experiment: True|False} (see sync_experiment for what False means).
    A MirrorError from one experiment's listing calls is caught here so the
    rest of the batch still runs. A MirrorConnectivityError is deliberately
    NOT caught here -- it propagates out of sync() (and from there out of
    main()'s dispatch) so the process exits 2, not 1, for a genuine HPC
    connectivity/configuration failure, even one that happens mid-batch
    rather than during the initial experiment-discovery call. "Delete
    nothing" already holds either way; this only fixes the exit code.
    """
    results: dict = {}
    for experiment in experiments:
        try:
            results[experiment] = sync_experiment(runner, config, experiment)
        except MirrorConnectivityError:
            raise
        except MirrorError as exc:
            logger.error("sync failed for %s: %s", experiment, exc)
            results[experiment] = False
    return results


def prune(runner: CommandRunner, config: MirrorConfig, experiments: list) -> dict:
    """Run prune_experiment for each of *experiments*, continuing past per-experiment errors.

    Returns {experiment: {"deleted_epochs": [...], "freed_bytes": N, "error": str|None}}.
    As in sync(), a MirrorConnectivityError is deliberately NOT caught here --
    it propagates so the process exits 2 for a genuine connectivity or
    configuration failure, instead of being folded into a per-experiment
    "error" string that main() would otherwise classify as exit 1.
    """
    results: dict = {}
    for experiment in experiments:
        try:
            deleted, freed = prune_experiment(runner, config, experiment)
            results[experiment] = {"deleted_epochs": deleted, "freed_bytes": freed, "error": None}
        except MirrorConnectivityError:
            raise
        except MirrorError as exc:
            logger.error("prune failed for %s: %s", experiment, exc)
            results[experiment] = {"deleted_epochs": [], "freed_bytes": 0, "error": str(exc)}
    return results


def build_inventory(runner: CommandRunner, config: MirrorConfig, experiments: list) -> dict:
    """Read-only report of HPC/NAS checkpoint state and manifest status for each experiment."""
    report: dict = {}
    for experiment in experiments:
        hpc = remote_checkpoints(runner, config, experiment)
        nas = local_nas_checkpoints(config, experiment)
        manifest = read_manifest(runner, config, experiment)
        report[experiment] = {
            "hpc_epochs": sorted(hpc),
            "nas_epochs": sorted(nas),
            "missing_on_nas": sorted(set(hpc) - set(nas)),
            "archived_through_epoch": manifest.get("archived_through_epoch", 0),
        }
    return report


# ---------------------------------------------------------------------------
# NAS mount guard
# ---------------------------------------------------------------------------

def _find_mount_point(path: Path) -> Path:
    """Walk up from *path* (which must exist) to the nearest ancestor that is a mount point.

    Always terminates: '/' is always a mount point, so the walk cannot run
    past it.
    """
    candidate = path
    while not candidate.is_mount():
        candidate = candidate.parent
    return candidate


def _nearest_existing_ancestor(path: Path) -> Path:
    """Walk up from *path* to the nearest ancestor that actually exists on disk.

    Always terminates: the walk stops as soon as parent == candidate, which
    happens at '/' at the latest.
    """
    existing = path
    while not existing.exists():
        parent = existing.parent
        if parent == existing:
            raise MirrorError(f"{path} does not exist and has no existing ancestor")
        existing = parent
    return existing


def ensure_nas_mounted(config: MirrorConfig) -> None:
    """Verify the NAS is actually mounted, not a plain directory standing in for it.

    Approach: walk up from nas_root to the nearest existing ancestor, then to
    the nearest mount point from there (Path.is_mount()), and require that
    mount point to not be '/'. A missing CIFS mount leaves nas_root as an
    ordinary directory on the root filesystem, whose nearest mount point is
    '/' -- exactly the dangerous case this guards against, since the root
    filesystem only has ~20 GB free and would silently fill up.

    This is the STARTUP check -- called once from main() before anything else
    runs. It alone is not enough to keep the NAS honest for a run that then
    holds the lock for hours (issue #312): see assert_nas_live() below for
    the re-check every later read/write of the NAS as authoritative must make.

    Raises MirrorError when nas_root's nearest mount point is '/'.
    """
    existing = _nearest_existing_ancestor(Path(config.nas_root))
    mount_point = _find_mount_point(existing)
    if mount_point == Path("/"):
        raise MirrorError(
            f"NAS not mounted: {config.nas_root} resolves to the root filesystem, not a CIFS mount "
            f"(nearest mount point is '/'). Refusing to proceed -- this would silently fill the "
            f"workstation's local disk."
        )
    logger.info("NAS mount verified: %s is under mount point %s", config.nas_root, mount_point)


def nas_device_id(config: MirrorConfig) -> int:
    """Return the st_dev of nas_root's nearest existing ancestor.

    main() captures this once, right after the startup ensure_nas_mounted(),
    and stores it on config.nas_dev so assert_nas_live() can later tell a
    live mount apart from a drop-then-remount: the path looks mounted again
    either way, but a fresh/different share has a different device id.
    """
    return os.stat(_nearest_existing_ancestor(Path(config.nas_root))).st_dev


def assert_nas_live(config: MirrorConfig) -> None:
    """Re-verify the NAS mount is live, right before treating it as authoritative.

    ensure_nas_mounted() historically ran exactly once, in main(), before the
    lock was even acquired -- but a run holds that lock for as long as a
    multi-day watch loop. When the CIFS mount drops mid-run, nas_root simply
    stops existing; _two_stage_copy's own `nas_dest.parent.mkdir(parents=True)`
    then silently recreates the NAS tree ON THE ROOT FILESYSTEM, cp/mv write
    there, and the byte-size check passes because the (now-local) copy really
    does match. prune_experiment re-reads those local files as if they were
    the NAS and deletes the verified HPC originals (issue #312).

    Call this at every point that is about to read or write the NAS as
    ground truth: before a transfer starts and right before it is trusted as
    verified, before a prune decision, and once per watch pass. It is cheap
    -- os.stat() is a stat() call, not I/O.

    Unlike ensure_nas_mounted(), a failure here raises MirrorConnectivityError
    (not the plain MirrorError base class), so it propagates through
    sync()/prune()/watch() as a connectivity failure (CONTRACT 6: exit 2)
    instead of being folded into an ordinary per-checkpoint or per-experiment
    verification failure.

    Deliberately does NOT repeat ensure_nas_mounted()'s "resolves to the root
    filesystem" check -- that check requires nas_root to sit under a REAL
    Path.is_mount() boundary, which only holds in production, never in a test
    built against tmp_path. The device-id comparison below is strictly
    stronger anyway: a mount that vanished (nothing left at nas_root) or one
    silently recreated on the root filesystem both resolve to a different
    st_dev than the one captured when the mount was known-good, so either
    case is still caught without needing is_mount() at all.

    When config.nas_dev is None (every test that builds a MirrorConfig
    directly, without going through main()), this is a no-op -- there is
    nothing captured yet to compare against.
    """
    if config.nas_dev is None:
        return
    try:
        existing = _nearest_existing_ancestor(Path(config.nas_root))
    except MirrorError as exc:
        raise MirrorConnectivityError(f"NAS mount lost: {exc}") from exc

    current_dev = os.stat(existing).st_dev
    if current_dev != config.nas_dev:
        raise MirrorConnectivityError(
            f"NAS mount changed since this run started (device id {config.nas_dev} -> {current_dev}): "
            f"a drop-and-remount occurred mid-run. Refusing to treat {config.nas_root} as authoritative."
        )


# ---------------------------------------------------------------------------
# Exclusive lock -- guards the whole process so two runs cannot overlap.
# ---------------------------------------------------------------------------

def _lock_path_for(config: MirrorConfig) -> Path:
    """Return the exclusive-lock file path for config.nas_root, under the fixed _LOCK_DIR.

    A short hash, not the raw nas_root string, because nas_root is an
    arbitrary filesystem path (could be long, could contain characters that
    are awkward in a filename) -- the hash just needs to be stable and
    collision-free in practice, not reversible.
    """
    nas_key = hashlib.sha256(str(Path(config.nas_root)).encode()).hexdigest()[:16]
    return _LOCK_DIR / f"{nas_key}.lock"


@contextlib.contextmanager
def exclusive_lock(config: MirrorConfig) -> Iterator[None]:
    """Take an exclusive, non-blocking flock scoped to config.nas_root.

    Raises MirrorError immediately if another run already holds it. flock is
    per-open-file-description, so this correctly rejects a second concurrent
    acquisition even from within the same process/tests, not just across
    processes.

    The lock file lives under the FIXED _LOCK_DIR, named from a hash of
    nas_root -- deliberately NOT under config.stage_dir (issue #317). Two
    runs started with different --stage-dir values used to not exclude each
    other at all; combined with _two_stage_copy's nas_tmp name ALSO being
    fixed (see its own per-pid fix above), that let two concurrent runs
    interleave cp/mv writes into the same NAS temp path and produce a
    same-size but corrupted file that still passed size verification.
    Keying the lock on nas_root instead means every run that could actually
    collide on the same NAS destination always contends for the same lock,
    regardless of what --stage-dir each one happens to use.

    A NAS-side lock was considered instead and rejected: CIFS lock semantics
    are not reliable enough to depend on for correctness, and a lock that
    lives ON the NAS cannot even be acquired to protect against the NAS
    being unreachable in the first place -- the exact failure mode issue
    #312 is about. A local lock, keyed by the NAS this run targets, needs no
    NAS I/O to acquire and is not exposed to the same failure mode it exists
    to guard against.
    """
    _LOCK_DIR.mkdir(parents=True, exist_ok=True)
    lock_path = _lock_path_for(config)
    fh = open(lock_path, "w")
    try:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        fh.close()
        raise MirrorError(
            f"another nas_mirror run holds the lock for {config.nas_root} at {lock_path}; exiting"
        ) from exc

    try:
        fh.write(f"{os.getpid()} {config.nas_root}\n")
        fh.flush()
        yield
    finally:
        fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        fh.close()


# ---------------------------------------------------------------------------
# Watch
# ---------------------------------------------------------------------------

def watch(runner: CommandRunner, config: MirrorConfig, experiments: list, interval_seconds: int) -> int:
    """Loop sync-then-prune across *experiments* every interval_seconds, until Ctrl-C.

    Ctrl-C during time.sleep() raises KeyboardInterrupt in the main thread
    immediately (Python's default SIGINT handling) -- no extra signal
    plumbing is needed since, unlike scripts/sync/sync_daemon.py, this loop
    holds no long-running child process open across the sleep.

    Returns 0 if the most recently completed pass was fully clean, 1 if it
    had any recoverable issue outstanding (mirrors the sync/prune exit codes).

    Three things this loop must never do (issues #312, #315, #319):
      - trust the NAS as live for a whole multi-day run off one check made
        before the loop even started -- assert_nas_live() re-checks it once
        every pass, cheaply, and skips the whole pass (no sync, no prune) if
        it fails, rather than proceeding against a possibly-local directory;
      - prune an experiment whose sync did not fully succeed this pass (a
        still-being-written checkpoint, or -- the common case right now,
        with the HPC over quota -- a manifest write that was blocked). The
        manifest is the only proof of what is archived; pruning against one
        that failed to write is exactly how a freshly-archived checkpoint
        gets deleted before it was ever recorded as safe;
      - stop looping just because one pass hit a connectivity failure. That
        is caught explicitly (not merely via the MirrorError parent class)
        so it is unambiguous in the logs, but a transient blip must not kill
        a watcher meant to run for days unattended -- it is logged, and the
        loop retries next pass;
      - start a pass at all with a bad --keep-on-hpc (issue #325): validated
        up front, once, so a misconfiguration fails immediately instead of
        after a full, possibly multi-hour sync pass has already run.
    """
    _validate_keep_on_hpc(config.keep_on_hpc)
    logger.info(
        "watch starting: experiments=%s interval=%ds keep_on_hpc=%d", experiments, interval_seconds, config.keep_on_hpc
    )
    last_ok = True
    try:
        while True:
            last_ok = True
            try:
                assert_nas_live(config)
            except MirrorError as exc:
                logger.error("NAS mount check failed this pass, skipping sync/prune entirely: %s", exc)
                last_ok = False
            else:
                for experiment in experiments:
                    try:
                        synced = sync_experiment(runner, config, experiment)
                    except MirrorConnectivityError as exc:
                        logger.error("sync connectivity failure for %s, skipping prune this pass: %s", experiment, exc)
                        last_ok = False
                        continue
                    except MirrorError as exc:
                        logger.error("sync failed for %s, skipping prune this pass: %s", experiment, exc)
                        last_ok = False
                        continue
                    if not synced:
                        last_ok = False
                        continue  # #315: never prune off a sync that did not fully succeed
                    try:
                        prune_experiment(runner, config, experiment)
                    except MirrorConnectivityError as exc:
                        logger.error("prune connectivity failure for %s: %s", experiment, exc)
                        last_ok = False
                    except MirrorError as exc:
                        logger.error("prune failed for %s: %s", experiment, exc)
                        last_ok = False
            logger.info("watch pass complete; sleeping %ds", interval_seconds)
            time.sleep(interval_seconds)
    except KeyboardInterrupt:
        logger.info("watch stopped by KeyboardInterrupt")
    return 0 if last_ok else 1


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def _setup_logging(log_file: Optional[Path]) -> None:
    """Configure the root logger: always stderr, plus a rotating *log_file* when given."""
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")

    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setFormatter(fmt)
    root_logger.addHandler(stderr_handler)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(str(log_file), maxBytes=5 * 1024 * 1024, backupCount=3)
        file_handler.setFormatter(fmt)
        root_logger.addHandler(file_handler)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_experiments_arg(raw: Optional[str]) -> Optional[list]:
    """Split a comma-separated --experiments value into a list, or None when not given."""
    if not raw:
        return None
    names = [item.strip() for item in raw.split(",") if item.strip()]
    return names or None


def _emit_result(data: object, *, as_json: bool) -> None:
    """Print *data* to stdout: JSON when as_json, else one line per top-level entry."""
    if as_json:
        print(json.dumps(data, indent=2, sort_keys=True))
        return
    if isinstance(data, dict):
        for key, value in sorted(data.items()):
            print(f"{key}: {value}")
    else:
        print(data)


def _add_common_args(sp: argparse.ArgumentParser, *, keep_on_hpc_required: bool = False) -> None:
    """Add the flags shared by every subcommand to *sp*."""
    sp.add_argument(
        "--hpc-root", default=DEFAULT_HPC_ROOT, metavar="PATH",
        help="Root of the checkpoint tree on the HPC (default: %(default)s)",
    )
    sp.add_argument(
        "--nas-root", default=DEFAULT_NAS_ROOT, metavar="PATH",
        help="Root of the mirrored tree on the NAS (default: %(default)s)",
    )
    sp.add_argument(
        "--stage-dir", type=Path, default=DEFAULT_STAGE_DIR, metavar="PATH",
        help="Local staging directory for two-stage transfers (default: %(default)s)",
    )
    sp.add_argument(
        "--experiments", default=None, metavar="NAME1,NAME2",
        help="Comma-separated experiment names to restrict to (default: discover all)",
    )
    if keep_on_hpc_required:
        sp.add_argument(
            "--keep-on-hpc", type=int, required=True, metavar="N",
            help="Number of newest checkpoints to always keep on the HPC (required)",
        )
    else:
        sp.add_argument(
            "--keep-on-hpc", type=int, default=DEFAULT_KEEP_ON_HPC, metavar="N",
            help="Number of newest checkpoints to always keep on the HPC (default: %(default)s)",
        )
    sp.add_argument("--dry-run", action="store_true", help="Print every action without running rsync, cp, mv or rm.")
    sp.add_argument("--json", action="store_true", help="Emit machine-readable JSON to stdout instead of text.")
    sp.add_argument(
        "--log-file", type=Path, default=None, metavar="PATH",
        help="Also log to this file, in addition to stderr (default: stderr only)",
    )


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the argparse CLI: subcommands inventory, sync, prune, watch."""
    parser = argparse.ArgumentParser(
        prog="python -m src.archive.nas_mirror",
        description=(
            "Mirror HPC training checkpoints to the bronknas NAS, verify each copy, record a manifest "
            "on the HPC, then prune the verified HPC copies. Runs on the local workstation only -- the "
            "HPC has no route to the NAS."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    sp_inventory = subparsers.add_parser(
        "inventory", help="Report HPC/NAS checkpoint state per experiment (read-only)."
    )
    _add_common_args(sp_inventory)

    sp_sync = subparsers.add_parser(
        "sync", help="Transfer missing/mismatched checkpoints to the NAS and write the manifest."
    )
    _add_common_args(sp_sync)

    sp_prune = subparsers.add_parser(
        "prune", help="Delete HPC checkpoints already safely archived, keeping the newest N."
    )
    _add_common_args(sp_prune, keep_on_hpc_required=True)

    sp_watch = subparsers.add_parser("watch", help="Loop sync then prune on an interval until interrupted (Ctrl-C).")
    # keep_on_hpc_required (issue #325): watch prunes just like the standalone
    # prune subcommand does, on every pass, so it must not silently fall back
    # to DEFAULT_KEEP_ON_HPC -- the same explicit-retention requirement STEP 5
    # already put on prune applies here for the same reason.
    _add_common_args(sp_watch, keep_on_hpc_required=True)
    sp_watch.add_argument(
        "--interval-seconds", type=int, default=600, metavar="SECONDS",
        help="Seconds to sleep between watch passes (default: %(default)s)",
    )

    return parser


def _sweep_stale_stage_files(config: MirrorConfig) -> None:
    """Delete leftover .stage files in config.stage_dir from a run that never cleaned up after itself.

    _two_stage_copy's own cleanup runs in a `finally` block, but a SIGKILL
    skips finally blocks entirely (issue #325) -- the staged file (up to
    ~1.66 GB for a VGG16 checkpoint) then survives indefinitely. Nothing
    else ever reclaims it, so repeated kills eventually exhaust the
    workstation's ~20 GB root filesystem and _check_stage_space starts
    refusing every transfer -- a safe failure (fails closed), but a
    confusing one with no message pointing at the actual cause.

    Must only be called AFTER the exclusive lock is held: that is what makes
    this safe. No other nas_mirror run can be using stage_dir at the same
    time, so anything matching the `.stage` naming convention (see
    transfer_checkpoint/_mirror_one_small_file) here is provably a leftover,
    never a file in active use.
    """
    if not config.stage_dir.is_dir():
        return
    for stale in config.stage_dir.glob(".*.stage"):
        try:
            size = stale.stat().st_size
            stale.unlink()
            logger.warning("swept stale stage file left by a previous run: %s (%d bytes)", stale, size)
        except OSError as exc:
            logger.warning("failed to sweep stale stage file %s: %s", stale, exc)


def main(argv: Optional[list] = None) -> int:
    """CLI entry point. Returns the process exit code (0 success, 1 verification failure, 2 config/connectivity)."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    _setup_logging(args.log_file)

    config = MirrorConfig.from_args(args)
    requested = _parse_experiments_arg(args.experiments)
    runner = CommandRunner()

    try:
        ensure_nas_mounted(config)
        # Capture the mount's device id now so every downstream read/write of
        # the NAS as authoritative can detect a later drop-and-remount, not
        # just a drop (#312) -- config is frozen, so this is a new instance.
        config = _dc_replace(config, nas_dev=nas_device_id(config))

        with exclusive_lock(config):
            if args.command != "inventory":
                # Issue #325: sweep once, here, after the lock but before any
                # subcommand runs -- inventory stays strictly read-only (its
                # own documented contract), the other three already mutate
                # local/HPC/NAS state and a stale .stage leftover from a
                # SIGKILLed previous run is safe to reclaim under the lock.
                _sweep_stale_stage_files(config)

            experiments = resolve_experiments(runner, config, requested)

            if args.command == "inventory":
                _emit_result(build_inventory(runner, config, experiments), as_json=args.json)
                return 0

            if args.command == "sync":
                results = sync(runner, config, experiments)
                _emit_result(results, as_json=args.json)
                return 0 if all(results.values()) else 1

            if args.command == "prune":
                results = prune(runner, config, experiments)
                _emit_result(results, as_json=args.json)
                return 0 if all(r["error"] is None for r in results.values()) else 1

            if args.command == "watch":
                return watch(runner, config, experiments, args.interval_seconds)

            parser.error(f"unknown command {args.command!r}")
            return 2  # unreachable: parser.error() exits, but keeps type checkers honest
    except MirrorError as exc:
        logger.error("%s", exc)
        return 2
    except ValueError as exc:
        logger.error("%s", exc)
        return 2
    except OSError as exc:
        # Issue #321: a defensive backstop, not the primary fix -- the known
        # site (local_nas_checkpoints, a stale CIFS handle) already wraps and
        # re-raises as MirrorError. This catches anything else that reaches
        # here as a bare OSError (a Python built-in, not a MirrorError
        # subclass) so it still exits 2 (connectivity/environment failure)
        # instead of an uncaught traceback and exit 1.
        logger.error("unexpected OS error: %s", exc)
        return 2


if __name__ == "__main__":
    sys.exit(main())
