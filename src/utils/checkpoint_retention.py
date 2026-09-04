"""
Checkpoint retention policy for training runs.

`RunLoader.save_checkpoint()` (src/analysis/run_loader.py) writes one
checkpoint file per epoch and then must get rid of old ones, or a long run
fills the disk quota. Deleting a checkpoint that has no copy anywhere else
is data loss, so this module makes deletion conditional on proof that a
checkpoint has already been copied off the training node.

That proof is an archive manifest written by an external process (the
workstation mirror), read here and never written here.

Archive manifest
-----------------
Path: `RUN_FOLDER/.archive_state.json`, next to (NOT inside)
`RUN_FOLDER/checkpoints/`, where `RUN_FOLDER` is the run's `NNNNNN` folder
that already holds `checkpoints/` and `logs/`.

Expected shape:

    {
        "schema_version": 1,
        "experiment": "<name>",
        "archived_epochs": [0, 1, 2, ...],
        "archived_through_epoch": 40,
        "nas_root": "<path>",
        "updated_utc": "2026-08-06T12:00:00Z",
        "writer": "<hostname or process name>"
    }

`schema_version`, `experiment`, and `archived_epochs` are consulted below;
`nas_root`, `updated_utc`, and `writer` are informational only and never
read by this module. A manifest that is missing, unreadable (including
non-UTF-8 bytes or pathologically nested JSON), malformed, of an
unrecognised schema version, that names a different `experiment` than the
one implied by `RUN_FOLDER`'s parent directory (a copied or restored run
folder can carry a stale manifest describing a *different* run -- see
`read_archive_state`), or whose `archived_epochs` entries are not all plain
integers, is treated as "nothing is archived" -- never as "everything is
archived". Read failures must fail closed, because failing open would
delete the only copy of a checkpoint.

Config keys
-----------
Read from the run's `configurations.json` (a plain dict; see
`RunLoader._load_config`). Neither key needs to exist in
`configs/configurations.yml` -- this docstring is the source of truth for
their meaning and defaults:

    - `keep_latest_checkpoints` (int, default 2): always keep this many of
      the highest-epoch checkpoints on disk, independent of archive state.
    - `keep_every_n_epochs` (int, optional): milestone epochs (epoch > 0
      and epoch % N == 0) are always kept, independent of archive state.
      This is the existing milestone behaviour `save_checkpoint` had before
      archive-awareness was added; it still wins over everything else, so
      e.g. `keep_every_n_epochs: 1` (the dense-run configuration) keeps
      every epoch forever regardless of what is archived.

`configurations.json` is JSON, so either key can arrive as the wrong type
(a quoted number, `null`, a float, a bool, an out-of-range int) if the file
was hand-edited on the HPC. `prune_checkpoints` coerces both at the
boundary: an unusable `keep_latest_checkpoints` falls back to the default
of 2, an unusable `keep_every_n_epochs` falls back to `None`, and a
well-typed but non-positive `keep_latest` (0 or negative -- still
nonsensical, since a run with no checkpoint on disk cannot resume) skips
pruning for that cycle instead of guessing what value was intended. Every
fallback logs a warning naming the bad key and value; none of them can
raise -- see the "never raises" contract on `prune_checkpoints` itself.

Operational note, not a bug: with the live `keep_every_n_epochs: 1` setting
on the two dense runs that filled the disk quota on 2026-08-06
(`test_live_vgg_dense_configuration_deletes_nothing` pins this), the
`keep_every_n` guard is a *permanent no-op* -- every epoch is a milestone,
so nothing is ever deleted through this function for those two runs. The
identical epoch/archived state with the key merely *absent*
(`keep_every_n=None`) deletes forty checkpoints. The entire difference is
one hand-edited JSON value with no other guard behind it; be deliberate
before touching that key on a live run.
"""

import json
import re
from pathlib import Path
from typing import Optional

# [0-9], not \d: \d also matches non-ASCII Unicode decimal digits (e.g. the
# Arabic-Indic digit five), which int() parses without complaint and which
# would otherwise collide with an ASCII-named checkpoint for the same epoch.
_CHECKPOINT_NAME_RE = re.compile(r"checkpoint_([0-9]+)\.pth")

# An epoch number above this is not something this trainer could plausibly
# have written; a stray file this large in its name must not be allowed to
# sort to the top of the epoch list and consume a keep_latest protection
# slot meant for a real recent checkpoint.
_MAX_PLAUSIBLE_EPOCH = 1_000_000

_DEFAULT_KEEP_LATEST = 2


def read_archive_state(run_folder: Path) -> set[int]:
    """
    Read the workstation mirror's archive manifest for a run.

    Never raises. Any problem with the manifest -- missing, unreadable
    (including non-UTF-8 bytes or pathologically nested JSON), not valid
    JSON, wrong schema version, naming a different `experiment` than this
    run folder, missing `archived_epochs`, or a non-integer entry in
    `archived_epochs` -- degrades to "nothing is archived" (the empty
    set), logged as one warning line.

    Parameters:
        - run_folder (Path): the run's root directory (the one containing
          `checkpoints/` and `logs/`), i.e. `RunLoader.run_folder`.

    Returns:
        - set[int]: epoch numbers the manifest confirms are archived
          elsewhere.
    """
    run_folder = Path(run_folder)
    manifest_path = run_folder / ".archive_state.json"

    try:
        with manifest_path.open("r", encoding="utf-8", errors="replace") as fh:
            data = json.load(fh)
    except FileNotFoundError:
        print(f"[WARNING] archive manifest not found at {manifest_path}; treating as nothing archived")
        return set()
    except json.JSONDecodeError as e:
        print(f"[WARNING] archive manifest at {manifest_path} is not valid JSON ({e}); treating as nothing archived")
        return set()
    except OSError as e:
        print(f"[WARNING] archive manifest at {manifest_path} could not be read ({e}); treating as nothing archived")
        return set()
    except Exception as e:
        # Structural catch-all rather than a growing list of remembered
        # exception types: a non-UTF-8 byte raises UnicodeDecodeError (a
        # ValueError subclass, #326); a pathologically nested manifest
        # raises RecursionError or MemoryError while json.load parses it
        # (#337). Whichever it is, this function's one job is to never let
        # it escape into prune_checkpoints -> save_checkpoint -> the
        # training loop.
        print(
            f"[WARNING] archive manifest at {manifest_path} could not be parsed ({e!r}); "
            "treating as nothing archived"
        )
        return set()

    if not isinstance(data, dict):
        print(f"[WARNING] archive manifest at {manifest_path} is not a JSON object; treating as nothing archived")
        return set()

    schema_version = data.get("schema_version")
    if type(schema_version) is not int or schema_version != 1:
        # type(...) is int, not isinstance(...), deliberately: isinstance
        # would accept bool (a int subclass, and True == 1) and would still
        # accept the value 1.0 == 1 if isinstance(..., int) were used
        # instead of a value check. CONTRACT 3 specifies a plain int.
        print(
            f"[WARNING] archive manifest at {manifest_path} has unsupported schema_version "
            f"{schema_version!r} (expected 1); treating as nothing archived"
        )
        return set()

    experiment = data.get("experiment")
    expected_experiment = run_folder.parent.name
    if not isinstance(experiment, str) or experiment != expected_experiment:
        # Fail closed, not open: a missing/non-string experiment is treated
        # the same as a mismatched one. A copied or NAS-restored run folder
        # can carry a stale manifest describing a *different* run (#328);
        # trusting it would authorise deleting checkpoints that were never
        # mirrored under this run at all.
        print(
            f"[WARNING] archive manifest at {manifest_path} has experiment {experiment!r}, expected "
            f"{expected_experiment!r} for this run folder; treating as nothing archived"
        )
        return set()

    archived_epochs = data.get("archived_epochs")
    if not isinstance(archived_epochs, list):
        print(
            f"[WARNING] archive manifest at {manifest_path} is missing an 'archived_epochs' list; "
            "treating as nothing archived"
        )
        return set()

    archived: set[int] = set()
    for entry in archived_epochs:
        # bool is a subclass of int in Python; True/False are not epochs.
        if isinstance(entry, bool) or not isinstance(entry, int):
            print(
                f"[WARNING] archive manifest at {manifest_path} has a non-integer entry in "
                f"'archived_epochs' ({entry!r}); treating as nothing archived"
            )
            return set()
        archived.add(entry)

    return archived


def select_checkpoints_to_delete(
    epochs: list[int],
    archived: set[int],
    keep_latest: int,
    keep_every_n: Optional[int],
    current_epoch: int,
) -> list[int]:
    """
    Decide which on-disk checkpoint epochs are safe to delete.

    Does not touch the filesystem. An epoch is deletable only when all of
    these hold:
        a. it is not `current_epoch` (just written, cannot be archived yet)
        b. it is not among the `keep_latest` highest epochs in `epochs`
        c. it is present in `archived`
        d. it is not a milestone: `keep_every_n` is not a positive int
           dividing it, or the epoch is 0 (0 gets no free pass from the
           `% keep_every_n == 0` identity)

    Parameters:
        - epochs (list[int]): epoch numbers of the checkpoints on disk.
        - archived (set[int]): epoch numbers confirmed archived elsewhere,
          e.g. the return value of `read_archive_state`.
        - keep_latest (int): always keep this many of the highest epochs.
          Must be a positive integer -- a run with no checkpoint on disk
          cannot resume.
        - keep_every_n (Optional[int]): if a positive integer, epochs > 0
          divisible by it are kept regardless of archive state.
        - current_epoch (int): the epoch just written; always kept.

    Returns:
        - list[int]: epochs safe to delete, sorted ascending.

    Raises:
        - ValueError: if `keep_latest` is not a positive integer.
    """
    if keep_latest <= 0:
        raise ValueError(f"keep_latest must be a positive integer (need a checkpoint to resume), got {keep_latest}")

    # De-duplicate BEFORE slicing and BEFORE iterating (#333): slicing the
    # raw (possibly repeated) epoch list first lets a repeated high epoch
    # consume more than one protection slot, shrinking effective keep_latest
    # below what was asked for; iterating the raw list can also return the
    # same epoch to delete more than once, violating the sorted-unique
    # postcondition documented below.
    unique_epochs_desc = sorted(set(epochs), reverse=True)
    protected_latest = set(unique_epochs_desc[:keep_latest])

    to_delete = []
    for ep in sorted(set(epochs)):
        if ep == current_epoch:
            continue
        if ep in protected_latest:
            continue
        if ep not in archived:
            continue
        if keep_every_n and keep_every_n > 0 and ep > 0 and ep % keep_every_n == 0:
            continue
        to_delete.append(ep)

    return sorted(to_delete)


def _coerce_int_config(value, key: str, default):
    """
    Coerce a JSON config value for `key` to `int`, falling back to
    `default` -- logging one warning naming the bad key and value -- for
    anything that is not unambiguously an integer.

    `bool` is rejected even though it is an `int` subclass in Python: a
    JSON `true`/`false` is never a sane checkpoint count. Range validity
    (e.g. `keep_latest <= 0`) is deliberately NOT checked here -- whether a
    well-typed value makes sense is a policy question for
    `select_checkpoints_to_delete`, not a type question for this helper.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        print(f"[WARNING] {key}={value!r} is not a valid number; using default {default!r}")
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        print(f"[WARNING] {key}={value!r} cannot be parsed as an integer; using default {default!r}")
        return default


def prune_checkpoints(
    ckpt_dir: Path,
    run_folder: Path,
    keep_latest: int,
    keep_every_n: Optional[int],
    current_epoch: int,
) -> list[Path]:
    """
    Delete checkpoints that are archived, unprotected, and safe to lose.

    Globs `checkpoint_<digits>.pth` in `ckpt_dir`, reads the archive
    manifest via `read_archive_state`, applies
    `select_checkpoints_to_delete`, and removes the resulting files. A
    filename that does not match `checkpoint_<digits>.pth` exactly --
    including the atomic-write temp file `.checkpoint_<N>.pth.tmp`, a
    non-canonical digit string (e.g. a leading zero), a directory, or an
    epoch number past a sane ceiling -- is never parsed and never removed.

    Never raises. `save_checkpoint` (src/analysis/run_loader.py) calls
    this immediately after writing a checkpoint with no try/except of its
    own, and nothing above it in the training loop catches an exception
    either -- see the module docstring for what that costs. `keep_latest`
    and `keep_every_n` are config-derived and may arrive as any JSON type;
    both are coerced to a usable value before use, falling back to the
    documented default (with a warning) on anything unusable. A missing
    `ckpt_dir`, a config value `select_checkpoints_to_delete` still rejects
    after coercion, a failed individual file removal, or any other
    unexpected problem is logged and treated as "nothing more to prune
    this cycle"; none of it propagates.

    Parameters:
        - ckpt_dir (Path): directory containing `checkpoint_<N>.pth` files.
        - run_folder (Path): the run's root directory, forwarded to
          `read_archive_state` (the manifest lives beside `checkpoints/`,
          not inside it).
        - keep_latest (int): see `select_checkpoints_to_delete`.
        - keep_every_n (Optional[int]): see `select_checkpoints_to_delete`.
        - current_epoch (int): see `select_checkpoints_to_delete`.

    Returns:
        - list[Path]: the checkpoint files actually removed. Empty on any
          failure -- deleting nothing is always a safe outcome.
    """
    ckpt_dir = Path(ckpt_dir)

    try:
        if not ckpt_dir.is_dir():
            # Path.glob() raises FileNotFoundError on a missing directory;
            # save_checkpoint() creates ckpt_dir before calling this, but a
            # future caller is not guaranteed to (#341 TWO).
            print(f"[WARNING] checkpoint directory {ckpt_dir} does not exist; nothing to prune")
            return []

        keep_latest = _coerce_int_config(keep_latest, "keep_latest_checkpoints", _DEFAULT_KEEP_LATEST)
        if keep_every_n is not None:
            # Unlike keep_latest, None is keep_every_n's own valid "no
            # milestone policy" default -- do not warn about it.
            keep_every_n = _coerce_int_config(keep_every_n, "keep_every_n_epochs", None)

        epoch_to_path: dict[int, Path] = {}
        for f in ckpt_dir.glob("checkpoint_*.pth"):
            match = _CHECKPOINT_NAME_RE.fullmatch(f.name)
            if match is None:
                continue
            digits = match.group(1)
            epoch = int(digits)
            if str(epoch) != digits:
                # Rejects a leading zero ("007" != "7") and any surviving
                # non-ASCII digit string outright instead of letting it
                # collide with the canonical name for the same epoch
                # (#341 FOUR).
                print(f"[WARNING] skipping {f.name}: non-canonical epoch digits {digits!r}")
                continue
            if epoch > _MAX_PLAUSIBLE_EPOCH:
                print(f"[WARNING] skipping {f.name}: epoch {epoch} exceeds plausible ceiling {_MAX_PLAUSIBLE_EPOCH}")
                continue
            if f.is_dir():
                # A directory cannot be a checkpoint; do not let it occupy
                # a keep_latest protection slot (#332).
                print(f"[WARNING] skipping {f.name}: is a directory, not a checkpoint file")
                continue
            epoch_to_path[epoch] = f

        archived = read_archive_state(run_folder)

        try:
            to_delete = select_checkpoints_to_delete(
                epochs=sorted(epoch_to_path.keys()),
                archived=archived,
                keep_latest=keep_latest,
                keep_every_n=keep_every_n,
                current_epoch=current_epoch,
            )
        except (ValueError, TypeError) as e:
            # select_checkpoints_to_delete is a pure function and is right
            # to reject a nonsensical keep_latest (e.g. 0 or negative,
            # still possible after coercion above); the bug being fixed is
            # this caller treating that rejection as fatal (#327).
            print(
                f"[WARNING] checkpoint retention policy rejected keep_latest={keep_latest!r} "
                f"keep_every_n={keep_every_n!r} ({e}); skipping prune this cycle"
            )
            return []

        removed = []
        total_freed = 0
        for ep in to_delete:
            f = epoch_to_path[ep]
            try:
                # lstat/unlink, not stat: a dangling symlink must be
                # removable without its target existing. stat() follows
                # the link, raises FileNotFoundError on a dangling one, and
                # the entry is then skipped forever (#341 ONE).
                freed = f.lstat().st_size
                f.unlink()
            except OSError as e:
                print(f"[WARNING] failed to remove checkpoint {f.name}: {e}")
                continue
            removed.append(f)
            total_freed += freed
            print(f"[INFO] pruned checkpoint {f.name} (epoch {ep}), freed {freed} bytes")

        print(f"[INFO] checkpoint prune complete: removed {len(removed)} file(s), freed {total_freed} bytes total")
        return removed
    except Exception as e:
        # Final defence in depth: retention is housekeeping and must never
        # be able to end a run that has just checkpointed successfully.
        print(f"[WARNING] checkpoint prune failed unexpectedly ({e!r}); nothing pruned this cycle")
        return []
