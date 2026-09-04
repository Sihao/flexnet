import errno
import json
from pathlib import Path

import pandas as pd


class SimpleLogger:
    """Append-only JSON-lines metric logger with best-effort persistence.

    A metrics line failing to write (e.g. a quota-exceeded disk) must never
    take down a training run that has already checkpointed the epoch, so
    `log` degrades gracefully on a full filesystem instead of raising.
    """

    def __init__(self, filename):
        self.filename = Path(filename)
        self.data = []
        self._pending = []  # entries dropped by a full-disk write, oldest first,
        # stored as already-serialised JSON line strings (see `log`).
        self._dropped_count = 0
        self._drops_since_flush = 0  # resets when the backlog fully drains (#307)
        try:
            self.filename.parent.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            if e.errno in (errno.EDQUOT, errno.ENOSPC):
                print(
                    f"[SimpleLogger] WARNING: cannot create log directory "
                    f"{self.filename.parent} ({e}); logging will stay in-memory "
                    "only for this process."
                )
            else:
                print(f"[SimpleLogger] ERROR: unexpected OSError creating {self.filename.parent}: {e}")
                raise
        self._jsonl_path = self._to_jsonl_path(self.filename)

    @staticmethod
    def _to_jsonl_path(path: Path) -> Path:
        """Return the ``.jsonl`` path used to persist `path`.

        `Path.with_suffix(".jsonl")` replaces everything after the LAST dot
        of the final path component, not just a real extension. A name that
        already contains a dot for another reason (e.g. an embedded date,
        "run.2026-08-06") silently loses that part of the name, and two
        differently-named logs can then collide on one file
        ("run.2026-08-06" and "run.2026-08-07" would both become
        "run.jsonl").

        Rule: append ".jsonl" to the full file name via `with_name`, so no
        part of the name is ever dropped -- except the one legacy suffix
        ".db" (the live construction site passes "metrics.db"), which is
        stripped first so it keeps resolving to the "metrics.jsonl" files
        already on disk. Any other suffix, dotted or not, is preserved.

        Edge cases handled on top of that rule:
          - A name that is already ``*.jsonl`` is returned unchanged instead
            of doubling the suffix ("metrics.jsonl" -> "metrics.jsonl", not
            "metrics.jsonl.jsonl").
          - The legacy ".db" suffix is matched case-insensitively, so
            "metrics.DB" resolves the same as "metrics.db".
          - A path with no usable file name (``Path("")``, ``Path(".")``,
            ``Path("/")``) raises `ValueError` up front with a message that
            names the problem, instead of the more cryptic
            "PosixPath('.') has an empty name" raised deep inside
            `Path.with_name`.
        """
        if not path.name:
            raise ValueError(f"SimpleLogger: filename {str(path)!r} has no usable file name")
        if path.suffix == ".jsonl":
            return path
        name = path.stem if path.suffix.lower() == ".db" else path.name
        return path.with_name(name + ".jsonl")

    def _truncate_to(self, size: int):
        """Best-effort: cut the jsonl file back to `size` bytes.

        Called after a failed write so a partial line never survives (see
        `_write_line`). If the truncate itself fails, there is nothing more
        we can safely do about the on-disk state -- report it loudly and
        move on. `get_dataframe` tolerates an unreadable line as a second
        line of defence for exactly this residual case.
        """
        try:
            with open(self._jsonl_path, "r+") as f:
                f.truncate(size)
        except OSError as e:
            print(
                f"[SimpleLogger] ERROR: could not truncate {self._jsonl_path} back to "
                f"{size} bytes after a failed write ({e}); the file may contain a partial line."
            )

    def _write_line(self, line: str):
        """Append one already-serialised JSON line (with trailing newline).

        Returns None on success. On a full disk / exceeded quota (EDQUOT or
        ENOSPC) returns the OSError instead of raising, so the caller can
        queue the line for a later retry. Any other OSError is unexpected,
        so it is reported and re-raised -- a real IO problem should stay
        visible rather than be silently swallowed.

        A write that fails partway through (e.g. hitting the quota mid
        flush) can leave a truncated fragment already on disk, which then
        breaks every later JSON reader permanently. Record the file size
        before attempting the write and, on ANY failure, truncate back to
        that size first so a torn fragment never survives on disk.
        """
        try:
            pre_size = self._jsonl_path.stat().st_size
        except OSError:
            pre_size = 0

        try:
            with open(self._jsonl_path, "a") as f:
                f.write(line)
            return None
        except OSError as e:
            self._truncate_to(pre_size)
            if e.errno in (errno.EDQUOT, errno.ENOSPC):
                return e
            print(f"[SimpleLogger] ERROR: unexpected OSError writing to {self._jsonl_path}: {e}")
            raise

    def log(self, entry: dict):
        # Retry any backlog first so a run recovers on its own once space is
        # freed, instead of staying stuck behind older dropped entries.
        self.flush_pending()

        # Serialise NOW, once, and queue the resulting (immutable) string --
        # never the caller's dict. Callers may mutate-and-reuse the same
        # dict across calls (e.g. a training loop's per-epoch metrics dict
        # updated in place), so queuing the dict itself means every pending
        # entry silently becomes an alias of whatever the caller last wrote
        # into it; recovery would then flush N copies of the LAST epoch
        # instead of the N distinct epochs that were actually dropped. A
        # freshly rendered string has no such aliasing hazard.
        #
        # This also means a value that can never be serialised (e.g. a raw
        # torch.Tensor) fails right here, once, for this one entry -- not
        # later, repeatedly, inside flush_pending, where it would wedge
        # every entry queued behind it.
        try:
            line = json.dumps(entry) + "\n"
        except (TypeError, ValueError) as e:
            print(f"[SimpleLogger] WARNING: dropping entry that is not JSON-serialisable ({e}): {entry!r}")
            return

        # `self.data` keeps the caller's dict by reference, same as before
        # this fix. That is a pre-existing aliasing hazard for in-process
        # readers of `self.data` if the caller mutates-and-reuses one dict
        # across calls -- unrelated to the on-disk/pending-queue duplication
        # this fix closes, and out of scope here (see the caller in
        # src/training/train.py). `get_dataframe` does not read `self.data`,
        # so it is unaffected either way.
        self.data.append(entry)

        error = self._write_line(line)
        if error is not None:
            self._pending.append(line)
            self._dropped_count += 1
            self._drops_since_flush += 1
            # Loud on the first drop of a fresh outage, then only every
            # 100th after that -- a full disk should be visible, not spam a
            # line per epoch. `_drops_since_flush` (unlike `_dropped_count`,
            # the lifetime total) resets to 0 once the backlog fully drains,
            # so a SECOND, later outage warns immediately instead of staying
            # silent until the lifetime count next happens to hit a
            # multiple of 100.
            if self._drops_since_flush == 1 or self._drops_since_flush % 100 == 0:
                print(
                    f"[SimpleLogger] WARNING: cannot write to {self._jsonl_path} ({error}); "
                    f"dropped {self._dropped_count} entries so far ({self._drops_since_flush} since "
                    "the last successful flush). Entries are kept in memory and flush_pending() "
                    "retries them once space is freed."
                )

    def flush_pending(self):
        """Retry writing entries dropped by a previous full-disk error.

        Writes the backlog in order (oldest first) and stops at the first
        failure, so a still-full disk leaves the remainder queued -- nothing
        lost, nothing duplicated -- instead of raising.
        """
        while self._pending:
            error = self._write_line(self._pending[0])
            if error is not None:
                return
            self._pending.pop(0)
        # The backlog is fully drained: a NEW outage should be loud again
        # right away (issue #307).
        self._drops_since_flush = 0

    def add_metric_entry(self, entry: dict):
        # Alias for log if needed, but we will revert train.py to use log()
        self.log(entry)

    def show_last_row(self):
        if self.data:
            print(self.data[-1])

    def get_dataframe(self):
        # Read the rows that made it to disk, then append any still-pending
        # entries (dropped by a write failure, not yet flushed) so a caller
        # mid-outage still sees every entry logged so far in this process --
        # `self._pending` holds entries that by definition never landed on
        # disk, so this cannot duplicate a row that did (issue #306).
        data = []
        if self._jsonl_path.exists():
            with open(self._jsonl_path, "r") as f:
                for lineno, line in enumerate(f, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        # A write that failed mid-flush can leave a torn
                        # fragment on disk despite the truncate-on-failure
                        # guard in `_write_line` (e.g. the truncate itself
                        # also failed). One unreadable line must not make
                        # the whole run's metrics unreadable (issue #302).
                        print(
                            f"[SimpleLogger] WARNING: skipping unreadable line {lineno} in "
                            f"{self._jsonl_path}: {e}"
                        )
        for line in self._pending:
            data.append(json.loads(line))
        return pd.DataFrame(data)

    def save_as_pandas_dataframe(self, save_dir):
        df = self.get_dataframe()
        df.to_csv(save_dir, index=False)
