#!/usr/bin/env python3
"""
Assemble per-tag brain-score sweep results into the JSONs that turn on Fig5
Panel C (chainlink #397).

Each VGG16 layer sweep job (scripts/submit_vgg16_brainscore_layers.sh, via
scripts/hpc/brain_score_checkpoint.py) writes ONE result file per tag,
already shaped as::

    {"<tag>": {"<raw_layer_key>": {"<benchmark_id>": <float>, ...}, ...}}

with one entry per swept layer (features.3, features.10, features.20,
features.30, features.40) and one entry per benchmark (the four
FreemanZiemba/MajajHong ids in BENCHMARKS below -- see
scripts/reproduce_manuscript_styled.py's BENCH dict, which this module
matches verbatim; no new region-name mapping is invented here).

IMPORTANT -- flex vs vanilla key naming: brain_score's resolve_layers() in
scripts/brain_score_checkpoint.py only matches torch.nn.Conv2d instances by
name. A vanilla VGG16 conv at position N is itself an nn.Conv2d, so it is
keyed by the bare name "features.N". A flex VGG16 conv at the same position
is a Flex2D wrapper (src/modules/models/VGG.py) whose actual nn.Conv2d
submodule is ".flex_conv" (src/modules/layers/flex.py:85), so the SAME
probed layer is keyed "features.N.flex_conv" in a flex tag's sweep result.
resolve_layer_key() below accepts either form and always STORES the result
under the bare canonical name, so a flex tag and a vanilla tag for the same
probed layer land on the same key -- exactly what
reproduce_manuscript_styled.py's layer_base()/resolve_final_layer() expect
on the read side (they strip the same ".flex_conv" suffix).

Two output modes, both writing role-keyed files named
flex-7L_brainscore.json / vanilla-7L_brainscore.json:

  * Multi-tag merge (default): every tag in --flex-tags / --vanilla-tags
    becomes its own top-level entry --
    {tag_1: {layer: {bench: value}}, tag_2: {...}, ...}.
    NOTE: reproduce_manuscript_styled.py's nested_vals() reads ONLY the
    FIRST top-level tag in a brain-score json (`next(iter(d.values()))`) --
    it has no tag selector. A multi-tag file therefore does NOT drive Fig5
    for every tag it contains; it is a convenient merged archive, useful
    for inspection or a future reader that does iterate every tag, but only
    the first-inserted tag's data would be picked up if fed to fig5() as-is.

  * Single-pair selection (--flex-pair-tag / --vanilla-pair-tag): restricts
    each output file to exactly the one matching tag, so the file's sole
    top-level entry is `{tag: {layer: {bench: value}}}` -- the precise
    shape nested_vals() consumes for one flex-vs-vanilla pair. This is the
    mode #398's per-pair rendering loop calls, once per iso pair.

A tag missing a swept layer entirely gets that layer OMITTED (never
zero-filled) from its entry, and the omission is logged so a silent gap
does not read as "scored zero correlation". A non-finite (inf/nan)
benchmark value is treated the same way: omitted and logged, never written
(json.dump uses allow_nan=False as a fail-fast backstop). Writes are atomic
(temp file + os.replace) and a role that resolves to zero usable tags
refuses to write at all, so a bad sweep can never silently clobber a good
existing output file with an empty one.

Usage:
    python scripts/assemble_7L_brainscore.py \\
        --input-dir /path/to/brainscore_hpc/results \\
        --flex-tags vgg16-flex-e32,vgg16-flex-e36,... \\
        --vanilla-tags vgg16-vanilla-e22,vgg16-vanilla-e25,... \\
        --out-dir /tmp/manuscript_repro

    python scripts/assemble_7L_brainscore.py \\
        --input-dir /path/to/brainscore_hpc/results \\
        --flex-pair-tag vgg16-flex-e32 --vanilla-pair-tag vgg16-vanilla-e22

    python scripts/assemble_7L_brainscore.py --self-test
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Mapping

LOG = logging.getLogger("assemble_7L_brainscore")

# Layers swept by scripts/submit_vgg16_brainscore_layers.sh.
LAYERS = ("features.3", "features.10", "features.20", "features.30", "features.40")

# Same benchmark ids used as BENCH's values in reproduce_manuscript_styled.py
# (lines 38-41) -- reproduced verbatim, not re-derived.
BENCHMARKS = (
    "FreemanZiemba2013.V1.public-pls",
    "FreemanZiemba2013.V2.public-pls",
    "MajajHong2015.public.V4-pls",
    "MajajHong2015.public.IT-pls",
)

DEFAULT_FLEX_TAGS = (
    "vgg16-flex-e32",
    "vgg16-flex-e36",
    "vgg16-flex-e40",
    "vgg16-flex-e44",
    "vgg16-flex-e48",
    "vgg16-flex-e52",
)
DEFAULT_VANILLA_TAGS = (
    "vgg16-vanilla-e22",
    "vgg16-vanilla-e25",
    "vgg16-vanilla-e28",
    "vgg16-vanilla-e29",
    "vgg16-vanilla-e40",
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT_DIR = Path("/tmp/manuscript_repro")
DEFAULT_PROJECT_COPY_DIR = REPO_ROOT / "__local__" / "manuscript_repro_vgg16"


class TagResultError(Exception):
    """Raised when a per-tag sweep result JSON cannot be read or parsed."""


def load_tag_result(input_dir: Path, tag: str) -> Mapping[str, Mapping[str, float]]:
    """Load one tag's {layer: {benchmark: value}} dict from <input_dir>/<tag>.json.

    Raises TagResultError with the tag and path in the message on any
    failure (missing file, bad JSON, unexpected top-level shape) so a
    caller can log a specific, actionable error instead of a bare
    traceback.
    """
    path = input_dir / f"{tag}.json"
    if not path.is_file():
        raise TagResultError(f"{tag}: no result file at {path}")
    try:
        raw_text = path.read_text()
    except OSError as exc:
        raise TagResultError(f"{tag}: cannot read {path}: {exc}") from exc
    try:
        doc = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise TagResultError(f"{tag}: invalid JSON in {path}: {exc}") from exc
    if not isinstance(doc, dict):
        raise TagResultError(
            f"{tag}: {path} top level is {type(doc).__name__}, expected an object"
        )
    if tag in doc:
        layers = doc[tag]
    elif len(doc) == 1:
        # Tolerate an exp-name that doesn't exactly match the filename stem
        # (e.g. a manual rename), as long as there is exactly one candidate.
        (only_key, layers), = doc.items()
        LOG.warning(
            "%s: top-level key %r != tag %r in %s; using it anyway "
            "(single-entry file)", tag, only_key, tag, path,
        )
    else:
        raise TagResultError(
            f"{tag}: {path} has no top-level key {tag!r} and is not a "
            f"single-entry file (keys: {list(doc)})"
        )
    if not isinstance(layers, dict):
        raise TagResultError(
            f"{tag}: {path}[{tag!r}] is {type(layers).__name__}, expected an object"
        )
    return layers


def resolve_layer_key(layers: Mapping[str, object], canonical: str) -> str | None:
    """Find the raw sweep-result key for a canonical layer name (e.g.
    "features.40"), accepting either the vanilla bare form or the flex
    ".flex_conv"-suffixed form (see the module docstring). Mirrors
    reproduce_manuscript_styled.py's layer_base(), which strips the same
    suffix on the read side, so this must accept exactly what that strips.

    Returns the raw key if found, else None. If more than one key matches
    the dotted-prefix form, the layer is ambiguous in this tag's data --
    skip it (return None) rather than guess which one is right.
    """
    if canonical in layers:
        return canonical
    prefix = canonical + "."
    matches = [k for k in layers if k.startswith(prefix)]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        LOG.warning(
            "layer %r is ambiguous in sweep result (candidates: %s) -- skipping",
            canonical, matches,
        )
    return None


def assemble_tag(input_dir: Path, tag: str) -> dict[str, dict[str, float]]:
    """Build one tag's {layer: {benchmark: value}} entry, restricted to the
    swept LAYERS and keyed by the BARE canonical layer name regardless of
    whether the raw result used the bare form (vanilla) or the
    ".flex_conv"-suffixed form (flex). A layer absent from the raw result,
    or present only with non-finite/invalid benchmark values, is OMITTED
    (never zero-filled) and logged.
    """
    layers = load_tag_result(input_dir, tag)
    entry: dict[str, dict[str, float]] = {}
    for canonical in LAYERS:
        raw_key = resolve_layer_key(layers, canonical)
        if raw_key is None:
            LOG.warning("%s: layer %r absent from sweep result -- omitting", tag, canonical)
            continue
        bench = layers[raw_key]
        if not isinstance(bench, dict) or not bench:
            LOG.warning(
                "%s: layer %r (raw key %r) has no usable benchmark data (%r) -- omitting",
                tag, canonical, raw_key, bench,
            )
            continue
        clean_bench: dict[str, float] = {}
        for bench_id, value in bench.items():
            try:
                fval = float(value)
            except (TypeError, ValueError):
                LOG.warning(
                    "%s: layer %r benchmark %r has non-numeric value %r -- omitting",
                    tag, canonical, bench_id, value,
                )
                continue
            if not math.isfinite(fval):
                LOG.warning(
                    "%s: layer %r benchmark %r is non-finite (%r) -- omitting",
                    tag, canonical, bench_id, value,
                )
                continue
            clean_bench[bench_id] = fval
        if not clean_bench:
            LOG.warning(
                "%s: layer %r (raw key %r) had only non-finite/invalid "
                "benchmark values -- omitting", tag, canonical, raw_key,
            )
            continue
        entry[canonical] = clean_bench
    if not entry:
        LOG.warning("%s: no layers scored at all -- entry will be empty", tag)
    return entry


def assemble_role(input_dir: Path, tags: list[str]) -> dict[str, dict[str, dict[str, float]]]:
    """Merge a role's (flex or vanilla) per-tag results into tag -> layer ->
    benchmark -> value. Missing tag files are logged and skipped (not
    fatal) so one bad tag doesn't block the rest of the sweep -- but the
    caller decides whether an empty overall result is acceptable.

    A tag whose file exists but resolves to zero usable layers (every layer
    absent/non-finite -- assemble_tag() returns {} for it) is EXCLUDED from
    the role entirely, never stored as `{tag: {}}` (chainlink #399): a
    hollow-but-present entry used to make the role dict truthy and slip
    past run()'s empty-role guard, silently clobbering a good existing
    output file with one containing that hollow tag.
    """
    out: dict[str, dict[str, dict[str, float]]] = {}
    for tag in tags:
        try:
            entry = assemble_tag(input_dir, tag)
        except TagResultError as exc:
            LOG.error("%s", exc)
            continue
        if not entry:
            LOG.error(
                "%s: resolved to zero usable layers -- excluding this tag "
                "from the output (not storing an empty %r: {})", tag, tag,
            )
            continue
        out[tag] = entry
    return out


def _role_has_content(role: dict[str, dict[str, dict[str, float]]]) -> bool:
    """True iff at least one tag in role has at least one non-empty layer
    mapping. A tag present with an empty ({}) entry must NOT count as
    content -- checking bare dict truthiness (`if not role`) is exactly
    what let `{tag: {}}` slip past the guard in chainlink #399. Checked
    independently of assemble_role()'s own filtering as defense in depth.
    """
    return any(layers for layers in role.values())


class EmptyAssemblyError(RuntimeError):
    """Raised when a role (flex or vanilla) resolved zero usable tags.

    Writing an empty {} over an existing, good output file would silently
    blank out prior good data (nested_vals() then StopIterations on the
    empty dict downstream) -- refuse instead, naming the input dir and the
    tags that were tried so the fix is obvious.
    """


def write_json_atomic(data: object, path: Path) -> None:
    """Write JSON to path atomically: build in a temp file in the same
    directory, then os.replace() it into place, so a crash mid-write can
    never leave a half-written or truncated file where a good one used to
    be. allow_nan=False is a fail-fast backstop -- assemble_tag() already
    filters non-finite values, so this should never actually trigger, but
    if it ever does, a stray inf/nan must be a hard error, not a silent
    write that later blows an axis downstream.

    tempfile.mkstemp() creates the temp file mode 0600, which os.replace()
    would otherwise carry straight through to the final path -- unreadable
    by a group collaborator on the shared lab HPC (chainlink #400). Restore
    the umask-masked 0644 a normal Path.write_text() would have produced.
    os.umask() can only be READ by setting it, so query-and-restore
    back-to-back (os.umask() itself is documented as not thread-safe; this
    script is single-threaded, so the brief window is not a concern here).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp",
                                     dir=str(path.parent))
    try:
        cur_umask = os.umask(0)
        os.umask(cur_umask)
        with os.fdopen(fd, "w") as fh:
            os.fchmod(fh.fileno(), 0o644 & ~cur_umask)
            json.dump(data, fh, indent=2, sort_keys=True, allow_nan=False)
        os.replace(tmp_name, path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise


def run(input_dir: Path, flex_tags: list[str], vanilla_tags: list[str],
        out_dir: Path, project_copy_dir: Path | None) -> tuple[Path, Path]:
    if not input_dir.is_dir():
        raise NotADirectoryError(f"--input-dir does not exist or is not a directory: {input_dir}")

    flex = assemble_role(input_dir, flex_tags)
    vanilla = assemble_role(input_dir, vanilla_tags)

    flex_path = out_dir / "flex-7L_brainscore.json"
    vanilla_path = out_dir / "vanilla-7L_brainscore.json"

    if not _role_has_content(flex):
        raise EmptyAssemblyError(
            f"flex role produced NO tags with usable layer data from "
            f"{input_dir} (tried: {flex_tags}); refusing to write an "
            f"empty/hollow {flex_path}"
        )
    if not _role_has_content(vanilla):
        raise EmptyAssemblyError(
            f"vanilla role produced NO tags with usable layer data from "
            f"{input_dir} (tried: {vanilla_tags}); refusing to write an "
            f"empty/hollow {vanilla_path}"
        )

    write_json_atomic(flex, flex_path)
    write_json_atomic(vanilla, vanilla_path)
    LOG.info("wrote %s (%d tags)", flex_path, len(flex))
    LOG.info("wrote %s (%d tags)", vanilla_path, len(vanilla))

    if project_copy_dir is not None:
        project_copy_dir.mkdir(parents=True, exist_ok=True)
        for src in (flex_path, vanilla_path):
            dst = project_copy_dir / src.name
            shutil.copy2(src, dst)
            LOG.info("copied %s -> %s (survives /tmp cleanup)", src, dst)

    return flex_path, vanilla_path


# --------------------------------------------------------------------- self-test
def _fabricate_sweep_results(input_dir: Path, tags: list[str], missing: dict[str, str],
                              key_suffix: str = "", inject_nonfinite: dict[str, str] | None = None) -> None:
    """Write synthetic per-tag sweep result JSONs into input_dir, one file
    per tag, in the exact shape brain_score_checkpoint.py produces.

    missing maps a tag to one canonical layer name to leave out of that
    tag's file entirely, to exercise the "absent, not zero" path.

    key_suffix is appended to every raw layer key ("" for vanilla's bare
    "features.N"; ".flex_conv" for flex's wrapped conv submodule -- see the
    module docstring), so this can fabricate BOTH shapes and prove the
    assembler resolves either one to the same bare canonical key.

    inject_nonfinite maps a tag to one canonical layer name whose first
    benchmark value is set to float("inf"), to exercise the non-finite
    filter.
    """
    inject_nonfinite = inject_nonfinite or {}
    for i, tag in enumerate(tags):
        skip_layer = missing.get(tag)
        nonfinite_layer = inject_nonfinite.get(tag)
        layers = {}
        for j, layer in enumerate(LAYERS):
            if layer == skip_layer:
                continue
            raw_key = f"{layer}{key_suffix}"
            bench = {
                bench: round(0.1 + 0.01 * i + 0.001 * j + 0.0001 * k, 6)
                for k, bench in enumerate(BENCHMARKS)
            }
            if layer == nonfinite_layer:
                bench[BENCHMARKS[0]] = float("inf")
            layers[raw_key] = bench
        (input_dir / f"{tag}.json").write_text(json.dumps({tag: layers}))


def self_test() -> None:
    with tempfile.TemporaryDirectory(prefix="assemble_7L_selftest_") as tmp:
        tmp_path = Path(tmp)
        input_dir = tmp_path / "sweep_results"
        input_dir.mkdir()
        out_dir = tmp_path / "manuscript_repro"
        project_copy_dir = tmp_path / "project_copy"

        flex_tags = list(DEFAULT_FLEX_TAGS[:6])
        vanilla_tags = list(DEFAULT_VANILLA_TAGS[:6]) if len(DEFAULT_VANILLA_TAGS) >= 6 else (
            list(DEFAULT_VANILLA_TAGS) + ["vgg16-vanilla-e99"]
        )
        # Deliberately drop one layer for one flex tag and one vanilla tag to
        # exercise the "absent, not zero" path on both output files, and
        # inject one non-finite benchmark value into a different flex tag to
        # exercise the finite filter.
        missing = {flex_tags[2]: "features.20", vanilla_tags[1]: "features.30"}
        nonfinite = {flex_tags[3]: "features.10"}
        # Flex tags are fabricated with the ".flex_conv"-suffixed raw keys a
        # REAL flex sweep produces; vanilla tags with the bare form. If
        # resolve_layer_key() regresses to an exact-match-only lookup, every
        # flex layer silently fails to resolve and this assertion block
        # fails (entries come back empty) -- this is the regression test for
        # finding #1.
        _fabricate_sweep_results(input_dir, flex_tags, missing, key_suffix=".flex_conv",
                                  inject_nonfinite=nonfinite)
        _fabricate_sweep_results(input_dir, vanilla_tags, missing, key_suffix="")

        flex_path, vanilla_path = run(
            input_dir=input_dir,
            flex_tags=flex_tags,
            vanilla_tags=vanilla_tags,
            out_dir=out_dir,
            project_copy_dir=project_copy_dir,
        )

        flex_doc = json.loads(flex_path.read_text())
        vanilla_doc = json.loads(vanilla_path.read_text())

        assert set(flex_doc) == set(flex_tags), (set(flex_doc), flex_tags)
        assert set(vanilla_doc) == set(vanilla_tags), (set(vanilla_doc), vanilla_tags)

        for role_name, doc, tags, miss in (
            ("flex", flex_doc, flex_tags, missing),
            ("vanilla", vanilla_doc, vanilla_tags, missing),
        ):
            for tag in tags:
                layer_entries = doc[tag]
                assert len(layer_entries) >= 2, (
                    f"{role_name} tag {tag} has only {len(layer_entries)} "
                    f"layer keys, need >= 2 to trip the multi_layer gate "
                    f"(if this is a flex tag: the .flex_conv suffix was not "
                    f"resolved -- finding #1 regressed)"
                )
                # Every stored key must be a BARE canonical name -- never
                # the raw ".flex_conv"-suffixed key -- so flex and vanilla
                # entries for the same probed layer line up under fig5's
                # layer_base()/resolve_final_layer().
                for layer_key in layer_entries:
                    assert layer_key in LAYERS, (
                        f"{role_name} tag {tag} stored non-canonical layer "
                        f"key {layer_key!r} (raw suffix leaked through)"
                    )
                for layer, bench in layer_entries.items():
                    for value in bench.values():
                        assert math.isfinite(value), (tag, layer, value)
                dropped = miss.get(tag)
                if dropped is not None:
                    assert dropped not in layer_entries, (
                        f"{role_name} tag {tag} should have layer {dropped!r} "
                        f"absent, found it present instead"
                    )

        # The injected inf was omitted, not written (finding #5): that
        # tag/layer must still be present (other benchmarks are fine) but
        # missing the poisoned benchmark id specifically.
        poisoned_tag = flex_tags[3]
        poisoned_bench = flex_doc[poisoned_tag]["features.10"]
        assert BENCHMARKS[0] not in poisoned_bench, (
            f"non-finite value for {BENCHMARKS[0]} was written instead of omitted"
        )
        assert len(poisoned_bench) == len(BENCHMARKS) - 1

        # Project copy landed too.
        assert (project_copy_dir / "flex-7L_brainscore.json").is_file()
        assert (project_copy_dir / "vanilla-7L_brainscore.json").is_file()

        # --- single-pair selection mode (finding #2): restricting each role
        # to one tag produces the exact single-top-level-entry shape
        # nested_vals() consumes for one flex-vs-vanilla pair. ---
        pair_out_dir = tmp_path / "manuscript_repro_pair"
        one_flex, one_vanilla = flex_tags[0], vanilla_tags[0]
        pair_flex_path, pair_vanilla_path = run(
            input_dir=input_dir,
            flex_tags=[one_flex],
            vanilla_tags=[one_vanilla],
            out_dir=pair_out_dir,
            project_copy_dir=None,
        )
        pair_flex_doc = json.loads(pair_flex_path.read_text())
        pair_vanilla_doc = json.loads(pair_vanilla_path.read_text())
        assert list(pair_flex_doc) == [one_flex], pair_flex_doc
        assert list(pair_vanilla_doc) == [one_vanilla], pair_vanilla_doc
        assert len(pair_flex_doc[one_flex]) >= 2, "pair-mode flex arm is hollow"

        # A completely missing tag file is logged, not fatal, and simply
        # absent from the merged output -- as long as at least one tag in
        # the role still resolves.
        ghost_tags = flex_tags + ["vgg16-flex-does-not-exist"]
        flex_path2, _ = run(
            input_dir=input_dir,
            flex_tags=ghost_tags,
            vanilla_tags=vanilla_tags,
            out_dir=out_dir,
            project_copy_dir=None,
        )
        flex_doc2 = json.loads(flex_path2.read_text())
        assert "vgg16-flex-does-not-exist" not in flex_doc2
        assert set(flex_tags) <= set(flex_doc2)

        # --- empty-role guard (finding #3): if EVERY tag in a role is
        # missing, refuse to write -- and the prior good output must
        # survive untouched (no clobber). ---
        flex_before = flex_path.read_text()
        vanilla_before = vanilla_path.read_text()
        try:
            run(
                input_dir=input_dir,
                flex_tags=["vgg16-flex-does-not-exist"],
                vanilla_tags=vanilla_tags,
                out_dir=out_dir,
                project_copy_dir=None,
            )
        except EmptyAssemblyError:
            pass
        else:
            raise AssertionError("expected EmptyAssemblyError when the whole flex role is missing")
        assert flex_path.read_text() == flex_before, "empty-role write clobbered the existing good flex file"
        assert vanilla_path.read_text() == vanilla_before

        # --- #399: a tag whose FILE exists but resolves to zero usable
        # layers (every benchmark non-finite here) must NOT produce a
        # hollow {tag: {}} entry that slips past the empty-role guard --
        # the role must be treated as empty and refuse to write, and the
        # existing good file must survive untouched. ---
        hollow_tag = "vgg16-flex-all-nonfinite"
        hollow_layers = {
            f"{layer}.flex_conv": {b: float("inf") for b in BENCHMARKS}
            for layer in LAYERS
        }
        (input_dir / f"{hollow_tag}.json").write_text(json.dumps({hollow_tag: hollow_layers}))

        flex_before2 = flex_path.read_text()
        try:
            run(
                input_dir=input_dir,
                flex_tags=[hollow_tag],
                vanilla_tags=vanilla_tags,
                out_dir=out_dir,
                project_copy_dir=None,
            )
        except EmptyAssemblyError:
            pass
        else:
            raise AssertionError(
                "expected EmptyAssemblyError when the only flex tag "
                "resolves to zero usable layers (finding #399)"
            )
        assert flex_path.read_text() == flex_before2, (
            "an all-hollow-tag flex role clobbered the existing good flex "
            "file (finding #399 regressed)"
        )

        # Same check in single-pair mode specifically -- this is exactly
        # the #398 per-pair call pattern the finding was reproduced against.
        pair_flex_before = pair_flex_path.read_text()
        try:
            run(
                input_dir=input_dir,
                flex_tags=[hollow_tag],
                vanilla_tags=[one_vanilla],
                out_dir=pair_out_dir,
                project_copy_dir=None,
            )
        except EmptyAssemblyError:
            pass
        else:
            raise AssertionError(
                "expected EmptyAssemblyError in single-pair mode when the "
                "flex tag resolves to zero usable layers (finding #399)"
            )
        assert pair_flex_path.read_text() == pair_flex_before, (
            "single-pair mode: a hollow-tag flex re-run clobbered the "
            "existing good pair file (finding #399 regressed)"
        )

        # --- #400: outputs (and their project copies) must be group/other
        # readable, not mode 0600 from tempfile.mkstemp() leaking through
        # os.replace() into the final path. ---
        for p in (flex_path, vanilla_path,
                  project_copy_dir / "flex-7L_brainscore.json",
                  project_copy_dir / "vanilla-7L_brainscore.json"):
            mode = p.stat().st_mode
            assert mode & 0o044, (
                f"{p} is not group/other readable (mode {oct(mode)}) -- "
                f"finding #400 regressed"
            )

        # A bad --input-dir raises a specific, actionable error.
        try:
            run(
                input_dir=tmp_path / "does_not_exist",
                flex_tags=flex_tags,
                vanilla_tags=vanilla_tags,
                out_dir=out_dir,
                project_copy_dir=None,
            )
        except NotADirectoryError:
            pass
        else:
            raise AssertionError("expected NotADirectoryError for a missing --input-dir")

    print("[self-test] PASS: 6 core tags/file, >=2 layer keys each (flex "
          ".flex_conv keys resolved to bare canonical names), finite "
          "values, missing layers/non-finite values recorded absent (not "
          "zero), single-pair selection shape correct, missing-tag / "
          "empty-role (no clobber) / hollow-tag-role (no clobber, #399) / "
          "bad-input-dir handled cleanly, outputs group-readable (#400)")


# --------------------------------------------------------------------------- CLI
def _csv(value: str) -> list[str]:
    return [t.strip() for t in value.split(",") if t.strip()]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", type=Path,
                     help="Directory of per-tag sweep result JSONs "
                          "(<tag>.json each, e.g. brainscore_hpc/results)")
    ap.add_argument("--flex-tags", type=_csv, default=list(DEFAULT_FLEX_TAGS),
                     help="Comma-separated flex tags (default: the 6 VGG16 "
                          "flex iso tags from submit_vgg16_brainscore_layers.sh)")
    ap.add_argument("--vanilla-tags", type=_csv, default=list(DEFAULT_VANILLA_TAGS),
                     help="Comma-separated vanilla tags (default: the 5 "
                          "matched VGG16 vanilla tags)")
    ap.add_argument("--flex-pair-tag", default=None,
                     help="Restrict the flex output file to just this ONE "
                          "tag (must be given together with "
                          "--vanilla-pair-tag). Overrides --flex-tags and "
                          "produces the single-top-level-entry shape "
                          "reproduce_manuscript_styled.py's nested_vals() "
                          "consumes for one flex-vs-vanilla pair.")
    ap.add_argument("--vanilla-pair-tag", default=None,
                     help="Restrict the vanilla output file to just this "
                          "ONE tag (must be given together with "
                          "--flex-pair-tag). See --flex-pair-tag.")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR,
                     help=f"Where to write flex-7L_brainscore.json / "
                          f"vanilla-7L_brainscore.json (default: {DEFAULT_OUT_DIR})")
    ap.add_argument("--project-copy-dir", type=str, default=str(DEFAULT_PROJECT_COPY_DIR),
                     help="Also copy both outputs here so they survive /tmp "
                          f"cleanup (default: {DEFAULT_PROJECT_COPY_DIR}; pass "
                          "an empty string to skip)")
    ap.add_argument("--self-test", action="store_true",
                     help="Run the built-in synthetic self-test and exit")
    ap.add_argument("-v", "--verbose", action="store_true")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="[%(levelname)s] %(message)s",
    )

    if args.self_test:
        self_test()
        return 0

    if args.input_dir is None:
        print("error: --input-dir is required (or pass --self-test)", file=sys.stderr)
        return 2

    if bool(args.flex_pair_tag) != bool(args.vanilla_pair_tag):
        print("error: --flex-pair-tag and --vanilla-pair-tag must be given "
              "together", file=sys.stderr)
        return 2

    flex_tags = [args.flex_pair_tag] if args.flex_pair_tag else args.flex_tags
    vanilla_tags = [args.vanilla_pair_tag] if args.vanilla_pair_tag else args.vanilla_tags

    project_copy_dir = Path(args.project_copy_dir) if args.project_copy_dir else None

    try:
        run(
            input_dir=args.input_dir,
            flex_tags=flex_tags,
            vanilla_tags=vanilla_tags,
            out_dir=args.out_dir,
            project_copy_dir=project_copy_dir,
        )
    except (NotADirectoryError, EmptyAssemblyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
