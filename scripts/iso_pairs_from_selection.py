#!/usr/bin/env python3
"""
Single source of truth for the VGG16 iso-accuracy pair set.

Reads a frozen iso-accuracy selection JSON (as written by
``select_iso_accuracy_checkpoints.py``) and emits the canonical pair list in
whatever format a downstream driver needs, so no submit/stage/render/report
script ever hardcodes epoch numbers again.

The frozen selection for the current publication campaign lives at
``data/iso_accuracy_selection.frozen.json``.

Emitted formats (``--format``):
  pairs    : "<flextag> <vantag> <iso> <fe> <ve>" per line  (render PAIRS arrays)
  targets  : "<tag>|<exp>|<epoch>" per line, deduped        (submit TARGETS block)
  tags     : "<tag>" per line, deduped                      (stage / layers / extras)
  json     : the normalized pair rows as JSON                (debugging)

Library use:
  from iso_pairs_from_selection import load_pairs, Pair
  pairs = load_pairs(Path("data/iso_accuracy_selection.frozen.json"))

Pure stdlib.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

FLEX_EXP = "experiment-flex-vgg16-imagenet-dense"
VANILLA_EXP = "experiment-vanilla-vgg16-imagenet-dense"


@dataclass(frozen=True)
class Pair:
    """One iso-accuracy matched pair, normalized for every downstream driver."""

    iso: str            # 2-decimal accuracy label used in filenames, e.g. "0.53"
    flex_epoch: int
    vanilla_epoch: int
    flex_top1: float
    vanilla_top1: float
    delta: float        # |flex_top1 - vanilla_top1|

    @property
    def flextag(self) -> str:
        return f"vgg16-flex-e{self.flex_epoch}"

    @property
    def vantag(self) -> str:
        return f"vgg16-vanilla-e{self.vanilla_epoch}"

    @property
    def infix(self) -> str:
        """Filename infix used by comparison/manuscript PNGs."""
        return f"flexvggE{self.flex_epoch}_vs_vanillavggE{self.vanilla_epoch}"

    @property
    def short_label(self) -> str:
        return f"flexE{self.flex_epoch} / vanE{self.vanilla_epoch}"


def load_pairs(selection_path: Path) -> list[Pair]:
    """Parse the selection JSON into normalized, filename-ready Pair rows.

    Raises FileNotFoundError / json.JSONDecodeError / KeyError on bad input so
    callers fail loudly rather than silently rendering the wrong pairs.
    """
    data = json.loads(selection_path.read_text())
    raw = data["pairs"]
    if not raw:
        raise ValueError(f"selection {selection_path} contains no pairs")
    pairs: list[Pair] = []
    for p in raw:
        fe = int(p["flex_epoch"])
        ve = int(p["vanilla_epoch"])
        fa = float(p["flex_top1"])
        va = float(p["vanilla_top1"])
        # Label each pair by its matched accuracy, rounded to 2 dp. The selection
        # is monotone in accuracy, so labels are distinct across the 12 pairs.
        iso = f"{round(float(p['target_top1']), 2):.2f}"
        pairs.append(
            Pair(
                iso=iso,
                flex_epoch=fe,
                vanilla_epoch=ve,
                flex_top1=fa,
                vanilla_top1=va,
                delta=round(abs(fa - va), 5),
            )
        )
    # Guard the distinctness assumption the filename scheme relies on.
    labels = [p.iso for p in pairs]
    if len(set(labels)) != len(labels):
        dupes = sorted({lab for lab in labels if labels.count(lab) > 1})
        raise ValueError(
            f"non-unique iso labels {dupes} in {selection_path}; two pairs would "
            f"collide on the same filename. Widen the rounding in load_pairs()."
        )
    return pairs


def _emit(pairs: list[Pair], fmt: str) -> str:
    if fmt == "pairs":
        return "\n".join(
            f"{p.flextag} {p.vantag} {p.iso} {p.flex_epoch} {p.vanilla_epoch}"
            for p in pairs
        )
    if fmt == "targets":
        seen: set[str] = set()
        lines: list[str] = []
        for p in pairs:
            for tag, exp, ep in (
                (p.flextag, FLEX_EXP, p.flex_epoch),
                (p.vantag, VANILLA_EXP, p.vanilla_epoch),
            ):
                if tag not in seen:
                    seen.add(tag)
                    lines.append(f"{tag}|{exp}|{ep}")
        return "\n".join(lines)
    if fmt == "tags":
        seen_tags: set[str] = set()
        tag_lines: list[str] = []
        for p in pairs:
            for tag in (p.flextag, p.vantag):
                if tag not in seen_tags:
                    seen_tags.add(tag)
                    tag_lines.append(tag)
        return "\n".join(tag_lines)
    if fmt == "json":
        return json.dumps([asdict(p) for p in pairs], indent=2)
    raise ValueError(f"unknown format: {fmt}")


def default_selection() -> Path:
    """Repo-local frozen selection path (single campaign source of truth)."""
    repo_root = Path(__file__).resolve().parent.parent
    return repo_root / "data" / "iso_accuracy_selection.frozen.json"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("selection", nargs="?", type=Path, default=None,
                    help="selection JSON (default: data/iso_accuracy_selection.frozen.json)")
    ap.add_argument("--format", choices=["pairs", "targets", "tags", "json"],
                    default="pairs")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    sel = args.selection or default_selection()
    if not sel.is_file():
        print(f"[error] selection not found: {sel}", file=sys.stderr)
        return 2
    try:
        pairs = load_pairs(sel)
    except (ValueError, KeyError, json.JSONDecodeError) as e:
        print(f"[error] bad selection {sel}: {e}", file=sys.stderr)
        return 2
    print(_emit(pairs, args.format))
    return 0


if __name__ == "__main__":
    sys.exit(main())
