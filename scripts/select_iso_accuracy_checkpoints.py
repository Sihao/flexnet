#!/usr/bin/env python3
"""
Select iso-accuracy matched checkpoint pairs for the flex-vs-vanilla
downstream comparison.

Reads the per-epoch full-validation results (``full_val_epoch*.json``) for both
models, finds the overlapping top-1 accuracy band, picks N evenly spaced target
accuracies inside it, and for each target selects the epoch of each model whose
full-val top-1 is closest. The result is N matched (flex_epoch, vanilla_epoch)
pairs at (approximately) equal accuracy, plus their checkpoint paths.

Exit codes:
  0  selection written (ready)
  3  not enough data / no usable overlap yet (cron should retry later)
  2  usage / IO error
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional


def load_trajectory(logs_dir: Path) -> list[tuple[int, float]]:
    """Return sorted [(epoch, top1), ...] from full_val_epoch*.json files."""
    points: dict[int, float] = {}
    for f in logs_dir.glob("full_val_epoch*.json"):
        try:
            d = json.loads(f.read_text())
            ep = int(d["epoch"])
            top1 = float(d["top1"])
        except (ValueError, KeyError, json.JSONDecodeError) as e:
            print(f"[warn] skipping unreadable {f.name}: {e}", file=sys.stderr)
            continue
        points[ep] = top1
    return sorted(points.items())


def nearest(traj: list[tuple[int, float]], target: float) -> tuple[int, float]:
    """Epoch whose top1 is closest to target."""
    return min(traj, key=lambda ep_acc: abs(ep_acc[1] - target))


def select_pairs(
    flex: list[tuple[int, float]],
    vanilla: list[tuple[int, float]],
    n: int,
    band_floor: float = 0.40,
) -> Optional[list[dict]]:
    """N iso-accuracy matched pairs across the overlapping accuracy band.

    The band is anchored at max(band_floor, min-overlap): near-chance / barely
    trained checkpoints are excluded because downstream metrics (brain-score,
    robustness, frequency, Hessian) are noise-dominated there and the
    flex-vs-vanilla difference is uninterpretable. Set band_floor=0 to span the
    full overlap including low-accuracy points.
    """
    if len(flex) < 2 or len(vanilla) < 2:
        return None
    flex_acc = [a for _, a in flex]
    van_acc = [a for _, a in vanilla]
    lo = max(min(flex_acc), min(van_acc), band_floor)
    hi = min(max(flex_acc), max(van_acc))
    if not (hi > lo):
        return None
    pairs = []
    for i in range(n):
        # evenly spaced inclusive of endpoints
        frac = i / (n - 1) if n > 1 else 0.0
        target = lo + frac * (hi - lo)
        fe, fa = nearest(flex, target)
        ve, va = nearest(vanilla, target)
        pairs.append(
            {
                "target_top1": round(target, 5),
                "flex_epoch": fe,
                "flex_top1": round(fa, 5),
                "vanilla_epoch": ve,
                "vanilla_top1": round(va, 5),
                "acc_gap": round(abs(fa - va), 5),
            }
        )
    return pairs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--flex-logs", required=True, type=Path,
                   help="Dir with flex full_val_epoch*.json")
    p.add_argument("--vanilla-logs", required=True, type=Path,
                   help="Dir with vanilla full_val_epoch*.json")
    p.add_argument("--flex-ckpt-dir", required=True, type=Path,
                   help="Dir holding flex checkpoint_<epoch>.pth")
    p.add_argument("--vanilla-ckpt-dir", required=True, type=Path,
                   help="Dir holding vanilla checkpoint_<epoch>.pth")
    p.add_argument("--n", type=int, default=10, help="Number of matched pairs.")
    p.add_argument("--band-floor", type=float, default=0.40,
                   help="Lower accuracy bound for matched pairs (default 0.40; "
                        "near-chance points are noise-dominated). Set 0 to span full overlap.")
    p.add_argument("--max-gap", type=float, default=0.015,
                   help="Warn if any matched pair's |acc gap| exceeds this (default 0.015).")
    p.add_argument("--out", required=True, type=Path,
                   help="Output JSON path for the selection.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    for d in (args.flex_logs, args.vanilla_logs):
        if not d.is_dir():
            print(f"[error] logs dir not found: {d}", file=sys.stderr)
            return 2

    flex = load_trajectory(args.flex_logs)
    vanilla = load_trajectory(args.vanilla_logs)
    print(f"[info] flex points={len(flex)} vanilla points={len(vanilla)}")

    pairs = select_pairs(flex, vanilla, args.n, band_floor=args.band_floor)
    if pairs is None:
        print("[not-ready] insufficient trajectory overlap above band_floor "
              f"({args.band_floor}); retry later", file=sys.stderr)
        return 3

    # attach checkpoint paths and flag missing ones
    ready = True
    for pr in pairs:
        fp = args.flex_ckpt_dir / f"checkpoint_{pr['flex_epoch']}.pth"
        vp = args.vanilla_ckpt_dir / f"checkpoint_{pr['vanilla_epoch']}.pth"
        pr["flex_ckpt"] = str(fp)
        pr["vanilla_ckpt"] = str(vp)
        pr["flex_ckpt_exists"] = fp.is_file()
        pr["vanilla_ckpt_exists"] = vp.is_file()
        if not (fp.is_file() and vp.is_file()):
            ready = False
        if pr["acc_gap"] > args.max_gap:
            print(f"[warn] target {pr['target_top1']}: acc gap "
                  f"{pr['acc_gap']} > {args.max_gap} (coarse trajectory)",
                  file=sys.stderr)

    n_distinct_flex = len({p["flex_epoch"] for p in pairs})
    n_distinct_van = len({p["vanilla_epoch"] for p in pairs})
    if n_distinct_flex < args.n or n_distinct_van < args.n:
        print(f"[warn] duplicate epochs after matching "
              f"(flex distinct={n_distinct_flex}, vanilla={n_distinct_van}); "
              f"trajectory too coarse for {args.n} distinct points",
              file=sys.stderr)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"n": args.n, "pairs": pairs}, indent=2))
    print(f"[ok] wrote {args.out} ({len(pairs)} pairs, "
          f"all_ckpts_present={ready})")
    for pr in pairs:
        print(f"  acc~{pr['target_top1']:.3f} | "
              f"flex e{pr['flex_epoch']}={pr['flex_top1']:.3f} | "
              f"vanilla e{pr['vanilla_epoch']}={pr['vanilla_top1']:.3f} | "
              f"gap={pr['acc_gap']:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
