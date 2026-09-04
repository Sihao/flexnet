#!/usr/bin/env python3
"""
Run the GPU/ImageNet downstream analyses (adversarial attacks, frequency,
Hessian) on a single restored checkpoint directory, for the iso-accuracy
trajectory comparison.

The analysis functions load the *latest* checkpoint in the given run folder, so
the caller restores exactly one checkpoint_<epoch>.pth (+ configurations.json)
into ``run_dir`` before invoking this. Each analysis writes a marker file under
``run_dir/results/`` so re-runs skip completed work (idempotent).

Intended to run on the HPC inside the ``flexnet`` conda env, from the project
root (so relative data paths like ``data/...`` and ImageNet resolve).
"""
from __future__ import annotations

import argparse
import os
import sys
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# DEFAULT_ANALYSES is what a bare run (and the auto-cron) executes. "perturbation"
# (ImageNet-C corruption robustness) is available but OFF by default: it needs the
# ImageNet-C data staged and adds ~1-2h, so it is opt-in via `--analyses perturbation`.
DEFAULT_ANALYSES = ("attacks", "frequency", "hessian")
ANALYSES = ("attacks", "frequency", "hessian", "perturbation")


def run_one(name: str, run_dir: str, batch_size: int, device: str) -> bool:
    """Run a single analysis. Returns True on success."""
    if name == "attacks":
        from src.analysis.run_attack_comparison import run_attack_comparison

        run_attack_comparison(
            experiment_id=run_dir,
            batch_size=batch_size,
            # Match the original convnet exp2-vs-exp4 comparison: FGSM/Jitter/APGD
            # only. SPSA (query-based black-box) costs ~54 min/epsilon x20 ~= 18h at
            # 2000 samples and blew the 12h walltime mid-sweep (so the results JSON
            # was never written and frequency/hessian never ran); OnePixel is
            # similarly slow and also not in the target plot. Override via env if
            # ever needed. These three finish in ~2.5h, leaving time for the rest.
            attacks=os.environ.get("ATTACKS", "FGSM,Jitter,APGD").split(","),
            viz_filter=None,
            resume=True,
            # Full-ImageNet val is 50k images; a balanced subset keeps the
            # attack sweep tractable while still being representative.
            # Override with ATTACK_MAX_SAMPLES.
            max_samples=int(os.environ.get("ATTACK_MAX_SAMPLES", "2000")),
            seed=0,
        )
    elif name == "frequency":
        from src.analysis.run_frequency_analysis import run_frequency_analysis

        run_frequency_analysis(
            experiment_id=run_dir,
            batch_size=batch_size,
            device=device,
        )
    elif name == "hessian":
        from src.analysis.run_hessian_analysis import analyze_hessian_input

        analyze_hessian_input(
            exp_id=run_dir,
            batch_size=1,
            num_batches=4,
            m_steps=50,
            device=device,
        )
    elif name == "perturbation":
        from src.analysis.run_perturbation_analysis import run_perturbation_analysis

        run_perturbation_analysis(
            experiment_id=run_dir,
            imagenet_c_path=os.environ.get(
                "IMAGENET_C_PATH",
                "/lustre/fs8/huds_lab/scratch/slu/Data/imagenet_c",
            ),
            batch_size=64,
            device_str=device,
        )
    else:
        raise ValueError(f"unknown analysis '{name}'")
    return True


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True,
                   help="Restored run folder (contains checkpoints/ + configurations.json)")
    p.add_argument("--analyses", nargs="+", default=list(DEFAULT_ANALYSES),
                   choices=ANALYSES)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--device", default="cuda")
    p.add_argument("--force", action="store_true",
                   help="Re-run even if a completion marker exists.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir)
    if not (run_dir / "configurations.json").is_file():
        print(f"[error] no configurations.json in {run_dir}", file=sys.stderr)
        return 2
    if not list((run_dir / "checkpoints").glob("checkpoint_*.pth")):
        print(f"[error] no checkpoint in {run_dir}/checkpoints", file=sys.stderr)
        return 2

    marker_dir = run_dir / "results"
    marker_dir.mkdir(parents=True, exist_ok=True)

    failures = 0
    for name in args.analyses:
        marker = marker_dir / f".iso_done_{name}"
        if marker.exists() and not args.force:
            print(f"[skip] {name} already done ({marker})")
            continue
        print(f"[run] {name} on {run_dir} (device={args.device}) ...")
        try:
            run_one(name, str(run_dir), args.batch_size, args.device)
            marker.write_text("ok\n")
            print(f"[ok] {name} complete -> marker {marker}")
        except Exception as e:  # noqa: BLE001 - log and continue other analyses
            failures += 1
            print(f"[error] {name} failed: {e}", file=sys.stderr)
            traceback.print_exc()

    if failures:
        print(f"[done] {failures}/{len(args.analyses)} analyses failed", file=sys.stderr)
        return 1
    print("[done] all requested analyses complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
