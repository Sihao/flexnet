#!/usr/bin/env python3
"""
Draft Fig4C principled probe selection: find the TWO validation images with
the most obvious baseline-vs-flex curvature difference.

Protocol (replaces "two randomly selected validation images"):
  1. Gate-scan seeded-random validation candidates (plus forced known probes)
     until POOL_SIZE images pass probe_sanity for BOTH models -- the same
     correct-classification gate as the manuscript-extras pipeline.
  2. For every pooled image and both models, compute the top-2 input-Hessian
     eigenvalues/eigenvectors (compute_top_eigenvectors, float64 -- the same
     math the loss surface uses).
  3. Rank images by the curvature ratio
         score = (lam1+lam2)_baseline / (lam1+lam2)_flex
     restricted to images whose baseline curvature is at or above the pool
     median (a large ratio over two flat surfaces shows nothing). Fall back
     to the unrestricted ratio if fewer than 2 images qualify.
  4. Evaluate the 51x51 loss surface for the top-2 images x both models,
     REUSING the scan's eigenvectors, and write rank1/ rank2/ under
     results/<outdir-name>/ in each run dir. The full ranking table goes to
     summary.json so the choice is auditable and overridable.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import analyze_manuscript_extras_hpc as extras  # noqa: E402


def parse_candidates(path: Path) -> list[tuple[str, str]]:
    pairs = []
    for ln in path.read_text().splitlines():
        ln = ln.strip()
        if not ln:
            continue
        wnid, _, fname = ln.partition("/")
        if not wnid or not fname:
            raise ValueError(f"malformed candidate line: {ln!r}")
        pairs.append((wnid, fname))
    return pairs


def main() -> int:
    from cli_tool import get_model_for_experiment  # lazy: heavy import chain
    from src.analysis.run_loss_surface import (  # lazy: heavy import chain
        compute_top_eigenvectors,
        evaluate_loss_surface,
    )

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--flex-run-dir", required=True)
    ap.add_argument("--vanilla-run-dir", required=True)
    ap.add_argument("--candidates", required=True,
                    help="wnid/filename lines (pick_train_files.py output)")
    ap.add_argument("--files-root", required=True)
    ap.add_argument("--classes", required=True,
                    help="one wnid per line; sorted rank = class index")
    ap.add_argument("--forced", action="append", default=[],
                    help="'path:class_index' probe scanned before the random "
                         "candidates (repeatable; use for the known probes)")
    ap.add_argument("--pool-size", type=int, default=16)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--grid-points", type=int, default=51)
    ap.add_argument("--range-scale", type=float, default=10.0)
    ap.add_argument("--outdir-name", default="manuscript_extras_best")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    classes = sorted(ln.strip() for ln in Path(args.classes).read_text().splitlines()
                     if ln.strip())
    class_rank = {w: i for i, w in enumerate(classes)}
    files_root = Path(args.files_root)

    runs = {"flex": Path(args.flex_run_dir), "vanilla": Path(args.vanilla_run_dir)}
    outdirs = {k: rd / "results" / args.outdir_name for k, rd in runs.items()}
    if all((od / ".done").exists() for od in outdirs.values()) and not args.force:
        print("[skip] both sides already done")
        return 0

    models = {}
    load_diags = {}
    for k, rd in runs.items():
        m = get_model_for_experiment(str(rd))
        m.to(device).eval()
        load_diags[k] = extras.verify_checkpoint_load(m, rd, device)
        # the whole job runs in float64: the gate forward, the eigenvector
        # power iterations, and the surface evaluation (same precision fix
        # as dump_loss_surface -- float32 softmax saturation zeroes the
        # curvature of confidently-classified probes)
        m.double()
        models[k] = m
        print(f"[model] {k}: {rd}", flush=True)

    # candidate stream: forced probes first, then the seeded random picks
    stream: list[tuple[str, int, str]] = []
    for spec in args.forced:
        pth, _, cls = spec.rpartition(":")
        if not pth or not cls.lstrip("-").isdigit():
            raise ValueError(f"malformed --forced spec: {spec!r} (want path:class)")
        stream.append((pth, int(cls), Path(pth).name))
    for wnid, fname in parse_candidates(Path(args.candidates)):
        if wnid not in class_rank:
            print(f"[scan] {wnid}/{fname}: wnid not in class list; skipped")
            continue
        stream.append((str(files_root / wnid / fname), class_rank[wnid],
                       f"{wnid}/{fname}"))

    criterion = nn.CrossEntropyLoss()
    pool = []          # dicts: name, cls, inputs, targets, probe per model
    seen_names = set()
    for pth, cls, name in stream:
        if len(pool) >= args.pool_size:
            break
        if name in seen_names:
            continue
        seen_names.add(name)
        try:
            inputs, targets, _ = extras.load_override_image(pth, cls)
        except (FileNotFoundError, OSError) as e:
            print(f"[scan] {name}: load failed ({e})", flush=True)
            continue
        inputs = inputs.double()
        verdicts = {}
        infos = {}
        with torch.no_grad():
            for k, m in models.items():
                logits = m(inputs.to(device))
                loss = criterion(logits, targets.to(device)).item()
                ok, info = extras.probe_sanity(logits, int(targets.item()), loss)
                verdicts[k] = ok
                infos[k] = info
        both = all(verdicts.values())
        print(f"[scan] {name} cls={cls} "
              f"flex={'PASS' if verdicts['flex'] else 'fail'} "
              f"vanilla={'PASS' if verdicts['vanilla'] else 'fail'} "
              f"pool={len(pool) + int(both)}/{args.pool_size}", flush=True)
        if both:
            pool.append({"name": name, "cls": cls, "inputs": inputs,
                         "targets": targets, "probe": infos})

    if len(pool) < 2:
        print(f"[error] only {len(pool)} images passed the gate for both "
              "models; need >= 2. Stage more candidates.", file=sys.stderr)
        return 1
    print(f"[pool] {len(pool)} images pass for both models", flush=True)

    # top-2 input-Hessian eigenpairs per (image, model)
    for i, entry in enumerate(pool):
        entry["eig"] = {}
        for k, m in models.items():
            t0 = time.time()
            eigvals, eigvecs = compute_top_eigenvectors(
                m, criterion, entry["inputs"].clone(), entry["targets"],
                k=2, steps=args.steps, device=device)
            entry["eig"][k] = {"vals": [float(v) for v in eigvals[:2]],
                               "vecs": eigvecs}
            print(f"[eig] {i + 1}/{len(pool)} {entry['name']} {k}: "
                  f"{[round(float(v), 4) for v in eigvals[:2]]} "
                  f"({time.time() - t0:.0f}s)", flush=True)

    eps = 1e-9
    for entry in pool:
        v = sum(entry["eig"]["vanilla"]["vals"])
        f = sum(entry["eig"]["flex"]["vals"])
        entry["van_sum"] = v
        entry["flex_sum"] = f
        entry["ratio"] = v / max(f, eps)
    med = float(np.median([e["van_sum"] for e in pool]))
    eligible = [e for e in pool if e["van_sum"] >= med]
    if len(eligible) < 2:
        eligible = pool
        print("[rank] WARNING: median guard left <2 images; ranking the "
              "full pool by ratio", flush=True)
    ranked = sorted(eligible, key=lambda e: e["ratio"], reverse=True)
    winners = ranked[:2]
    table = [{"image": e["name"], "class_index": e["cls"],
              "vanilla_top2": e["eig"]["vanilla"]["vals"],
              "flex_top2": e["eig"]["flex"]["vals"],
              "vanilla_sum": e["van_sum"], "flex_sum": e["flex_sum"],
              "ratio": e["ratio"],
              "eligible": e["van_sum"] >= med}
             for e in sorted(pool, key=lambda e: e["ratio"], reverse=True)]
    print(f"[rank] median vanilla curvature {med:.4f}; winners: "
          f"{[w['name'] for w in winners]} "
          f"ratios {[round(w['ratio'], 2) for w in winners]}", flush=True)

    rc = 0
    for k, m in models.items():
        outdir = outdirs[k]
        outdir.mkdir(parents=True, exist_ok=True)
        for r, w in enumerate(winners, start=1):
            rdir = outdir / f"rank{r}"
            rdir.mkdir(parents=True, exist_ok=True)
            eig = w["eig"][k]
            ma, mb, surface = evaluate_loss_surface(
                m, criterion, w["inputs"].clone(), w["targets"],
                eig["vecs"][0], eig["vecs"][1],
                grid_points=args.grid_points, range_scale=args.range_scale,
                device=device)
            np.save(rdir / "loss_surface_z.npy", surface)
            np.save(rdir / "loss_surface_alpha.npy", ma)
            np.save(rdir / "loss_surface_beta.npy", mb)
            (rdir / "loss_surface_meta.json").write_text(json.dumps({
                "eigenvalues": eig["vals"], "grid_points": args.grid_points,
                "range_scale": args.range_scale, "probe_image": w["name"],
                "class_index": w["cls"], "rank": r,
                "selection": "max baseline/flex top-2 curvature ratio, "
                             "baseline curvature >= pool median",
            }, indent=2))
            print(f"[surface] {k} rank{r} {w['name']} "
                  f"z=[{surface.min():.3f},{surface.max():.3f}]", flush=True)
        (outdir / "summary.json").write_text(json.dumps({
            "run_dir": str(runs[k]), "outdir_name": args.outdir_name,
            "load_diagnostics": load_diags[k],
            "pool_size": len(pool), "median_vanilla_curvature": med,
            "winners": [w["name"] for w in winners],
            "ranking_table": table,
            "probe": {w["name"]: w["probe"][k] for w in winners},
        }, indent=2))
        (outdir / ".done").write_text("ok\n")
        print(f"[done] {k} -> {outdir}", flush=True)
    return rc


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise
