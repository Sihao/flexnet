#!/usr/bin/env python3
"""
Produce the two MANUSCRIPT figures that need a live model forward pass and are
NOT emitted by the standard iso-analysis pipeline, for one restored checkpoint:

  Fig 2E  feature maps of an early layer for a fixed example image
  Fig 4C  input-space loss surface projected on the top-2 Hessian eigenvectors

INVARIANT: both models of an iso-accuracy pair MUST be run on the SAME fixed
image (same class index + same in-class file rank), or the flex-vs-vanilla
panels are not comparable and the figure is invalid. The image is selected
deterministically by (class_index, in-class file rank) rather than the
shuffled ImageFolder sampling used elsewhere. ImageFolder's full-tree scan is
avoided (one readdir of the val root + one class dir + one file open) to stay
light on the fs8 Lustre client.

By DEFAULT (neither --file-rank nor --auto-pick-image given) the file rank is
the fixed constant DEFAULT_FILE_RANK (0), so both models of a pair already use
the same image with zero flags. The probe sanity gate (probe_sanity) is the
safety net for that fixed image: a degenerate probe makes this script SKIP the
loss surface rather than ship a chance-plateau figure.

--auto-pick-image is an opt-in, per-model, model-dependent confidence scan
(pick_probe_image) and is NOT pair-safe by itself: it picks each model's OWN
best-confidence image, which can differ between flex and vanilla. Treat it as
a per-pair RESOLUTION step, never as something run independently per model:
run it once, read the printed CHOSEN_FILE_RANK=<r> line, then pass that SAME
--file-rank <r> explicitly to BOTH models of the pair. That re-invocation is
the orchestrator's job, not this script's. An explicit --file-rank always
wins over --auto-pick-image and skips the scan.

Outputs land under <run_dir>/results/manuscript_extras/ (npy + png), pulled and
re-plotted flex-vs-vanilla locally afterwards.

COMPUTE: loads torch + a model + one image -> MUST run via sbatch on a GPU node,
never on the login node.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# `cli_tool` and `src.analysis.run_loss_surface` both transitively import
# RunLoader's heavy chain (dataset loaders, the full model zoo). Both are
# imported lazily, right where they are used (main(), dump_loss_surface()),
# so probe_sanity() and pick_probe_image() stay importable on their own,
# without a GPU or the training stack -- see tests/test_extras_probe.py.

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Fixed, pair-safe default in-class file rank: used whenever neither
# --file-rank nor --auto-pick-image is given, so both models of an
# iso-accuracy pair land on the SAME image with zero flags. See module
# docstring for the full invariant.
DEFAULT_FILE_RANK = 0

# loss_surface_{z,alpha,beta}.npy: the three arrays dump_loss_surface() writes
# alongside loss_surface_meta.json. Named once here so the skip path
# (write_skipped_surface) can delete exactly these and nothing else.
SURFACE_NPY_NAMES = ("loss_surface_z.npy", "loss_surface_alpha.npy", "loss_surface_beta.npy")


def resolve_val_dir() -> str:
    root = os.environ.get("IMAGENET_LOCAL_DIR",
                          "/lustre/fs8/huds_lab/scratch/slu/Data/imagenet_full")
    for cand in (os.path.join(root, "val"), root):
        if os.path.isdir(cand) and any(
            os.path.isdir(os.path.join(cand, d)) for d in os.listdir(cand)[:5]
        ):
            return cand
    raise FileNotFoundError(f"no ImageNet val dir under {root}")


def load_fixed_image(val_dir: str, class_index: int, file_rank: int):
    """Return (inputs[1,3,224,224], targets[1], fpath) for a deterministic
    image. class_index indexes the alphabetically-sorted WNID dirs (== model
    labels)."""
    import torchvision.transforms as T
    from PIL import Image

    wnids = sorted(d for d in os.listdir(val_dir)
                   if os.path.isdir(os.path.join(val_dir, d)))
    wnid = wnids[class_index]
    cdir = os.path.join(val_dir, wnid)
    files = sorted(f for f in os.listdir(cdir)
                   if f.lower().endswith((".jpeg", ".jpg", ".png")))
    if not files:
        raise FileNotFoundError(f"no candidate images under {cdir}")
    fpath = os.path.join(cdir, files[file_rank % len(files)])
    tf = T.Compose([
        T.Resize(256), T.CenterCrop(224), T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    img = Image.open(fpath).convert("RGB")
    inputs = tf(img).unsqueeze(0)
    targets = torch.tensor([class_index], dtype=torch.long)
    print(f"[img] class_index={class_index} wnid={wnid} file={files[file_rank % len(files)]}")
    return inputs, targets, fpath


def load_override_image(image_path: str, class_index: int):
    """Return (inputs[1,3,224,224], targets[1], fpath) for an EXPLICIT probe
    image path, bypassing load_fixed_image / pick_probe_image's WNID +
    file-rank indexing entirely.

    This is the --image-path override: it forces BOTH models of an
    iso-accuracy pair onto the exact same original manuscript probe JPEG
    (e.g. the ImageNet-100 n01632777/ILSVRC2012_val_00034583.JPEG image also
    referenced as the default --image in cli_tool.py's `visualize`
    subcommand), independent of whatever ImageNet tree --class-index/
    --file-rank would otherwise scan.

    NOTE (chainlink #396 audit correction): n01632777 is AXOLOTL, not golden
    retriever -- an earlier pass here wrongly assumed it matched the
    unrelated --class-index default of 207 (207 IS correct for golden
    retriever / WNID n02099601 in the full-ImageNet-1k alphabetical scan
    default path, but that has nothing to do with n01632777). Do not
    reintroduce that assumption.

    class_index is still the CrossEntropyLoss / probe_sanity target label --
    the caller MUST supply the correct WNID index for the image being
    loaded; there is no safe default when this override is used (main()
    requires --class-index explicitly whenever --image-path is given). This
    function does not, and cannot in general, infer that index itself: the
    override path may live under a completely different class-tree layout
    (e.g. ImageNet-100 val.X) than the default full-ImageNet val dir, so
    guessing would risk silently mismatching image and label -- exactly the
    bug this note corrects.

    Raises FileNotFoundError if the path does not exist: the original
    manuscript image is required whenever this override is given, and must
    NEVER be silently substituted with the class-207 scan default.
    """
    import torchvision.transforms as T
    from PIL import Image

    fpath = Path(image_path)
    if not fpath.is_file():
        raise FileNotFoundError(
            f"--image-path {fpath} does not exist. The original manuscript "
            "probe image is required when this override is given; refusing "
            "to silently fall back to the --class-index scan default."
        )
    tf = T.Compose([
        T.Resize(256), T.CenterCrop(224), T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    img = Image.open(fpath).convert("RGB")
    inputs = tf(img).unsqueeze(0)
    targets = torch.tensor([class_index], dtype=torch.long)
    print(f"[img] OVERRIDE --image-path={fpath} class_index={class_index}")
    return inputs, targets, str(fpath)


def resolve_probe_rank(args) -> int | None:
    """Resolve the file_rank_override main() passes to pick_probe_image, from
    parsed CLI args. Precedence:

      1. args.file_rank explicit (not None) -> always wins, return it verbatim
         (pick_probe_image then skips the scan entirely).
      2. args.auto_pick_image set -> return None, so pick_probe_image runs its
         per-model confidence scan. That scan is NOT pair-safe by itself (see
         module docstring); main() only reaches it when this flag is set.
      3. neither set -- the DEFAULT -> DEFAULT_FILE_RANK, so both models of an
         iso-accuracy pair use the SAME image with zero flags.

    Pure function of args (no model, no filesystem, no scan), so the default
    wiring is unit-testable without a GPU -- see tests/test_extras_probe.py.
    """
    if args.file_rank is not None:
        return args.file_rank
    if getattr(args, "auto_pick_image", False):
        return None
    return DEFAULT_FILE_RANK


def pick_probe_image(model, val_dir: str, class_index: int, device,
                     n_scan: int = 10, file_rank_override: int | None = None):
    """Pick the probe image used for Fig2E / Fig4C.

    An explicit file_rank_override always wins (both models of an
    iso-accuracy pair must share the same image) and skips the scan.

    Otherwise this function runs a MODEL-DEPENDENT scan of the first n_scan
    files in the class directory and keeps the highest-confidence
    CORRECTLY-classified one (argmax == class_index); if none classify
    correctly, it falls back to the highest-confidence candidate and lets
    probe_sanity() catch the failure downstream. This scan branch is NOT
    pair-safe by itself -- flex and vanilla can pick different images -- so
    main() only reaches it when --auto-pick-image is explicitly set. On that
    branch this function prints a CHOSEN_FILE_RANK=<r> line so the calling
    orchestrator can re-invoke BOTH models of the pair with that same
    --file-rank <r>; never rely on this scan independently per model.

    Returns (inputs[1,3,224,224], targets[1], chosen_path).
    """
    if file_rank_override is not None:
        inputs, targets, fpath = load_fixed_image(val_dir, class_index, file_rank_override)
        print(f"[probe] explicit --file-rank={file_rank_override}: {fpath}")
        return inputs, targets, fpath

    wnids = sorted(d for d in os.listdir(val_dir)
                   if os.path.isdir(os.path.join(val_dir, d)))
    wnid = wnids[class_index]
    cdir = os.path.join(val_dir, wnid)
    n_files = len([f for f in os.listdir(cdir)
                   if f.lower().endswith((".jpeg", ".jpg", ".png"))])
    n_candidates = min(n_scan, n_files)
    if n_candidates <= 0:
        raise FileNotFoundError(f"no candidate images under {cdir}")

    best_correct = None   # (confidence, rank, inputs, targets, fpath)
    best_overall = None
    with torch.no_grad():
        for rank in range(n_candidates):
            inputs, targets, fpath = load_fixed_image(val_dir, class_index, rank)
            logits = model(inputs.to(device))
            probs = torch.softmax(logits.detach().float().reshape(-1), dim=0)
            conf, idx = torch.max(probs, dim=0)
            conf, idx = float(conf.item()), int(idx.item())
            candidate = (conf, rank, inputs, targets, fpath)
            if best_overall is None or conf > best_overall[0]:
                best_overall = candidate
            if idx == class_index and (best_correct is None or conf > best_correct[0]):
                best_correct = candidate

    chosen = best_correct if best_correct is not None else best_overall
    conf, chosen_rank, inputs, targets, fpath = chosen
    status = "correct" if best_correct is not None else "FALLBACK-none-classified-correctly"
    print(f"[probe] scanned {n_candidates} candidates ({wnid}), picked ({status}, "
          f"confidence={conf:.4f}): {fpath}")
    # Machine-greppable: an orchestrator resolving a pair-safe rank from this
    # per-model scan parses this exact line, then passes the SAME
    # --file-rank <r> to both models of the iso-accuracy pair.
    print(f"CHOSEN_FILE_RANK={chosen_rank}")
    print(f"CHOSEN_FILE_PATH={fpath}")
    return inputs, targets, fpath


def first_spatial_layers(model, n=2):
    """Return up to n (name, module) early conv/flex layers (for feature maps)."""
    out = []
    for name, m in model.named_modules():
        cls = m.__class__.__name__
        if ("Conv2d" in cls or "Flex" in cls) and hasattr(m, "forward"):
            out.append((name, m))
        if len(out) >= n * 6:
            break
    # pick the 1st (stem) and an early mid layer for variety
    picks = []
    if out:
        picks.append(out[0])
    if len(out) > 4:
        picks.append(out[4])
    return picks[:n]


def dump_feature_maps(model, inputs, device, outdir: Path):
    """Fig 2E: save early-layer channel activations for the example image."""
    picks = first_spatial_layers(model, n=2)
    acts = {}
    handles = []
    for name, mod in picks:
        def mk(nm):
            def hook(_m, _i, o):
                acts[nm] = o.detach().float().cpu().numpy()
            return hook
        handles.append(mod.register_forward_hook(mk(name)))
    with torch.no_grad():
        model(inputs.to(device))
    for h in handles:
        h.remove()
    saved = {}
    for name, arr in acts.items():
        safe = name.replace(".", "_")
        np.save(outdir / f"featmap_{safe}.npy", arr)      # (1,C,H,W)
        saved[name] = list(arr.shape)
    print(f"[Fig2E] feature maps saved for layers: {list(acts)}")
    return saved


# Absolute-value threshold below which BOTH top-2 eigenvalues are treated as
# a genuinely degenerate (numerically flat) loss surface, even after the
# float64 root-cause fix in dump_loss_surface. Comfortably above float64
# rounding noise, far below any real curvature magnitude seen on a trained
# checkpoint -- see dump_loss_surface's docstring and lessons.md for why this
# guard exists.
DEGENERATE_EIGENVALUE_ABS_TOL = 1e-8


class DegenerateLossSurfaceError(RuntimeError):
    """Raised by dump_loss_surface when the top-2 eigenvalues are still ~0
    even in float64 -- a genuinely degenerate probe/checkpoint, not the
    float32-saturation precision artifact the float64 cast fixes. main()
    catches this SPECIFIC type (not the generic per-task except Exception)
    and routes it through write_skipped_surface, so it lands on the SAME
    "skipped" markers #391's summary.json scanner already relies on for a
    probe_sanity failure, instead of an invisible generic ERROR string."""


def dump_loss_surface(model, inputs, targets, device, outdir: Path,
                      grid_points=51, range_scale=1.0):
    """Fig 4C: loss on a grid spanned by the top-2 input-Hessian eigenvectors.

    ROOT CAUSE of the previously observed vanilla-model 0.00/0.00 top-2
    eigenvalues (see lessons.md, chainlink #396): a vanilla VGG16 is built
    entirely from piecewise-linear operators (Conv2d, BatchNorm2d-in-eval
    (an affine transform), ReLU, MaxPool2d, Linear), so the network's LOGITS
    are locally linear in the input almost everywhere -- the only source of
    input-Hessian curvature is the softmax/cross-entropy term itself,
    diag(p) - p*p^T, propagated through the (locally constant) output
    Jacobian. When the probe image is classified with a large logit margin,
    float32 softmax saturates to an EXACT 0.0/1.0 one-hot vector, so
    diag(p) - p*p^T -- and therefore the double-backprop Hessian-vector
    product compute_top_eigenvectors relies on -- underflows to a
    bit-for-bit hard zero, not merely "small". float64 has ~300 extra orders
    of magnitude of dynamic range before that underflow, so running this
    analysis in double precision recovers the real (small but non-zero)
    curvature instead of a spurious exact zero.

    FIX: cast model + inputs to float64 for this analysis only (restored to
    the original dtype afterward), verify AFTER the cast that d(loss)/d(input)
    is non-zero (a cheap guard against a truly flat/dead batch, distinct from
    the softmax-saturation precision issue above), and refuse to write a
    surface whose top-2 eigenvalues are both still ~0 after the fix.

    AUDIT CORRECTION (chainlink #396): the zero-gradient pre-check MUST run
    in float64, after the cast -- NOT in float32 beforehand. probe_sanity()
    upstream only lets a confidently CORRECT probe reach this function (a
    confidently wrong one is already skipped), and that is exactly the
    saturated-softmax regime described above: dL/dx = J^T(p - y) is ALSO
    exactly zero in float32 there, for the same underflow reason the
    Hessian is. A float32 pre-check would therefore raise precisely on the
    probe this fix targets, aborting Fig4C before the float64 recovery ever
    runs. Running the same check in float64 avoids that self-defeat.
    """
    from src.analysis.run_loss_surface import (  # lazy: heavy chain, see top of file
        compute_top_eigenvectors,
        evaluate_loss_surface,
    )

    if model.training:
        raise RuntimeError(
            "dump_loss_surface: model is in train() mode -- BatchNorm/Dropout "
            "would use batch statistics instead of the checkpoint's running "
            "stats, corrupting the Hessian. Call model.eval() before this."
        )

    criterion = nn.CrossEntropyLoss()

    # Root-cause fix: run the double-backprop Hessian work -- INCLUDING the
    # zero-gradient pre-check -- in float64, so a confidently-classified
    # probe cannot silently underflow the softmax Jacobian to an exact 0.0
    # in float32 (see docstring above). Always restored to the original
    # dtype, even on failure.
    model_dtype = next(model.parameters()).dtype
    try:
        model.double()
        inputs64 = inputs.double()
        targets_dev = targets.to(device)

        # Cheap first-order guard *before* paying for 50 power-iteration
        # steps x k eigenvectors: if d(loss)/d(input) is zero even in
        # float64, the Hessian is meaningless everywhere on this batch (a
        # genuinely flat/dead probe -- e.g. a constant-color image -- NOT
        # the float32-saturation case the cast above already handles).
        probe_inputs = inputs64.clone().to(device).requires_grad_(True)
        probe_loss = criterion(model(probe_inputs), targets_dev)
        (grad_x,) = torch.autograd.grad(probe_loss, probe_inputs)
        grad_norm = float(torch.norm(grad_x).item())
        if grad_norm == 0.0:
            raise RuntimeError(
                "dump_loss_surface: d(loss)/d(input) is exactly zero even in "
                "float64 -- the Hessian/eigenvector computation would run on "
                "a flat batch. Refusing to emit a degenerate loss surface."
            )
        print(f"[Fig4C] pre-check (float64): ||d(loss)/d(input)|| = {grad_norm:.3e}")

        eigvals, eigvecs = compute_top_eigenvectors(
            model, criterion, inputs64.clone(), targets, k=2, steps=50, device=device)
        ma, mb, surface = evaluate_loss_surface(
            model, criterion, inputs64.clone(), targets, eigvecs[0], eigvecs[1],
            grid_points=grid_points, range_scale=range_scale, device=device)
    finally:
        model.to(model_dtype)

    if all(abs(e) < DEGENERATE_EIGENVALUE_ABS_TOL for e in eigvals[:2]):
        raise DegenerateLossSurfaceError(
            f"top-2 eigenvalues {eigvals[:2]} are both below "
            f"{DEGENERATE_EIGENVALUE_ABS_TOL:.1e} even in float64 -- this is "
            "a genuinely degenerate probe/checkpoint, not a precision "
            "artifact. Refusing to write a flat loss surface; pick a "
            "different probe image or investigate the checkpoint."
        )

    np.save(outdir / "loss_surface_z.npy", surface)
    np.save(outdir / "loss_surface_alpha.npy", ma)
    np.save(outdir / "loss_surface_beta.npy", mb)
    (outdir / "loss_surface_meta.json").write_text(json.dumps({
        "eigenvalues": [float(e) for e in eigvals],
        "grid_points": grid_points, "range_scale": range_scale,
    }, indent=2))
    print(f"[Fig4C] loss surface {surface.shape} eig={[round(float(e),3) for e in eigvals]} "
          f"z=[{surface.min():.3f},{surface.max():.3f}]")
    return {"eigenvalues": [float(e) for e in eigvals],
            "z_min": float(surface.min()), "z_max": float(surface.max())}


def write_skipped_surface(outdir: Path, reason: str, probe_info: dict) -> dict:
    """Record a skipped loss surface: write loss_surface_meta.json with
    {"skipped": true, ...} AND delete any loss_surface_{z,alpha,beta}.npy
    already sitting in outdir.

    A prior good run (or a stale one left by --force re-running a probe that
    now fails) can leave those three npys on disk. If a skip left them in
    place, the downstream plotter would load the OLD arrays next to a meta
    that has no "eigenvalues" key -> KeyError, or worse, silently plot stale
    data as if it were current. The on-disk state after this function runs is
    always: skipped meta present, all three npys ABSENT -- including a
    DANGLING symlink, which exists() alone would miss (it follows the link
    and reports False for a broken target), so the guard also checks
    is_symlink() directly. A skip with nothing stale to clean is a no-op.

    Returns the meta dict that was written, so callers can fold it into
    summary.json without re-deriving it.
    """
    for name in SURFACE_NPY_NAMES:
        stale = outdir / name
        if stale.is_symlink() or stale.exists():
            stale.unlink(missing_ok=True)
            print(f"[SANITY-GATE-FAILED] removed stale {stale.name} so it cannot "
                  f"be read next to this skip's meta")

    meta = {
        "skipped": True,
        "reason": reason,
        "base_loss": probe_info["loss"],
        "top1": probe_info["top1"],
        "confidence": probe_info["top1_confidence"],
    }
    (outdir / "loss_surface_meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def resolve_latest_checkpoint(ckpt_dir: Path) -> Path:
    """Pick the highest-epoch checkpoint_<N>.pth. Replicates ONLY the
    path-resolution half of RunLoader._load_model_and_optimizer
    (src/analysis/run_loader.py) so this script can re-inspect what got
    loaded, without editing that class."""
    candidates = []
    for f in ckpt_dir.glob("checkpoint_*.pth"):
        try:
            epoch = int(f.stem.replace("checkpoint_", ""))
        except ValueError:
            continue
        candidates.append((epoch, f))
    if not candidates:
        raise FileNotFoundError(f"No checkpoint_<N>.pth found in {ckpt_dir}")
    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1]


def verify_checkpoint_load(model, run_dir: Path, device,
                           max_missing_frac: float = 0.10) -> dict:
    """Re-apply the run's checkpoint state_dict with strict=False, the same
    way RunLoader already does inside get_model_for_experiment, and report
    how many keys actually matched. RunLoader drops mismatched keys
    silently; a checkpoint that matches almost nothing leaves the model at
    Kaiming init, which is the suspected root cause of the observed
    chance-plateau (base loss ~= ln(1000)) loss surface. Raises before any
    figure is written in that case instead of silently emitting a
    degenerate one."""
    ckpt_path = resolve_latest_checkpoint(run_dir / "checkpoints")
    save_content = torch.load(ckpt_path, map_location=device)
    state_dict = save_content["model_state_dict"]
    incompat = model.load_state_dict(state_dict, strict=False)
    total = len(model.state_dict())
    missing = list(incompat.missing_keys)
    unexpected = list(incompat.unexpected_keys)
    diag = {
        "checkpoint": str(ckpt_path),
        "missing_keys": len(missing),
        "unexpected_keys": len(unexpected),
        "total_keys": total,
        "missing_keys_sample": missing[:8],
        "unexpected_keys_sample": unexpected[:8],
    }
    print(f"[load-verify] checkpoint={ckpt_path.name} "
          f"missing={len(missing)}/{total} unexpected={len(unexpected)}")
    if total and (len(missing) / total) > max_missing_frac:
        print(f"[load-verify] first missing keys: {missing[:8]}")
        print(f"[load-verify] first unexpected keys: {unexpected[:8]}")
        raise RuntimeError(
            f"Checkpoint {ckpt_path} left {len(missing)}/{total} model keys "
            f"unmatched (> {max_missing_frac:.0%}); model is likely still at "
            f"Kaiming init. Refusing to emit a degenerate loss surface."
        )
    return diag


def probe_sanity(logits: torch.Tensor, target_idx: int, loss: float,
                 loss_threshold: float = 3.0) -> tuple[bool, dict]:
    """Pure sanity check on one forward pass. Catches a chance plateau
    (near-uniform logits, loss ~= ln(1000) = 6.908) before it reaches a
    shipped loss-surface figure. ok is False when the top1 prediction misses
    target_idx OR loss exceeds loss_threshold."""
    probs = torch.softmax(logits.detach().float().reshape(-1), dim=0)
    k = min(5, probs.numel())
    top_conf, top_idx = torch.topk(probs, k=k)
    top1_idx = int(top_idx[0].item())
    top1_conf = float(top_conf[0].item())
    loss = float(loss)
    ok = (top1_idx == int(target_idx)) and (loss <= loss_threshold)
    info = {
        "top1": top1_idx,
        "top1_confidence": top1_conf,
        "top5": [int(i) for i in top_idx.tolist()],
        "top5_confidence": [float(c) for c in top_conf.tolist()],
        "target_idx": int(target_idx),
        "loss": loss,
        "loss_threshold": float(loss_threshold),
    }
    return ok, info


def main() -> int:
    from cli_tool import get_model_for_experiment  # lazy: heavy chain, see top of file

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--class-index", type=int, default=None,
                    help="alphabetical WNID index (== model label) for the "
                         "probe. Default when --image-path is NOT given: 207 "
                         "(golden retriever, WNID n02099601, in the "
                         "full-ImageNet-1k alphabetically-sorted class list "
                         "load_fixed_image/pick_probe_image scan over). "
                         "REQUIRED (no default) when --image-path IS given -- "
                         "see --image-path's help for why.")
    ap.add_argument("--file-rank", type=int, default=None,
                    help="explicit in-class file rank; always wins over "
                         "--auto-pick-image and skips the scan. Use this to force "
                         "the SAME image onto both models of an iso-accuracy pair "
                         f"(default when unset: fixed rank {DEFAULT_FILE_RANK}, "
                         "already shared across both models with no flags)")
    ap.add_argument("--auto-pick-image", action="store_true",
                    help="opt-in: run a per-model confidence scan "
                         "(pick_probe_image) instead of the fixed default rank. "
                         "NOT pair-safe by itself -- flex and vanilla can pick "
                         "different images. Prints CHOSEN_FILE_RANK=<r>; the "
                         "orchestrator must re-run BOTH models of the pair with "
                         "that same --file-rank <r>")
    ap.add_argument("--image-path", default=None,
                    help="explicit path to the ORIGINAL manuscript probe JPEG "
                         "(e.g. the ImageNet-100 "
                         "n01632777/ILSVRC2012_val_00034583.JPEG image "
                         "cli_tool.py's `visualize --image` also defaults to). "
                         "OVERRIDES --class-index's file-scan, --file-rank, and "
                         "--auto-pick-image entirely (load_fixed_image / "
                         "pick_probe_image are skipped) so BOTH models of an "
                         "iso-accuracy pair use the exact same image file. "
                         "--class-index still supplies the CrossEntropyLoss "
                         "target label and is REQUIRED (no default) when this "
                         "flag is given: n01632777 is AXOLOTL, NOT golden "
                         "retriever, and its true index in this model's "
                         "class-index space is NOT known to be 207 -- that "
                         "value is only correct for the unrelated default "
                         "(no --image-path) golden-retriever scan. Supply the "
                         "override image's real WNID index explicitly. Raises "
                         "FileNotFoundError if the path does not exist; never "
                         "silently falls back to the class-207 default.")
    ap.add_argument("--grid-points", type=int, default=51)
    ap.add_argument("--range-scale", type=float, default=10.0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if args.image_path is not None and args.class_index is None:
        ap.error(
            "--class-index is REQUIRED when --image-path is given: the "
            "override image's true label in the model's class-index space "
            "cannot be inferred from the file path alone (chainlink #396 "
            "audit: n01632777, the default --image-path's WNID, is AXOLOTL, "
            "not golden retriever -- do not assume class-index 207 for it). "
            "Pass the override image's real WNID index explicitly."
        )
    if args.class_index is None:
        args.class_index = 207  # golden retriever -- default (no --image-path) path only

    run_dir = Path(args.run_dir)
    outdir = run_dir / "results" / "manuscript_extras"
    outdir.mkdir(parents=True, exist_ok=True)
    marker = outdir / ".done"
    if marker.exists() and not args.force:
        print(f"[skip] already done ({marker})")
        return 0

    device = args.device if torch.cuda.is_available() else "cpu"
    model = get_model_for_experiment(str(run_dir))
    model.to(device).eval()
    load_diag = verify_checkpoint_load(model, run_dir, device)

    if args.image_path:
        # Override path: skip val_dir resolution and the WNID/file-rank scan
        # entirely (see load_override_image docstring). Default behavior
        # (this branch not taken) is unchanged below.
        inputs, targets, probe_path = load_override_image(args.image_path, args.class_index)
    else:
        val_dir = resolve_val_dir()
        probe_rank = resolve_probe_rank(args)
        inputs, targets, probe_path = pick_probe_image(
            model, val_dir, args.class_index, device, file_rank_override=probe_rank)

    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        probe_logits = model(inputs.to(device))
        probe_loss = criterion(probe_logits, targets.to(device)).item()
    probe_ok, probe_info = probe_sanity(probe_logits, int(targets.item()), probe_loss)

    summary = {"run_dir": str(run_dir), "class_index": args.class_index,
               "probe_image": probe_path, "image_override": args.image_path,
               "load_diagnostics": load_diag,
               "probe_ok": probe_ok, "probe": probe_info}

    failures = 0
    tasks = [("fig2E_feature_maps", lambda: dump_feature_maps(model, inputs, device, outdir))]
    for label, fn in tasks:
        try:
            summary[label] = fn()
        except Exception as e:  # noqa: BLE001
            failures += 1
            summary[label] = f"ERROR: {e}"
            print(f"[error] {label}: {e}", file=sys.stderr)
            traceback.print_exc()

    if probe_ok:
        try:
            summary["fig4C_loss_surface"] = dump_loss_surface(
                model, inputs, targets, device, outdir,
                args.grid_points, args.range_scale)
        except DegenerateLossSurfaceError as e:
            # Same skip contract as the probe-sanity-failure branch below:
            # #391's summary.json scanner keys off summary["loss_surface"] /
            # write_skipped_surface's markers, not a generic ERROR string, so
            # a still-degenerate-after-float64 surface must land here, NOT
            # in the generic except Exception path further down.
            reason = f"loss surface degenerate after float64 fix: {e}"
            print(f"[SANITY-GATE-FAILED] {reason}")
            skip_meta = write_skipped_surface(outdir, reason, probe_info)
            print("[SANITY-GATE-FAILED] loss_surface SKIPPED for this run "
                  "(summary.json['loss_surface'] == 'skipped')")
            summary["loss_surface"] = "skipped"
            summary["skipped_surfaces"] = [{"surface": "fig4C_loss_surface", **skip_meta}]
            summary["fig4C_loss_surface"] = {
                "skipped": True,
                "reason": reason,
                "probe": probe_info,
            }
        except Exception as e:  # noqa: BLE001
            failures += 1
            summary["fig4C_loss_surface"] = f"ERROR: {e}"
            print(f"[error] fig4C_loss_surface: {e}", file=sys.stderr)
            traceback.print_exc()
    else:
        reason = "probe failed sanity gate"
        print("[SANITY-GATE-FAILED] probe image failed the sanity gate -- "
              "refusing to emit a loss surface that may be a chance plateau")
        print(f"[SANITY-GATE-FAILED] top5={probe_info['top5']} "
              f"confidence={probe_info['top1_confidence']:.4f} "
              f"loss={probe_info['loss']:.4f} target={probe_info['target_idx']}")
        print(f"[SANITY-GATE-FAILED] checkpoint load diagnostics: {load_diag}")
        skip_meta = write_skipped_surface(outdir, reason, probe_info)
        print("[SANITY-GATE-FAILED] loss_surface SKIPPED for this run "
              "(summary.json['loss_surface'] == 'skipped')")
        # Top-level, unmissable markers: #391 scans many summary.json files
        # across runs/tags and must be able to tell a skip apart from a
        # completed surface without digging into the fig4C_loss_surface dict.
        summary["loss_surface"] = "skipped"
        summary["skipped_surfaces"] = [{"surface": "fig4C_loss_surface", **skip_meta}]
        summary["fig4C_loss_surface"] = {
            "skipped": True,
            "reason": reason,
            "probe": probe_info,
        }

    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))
    if not failures:
        marker.write_text("ok\n")
    print(f"[done] manuscript extras ({failures} failures) -> {outdir}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
