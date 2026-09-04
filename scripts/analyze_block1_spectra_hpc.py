#!/usr/bin/env python3
"""
Draft Fig2 C/D data: per-channel power spectra of the LAST conv/flex layer in
block 1, over N random ImageNet-1k TRAIN images (draft: 512 images x 64
channels = 32,768 activations).

Faithful to the original pipeline (src/analysis/spectral.py, which cannot run
any more because load_experiment_model's RunLoader was retired):
  - forward hook on the layer output; torch.fft.fftn over dims (-2,-1),
    fftshift, |.|^2
  - per-channel radial profile via spectral_utils.get_radial_profile
    (mean-normalized, the draft's "Power" scale)
  - slope via spectral_utils.compute_slope (log-log fit over 0 < k <= nyquist,
    y > 1e-12)

Model loading follows analyze_manuscript_extras_hpc.py
(cli_tool.get_model_for_experiment on a restored iso run dir). Train images
are picked WITHOUT an ImageFolder tree scan (Lustre-safe): a seeded rng samples
(class, file-rank) pairs and only the chosen class dirs are listed.

Output: <run_dir>/results/block1_spectra/block1_spectra.npz
  profiles (N*C, L) float32, slopes (N*C,) float32, layer_name, files, meta.
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# --- copied VERBATIM from src/flex_neurons/utils/spectral_utils.py (the HPC
# --- deployed tree predates the reorg and lacks that module; inlining keeps
# --- this job dependency-free and byte-identical to the original math)
def get_radial_profile(power_spectrum, normalize=True):
    h, w = power_spectrum.shape
    cy, cx = h // 2, w // 2
    y, x = np.indices((h, w))
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2).astype(int)
    tbin = np.bincount(r.ravel(), power_spectrum.ravel())
    nr = np.bincount(r.ravel())
    non_zero = nr > 0
    radialprofile = tbin[non_zero] / nr[non_zero]
    if normalize and len(radialprofile) > 0:
        mean_power = radialprofile.mean()
        if mean_power > 1e-12:
            radialprofile /= mean_power
    return radialprofile


def compute_slope(profile, nyquist_limit):
    freqs = np.arange(len(profile))
    mask = (freqs > 0) & (freqs <= nyquist_limit)
    x_fit = freqs[mask]
    y_fit = profile[mask]
    valid_mask = y_fit > 1e-12
    x_fit = x_fit[valid_mask]
    y_fit = y_fit[valid_mask]
    if len(x_fit) < 2:
        return np.nan, np.nan, x_fit, y_fit
    log_x = np.log(x_fit)
    log_y = np.log(y_fit)
    slope, intercept = np.polyfit(log_x, log_y, 1)
    return slope, intercept, x_fit, y_fit


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def block1_last_layer(model: nn.Module):
    """Last Conv2d/Flex2D DIRECT child of model.features before the first
    top-level MaxPool2d — the draft's 'last convolution layer in block 1'
    (64 channels), robust to the different flex/vanilla submodule layouts."""
    feats = getattr(model, "features", None)
    if feats is None:
        raise RuntimeError("model has no .features sequential")
    last = None
    for name, child in feats.named_children():
        if isinstance(child, nn.MaxPool2d):
            break
        cls = child.__class__.__name__
        if "Conv2d" in cls or "Flex" in cls:
            last = (f"features.{name}", child)
    if last is None:
        raise RuntimeError("no conv/flex layer found before the first MaxPool2d")
    return last


def pick_train_images(train_dir: Path, n: int, seed: int) -> list[Path]:
    classes = sorted(d.name for d in train_dir.iterdir() if d.is_dir())
    if not classes:
        raise FileNotFoundError(f"no class dirs under {train_dir}")
    rng = np.random.default_rng(seed)
    listing_cache: dict[str, list[str]] = {}
    files = []
    for _ in range(n):
        wnid = classes[int(rng.integers(len(classes)))]
        if wnid not in listing_cache:
            listing_cache[wnid] = sorted(
                p.name for p in (train_dir / wnid).iterdir() if p.is_file())
        names = listing_cache[wnid]
        if not names:
            raise FileNotFoundError(f"empty class dir {train_dir / wnid}")
        files.append(train_dir / wnid / names[int(rng.integers(len(names)))])
    return files


def load_batch(paths: list[Path]) -> torch.Tensor:
    import torchvision.transforms as T
    from PIL import Image

    tf = T.Compose([
        T.Resize(256), T.CenterCrop(224), T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return torch.stack([tf(Image.open(p).convert("RGB")) for p in paths])


def main() -> int:
    from cli_tool import get_model_for_experiment  # lazy: heavy import chain

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--train-dir", default=None,
                    help="ImageNet-1k train split root (wnid subdirs); the "
                         "in-process listing this triggers WEDGES on the fs8 "
                         "Lustre client -- prefer --file-list + --files-root")
    ap.add_argument("--file-list", default=None,
                    help="pre-picked wnid/filename lines (pick_train_files.py); "
                         "skips all directory listing")
    ap.add_argument("--files-root", default=None,
                    help="root the --file-list paths resolve against (a "
                         "node-local stage)")
    ap.add_argument("--probe-image", default=None,
                    help="optional probe JPEG: its activation at the hooked "
                         "layer + per-channel mean slopes are saved for the "
                         "fig2E quantile-of-slope tiles")
    ap.add_argument("--num-images", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if not args.file_list and not args.train_dir:
        ap.error("give --file-list/--files-root (preferred) or --train-dir")

    run_dir = Path(args.run_dir)
    outdir = run_dir / "results" / "block1_spectra"
    outdir.mkdir(parents=True, exist_ok=True)
    marker = outdir / ".done"
    if marker.exists() and not args.force:
        print(f"[skip] already done ({marker})")
        return 0

    device = args.device if torch.cuda.is_available() else "cpu"
    model = get_model_for_experiment(str(run_dir))
    model.to(device).eval()

    layer_name, layer = block1_last_layer(model)
    print(f"[layer] {layer_name} ({layer.__class__.__name__})", flush=True)

    if args.file_list:
        root = Path(args.files_root or ".")
        files = [root / ln.strip()
                 for ln in Path(args.file_list).read_text().splitlines()
                 if ln.strip()]
    else:
        files = pick_train_images(Path(args.train_dir), args.num_images, args.seed)
    print(f"[images] {len(files)} train images (seed {args.seed})", flush=True)

    captured: list[torch.Tensor] = []

    def hook(_m, _i, out):
        captured.append(out.detach())

    handle = layer.register_forward_hook(hook)

    profiles: list[np.ndarray] = []
    slopes: list[float] = []
    failures = 0
    nyquist = None
    with torch.no_grad():
        for i in range(0, len(files), args.batch_size):
            batch_paths = files[i:i + args.batch_size]
            try:
                images = load_batch(batch_paths).to(device)
            except OSError as e:
                failures += len(batch_paths)
                print(f"[warn] batch load failed ({e}); skipping", file=sys.stderr)
                continue
            captured.clear()
            model(images)
            acts = captured[-1].float()                       # (B, C, H, W)
            fft = torch.fft.fftn(acts, dim=(-2, -1))
            power = torch.abs(torch.fft.fftshift(fft, dim=(-2, -1))) ** 2
            power_np = power.cpu().numpy()
            h, w = power_np.shape[-2:]
            nyquist = min(h, w) // 2
            for img_pow in power_np:                          # (C, H, W)
                for ch_pow in img_pow:
                    if np.isnan(ch_pow).any():
                        failures += 1
                        continue
                    prof = get_radial_profile(ch_pow)         # mean-normalized
                    profiles.append(prof.astype(np.float32))
                    s, _, _, _ = compute_slope(prof, nyquist)
                    slopes.append(float(s))
            done = min(i + args.batch_size, len(files))
            print(f"[progress] {done}/{len(files)} images, "
                  f"{len(profiles)} profiles", flush=True)
    # fig2E probe capture MUST happen while the hook is still registered
    # (jobs 6070983/6070984 crashed on captured[-1] because the probe forward
    # ran after handle.remove())
    probe_featmap = None
    if args.probe_image:
        try:
            with torch.no_grad():
                probe = load_batch([Path(args.probe_image)]).to(device)
                captured.clear()
                model(probe)
            probe_featmap = captured[-1].float().cpu().numpy()
            print(f"[probe] featmap {probe_featmap.shape}", flush=True)
        except OSError as e:
            print(f"[warn] probe image failed ({e})", file=sys.stderr)
    handle.remove()

    if not profiles:
        print("[error] no profiles computed", file=sys.stderr)
        return 1

    P = np.stack(profiles)
    S = np.asarray(slopes, dtype=np.float32)
    valid = S[~np.isnan(S)]

    # fig2E quantile-of-slope inputs: the probe activation at THIS layer plus
    # per-channel mean slopes over the train images (stable ranking; only
    # valid when no channel was dropped, so the (image, channel) rectangle
    # is intact)
    extra = {}
    if probe_featmap is not None:
        extra["probe_featmap"] = probe_featmap
        n_ch = probe_featmap.shape[1]
        if failures == 0 and len(S) % n_ch == 0:
            extra["channel_mean_slopes"] = np.nanmean(
                S.reshape(-1, n_ch), axis=0).astype(np.float32)
            print(f"[probe] per-channel mean slopes over "
                  f"{len(S) // n_ch} images", flush=True)

    meta = {
        "run_dir": str(run_dir), "layer_name": layer_name,
        "layer_class": layer.__class__.__name__,
        "num_images": args.num_images, "seed": args.seed,
        "nyquist": int(nyquist), "n_profiles": int(P.shape[0]),
        "n_valid_slopes": int(valid.size), "n_failures": int(failures),
        "mean_slope": float(np.mean(valid)), "std_slope": float(np.std(valid)),
        "train_dir": str(args.train_dir),
        "normalize": "get_radial_profile mean-normalized (original pipeline)",
    }
    np.savez_compressed(
        outdir / "block1_spectra.npz", profiles=P, slopes=S,
        layer_name=np.array(layer_name),
        files=np.array([str(f) for f in files]),
        meta=np.array(json.dumps(meta)), **extra)
    marker.write_text(json.dumps(meta, indent=2))
    print(f"[done] {P.shape[0]} profiles, mu={meta['mean_slope']:.3f} "
          f"sigma={meta['std_slope']:.3f} -> {outdir / 'block1_spectra.npz'}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise
