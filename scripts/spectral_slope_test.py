#!/usr/bin/env python3
"""
Paired statistical test for the Fig 2C/D spectral-slope difference.

The 32,768 per-map slopes (512 images x 64 channels, per network) are not
independent: maps within an image share content, and channels share every
image. The test therefore pairs at the image level, the independent sampling
unit: both networks were probed on the SAME 512 training images (verified
here), so each image yields one per-image mean slope per network and one
paired difference. Reported: paired t-test, Wilcoxon signed-rank, a
sign-flip permutation test (10,000 draws, seed 0), and Cohen's d_z.

Maps whose radial profile had fewer than two positive points carry no slope
(NaN) and are excluded from the per-image means; the count is reported.

Usage:
  python scripts/spectral_slope_test.py \
      [--flex data/iso_analysis_staged/spectra512/vgg16-flex-e89.npz] \
      [--vanilla data/iso_analysis_staged/spectra512/vgg16-vanilla-e70.npz]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy import stats

N_PERMUTATIONS = 10_000
PERMUTATION_SEED = 0


def load_slopes(path: Path) -> tuple[np.ndarray, list[str]]:
    """Return (slopes, relative image paths) from a block1-spectra npz."""
    z = np.load(path, allow_pickle=True)
    slopes = np.asarray(z["slopes"], dtype=float)
    # Strip the per-job staging prefix (/tmp/block1_spectra_stage_<jobid>/files/)
    # so image identity is comparable across the two jobs.
    rel = [str(f).split("/files/", 1)[-1] for f in z["files"]]
    if slopes.size % len(rel) != 0:
        raise ValueError(f"{path}: {slopes.size} slopes not divisible by "
                         f"{len(rel)} images")
    return slopes, rel


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    repo = Path(__file__).resolve().parent.parent
    ap.add_argument("--flex", type=Path, default=repo /
                    "data/iso_analysis_staged/spectra512/vgg16-flex-e89.npz")
    ap.add_argument("--vanilla", type=Path, default=repo /
                    "data/iso_analysis_staged/spectra512/vgg16-vanilla-e70.npz")
    args = ap.parse_args()

    try:
        sf, rf = load_slopes(args.flex)
        sv, rv = load_slopes(args.vanilla)
    except (OSError, KeyError, ValueError) as e:
        print(f"[error] {e}", file=sys.stderr)
        return 2
    if rf != rv:
        print("[error] the two npz files probe different image lists; "
              "the paired design is invalid", file=sys.stderr)
        return 2

    n_img = len(rf)
    F = sf.reshape(n_img, -1)
    V = sv.reshape(n_img, -1)
    print(f"paired design: {n_img} shared images x {F.shape[1]} channels")
    print(f"NaN maps excluded: flex {int(np.isnan(F).sum())}, "
          f"vanilla {int(np.isnan(V).sum())} (of {F.size} each)")

    fi, vi = np.nanmean(F, axis=1), np.nanmean(V, axis=1)
    d = fi - vi
    print(f"per-image mean slope: flex {fi.mean():+.3f} (sd {fi.std():.3f}), "
          f"vanilla {vi.mean():+.3f} (sd {vi.std():.3f})")
    print(f"paired difference (flex - vanilla): mean {d.mean():+.4f}, "
          f"sd {d.std(ddof=1):.4f}; flex steeper on {(d < 0).sum()}/{n_img}")

    t, pt = stats.ttest_rel(fi, vi)
    w, pw = stats.wilcoxon(fi, vi)
    rng = np.random.default_rng(PERMUTATION_SEED)
    null = (rng.choice([-1.0, 1.0], size=(N_PERMUTATIONS, n_img)) * d).mean(1)
    p_perm = float((np.abs(null) >= abs(d.mean())).mean())
    p_perm_str = (f"< {1.0 / N_PERMUTATIONS:g}" if p_perm == 0.0
                  else f"= {p_perm:g}")
    print(f"paired t-test:        t({n_img - 1}) = {t:.2f}, p = {pt:.3e}")
    print(f"Wilcoxon signed-rank: W = {w:.0f}, p = {pw:.3e}")
    print(f"sign-flip permutation ({N_PERMUTATIONS} draws, seed "
          f"{PERMUTATION_SEED}): p {p_perm_str}")
    print(f"effect size (Cohen's d_z): {d.mean() / d.std(ddof=1):.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
