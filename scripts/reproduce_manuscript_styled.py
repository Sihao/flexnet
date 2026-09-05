#!/usr/bin/env python3
"""
Render the manuscript figures (Figs 1-6) in their EXACT house style, for one
iso-accuracy flex-vs-baseline VGG16 pair (e.g. ISO0.60: flex epoch 89 vs
vanilla epoch 70).

House style (read off the compiled manuscript):
  baseline/vanilla = solid navy line + filled circles ; flex = dashed orange
  line + filled squares ; axes use x10^-1 offset scaling ; chip labels
  ("Baseline"/"Flex") ; plasma colormap for feature maps (2E), shaded surfaces
  for loss landscapes (4C) ; 3x5 alphabetical corruption grid (3A) ;
  1x4 attack row (3B) ; 2x2 per-layer grid (5C) ; routing statistics (6).

Data: every input is read from a staging directory (default
/tmp/manuscript_repro, override with the MANUSCRIPT_RP environment variable)
that scripts/render_figures.sh assembles from data/. Inputs: frequency npz,
perturbation + attack json, hessian eigenvalues npy, block1_spectra npz,
loss-surface npy, brain-score jsons, metrics.jsonl + full_val_epoch*.json
training logs, flex_selection_stats json/npz. Nothing here loads torch.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# House typography: Helvetica for every glyph, math text included. On systems
# without a licensed Helvetica the stack resolves to URW Nimbus Sans, the
# metrically identical Helvetica clone shipped with Ghostscript/TeX.
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Helvetica Neue", "Arial",
                        "Nimbus Sans", "Liberation Sans"],
    "mathtext.fontset": "custom",
    "mathtext.rm": "sans",
    "mathtext.it": "sans:italic",
    "mathtext.bf": "sans:bold",
    "mathtext.cal": "sans",
    # PDF/SVG keep text and lines vector; this sets the sampling resolution
    # of embedded raster content (feature maps, loss surfaces, schematic).
    # Without it the vector backends resample images at 100 ppi.
    "savefig.dpi": 600,
})
from matplotlib.colors import LightSource
from matplotlib.ticker import FuncFormatter
from PIL import Image
from scipy.interpolate import griddata
from scipy.stats import wasserstein_distance

BASE = "#26263f"   # baseline / vanilla  (dark navy)
FLEX = "#F2A900"   # flex                (orange)
RP = Path(os.environ.get("MANUSCRIPT_RP", "/tmp/manuscript_repro"))

BENCH = {"V1": "FreemanZiemba2013.V1.public-pls",
         "V2": "FreemanZiemba2013.V2.public-pls",
         "V4": "MajajHong2015.public.V4-pls",
         "IT": "MajajHong2015.public.IT-pls"}
LAYERS7 = ["layer1.0.conv1", "layer1.2.conv3", "layer2.3.conv3", "layer3.1.conv3",
           "layer3.5.conv3", "layer4.0.conv3", "layer4.2.conv3"]
# real ResNet-50 module names + true depth = cumulative bottleneck-block index
# across the 16 blocks (stages [3,4,6,3]); positions the probes by architecture.
LAYER_TICKS = ["1.0\nc1", "1.2\nc3", "2.3\nc3", "3.1\nc3", "3.5\nc3", "4.0\nc3", "4.2\nc3"]
LAYER_DEPTH = np.array([0, 2, 6, 8, 12, 13, 15])
# manuscript "Layer ID" (block.position) for each VGG16 conv features index --
# the exact mapping the draft's Fig 5C x axis uses (plotting._map_vgg_layer_label)
VGG_LAYER_ID = {0: "1.1", 3: "1.2", 7: "2.1", 10: "2.2", 14: "3.1", 17: "3.2",
                20: "3.3", 24: "4.1", 27: "4.2", 30: "4.3", 34: "5.1",
                37: "5.2", 40: "5.3"}
CORR_ALPHA = ["brightness", "contrast", "defocus_blur", "elastic_transform", "fog",
              "frost", "gaussian_noise", "glass_blur", "impulse_noise",
              "jpeg_compression", "motion_blur", "pixelate", "shot_noise", "snow",
              "zoom_blur"]


# ------------------------------------------------------------------ style helpers
def chip(ax, text):
    ax.annotate(text, xy=(0.5, 1.02), xycoords="axes fraction", ha="center",
                va="bottom", fontsize=8.5, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", fc="#e9e9e9", ec="#9a9a9a"))


def sci_y(ax, p=-1):
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 10.0**p:.0f}"))
    ax.text(0.0, 1.0, f"x10$^{{{p}}}$", transform=ax.transAxes, fontsize=7,
            va="bottom", ha="left")


def series(ax, x, yb, yf):
    ax.plot(x, yb, "-o", color=BASE, ms=4, lw=1.4, label="Baseline")
    ax.plot(x, yf, "--s", color=FLEX, ms=4, lw=1.4, label="Flex")


def radial(power2d):
    h, w = power2d.shape
    cy, cx = h // 2, w // 2
    y, x = np.indices((h, w))
    r = np.hypot(x - cx, y - cy).astype(int)
    return np.bincount(r.ravel(), power2d.ravel()) / np.maximum(np.bincount(r.ravel()), 1)


def slope_of(P):
    k = np.arange(1, len(P)); Pk = P[1:]
    m = Pk > 0
    return np.polyfit(np.log10(k[m]), np.log10(Pk[m]), 1)[0]


def freq_maps(npz):
    z = np.load(npz, allow_pickle=True)
    return [np.asarray(z[k]) for k in z.keys() if np.asarray(z[k]).ndim == 2]


# ------------------------------------------------------------------ Fig 1
def fig1(tag, n_classes=1000):
    """Draft Fig 1 schematic, redrawn in vector form. (A) flexible-layer streams,
    (B) element-wise max dot grids, (C) VGG16 stack. Purely conceptual: the only
    checkpoint-dependent element is the classifier head (1x100 in the draft's
    ImageNet-100 -> 1x1000 here). Geometry measured off main.pdf p12 at 300 dpi."""
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

    # canvas drawn in 6.7x3.5 design units, scaled uniformly to the common
    # 7.4-inch figure width shared by all manuscript figures
    fig = plt.figure(figsize=(7.4, 3.5 * 7.4 / 6.7))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 6.7); ax.set_ylim(0, 3.5)
    ax.set_aspect("equal"); ax.axis("off")

    def rbox(x, y, w, h, fc, ec, lw=0.8, r=0.035, zorder=2):
        p = FancyBboxPatch((x, y), w, h, fc=fc, ec=ec, lw=lw, zorder=zorder,
                           boxstyle=f"round,pad=0,rounding_size={r}")
        ax.add_patch(p); return p

    def elbow(pts, head):
        # polyline arrow: plain segments, arrowhead only on the last leg
        xs, ys = zip(*pts)
        ax.plot(xs, ys, color="black", lw=1.0, solid_capstyle="round", zorder=1)
        ax.add_patch(FancyArrowPatch(pts[-1], head, arrowstyle="-|>", color="black",
                                     lw=1.0, mutation_scale=8, zorder=1,
                                     shrinkA=0, shrinkB=0))

    # ---- Panel A: input -> {conv, pooling} stacks -> MAX -> output
    ax.text(0.06, 3.30, "A", fontsize=11, fontweight="bold")
    sq, off = 0.44, 0.05
    rbox(0.10, 2.13, 0.42, 0.42, "#e2e2e2", "#555555")
    ax.text(0.36, 2.60, "Input", fontsize=9, ha="center")
    conv_shades = ["#151b3d", "#1e2757", "#2b3775", "#3a4a9c", "#5165c4"]
    for i, c in enumerate(conv_shades):  # back -> front
        d = (4 - i) * off
        rbox(0.86 - d, 2.58 + d, sq, sq, c, "black")
    ax.text(1.08, 3.28, "Convolution", fontsize=9, ha="center")
    for i in range(5):
        d = (4 - i) * off
        rbox(0.72 - d, 1.62 + d, sq, sq, "#f5a728", "black")
    ax.text(0.94, 2.32, "Pooling", fontsize=9, ha="center")
    out_shades = ["#181d3a", "#262c55", "#3b3a68", "#565177"]
    for i, c in enumerate(out_shades):
        d = (4 - i) * off
        rbox(2.10 - d, 2.16 + d, sq, sq, c, "black")
    grad_cmap = LinearSegmentedColormap.from_list(
        "flexgrad", ["#1d2452", "#4a4e7c", "#8a6a76", "#d78b3f", "#f2a437"])
    gx, gy = np.meshgrid(np.linspace(0, 1, 64), np.linspace(0, 1, 64))
    front = rbox(2.10, 2.16, sq, sq, "none", "black", zorder=4)
    im = ax.imshow((gx + gy) / 2, cmap=grad_cmap, extent=(2.10, 2.10 + sq, 2.16, 2.16 + sq),
                   origin="upper", zorder=3, interpolation="bilinear")
    im.set_clip_path(front)
    ax.text(2.36, 2.87, "Output", fontsize=9, ha="center")
    max_c = Circle((1.52, 2.38), 0.155, fc="#e8e8e8", ec="#555555", lw=0.9, zorder=2)
    ax.add_patch(max_c)
    ax.text(1.52, 2.38, "MAX", fontsize=6.2, ha="center", va="center")
    elbow([(0.18, 2.57), (0.18, 2.90)], (0.60, 2.90))          # input -> conv
    elbow([(0.18, 2.11), (0.18, 1.86)], (0.46, 1.86))          # input -> pooling
    elbow([(1.34, 2.85), (1.52, 2.85)], (1.52, 2.56))          # conv -> MAX
    elbow([(1.18, 1.88), (1.52, 1.88)], (1.52, 2.20))          # pooling -> MAX
    elbow([(1.70, 2.38)], (1.88, 2.38))                        # MAX -> output

    # ---- Panel B: 9x9 dot grids (conv values, pool values, element-wise max)
    ax.text(0.06, 1.52, "B", fontsize=11, fontweight="bold")
    cm_conv = LinearSegmentedColormap.from_list(
        "dotconv", ["#dfe4f6", "#93a1d8", "#2e3f96"])
    cm_pool = LinearSegmentedColormap.from_list(
        "dotpool", ["#fdf0d8", "#f8c96d", "#f09d1c"])
    rng = np.random.default_rng(89)  # flex epoch of the iso pair, fixed for reproducibility
    cvals = rng.uniform(0.25, 1.0, size=(9, 9))   # draft's conv grid skews dark
    pvals = rng.uniform(0.10, 0.85, size=(9, 9))
    pitch, y_top = 0.0875, 1.16
    grids = [(0.10, cm_conv(cvals), "Convolution"),
             (1.02, cm_pool(pvals), "Pooling"),
             (1.94, np.where((cvals >= pvals)[..., None], cm_conv(cvals), cm_pool(pvals)),
              "Element-wise\nmax pooling")]
    for x0, colors, label in grids:
        xx, yy = np.meshgrid(x0 + np.arange(9) * pitch, y_top - np.arange(9) * pitch)
        ax.scatter(xx, yy, s=21, c=colors.reshape(-1, 4), edgecolors="black",
                   linewidths=0.45, zorder=2)
        ax.text(x0 + 4 * pitch, 0.335, label, fontsize=9, ha="center", va="top")
    mt = dict(fontsize=9, ha="center")
    ax.text(0.45, 1.29, r"$f_{conv}(i,j,k)$", **mt)
    ax.text(1.37, 1.29, r"$f_{max}(i,j,k)$", **mt)
    ax.text(2.29, 1.44, r"$max(f_{conv}(i,j,k),$", **mt)
    ax.text(2.36, 1.28, r"$f_{max}(i,j,k))$", **mt)

    # ---- Panel C: VGG16 stack (widths taper; MaxPool bars a hair wider)
    ax.text(2.80, 3.30, "C", fontsize=11, fontweight="bold")
    C_IN, C_MP, C_FC = ("#ccd3e8", "#8e99b8"), ("#f9d8ec", "#d193bd"), ("#d8edcf", "#99c489")
    cx, lx = 3.92, 5.52
    h, gap, aslot = 0.10, 0.02, 0.105
    # conv-row sizes are the PRE-pool activation shapes: Flex2D runs at
    # stride 1 (flex.py), so spatial size drops only at each block's
    # MaxPool2d(2,2) -- verified against the e89 routing-stats p_elem shapes
    blocks = [("input", [("224x224x3", 2.18, ("#d8d8d8", "#999999"))], "Input"),
              ("b1", [("224x224x64", 1.80, C_IN)] * 2 + [("MaxPool", 1.90, C_MP)],
               "Flex2D\nBlock 1"),
              ("b2", [("112x112x128", 1.55, C_IN)] * 2 + [("MaxPool", 1.65, C_MP)],
               "Flex2D\nBlock 2"),
              ("b3", [("56x56x256", 1.35, C_IN)] * 3 + [("MaxPool", 1.45, C_MP)],
               "Flex2D\nBlock 3"),
              ("b4", [("28x28x512", 1.12, C_IN)] * 3 + [("MaxPool", 1.22, C_MP)],
               "Flex2D\nBlock 4"),
              ("b5", [("14x14x512", 0.92, C_IN)] * 3 + [("MaxPool", 1.02, C_MP)],
               "Flex2D\nBlock 5"),
              ("fc", [("1x4096", 1.00, C_FC)] * 2, "Fully connected\nblock"),
              ("out", [(f"1x{n_classes}", 1.00, C_FC)], "Output")]
    y = 3.30
    for gi, (_, bars, label) in enumerate(blocks):
        y0 = y; y_conv_end = y
        for bi, (txt, w, (fc, ec)) in enumerate(bars):
            rbox(cx - w / 2, y - h, w, h, fc, ec, lw=0.7, r=0.045)
            ax.text(cx, y - h / 2, txt, fontsize=6.8, style="italic",
                    ha="center", va="center", color="#222222")
            y -= h + (gap if bi < len(bars) - 1 else 0)
            if txt != "MaxPool":
                y_conv_end = y
        # the draft centers each block label on the conv bars, above the MaxPool line
        ax.text(lx, (y0 + y_conv_end) / 2, label, fontsize=9, ha="center", va="center")
        if gi < len(blocks) - 1:
            ax.add_patch(FancyArrowPatch((cx, y - 0.012), (cx, y - aslot + 0.012),
                                         arrowstyle="-|>", color="black", lw=1.0,
                                         mutation_scale=7, shrinkA=0, shrinkB=0))
            y -= aslot

    out = f"manuscript_styled_fig1_{tag}.png"
    fig.savefig(out, dpi=250)
    fig.savefig(f"manuscript_styled_fig1_{tag}.svg")
    fig.savefig(f"manuscript_styled_fig1_{tag}.pdf")
    plt.close(fig); print(f"[fig1] -> {out} (+svg +pdf)")


# ------------------------------------------------------------------ Fig 2
def fig2(tag, vanilla_metrics=None, flex_metrics=None):
    """Draft-exact Fig 2: A/B training dynamics (10-epoch sliding mean +- std),
    C/D per-channel activation power spectra + slope histograms (from the extras
    featmap of the first flex/conv layer, 64 channels, axolotl probe), E feature
    map strips with rotated side chips. Same style grammar as the fig3 pass:
    despined, labelsize 7, no suptitle, no grid, shared axes within panel pairs."""

    def sane(v):
        # Mask the CE-overflow artifact (lessons.md 2026-08-29): the flex
        # full-val loss path overflows on 79/90 epochs (217 .. 1e20). A
        # 1000-class CE starts at ln(1000)=6.9; anything above 20 is broken.
        return v if (isinstance(v, (int, float)) and 0 <= v < 20) else np.nan

    def metrics(f):
        # metrics.jsonl logs many mini-batch rows per epoch -> average them so
        # the training curves are per-epoch means, not last-batch noise.
        rows = [json.loads(l) for l in Path(f).read_text().splitlines() if l.strip()]
        ep = {}
        for r in rows:
            e = r.get("Epoch")
            if e is None:
                continue
            b = ep.setdefault(e, {"tl": [], "ta": [], "vl": set()})
            if isinstance(r.get("Train Loss"), (int, float)):
                b["tl"].append(sane(r["Train Loss"]))
            if isinstance(r.get("Train Accuracy"), (int, float)):
                b["ta"].append(r["Train Accuracy"])
            if isinstance(r.get("Valid Loss"), (int, float)):
                b["vl"].add(r["Valid Loss"])  # repeated across rows -> dedupe
        E = sorted(ep)
        tl = np.array([np.nanmean(ep[e]["tl"]) if ep[e]["tl"] else np.nan for e in E])
        ta = np.array([np.nanmean(ep[e]["ta"]) if ep[e]["ta"] else np.nan for e in E])
        # Validation loss: mean of the epoch's distinct logged validation losses
        # (the flex full-val loss path overflows on most epochs, so the full_val
        # jsons are NOT usable for loss; the jsonl values are sane, just sparse).
        def mean_sane(vals):
            s = [sane(v) for v in vals]
            s = [v for v in s if not np.isnan(v)]
            return float(np.mean(s)) if s else np.nan

        vl = np.array([mean_sane(ep[e]["vl"]) for e in E])
        # TRUE validation accuracy: per-epoch full-val evaluations (n=50000)
        # written as full_val_epoch<N>.json next to metrics.jsonl. The jsonl's
        # own "Valid Accuracy" is a single 64-image mini-batch (e.g. 0.90625 =
        # 58/64) and must never be plotted as validation.
        logs = Path(f).resolve().parent
        fv = {}
        for p in logs.glob("full_val_epoch*.json"):
            try:
                d = json.loads(p.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(d.get("epoch"), int):
                fv[d["epoch"]] = d.get("top1", np.nan)
        if not fv:
            raise FileNotFoundError(f"no full_val_epoch*.json under {logs}; "
                                    "refusing to plot mini-batch 'Valid Accuracy' "
                                    "as validation")
        va = np.array([fv.get(e, np.nan) for e in E], dtype=float)
        return np.array(E), tl, ta, vl, va

    def roll(y, w=10):
        y = np.asarray(y, float)
        mu = np.full(len(y), np.nan); sd = np.full(len(y), np.nan)
        for i in range(len(y)):
            seg = y[max(0, i - w + 1):i + 1]
            seg = seg[~np.isnan(seg)]
            if seg.size:
                mu[i] = seg.mean(); sd[i] = seg.std()
        return mu, sd

    def despine(ax):
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=7, length=2.5)

    def chip_top(ax, text):
        # chip whose BOX top aligns with the top edge of the panel letters
        # (letters sit at baseline y=1.14; cap top ~0.111in above, minus the
        # chip bbox pad ~0.035in -> text top 0.076in above the 1.14 line)
        h_in = ax.get_position().height * fig.get_figheight()
        y = 1.14 + 0.076 / h_in
        ax.annotate(text, xy=(0.5, y), xycoords="axes fraction", ha="center",
                    va="top", fontsize=8.5, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.3", fc="#e9e9e9",
                              ec="#9a9a9a"))

    def band(ax, x, y, color, label=None):
        mu, sd = roll(y)
        ax.fill_between(x, mu - sd, mu + sd, color=color, alpha=0.25, lw=0)
        ax.plot(x, mu, color=color, lw=1.2, label=label)
        return np.nanmax(mu + sd)

    # original pipeline math (src/flex_neurons/utils/spectral_utils.py):
    # mean-normalized radial profile + log-log slope fit over 0 < k <= nyquist
    def orig_profile(power2d):
        p = radial(power2d)
        m = p.mean()
        return p / m if m > 1e-12 else p

    def orig_slope(profile, nyquist):
        k = np.arange(len(profile))
        m = (k > 0) & (k <= nyquist)
        x, y = k[m], profile[m]
        keep = y > 1e-12
        x, y = x[keep], y[keep]
        if len(x) < 2:
            return np.nan
        return float(np.polyfit(np.log(x), np.log(y), 1)[0])

    def spectra_source(side):
        """(profiles, slopes, nyquist): the 512-train-image job output when
        staged (draft-faithful), else the single-probe featmap channels."""
        z = RP / side / "block1_spectra.npz"
        if z.exists():
            d = np.load(z, allow_pickle=True)
            profs = np.asarray(d["profiles"], dtype=float)
            slopes = np.asarray(d["slopes"], dtype=float)
            ny = int(json.loads(str(d["meta"]))["nyquist"])
            return profs, slopes[~np.isnan(slopes)], ny
        dd = RP / f"{side}_extras"
        fp = dd / "featmap_features_0.npy"
        if not fp.exists():
            fp = next(p for p in sorted(dd.glob("featmap_*.npy"))
                      if np.load(p).ndim == 4)
        fm = np.load(fp)[0].astype(float)  # (C, H, W)
        ny = fm.shape[-1] // 2
        profs, slopes = [], []
        for ch in fm:
            P = np.abs(np.fft.fftshift(np.fft.fft2(ch))) ** 2
            prof = orig_profile(P)
            profs.append(prof)
            slopes.append(orig_slope(prof, ny))
        return np.stack(profs), np.asarray(slopes), ny

    vanilla_metrics = vanilla_metrics or RP / "vanilla_metrics.jsonl"
    flex_metrics = flex_metrics or RP / "flex_metrics.jsonl"

    fig = plt.figure(figsize=(7.4, 4.6))
    # A/B and C/D get separate gridspecs: C/D's top row carries an xlabel
    # ("Spatial Frequency"), so its row gap must be wider than A/B's
    gsT = fig.add_gridspec(2, 2, left=0.075, right=0.500, top=0.90, bottom=0.445,
                           hspace=0.30, wspace=0.30)
    gsCD = fig.add_gridspec(2, 2, left=0.575, right=0.985, top=0.90, bottom=0.445,
                            hspace=0.62, wspace=0.30)
    # panel E: input column left-aligned with panel A's axes (0.075), then a
    # clear gap before the 10 activation tiles. All 11 cells share one width:
    # 11.45*w + gap = span (10 tiles + 9 tile-tile gaps of 0.05*w).
    E_L, E_R, E_GAP = 0.075, 0.985, 0.018
    e_w = (E_R - E_L - E_GAP) / 11.45
    gsE0 = fig.add_gridspec(2, 1, left=E_L, right=E_L + e_w, top=0.335,
                            bottom=0.025, hspace=0.07)
    gsE1 = fig.add_gridspec(2, 10, left=E_L + e_w + E_GAP, right=E_R, top=0.335,
                            bottom=0.025, hspace=0.07, wspace=0.05)

    # --- A/B training dynamics (10-epoch sliding mean +- std, shared ylims) ---
    mets = [metrics(vanilla_metrics), metrics(flex_metrics)]
    loss_max = 0.0
    axesAB = []
    for col, ((E, tl, ta, vl, va), name, lab) in enumerate(
            zip(mets, ("Baseline", "Flex"), ("A", "B"))):
        axl = fig.add_subplot(gsT[0, col]); axa = fig.add_subplot(gsT[1, col])
        loss_max = max(loss_max, band(axl, E, tl, BASE, "Training"))
        loss_max = max(loss_max, band(axl, E, vl, FLEX, "Validation"))
        band(axa, E, ta, BASE); band(axa, E, va, FLEX)
        leg = axl.legend(fontsize=6, frameon=False, loc="upper right",
                         handlelength=1.5, borderaxespad=0.2)
        for h in leg.legend_handles:
            h.set_linewidth(2.2)
        chip_top(axl, name)
        axa.set_ylim(0, 1.0); axa.set_yticks(np.arange(0, 1.01, 0.2))
        axa.set_xlabel("Epoch", fontsize=8)
        axl.tick_params(labelbottom=False)
        if col == 0:
            axl.set_ylabel("Loss", fontsize=8)
            axa.set_ylabel("Accuracy", fontsize=8)
        else:
            axl.tick_params(labelleft=False); axa.tick_params(labelleft=False)
        for ax in (axl, axa):
            despine(ax)
        axl.text(-0.30 if col == 0 else -0.14, 1.14, lab, transform=axl.transAxes,
                 fontsize=11, fontweight="bold")
        axesAB.append((axl, axa))
    for axl, _ in axesAB:
        axl.set_ylim(0, loss_max * 1.06)
        axl.set_yticks(np.arange(0, loss_max * 1.06, 2))

    # --- C/D per-channel power spectra + slope histograms ---
    spec_sets = [spectra_source("vanilla"), spectra_source("flex")]
    allv = np.concatenate([P[:, 1:P.shape[1] if ny is None else ny + 1].ravel()
                           for P, _, ny in spec_sets])
    allv = allv[allv > 0]
    ylo, yhi = allv.min() * 0.5, allv.max() * 2
    MAX_FAN = 1500  # draft plots a dense fan; cap line count to keep vector output sane
    hist_axes = []
    for col, ((P, slopes, ny), name, lab) in enumerate(
            zip(spec_sets, ("Baseline", "Flex"), ("C", "D"))):
        axp = fig.add_subplot(gsCD[0, col]); axh = fig.add_subplot(gsCD[1, col])
        L = ny + 1 if ny is not None else P.shape[1]
        fan = P if len(P) <= MAX_FAN else P[np.linspace(0, len(P) - 1, MAX_FAN,
                                                        dtype=int)]
        dense = len(P) > 200
        k = np.arange(1, L)
        for p in fan:
            axp.loglog(k, p[1:L], color=BASE, lw=0.25 if dense else 0.3,
                       alpha=0.04 if dense else 0.15)
        axp.loglog(k, P.mean(0)[1:L], color=FLEX, lw=1.6)
        axp.set_ylim(ylo, yhi)
        axp.set_xlabel("Spatial Frequency", fontsize=8)
        chip_top(axp, name)
        mu, sig = float(np.mean(slopes)), float(np.std(slopes))
        cnts, _, _ = axh.hist(slopes, bins=13, color="#ececec", edgecolor="0.35",
                              lw=0.7)
        axh.axvline(mu, color=FLEX, ls="--", lw=1.6)
        axh.text(0.97, 0.95, f"$\\mu$ = {mu:.2f}\n$\\sigma$ = {sig:.2f}", ha="right",
                 va="top", transform=axh.transAxes, fontsize=6.5)
        if cnts.max() >= 1000:  # draft's x10^4 count-axis offset grammar
            p10 = int(np.floor(np.log10(cnts.max())))
            axh.yaxis.set_major_formatter(
                FuncFormatter(lambda v, _, p=p10: f"{v / 10**p:g}"))
            if col == 0:
                axh.text(-0.02, 1.03, rf"$\times10^{{{p10}}}$",
                         transform=axh.transAxes, fontsize=6, va="bottom")
        axh.set_xlabel("Slope", fontsize=8)
        if col == 0:
            axp.set_ylabel("Power", fontsize=8)
            axh.set_ylabel("Count", fontsize=8)
        else:
            axp.tick_params(labelleft=False)
            axh.tick_params(labelleft=False)
        for ax in (axp, axh):
            despine(ax)
        axp.text(-0.30 if col == 0 else -0.14, 1.14, lab, transform=axp.transAxes,
                 fontsize=11, fontweight="bold")
        hist_axes.append(axh)
    hmax = max(ax.get_ylim()[1] for ax in hist_axes)
    for ax in hist_axes:
        ax.set_ylim(0, hmax * 1.2)  # headroom so the mu/sigma text clears the bars

    # --- E feature maps: quantile-of-slope selection ---
    # For each model, channels are ranked by spectral slope (the C/D metric)
    # and the tiles show the channels at even quantiles (5%..95%), steepest
    # (most low-frequency) first. This replaces the draft's top-10-by-mean-
    # activation rule: that rule measures output magnitude, not frequency
    # content, and left the two strips positionally uncomparable.
    def e_row_data(side):
        """(featmap (C,H,W), per-channel slopes) for one model row. Prefers
        the 512-image job output (same hooked layer as C/D, mean slope over
        all train images); falls back to probe-image slopes on the extras
        featmap."""
        z = RP / side / "block1_spectra.npz"
        if z.exists():
            d = np.load(z, allow_pickle=True)
            if "probe_featmap" in d.files and "channel_mean_slopes" in d.files:
                return (np.asarray(d["probe_featmap"])[0],
                        np.asarray(d["channel_mean_slopes"], dtype=float))
        dd = RP / f"{side}_extras"
        fp = dd / "featmap_features_0.npy"
        if not fp.exists():
            fp = next(p for p in sorted(dd.glob("featmap_*.npy"))
                      if np.load(p).ndim == 4)
        fm = np.load(fp)[0].astype(float)
        ny = fm.shape[-1] // 2
        sl = np.array([orig_slope(
            orig_profile(np.abs(np.fft.fftshift(np.fft.fft2(ch))) ** 2), ny)
            for ch in fm])
        return fm, sl

    def quantile_order(sl, n=10):
        valid = np.where(~np.isnan(sl))[0]
        ranked = valid[np.argsort(sl[valid])]           # steepest first
        q = np.round(np.linspace(0.05, 0.95, n) * (len(ranked) - 1)).astype(int)
        return ranked[q]

    probe_imgs = sorted(RP.glob("input_*.JPEG"))
    inp = None
    if probe_imgs:
        try:
            # original code (plotting.py visualize_activations) displays the
            # inverse-normalized network input: Resize(256) -> CenterCrop(224)
            im = Image.open(probe_imgs[0]).convert("RGB")
            s = 256 / min(im.size)
            im = im.resize((round(im.width * s), round(im.height * s)))
            lft, top = (im.width - 224) // 2, (im.height - 224) // 2
            inp = im.crop((lft, top, lft + 224, top + 224))
        except OSError:
            inp = None
    for row, (side, name) in enumerate((("vanilla", "Baseline"),
                                        ("flex", "Flex"))):
        fm, sl = e_row_data(side)
        order = quantile_order(sl)
        ax0 = fig.add_subplot(gsE0[row, 0])
        if inp is not None:
            ax0.imshow(inp)
        ax0.axis("off")
        ax0.annotate(name, xy=(-0.22, 0.5), xycoords="axes fraction", ha="center",
                     va="center", fontsize=7.5, fontweight="bold", rotation=90,
                     bbox=dict(boxstyle="round,pad=0.3", fc="#e9e9e9", ec="#9a9a9a"))
        for j, ch in enumerate(order):
            ax = fig.add_subplot(gsE1[row, j])
            ax.imshow(fm[ch], cmap="plasma"); ax.axis("off")
        if row == 0:
            ax0.text(-0.40, 1.12, "E", transform=ax0.transAxes, fontsize=11,
                     fontweight="bold")
    out = f"manuscript_styled_fig2_{tag}.png"
    fig.savefig(out, dpi=250)
    fig.savefig(f"manuscript_styled_fig2_{tag}.svg")
    fig.savefig(f"manuscript_styled_fig2_{tag}.pdf")
    plt.close(fig); print(f"[fig2] -> {out} (+svg +pdf)")


# ------------------------------------------------------------------ Fig 3
def fig3(tag):
    # EXACT draft layout (main.pdf p.14): no suptitle; Panel A = 3x5 grid,
    # ONE shared y-scale for all 15 corruptions with a single x10^-1 offset,
    # tick labels only on the left column / bottom row, one centred
    # "Severity", sentence-case titles, in-panel legends (Fog + SPSA),
    # top/right spines removed, no grid, compact spacing. This supersedes the
    # earlier per-panel y-window variant: the draft uses a common scale, so
    # panels are directly comparable by eye. The only departure from the
    # draft's literal numbers is the shared y-maximum, which is data-driven
    # (draft data capped at 0.2; full-ImageNet accuracies here reach ~0.6).
    def loadp(p):
        d = json.loads(Path(p).read_text())
        return {k.lower().replace("-", "_").replace(" ", "_"): v
                for k, v in d.items() if k != "metadata" and isinstance(v, dict)}
    fb = loadp(RP / "vanilla/perturbation_analysis/perturbation_analysis_results.json")
    ff = loadp(RP / "flex/perturbation_analysis/perturbation_analysis_results.json")
    gmax = max((v for m in (fb, ff) for c in m.values() if c.get("accuracies")
                for v in c["accuracies"]), default=0.2)
    if gmax <= 0.21:
        ylim, yticks = (0, 0.21), [0.0, 0.1, 0.2]
    elif gmax <= 0.42:
        ylim, yticks = (0, 0.42), [0.0, 0.2, 0.4]
    else:
        ylim, yticks = (0, gmax * 1.06), [0.0, 0.2, 0.4, 0.6]

    fig = plt.figure(figsize=(7.4, 5.6))
    # Panel A = adversarial attacks (headline result), Panel B = corruptions.
    gsA = fig.add_gridspec(1, 4, left=0.09, right=0.985, top=0.93, bottom=0.76,
                           wspace=0.18)
    gsB = fig.add_gridspec(3, 5, left=0.09, right=0.985, top=0.62, bottom=0.075,
                           hspace=0.55, wspace=0.18)

    def despine(ax):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=7, length=2.5)

    axB0 = None
    for i, c in enumerate(CORR_ALPHA):
        r, k = divmod(i, 5)
        ax = fig.add_subplot(gsB[r, k], sharey=axB0)
        if axB0 is None:
            axB0 = ax
        for model, col, mk, ls in ((fb, BASE, "o", "-"), (ff, FLEX, "s", "--")):
            if c in model:
                ax.plot(model[c]["severity"], model[c]["accuracies"], ls + mk,
                        color=col, ms=2.8, lw=1.1,
                        label={BASE: "Baseline", FLEX: "Flex"}[col])
        ax.set_title(c.replace("_", " ").capitalize(), fontsize=8, pad=3)
        ax.set_ylim(*ylim)
        ax.set_xticks([1, 2, 3, 4, 5])
        ax.set_xlim(0.7, 5.3)
        despine(ax)
        if k == 0:
            ax.set_yticks(yticks)
            ax.yaxis.set_major_formatter(
                FuncFormatter(lambda v, _: f"{v * 10:.0f}"))
            ax.set_ylabel("Accuracy", fontsize=8)
            ax.text(-0.22, 1.01, r"$\times10^{-1}$", transform=ax.transAxes,
                    fontsize=6, va="bottom", ha="left")
        else:
            ax.tick_params(labelleft=False)
        if r < 2:
            ax.tick_params(labelbottom=False)
        if r == 2 and k == 2:
            ax.set_xlabel("Severity", fontsize=8)
        if i == 0:
            ax.text(-0.52, 1.22, "B", transform=ax.transAxes, fontsize=12,
                    fontweight="bold")
        if r == 0 and k == 4:
            ax.legend(loc="upper right", fontsize=6.5, frameon=False,
                      handlelength=1.8, borderaxespad=0.1, labelspacing=0.3)

    # --- A: attacks 1x4, full width (headline result) ---
    fd = json.loads((next(RP.glob("flex/attack_comparison/*/attack_comparison_results.json"))).read_text())
    vd = json.loads((next(RP.glob("vanilla/attack_comparison/*/attack_comparison_results.json"))).read_text())
    # APGD/Jitter: show only the lower half of the evaluated epsilon range --
    # both models are at chance beyond it, and the flex-vs-baseline separation
    # lives entirely in this window. FGSM/SPSA keep the full range.
    HALF_RANGE = {"APGD", "Jitter"}
    axA0 = None
    for j, atk in enumerate(["FGSM", "APGD", "Jitter", "SPSA"]):
        ax = fig.add_subplot(gsA[0, j], sharey=axA0)
        if axA0 is None:
            axA0 = ax
        if atk in fd and atk in vd:
            p = None
            for d, mk, col, lab in ((vd, "-o", BASE, "Baseline"),
                                    (fd, "--s", FLEX, "Flex")):
                eps = np.asarray(d[atk]["epsilons"], dtype=float)
                acc = np.asarray(d[atk]["accuracies"], dtype=float)
                if atk in HALF_RANGE:
                    m = eps <= eps.max() / 2
                    eps, acc = eps[m], acc[m]
                if p is None and eps.max() > 0:
                    p = int(np.floor(np.log10(eps.max())))
                ax.plot(eps, acc, mk, color=col, ms=2.8, lw=1.1, label=lab)
            p = 0 if p is None else p
            ax.xaxis.set_major_formatter(
                FuncFormatter(lambda v, _, p=p: f"{v / 10.0**p:g}"))
            ax.text(1.02, -0.24, rf"$\times10^{{{p}}}$", transform=ax.transAxes,
                    fontsize=6, va="top", ha="right")
        else:
            ax.text(0.5, 0.5, "SPSA\n(not in pipeline)", ha="center",
                    va="center", transform=ax.transAxes, fontsize=8,
                    color="0.5")
        ax.set_title(atk, fontsize=8, pad=3)
        ax.set_ylim(0, 1.05)
        despine(ax)
        ax.set_xlabel(r"$\epsilon$", fontsize=8, labelpad=1)
        if j == 0:
            ax.set_yticks([0.0, 0.5, 1.0])
            ax.set_ylabel("Accuracy", fontsize=8)
            ax.text(-0.42, 1.14, "A", transform=ax.transAxes, fontsize=12,
                    fontweight="bold")
        else:
            ax.tick_params(labelleft=False)
        if j == 3 and atk in fd:
            ax.legend(loc="upper right", fontsize=6.5, frameon=False,
                      handlelength=1.8, borderaxespad=0.1, labelspacing=0.3)

    out = f"manuscript_styled_fig3_{tag}.png"
    fig.savefig(out, dpi=250)
    fig.savefig(f"manuscript_styled_fig3_{tag}.svg")
    fig.savefig(f"manuscript_styled_fig3_{tag}.pdf")
    plt.close(fig); print(f"[fig3] -> {out} (+svg +pdf)")


# ------------------------------------------------------------------ Fig 4
def fig4(tag, seed=0):
    """Draft-exact Fig 4: A input-Hessian ESD overlay, B Wasserstein
    permutation test, C 2x2 hillshaded loss surfaces (rows = the two shared
    validation probe images, columns = Baseline | Flex). Panel code follows
    the original plotting.py plot_hessian_comparison /
    plot_loss_surface_comparison + statistical_analysis_hessian.py."""
    def hess(side):
        direct = RP / side / "hessian_eigenvalues.npy"
        if direct.is_file():
            return np.load(direct).astype(float)
        return np.load(next(RP.glob(
            f"{side}/hessian_analysis/*/hessian_eigenvalues.npy"))).astype(float)
    ef, ev = hess("flex"), hess("vanilla")
    wd = float(wasserstein_distance(ef, ev))
    rng = np.random.default_rng(seed); pooled = np.concatenate([ef, ev]); nf = len(ef)
    null = np.array([wasserstein_distance(*(lambda P: (P[:nf], P[nf:]))(rng.permutation(pooled)))
                     for _ in range(1000)])
    pval = float((null >= wd).mean())
    # A permutation test cannot resolve p below 1/n_permutations: when no null
    # draw reaches the observed distance, report the bound, not "p = 0".
    p_label = (f"$p < {1.0 / len(null):g}$" if pval < 1.0 / len(null)
               else f"$p = {pval:.4f}$")

    fig = plt.figure(figsize=(7.4, 4.35))
    gsL = fig.add_gridspec(2, 1, left=0.095, right=0.355, top=0.90, bottom=0.115,
                           hspace=0.58)
    axA = fig.add_subplot(gsL[0, 0]); axB = fig.add_subplot(gsL[1, 0])
    for ax in (axA, axB):
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=7, length=2.5)
        ax.ticklabel_format(axis="both", style="sci", scilimits=(0, 0),
                            useMathText=True)
        ax.xaxis.get_offset_text().set_fontsize(6)
        ax.yaxis.get_offset_text().set_fontsize(6)
    # A -- original plot_hessian_comparison: 30 shared bins from min(.,0),
    # unit-area densities, alpha 0.9, black bar edges, baseline under flex
    bins = np.linspace(min(pooled.min(), 0.0), pooled.max(), 30)
    axA.hist(ev, bins=bins, density=True, alpha=0.9, label="Baseline",
             color=BASE, edgecolor="black", linewidth=0.5)
    axA.hist(ef, bins=bins, density=True, alpha=0.9, label="Flex",
             color=FLEX, edgecolor="black", linewidth=0.5)
    axA.set_xlabel("Eigenvalue", fontsize=8); axA.set_ylabel("Density", fontsize=8)
    axA.legend(frameon=False, fontsize=6.5)
    axA.text(-0.28, 1.06, "A", transform=axA.transAxes, fontsize=11,
             fontweight="bold")
    # B -- original plot_permutation_results: 15 bins, white bar edges,
    # dashed observed line
    axB.hist(null, bins=15, alpha=0.7, color=BASE, edgecolor="white",
             linewidth=0.5, label="Null Distribution")
    axB.axvline(wd, color=FLEX, linestyle="--", linewidth=2.5,
                label=f"Observed ({p_label})")
    axB.set_xlabel("Wasserstein Distance", fontsize=8)
    axB.set_ylabel("Count", fontsize=8)
    axB.legend(frameon=False, fontsize=6.5)
    axB.text(-0.28, 1.06, "B", transform=axB.transAxes, fontsize=11,
             fontweight="bold")

    # C -- original plot_loss_surface_comparison: cubic 300x300 upsample,
    # LightSource(315,45) plasma hillshade, 15 floor contours at
    # z_floor = z_min - 2*span, z-limits shared within a row (one image)
    def surf(dd):
        return (np.load(dd / "loss_surface_alpha.npy"),
                np.load(dd / "loss_surface_beta.npy"),
                np.load(dd / "loss_surface_z.npy"))

    def upsample(a, b, z, res=300j):
        pts = np.column_stack((a.ravel(), b.ravel()))
        ga, gb = np.mgrid[a.min():a.max():res, b.min():b.max():res]
        return ga, gb, griddata(pts, z.ravel(), (ga, gb), method="cubic")

    # Row sources, in priority order: dedicated fig4_row<N>_<side> links (the
    # principled max-curvature-ratio winners from analyze_fig4c_best_pair_hpc)
    # else the extras/extras2 probe pairs.
    rows = []
    for r in ("1", "2"):
        dv, df = RP / f"fig4_row{r}_vanilla", RP / f"fig4_row{r}_flex"
        if all((d / "loss_surface_z.npy").is_file() for d in (dv, df)):
            rows.append((dv, df))
    if not rows:
        for suf in ("", "2"):
            dv, df = RP / f"vanilla_extras{suf}", RP / f"flex_extras{suf}"
            if all((d / "loss_surface_z.npy").is_file() for d in (dv, df)):
                rows.append((dv, df))
    if not rows:
        print("[fig4] WARNING: no loss-surface pair staged; Panel C empty")
    n_rows = max(len(rows), 1)
    gsC = fig.add_gridspec(n_rows, 2, left=0.36, right=1.0, top=0.97,
                           bottom=0.0, wspace=0.0, hspace=0.0)
    top_axes = []
    for r, (dv, df) in enumerate(rows):
        panels = [upsample(*surf(d)) for d in (dv, df)]
        zmin = min(np.nanmin(gz) for _, _, gz in panels)
        zmax = max(np.nanmax(gz) for _, _, gz in panels)
        z_floor = zmin - 2.0 * (zmax - zmin)
        light = LightSource(azdeg=315, altdeg=45)
        for c, (ga, gb, gz) in enumerate(panels):
            ax = fig.add_subplot(gsC[r, c], projection="3d")
            ax.grid(False)
            for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
                axis.pane.fill = False
                axis.pane.set_linewidth(0.4)
                axis.pane.set_edgecolor("0.75")
                axis.line.set_linewidth(0.4)
                axis.line.set_color("0.75")
            ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
            rgb = light.shade(gz, cmap=plt.cm.plasma, vert_exag=0.1,
                              blend_mode="soft")
            ax.plot_surface(ga, gb, gz, facecolors=rgb, edgecolor="none",
                            linewidth=0, alpha=0.9, shade=False,
                            antialiased=False, rcount=300, ccount=300,
                            rasterized=True)
            ax.contour(ga, gb, gz, levels=15, zdir="z", offset=z_floor,
                       cmap="plasma", linewidths=0.2, alpha=0.8)
            ax.set_zlim(z_floor, zmax)
            if r == 0:
                top_axes.append(ax)
    for c, name in enumerate(("Baseline", "Flex")):
        if c < len(top_axes):
            bb = top_axes[c].get_position()
            fig.text((bb.x0 + bb.x1) / 2, 0.955, name, ha="center", va="center",
                     fontsize=8.5, fontweight="bold",
                     bbox=dict(boxstyle="round,pad=0.3", fc="#e9e9e9",
                               ec="#9a9a9a"))
    fig.text(0.385, 0.945, "C", fontsize=11, fontweight="bold")

    out = f"manuscript_styled_fig4_{tag}.png"
    fig.savefig(out, dpi=250)
    fig.savefig(f"manuscript_styled_fig4_{tag}.svg")
    fig.savefig(f"manuscript_styled_fig4_{tag}.pdf")
    plt.close(fig)
    print(f"[fig4] Wasserstein={wd:.3f} p={pval:.4f} rows={len(rows)} -> {out} (+svg +pdf)")


# ------------------------------------------------------------------ Fig 5
def layer_base(layer_key):
    """Strip a trailing non-numeric module-name segment so variant keys for
    the SAME probed layer compare equal: "features.40.flex_conv" ->
    "features.40". A key with a numeric (or absent) trailing segment is
    returned unchanged.
    """
    head, sep, tail = layer_key.rpartition(".")
    return head if sep and not tail.isdigit() else layer_key


def vgg_layer_id(layer_key):
    """Map a VGG16 "features.X[.suffix]" layer key to the manuscript's
    "Layer ID" label ("block.position", e.g. features.20 -> "3.3")."""
    base = layer_base(layer_key)
    head, _, tail = base.partition(".")
    if head == "features" and tail.isdigit():
        return VGG_LAYER_ID.get(int(tail), base)
    return base


def fig5(tag, arch="resnet50", flex_bs=None, vanilla_bs=None):
    """Draft-exact Figure 5 (neural predictivity). Layout read off main.pdf
    p.16: A = brain schematic (vector art lifted from the compiled draft --
    data-independent), B = per-region score with the layer assignment made
    separately per network (per-region best scored layer), C = 2x2 per-layer
    grid (V1,V2 / V4,IT) with rotated "Layer ID" ticks, one global ylim
    (0, 1.1*max) on every panel, x10^-1 offset notation, right-column y tick
    labels hidden, frameless legends in B and in the V2 panel.
    """
    def nested_vals(path):
        d = json.loads(Path(path).read_text()); e = next(iter(d.values()))
        out = {}
        for lk, bench in e.items():
            # merge variant keys of the SAME probed layer: resnet nested
            # ".layerX" suffixes, and VGG flex module names ("features.40.
            # flex_conv" scores the layer vanilla keys as "features.40").
            # layer_base only applies to VGG keys -- it would wrongly strip
            # a resnet ".conv3" tail.
            key = lk.split(".layer")[0]
            if key.startswith("features."):
                key = layer_base(key)
            out.setdefault(key, {}).update(bench)
        return out

    van = nested_vals(vanilla_bs or str(RP / "vanilla-7L_brainscore.json"))
    flx = nested_vals(flex_bs or str(RP / "flex-7L_brainscore.json"))

    def clean(v):
        return v if (v is not None and v > 1e-6) else np.nan

    # per-arch probe ordering for panel C
    if arch == "resnet50":
        layers, ticks, xs = LAYERS7, LAYER_TICKS, LAYER_DEPTH
    else:
        def layer_num(k):
            tail = k.rsplit(".", 1)[-1]
            return (0, int(tail)) if tail.isdigit() else (1, k)
        layers = sorted(set(van) | set(flx), key=layer_num)
        ticks, xs = [vgg_layer_id(k) for k in layers], np.arange(len(layers))
    if len(layers) < 2:
        raise ValueError(
            f"Fig5: only {len(layers)} scored layer(s) in the brain-score "
            f"jsons; the draft-exact figure needs a multi-layer sweep "
            f"(stage the per-layer rescore before rendering)")

    scores = {"van": {L: {r: clean(van.get(L, {}).get(BENCH[r])) for r in BENCH}
                      for L in layers},
              "flx": {L: {r: clean(flx.get(L, {}).get(BENCH[r])) for r in BENCH}
                      for L in layers}}
    allvals = [v for m in scores.values() for b in m.values()
               for v in b.values() if not np.isnan(v)]
    if not allvals:
        raise ValueError("Fig5: no usable Brain-Score cell in either json")
    ymax = 1.1 * max(allvals)
    yticks = np.arange(0, ymax + 1e-9, 0.2)

    # B: layer assignment separate per network = per-region best scored layer
    def best(model):
        out = {}
        for r in BENCH:
            vals = {L: scores[model][L][r] for L in layers
                    if not np.isnan(scores[model][L][r])}
            if not vals:
                raise ValueError(f"Fig5 panel B: no usable {r} score for "
                                 f"{model} in any scored layer")
            L = max(vals, key=vals.get)
            out[r] = (vals[L], L)
        return out
    bestv, bestf = best("van"), best("flx")

    def style_axis(ax):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=7.5, length=2.5)
        ax.set_ylim(0, ymax); ax.set_yticks(yticks)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0),
                            useMathText=True)
        ax.yaxis.get_offset_text().set_fontsize(7)

    fig = plt.figure(figsize=(7.4, 3.8))
    # A: the draft's brain schematic (data-independent vector art, rasterised
    # at 600 dpi from main.pdf p.16); grey placeholder if not staged
    axA = fig.add_axes([0.02, 0.42, 0.255, 0.50], anchor="NW")
    axA.axis("off")
    sch = RP / "fig5A_schematic.png"
    if not sch.exists():
        sch = Path("data/iso_analysis_staged/fig5A_schematic.png")
    if sch.exists():
        axA.imshow(np.asarray(Image.open(sch)))
    else:
        print("[fig5] WARNING: fig5A_schematic.png not staged; drawing a "
              "placeholder box for panel A")
        axA.add_patch(plt.Rectangle((0.05, 0.1), 0.9, 0.8, fc="#f0f0f4",
                                    ec="0.6", transform=axA.transAxes))
        axA.text(0.5, 0.5, "biological alignment\n(schematic)", ha="center",
                 va="center", transform=axA.transAxes, fontsize=8, color="0.4")

    # B
    axB = fig.add_axes([0.075, 0.115, 0.205, 0.28])
    regs = list(BENCH)
    x = np.arange(len(regs))
    series(axB, x, [bestv[r][0] for r in regs], [bestf[r][0] for r in regs])
    style_axis(axB)
    axB.set_xticks(x); axB.set_xticklabels(regs, fontsize=7.5)
    axB.set_ylabel("Correlation", fontsize=8.5)
    axB.set_xlabel("Visual Area", fontsize=8.5)
    axB.legend(loc="lower right", frameon=False, fontsize=7.5,
               borderaxespad=0.3)

    # C: 2x2 per-layer grid
    gsC = fig.add_gridspec(2, 2, left=0.375, right=0.985, top=0.855,
                           bottom=0.145, hspace=0.9, wspace=0.10)
    for idx, reg in enumerate(["V1", "V2", "V4", "IT"]):
        ax = fig.add_subplot(gsC[idx // 2, idx % 2])
        series(ax, xs,
               [scores["van"][L][reg] for L in layers],
               [scores["flx"][L][reg] for L in layers])
        style_axis(ax)
        ax.set_title(reg, fontsize=10, pad=10)
        ax.set_xticks(xs)
        if arch == "resnet50":
            ax.set_xticklabels(ticks, fontsize=6.5)
            ax.set_xlim(-0.6, 15.6)
        else:
            ax.set_xticklabels(ticks, rotation=45, ha="right", fontsize=7.5)
        if idx % 2 == 0:
            ax.set_ylabel("Correlation", fontsize=8.5)
        else:
            plt.setp(ax.get_yticklabels(), visible=False)
        if idx >= 2:
            ax.set_xlabel("Layer ID", fontsize=8.5)
        if idx == 1:
            ax.legend(loc="upper right", frameon=False, fontsize=7.5,
                      borderaxespad=0.3)

    for lx, ly, letter in ((0.012, 0.93, "A"), (0.012, 0.415, "B"),
                           (0.315, 0.93, "C")):
        fig.text(lx, ly, letter, fontsize=13, fontweight="bold")

    out = f"manuscript_styled_fig5_{tag}.png"
    fig.savefig(out, dpi=250)
    fig.savefig(f"manuscript_styled_fig5_{tag}.svg")
    fig.savefig(f"manuscript_styled_fig5_{tag}.pdf")
    plt.close(fig)
    for name, b in (("Baseline", bestv), ("Flex", bestf)):
        print(f"[fig5] B {name} best layers: " + "  ".join(
            f"{r}={b[r][0]:.3f}@{b[r][1]}" for r in regs))
    print(f"[fig5] layers={list(layers)} -> {out} (+svg +pdf)")


def fig6(tag):
    """NEW figure (no draft counterpart): how the Flex mechanism routes
    across depth on the pair's flex checkpoint. Data = full-validation
    selection statistics (analyze_flex_selection_stats_hpc.py): for every
    Flex2D site, per-element P(conv path wins the element-wise max against
    the pool path), plus per-image and per-class aggregates.
    Single row of three panels:
      A = per-layer path shares (stacked conv/max bars, +-1 s.d. over images)
      B = routing determinism (always-max / always-conv / input-dependent)
      C = channel commitment strip (one dot per filter: channel-mean P(conv))
    Reading order = aggregate -> category -> identity: A unit-level mean,
    B unit-level determinism split, C per-filter localisation.
    House style follows figs 2-5: orange = conv path, navy = max-pool path
    (the plot_flex_conv_ratio convention), x10^-1 y offsets, rotated
    "Layer ID" ticks, frameless legends, top/right spines off.
    """
    import re

    def staged(suffix):
        p = RP / f"flex_selection_stats.{suffix}"
        if p.exists():
            return p
        m = re.search(r"flexvggE(\d+)", tag)
        if m:
            q = (Path("data/flex_selection_stats")
                 / f"vgg16-flex-e{m.group(1)}.{suffix}")
            if q.exists():
                return q
        raise ValueError(
            f"fig6: no flex_selection_stats .{suffix} for {tag} -- run "
            "sbatch scripts/hpc/flex_selection_stats.sh <run_dir> <flextag> "
            "and stage to data/flex_selection_stats/<flextag>.json/.npz")

    meta = json.loads(staged("json").read_text())
    rows = sorted(meta.get("layers") or [], key=lambda r: r.get("order", 0))
    if len(rows) < 2:
        raise ValueError("fig6: selection-stats json holds <2 flex layers")
    names = [r["name"] for r in rows]
    conv = np.array([r["conv"] for r in rows], float)
    imstd = np.array([r["per_image_std"] for r in rows], float)

    z = np.load(staged("npz"))
    n = len(names)
    lock_max, lock_conv = np.zeros(n), np.zeros(n)
    chans = []
    for i, name in enumerate(names):
        try:
            m = np.asarray(z[f"{name}/p_elem"], float)
        except KeyError as e:
            raise ValueError(f"fig6: npz misses key for layer {name}: {e}")
        p = m.ravel()
        lock_max[i] = float((p < 0.05).mean())
        lock_conv[i] = float((p > 0.95).mean())
        chans.append(m.reshape(m.shape[0], -1).mean(1))
    switching = 1.0 - lock_max - lock_conv

    ticks = [vgg_layer_id(k) for k in names]
    xs = np.arange(n)

    fig = plt.figure(figsize=(7.4, 2.6))
    gs = fig.add_gridspec(1, 3, left=0.075, right=0.975, top=0.84,
                          bottom=0.225, wspace=0.55)

    def style_axis(ax):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=7.5, length=2.5)
        ax.set_xticks(xs)
        ax.set_xticklabels(ticks, rotation=45, ha="right", fontsize=7)
        ax.set_xlim(-0.7, n - 0.3)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0),
                            useMathText=True)
        ax.yaxis.get_offset_text().set_fontsize(7)
        ax.grid(False)

    # A -- path shares across depth
    axA = fig.add_subplot(gs[0, 0])
    axA.bar(xs, conv, width=0.72, color=FLEX, label="Conv path")
    axA.bar(xs, 1.0 - conv, bottom=conv, width=0.72, color=BASE,
            label="Max-pool path")
    axA.errorbar(xs, conv, yerr=imstd, fmt="none", ecolor="white",
                 elinewidth=0.9, capsize=1.5, capthick=0.9)
    axA.axhline(0.5, ls="--", lw=0.8, color="#9a9a9a")
    style_axis(axA)
    axA.set_ylim(0, 1)
    axA.set_yticks(np.arange(0, 1.01, 0.2))
    axA.set_ylabel("Path share", fontsize=8.5)
    axA.set_xlabel("Layer ID", fontsize=8.5)
    # horizontal legend north of the panel (draft frameless style)
    axA.legend(loc="lower left", bbox_to_anchor=(-0.02, 1.02), ncol=2,
               frameon=False, fontsize=7, handlelength=1.2,
               columnspacing=0.9, handletextpad=0.5, borderaxespad=0.0)

    # B -- routing determinism (unit level, decomposes A's mean share)
    axB = fig.add_subplot(gs[0, 1])
    axB.plot(xs, lock_max, "-o", color=BASE, ms=4, lw=1.4,
             label="Always max-pool")
    axB.plot(xs, lock_conv, "--s", color=FLEX, ms=4, lw=1.4,
             label="Always conv")
    axB.plot(xs, switching, ":^", color="#7a7a7a", ms=4, lw=1.4,
             label="Input-dependent")
    style_axis(axB)
    axB.set_ylim(0, 1.2)
    axB.set_yticks(np.arange(0, 1.01, 0.2))
    axB.set_ylabel("Fraction of units", fontsize=8.5)
    axB.set_xlabel("Layer ID", fontsize=8.5)
    axB.legend(loc="upper center", frameon=False, fontsize=6.5,
               borderaxespad=0.0, handlelength=1.8, labelspacing=0.25)

    # C -- channel commitment strip (one dot per filter)
    axC = fig.add_subplot(gs[0, 2])
    rng = np.random.default_rng(0)
    for i, ch in enumerate(chans):
        jx = i + rng.uniform(-0.27, 0.27, ch.size)
        axC.plot(jx, ch, ".", color=BASE, ms=1.6, alpha=0.30)
        axC.plot([i - 0.33, i + 0.33], [float(np.median(ch))] * 2, color=FLEX,
                 lw=1.8, solid_capstyle="butt", zorder=3)
    style_axis(axC)
    axC.set_ylim(-0.02, 1.02)
    axC.set_yticks(np.arange(0, 1.01, 0.2))
    axC.set_ylabel("Channel-mean P(conv path)", fontsize=8.5)
    axC.set_xlabel("Layer ID", fontsize=8.5)
    axC.plot([], [], ".", color=BASE, ms=4, label="Channel")
    axC.plot([], [], color=FLEX, lw=1.8, label="Median")
    axC.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2,
               frameon=False, fontsize=7, handlelength=1.2,
               columnspacing=0.9, handletextpad=0.5, borderaxespad=0.0)

    for lx, ly, letter in ((0.012, 0.93, "A"), (0.355, 0.93, "B"),
                           (0.68, 0.93, "C")):
        fig.text(lx, ly, letter, fontsize=13, fontweight="bold")

    out = f"manuscript_styled_fig6_{tag}.png"
    fig.savefig(out, dpi=250)
    fig.savefig(f"manuscript_styled_fig6_{tag}.svg")
    fig.savefig(f"manuscript_styled_fig6_{tag}.pdf")
    plt.close(fig)
    print(f"[fig6] conv share {conv[0]:.3f}@{ticks[0]} -> "
          f"{conv[-1]:.3f}@{ticks[-1]}; locked units "
          f"{(lock_max + lock_conv)[0]:.2f} -> {(lock_max + lock_conv)[-1]:.2f}")
    print(f"[fig6] n_images={meta.get('n_images')} "
          f"mechanism={meta.get('joint_mechanism')} -> {out} (+svg +pdf)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", default="ISO0.686")
    ap.add_argument("--only", default=None, help="comma list: 1,2,3,4,5,6")
    ap.add_argument("--arch", choices=["resnet50", "vgg16"], default="resnet50",
                     help="architecture driving Fig5's layer probes")
    ap.add_argument("--vanilla-metrics", default=str(RP / "vanilla_metrics.jsonl"),
                     help="Fig2 A/B baseline metrics.jsonl path")
    ap.add_argument("--flex-metrics", default=str(RP / "flex_metrics.jsonl"),
                     help="Fig2 A/B flex metrics.jsonl path")
    ap.add_argument("--vanilla-bs", default=str(RP / "vanilla-7L_brainscore.json"),
                     help="Fig5 baseline brain-score json path")
    ap.add_argument("--flex-bs", default=str(RP / "flex-7L_brainscore.json"),
                     help="Fig5 flex brain-score json path")
    args = ap.parse_args()
    which = args.only.split(",") if args.only else ["1", "2", "3", "4", "5"]
    if "1" in which: fig1(args.tag)
    if "2" in which: fig2(args.tag, args.vanilla_metrics, args.flex_metrics)
    if "3" in which: fig3(args.tag)
    if "4" in which: fig4(args.tag)
    if "5" in which: fig5(args.tag, args.arch, args.flex_bs, args.vanilla_bs)
    if "6" in which: fig6(args.tag)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
