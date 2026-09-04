#!/usr/bin/env bash
# ============================================================================
# Render the manuscript figures (Figs 1-6) for every iso-accuracy pair whose
# staged inputs exist under data/. Pure matplotlib: no GPU, no torch, no
# cluster access needed -- everything renders from the shipped data/ tree.
#
# Inputs per checkpoint tag under data/iso_analysis_staged/:
#   frequency/<tag>.npz      attacks/<tag>.json    perturbation/<tag>.json
#   hessian/<tag>.npy        spectra512/<tag>.npz
#   extras_best/<tag>/rank{1,2}/
#   brainscore7L/<pairkey>_{flex,vanilla}.json
# plus training logs under data/logs/<experiment>/ (metrics.jsonl +
# full_val_epoch*.json), data/iso_analysis_staged/probe/input_29.JPEG,
# data/iso_analysis_staged/fig5A_schematic.png, and
# data/flex_selection_stats/<flextag>.{json,npz} for Fig 6.
#
# Usage:
#   scripts/render_figures.sh                       # figures 1-6, all pairs
#   ONLY=3,4 scripts/render_figures.sh              # subset of figures
#   PYTHON=.venv/bin/python scripts/render_figures.sh
#
# Outputs land in the repository root as
# manuscript_styled_fig<N>_ISO<acc>_flexvggE<fe>_vs_vanillavggE<ve>.{png,svg,pdf}
# ============================================================================
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON="${PYTHON:-python3}"
S="$REPO/data/iso_analysis_staged"
LOGS="$REPO/data/logs"
RP="${MANUSCRIPT_RP:-/tmp/manuscript_repro}"
export MANUSCRIPT_RP="$RP"
ONLY="${ONLY:-1,2,3,4,5,6}"
FLEX_EXP=experiment-flex-vgg16-imagenet-dense
VAN_EXP=experiment-vanilla-vgg16-imagenet-dense

"$PYTHON" -c "import numpy, matplotlib, scipy, PIL" || {
    echo "[err] python deps missing: numpy matplotlib scipy pillow" >&2; exit 2; }

ok=0; fail=0; skip=0
while read -r flextag vantag iso fe ve; do
    tag="ISO${iso}_flexvggE${fe}_vs_vanillavggE${ve}"
    missing=0
    for f in "$S/frequency/${flextag}.npz" "$S/frequency/${vantag}.npz" \
             "$S/attacks/${flextag}.json" "$S/attacks/${vantag}.json" \
             "$S/perturbation/${flextag}.json" "$S/perturbation/${vantag}.json" \
             "$S/hessian/${flextag}.npy" "$S/hessian/${vantag}.npy"; do
        [ -f "$f" ] || missing=1
    done
    if [ "$missing" -eq 1 ]; then
        echo "[skip] $tag (staged inputs incomplete)"; skip=$((skip+1)); continue
    fi
    rm -rf "$RP"
    mkdir -p "$RP/flex/frequency_analysis" "$RP/vanilla/frequency_analysis" \
        "$RP/flex/perturbation_analysis" "$RP/vanilla/perturbation_analysis" \
        "$RP/flex/attack_comparison/staged" "$RP/vanilla/attack_comparison/staged" \
        "$RP/flex/hessian_analysis/staged" "$RP/vanilla/hessian_analysis/staged"
    ln -sf "$S/frequency/${flextag}.npz" "$RP/flex/frequency_analysis/frequency_analysis_data.npz"
    ln -sf "$S/frequency/${vantag}.npz" "$RP/vanilla/frequency_analysis/frequency_analysis_data.npz"
    ln -sf "$S/perturbation/${flextag}.json" "$RP/flex/perturbation_analysis/perturbation_analysis_results.json"
    ln -sf "$S/perturbation/${vantag}.json" "$RP/vanilla/perturbation_analysis/perturbation_analysis_results.json"
    ln -sf "$S/attacks/${flextag}.json" "$RP/flex/attack_comparison/staged/attack_comparison_results.json"
    ln -sf "$S/attacks/${vantag}.json" "$RP/vanilla/attack_comparison/staged/attack_comparison_results.json"
    ln -sf "$S/hessian/${flextag}.npy" "$RP/flex/hessian_analysis/staged/hessian_eigenvalues.npy"
    ln -sf "$S/hessian/${vantag}.npy" "$RP/vanilla/hessian_analysis/staged/hessian_eigenvalues.npy"
    # fig2 C/D/E: 512-train-image block1 spectra (+ probe featmap)
    [ -f "$S/spectra512/${flextag}.npz" ] && ln -sf "$S/spectra512/${flextag}.npz" "$RP/flex/block1_spectra.npz"
    [ -f "$S/spectra512/${vantag}.npz" ] && ln -sf "$S/spectra512/${vantag}.npz" "$RP/vanilla/block1_spectra.npz"
    # fig4 C: loss-surface probe rows, max baseline/flex curvature ratio
    for r in 1 2; do
        if [ -d "$S/extras_best/${flextag}/rank${r}" ] && [ -d "$S/extras_best/${vantag}/rank${r}" ]; then
            ln -sfn "$S/extras_best/${flextag}/rank${r}" "$RP/fig4_row${r}_flex"
            ln -sfn "$S/extras_best/${vantag}/rank${r}" "$RP/fig4_row${r}_vanilla"
        fi
    done
    # fig2 E input tile + fig5 A schematic
    [ -f "$S/probe/input_29.JPEG" ] && ln -sf "$S/probe/input_29.JPEG" "$RP/input_29.JPEG"
    [ -f "$S/fig5A_schematic.png" ] && ln -sf "$S/fig5A_schematic.png" "$RP/fig5A_schematic.png"
    # fig6: flex routing statistics (analyze_flex_selection_stats output)
    for ext in json npz; do
        [ -f "$REPO/data/flex_selection_stats/${flextag}.${ext}" ] && \
            ln -sf "$REPO/data/flex_selection_stats/${flextag}.${ext}" "$RP/flex_selection_stats.${ext}"
    done
    BSARGS=()
    if [ -f "$S/brainscore7L/${tag}_flex.json" ] && [ -f "$S/brainscore7L/${tag}_vanilla.json" ]; then
        BSARGS=(--flex-bs "$S/brainscore7L/${tag}_flex.json" --vanilla-bs "$S/brainscore7L/${tag}_vanilla.json")
    fi
    if (cd "$REPO" && "$PYTHON" scripts/reproduce_manuscript_styled.py \
            --arch vgg16 --tag "$tag" --only "$ONLY" \
            --flex-metrics "$LOGS/$FLEX_EXP/metrics.jsonl" \
            --vanilla-metrics "$LOGS/$VAN_EXP/metrics.jsonl" \
            "${BSARGS[@]}"); then
        echo "[styled OK] $tag (figs $ONLY)"; ok=$((ok+1))
    else
        echo "[styled FAIL] $tag" >&2; fail=$((fail+1))
    fi
done < <("$PYTHON" "$REPO/scripts/iso_pairs_from_selection.py" \
             "$REPO/data/iso_accuracy_selection.frozen.json" --format pairs)
echo "[render] done: $ok rendered, $fail failed, $skip skipped (ONLY=$ONLY)"
[ "$fail" -eq 0 ]
