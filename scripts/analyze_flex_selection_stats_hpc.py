#!/usr/bin/env python3
"""Rich conv-vs-max selection statistics for a trained flex network over
ImageNet val — the data behind the "where does flex choose conv?" figure.

Extends scripts/analyze_flex_conv_ratio_hpc.py (same faithful eval-time hook:
re-runs each Flex2D's own conv/pool branches through the project's
channel_expand_view + channel_wise_maxpool; no training-mode flip). Instead of
only a per-layer scalar, it accumulates per layer:

  - elem_sum   (C,H,W) float32 : per-ELEMENT count of images where conv won.
                 Everything else derives from it: per-channel fractions
                 (channel specialization), per-position fractions (spatial
                 preference maps), per-element p(conv) (routing determinism),
                 and the global element-weighted conv fraction.
  - per_image  (N,) float32    : per-image mean conv fraction (stability).
  - class_sum/class_cnt (K,)   : per-class sums of the per-image fraction
                 (class dependence of the routing).

Output: one .npz (arrays, keyed <layer>/elem_sum etc.) + one .json summary.

COMPUTE: loads torch + a model + ImageNet val -> MUST run via sbatch on a GPU
node, never on the login node.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from cli_tool import get_model_for_experiment  # noqa: E402
from src.modules.layers._utils import channel_expand_view  # noqa: E402
from src.modules.joint.channelwise_maxpool import channel_wise_maxpool  # noqa: E402
from src.training.dataset_select import get_dataset_obj  # noqa: E402
from torch.utils.data import DataLoader, Subset  # noqa: E402

N_CLASSES = 1000


def find_flex_layers(model):
    """Flex2D modules (or subclasses) in network-definition (depth) order.

    Matched by MRO class NAME, not exact type: VGG.py builds its flex convs as
    a locally-defined subclass literally named ``Conv2d`` (``class
    Conv2d(Flex2D)``), so an exact ``type(m).__name__ == "Flex2D"`` check finds
    nothing on FlexVGG. Walking the MRO catches Flex2D and every subclass while
    staying robust to the src.modules vs src.flex_neurons duplication.
    """
    return [(name, m) for name, m in model.named_modules()
            if any(c.__name__ == "Flex2D" for c in type(m).__mro__)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--exp-name", required=True)
    ap.add_argument("--num-images", type=int, default=0,
                    help="<=0 means FULL val (default); else stratified subset")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--out-npz", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    cfg_path = run_dir / "configurations.json"
    if not cfg_path.is_file():
        print(f"[error] config not found: {cfg_path}", file=sys.stderr)
        return 2
    config = json.loads(cfg_path.read_text())
    print(f"[info] network={config.get('network')} use_flex={config.get('use_flex')} "
          f"joint={config.get('joint_mechanism')}", flush=True)

    device = args.device if torch.cuda.is_available() else "cpu"
    model = get_model_for_experiment(str(run_dir))
    model.to(device).eval()

    flex_layers = find_flex_layers(model)
    if not flex_layers:
        print("[error] no Flex2D layers found", file=sys.stderr)
        return 3
    order = [name for name, _ in flex_layers]
    print(f"[info] {len(order)} Flex2D layers", flush=True)

    # accumulators (created lazily on the first batch, once shapes are known)
    elem_sum: dict[str, torch.Tensor] = {}
    per_image: dict[str, list[float]] = {n: [] for n in order}
    class_sum = {n: torch.zeros(N_CLASSES, dtype=torch.float64) for n in order}
    class_cnt = torch.zeros(N_CLASSES, dtype=torch.int64)
    batch_labels: list[torch.Tensor] = []   # labels of the CURRENT batch

    def make_hook(layer_name):
        def hook(module, inp, out):  # noqa: ANN001
            x = inp[0]
            with torch.no_grad():
                pool = channel_expand_view(module.flex_pool(x), module.out_channels)
                conv = module.flex_conv(x)
                _pooled, _ratio, cp = channel_wise_maxpool(pool, conv)
                cpf = cp.float()                       # (B,C,H,W), 1 == conv won
                if layer_name not in elem_sum:
                    elem_sum[layer_name] = torch.zeros(
                        cpf.shape[1:], dtype=torch.float32, device=cpf.device)
                elif elem_sum[layer_name].shape != cpf.shape[1:]:
                    raise ValueError(
                        f"{layer_name}: activation shape changed "
                        f"{tuple(elem_sum[layer_name].shape)} -> {tuple(cpf.shape[1:])}; "
                        f"fixed input size expected")
                elem_sum[layer_name] += cpf.sum(dim=0)
                frac = cpf.mean(dim=(1, 2, 3)).cpu()   # (B,)
                per_image[layer_name].extend(frac.tolist())
                lbl = batch_labels[0]
                class_sum[layer_name].index_add_(0, lbl, frac.double())
        return hook

    handles = [m.register_forward_hook(make_hook(n)) for n, m in flex_layers]

    val_ds = get_dataset_obj(config.get("dataset", "imagenet"), "VAL")
    n_total = len(val_ds)
    if 0 < args.num_images < n_total:
        stride = max(1, n_total // args.num_images)
        idx = list(range(0, n_total, stride))[: args.num_images]
        val_ds = Subset(val_ds, idx)
    loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)
    print(f"[info] scoring {len(val_ds)} / {n_total} val images", flush=True)

    seen = 0
    with torch.no_grad():
        for i, batch in enumerate(loader):
            images, labels = batch[0], batch[1]
            lbl = labels.long().clamp_(0, N_CLASSES - 1)
            batch_labels.clear()
            batch_labels.append(lbl)
            model(images.to(device, non_blocking=True))
            class_cnt.index_add_(0, lbl, torch.ones_like(lbl))
            seen += images.size(0)
            if i % 20 == 0:
                print(f"  batch {i:4d} | seen {seen:6d}", flush=True)

    for h in handles:
        h.remove()

    # ---- derive + save ----
    npz: dict[str, np.ndarray] = {"__order__": np.array(order)}
    summary = []
    for k, name in enumerate(order):
        es = elem_sum[name].cpu().numpy()                # (C,H,W) counts
        p = es / max(seen, 1)                            # per-element p(conv)
        pi = np.asarray(per_image[name], dtype=np.float64)
        cs = class_sum[name].numpy()
        cc = class_cnt.numpy()
        cls_mean = np.divide(cs, cc, out=np.full(N_CLASSES, np.nan), where=cc > 0)
        npz[f"{name}/p_elem"] = p.astype(np.float32)
        npz[f"{name}/class_mean"] = cls_mean.astype(np.float32)
        conv = float(p.mean())
        static_share = float(((p < 0.05) | (p > 0.95)).mean())
        summary.append({
            "name": name, "order": k, "conv": conv, "max": 1.0 - conv,
            "per_image_std": float(pi.std()) if pi.size else 0.0,
            "static_share": static_share,
            "shape": list(p.shape),
        })
        print(f"  {name:20s} conv={conv:.3f} static={static_share:.3f}", flush=True)

    out_npz = Path(args.out_npz)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_npz, **npz)
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps({
        "exp_name": args.exp_name, "run_dir": str(run_dir), "n_images": seen,
        "network": config.get("network"),
        "joint_mechanism": config.get("joint_mechanism"),
        "layers": summary,
    }, indent=2))
    print(f"[done] {seen} images -> {out_npz} + {out_json}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
