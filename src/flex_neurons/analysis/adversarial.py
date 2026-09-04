import warnings
from typing import Dict, Iterable, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

SUPPORTED_ATTACKS = ("fgsm", "apgd", "spsa", "jitter")


def _fgsm_manual(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    eps: float,
    clip_min: Optional[float] = None,
    clip_max: Optional[float] = None,
) -> torch.Tensor:
    """Hand-rolled FGSM. Applies L_inf perturbation of size eps.

    Pixel-range clamping is applied only when clip_min/clip_max are provided.
    If your inputs are normalized (e.g. ImageNet standardized tensors), either
    pass clip_min/clip_max in the normalized range or omit them to skip clamping.
    """
    images = images.clone().detach().requires_grad_(True)
    outputs = model(images)
    loss = nn.CrossEntropyLoss()(outputs, labels)
    grad = torch.autograd.grad(loss, images)[0]
    grad_sign = grad.sign()
    adv = images.detach() + eps * grad_sign
    if clip_min is not None and clip_max is not None:
        adv = adv.clamp(clip_min, clip_max)
    return adv


def _pgd_manual(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    eps: float,
    alpha: float,
    steps: int,
    norm: str,
    clip_min: Optional[float] = None,
    clip_max: Optional[float] = None,
) -> torch.Tensor:
    """Hand-rolled PGD (used as APGD fallback) for Linf and L2.

    Pixel-range clamping is applied only when clip_min/clip_max are provided.
    If your inputs are normalized (e.g. ImageNet standardized tensors), either
    pass clip_min/clip_max in the normalized range or omit them to skip clamping.
    """
    orig = images.clone().detach()
    adv = images.clone().detach()

    for _ in range(steps):
        adv = adv.requires_grad_(True)
        outputs = model(adv)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        grad = torch.autograd.grad(loss, adv, retain_graph=False)[0].detach()

        if norm == "Linf":
            adv = adv.detach() + alpha * grad.sign()
            delta = torch.clamp(adv - orig, -eps, eps)
            adv = orig + delta
            if clip_min is not None and clip_max is not None:
                adv = adv.clamp(clip_min, clip_max)
        else:  # L2
            grad_norm = grad.view(grad.size(0), -1).norm(2, dim=1)
            # Avoid div-by-zero
            grad_norm = grad_norm.view(-1, 1, 1, 1).clamp(min=1e-12)
            adv = adv.detach() + alpha * grad / grad_norm
            delta = adv - orig
            delta_norm = delta.view(delta.size(0), -1).norm(2, dim=1).view(-1, 1, 1, 1).clamp(min=1e-12)
            delta = delta * (eps / delta_norm).clamp(max=1.0)
            adv = orig + delta
            if clip_min is not None and clip_max is not None:
                adv = adv.clamp(clip_min, clip_max)

    return adv.detach()


def run_attack(
    model: nn.Module,
    loader: DataLoader,
    attack_name: str,
    eps: float,
    *,
    steps: int = 10,
    alpha: Optional[float] = None,
    targeted: bool = False,
    device: str = "cuda",
    norm: str = "Linf",
    clip_min: Optional[float] = None,
    clip_max: Optional[float] = None,
    nb_sample: int = 128,
    clean_acc: Optional[float] = None,
) -> dict:
    """Run a single attack at one epsilon. Returns:
        {
            'clean_acc': float,
            'adv_acc': float,
            'eps': float,
            'attack': str,
            'norm': str,
            'n_samples': int,
        }

    Inputs are passed to the model as-is. If your loader produces normalized
    tensors (e.g. ImageNet mean/std standardized), pass clip_min and clip_max
    corresponding to the pixel range AFTER normalization (e.g. roughly -2.1 and
    2.6 for standard ImageNet normalization), or omit them to skip pixel-range
    clamping and preserve only the L_inf ball constraint.

    nb_sample (SPSA only) is the number of Monte Carlo samples used per
    finite-difference gradient estimate. It is independent of the loader's
    batch size and defaults to 128; max_batch_size is derived from the
    loader's batch size separately and only caps how many of those samples
    are evaluated per forward pass.

    clean_acc, when provided, is used verbatim as the 'clean_acc' entry of
    the returned dict instead of being recomputed from a clean forward pass
    over the loader. Clean accuracy does not depend on eps, so callers that
    invoke run_attack for several eps values against the same model/loader
    (e.g. sweep_attack) can compute it once and pass it in on subsequent
    calls to avoid redundant full passes over the loader. When None (the
    default), behavior is unchanged: clean accuracy is computed here.
    """
    attack_name = attack_name.lower()
    if attack_name not in SUPPORTED_ATTACKS:
        raise ValueError(
            f"Unknown attack '{attack_name}'. Supported: {SUPPORTED_ATTACKS}"
        )

    if nb_sample < 1:
        raise ValueError(f"nb_sample must be a positive integer, got {nb_sample}")

    if norm not in ("Linf", "L2"):
        raise ValueError(f"Unknown norm '{norm}'. Supported: ('Linf', 'L2')")

    if eps < 0:
        raise ValueError(f"eps must be >= 0, got {eps}")

    if norm == "Linf" and eps > 1.0:
        warnings.warn(
            f"eps={eps} exceeds the typical Linf upper bound of 1.0 for "
            f"[0,1]-space inputs; this eps value is likely nonsensical.",
            UserWarning,
        )
    elif norm == "L2" and eps > 10.0:
        warnings.warn(
            f"eps={eps} exceeds the typical L2 upper bound of 10.0 for "
            f"[0,1]-space inputs; this eps value is likely nonsensical.",
            UserWarning,
        )

    # Resolve alpha defaults
    if alpha is None:
        if attack_name == "fgsm":
            alpha = eps
        elif attack_name == "apgd":
            alpha = eps / 4.0
        elif attack_name == "spsa":
            alpha = eps / 100.0
        else:  # jitter
            alpha = eps / steps if steps > 0 else eps

    # Try to import torchattacks; required for apgd/spsa/jitter
    torchattacks = None
    try:
        import torchattacks as _ta
        torchattacks = _ta
    except ImportError:
        if attack_name != "fgsm":
            raise ImportError(
                f"torchattacks is required for '{attack_name}'. "
                "install torchattacks for full attack support: pip install torchattacks"
            )

    was_training = model.training
    model.eval()
    try:
        dev = torch.device(device)
        model.to(dev)

        # Build the torchattacks attack object once (outside the loop).
        # torchattacks attacks are stateless per call, so reuse is fine.
        if torchattacks is not None and eps > 0:
            if attack_name == "fgsm":
                attack_obj = torchattacks.FGSM(model, eps=eps)
            elif attack_name == "apgd":
                attack_obj = torchattacks.APGD(model, eps=eps, steps=steps, norm=norm)
            elif attack_name == "spsa":
                attack_obj = torchattacks.SPSA(
                    model,
                    eps=eps,
                    delta=0.001,
                    lr=alpha,
                    nb_iter=steps,
                    nb_sample=nb_sample,
                    max_batch_size=max(1, loader.batch_size or 2),
                )
            elif attack_name == "jitter":
                attack_obj = torchattacks.Jitter(
                    model,
                    eps=eps,
                    alpha=alpha,
                    steps=steps,
                    scale=10,
                )
        else:
            attack_obj = None  # eps == 0 or no torchattacks + fgsm

        compute_clean = clean_acc is None

        clean_correct = 0
        adv_correct = 0
        n_samples = 0

        for images, labels in loader:
            images = images.to(dev)
            labels = labels.to(dev)

            # Clean forward pass: needed to compute clean accuracy (unless the
            # caller already supplied it) and, when eps == 0, to reuse as the
            # (unperturbed) adversarial output.
            out = None
            if compute_clean or eps == 0:
                with torch.no_grad():
                    out = model(images)
                if compute_clean:
                    clean_correct += out.argmax(1).eq(labels).sum().item()

            # Adversarial accuracy
            if eps == 0:
                # Zero perturbation: adversarial == clean
                adv_correct += out.argmax(1).eq(labels).sum().item()
            elif attack_obj is not None:
                adv_images = attack_obj(images, labels)
                with torch.no_grad():
                    adv_out = model(adv_images)
                adv_correct += adv_out.argmax(1).eq(labels).sum().item()
            else:
                # Manual FGSM fallback (torchattacks not available)
                adv_images = _fgsm_manual(model, images, labels, eps,
                                          clip_min=clip_min, clip_max=clip_max)
                with torch.no_grad():
                    adv_out = model(adv_images)
                adv_correct += adv_out.argmax(1).eq(labels).sum().item()

            n_samples += labels.size(0)

    finally:
        if was_training:
            model.train()

    if compute_clean:
        clean_acc = clean_correct / n_samples if n_samples > 0 else 0.0
    adv_acc = adv_correct / n_samples if n_samples > 0 else 0.0

    return {
        "clean_acc": clean_acc,
        "adv_acc": adv_acc,
        "eps": eps,
        "attack": attack_name,
        "norm": norm,
        "n_samples": n_samples,
    }


def sweep_attack(
    model: nn.Module,
    loader: DataLoader,
    attack_name: str,
    eps_list: Iterable[float],
    **kwargs,
) -> Dict[float, dict]:
    """Run attack at each eps. Returns dict[eps]=run_attack output.

    Clean accuracy does not depend on eps, so it is computed once (during the
    first eps's run_attack call) and reused for every subsequent eps via
    run_attack's clean_acc parameter, instead of being recomputed from
    scratch for each of the K eps values. Every entry in the returned dict
    still has a 'clean_acc' key, identical across all eps.
    """
    results: Dict[float, dict] = {}
    clean_acc: Optional[float] = None
    for eps in eps_list:
        result = run_attack(model, loader, attack_name, eps, clean_acc=clean_acc, **kwargs)
        clean_acc = result["clean_acc"]
        results[eps] = result
    return results
