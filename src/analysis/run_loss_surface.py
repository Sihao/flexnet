import torch
import torch.nn as nn
import numpy as np
import argparse
import math
import sys
import os
import matplotlib.pyplot as plt
from pathlib import Path
import datetime
import json
from mpl_toolkits.mplot3d import Axes3D

# Add project root to path
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from src.analysis.run_loader import RunLoader
from cli_tool import get_model_for_experiment


# Absolute-value threshold below which BOTH top-2 eigenvalues are treated as
# a genuinely degenerate (numerically flat) loss surface, even after the
# float64 precision fix in compute_top_eigenvectors / evaluate_loss_surface
# below. Comfortably above float64 rounding noise, far below any real
# curvature magnitude seen on a trained checkpoint (chainlink #401, ported
# from the #396 fix in scripts/analyze_manuscript_extras_hpc.py).
DEGENERATE_EIGENVALUE_ABS_TOL = 1e-8


class DegenerateLossSurfaceError(RuntimeError):
    """Raised by run_loss_surface_analysis when the top-2 eigenvalues are
    still ~0 even in float64 -- a genuinely degenerate probe/checkpoint,
    not the float32 softmax-saturation precision artifact the float64 cast
    fixes. cli_tool.py's run_loss_surface command catches this through its
    generic `except Exception`, so it surfaces as a reported error instead
    of silently saving a flat surface."""


def get_data_sample(val_dir="data/imagenet100/val.X", batch_size=1):
    import torchvision.transforms as transforms
    from torchvision.datasets import ImageFolder
    from torch.utils.data import DataLoader

    if not os.path.exists(val_dir):
        print(f"[WARNING] {val_dir} not found. Using random tensors.")
        inputs = torch.randn(batch_size, 3, 224, 224)
        targets = torch.randint(0, 100, (batch_size,))
        return inputs, targets

    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    dataset = ImageFolder(val_dir, transform=transform)
    # Shuffle true to get random imagew
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    inputs, targets = next(iter(data_loader))
    return inputs, targets


def hvp(loss, inputs, v):
    """
    Compute Hessian-Vector Product: H*v = \nabla_x (\nabla_x L \cdot v)
    """
    # 1. First gradient: \nabla_x L
    # Note: create_graph=True is needed for the second derivative
    grads = torch.autograd.grad(loss, inputs, create_graph=True, retain_graph=True)[0]

    # 2. Dot product: \nabla_x L \cdot v
    dot_prod = torch.sum(grads * v)

    # 3. Second gradient: \nabla_x (\nabla_x L \cdot v)
    # We don't need graph for the output unless we differentiate again (we don't)
    hvp_result = torch.autograd.grad(dot_prod, inputs, retain_graph=True)[0]

    return hvp_result


def compute_top_eigenvectors(
    model, criterion, inputs, targets, k=2, steps=50, device="cpu"
):
    # --- float64 precision fix (chainlink #401) ---
    # A piecewise-linear network (e.g. VGG16 in eval) has input-Hessian
    # curvature ONLY from the softmax cross-entropy term. Once softmax
    # saturates on a confident-correct probe, that term underflows to an
    # EXACT 0.0 in float32 (~88 nat floor), so the double-backprop Hvp
    # below returns an exact-zero Hessian and the top-2 eigenvalues come
    # back 0.00/0.00. float64 pushes the underflow floor to ~709 nats.
    #
    # Run the whole eigen-solve in float64, then restore the model and
    # the returned eigenvectors to whatever dtype they had on entry, so
    # a caller that already passed float64 (e.g.
    # scripts/analyze_manuscript_extras_hpc.py, which double()s the
    # model itself before calling in) sees a harmless no-op.
    input_dtype = inputs.dtype
    model_params = list(model.parameters())
    model_dtype = model_params[0].dtype if model_params else None

    eigenvalues = []
    eigenvectors = []

    inputs = inputs.to(device).double()
    targets = targets.to(device)
    inputs.requires_grad = True

    try:
        model.double()

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        tol = 1e-6

        def deflate(vec, existing_vectors):
            # Project out the component of `vec` along each vector already
            # found, so later eigenvectors stay orthogonal to earlier ones.
            for existing_v in existing_vectors:
                proj = torch.dot(vec.flatten(), existing_v.flatten())
                vec = vec - proj * existing_v
            return vec

        for i in range(k):
            print(f"Computing Eigenvector {i+1}...")

            # Init v, then deflate it against previously found eigenvectors
            # *before* the first Hvp. Without this, power iteration for the
            # 2nd+ eigenvector can collapse straight back onto the 1st.
            v = torch.randn_like(inputs, device=device)
            v = v / torch.norm(v)
            v = deflate(v, eigenvectors)
            v_norm = torch.norm(v)
            if v_norm <= tol:
                # The random draw landed (almost) entirely inside the
                # already-found subspace. Redraw once and deflate again.
                v = torch.randn_like(inputs, device=device)
                v = v / torch.norm(v)
                v = deflate(v, eigenvectors)
                v_norm = torch.norm(v)
                if v_norm <= tol:
                    # Even the redraw collapsed into the already-found
                    # subspace (e.g. k exceeds the input's true
                    # dimensionality, so no independent direction is left
                    # to initialize from). Report the same kind of
                    # degenerate (~0) eigenpair the in-loop path reports
                    # below, WITHOUT dividing by the near-zero v_norm.
                    print(
                        f"[WARNING] Eigenvector {i+1}: random init collapsed "
                        f"into the existing subspace even after a redraw "
                        f"(||v|| <= {tol}); reporting a degenerate (~0) "
                        f"eigenvalue for this direction."
                    )
                    eigenvalues.append(0.0)
                    eigenvectors.append(v.detach())
                    print(f"  Eigenvalue {i+1}: 0.0000 (degenerate)")
                    continue
            v = v / v_norm

            degenerate = False

            for step in range(steps):
                w = hvp(loss, inputs, v)

                # Deflate: Project out components of previous eigenvectors from w
                # w' = w - sum( (w . vj) * vj )
                w = deflate(w, eigenvectors)

                v_norm = torch.norm(w)

                if v_norm <= tol:
                    # Curvature vanished along this direction: H*v deflated to
                    # ~0, so there is nothing left to iterate on. Stop instead
                    # of advancing v to a near-zero/un-normalisable direction.
                    degenerate = True
                    print(
                        f"[WARNING] Eigenvector {i+1}: curvature vanished at "
                        f"step {step+1} (||H v|| <= {tol}); reporting a "
                        f"degenerate (~0) eigenvalue for this direction."
                    )
                    break

                v = w / v_norm

            # Recompute the eigenvalue for the FINAL v so the reported
            # (eigenvalue, eigenvector) pair is matched. (The old code paired
            # the Rayleigh quotient of the pre-update v with the post-update
            # v -- a mismatched pair.)
            w_final = hvp(loss, inputs, v)
            w_final = deflate(w_final, eigenvectors)
            eigenval = torch.dot(v.flatten(), w_final.flatten()).item()

            eigenvalues.append(eigenval)
            eigenvectors.append(v.detach())  # Detach to stop graph growth
            status = " (degenerate)" if degenerate else ""
            print(f"  Eigenvalue {i+1}: {eigenval:.4f}{status}")
    finally:
        if model_dtype is not None:
            model.to(model_dtype)

    # Restore the eigenvectors to the dtype `inputs` had on entry, so a
    # caller that passed float32 sees float32 back out (matching the
    # pre-#401 return type), and a caller that already passed float64
    # (the manuscript-extras script) sees float64 back out unchanged.
    eigenvectors = [v.to(input_dtype) for v in eigenvectors]

    return eigenvalues, eigenvectors


def evaluate_loss_surface(
    model,
    criterion,
    inputs,
    targets,
    v1,
    v2,
    grid_points=21,
    range_scale=10.0,
    device="cpu",
):
    alphas = np.linspace(-range_scale, range_scale, grid_points)
    betas = np.linspace(-range_scale, range_scale, grid_points)

    loss_surface = np.zeros((grid_points, grid_points))
    # indexing="ij" makes mesh_alpha[i, j] == alphas[i] and
    # mesh_beta[i, j] == betas[j], matching the fill loop below which
    # writes loss_surface[i, j] for the i-th alpha and j-th beta.
    mesh_alpha, mesh_beta = np.meshgrid(alphas, betas, indexing="ij")

    # --- float64 precision fix (chainlink #401); see
    # compute_top_eigenvectors above for the full rationale. Restore the
    # model to whatever dtype it had on entry; mesh_alpha/mesh_beta/
    # loss_surface are plain numpy arrays and were always float64 in the
    # original code, so no cast-back is needed for them.
    model_params = list(model.parameters())
    model_dtype = model_params[0].dtype if model_params else None

    inputs = inputs.to(device).double()
    targets = targets.to(device)
    v1 = v1.to(device).double()
    v2 = v2.to(device).double()

    print(f"[INFO] Computing Loss Surface ({grid_points}x{grid_points})...")

    try:
        model.double()

        # Pre-calculate base loss
        with torch.no_grad():
            base_out = model(inputs)
            base_loss = criterion(base_out, targets).item()
            print(f"Base Loss: {base_loss:.4f}")

        with torch.no_grad():
            for i, alpha in enumerate(alphas):
                for j, beta in enumerate(betas):
                    perturbation = alpha * v1 + beta * v2
                    perturbed_input = inputs + perturbation

                    output = model(perturbed_input)
                    loss = criterion(output, targets)
                    loss_surface[i, j] = loss.item()
    finally:
        if model_dtype is not None:
            model.to(model_dtype)

    return mesh_alpha, mesh_beta, loss_surface


def run_loss_surface_analysis(exp_id, grid_points=21, range_scale=10.0, device="cpu", show_plot=False):
    # Setup Output Directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = Path(f"__local__/experiment-{exp_id}/000000/results/loss_surface")
    output_dir = base_dir / timestamp

    print(f"[INFO] Loading Experiment {exp_id}...")
    model = get_model_for_experiment(exp_id)
    model.to(device)
    model.eval()

    criterion = nn.CrossEntropyLoss()

    # 1. Get One Image Sample (Randomized from val set)
    # Using existing utility
    inputs, targets = get_data_sample(batch_size=1)
    inputs = inputs.to(device)
    targets = targets.to(device)

    print(f"[INFO] Computing Top 2 Eigenvectors...")
    eigenvalues, eigenvectors = compute_top_eigenvectors(
        model, criterion, inputs, targets, k=2, steps=50, device=device
    )

    # Guard (chainlink #401): the float64 cast inside compute_top_eigenvectors
    # already fixes the float32 softmax-saturation underflow that used to
    # report a spurious 0.00/0.00 top-2 pair. If both are STILL ~0 here, the
    # probe/checkpoint is genuinely flat -- refuse to save a degenerate
    # surface instead of writing one silently. A NaN/Inf top-2 eigenvalue is
    # also degenerate: abs(nan) < tol is False, so it would otherwise slip
    # past a flatness-only check and get written silently as a NaN surface.
    top2 = eigenvalues[:2]
    non_finite = [e for e in top2 if not math.isfinite(e)]
    if non_finite:
        raise DegenerateLossSurfaceError(
            f"Experiment {exp_id}: top-2 eigenvalues {top2} contain a "
            f"non-finite value {non_finite} -- refusing to save a "
            "NaN/Inf-contaminated loss surface."
        )
    if all(abs(e) < DEGENERATE_EIGENVALUE_ABS_TOL for e in top2):
        raise DegenerateLossSurfaceError(
            f"Experiment {exp_id}: top-2 eigenvalues {top2} are "
            f"both below {DEGENERATE_EIGENVALUE_ABS_TOL:.1e} even in "
            "float64 -- this is a genuinely degenerate probe/checkpoint, "
            "not a precision artifact. Refusing to save a flat loss surface."
        )

    v1 = eigenvectors[0]
    v2 = eigenvectors[1]

    # Output directory is created only after the degenerate guard passes
    # (chainlink #401 fix 2), so a degenerate/non-finite run leaves no
    # empty timestamped directory behind.
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save vectors/values
    np.save(output_dir / "eigenvalues.npy", np.array(eigenvalues))
    # Saving vectors might be heavy, skipping for now unless needed

    # 2. Evaluate Surface
    mesh_x, mesh_y, surface_z = evaluate_loss_surface(
        model,
        criterion,
        inputs,
        targets,
        v1,
        v2,
        grid_points=grid_points,
        range_scale=range_scale,
        device=device,
    )

    # Save Surface Data
    np.save(output_dir / "surface_x.npy", mesh_x)
    np.save(output_dir / "surface_y.npy", mesh_y)
    np.save(output_dir / "surface_z.npy", surface_z)

    # Plot using shared plotting logic
    from plotting import plot_loss_surface

    plot_loss_surface(
        mesh_x,
        mesh_y,
        surface_z,
        eigenvalues=eigenvalues,
        exp_id=exp_id,
        output_dir=output_dir,
        show_plot=show_plot,
    )

    print(f"[INFO] Analysis Complete. Saved to {output_dir}")
    return output_dir


def compare_loss_surfaces(exp1, exp2, output_path=None, show_plot=False):
    from plotting import plot_loss_surface_comparison

    def get_latest_results(eid):
        base_dir = Path(f"__local__/experiment-{eid}/000000/results/loss_surface")
        if not base_dir.exists():
            raise FileNotFoundError(
                f"No loss surface results found for Experiment {eid}"
            )
        timestamps = sorted(
            [d for d in base_dir.iterdir() if d.is_dir()],
            key=lambda x: x.name,
            reverse=True,
        )
        if not timestamps:
            raise FileNotFoundError(f"No results found in {base_dir}")
        return timestamps[0]

    dir1 = get_latest_results(exp1)
    dir2 = get_latest_results(exp2)

    print(f"Comparing {dir1} vs {dir2}")

    # Prepare data dicts
    data1 = {
        "x": np.load(dir1 / "surface_x.npy"),
        "y": np.load(dir1 / "surface_y.npy"),
        "z": np.load(dir1 / "surface_z.npy"),
        "vals": np.load(dir1 / "eigenvalues.npy"),
        "exp_id": exp1,
    }

    data2 = {
        "x": np.load(dir2 / "surface_x.npy"),
        "y": np.load(dir2 / "surface_y.npy"),
        "z": np.load(dir2 / "surface_z.npy"),
        "vals": np.load(dir2 / "eigenvalues.npy"),
        "exp_id": exp2,
    }

    if output_path is None:
        # Default to saving in Exp 1's results directory
        output_path = dir1 / f"comparison_{exp1}_vs_{exp2}.png"

    plot_loss_surface_comparison(
        data1, data2, output_path=output_path, show_plot=show_plot
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="action")

    # Run Parser
    p_run = subparsers.add_parser("run")

    p_run.add_argument("-e", "--experiment", type=str, required=True)
    p_run.add_argument("--grid-points", type=int, default=51)
    p_run.add_argument("--range", type=float, default=10.0)
    p_run.add_argument("--device", type=str, default="cpu")

    # Compare Parser
    p_comp = subparsers.add_parser("compare")
    p_comp.add_argument("--exp1", required=True)
    p_comp.add_argument("--exp2", required=True)
    p_comp.add_argument("--output", default=None)

    args = parser.parse_args()

    if args.action == "run":
        device = args.device
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        run_loss_surface_analysis(
            args.experiment,
            grid_points=args.grid_points,
            range_scale=args.range,
            device=device,
        )

    elif args.action == "compare":
        compare_loss_surfaces(args.exp1, args.exp2, args.output)

    # Default behavior for backward compatibility if just args provided (legacy call from CLI might need update)
    if args.action is None and hasattr(args, "experiment"):
        # Fallback if no subparser selected but arguments match run
        device = args.device
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        run_loss_surface_analysis(
            args.experiment,
            grid_points=args.grid_points,
            range_scale=args.range,
            device=device,
        )
