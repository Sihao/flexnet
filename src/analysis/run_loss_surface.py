import torch
import torch.nn as nn
import numpy as np
import argparse
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


def power_iteration(
    model, criterion, inputs, targets, num_bonds=0, steps=20, device="cpu"
):
    """
    Find the dominant eigenvector of the Hessian using Power Iteration.
    If num_bonds > 0 (deflation), we project out existing eigenvectors.
    """
    inputs = inputs.to(device)
    targets = targets.to(device)
    inputs.requires_grad = True

    outputs = model(inputs)
    loss = criterion(outputs, targets)

    # Random initialization
    v = torch.randn_like(inputs, device=device)
    v = v / torch.norm(v)

    eigenvalue = 0.0

    for i in range(steps):
        # Apply H * v
        w = hvp(loss, inputs, v)

        # Deflation if we want to find 2nd, 3rd eigenvectors
        # Orthogonalize w against known eigenvectors (not implemented here, passed as arg?)
        # For simple deflation, we assume we subtract lambda_i * v_i * (v_i^T v) from H
        # But H is operator.
        # Easier: explicitly project v to be orthogonal to previous vectors at each step.

        # But wait, Power Iteration naturally finds dominant. To find 2nd, we need to remove component of 1st.
        # Handled in the calling function via projection.
        # Actually better to handle it here if we pass the known vectors.

        eigenvalue = torch.dot(v.flatten(), w.flatten()).item()

        v = w / torch.norm(w)

    return eigenvalue, v


def compute_top_eigenvectors(
    model, criterion, inputs, targets, k=2, steps=50, device="cpu"
):
    eigenvalues = []
    eigenvectors = []

    inputs = inputs.to(device)
    targets = targets.to(device)
    inputs.requires_grad = True

    outputs = model(inputs)
    loss = criterion(outputs, targets)

    for i in range(k):
        print(f"Computing Eigenvector {i+1}...")

        # Init v
        v = torch.randn_like(inputs, device=device)
        v = v / torch.norm(v)

        for step in range(steps):
            w = hvp(loss, inputs, v)

            # Defalte: Project out components of previous eigenvectors from w
            # w' = w - sum( (w . vj) * vj )
            for existing_v in eigenvectors:
                proj = torch.dot(w.flatten(), existing_v.flatten())
                w = w - proj * existing_v

            eigenval = torch.dot(v.flatten(), w.flatten()).item()
            v_norm = torch.norm(w)

            if v_norm > 1e-6:
                v = w / v_norm
            else:
                pass  # w is zero vector?

        eigenvalues.append(eigenval)
        eigenvectors.append(v.detach())  # Detach to stop graph growth
        print(f"  Eigenvalue {i+1}: {eigenval:.4f}")

    return eigenvalues, eigenvectors


def evaluate_loss_surface(
    model,
    criterion,
    inputs,
    targets,
    v1,
    v2,
    grid_points=21,
    range_scale=1.0,
    device="cpu",
):
    alphas = np.linspace(-range_scale, range_scale, grid_points)
    betas = np.linspace(-range_scale, range_scale, grid_points)

    loss_surface = np.zeros((grid_points, grid_points))
    mesh_alpha, mesh_beta = np.meshgrid(alphas, betas)

    inputs = inputs.to(device)
    targets = targets.to(device)
    v1 = v1.to(device)
    v2 = v2.to(device)

    print(f"[INFO] Computing Loss Surface ({grid_points}x{grid_points})...")

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

    return mesh_alpha, mesh_beta, loss_surface


def run_loss_surface_analysis(exp_id, grid_points=21, range_scale=1.0, device="cpu", show_plot=False):
    # Setup Output Directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = Path(f"__local__/experiment-{exp_id}/000000/results/loss_surface")
    output_dir = base_dir / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

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

    v1 = eigenvectors[0]
    v2 = eigenvectors[1]

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
