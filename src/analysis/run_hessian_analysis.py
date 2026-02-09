import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
import sys
import psutil
import os
import matplotlib.pyplot as plt
from pathlib import Path
import datetime
import json

# Add project root to path
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from src.analysis.run_loader import RunLoader
from cli_tool import get_model_for_experiment


def estimate_ram_usage(input_shape, batch_size, model):
    """
    Estimates the RAM usage for the Hessian-Vector Product computation.
    """
    process = psutil.Process(os.getpid())
    current_mem_mb = process.memory_info().rss / 1024 / 1024

    # Estimate Gradient Size
    # Input gradients: B x C x H x W x 4 bytes (float32)
    grad_size_mb = (np.prod(input_shape) * batch_size * 4) / 1024 / 1024

    print(f"[INFO] Current Process RAM: {current_mem_mb:.2f} MB")
    print(f"[INFO] Estimated Input Gradient Size (per HVP): {grad_size_mb:.2f} MB")
    return current_mem_mb, grad_size_mb


def hvp(loss, inputs, v):
    """
    Compute Hessian-Vector Product: H*v = \nabla_x (\nabla_x L \cdot v)
    """
    # 1. First gradient: \nabla_x L
    grads = torch.autograd.grad(loss, inputs, create_graph=True, retain_graph=True)[0]

    # 2. Dot product: \nabla_x L \cdot v
    # v should have same shape as inputs
    dot_prod = torch.sum(grads * v)

    # 3. Second gradient: \nabla_x (\nabla_x L \cdot v)
    hvp_result = torch.autograd.grad(dot_prod, inputs, retain_graph=True)[0]

    return hvp_result


def lanczos_algorithm(model, criterion, inputs, targets, m_steps=50, device="cpu"):
    """
    Run Lanczos algorithm to approximate the eigenvalues of the Hessian w.r.t. inputs.
    """
    inputs = inputs.to(device)
    targets = targets.to(device)
    inputs.requires_grad = True

    # Initial forward pass and loss
    outputs = model(inputs)
    loss = criterion(outputs, targets)

    # Initial random vector v1 (normalized)
    v = torch.randn_like(inputs, device=device)
    v = v / torch.norm(v)

    # Storage for Lanczos vectors (alphas and betas)
    alphas = []
    betas = []

    v_prev = None

    print(f"[INFO] Starting Lanczos Iteration (m={m_steps})...")

    for i in range(m_steps):
        # w_prime = H * v_i
        w_prime = hvp(loss, inputs, v)

        # alpha_i = w_prime^T * v_i
        alpha = torch.sum(w_prime * v).item()
        alphas.append(alpha)

        # w = w_prime - alpha_i * v_i - beta_{i-1} * v_{i-1}
        w = w_prime - alpha * v
        if v_prev is not None:
            w = w - betas[-1] * v_prev

        # beta_i = ||w||
        beta = torch.norm(w).item()

        # Check for breakdown (if beta is effectively 0)
        if beta < 1e-6:
            print(
                f"[WARNING] Lanczos breakdown at step {i+1} (beta approx 0). Stopping early."
            )
            break

        betas.append(beta)

        # Prepare for next step
        v_prev = v.clone()
        v = w / beta

        # Log progress and RAM every 10 steps
        if (i + 1) % 10 == 0:
            ram = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
            print(f"Step {i+1}/{m_steps} | RAM: {ram:.2f} MB")

    return np.array(alphas), np.array(betas)


def analyze_hessian_input(
    exp_id, batch_size=1, num_batches=1, m_steps=50, device="cpu"
):
    # Setup Output Directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = Path(f"__local__/experiment-{exp_id}/000000/results/hessian_analysis")
    output_dir = base_dir / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading Experiment {exp_id}...")
    model = get_model_for_experiment(exp_id)
    model.to(device)
    model.eval()

    # Data Loading
    try:
        import torchvision.transforms as transforms
        from torchvision.datasets import ImageFolder
        from torch.utils.data import DataLoader

        val_dir = "data/imagenet100/val.X"
        if not os.path.exists(val_dir):
            print(f"[WARNING] {val_dir} not found. Using random tensors.")
            use_random = True
        else:
            use_random = False

    except Exception as e:
        print(f"[ERROR] Data setup failed: {e}. Using random tensors.")
        use_random = True

    data_loader = None
    data_iter = None
    if not use_random:
        transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        dataset = ImageFolder(val_dir, transform=transform)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        data_iter = iter(data_loader)

    criterion = nn.CrossEntropyLoss()

    all_alphas = []
    all_betas = []

    print(f"[INFO] Running Analysis on {device}...")

    ram_usage_log = []

    for b in range(num_batches):
        print(f"--- Batch {b+1}/{num_batches} ---")

        if use_random:
            inputs = torch.randn(batch_size, 3, 224, 224)
            targets = torch.randint(0, 100, (batch_size,))
        else:
            try:
                inputs, targets = next(data_iter)
            except StopIteration:
                data_iter = iter(data_loader)
                inputs, targets = next(data_iter)

        current_ram, grad_size = estimate_ram_usage(inputs.shape[1:], batch_size, model)
        ram_usage_log.append(
            {"batch": b, "ram_mb": current_ram, "grad_size_mb": grad_size}
        )

        alphas, betas = lanczos_algorithm(
            model, criterion, inputs, targets, m_steps=m_steps, device=device
        )
        all_alphas.append(alphas)
        all_betas.append(betas)

    # Aggregate and Compute Random Ritz Values
    all_sub_eigenvalues = []

    for i in range(len(all_alphas)):
        alpha = all_alphas[i]
        beta = all_betas[i]
        m = len(alpha)
        T = np.zeros((m, m))
        np.fill_diagonal(T, alpha)
        np.fill_diagonal(T[1:], beta[:-1])
        np.fill_diagonal(T[:, 1:], beta[:-1])

        w, _ = np.linalg.eigh(T)
        all_sub_eigenvalues.extend(w)

    eigenvalues = np.array(all_sub_eigenvalues)

    # Save Results
    np.save(output_dir / "hessian_eigenvalues.npy", eigenvalues)

    # Save Plot
    plot_spectrum(eigenvalues, output_dir / "hessian_density.png")

    # Save Metadata
    metadata = {
        "experiment_id": exp_id,
        "batch_size": batch_size,
        "num_batches": num_batches,
        "m_steps": m_steps,
        "device": device,
        "timestamp": timestamp,
        "num_eigenvalues": len(eigenvalues),
    }

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"[INFO] Analysis Complete. Results saved to {output_dir}")
    return output_dir


def plot_spectrum(eigenvalues, output_path):
    plt.figure(figsize=(10, 6))
    plt.hist(
        eigenvalues, bins=50, density=True, alpha=0.7, color="blue", edgecolor="black"
    )
    plt.title("Hessian Eigenvalue Density (Input Space)")
    plt.xlabel("Eigenvalue")
    plt.ylabel("Density")
    plt.grid(axis="y", alpha=0.5)

    plt.savefig(output_path, dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Run Hessian Analysis (Input Space)")
    parser.add_argument(
        "-e", "--experiment", type=str, required=True, help="Experiment ID"
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--batches", type=int, default=1, help="Number of batches to average."
    )
    parser.add_argument(
        "--m-steps", type=int, default=50, help="Number of Lanczos steps."
    )
    parser.add_argument("--device", type=str, default="cpu")

    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[WARNING] CUDA not available, switching to CPU.")
        device = "cpu"

    analyze_hessian_input(
        exp_id=args.experiment,
        batch_size=args.batch_size,
        num_batches=args.batches,
        m_steps=args.m_steps,
        device=device,
    )


def compare_hessian_spectra(exp1, exp2, output_path=None, show_plot=False):
    from plotting import plot_hessian_comparison

    def get_latest_results(eid):
        base_dir = Path(f"__local__/experiment-{eid}/000000/results/hessian_analysis")
        if not base_dir.exists():
            raise FileNotFoundError(
                f"No hessian analysis results found for Experiment {eid}"
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

    print(f"Comparing Hessian Spectra: {dir1} vs {dir2}")

    # Load Data
    data1 = np.load(dir1 / "hessian_eigenvalues.npy")
    data2 = np.load(dir2 / "hessian_eigenvalues.npy")

    if output_path is None:
        output_path = dir1 / f"comparison_hessian_{exp1}_vs_{exp2}.png"

    plot_hessian_comparison(data1, data2, output_path=output_path, show_plot=show_plot)


if __name__ == "__main__":
    main()
