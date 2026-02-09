import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import sys
import os

# Add project root to path
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

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

def load_eigenvalues(eid):
    dir_path = get_latest_results(eid)
    print(f"[INFO] Loading eigenvalues for Experiment {eid} from {dir_path}")
    return np.load(dir_path / "hessian_eigenvalues.npy")

def permutation_test(dist1, dist2, num_permutations=1000):
    """
    Perform permutation test using Wasserstein distance as the statistic.
    """
    n = len(dist1)
    m = len(dist2)
    pool = np.concatenate([dist1, dist2])
    
    # Observed statistic
    obs_stat = scipy.stats.wasserstein_distance(dist1, dist2)
    
    null_stats = []
    
    print(f"[INFO] Running {num_permutations} permutations...")
    for _ in range(num_permutations):
        # Using a copy of pool is safer if we were modifying it in place, 
        # but shuffle modifies in place. We should probably shuffle a copy or index.
        # np.random.permutation returns a shuffled copy.
        shuffled_pool = np.random.permutation(pool)
        perm_a = shuffled_pool[:n]
        perm_b = shuffled_pool[n:]
        
        stat = scipy.stats.wasserstein_distance(perm_a, perm_b)
        null_stats.append(stat)
        
    null_stats = np.array(null_stats)
    
    # P-value: proportion of null stats >= observed stat
    p_value = np.mean(null_stats >= obs_stat)
    
    return obs_stat, p_value, null_stats

from plotting import set_style

# Apply style
set_style()

def plot_permutation_results(obs_stat, null_stats, p_value, output_path):
    # Dimensions match plot_hessian_comparison in plotting.py
    # Width 80mm, Ratio 4:3
    inch_in_mm = 25.4
    width_mm = 80
    fig_ratio = 4 / 3
    height_mm = width_mm / fig_ratio
    figsize = (width_mm / inch_in_mm, height_mm / inch_in_mm)
    
    plt.figure(figsize=figsize)
    
    # Colors from plotting.py
    # Dark Blue 1 (#14213D), Dark Blue 2 (#003D71), Orange (#FCA311), Light Grey (#E5E5E5)
    c_null = "#14213D"  # Dark Blue for bars
    c_obs = "#FCA311"   # Orange for observed line
    
    # Plot Null Distribution
    # "Increase bin size" -> reduce number of bins (e.g., from 30 to 15)
    plt.hist(null_stats, bins=15, alpha=0.7, color=c_null, edgecolor='white', linewidth=0.5, label='Null Distribution')
    
    # Plot Observed Statistic
    plt.axvline(obs_stat, color=c_obs, linestyle='--', linewidth=2.5, label=f'Observed ($p={p_value:.4f}$)')
    
    # Remove title as requested
    # plt.title("Permutation Test Results (Wasserstein Distance)")
    plt.xlabel("Wasserstein Distance")
    plt.ylabel("Count")
    
    # Legend style matching plotting.py (frameon=False)
    plt.legend(frameon=False)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    
    # Save SVG as well
    svg_path = str(output_path).replace(".png", ".svg")
    plt.savefig(svg_path)
    
    plt.close()
    print(f"[INFO] Plot saved to {output_path} and {svg_path}")

def main():
    parser = argparse.ArgumentParser(description="Statistical Analysis of Hessian Eigenvalues")
    parser.add_argument("--exp1", type=str, required=True, help="Experiment ID 1")
    parser.add_argument("--exp2", type=str, required=True, help="Experiment ID 2")
    parser.add_argument("--permutations", type=int, default=1000, help="Number of permutations")
    parser.add_argument("--output", type=str, default=None, help="Output directory")

    args = parser.parse_args()

    try:
        evals1 = load_eigenvalues(args.exp1)
        evals2 = load_eigenvalues(args.exp2)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    obs_stat, p_value, null_stats = permutation_test(evals1, evals2, args.permutations)
    
    print(f"\n--- Results ---")
    print(f"Observed Wasserstein Distance: {obs_stat:.6f}")
    print(f"P-value: {p_value:.6f}")
    
    if args.output:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        # Save to the latest directory of exp1
        output_dir = get_latest_results(args.exp1)
        
    plot_path = output_dir / f"perm_test_wasserstein_{args.exp1}_vs_{args.exp2}.png"
    plot_permutation_results(obs_stat, null_stats, p_value, plot_path)
    
    # Save text results
    with open(output_dir / f"stats_{args.exp1}_vs_{args.exp2}.txt", "w") as f:
        f.write(f"Experiment {args.exp1} vs Experiment {args.exp2}\n")
        f.write(f"Permutations: {args.permutations}\n")
        f.write(f"Observed Wasserstein Distance: {obs_stat}\n")
        f.write(f"P-value: {p_value}\n")
    
    print(f"[INFO] Results saved to {output_dir}")

if __name__ == "__main__":
    main()
