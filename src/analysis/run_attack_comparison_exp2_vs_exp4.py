#!/Users/donyin/miniconda3/envs/imperial/bin/python

"""
Script to compare attacks between Experiment 2 and Experiment 4.
Modified to run FGSM, Jitter, and APGD instead of SPSA.
"""

import sys
import os
sys.path.append(os.getcwd())
import torch
import torchattacks
import numpy as np
import json
from pathlib import Path
from src.analysis.run_loader import RunLoader
from src.flex_neurons.utils.device import select_device
from src.flex_neurons.utils.normalization import Normalize, IMAGENET_MEAN, IMAGENET_STD, denormalize_batch
from src.analysis.run_attack_comparison import run_attack_comparison
from plotting import compare_attack_results
from torch.utils.data import DataLoader, Subset

def main():
    device = select_device()
    
    exp2_id = "2" 
    exp4_id = "4"
    
    # 1. Define Attacks (No tuning needed for standard attacks usually)
    attacks_to_run = ["FGSM", "Jitter", "APGD"]
    
    # Use default params (or specify if needed)
    attack_params = {}
    
    # 2. Run Attacks 
    # Exp 2
    print(f"\nRunning Attacks for Exp {exp2_id}...")
    run_attack_comparison(
        exp2_id, 
        batch_size=32, # Batch size can be larger for FGSM/Jitter
        attacks=attacks_to_run, 
        resume=False, 
        attack_params_dict=attack_params,
        max_samples=200,
        seed=42
    )
    
    # Exp 4
    print(f"\nRunning Attacks for Exp {exp4_id}...")
    run_attack_comparison(
        exp4_id, 
        batch_size=32, 
        attacks=attacks_to_run,
        resume=False,
        attack_params_dict=attack_params,
        max_samples=200,
        seed=42
    )
    
    # 3. Plot Comparison
    def load_results(eid):
        base_dir = Path(f"__local__/experiment-{eid}/000000/results/attack_comparison")
        if not base_dir.exists():
            return None
        # Find latest timestamp dir
        dirs = sorted([d for d in base_dir.iterdir() if d.is_dir()], key=lambda x: x.name)
        if not dirs:
            return None
        latest = dirs[-1]
        print(f"Loading results from {latest} for Exp {eid}")
        res_file = latest / "attack_comparison_results.json"
        if not res_file.exists():
            return None
        
        # Robust load
        with open(res_file, "r") as f:
            content = f.read()
            content = content.replace(",\n}", "\n}")
            return json.loads(content)

    res2 = load_results(exp2_id)
    res4 = load_results(exp4_id)
    
    if res2 and res4:
        print("Generating Comparison Plot...")
        output_plot = Path(f"comparison_exp{exp2_id}_vs_exp{exp4_id}_fgsm_jitter_apgd.png").resolve()
        compare_attack_results(
            res2, 
            res4, 
            label1=f"Exp {exp2_id}", 
            label2=f"Exp {exp4_id}", 
            output_path=output_plot,
            show_plot=False
        )
        print(f"Done. Plot saved to {output_plot}")
    else:
        print("Error loading results for plotting.")

if __name__ == "__main__":
    main()
