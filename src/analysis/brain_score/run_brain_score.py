
import sys
import os
import json
import traceback
from pathlib import Path
import natsort
import torch
import torch.nn as nn

# Set data directories via env var, default to relative 'data' dir if not set
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
DATA_ROOT = Path(os.environ.get("FLEX_DATA_ROOT", PROJECT_ROOT / "data" / "brain_score_data"))

os.environ['BRAINIO_HOME'] = str(DATA_ROOT / "brainio")
os.environ['BRAINSCORE_HOME'] = str(DATA_ROOT / "brain_score")
# os.environ['RESULTCACHING_HOME'] = str(DATA_ROOT / "result_caching") 

print(f"Brain-Score Data Paths set to: {DATA_ROOT}")

print(f"Brain-Score Data Paths set to: {DATA_ROOT}")
print(f"BRAINIO_HOME: {os.environ.get('BRAINIO_HOME')}")
print(f"BRAINSCORE_HOME: {os.environ.get('BRAINSCORE_HOME')}")
print(f"RESULTCACHING_HOME: {os.environ.get('RESULTCACHING_HOME')}")

from brainscore_vision import score
from src.analysis.brain_score.wrapper import get_brain_model
from src.modules.models.VGG import VGG
from src.modules.layers.flex import Flex2D

def clear_model_hooks(model):
    """
    Removes all forward and backward hooks from the model and its submodules.
    This is critical when reusing a model instance across Brain-Score wrappers
    to prevent hook accumulation and memory leaks.
    """
    for module in model.modules():
        if hasattr(module, "_forward_hooks"):
            module._forward_hooks.clear()
        if hasattr(module, "_forward_pre_hooks"):
            module._forward_pre_hooks.clear()
        if hasattr(module, "_backward_hooks"):
            module._backward_hooks.clear()


def get_latest_checkpoint(run_dir):
    ckpt_dir = run_dir / "checkpoints"
    ckpts = list(ckpt_dir.glob("*.pth"))
    if not ckpts:
        return None
    return natsort.natsorted(ckpts)[0]

def load_results(results_file):
    """
    Loads existing results from JSON file.
    """
    if results_file.exists():
        with open(results_file, "r") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return {}
    return {}

def save_results(results_file, data):
    """
    Saves the results dictionary to JSON file.
    """
    # Load fresh data to avoid overwriting parallel processes
    current_data = load_results(results_file)
    
    # Merge new data into current_data
    for exp, exp_data in data.items():
        if exp not in current_data:
            current_data[exp] = {}
        for layer, layer_data in exp_data.items():
            current_data[exp][layer] = layer_data

    with open(results_file, "w") as f:
        json.dump(current_data, f, indent=4)
    print(f"Saved results to {results_file}")


def process_layer(model, layer, exp_name, config, results_data, output_file, benchmarks, early_layers):
    print(f"\n  Scoring Layer: {layer}")
    # Ensure we start with a clean model (no hooks from previous iterations)
    clear_model_hooks(model)
    
    # Map ALL regions to this single layer to support all benchmarks
    region_map = {
        "V1": layer,
        "V2": layer,
        "V4": layer,
        "IT": layer
    }
    
    identifier = f"flexible_neurons_{exp_name}_{layer}"
    
    # We reuse the loaded model object!
    try:
        brain_model = get_brain_model(
            model_name=identifier,
            region_layer_map=region_map,
            model=model,
            config=config
        )
    except Exception as e:
        print(f"Failed to create brain model for layer {layer}: {e}")
        return


    for bench_id in benchmarks:
        # OPTIMIZATION: Skip late-visual-area benchmarks on early layers
        if layer in early_layers:
            if "MajajHong" in bench_id or "V4" in bench_id or "IT" in bench_id:
                    # Double check to ensure we don't accidentally skip V1/V2 if they had V4 in name (unlikely)
                    # Strict check:
                    if "V4" in bench_id or "IT" in bench_id:
                        print(f"    Skipping {bench_id} for early layer {layer}")
                        continue

        print(f"    Running benchmark {bench_id}...")
        
        # Ensure dictionary structure exists
        if exp_name not in results_data:
            results_data[exp_name] = {}
        if layer not in results_data[exp_name]:
            results_data[exp_name][layer] = {}

        try:
            
            if "FreemanZiemba" in bench_id:
                from brainscore_vision.benchmarks.freemanziemba2013.benchmarks.public_benchmarks import FreemanZiembaV1PublicBenchmark, FreemanZiembaV2PublicBenchmark
                if "V1" in bench_id:
                    benchmark = FreemanZiembaV1PublicBenchmark()
                elif "V2" in bench_id:
                    benchmark = FreemanZiembaV2PublicBenchmark()
                else:
                    raise ValueError(f"Unknown FreemanZiemba benchmark {bench_id}")

            elif "MajajHong" in bench_id:
                from brainscore_vision.benchmarks.majajhong2015.benchmark import MajajHongV4PublicBenchmark, MajajHongITPublicBenchmark
                if "V4" in bench_id:
                    benchmark = MajajHongV4PublicBenchmark()
                elif "IT" in bench_id:
                    benchmark = MajajHongITPublicBenchmark()
                else:
                    raise ValueError(f"Unknown MajajHong benchmark {bench_id}")
            else:
                    raise ValueError(f"Unknown benchmark family for {bench_id}")

            s = benchmark(brain_model)
            # s is an xarray/DataAssembly
            
            # 1. Center Score
            if 'aggregation' in s.coords:
                center_val = s.sel(aggregation='center').item()
            else:
                center_val = s.values.item()
            center_val = float(center_val)

            # 2. Error (Standard Error/Deviation)
            error_val = None
            if 'aggregation' in s.coords and 'error' in s.coords['aggregation'].values:
                try:
                    error_val = float(s.sel(aggregation='error').item())
                except:
                    pass

            # 3. Raw Folds (Cross-Validation Splits)
            folds = []
            # Brain-Score typically stores raw un-aggregated scores in attrs['raw']
            if 'raw' in s.attrs:
                raw_obj = s.attrs['raw']
                # raw_obj is typically an xarray with a 'split' dimension
                if hasattr(raw_obj, 'values'):
                        # Convert to list. Handle generic numpy types.
                        try:
                            val = raw_obj.values.tolist()
                            if isinstance(val, list):
                                folds = val
                            else:
                                folds = [val]
                        except:
                            pass
            
            print(f"    Score: {center_val} (Folds: {len(folds)})")

            # Store complex object
            # Store simple score
            results_data[exp_name][layer][bench_id] = center_val

        except Exception as e:
            print(f"    Error running {bench_id} on {layer}: {e}")
            results_data[exp_name][layer][bench_id] = 0.0

        # Save immediately to show progress
        save_results(output_file, results_data)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, choices=["Exp2", "Exp4"], help="Specific experiment to run (Exp2 or Exp4)")
    parser.add_argument("--layer", type=str, help="Specific layer to run (e.g., features.30)")
    args = parser.parse_args()

    experiments = {
        "Exp2": "2",
        "Exp4": "4"
    }
    
    
    # Filter based on argument
    if args.experiment:
        experiments = {args.experiment: experiments[args.experiment]}
        print(f"Running ONLY for {args.experiment}")
        output_filename = f"results/brain_score_{args.experiment}.json"
    else:
        print("Running for ALL experiments")
        output_filename = "results/brain_score_results.json"

    if args.layer:
        print(f"Running ONLY for layer: {args.layer}")

    output_file = Path(output_filename)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Load existing results into memory once
    results_data = load_results(output_file)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cuda":
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")

    for exp_name, exp_id in experiments.items():
        print(f"\nProcessing {exp_name}...")
        
        # Locate Experiment Dir
        base_dir = Path(f"__local__/experiment-{exp_id}/000000")
        if not base_dir.exists():
            print(f"Warning: Directory not found for {exp_name} at {base_dir}")
            continue

        config_path = base_dir / "configurations.json"
        if not config_path.exists():
            print(f"Warning: No config found for {exp_name}")
            continue
            
        with open(config_path, "r") as f:
            config = json.load(f)

        # Locate Checkpoint
        ckpt_path = get_latest_checkpoint(base_dir)
        if not ckpt_path:
            print(f"Warning: No checkpoint found for {exp_name}")
            continue
            
        print(f"Loading model from {ckpt_path}")
        
        # Instantiate and Load Model manually to inspect layers
        model = VGG(config)
        model.to(device)
        
        # Load weights
        checkpoint = torch.load(ckpt_path, map_location=device)
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        model.eval()
        
        # Identify Target Layers
        target_layers = []
        for name, module in model.named_modules():
            # Exp2: standard Conv2d (check config or name)
            # Exp4: Flex2D
            if exp_name == "Exp2":
                if isinstance(module, nn.Conv2d):
                    target_layers.append(name)
            elif exp_name == "Exp4":
                if isinstance(module, Flex2D):
                    target_layers.append(name)
        
        print(f"Found {len(target_layers)} target layers for {exp_name}")

        # Filter by single layer argument if present
        if args.layer:
            if args.layer not in target_layers:
                print(f"Warning: Requested layer {args.layer} not found in model.")
                # We do NOT skip if not found? Or should we?
                # Let's strictly run only if found or warn.
                # If user asked for 'features.30' but it's not a Conv2d/Flex2D, we might skip.
                # But let's check exact match.
                pass 
            # Overwrite target_layers with just this one
            target_layers = [args.layer]
        
        benchmarks = [
            "FreemanZiemba2013.V1.public-pls",
            "FreemanZiemba2013.V2.public-pls",
            "MajajHong2015.public.V4-pls",
            "MajajHong2015.public.IT-pls"
        ]

        # Calculate cutoff for "early layers" (e.g., first 30%)
        # NOTE: If running a single layer, we lost the global context of "where in the network is this layer".
        # However, earlier logic had `target_layers` as the FULL list before filtering?
        # WAIT. I filtered `target_layers` ABOVE. This messes up `early_layer_cutoff_idx`.
        # FIX: Calculate early layers BEFORE filtering.
        
        # RE-CALCULATE FULL LIST just for early layer detection if we filtered
        # Actually better to just calculate indices from the full list.
        # But wait, I just filtered target_layers.
        # Let's rebuild the full list to determine early/late status.
        
        all_layers = []
        for name, module in model.named_modules():
             if exp_name == "Exp2" and isinstance(module, nn.Conv2d):
                 all_layers.append(name)
             elif exp_name == "Exp4" and isinstance(module, Flex2D):
                 all_layers.append(name)

        early_layer_cutoff_idx = int(len(all_layers) * 0.3)
        early_layers = set(all_layers[:early_layer_cutoff_idx])
        print(f"Defining early layers (skipping V4/IT): {early_layers}")

        for layer in target_layers:
             process_layer(model, layer, exp_name, config, results_data, output_file, benchmarks, early_layers)


    print(f"\nResults saved to {output_file}")

if __name__ == "__main__":
    main()
