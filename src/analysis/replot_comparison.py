
import sys
import os
import json
from pathlib import Path
sys.path.append(os.getcwd())

from plotting import compare_attack_results

def main():
    exp2_id = "2"
    exp4_id = "4"

    print(f"\nRunning Replot...")
    def load_results(eid, specific_timestamp=None):
        base_dir = Path(f"__local__/experiment-{eid}/000000/results/attack_comparison")
        if not base_dir.exists():
            print(f"Base dir not found: {base_dir}")
            return None
        
        if specific_timestamp:
            latest = base_dir / specific_timestamp
        else:
            # Find latest timestamp dir, ignoring non-timestamp dirs like 'attack_examples'
            dirs = sorted([d for d in base_dir.iterdir() if d.is_dir() and d.name.split('_')[0].isdigit()], key=lambda x: x.name)
            if not dirs:
                print(f"No results directories found in {base_dir}")
                return None
            latest = dirs[-1]

        print(f"Loading results from {latest} for Exp {eid}")
        res_file = latest / "attack_comparison_results.json"
        if not res_file.exists():
            print(f"Result file not found: {res_file}")
            return None
        
        # Robust load
        with open(res_file, "r") as f:
            content = f.read()
            # Fix potential trailing comma issue if present (from original script)
            content = content.replace(",\n}", "\n}")
            try:
                return json.loads(content)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON in {res_file}: {e}")
                return None

    # Use explicit valid timestamp for Exp 2 to avoid the partial run
    res2 = load_results(exp2_id, specific_timestamp="20260105_182152")
    res4 = load_results(exp4_id) # Let it pick latest for Exp 4 (which was 20260105_183223)
    
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
