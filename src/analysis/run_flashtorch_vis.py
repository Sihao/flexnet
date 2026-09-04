
import argparse
import sys
import os
from pathlib import Path
import random
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import json

# Add project root to sys.path
sys.path.append(os.getcwd())

from src.analysis.run_loader import RunLoader
from src.flex_neurons.utils.device import select_device
from src.flex_neurons.utils.normalization import Normalize, IMAGENET_MEAN, IMAGENET_STD

from flashtorch.activmax import GradientAscent
from flashtorch.utils import format_for_plotting, apply_transforms
import numpy as np
import torch.nn as nn

class ForgivingGradientAscent(GradientAscent):
    def optimize(self, layer, filter_idx, input_=None, num_iter=30):
        """Generates an image that maximally activates the target filter.
           Bypasses strict type checking for Conv2d.
        """

        # Validate the type of the layer - RELAXED CHECK
        if not isinstance(layer, (nn.Conv2d, nn.Linear)):
             # Try to see if it behaves like one (has out_channels or out_features)
             if not hasattr(layer, 'out_channels') and not hasattr(layer, 'out_features'):
                raise TypeError('The layer must be nn.Conv2d/nn.Linear or have out_channels/out_features attribute.')

        # Validate filter index
        if hasattr(layer, 'out_channels'):
            num_total_filters = layer.out_channels
        else:
            num_total_filters = layer.out_features
            
        self._validate_filter_idx(num_total_filters, filter_idx)

        # Inisialize input (as noise) if not provided
        if input_ is None:
            input_ = np.uint8(np.random.uniform(
                150, 180, (self._img_size, self._img_size, 3)))
            input_ = apply_transforms(input_, size=self._img_size)

        if torch.cuda.is_available() and self.use_gpu:
            self.model = self.model.to('cuda')
            input_ = input_.to('cuda')

        # Remove previous hooks if any
        while len(self.handlers) > 0:
            self.handlers.pop().remove()

        # Register hooks to record activation and gradients
        self.handlers.append(self._register_forward_hooks(layer, filter_idx))
        self.handlers.append(self._register_backward_hooks())

        # Inisialize gradients
        self.gradients = torch.zeros(input_.shape)

        # Optimize
        return self._ascent(input_, num_iter)

    def _register_forward_hooks(self, layer, filter_idx):
        def _record_activation(module, input_, output):
            if output.ndim == 2:
                self.activation = torch.mean(output[:, filter_idx])
            else:
                self.activation = torch.mean(output[:, filter_idx, :, :])

        return layer.register_forward_hook(_record_activation)



from src.flex_neurons.models.layers.flex import Flex2D

def run_flashtorch_vis(experiment_id, device_str=None, num_filters=4, layers_to_vis=None):
    """
    Run activation maximization visualization using FlashTorch.
    
    Args:
        experiment_id: Experiment ID or path.
        device_str: Device string (e.g., 'cuda:0', 'cpu').
        num_filters: Number of filters to visualize per layer.
        layers_to_vis: List of layer indices to visualize. If None, selects a representative set.
    """
    
    # 1. Setup
    if isinstance(experiment_id, int) or (isinstance(experiment_id, str) and experiment_id.isdigit()):
        exp_path = f"__local__/experiment-{experiment_id}/000000"
    else:
        exp_path = experiment_id

    print(f"Loading Experiment from {exp_path}...")
    
    output_dir = Path(exp_path) / "results" / "flashtorch_vis"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results will be saved to: {output_dir}")

    if device_str:
        device = torch.device(device_str)
    else:
        device = select_device()
    print(f"Using device: {device}")
    
    use_gpu = device.type == 'cuda'

    # 2. Load Model
    try:
        loader = RunLoader(exp_path, device=device)
    except Exception as e:
        print(f"Error loading experiment from {exp_path}: {e}")
        return

    # Wrap model with Normalize so we can optimize in [0,1] space
    normalization = Normalize(IMAGENET_MEAN, IMAGENET_STD).to(device)
    raw_model = loader.model
    model = nn.Sequential(normalization, raw_model)
    model.eval()
    model.to(device)  
    
    # 3. Identify Layers (Conv2d or Flex2D)
    print("Initializing ActivationMaximization (ForgivingGradientAscent)...")
    visualizer = ForgivingGradientAscent(model, img_size=224, lr=1.0, use_gpu=use_gpu)

    # Find interesting layers in the raw model (index 1 of Sequential)
    print("Searching for Conv2d/Flex2D layers...")
    candidate_layers = []
    
    # We want to avoid picking children if parent is selected.
    # Strategy: detailed iteration or just use named_modules and filter names.
    # If we see a Flex2D, we accept it.
    # If we see a Conv2d, we check if it is a child of a Flex2D we already saw?
    # Actually, iterate, store all, then filter.
    
    all_modules = dict(raw_model.named_modules())
    
    # Identify Flex2D parents
    flex_parents = {}
    for name, module in all_modules.items():
        if isinstance(module, Flex2D):
             flex_parents[name] = module

    final_layers = []
    
    for name, module in all_modules.items():
        # Check if this module is a Flex2D or Conv2d
        is_flex = isinstance(module, Flex2D)
        is_conv = isinstance(module, nn.Conv2d)
        
        if is_flex:
            full_name = name
            final_layers.append((full_name, module))
        elif is_conv:
            # Check if this conv is inside a Flex2D
            # If `name` starts with any flex parent name + '.', skip it
            is_child_of_flex = False
            for parent_name in flex_parents:
                if name.startswith(parent_name + "."):
                    is_child_of_flex = True
                    break
            
            if not is_child_of_flex:
                full_name = name
                final_layers.append((full_name, module))
            
    # Sort by name length or just assume ordered? named_modules is depth-first usually.
    # But dictionary iteration might not be ordered in older python? Python 3.7+ is ordered.
    # Let's rely on stored list order if we iterate over named_modules again to be safe?
    # Re-doing properly:
    
    final_layers = []
    processed_prefixes = set()
    
    for name, module in raw_model.named_modules():
        # Clean logic: 
        # If it is Flex2D, take it, mark name as processed prefix.
        # If it is Conv2d, check if prefix processed.
        
        if isinstance(module, Flex2D):
            final_layers.append((name, module))
            processed_prefixes.add(name)
        elif isinstance(module, nn.Conv2d):
            # Check if child of processed
            is_child = any(name.startswith(p + ".") for p in processed_prefixes)
            if not is_child:
                final_layers.append((name, module))
            
    print(f"Found {len(final_layers)} interesting layers.")
    
    if not final_layers:
        print("No Conv2d or Flex2D layers found!")
        return

    # Select representative layers if not specified
    if args.visualize_logits:
        print("Mode: Visualizing Classifier Logits")
        # Find the last Linear layer
        classifier_layer = None
        layer_name = None
        for name, module in raw_model.named_modules():
            if isinstance(module, nn.Linear):
                classifier_layer = module
                layer_name = name
        
        if classifier_layer:
            # Override target layers to just this one
            target_layers = [(layer_name, classifier_layer)]
        else:
            print("No linear layer found for logits visualization.")
            return
    elif layers_to_vis is None:
        # Pick start, middle, end
        n = len(final_layers)
        indices = sorted(list(set([0, n//4, n//2, 3*n//4, n-1])))
        target_layers = [final_layers[i] for i in indices]
    else:
        target_layers = [final_layers[i] for i in layers_to_vis if i < len(final_layers)]

    print(f"Targeting {len(target_layers)} layers: {[n for n, _ in target_layers]}")

    # 4. Generate Visualizations
    for layer_name, layer_module in target_layers:
        print(f"Visualizing layer: {layer_name}")
        
        if args.visualize_logits:
             # For logits, num_filters is interpreted as number of classes to visualize
             # Let's visualize random classes or first N
             filter_idxs = range(min(num_filters, layer_module.out_features))
        else:
             filter_idxs = range(min(num_filters, layer_module.out_channels))
        
        for filter_idx in filter_idxs:
            print(f"  Filter/Class {filter_idx}...")
            try:
                # optimize returns output
                output = visualizer.optimize(layer_module, filter_idx, num_iter=30)
                
                # Check output type
                # If output is a list of tensors, take the last one
                if isinstance(output, list):
                     img_tensor = output[-1]
                else:
                     img_tensor = output

                # Move to CPU
                img_tensor = img_tensor.cpu()
                
                # Save image
                if args.visualize_logits:
                     save_name = f"logit_{layer_name}_class_{filter_idx}.png"
                else:
                     save_name = f"layer_{layer_name}_filter_{filter_idx}.png"
                
                save_path = output_dir / save_name
                
                plt.figure(figsize=(4, 4))
                img_np = format_for_plotting(img_tensor) 
                plt.imshow(img_np)
                if args.visualize_logits:
                    plt.title(f"Class {filter_idx}")
                else:
                    plt.title(f"{layer_name} Filter {filter_idx}")
                plt.axis('off')
                plt.savefig(save_path, bbox_inches='tight')
                plt.close()
                
            except Exception as e:
                print(f"Failed to visualize {layer_name} filter {filter_idx}: {e}")
                import traceback
                traceback.print_exc()

    print(f"Done. Check results in {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run FlashTorch Visualization")
    parser.add_argument("--experiment", type=str, required=True, help="Experiment ID or path")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    parser.add_argument("--num_filters", type=int, default=4, help="Number of filters to visualize per layer")
    parser.add_argument("--visualize_logits", action="store_true", help="Visualize classifier logits instead of conv layers")
    
    args = parser.parse_args()
    
    run_flashtorch_vis(
        experiment_id=args.experiment,
        device_str=args.device,
        num_filters=args.num_filters,
        layers_to_vis=None # Handled inside
    )
