import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import numpy as np
import pandas as pd
import json
import glob
import math
from pathlib import Path
from contextlib import contextmanager
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from matplotlib.colors import LinearSegmentedColormap, LightSource
try:
    from scipy.interpolate import griddata
except ImportError:
    pass

# --- Constants ---
FONT_SIZE = 10.5
FONT_FAMILY = "sans-serif"
FIGURE_SIZE_DEFAULT = (18.3 / 2.54, 9 / 2.54) # approx 7.2 x 3.5 inches
TITLE_PAD = 17
COLORS = ["#14213D", "#003D71", "#FCA311", "#E5E5E5"] # Dark Blue 1, Dark Blue 2, Orange, Light Grey

def get_style_params(font_size=FONT_SIZE):
    """Returns the dictionary of style parameters."""
    return {
        "axes.titlesize": 10,
        "axes.labelsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "figure.figsize": FIGURE_SIZE_DEFAULT,
        "axes.titlepad": TITLE_PAD,
        "text.usetex": True,
        "font.family": FONT_FAMILY,
        "text.latex.preamble": r"\usepackage{helvet} \usepackage[T1]{fontenc} \usepackage{sfmath}",
        "font.size": font_size,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.prop_cycle": plt.cycler(color=COLORS),
        "axes.formatter.limits": (0, 0),
        "axes.formatter.useoffset": False,
    }

@contextmanager
def plotting_style(font_size=FONT_SIZE):
    """
    Context manager for applying the project's plotting style temporarily.
    """
    style_params = get_style_params(font_size)
    with plt.rc_context(style_params):
        yield

def set_style():
    """
    Sets the global style.
    Deprecated: Prefer using the `plotting_style` context manager.
    """
    style_params = get_style_params()
    for key, value in style_params.items():
        plt.rcParams[key] = value

def save_plot(fig, output_path, dpi=300, pad_inches=None):
    """
    Helper to save a figure to PNG and SVG.
    """
    if not output_path:
        return

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save PNG
    fig.savefig(path, dpi=dpi, bbox_inches='tight', pad_inches=pad_inches)
    
    # Save SVG
    svg_path = path.with_suffix(".svg")
    fig.savefig(svg_path, bbox_inches='tight', pad_inches=pad_inches)
    
    print(f"Plots saved to {path} and {svg_path}")

def get_module_by_name(model, layer_name):
    """
    Recursively get a module by dot path or integer index or direct attribute.
    Handles 'features.0', 'layer1', '0', etc.
    """
    
    # helper for recursion
    def recursive_get(curr, path):
        for part in path.split("."):
            if part.isdigit() and hasattr(curr, "__getitem__"):
                 # Handle nn.Sequential or list-like
                curr = curr[int(part)]
            elif hasattr(curr, part):
                curr = getattr(curr, part)
            elif part.isdigit() and isinstance(curr, torch.nn.Sequential):
                curr = curr[int(part)]
            else:
                return None
        return curr

    target = None
    
    # Case 1: Integer or Digit String (common for VGG features)
    if isinstance(layer_name, int) or (isinstance(layer_name, str) and layer_name.isdigit()):
        idx = int(layer_name)
        if hasattr(model, "features"):
            target = model.features[idx]
        elif isinstance(model, torch.nn.Sequential):
             # If the model itself is sequential
            target = model[idx]
            
    # Case 2: Dot notation or Attribute
    if target is None:
        target = recursive_get(model, str(layer_name))
        
    # Case 3: Fallback direct attribute check if dot split logic failed or wasn't applicable
    if target is None and hasattr(model, str(layer_name)):
        target = getattr(model, str(layer_name))
        
    return target


def plot_training_progress(
    epochs,
    means,
    stds,
    exp_id,
    window=1,
    output_file=None,
    show_plot=False,
    figsize=None,
):
    """
    Generates and saves a training progress plot using aggregated data.
    Uses the global style settings.

    Args:
        epochs (array-like): List or array of epoch numbers.
        means (pd.DataFrame): DataFrame containing mean values for metrics (Train Accuracy, Valid Accuracy, Train Loss, Valid Loss).
        stds (pd.DataFrame): DataFrame containing standard deviation values for metrics.
        exp_id (int or str): Experiment identifier for the title.
        window (int, optional): Smoothing window size used for the data. Defaults to 1.
        output_file (str, optional): Path to save the plot image. If None, plot is not saved. Defaults to None.
        show_plot (bool, optional): Whether to display the plot interactively. Defaults to False.
        figsize (tuple, optional): Figure size (width, height) in inches. If None, uses a default vertical layout size. Defaults to None.
    """
    if figsize is None:
        base_width = 5 / 2.54
        base_height = base_width * 1.618
        figsize = (base_width, base_height)

    # Use subplot_mosaic
    # Top: Loss, Bottom: Accuracy
    # Legend is now inside the Loss plot
    fig, axes_dict = plt.subplot_mosaic(
        [["loss"], ["accuracy"]], figsize=figsize, layout="constrained"
    )

    ax1 = axes_dict["loss"]  # Loss is now Top
    ax2 = axes_dict["accuracy"]  # Accuracy is now Bottom

    # Get colors from the cycler
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    c_train = colors[0]  # Dark Blue 1
    c_valid = colors[2]  # Orange

    # --- Loss (Top Plot) ---
    ax1.plot(epochs, means["Train Loss"], label="Training", color=c_train)
    ax1.fill_between(
        epochs,
        means["Train Loss"] - stds["Train Loss"],
        means["Train Loss"] + stds["Train Loss"],
        color=c_train,
        alpha=0.2,
    )

    ax1.plot(epochs, means["Valid Loss"], label="Validation", color=c_valid)
    ax1.fill_between(
        epochs,
        means["Valid Loss"] - stds["Valid Loss"],
        means["Valid Loss"] + stds["Valid Loss"],
        color=c_valid,
        alpha=0.2,
    )

    ax1.set_ylabel("Loss")
    ax1.set_ylim(0, 5)
    # Share X axis
    ax1.sharex(ax2)
    plt.setp(ax1.get_xticklabels(), visible=False)

    # --- Legend (Inside Top Plot) ---
    # Custom legend with shorter marker lines (handlelength)
    custom_lines = [
        plt.Line2D([0], [0], color=c_train, lw=4),
        plt.Line2D([0], [0], color=c_valid, lw=4),
    ]

    # "put the legend inside the first plot in the top right with no bounding box"
    # "make the marker line a little shorter inside the legend than it currently is"
    ax1.legend(
        custom_lines,
        ["Training", "Validation"],
        frameon=False,
        loc="upper right",
        ncol=1,
        handlelength=1.0,
    )  # Reduced handle length

    # --- Accuracy (Bottom Plot) ---
    ax2.plot(epochs, means["Train Accuracy"], label="Train Acc", color=c_train)
    ax2.fill_between(
        epochs,
        means["Train Accuracy"] - stds["Train Accuracy"],
        means["Train Accuracy"] + stds["Train Accuracy"],
        color=c_train,
        alpha=0.2,
    )

    ax2.plot(epochs, means["Valid Accuracy"], label="Valid Acc", color=c_valid)
    ax2.fill_between(
        epochs,
        means["Valid Accuracy"] - stds["Valid Accuracy"],
        means["Valid Accuracy"] + stds["Valid Accuracy"],
        color=c_valid,
        alpha=0.2,
    )

    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_ylim(0, 1.0)

    # Explicitly turn off offset (scientific notation) on x-axis to prevent 10^1 on top plot
    ax1.ticklabel_format(style="plain", axis="x", useOffset=False)
    ax2.ticklabel_format(style="plain", axis="x", useOffset=False)

    if output_file:
        save_plot(fig, output_file)

    if show_plot:
        plt.show()


def plot_conv_ratios(
    epochs,
    means,
    stds,
    layer_num,
    exp_id,
    window=1,
    output_file=None,
    show_plot=False,
    figsize=None,
):
    """
    Generates and saves a Conv Ratio plot.
    If layer_num is None, plots ALL layers found in means as a heatmap.
    If layer_num is specified, plots a line graph for that specific layer.

    Args:
        epochs (array-like): List or array of epoch numbers.
        means (pd.DataFrame): DataFrame containing mean values for metrics. Must contain "Conv Ratio {layer_num}" columns.
        stds (pd.DataFrame): DataFrame containing standard deviation values for metrics.
        layer_num (int, optional): The specific convolutional layer to plot. If None, plots all layers. Defaults to None.
        exp_id (int or str): Experiment identifier for the title.
        window (int, optional): Smoothing window size used for the data. Defaults to 1.
        output_file (str, optional): Path to save the plot image. If None, plot is not saved. Defaults to None.
        show_plot (bool, optional): Whether to display the plot interactively. Defaults to False.
        figsize (tuple, optional): Figure size (width, height) in inches. If None, uses the global rcParams default. Defaults to None.
    """
    if figsize is None:
        # Use default figsize from rcParams
        figsize = plt.rcParams["figure.figsize"]

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    if layer_num is not None:
        # Plot single layer
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        c_line = colors[0]  # Dark Blue 1

        col_name = f"Conv Ratio {layer_num}"
        ax.plot(epochs, means[col_name], label=f"Layer {layer_num}", color=c_line)
        ax.fill_between(
            epochs,
            means[col_name] - stds[col_name],
            means[col_name] + stds[col_name],
            color=c_line,
            alpha=0.2,
        )

        ax.set_title(
            f"Experiment {exp_id}: Conv Ratio Layer {layer_num}"
            + (f" (Smoothing: {window})" if window > 1 else "")
        )
        custom_lines = [plt.Line2D([0], [0], color=c_line, lw=4)]
        ax.legend(custom_lines, [f"Layer {layer_num}"], frameon=False)
    else:
        # Plot all layers as Heatmap
        # Create custom colormap: #14213D -> #FCA311 (mid) -> #E5E5E5
        # Prepare data: Rows = Layers, Cols = Epochs
        c_start = "#14213D"
        c_mid = "#FCA311"
        c_end = "#E5E5E5"

        # Create colormap
        cmap = LinearSegmentedColormap.from_list(
            "custom_blue_orange_grey", [c_start, c_mid, c_end]
        )

        # Prepare data: Rows = Layers, Cols = Epochs
        heatmap_data = means.T.values  # Shape: (n_layers, n_epochs)

        # Create Heatmap
        im = ax.imshow(
            heatmap_data,
            aspect="auto",
            cmap=cmap,
            origin="lower",
            interpolation="nearest",
            vmin=0,
            vmax=1,
            extent=[epochs[0], epochs[-1], -0.5, len(means.columns) - 0.5],
        )

        ax.set_title(
            f"Experiment {exp_id}: Conv Ratio Heatmap"
            + (f" (Smoothing: {window})" if window > 1 else "")
        )
        ax.set_ylabel("Layer")
        ax.set_xlabel("Epoch")

        # Set Y-ticks to layer numbers
        y_ticks = range(len(means.columns))
        ax.set_yticks(y_ticks)

        # Extract layer numbers for labels
        y_labels = []
        for col in means.columns:
            try:
                y_labels.append(col.split(" ")[-1])
            except:
                y_labels.append(col)
        ax.set_yticklabels(y_labels)

        # Add colorbar
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Conv Ratio")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Conv Ratio")

    if output_file:
        save_plot(fig, output_file)

    if show_plot:
        plt.show()


def visualize_activations(
    model, layer_name, image_path, output_file=None, top_k=10, show_plot=False
):
    """
    Visualizes activations of a specific layer in the model for a given image.

    Args:
        model (nn.Module): The PyTorch model.
        layer_name (str or int): The name or index of the layer to visualize.
        image_path (str): Path to the input image.
        output_file (str, optional): Path to save the plot.
        top_k (int, optional): Number of top active channels to display. Defaults to 10.
        show_plot (bool, optional): Whether to show the plot interactively.
    """
    device = next(model.parameters()).device

    # 1. Register Hook
    activation = {}

    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()

        return hook

    # Hook using helper
    print(f"Hooking into {layer_name}...")
    hook_handle = None
    
    target_layer = get_module_by_name(model, layer_name)
    if target_layer:
        hook_handle = target_layer.register_forward_hook(get_activation(str(layer_name)))
    else:
        raise ValueError(f"Error: Layer {layer_name} not found in model.")

    # 2. Prepare Image
    print(f"Loading image from {image_path}...")
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    try:
        raw_image = Image.open(image_path).convert("RGB")
        input_tensor = transform(raw_image).unsqueeze(0).to(device)
    except Exception as e:
        raise RuntimeError(f"Error loading image from {image_path}: {e}")

    # 3. Forward Pass
    print("Running forward pass...")
    model.eval()
    with torch.no_grad():
        model(input_tensor)

    # Remove hook
    # if hook_handle: hook_handle.remove() # Optional cleanup

    # 4. Visualization
    key = str(layer_name)
    if key not in activation:
        print(f"No activation captured for {layer_name}")
        return

    act = activation[key].squeeze()  # [C, H, W]

    # Calculate Mean Activation per channel
    mean_acts = act.mean(dim=(1, 2))
    # Get Top K indices
    top_indices = torch.argsort(mean_acts, descending=True)[:top_k]

    # Prepare Plot Layout
    # Layout: One row. Input Image (Left), Top 10 Activations (Right).

    num_plots = top_k + 1  # +1 for input

    # 177 mm width
    total_width_mm = 177
    total_width_inch = total_width_mm / 25.4

    # Estimate height: images are square.
    # Width per subplot approx total_width / num_plots (ignoring margins for calculation)
    # This keeps aspect ratio roughly 1:1 for the data area
    subplot_width_inch = total_width_inch / num_plots
    # Title needs a bit of space, but user wants to "fill space".
    # Let's keep it tight.
    total_height_inch = subplot_width_inch * 1.25

    fig = plt.figure(figsize=(total_width_inch, total_height_inch))
    # wspace=0.02 creates a tiny gap between images. left/right/bottom=0 fills headers.
    # top=0.8 leaves 20% for "Input" title.
    gs = fig.add_gridspec(
        1, num_plots, wspace=0.02, left=0.0, right=1.0, bottom=0.0, top=0.85
    )

    # --- Input Image ---
    ax_orig = fig.add_subplot(gs[0, 0])

    # Inverse Normalize for display
    inv_normalize = transforms.Normalize(
        mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
        std=[1 / 0.229, 1 / 0.224, 1 / 0.225],
    )
    img_disp = inv_normalize(input_tensor.squeeze().cpu()).permute(1, 2, 0).numpy()
    img_disp = np.clip(img_disp, 0, 1)

    ax_orig.imshow(img_disp)
    # ax_orig.set_title("Input", fontsize=8) # Small font for small plot
    ax_orig.axis("off")

    # --- Activations ---
    for i, idx in enumerate(top_indices):
        c = i + 1  # Offset by 1 for input image

        ax = fig.add_subplot(gs[0, c])

        channel_img = act[idx].cpu().numpy()
        im = ax.imshow(channel_img, cmap="plasma")
        # ax.set_title(f"Ch {idx}\n({mean_acts[idx]:.1f})", fontsize=8) # Removed labels
        ax.axis("off")

    # plt.suptitle(f"Top {top_k} Activations - Layer {layer_name}") # Removed suptitle for compactness? Or keep? User didn't explicitly ask to remove suptitle, but "remove channel labels".
    # For publication quality usually suptitle is caption. I'll maintain it but maybe make it optional or smaller?
    # Given constraint "make total width... 88mm", simpler is better. I will comment it out or keep it very small.
    # User instruction was precise about labels. I'll leave suptitle out to ensure clean 88mm fit.

    if output_file:
        save_plot(fig, output_file)

    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def visualize_filters(model, layer_name, output_file=None, top_k=64, show_plot=False):
    """
    Visualizes filters (kernels) of a specific layer.

    Args:
        model (nn.Module): The PyTorch model.
        layer_name (str or int): The name or index of the layer to visualize.
        output_file (str, optional): Path to save the plot.
        top_k (int, optional): Max number of filters to display.
        show_plot (bool, optional): Whether to show the plot interactively.
    """
    # 1. Retrieve Layer
    target_layer = get_module_by_name(model, layer_name)
    if target_layer is None:
        raise ValueError(f"Error: Layer {layer_name} not found in model.")

    # 2. Extract Weights
    weights = None
    if isinstance(target_layer, nn.Conv2d):
        weights = target_layer.weight.data
    elif hasattr(target_layer, "flex_conv") and isinstance(
        target_layer.flex_conv, nn.Conv2d
    ):
        weights = target_layer.flex_conv.weight.data
    else:
        raise ValueError(
            f"Error: Layer {layer_name} is not a Conv2d or Flex2D layer ({type(target_layer)})."
        )

    # Weights Shape: (Out, In, H, W)
    num_filters, in_channels, h, w = weights.shape
    print(f"Visualizing {num_filters} filters of shape {in_channels}x{h}x{w}...")

    # Limit to top_k
    # If RGB, we keep (Out, 3, H, W) and iterate Out
    # If Deep, user wants individual kernels. Flatten (Out, In, H, W) -> (Out*In, H, W)

    if in_channels == 3:
        if num_filters > top_k:
            weights = weights[:top_k]
            num_filters = top_k
        print(f"Visualizing {num_filters} RGB filters...")
    else:
        # Flatten: (Out, In, H, W) -> (Out*In, H, W)
        weights = weights.view(-1, h, w)
        total_kernels = weights.shape[0]
        if total_kernels > top_k:
            weights = weights[:top_k]
            num_filters = top_k
        else:
            num_filters = total_kernels
        print(
            f"Visualizing {num_filters} individual kernels (flattened from {in_channels}x{h}x{w})..."
        )

    weights = weights.cpu()

    # 3. Process Weights for Display
    # Grid calculation
    grid_size = math.ceil(math.sqrt(num_filters))
    fig, axes = plt.subplots(
        grid_size, grid_size, figsize=(12, 12), constrained_layout=True
    )

    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    axes = axes.flatten()

    for i in range(num_filters):
        ax = axes[i]

        if in_channels == 3:
            kernel = weights[i]  # (3, H, W)
            # Assume First Layer -> RGB Visualization
            # Normalize to 0-1 range for display
            # Min-Max Normalization per filter
            k_min, k_max = kernel.min(), kernel.max()
            if k_max - k_min > 1e-5:
                kernel = (kernel - k_min) / (k_max - k_min)
            else:
                kernel = torch.zeros_like(kernel)

            # (3, H, W) -> (H, W, 3)
            img = kernel.permute(1, 2, 0).numpy()
            ax.imshow(img, interpolation="nearest")

        else:
            # Deep Layer -> Individual Kernel Visualization
            kernel = weights[i]  # (H, W)
            ax.imshow(kernel.numpy(), cmap="viridis", interpolation="nearest")

        ax.axis("off")

    # Hide unused axes
    for i in range(num_filters, len(axes)):
        axes[i].axis("off")

    plt.suptitle(
        f"Filters - Layer {layer_name} ({'RGB' if in_channels==3 else 'Individual Kernels'})"
    )

    if output_file:
        save_plot(fig, output_file)

    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def plot_single_attack(
    epsilons,
    accuracies,
    attack_name,
    experiment_id,
    output_file=None,
    show_plot=False
):
    """
    Plots the accuracy vs. epsilon curve for a single attack.
    """
    # Match size to one subplot of progress plot
    base_width = 8 / 2.54
    base_height = (base_width * 1.618) / 2
    figsize = (base_width, base_height)

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

    c_line = COLORS[0] # Dark Blue 1
    
    ax.plot(
        epsilons, accuracies, marker="o", linewidth=2, markersize=5, label=attack_name, color=c_line
    )

    ax.set_xlabel("Epsilon")
    ax.set_ylabel("Top-1 Accuracy")
    ax.set_title(f"{attack_name} Attack (Exp {experiment_id})")
    ax.legend(frameon=False)
    ax.set_ylim(0, 1.0)
    
    if output_file:
        save_plot(fig, output_file)

    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def plot_experiment_attacks(
    results,
    experiment_id,
    output_path=None,
    show_plot=False
):
    """
    Plots the curves for all attacks found in a single experiment's results.
    Creates a row of subplots.
    
    Args:
        results (dict): Dictionary where keys are attack names and values have "epsilons", "accuracies".
    """
    attacks = sorted(list(results.keys()))
    if "metadata" in attacks:
        attacks.remove("metadata")
        
    num_attacks = len(attacks)
    if num_attacks == 0:
        print("No attacks to plot.")
        return

    # Dimensions
    total_width_mm = 177
    total_width_inch = total_width_mm / 25.4
    subplot_width_inch = total_width_inch / max(num_attacks, 1)
    total_height_inch = subplot_width_inch * 1.1

    fig, axes = plt.subplots(
        1,
        num_attacks,
        figsize=(total_width_inch, total_height_inch),
        constrained_layout=True,
    )

    if num_attacks == 1:
        axes = [axes]
    elif not isinstance(axes, np.ndarray):
        axes = [axes]

    for i, attack_name in enumerate(attacks):
        ax = axes[i]
        data = results[attack_name]
        
        eps = data.get("epsilons", [])
        acc = data.get("accuracies", [])
        
        c = COLORS[i % len(COLORS)]
        
        ax.plot(eps, acc, marker="o", linewidth=2, markersize=5, color=c, label=attack_name)
        
        ax.set_title(attack_name, fontsize=10)
        ax.set_xlabel("Epsilon")
        
        if i == 0:
            ax.set_ylabel("Accuracy")
        else:
            ax.set_yticklabels([])
            
        ax.set_ylim(-0.05, 1.05)
        ax.tick_params(axis="both", which="major", labelsize=8)
        if i > 0:
             ax.tick_params(axis='y', labelleft=False)

    if output_path:
        save_plot(fig, output_path)

    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def compare_experiment_attacks(
    results1,
    results2,
    label1="Exp 1",
    label2="Exp 2",
    output_path=None,
    show_plot=False
):
    """
    Plots a comparison of all attacks between two experiments.
    """
    all_keys = set(list(results1.keys()) + list(results2.keys()))
    if "metadata" in all_keys:
        all_keys.remove("metadata")
        
    attacks = sorted(list(all_keys))
    num_attacks = len(attacks)
    
    if num_attacks == 0:
        return

    # Geometry
    total_width_mm = 186
    total_width_inch = total_width_mm / 25.4
    subplot_width_inch = total_width_inch / num_attacks
    total_height_inch = subplot_width_inch * 1.1 * 0.8

    fig, axes = plt.subplots(
        1,
        num_attacks,
        figsize=(total_width_inch, total_height_inch),
        constrained_layout=True,
    )
    
    if num_attacks == 1:
        axes = [axes]
    elif not isinstance(axes, np.ndarray):
        axes = [axes]
        
    c1 = COLORS[0]
    c2 = COLORS[2]
    
    for i, attack_name in enumerate(attacks):
        ax = axes[i]
        
        if attack_name in results1:
            d = results1[attack_name]
            ax.plot(d.get("epsilons", []), d.get("accuracies", []), 
                    marker="o", markersize=4, linewidth=1.5, color=c1, label=label1)
            
        if attack_name in results2:
            d = results2[attack_name]
            ax.plot(d.get("epsilons", []), d.get("accuracies", []),
                    marker="s", markersize=4, linewidth=1.5, color=c2, linestyle="--", label=label2)
            
        ax.set_title(attack_name, fontsize=10)
        ax.set_xlabel(r"$\epsilon$")
        
        if i == 0:
            ax.set_ylabel("Accuracy")
        else:
            ax.set_yticklabels([])
            
        if i == num_attacks - 1:
            ax.legend(frameon=False, fontsize=8)
            
        ax.set_ylim(-0.05, 1.05)
        ax.tick_params(axis="both", which="major", labelsize=8)
        if i > 0:
             ax.tick_params(axis='y', labelleft=False)

    if output_path:
        save_plot(fig, output_path)

    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def plot_spectral_results(
    profiles,
    slopes,
    mean_slope,
    std_slope,
    layer_name,
    nyquist,
    output_dir,
    ylim_max=None,
):
    """
    Generates and saves the radial power spectra and slope distribution plots.
    Aligned with project styling (Helvetica, colors, spines).

    Args:
        profiles (list): List of radial profile arrays.
        slopes (list): List of slope values.
        mean_slope (float): Mean slope.
        std_slope (float): Std dev of slope.
        layer_name (str): Name of the layer analyzed.
        nyquist (int): Nyquist limit.
        output_dir (Path): Directory to save plots.
        ylim_max (int, optional): Manual override for slope histogram Y-axis limit.
    """
    # --- Style Settings ---
    # Match plot_training_progress sizing
    base_width = 5 / 2.54
    base_height = base_width * 1.618
    figsize = (base_width, base_height)

    # Use general settings but enforce size for this plot
    # Re-using the rcParams from global set_style where possible, but ensuring specific params here
    plt.rcParams["axes.titlesize"] = 10
    plt.rcParams["axes.labelsize"] = 8
    plt.rcParams["xtick.labelsize"] = 8
    plt.rcParams["ytick.labelsize"] = 8
    plt.rcParams["axes.titlepad"] = 16.8
    plt.rcParams["text.usetex"] = True
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["text.latex.preamble"] = (
        r"\usepackage{helvet} \usepackage[T1]{fontenc} \usepackage{sfmath}"
    )
    plt.rcParams["font.size"] = 10.45619
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["axes.prop_cycle"] = plt.cycler(
        color=["#14213D", "#003D71", "#FCA311", "#E5E5E5"]
    )

    # Colors
    c_blue1 = "#14213D"
    c_blue2 = "#003D71"
    c_orange = "#FCA311"
    c_grey = "#E5E5E5"

    # Vertical Layout: 2 rows, 1 col
    fig, axes = plt.subplots(2, 1, figsize=figsize, layout="constrained")

    # Plot 1: Radial Power Spectra (Top)
    ax = axes[0]
    # Restrict to Nyquist
    # Use 1 to nyquist (inclusive) to avoid log(0) and go up to cutoff
    valid_range = slice(1, int(nyquist) + 1)
    plot_freqs = np.arange(len(profiles[0]))[valid_range]

    # Subsample profiles for plotting if there are too many
    num_profiles = len(profiles)
    if num_profiles > 1000:
        indices = np.random.choice(num_profiles, 1000, replace=False)
        plot_profiles = [profiles[i][valid_range] for i in indices]
        title_suffix = f" - 1000/{num_profiles} Sampled"
    else:
        plot_profiles = [p[valid_range] for p in profiles]
        title_suffix = ""

    # Plot individual profiles (light blue/greyish, very transparent)
    for prof in plot_profiles:
        ax.loglog(plot_freqs, prof, alpha=0.05, color=c_blue1, linewidth=0.5)

    # Plot Mean Profile
    mean_profile = np.mean(profiles, axis=0)[valid_range]
    ax.loglog(
        plot_freqs, mean_profile, color=c_orange, linewidth=2, label="Mean Profile"
    )

    # ax.set_title(f"Radial Power Spectra\n({layer_name}){title_suffix}")
    ax.set_xlabel("Spatial Frequency")
    ax.set_ylabel("Power")
    # ax.legend(frameon=False)

    # Set explicit x-limits
    ax.set_xlim(1, nyquist)
    ax.set_ylim(1e-6, 1e2)

    # Plot 2: Slope Distribution
    ax = axes[1]
    counts, bins, patches = ax.hist(
        slopes, bins=10, color=c_grey, edgecolor=c_blue1, alpha=0.8
    )
    ax.axvline(
        mean_slope,
        color=c_orange,
        linestyle="--",
        linewidth=2,
        label=f"Mean: {mean_slope:.2f} +/- {std_slope:.2f}",
    )

    # Calculate Y-Limit
    if ylim_max is not None:
        final_ylim = ylim_max
    else:
        max_count = np.max(counts)
        # Nearest 1000 larger than max_count
        # If max_count is 2100 -> 3000
        # If max_count is 3000 -> 3000 or 4000?
        # User said "larger than". If exactly 3000, 3000 is not larger. So maybe +1?
        # But usually standard ceil logic is sufficient. Let's do ceil(max/1000)*1000.
        # If we really want "strictly larger", we could check.
        # But ceiling usually implies ensuring plot containment.

        final_ylim = math.ceil(max_count / 1000) * 1000
        if final_ylim == 0:
            final_ylim = 1000  # Safety

    # ax.set_title(f"Slope Distribution\n(Mean: {mean_slope:.2f} +/- {std_slope:.2f})")
    ax.set_xlabel("Slope")
    ax.set_ylabel("Count")
    ax.set_ylim(0, final_ylim)
    # ax.legend(frameon=False)

    # Add text annotation
    text_str = (
        r"$\mu = {:.2f}$".format(mean_slope)
        + "\n"
        + r"$\sigma = {:.2f}$".format(std_slope)
    )
    ax.text(0.95, 0.95, text_str, transform=ax.transAxes, ha="right", va="top")

    # Save
    # Save
    if output_dir:
        filename = f"spectral_analysis_{layer_name}_{len(slopes)}imgs.png"
        save_path = Path(output_dir) / filename
        save_plot(fig, save_path)
    else:
        plt.close(fig)


def plot_spectral_results_from_disk(paths, output_file=None, ylim_max=None):
    """
    Loads spectral analysis results from disk and generates plots.

    Args:
        paths (dict): Dictionary of explicit paths ('metadata', 'spectra', 'slopes', 'output_dir').
        output_file (str, optional): Output filename override.
        ylim_max (int, optional): Manual override for Y-limit.
    """
    if not paths or not isinstance(paths, dict):
        raise ValueError(
            "Error: 'paths' argument must be a dictionary containing file paths."
        )

    metadata_file = Path(paths["metadata"])
    spectra_file = Path(paths["spectra"])
    slopes_file = Path(paths["slopes"])
    # Optional result dir for saving if output_file not absolute or None
    results_dir = Path(paths.get("output_dir", metadata_file.parent))

    print(f"Using explicit paths:")
    print(f"  Metadata: {metadata_file}")
    print(f"  Spectra:  {spectra_file}")
    print(f"  Slopes:   {slopes_file}")

    if not metadata_file.exists():
        raise FileNotFoundError(f"Error: Metadata file not found: {metadata_file}")

    print(f"Loading metadata from {metadata_file}...")

    with open(metadata_file, "r") as f:
        metadata = json.load(f)

    mean_slope = metadata["mean_slope"]
    std_slope = metadata["std_slope"]
    nyquist = metadata["nyquist"]
    num_samples = metadata["num_samples"]
    layer_name = metadata.get("layer_name", "unknown_layer")

    # 2. Load Spectra
    if not spectra_file.exists():
        raise FileNotFoundError(f"Error: Spectra file not found: {spectra_file}")
    print(f"Loading spectra from {spectra_file}...")
    profiles = np.load(spectra_file)

    # 3. Load Slopes
    if not slopes_file.exists():
        raise FileNotFoundError(f"Error: Slopes file not found: {slopes_file}")
    print(f"Loading slopes from {slopes_file}...")
    df = pd.read_csv(slopes_file)
    slopes = df["slope"].values

    # 4. Plot
    plot_spectral_results(
        profiles=profiles,
        slopes=slopes,
        mean_slope=mean_slope,
        std_slope=std_slope,
        layer_name=layer_name,
        nyquist=nyquist,
        output_dir=results_dir,  # Or output_file? The core function saves to dir.
        ylim_max=ylim_max,
    )


# plot_attack_comparison removed or replaced by generic plot_experiment_attacks
# compare_attack_results removed or replaced by generic compare_experiment_attacks


def plot_ood_type_accuracy(json_path, output_file=None, show_plot=False):
    """
    Plots the accuracy of different image types from OOD analysis results.
    Aggregates accuracy per type and plots as a bar chart.

    Args:
        json_path (str): Path to the results JSON file.
        output_file (str, optional): Path to save the plot.
        show_plot (bool, optional): Whether to show the plot interactively.
    """
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
    except Exception as e:
        raise RuntimeError(f"Error loading JSON data from {json_path}: {e}")

    predictions = data.get("predictions", [])
    if not predictions:
        raise ValueError("Error: No predictions found in JSON.")

    # Convert to DataFrame
    df = pd.DataFrame(predictions)

    # Ensure correct column is numeric
    df["correct"] = df["correct"].astype(int)

    # Aggregate by type
    if "image_type" not in df.columns:
        raise ValueError("Error: 'image_type' column missing in predictions.")

    type_accuracy = (
        df.groupby("image_type")["correct"].mean().sort_values(ascending=False)
    )

    # Prepare Plot
    base_width = 12 / 2.54
    base_height = base_width * 0.618
    figsize = (base_width, base_height)

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

    types = type_accuracy.index.tolist()
    accuracies = type_accuracy.values.tolist()

    # Colors
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    c_bar = colors[0]  # Dark Blue 1

    bars = ax.bar(types, accuracies, color=c_bar)

    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.0)
    ax.set_title(f"OOD Accuracy by Image Type (Exp {data.get('experiment_id', 'N/A')})")

    # Rotate x labels
    plt.xticks(rotation=45, ha="right")

    # Remove top/right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if output_file:
        save_plot(fig, output_file)

    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def plot_ood_comparison(json_path1, json_path2, output_file=None, show_plot=False):
    """
    Plots a paired bar chart comparing OOD accuracy by image type for two experiments.

    Args:
        json_path1 (str): Path to first result JSON.
        json_path2 (str): Path to second result JSON.
        output_file (str, optional): Path to save the plot.
        show_plot (bool, optional): Interactive display.
    """
    def load_and_agg(path):
        with open(path, "r") as f:
            data = json.load(f)
        preds = data.get("predictions", [])
        df = pd.DataFrame(preds)
        if df.empty or "image_type" not in df.columns:
            raise ValueError(f"Invalid data in {path}")
        df["correct"] = df["correct"].astype(int)
        acc = df.groupby("image_type")["correct"].mean()
        return acc, data.get("experiment_id", "Unknown")

    try:
        acc1, id1 = load_and_agg(json_path1)
        acc2, id2 = load_and_agg(json_path2)
    except Exception as e:
         raise RuntimeError(f"Error comparing OOD results: {e}")

    # Outer merge
    merged = pd.DataFrame({"Exp1": acc1, "Exp2": acc2}).fillna(0)

    # Sort by Exp1 accuracy descending
    merged = merged.sort_values("Exp1", ascending=False)

    types = merged.index.tolist()
    vals1 = merged["Exp1"].values
    vals2 = merged["Exp2"].values

    # Dimensions
    base_width = 14 / 2.54
    base_height = base_width * 0.618
    figsize = (base_width, base_height)

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

    x = np.arange(len(types))
    width = 0.35

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    c1 = colors[0]  # Dark Blue
    c2 = colors[2]  # Orange (constrast)

    rects1 = ax.bar(x - width / 2, vals1, width, label=f"Exp {id1}", color=c1)
    rects2 = ax.bar(x + width / 2, vals2, width, label=f"Exp {id2}", color=c2)

    ax.set_ylabel("Accuracy")
    ax.set_title("OOD Accuracy Comparison by Image Type")
    ax.set_xticks(x)
    ax.set_xticklabels(types, rotation=45, ha="right")
    ax.legend(frameon=False)
    ax.set_ylim(0, 1.0)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if output_file:
        save_plot(fig, output_file)

    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def plot_confidence_dist(json_path, output_file=None, show_plot=False):
    """
    Plots the distribution of confidence differences (Prediction Confidence - True Class Confidence).

    Args:
        json_path (str): Path to results JSON.
        output_file (str): Output path.
        show_plot (bool): Show plot.
    """
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
    except Exception as e:
        raise RuntimeError(f"Error loading {json_path}: {e}")

    preds = data.get("predictions", [])
    if not preds:
        return

    df = pd.DataFrame(preds)
    if "confidence_diff" not in df.columns:
        raise ValueError("Error: 'confidence_diff' not found.")

    vals = df["confidence_diff"].dropna()

    # Plot
    base_width = 8 / 2.54
    base_height = base_width * 0.75
    figsize = (base_width, base_height)

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    c1 = colors[0]

    ax.hist(vals, bins=50, color=c1, alpha=0.7, edgecolor="white", linewidth=0.5)

    ax.set_xlabel("Confidence Difference (Pred - True)")
    ax.set_ylabel("Count")
    ax.set_title(f"Confidence Difference (Exp {data.get('experiment_id', '?')})")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if output_file:
        save_plot(fig, output_file)

    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def plot_confidence_dist_comparison(
    json_path1, json_path2, output_file=None, show_plot=False
):
    """
    Plots overlapping histograms of confidence differences for two experiments.
    """
    def get_data(path):
        with open(path, "r") as f:
            data = json.load(f)
        df = pd.DataFrame(data.get("predictions", []))
        if "confidence_diff" in df.columns:
            return df["confidence_diff"].dropna(), data.get("experiment_id", "?")
        return None, None

    try:
        d1, id1 = get_data(json_path1)
        d2, id2 = get_data(json_path2)
    except Exception as e:
         raise RuntimeError(f"Error processing files: {e}")

    if d1 is None or d2 is None:
        raise ValueError("Error loading data from one or both files.")

    base_width = 10 / 2.54
    base_height = base_width * 0.75
    figsize = (base_width, base_height)

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    c1 = colors[0]
    c2 = colors[2]  # Orange

    ax.hist(d1, bins=50, alpha=0.5, label=f"Exp {id1}", color=c1, density=True)
    ax.hist(d2, bins=50, alpha=0.5, label=f"Exp {id2}", color=c2, density=True)

    ax.set_xlabel("Confidence Difference (Pred - True)")
    ax.set_ylabel("Density")
    ax.set_title(f"Confidence Diff: Exp {id1} vs Exp {id2}")
    ax.legend(frameon=False)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if output_file:
        save_plot(fig, output_file)

    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def plot_perturbation_comparison(
    json_path1,
    json_path2,
    label1="Baseline",
    label2="Flex",
    output_path=None,
    show_plot=False,
):
    """
    Plots a comparison of two perturbation analyses (ImageNet-C).
    Generates a 3x5 grid of subplots for 15 perturbation types.

    Args:
        json_path1 (str): Path to first JSON result.
        json_path2 (str): Path to second JSON result.
        label1 (str): Label for first model.
        label2 (str): Label for second model.
        output_path (str, optional): Path to save plot.
        show_plot (bool): Whether to show plot.
    """
    # 1. Load Data
    try:
        with open(json_path1, "r") as f:
            data1 = json.load(f)
        with open(json_path2, "r") as f:
            data2 = json.load(f)
    except Exception as e:
        raise RuntimeError(f"Error loading JSONs: {e}")

    # 2. Setup Figure (3x5 Grid)
    # 15 types total
    perturbation_types = sorted(list(set(list(data1.keys()) + list(data2.keys()))))

    # Dimensions
    # Based on standard plot sizing:
    # Width 185mm (18.5cm), Height approx 11.1cm to maintain ~square subplots
    figsize = (18.5 / 2.54, 11.1 / 2.54)

    fig, axes = plt.subplots(3, 5, figsize=figsize, constrained_layout=True)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    axes = axes.flatten()

    # Colors
    try:
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        c1 = colors[0]
        c2 = colors[2]  # Contrast color (Orange vs Blue)
    except:
        c1 = "b"
        c2 = "r"

    # 3. Plot Loop
    for i, p_type in enumerate(perturbation_types):
        if i >= len(axes):
            break

        ax = axes[i]

        # Plot Model 1
        if p_type in data1:
            sev = data1[p_type].get("severity", [])
            acc = data1[p_type].get("accuracies", [])
            ax.plot(
                sev,
                acc,
                marker="o",
                color=c1,
                label=label1,
                linewidth=1.5,
                markersize=4,
            )

        # Plot Model 2
        if p_type in data2:
            sev = data2[p_type].get("severity", [])
            acc = data2[p_type].get("accuracies", [])
            ax.plot(
                sev,
                acc,
                marker="s",
                color=c2,
                label=label2,
                linewidth=1.5,
                markersize=4,
                linestyle="--",
            )

        # ax.set_title(p_type.replace('_', ' ').capitalize())

        # Set Y-Lim to 0 - 0.2 as requested
        ax.set_ylim(0, 0.2)

        # Set X-Ticks to 1, 2, 3, 4, 5
        ax.set_xticks([1, 2, 3, 4, 5])

        row, col = divmod(i, 5)

        # X-Tick Labels: Only bottom row
        if row < 2:
            ax.set_xticklabels([])

        # X-Label: Only middle column of bottom row
        if row == 2 and col == 2:
            ax.set_xlabel("Severity")
        else:
            ax.set_xlabel("")

        # Y Labels and Ticks Logic
        if col == 0:
            ax.set_ylabel("Accuracy")
        else:
            # Hide Y Tick Labels for inner columns
            ax.set_yticklabels([])
            ax.set_ylabel("")  # No label

        # Remove Gridlines
        ax.grid(False)

        # Legend on rightmost subplot of first row (index 4)
        if i == 4:
            ax.legend(frameon=False, fontsize=8)

    # Hide unused axes
    for j in range(len(perturbation_types), len(axes)):
        axes[j].axis("off")

    if output_path is None:
        try:
            output_path = Path(json_path1).parent / "perturbation_comparison.png"
            print(f"Defaulting output to: {output_path}")
        except:
            pass

    if output_path:
        save_plot(fig, output_path)

    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def style_3d_axis(ax):
    """
    Apply clean 3D axis styling: remove grid, transparent panes.
    """
    ax.grid(False)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    # Set pane border thickness
    ax.xaxis.pane.set_linewidth(0.4)
    ax.yaxis.pane.set_linewidth(0.4)
    ax.zaxis.pane.set_linewidth(0.4)

    # Thinner spines
    ax.xaxis.line.set_linewidth(0.4)
    ax.yaxis.line.set_linewidth(0.4)
    ax.zaxis.line.set_linewidth(0.4)

    # Remove ticks
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])


def plot_loss_surface_comparison(
    exp1_data, exp2_data, output_path=None, show_plot=False
):
    """
    Plots a side-by-side 3D comparison of loss surfaces.

    Args:
        exp1_data (dict): Data for Exp 1 with keys: 'x', 'y', 'z', 'vals', 'exp_id'.
        exp2_data (dict): Data for Exp 2.
        output_path (str or Path): Output file path.
        show_plot (bool): Whether to display plot.
    """
    # Width 115mm. Back to 1 row.
    inch_in_mm = 25.4
    width_mm = 115
    height_mm = 65
    fig = plt.figure(figsize=(width_mm / inch_in_mm, height_mm / inch_in_mm))

    # Upsample data for smooth rendering using griddata
    target_res = 300j  # complex number for mgrid step count

    def upsample_data(data):
        # Extract points and values
        points = np.column_stack((data["x"].flatten(), data["y"].flatten()))
        values = data["z"].flatten()

        # Define new grid range
        min_x, max_x = data["x"].min(), data["x"].max()
        min_y, max_y = data["y"].min(), data["y"].max()

        # Create fine grid
        grid_x, grid_y = np.mgrid[min_x:max_x:target_res, min_y:max_y:target_res]

        # Interpolate
        grid_z = griddata(points, values, (grid_x, grid_y), method="cubic")

        new_data = data.copy()
        new_data["x"] = grid_x
        new_data["y"] = grid_y
        new_data["z"] = grid_z
        return new_data

    exp1_data = upsample_data(exp1_data)
    exp2_data = upsample_data(exp2_data)

    z1 = exp1_data["z"]
    z2 = exp2_data["z"]

    # Shared Z-limits
    z_real_min = min(z1.min(), z2.min())
    z_max = max(z1.max(), z2.max())
    z_span = z_max - z_real_min

    # "Raise" surface by lowering the floor significantly
    z_floor = z_real_min - (z_span * 2.0)

    # Helper to plot one surface
    def plot_one(ax, data):
        style_3d_axis(ax)

        # Create light source for hillshading
        ls = LightSource(azdeg=315, altdeg=45)
        rgb = ls.shade(data["z"], cmap=plt.cm.plasma, vert_exag=0.1, blend_mode="soft")

        # Surface (Almost opaque, shaded, no mesh lines)
        # facecolors=rgb applies the hillshading. shade=False because we manually shaded.
        surf = ax.plot_surface(
            data["x"],
            data["y"],
            data["z"],
            facecolors=rgb,
            edgecolor="none",
            linewidth=0,
            alpha=0.9,
            shade=False,
            antialiased=False,
            rcount=1000,
            ccount=1000,
            rasterized=True,
        )

        # Contour on the "floor" (offset)
        ax.contour(
            data["x"],
            data["y"],
            data["z"],
            levels=15,
            zdir="z",
            offset=z_floor,
            cmap="plasma",
            linewidths=0.2,
            alpha=0.8,
        )

        # ax.set_xlabel("v1", fontsize=5, labelpad=-10)
        # ax.set_ylabel("v2", fontsize=5, labelpad=-10)

        # Remove tick labels (as requested)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

        # Remove ticks lines
        ax.tick_params(axis="both", which="both", length=0, width=0)

        # Make axis lines thinner?
        # ax.xaxis.line.set_linewidth(0.1) # 3D axis handling varies, but let's try just the requested changes first.

        ax.set_zlim(z_floor, z_max)
        return surf

    # Subplot 1: Exp 1
    ax1 = fig.add_subplot(121, projection="3d")
    plot_one(ax1, exp1_data)

    # Subplot 2: Exp 2
    ax2 = fig.add_subplot(122, projection="3d")
    plot_one(ax2, exp2_data)

    plt.tight_layout()

    if output_path:
        # Custom save for high dpi rasterization
        # Set dpi=600 for high-res rasterization in SVG/PNG
        save_plot(plt, output_path, dpi=600, pad_inches=0.02)

    if show_plot:
        plt.show()
    plt.close()


def plot_hessian_comparison(
    data1, data2, label1="Baseline", label2="Flex", output_path=None, show_plot=False
):
    """
    Plots a comparison of Hessian Eigenvalue Spectra (densities).
    """
    # 70mm width (2.76 in), 4:3 ratio -> height = 2.07 in
    inch_in_mm = 25.4
    fig_ratio = 4 / 3
    width_mm = 80
    height_mm = width_mm / fig_ratio
    fig, ax = plt.subplots(figsize=(width_mm / inch_in_mm, height_mm / inch_in_mm))

    # Colors (match perturbation plot)
    try:
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        c1 = colors[0]
        c2 = colors[2]  # Contrast color (Orange vs Blue)
    except:
        c1 = "b"
        c2 = "r"

    # Calculate stats
    max1 = data1.max()
    max2 = data2.max()

    # Plot Histograms
    # Use stepped histograms for cleaner overlay, or filled with alpha
    # Bins: use shared bins to make them comparable
    combined_data = np.concatenate([data1, data2])
    bins = np.linspace(min(combined_data.min(), 0), combined_data.max(), 30)

    ax.hist(
        data1,
        bins=bins,
        density=True,
        alpha=0.9,
        label=f"{label1}",
        color=c1,
        edgecolor="black",
        linewidth=0.5,
    )
    ax.hist(
        data2,
        bins=bins,
        density=True,
        alpha=0.9,
        label=f"{label2}",
        color=c2,
        edgecolor="black",
        linewidth=0.5,
    )

    ax.set_xlabel("Eigenvalue")
    ax.set_ylabel("Density")
    ax.legend(frameon=False)

    # Clean style
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    if output_path:
        save_plot(fig, output_path)

    if show_plot:
        plt.show()
    else:
        plt.close(fig)


# compare_attack_results previously removed/consolidated.


def plot_loss_surface(
    mesh_x,
    mesh_y,
    surface_z,
    eigenvalues,
    exp_id,
    output_dir=None,
    show_plot=False,
):
    """
    Plots the 3D loss surface and 2D contour.
    """
    # Ensure output_dir is Path
    if output_dir:
        output_dir = Path(output_dir)

    # 3D Plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    style_3d_axis(ax)

    surf = ax.plot_surface(
        mesh_x, mesh_y, surface_z, cmap="plasma", edgecolor="none", alpha=0.9
    )

    ax.set_title(
        f"Loss Surface (Exp {exp_id})\nTop EVs: {eigenvalues[0]:.2f}, {eigenvalues[1]:.2f}"
    )
    ax.set_xlabel("v1")
    ax.set_ylabel("v2")
    ax.set_zlabel("Loss")
    fig.colorbar(surf)

    if output_dir:
        save_plot(fig, output_dir / "loss_surface_3d.png")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    # Contour Plot
    fig2 = plt.figure(figsize=(8, 6))
    plt.contourf(mesh_x, mesh_y, surface_z, levels=20, cmap="plasma")
    plt.colorbar(label="Loss")
    plt.title(f"Loss Contour (Exp {exp_id})")
    plt.xlabel("v1")
    plt.ylabel("v2")

    if output_dir:
        save_plot(fig2, output_dir / "loss_surface_contour.png")

    if show_plot:
        plt.show()
    else:
        plt.close(fig2)



def _map_vgg_layer_label(layer_name):
    """
    Maps 'features.X' to 'Layer {Block}.{Index}' for VGG16.
    """
    if not isinstance(layer_name, str) or not layer_name.startswith("features."):
        return layer_name
    
    try:
        idx = int(layer_name.split(".")[1])
    except ValueError:
        return layer_name

    # VGG16 Mapping (Conv layers only) - simplified to ID only
    mapping = {
        0: "1.1", 3: "1.2",
        7: "2.1", 10: "2.2",
        14: "3.1", 17: "3.2", 20: "3.3",
        24: "4.1", 27: "4.2", 30: "4.3",
        34: "5.1", 37: "5.2", 40: "5.3"
    }
    
    return mapping.get(idx, layer_name)


def plot_brain_score_comparison(
    results_files,
    output_file=None,
    show_plot=False,
    title=None,
    xlim=None,
    ylim=None,
    scientific_notation=False,
    figsize=None,
):
    """
    Plots a 2x2 comparison of Brain-Score benchmarks.

    Args:
        results_files (dict): Dictionary mapping Experiment Name -> JSON File Path.
        output_file (str, optional): Path to save the plot.
        show_plot (bool, optional): Whether to show the plot.
        title (str, optional): Title override.
        xlim (tuple, optional): X-axis limits.
        ylim (tuple, optional): Y-axis limits.
        scientific_notation (bool, optional): Formatting toggle.
        figsize (tuple, optional): Figure size.
    """
    # 1. Load Data
    data_records = []
    
    for exp_name, filepath in results_files.items():
        if not Path(filepath).exists():
            print(f"Warning: File {filepath} not found.")
            continue
            
        with open(filepath, "r") as f:
            try:
                content = json.load(f)
            except json.JSONDecodeError:
                print(f"Error decoding {filepath}")
                continue

        for exp_key, layers_data in content.items():
            for layer, benchmarks in layers_data.items():
                layer_int = -1
                if layer.startswith("features."):
                    try:
                        layer_int = int(layer.split(".")[1])
                    except:
                        pass
                
                label = _map_vgg_layer_label(layer)
                
                for bench_id, value in benchmarks.items():
                    # Handle if value is dict (old format) or float (new format)
                    score = value
                    if isinstance(value, dict) and "score" in value:
                        score = value["score"]
                    
                    data_records.append({
                        "Experiment": exp_name,
                        "LayerOriginal": layer,
                        "LayerLabel": label,
                        "LayerInt": layer_int,
                        "Benchmark": bench_id,
                        "Score": score
                    })

    if not data_records:
        return

    df = pd.DataFrame(data_records)
    df = df.sort_values(by="LayerInt")

    # 2. Setup Plot Style
    # Benchmarks of interest: V1, V2, V4, IT
    target_benchmarks = [
        "FreemanZiemba2013.V1.public-pls",
        "FreemanZiemba2013.V2.public-pls",
        "MajajHong2015.public.V4-pls",
        "MajajHong2015.public.IT-pls"
    ]
    
    readable_titles = {
        "FreemanZiemba2013.V1.public-pls": "V1",
        "FreemanZiemba2013.V2.public-pls": "V2",
        "MajajHong2015.public.V4-pls": "V4",
        "MajajHong2015.public.IT-pls": "IT"
    }

    # Style: Dimensions 186mm width
    if figsize is None:
        # Default to 2/3 width of standard full figure (186mm)
        total_width_mm = 186 * (2/3)
        total_width_inch = total_width_mm / 25.4
        # Height: golden ratio or appropriate. 
        # Height for 2 rows approx.
        total_height_inch = total_width_inch * 0.8
        figsize = (total_width_inch, total_height_inch)

    with plotting_style():
        fig, axes = plt.subplots(2, 2, figsize=figsize, constrained_layout=True)
        axes = axes.flatten()
        
        # Colors from style guide
        # Primary: Dark Blue (#14213D), Secondary: Orange (#FCA311)
        colors = [COLORS[0], COLORS[2]] # Index 0 and 2 based on previous code reading (check `COLORS` definition in file)
        
        # Calculate global Y-limits if not provided
        if ylim is None:
            max_score = df["Score"].max()
            # Add 10% padding
            ylim = (0, max_score * 1.1)

        for i, bench_id in enumerate(target_benchmarks):
            ax = axes[i]
            
            bench_data = df[df["Benchmark"] == bench_id]
            
            if bench_data.empty:
                ax.text(0.5, 0.5, "No Data", ha='center')
                ax.set_title(readable_titles.get(bench_id, bench_id))
                continue
            
            # Use manual plotting for control or seaborn with manual palette
            experiments = bench_data["Experiment"].unique()
            experiments.sort() # Ensure consistent order
            
            for j, exp in enumerate(experiments):
                exp_df = bench_data[bench_data["Experiment"] == exp]
                
                # Assign color: Exp2 -> Primary, Exp4 -> Secondary (just heuristic or order based)
                c = colors[j % len(colors)]
                marker = 'o' if j == 0 else 's'
                linestyle = '-' if j == 0 else '--'
                
                ax.plot(
                    exp_df["LayerLabel"],
                    exp_df["Score"],
                    marker=marker,
                    linestyle=linestyle,
                    color=c,
                    linewidth=1.5,
                    label=exp,
                    markersize=4
                )
            
            # Style Guide: No titles by default, but subplots usually need distinction.
            # "Titles: No titles by default. Titles should be optional and enabled via a flag."
            # However, for subplots, we usually need to know which is V1/V2. 
            # I will use the readable title but keep font size spec (10pt).
            if title is not None:
                # If global title provided, maybe suppress subplot titles?
                # Or usually 'title' arg refers to Figure title or main Axes title.
                # Here we have 4 subplots. I'll put benchmark name as ax title 
                # because otherwise it's unreadable.
                ax.set_title(readable_titles.get(bench_id, bench_id))
            else:
                 # Implicitly use benchmark name as we need to distinguish them
                 ax.set_title(readable_titles.get(bench_id, bench_id))

            if i % 2 == 0:
                ax.set_ylabel("Correlation")
            else:
                ax.set_ylabel("")
                # Hide Y-tick labels for non-left column
                plt.setp(ax.get_yticklabels(), visible=False)

            # Set X-label for bottom plots (indices 2 and 3)
            # Or all? User requested "x-axis label 'Layer ID'". Usually done on bottom row.
            if i >= 2:
                ax.set_xlabel("Layer ID")
            else:
                ax.set_xlabel("")
            
            # Always show x-tick labels (user request)
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

            # Grid: Disabled by default per style guide.
            # "Grid: Disabled by default"
            ax.grid(False)

            # Spines: Top/Right removed (handled by plotting_style/get_style_params usually, ensuring here)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            # Limits
            if ylim:
                ax.set_ylim(ylim)
            else:
                ax.set_ylim(bottom=0)
                
            if xlim:
                ax.set_xlim(xlim)

            # Legend: No outline. Only on last plot or first?
            # Let's put it on the first one or top-right one.
            if i == 1: # Top Right
                ax.legend(frameon=False)
    
        if output_file:
            save_plot(fig, output_file)

        if show_plot:
            plt.show()
        else:
            plt.close(fig)
