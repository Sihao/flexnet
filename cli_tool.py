import argparse
import sys
import data_io
import utils
import plotting


def parse_args():
    parser = argparse.ArgumentParser(description="Plot experiment training progress.")
    parser.add_argument(
        "-e",
        "--experiment",
        type=int,
        required=False,
        help="Experiment ID (e.g., 6, 9)",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=1,
        help="Number of epochs to average over (rolling window)",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Enable interactive mode (show plots).",
    )

    subparsers = parser.add_subparsers(
        dest="command", help="Plotting command", required=True
    )

    # Subparser for progress plot
    parser_progress = subparsers.add_parser(
        "plot_progress", help="Plot training and validation accuracy/loss."
    )

    # Subparser for conv ratio plot

    # Subparser for visualization
    parser_viz = subparsers.add_parser("visualize", help="Visualize layer activations.")
    parser_viz.add_argument(
        "--layer",
        default=None,
        help="Layer index (int) or name (str). Required unless --all-conv-layers is set.",
    )
    parser_viz.add_argument(
        "--all-conv-layers",
        action="store_true",
        help="Visualize ALL Conv2d layers in the model.",
    )
    parser_viz.add_argument(
        "--image",
        default="data/imagenet100/val.X/n01632777/ILSVRC2012_val_00034583.JPEG",
        help="Path to input image.",
    )
    parser_viz.add_argument(
        "--output", default=None, help="Output filename (optional)."
    )

    # Subparser for filter visualization
    parser_filters = subparsers.add_parser(
        "visualize_filters", help="Visualize layer filters (weights)."
    )
    parser_filters.add_argument(
        "--layer", required=True, help="Layer index (int) or name (str)."
    )
    parser_filters.add_argument(
        "--output", default=None, help="Output filename (optional)."
    )

    # Subparser for attack analysis
    parser_attacks = subparsers.add_parser(
        "run_attacks", help="Run adversarial attack comparison."
    )
    parser_attacks.add_argument(
        "--batch-size", type=int, default=16, help="Batch size for validation loader."
    )
    parser_attacks.add_argument(
        "--attacks",
        nargs="+",
        default=None,
        help="List of attacks to run (e.g. FGSM PGD). Default: ALL.",
    )
    parser_attacks.add_argument(
        "--viz-filter",
        default=None,
        help="String to filter visualization examples (e.g. class ID or filename).",
    )
    parser_attacks.add_argument(
        "--continue",
        dest="continue_run",
        action="store_true",
        help="Continue from existing results (skip computed epsilons).",
    )

    # Subparser for plotting attack results
    parser_plot_attack = subparsers.add_parser(
        "plot_attack", help="Plot adversarial attack performance."
    )
    parser_plot_attack.add_argument(
        "--attack", required=True, help="Attack name (e.g. FGSM)."
    )

    parser_plot_attack.add_argument("--output", default=None, help="Output filename.")

    # Subparser for frequency analysis
    parser_freq = subparsers.add_parser(
        "run_frequency", help="Analyze frequency bias of layer activations."
    )
    parser_freq.add_argument(
        "--num-images",
        type=int,
        default=None,
        help="Number of images to use (default: All).",
    )

    # Subparser for Spectral Slope Analysis
    parser_spectral = subparsers.add_parser(
        "run_spectral_slope", help="Run spectral slope analysis on a specific layer."
    )
    parser_spectral.add_argument(
        "--layer", type=int, required=True, help="Index of Conv/Pool layer to analyze."
    )
    parser_spectral.add_argument(
        "--num-images", type=int, default=128, help="Number of images to process."
    )
    parser_spectral.add_argument(
        "--batch-size", type=int, default=100, help="Batch size (default: 100)."
    )
    parser_spectral.add_argument(
        "--device", type=str, default=None, help="Device (cpu/cuda)."
    )
    parser_spectral.add_argument(
        "--ylim-max", type=int, default=None, help="Y-axis limit for slope histogram."
    )

    # Subparser for Plotting Spectral Slope
    parser_plot_spectral = subparsers.add_parser(
        "plot_spectral_slope", help="Plot spectral slope results from disk."
    )
    parser_plot_spectral.add_argument(
        "--layer",
        type=str,
        required=True,
        help="Layer Name (e.g. features.7) or Index (e.g. 7).",
    )
    parser_plot_spectral.add_argument(
        "--ylim-max", type=int, default=None, help="Y-axis limit for slope histogram."
    )

    # Subparser for OOD Analysis
    parser_ood = subparsers.add_parser("run_ood", help="Run OOD Analysis (ImageNet-R).")
    parser_ood.add_argument("--batch-size", type=int, default=32, help="Batch size.")
    parser_ood.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of images per class (for debugging/quick check).",
    )
    parser_ood.add_argument(
        "--device", type=str, default=None, help="Device (cpu/cuda)."
    )

    # Subparser for Hessian Analysis
    parser_hessian = subparsers.add_parser(
        "run_hessian", help="Run Hessian Eigenvalue Analysis (Input Space)."
    )
    parser_hessian.add_argument(
        "--batch-size", type=int, default=1, help="Batch size for Hessian estimation."
    )
    parser_hessian.add_argument(
        "--batches", type=int, default=1, help="Number of batches to accumulate."
    )
    parser_hessian.add_argument(
        "--m-steps", type=int, default=50, help="Number of Lanczos steps."
    )
    parser_hessian.add_argument(
        "--device", type=str, default="cpu", help="Device to run on (cpu/cuda)."
    )

    # Compare Hessian Spectra
    parser_comp_hessian = subparsers.add_parser(
        "compare_hessian", help="Compare Hessian eigenvalue spectra."
    )
    parser_comp_hessian.add_argument("--exp1", required=True)
    parser_comp_hessian.add_argument("--exp2", required=True)
    parser_comp_hessian.add_argument("--output", default=None)

    # Subparser for plotting ALL attacks comparison
    parser_plot_compare = subparsers.add_parser(
        "plot_attack_comparison", help="Plot comparison of all attacks."
    )
    parser_plot_compare.add_argument("--output", default=None, help="Output filename.")

    # Subparser for generating attack examples
    parser_gen_examples = subparsers.add_parser(
        "generate_attack_examples",
        help="Generate adversarial examples for all attacks.",
    )
    parser_gen_examples.add_argument(
        "--image",
        default="data/imagenet100/val.X/n01632777/ILSVRC2012_val_00034583.JPEG",
        help="Path to input image.",
    )
    parser_gen_examples.add_argument(
        "--output-dir", default=None, help="Output directory."
    )
    parser_gen_examples.add_argument(
        "--attacks", nargs="+", default=None, help="Specific attacks to run."
    )

    # Subparser for ImageNet-C Perturbation Analysis
    parser_perturb = subparsers.add_parser(
        "run_perturbation_analysis", help="Run ImageNet-C perturbation analysis."
    )
    parser_perturb.add_argument(
        "--imagenet-c-path",
        default="/mnt/bronknas/Sihao/FlexNet/Datasets/ImageNet-C/",
        help="Path to ImageNet-C dataset.",
    )
    parser_perturb.add_argument(
        "--batch-size", type=int, default=32, help="Batch size."
    )
    parser_perturb.add_argument(
        "--device",
        default=None,
        help="Device to use (e.g., 'cuda:0'). Defaults to GPU if available.",
    )

    # Subparser for Comparing Perturbations
    parser_compare_perturb = subparsers.add_parser(
        "compare_perturbations",
        help="Compare perturbation results between two experiments.",
    )
    parser_compare_perturb.add_argument("--exp1", required=True, help="Experiment ID 1")
    parser_compare_perturb.add_argument("--exp2", required=True, help="Experiment ID 2")
    parser_compare_perturb.add_argument(
        "--output", default=None, help="Output image path (optional)."
    )
    parser_compare_perturb.add_argument(
        "--labels",
        nargs=2,
        default=None,
        help="Custom labels for the legend (e.g. 'ModelA' 'ModelB')",
    )

    # Subparser for Comparing Attacks
    parser_comp_att = subparsers.add_parser(
        "compare_attacks", help="Compare attack results between two experiments."
    )
    parser_comp_att.add_argument("--exp1", required=True, help="Experiment ID 1")
    parser_comp_att.add_argument("--exp2", required=True, help="Experiment ID 2")
    parser_comp_att.add_argument("--output", default=None, help="Output filename.")
    parser_comp_att.add_argument(
        "--labels", nargs=2, default=None, help="Legend labels."
    )

    # Input Loss Surface Analysis
    parser_loss_surf = subparsers.add_parser(
        "run_loss_surface", help="Run Input Loss Surface Analysis."
    )
    parser_loss_surf.add_argument(
        "--grid-points", type=int, default=51, help="Grid resolution."
    )
    parser_loss_surf.add_argument(
        "--range", type=float, default=10.0, help="Range scale."
    )
    parser_loss_surf.add_argument("--device", type=str, default="cpu")

    # Compare Loss Surfaces
    parser_comp_surf = subparsers.add_parser(
        "compare_loss_surfaces", help="Compare loss surfaces input space."
    )
    parser_comp_surf.add_argument("--exp1", required=True)
    parser_comp_surf.add_argument("--exp2", required=True)
    parser_comp_surf.add_argument("--output", default=None)

    return parser.parse_args()


def get_model_for_experiment(exp_id):
    import torch
    from src.analysis.run_loader import RunLoader

    # Determine path
    if isinstance(exp_id, int) or (isinstance(exp_id, str) and str(exp_id).isdigit()):
        exp_path = f"__local__/experiment-{exp_id}/000000"
    else:
        exp_path = exp_id

    print(f"Loading model for Experiment {exp_id} from {exp_path} using RunLoader...")

    # RunLoader handles config loading and model instantiation
    loader = RunLoader(exp_path)

    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = loader.model.to(device)
    model.eval()

    return model


def main():
    args = parse_args()

    if args.interactive:
        import matplotlib

        try:
            matplotlib.use("TkAgg")
        except ImportError:
            print("Warning: TkAgg backend not available. Using default.")

    # Handle visualization separately as it needs model loading, not just metrics log
    if args.command == "visualize":
        try:
            if not args.layer and not args.all_conv_layers:
                print("Error: You must specify either --layer or --all-conv-layers")
                return

            model = get_model_for_experiment(args.experiment)
            import torch

            layers_to_viz = []
            if args.all_conv_layers:
                print("Identifying all Conv2d layers...")
                for name, module in model.named_modules():
                    if isinstance(module, torch.nn.Conv2d):
                        layers_to_viz.append(name)
                print(f"Found {len(layers_to_viz)} Conv2d layers: {layers_to_viz}")
            else:
                layers_to_viz.append(args.layer)

            for layer_name in layers_to_viz:
                print(f"Visualizing layer: {layer_name}")

                # Dynamic output filename handling
                output_file = args.output
                results_dir = (
                    f"__local__/experiment-{args.experiment}/000000/results/plots"
                )
                utils.ensure_dir(results_dir)

                # If doing batch or if output not specified, generate default name
                if args.all_conv_layers or output_file is None:
                    safe_layer_name = str(layer_name).replace(".", "_")
                    output_file_path = f"{results_dir}/activation_viz_exp{args.experiment}_layer{safe_layer_name}.png"
                else:
                    # Check if it has directory, if not, prepend default
                    if "/" not in output_file:
                        output_file_path = f"{results_dir}/{output_file}"
                    else:
                        output_file_path = output_file

                plotting.visualize_activations(
                    model=model,
                    layer_name=layer_name,
                    image_path=args.image,
                    output_file=output_file_path,
                    show_plot=args.interactive,
                )

        except Exception as e:
            print(f"Error during visualization: {e}")
            import traceback

            traceback.print_exc()
        return  # Exit after visualization

    if args.command == "visualize_filters":
        try:
            model = get_model_for_experiment(args.experiment)

            output_file = args.output
            output_file = args.output
            if output_file is None:
                results_dir = (
                    f"__local__/experiment-{args.experiment}/000000/results/plots"
                )
                utils.ensure_dir(results_dir)
                output_file = f"{results_dir}/filter_viz_exp{args.experiment}_layer{args.layer}.png"
            else:
                if "/" not in output_file:
                    results_dir = (
                        f"__local__/experiment-{args.experiment}/000000/results/plots"
                    )
                    utils.ensure_dir(results_dir)
                    output_file = f"{results_dir}/{output_file}"

            plotting.visualize_filters(
                model=model,
                layer_name=args.layer,
                output_file=output_file,
                show_plot=args.interactive,
            )
        except Exception as e:
            print(f"Error during filter visualization: {e}")
            import traceback

            traceback.print_exc()
        return  # Exit after visualization

    if args.command == "run_attacks":
        try:
            from src.analysis.run_attack_comparison import run_attack_comparison

            print(f"Running attacks for Experiment {args.experiment}...")
            run_attack_comparison(
                experiment_id=args.experiment,
                batch_size=args.batch_size,
                attacks=args.attacks,
                viz_filter=args.viz_filter,
                resume=args.continue_run,
            )
        except Exception as e:
            print(f"Error during attack analysis: {e}")
            import traceback

            traceback.print_exc()
        return  # Exit after analysis

    if args.command == "plot_attack":
        try:
            # Construct path to JSON
            # __local__/experiment-13/000000/results/attack_comparison/FGSM/FGSM_model_accuracies_top_1.json
            json_path = f"__local__/experiment-{args.experiment}/000000/results/attack_comparison/{args.attack}/{args.attack}_model_accuracies_top_1.json"

            output_file = args.output
            if output_file is None:
                output_file = f"{args.attack}_performance_exp{args.experiment}.png"
                # Save to the attack directory by default if not specified?
                # Or just cwd. Let's do cwd for simplicity unless user specifies path.
                # Actually user requirement implied checking specific path in verification plan.
                # Let's save to the attack directory to be clean.
                output_file = f"__local__/experiment-{args.experiment}/000000/results/attack_comparison/{args.attack}/{args.attack}_performance.png"

            plotting.plot_attack_performance(
                json_path=json_path,
                attack_name=args.attack,
                experiment_id=args.experiment,
                output_file=output_file,
                show_plot=args.interactive,
            )
        except Exception as e:
            print(f"Error during attack plotting: {e}")
            import traceback

            traceback.print_exc()
        return

    if args.command == "run_frequency":
        try:
            from src.analysis.run_frequency_analysis import run_frequency_analysis

            print(f"Running frequency analysis for Experiment {args.experiment}...")
            run_frequency_analysis(
                experiment_id=args.experiment,
                batch_size=args.batch_size,
                num_images=args.num_images,
            )
        except Exception as e:
            print(f"Error during frequency analysis: {e}")
            import traceback

            traceback.print_exc()
        return

    if args.command == "run_spectral_slope":
        try:
            from src.analysis.spectral import analyze_layer_spectral_slope

            print(
                f"Running spectral slope analysis for Experiment {args.experiment}, Layer {args.layer}..."
            )
            # Unpack 2 return values (paths dict, layer_name)
            paths, layer_name = analyze_layer_spectral_slope(
                experiment_id=args.experiment,
                layer_idx=args.layer,
                num_images=args.num_images,
                batch_size=args.batch_size,
                device=args.device,
                dataset_name="imagenet100",
                split="TRAIN",
                ylim_max=args.ylim_max,
            )

            output_dir = paths["output_dir"]

            # Now plot
            print(f"Plotting results for {layer_name} from {output_dir}...")
            plotting.plot_spectral_results_from_disk(
                paths=paths,
                # output_dir not needed as paths has it, but function signature changed?
                # Let's check plotting.plot_spectral_results_from_disk signature
                # def plot_spectral_results_from_disk(paths, output_file=None, ylim_max=None):
                ylim_max=args.ylim_max,
            )

        except Exception as e:
            print(f"Error during spectral slope analysis: {e}")
            import traceback

            traceback.print_exc()
        return

    if args.command == "plot_spectral_slope":
        try:
            from pathlib import Path

            # 1. Determine base search directory
            if isinstance(args.experiment, int) or (
                isinstance(args.experiment, str) and args.experiment.isdigit()
            ):
                exp_results_dir = Path(
                    f"__local__/experiment-{args.experiment}/000000/results/spectral_analysis"
                )
            else:
                exp_results_dir = (
                    Path(args.experiment) / "results" / "spectral_analysis"
                )

            # 2. Find layer subdirectory
            layer_dir = exp_results_dir / args.layer
            if not layer_dir.exists():
                print(f"Error: Layer directory not found: {layer_dir}")
                # Try searching for layer name match?
                # For now assume exact match or user provides exact folder name
                return

            # 3. Find latest timestamped directory
            timestamps = sorted(
                [d for d in layer_dir.iterdir() if d.is_dir()],
                key=lambda x: x.name,
                reverse=True,
            )
            if not timestamps:
                print(f"Error: No timestamp subdirectories found in {layer_dir}")
                return

            latest_dir = timestamps[0]
            print(f"Found timestamp directories. Using latest: {latest_dir}")

            # 4. Construct paths dict
            # We need to find the files inside latest_dir.
            # They have variable names based on sample count.
            # Use glob to find them.

            # Metadata
            meta_files = list(latest_dir.glob(f"metadata_{args.layer}_*samples.json"))
            if not meta_files:
                print(f"Error: metadata file not found in {latest_dir}")
                return
            metadata_file = meta_files[0]

            # Extract sample count/suffix to find others
            # metadata_layer.name_100samples.json
            suffix = metadata_file.name.replace(f"metadata_{args.layer}_", "")

            spectra_file = (
                latest_dir / f"spectra_{args.layer}_{suffix.replace('.json', '.npy')}"
            )
            slopes_file = (
                latest_dir / f"slopes_{args.layer}_{suffix.replace('.json', '.csv')}"
            )

            paths = {
                "metadata": metadata_file,
                "spectra": spectra_file,
                "slopes": slopes_file,
                "output_dir": latest_dir,
            }

            plotting.plot_spectral_results_from_disk(
                paths=paths, ylim_max=args.ylim_max
            )
        except Exception as e:
            print(f"Error during spectral plotting: {e}")
            import traceback

            traceback.print_exc()
        return

    if args.command == "plot_attack_comparison":
        try:
            print(f"Plotting attack comparison for Experiment {args.experiment}...")

            # Smart loading logic
            from pathlib import Path
            import json

            if isinstance(args.experiment, int) or (
                isinstance(args.experiment, str) and args.experiment.isdigit()
            ):
                base_dir = Path(
                    f"__local__/experiment-{args.experiment}/000000/results/attack_comparison"
                )
            else:
                base_dir = Path(args.experiment) / "results" / "attack_comparison"

            # 1. Search for timestamped JSONs (New Format)
            found_new_format = False
            if base_dir.exists():
                # Look for subdirs
                subdirs = sorted(
                    [d for d in base_dir.iterdir() if d.is_dir()],
                    key=lambda x: x.name,
                    reverse=True,
                )

                for d in subdirs:
                    json_path = d / "attack_comparison_results.json"
                    if json_path.exists():
                        print(f"Found new format results in {d}")
                        with open(json_path, "r") as f:
                            results = json.load(f)

                        # Use dict-based plotting function
                        # Note: plotting.plot_attack_comparison expects (results, exp_id, ...)
                        # But wait, there are TWO functions with similar names in plotting.py?
                        # 1. plot_attack_performance(json_path...) - Single attack
                        # 2. plot_all_attacks_comparison(experiment_id...) - Legacy, reads from disk
                        # AND... I need to check if there is a monolithic plotter in plotting.py.
                        # Looking at plotting.py:
                        # "def plot_attack_comparison(results, experiment_id, output_path=None, show_plot=False):"
                        # Yes, it exists (lines 909+).

                        output_path = args.output
                        if output_path is None:
                            output_path = d / "attack_comparison_plot.png"

                        plotting.plot_attack_comparison(
                            results=results,
                            experiment_id=args.experiment,
                            output_path=output_path,
                            show_plot=args.interactive,
                        )
                        found_new_format = True
                        break

            if not found_new_format:
                print(
                    "New format results not found. Attempting legacy folder structure..."
                )
                plotting.plot_all_attacks_comparison(
                    experiment_id=args.experiment,
                    output_path=args.output,
                    show_plot=args.interactive,
                )

        except Exception as e:
            print(f"Error during comparison plotting: {e}")
            import traceback

            traceback.print_exc()
        return

    if args.command == "run_hessian":
        try:
            from src.analysis.run_hessian_analysis import analyze_hessian_input

            print(f"Running Hessian analysis for Experiment {args.experiment}...")

            analyze_hessian_input(
                exp_id=args.experiment,
                batch_size=args.batch_size,
                num_batches=args.batches,
                m_steps=args.m_steps,
                device=args.device,
            )
        except Exception as e:
            print(f"Error during Hessian analysis: {e}")
            import traceback

            traceback.print_exc()
        return

    if args.command == "compare_hessian":
        try:
            from src.analysis.run_hessian_analysis import compare_hessian_spectra

            compare_hessian_spectra(
                args.exp1, args.exp2, args.output, show_plot=args.interactive
            )
        except Exception as e:
            print(f"Error during Hessian comparison: {e}")
            import traceback

            traceback.print_exc()
        return

    if args.command == "run_ood":
        try:
            from src.analysis.run_ood_analysis import run_ood_analysis

            print(f"Running OOD Analysis for Experiment {args.experiment}...")
            run_ood_analysis(
                experiment_id=args.experiment,
                batch_size=args.batch_size,
                limit=args.limit,
                device=args.device,
            )
        except Exception as e:
            print(f"Error during OOD analysis: {e}")
            import traceback

            traceback.print_exc()
        return

    if args.command == "generate_attack_examples":
        try:
            from src.analysis.run_attack_comparison import generate_attack_examples

            print(f"Generating attack examples for Experiment {args.experiment}...")
            generate_attack_examples(
                experiment_id=args.experiment,
                image_path=args.image,
                output_dir=args.output_dir,
                attacks=args.attacks,
            )
        except Exception as e:
            print(f"Error during example generation: {e}")
            import traceback

            traceback.print_exc()
        return

    if args.command == "run_perturbation_analysis":
        try:
            from src.analysis.run_perturbation_analysis import run_perturbation_analysis

            print(f"Running Perturbation Analysis for Experiment {args.experiment}...")
            run_perturbation_analysis(
                experiment_id=args.experiment,
                imagenet_c_path=args.imagenet_c_path,
                batch_size=args.batch_size,
                device_str=args.device,
            )
        except Exception as e:
            print(f"Error during perturbation analysis: {e}")
            import traceback

            traceback.print_exc()
        return

    if args.command == "compare_perturbations":
        from plotting import plot_perturbation_comparison

        # Infer Paths
        # Assuming standard structure: __local__/experiment-{ID}/000000/results/perturbation_analysis/perturbation_analysis_results.json
        def get_json_path(eid):
            if isinstance(eid, int) or (isinstance(eid, str) and eid.isdigit()):
                return f"__local__/experiment-{eid}/000000/results/perturbation_analysis/perturbation_analysis_results.json"
            return eid  # If full path provided

        path1 = get_json_path(args.exp1)
        path2 = get_json_path(args.exp2)

        # Infer output path if not provided: Defer to plotting function logic
        # if args.output is None: ... (Removed to allow plotting.py to handle defaults)

        labels = (
            args.labels if args.labels else [f"Exp {args.exp1}", f"Exp {args.exp2}"]
        )

        print(f"Comparing Perturbations: {path1} vs {path2}")
        print(f"Labels: {labels}")
        print(f"Output: {args.output}")

        try:
            plot_perturbation_comparison(
                path1,
                path2,
                label1=labels[0],
                label2=labels[1],
                output_path=args.output,
            )
        except Exception as e:
            print(f"Error plotting comparison: {e}")
            import traceback

            traceback.print_exc()
        return

    if args.command == "compare_attacks":
        try:
            from pathlib import Path
            import json

            def get_latest_results(exp_id):
                if isinstance(exp_id, int) or (
                    isinstance(exp_id, str) and exp_id.isdigit()
                ):
                    base = Path(
                        f"__local__/experiment-{exp_id}/000000/results/attack_comparison"
                    )
                else:
                    base = Path(exp_id) / "results" / "attack_comparison"

                if not base.exists():
                    return None

                subdirs = sorted(
                    [d for d in base.iterdir() if d.is_dir()],
                    key=lambda x: x.name,
                    reverse=True,
                )
                for d in subdirs:
                    if (d / "attack_comparison_results.json").exists():
                        return d / "attack_comparison_results.json"
                return None

            path1 = get_latest_results(args.exp1)
            path2 = get_latest_results(args.exp2)

            if not path1:
                print(f"Error: No results found for Exp {args.exp1}")
                return
            if not path2:
                print(f"Error: No results found for Exp {args.exp2}")
                return

            print(f"Comparing: {path1} vs {path2}")

            with open(path1, "r") as f:
                r1 = json.load(f)
            with open(path2, "r") as f:
                r2 = json.load(f)

            labels = (
                args.labels if args.labels else [f"Exp {args.exp1}", f"Exp {args.exp2}"]
            )

            output_file = args.output
            if output_file is None:
                # Save in first experiment's directory for now, or a shared location?
                # Or just local dir. Let's save to exp1's folder.
                output_file = path1.parent / f"attack_comparison_vs_exp{args.exp2}.png"

            plotting.compare_attack_results(
                r1,
                r2,
                label1=labels[0],
                label2=labels[1],
                output_path=output_file,
                show_plot=args.interactive,
            )

        except Exception as e:
            print(f"Error during attack comparison: {e}")
            import traceback

            traceback.print_exc()
        return

    if args.command == "run_loss_surface":
        try:
            from src.analysis.run_loss_surface import run_loss_surface_analysis

            print(f"Running Loss Surface Analysis for Experiment {args.experiment}...")
            run_loss_surface_analysis(
                exp_id=args.experiment,
                grid_points=args.grid_points,
                range_scale=args.range,
                device=args.device,
                show_plot=args.interactive,
            )
        except Exception as e:
            print(f"Error during loss surface analysis: {e}")

            import traceback

            traceback.print_exc()
        return

    if args.command == "compare_loss_surfaces":
        try:
            from src.analysis.run_loss_surface import compare_loss_surfaces

            compare_loss_surfaces(
                args.exp1, args.exp2, args.output, show_plot=args.interactive
            )
        except Exception as e:
            print(f"Error during comparison: {e}")
            import traceback

            traceback.print_exc()
        return

    # Logic for log-based plots
    log_path = f"__local__/experiment-{args.experiment}/000000/logs/metrics.jsonl"

    try:
        print(f"Loading metrics from {log_path}...")
        data = data_io.load_metrics(log_path)

        if args.command == "plot_conv_ratio":
            if args.layer is not None:
                print(f"Processing Conv Ratio for Layer {args.layer}...")
                epochs, means, stds = utils.process_conv_ratio(
                    data, args.layer, args.window
                )
                epochs, means, stds = utils.process_conv_ratio(
                    data, args.layer, args.window
                )

                results_dir = (
                    f"__local__/experiment-{args.experiment}/000000/results/plots"
                )
                utils.ensure_dir(results_dir)
                output_file = f"{results_dir}/experiment_{args.experiment}_layer_{args.layer}_conv_ratio.png"

                plotting.plot_conv_ratios(
                    epochs,
                    means,
                    stds,
                    args.layer,
                    args.experiment,
                    args.window,
                    output_file,
                    show_plot=args.interactive,
                )

            elif args.all_layers:
                print(f"Processing Conv Ratio for ALL layers...")
                epochs, means, stds = utils.process_conv_ratio(
                    data, layer_num=None, window=args.window
                )
                epochs, means, stds = utils.process_conv_ratio(
                    data, layer_num=None, window=args.window
                )

                results_dir = (
                    f"__local__/experiment-{args.experiment}/000000/results/plots"
                )
                utils.ensure_dir(results_dir)
                output_file = f"{results_dir}/experiment_{args.experiment}_all_layers_conv_ratio.png"

                plotting.plot_conv_ratios(
                    epochs,
                    means,
                    stds,
                    None,
                    args.experiment,
                    args.window,
                    output_file,
                    show_plot=args.interactive,
                )
            else:
                print(
                    "Error: For 'plot_conv_ratio', please specify either --layer <N> or --all-layers"
                )
                sys.exit(1)

        elif args.command == "plot_progress":
            print("Processing data...")
            epochs, means, stds = utils.process_data(data, args.window)

            epochs, means, stds = utils.process_data(data, args.window)

            results_dir = f"__local__/experiment-{args.experiment}/000000/results/plots"
            utils.ensure_dir(results_dir)
            output_file = f"{results_dir}/experiment_{args.experiment}_progress.png"

            print(f"Generating plot...")
            plotting.plot_training_progress(
                epochs,
                means,
                stds,
                args.experiment,
                args.window,
                output_file,
                show_plot=args.interactive,
            )

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
