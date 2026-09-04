import argparse
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
import tifffile
import numpy as np

# Project imports
from src.modules.models.flex_unet import FlexUNet
from src.training.dataset_select import get_dataset_obj


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


class MockConfig(dict):
    """
    Mock config to behave like the loaded json config used in existing utils
    """

    def __getattr__(self, key):
        if key in self:
            return self[key]
        # Return None or raise error? Existing code might define specific behavior
        # reusing logic from run_flex_unet_cpu.py which raises Error
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{key}'")


def train(args):
    print(f"Starting training with Tif: {args.tiff_path}")
    ensure_dir(args.output_dir)

    device = torch.device(
        args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    )
    print(f"Using device: {device}")

    # Build Config Dictionary
    config_dict = {
        "network": "FlexUNet",
        "dataset": "deepinterpolation",
        "tiff_file_path": args.tiff_path,
        "frame_window_N": args.frame_window,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "optimizer": "ADAMW",
        "weight_decay": 1e-4,
        # Default Flex params (can be exposed if needed)
        "joint_mechanism": "CHANNELWISE_MAXPOOL",
        "logits_mechanism": "THRESHOLD",
        "masking_mechanism": "SIGMOID_HARD",
        "logits_use_batchnorm": False,
        "sigmoid_mul_factor": 1.0,
        "vanilla": args.vanilla,
    }

    # Save config for reproducibility
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(config_dict, f, indent=4)

    # Prepare Config Object
    config = MockConfig(config_dict)

    # Initialize Model
    print("Initializing FlexUNet...")
    model = FlexUNet(config).to(device)

    # Dataset and DataLoader
    print("Loading Dataset...")
    # get_dataset_obj expects a config object or dict.
    # Based on run_flex_unet_cpu, passing the config wrapper works best if code uses attributes
    dataset = get_dataset_obj("deepinterpolation", "TRAIN", config=config)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=0
    )  # num_workers=0 to avoid complications for now

    # Optimizer & Loss
    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=config.weight_decay
    )
    criterion = nn.MSELoss()

    # Resume logic
    start_epoch = 0
    if args.resume_checkpoint:
        print(f"Resuming from checkpoint: {args.resume_checkpoint}")
        checkpoint = torch.load(args.resume_checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        # If continuing, we assume we want to run for `epochs` MORE epochs,
        # or we could jump to the next epoch number.
        # User request: "continue ... for 10 more epochs".
        # So we should probably treat `args.epochs` as "additional epochs".
        # But for logging clarity, let's update the epoch counter.
        previous_epoch = checkpoint.get("epoch", 0)
        start_epoch = previous_epoch + 1
        print(
            f"Resumed model. Last epoch was {previous_epoch}. Starting at {start_epoch}."
        )

    # Training Loop
    print("Starting Training Loop...")
    model.train()

    # Run for `args.epochs` iterations, regardless of start_epoch
    # The epoch index in the loop will be adjusted for logging/saving

    for i in range(args.epochs):
        epoch = start_epoch + i
        epoch_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")

        for inputs, targets in pbar:
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1} Completed. Avg Loss: {avg_loss:.6f}")

        # Save Checkpoint
        checkpoint_path = os.path.join(
            args.output_dir, f"checkpoint_epoch_{epoch+1}.pth"
        )
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_loss,
                "config": config_dict,
            },
            checkpoint_path,
        )

    print("Training Completed.")


def inference(args):
    print(f"Starting inference on: {args.tiff_path}")
    ensure_dir(os.path.dirname(args.output_path))

    device = torch.device(
        args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    )
    print(f"Using device: {device}")

    # Load Checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    config_dict = checkpoint.get("config", {})

    # Override N if provided, else use config
    frame_window = (
        args.frame_window
        if args.frame_window is not None
        else config_dict.get("frame_window_N", 30)
    )
    config_dict["frame_window_N"] = frame_window
    config = MockConfig(config_dict)

    # Initialize Model
    model = FlexUNet(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Load Full Tiff for Inference
    print("Loading input tiff...")
    stack = tifffile.imread(args.tiff_path)
    # Ensure (T, H, W)
    if len(stack.shape) != 3:
        raise ValueError(f"Expected 3D Tiff (T, H, W), got {stack.shape}")

    T, H, W = stack.shape
    print(f"Input shape: {stack.shape}")

    # Check bounds
    if T < 2 * frame_window + 1:
        print("Video too short for the requested frame window.")
        return

    # Output storage
    # We will only valid frames: [N : T-N]
    # Or should we output full size with zeros? Stick to valid frames for now or pad.
    # DeepInterpolation usually denoises the center.
    # Let's create an output of size (T, H, W) and fill the valid range.
    output_stack = np.zeros_like(stack, dtype=np.float32)

    # Indices to predict
    start_idx = frame_window
    end_idx = T - frame_window

    print(f"Predicting frames {start_idx} to {end_idx}...")

    with torch.no_grad():
        for i in tqdm(range(start_idx, end_idx)):
            # Prepare Input
            # Frames [i-N : i] and [i+1 : i+N+1]
            pre = stack[i - frame_window : i]
            post = stack[i + 1 : i + frame_window + 1]

            # Combine
            inp = np.concatenate([pre, post], axis=0)  # (2N, H, W)
            inp_tensor = (
                torch.from_numpy(inp).unsqueeze(0).float().to(device)
            )  # (1, 2N, H, W)

            # Predict
            out = model(inp_tensor)  # (1, 1, H, W)

            output_stack[i] = out.squeeze().cpu().numpy()

    # Save - Trim the padding frames
    trimmed_output = output_stack[start_idx:end_idx]
    print(f"Trimming output: {output_stack.shape} -> {trimmed_output.shape}")
    print(f"Saving output to {args.output_path}...")
    tifffile.imwrite(args.output_path, trimmed_output.astype(np.float32))
    print("Done.")


def main():
    parser = argparse.ArgumentParser(description="FlexUNet DeepInterpolation CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Train Parser
    train_parser = subparsers.add_parser("train", help="Train FlexUNet")
    train_parser.add_argument(
        "--tiff_path", type=str, required=True, help="Path to input tiff file"
    )
    train_parser.add_argument(
        "--output_dir", type=str, required=True, help="Directory to save checkpoints"
    )
    train_parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    train_parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    train_parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    train_parser.add_argument(
        "--frame_window", type=int, default=5, help="Frame window size N (context 2N)"
    )
    train_parser.add_argument(
        "--resume_checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    train_parser.add_argument(
        "--vanilla",
        action="store_true",
        help="Use Vanilla UNet (Standard Convolutions)",
    )
    train_parser.add_argument(
        "--device", type=str, default="cuda", help="Device (cpu or cuda)"
    )

    # Inference Parser
    infer_parser = subparsers.add_parser("inference", help="Run Inference")
    infer_parser.add_argument(
        "--tiff_path", type=str, required=True, help="Path to input tiff"
    )
    infer_parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint"
    )
    infer_parser.add_argument(
        "--output_path", type=str, required=True, help="Path to save output tiff"
    )
    infer_parser.add_argument(
        "--frame_window", type=int, default=None, help="Override frame window size"
    )
    infer_parser.add_argument(
        "--device", type=str, default="cuda", help="Device (cpu or cuda)"
    )

    args = parser.parse_args()

    if args.command == "train":
        train(args)
    elif args.command == "inference":
        inference(args)


if __name__ == "__main__":
    main()
