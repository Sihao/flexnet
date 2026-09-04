import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import json
import sys


def plot_learning_curve(log_path):
    log_path = Path(log_path)
    if not log_path.exists():
        print(f"Log file not found: {log_path}")
        return

    data = []
    with open(log_path, "r") as f:
        for line in f:
            data.append(json.loads(line))

    df = pd.DataFrame(data)

    if df.empty:
        print("Log file is empty.")
        return

    # Create plot
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Loss Plot
    if "Train Loss" in df.columns:
        axes[0].plot(df["Epoch"], df["Train Loss"], label="Train Loss")
    if "Valid Loss" in df.columns:
        axes[0].plot(df["Epoch"], df["Valid Loss"], label="Valid Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Learning Curve: Loss")
    axes[0].legend()
    axes[0].grid(True)

    # Accuracy Plot
    if "Train Accuracy" in df.columns:
        axes[1].plot(df["Epoch"], df["Train Accuracy"], label="Train Accuracy")
    if "Valid Accuracy" in df.columns:
        axes[1].plot(df["Epoch"], df["Valid Accuracy"], label="Valid Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Learning Curve: Accuracy")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()

    # Save plot with name derived from log file to be clear
    log_name = log_path.stem  # metrics
    save_path = log_path.parent / f"{log_name}_learning_curve.png"
    plt.savefig(save_path)
    print(f"Saved plot to {save_path}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        plot_learning_curve(sys.argv[1])
    else:
        print("Usage: python plot_utils.py <path_to_metrics.jsonl>")
