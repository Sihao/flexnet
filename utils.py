import pandas as pd


def process_data(data, window=1):
    """
    Converts list of dicts to DataFrame, aggregates by Epoch, and applies smoothing.
    Returns:
        epochs (Index): The epoch numbers
        means (DataFrame): Smoothed means of metrics
        stds (DataFrame): Smoothed standard deviations of metrics
    """
    if not data:
        raise ValueError("No data provided to process")

    df = pd.DataFrame(data)

    required_cols = [
        "Epoch",
        "Train Accuracy",
        "Valid Accuracy",
        "Train Loss",
        "Valid Loss",
    ]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in log data: {missing_cols}")

    # Aggregate by Epoch
    grouped = df.groupby("Epoch")[
        ["Train Accuracy", "Valid Accuracy", "Train Loss", "Valid Loss"]
    ].agg(["mean", "std"])
    grouped = grouped.fillna(0)

    means = grouped.xs("mean", axis=1, level=1)
    stds = grouped.xs("std", axis=1, level=1)

    if window > 1:
        print(f"Applying rolling window of {window} epochs...")
        means = means.rolling(window=window, min_periods=1).mean()
        stds = stds.rolling(window=window, min_periods=1).mean()

    return grouped.index, means, stds


def process_conv_ratio(data, layer_num=None, window=1):
    """
    Extracts and processes Conv Ratio.
    If layer_num is None, extracts ALL columns starting with 'Conv Ratio '.
    """
    if not data:
        raise ValueError("No data provided to process")

    df = pd.DataFrame(data)

    if layer_num is not None:
        target_cols = [f"Conv Ratio {layer_num}"]
    else:
        # Find all Conv Ratio columns
        target_cols = [c for c in df.columns if c.startswith("Conv Ratio ")]
        # Sort them numerically
        try:
            target_cols.sort(key=lambda x: int(x.split(" ")[-1]))
        except:
            target_cols.sort()

    if not target_cols:
        raise ValueError("No 'Conv Ratio' columns found in log data.")

    missing_cols = [c for c in target_cols if c not in df.columns]
    if missing_cols:
        # Only relevant if specific layer requested
        if layer_num is not None:
            raise ValueError(f"Column '{missing_cols[0]}' not found in log data.")

    # Aggregate by Epoch
    grouped = df.groupby("Epoch")[target_cols].agg(["mean", "std"])
    grouped = grouped.fillna(0)

    means = grouped.xs("mean", axis=1, level=1)
    stds = grouped.xs("std", axis=1, level=1)

    if window > 1:
        means = means.rolling(window=window, min_periods=1).mean()
        stds = stds.rolling(window=window, min_periods=1).mean()

    return grouped.index, means, stds


def ensure_dir(path):
    """
    Ensures that the directory exists.
    """
    import os

    os.makedirs(path, exist_ok=True)
