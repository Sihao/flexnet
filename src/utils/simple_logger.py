import pandas as pd
from pathlib import Path
import json


class SimpleLogger:
    def __init__(self, filename):
        self.filename = Path(filename)
        self.data = []
        self.filename.parent.mkdir(parents=True, exist_ok=True)

    def log(self, entry: dict):
        self.data.append(entry)
        # Auto-save every log or so (inefficient but safe) or just keep in memory?
        # Let's save to csv incrementally if possible, or just append to list.
        # For simplicity, let's just append to list and have a save method.
        # But wait, original code expected `log` to persist?
        # We can append to a json lines file.
        with open(self.filename.with_suffix(".jsonl"), "a") as f:
            f.write(json.dumps(entry) + "\n")

    def add_metric_entry(self, entry: dict):
        # Alias for log if needed, but we will revert train.py to use log()
        self.log(entry)

    def show_last_row(self):
        if self.data:
            print(self.data[-1])

    def on_end(self):
        pass

    def get_dataframe(self):
        # Read from jsonl
        data = []
        if self.filename.with_suffix(".jsonl").exists():
            with open(self.filename.with_suffix(".jsonl"), "r") as f:
                for line in f:
                    data.append(json.loads(line))
        return pd.DataFrame(data)

    def save_as_pandas_dataframe(self, save_dir):
        df = self.get_dataframe()
        df.to_csv(save_dir, index=False)
