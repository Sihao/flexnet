#!/Users/dy323/micromamba/envs/flex/bin/python

"""
- this script takes a runs folder where all the semi-trained models are stored and continues training them for x epochs
- example: python _run_2_continue_train.py    run_name = "000000"
"""

from src.train_initialiser import continue_training
from src.analysis.run_loader import RunLoader
from src.utils.server import is_on_server
from pathlib import Path
import argparse

parser = argparse.ArgumentParser(description="Continue Train Handler")
parser.add_argument(
    "--run_name",
    type=str,
    default="",
    required=False,
    help="The run name to be continued",
)
parser.add_argument(
    "--experiment_name",
    type=str,
    default=None,
    help="The experiment name (e.g. experiment-2)",
)


class ContinueTrainSingleHandler:
    def __init__(self, run_loader: RunLoader):
        self.run_loader = run_loader

    def main(self):
        continue_training(self.run_loader, epochs=200, logs_per_epoch=10)


if __name__ == "__main__":
    # ---- [output of step 1] ----
    import re
    import yaml

    with open("configurations.yml", "r") as f:
        config = yaml.safe_load(f)

    if is_on_server():
        base = Path(config["system"]["experiment_dir"]["server"])
    else:
        base = Path(config["system"]["experiment_dir"]["local"])

    # Find latest experiment
    args = parser.parse_args()

    # Find latest experiment if not specified
    if args.experiment_name:
        experiment_name = args.experiment_name
    else:
        max_i = 0
        experiment_name = "experiment-1"
        for path in base.iterdir():
            if path.is_dir() and path.name.startswith("experiment-"):
                try:
                    i = int(path.name.split("-")[1])
                    if i > max_i:
                        max_i = i
                        experiment_name = path.name
                except ValueError:
                    continue

    if is_on_server():
        base_folder = (
            Path(config["system"]["experiment_dir"]["server"]) / experiment_name
        )
        # ---- [just run the appointed one] ----
        actions = vars(args)
        run_loader = RunLoader(base_folder / actions["run_name"])
        single_run_analyser = ContinueTrainSingleHandler(run_loader=run_loader)
        single_run_analyser.main()

    else:
        base_folder = (
            Path(config["system"]["experiment_dir"]["local"]) / experiment_name
        )
        # ---- [ run all sequentially ] ----
        for run_folder in (item for item in base_folder.iterdir() if item.is_dir()):
            run_loader = RunLoader(run_folder)
            single_run_analyser = ContinueTrainSingleHandler(run_loader=run_loader)
            single_run_analyser.main()
