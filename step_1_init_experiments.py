#!/Users/dy323/micromamba/envs/flex/bin/python
"""
the main script for grid search initialiser

- the user set the hyperparameters and define the experiment dir
- inside of the experiment dir, n sub directories will be created for each combination
- each sub directory will contain the hyperparameters in its configurations.json file

    [note: use bash renaming all the config.json to configurations.json as well as in the code]
"""

import yaml
from pathlib import Path
from src.utils.hyperparams.grid import GridSearch
from src.utils.server import is_on_server


if __name__ == "__main__":
    i = 1
    while True:
        experiment_name = f"experiment-{i}"

        with open("configurations.yml", "r") as f:
            params = yaml.safe_load(f)

        if is_on_server():
            base_check = Path(params["system"]["experiment_dir"]["server"])
        else:
            base_check = Path(params["system"]["experiment_dir"]["local"])

        if not (base_check / experiment_name).exists():
            break
        i += 1

    if is_on_server():
        base = Path(params["system"]["experiment_dir"]["server"])
    else:
        base = Path(params["system"]["experiment_dir"]["local"])

    experiment_dir = base / experiment_name
    # Separate system config from grid search params
    system_config = params.pop("system", None)

    # 3. initialize the grid search
    init = GridSearch(dir_experiments=experiment_dir, params=params)
    init.init_directories()
