"""
This module provides functionality for experimental parameter management and grid search.

    > it creates a directory structure with all possible parameter combinations
    > it creates a settings.json file with the parameter definitions
    > it creates a settings.md file with a markdown table of the parameter definitions

The module contains the GridSearch class which handles parameter combinations for
experimental configurations. It supports dependency management between parameters
and eliminates duplicate configurations.

Classes:
    GridSearch: Manages parameter combinations for experimental configurations with
               dependency handling and deduplication.

"""

import json
from pathlib import Path

from graphlib import TopologicalSorter


class GridSearch:
    def __init__(self, dir_experiments: Path, params: dict = None):
        self.dir_experiments = Path(dir_experiments)
        self.params = params or {}

        self._topological_sort_params()
        self._generate_combinations()
        self._remove_duplications()

    def _get_possible_values(self, param_name, current_combination):
        param_info = self.params[param_name]
        depends_on = param_info.get("depends_on", None)

        if depends_on is None:
            return param_info["values"]
        else:
            dependencies_met = True
            for dep_param, dep_values in depends_on.items():
                dep_value_in_combination = current_combination.get(dep_param)

                if dep_value_in_combination is None:
                    dependencies_met = False
                    break
                else:
                    if isinstance(dep_values, list):
                        if dep_value_in_combination not in dep_values:
                            dependencies_met = False
                            break
                    else:
                        if dep_value_in_combination != dep_values:
                            dependencies_met = False
                            break

            if dependencies_met:
                return param_info["values"]
            else:
                return [None]

    def _topological_sort_params(self):
        graph = {}
        for param_name, param_info in self.params.items():
            depends_on = param_info.get("depends_on", {})
            if depends_on:
                # collect all dependent parameter names
                dep_params = set()
                for dep_param in depends_on.keys():
                    if dep_param not in self.params:
                        raise ValueError(
                            f"Undefined dependency: '{dep_param}' in parameter '{param_name}'"
                        )
                    dep_params.add(dep_param)
                graph[param_name] = dep_params
            else:
                graph.setdefault(param_name, set())

        ts = TopologicalSorter(graph)
        sorted_params = list(ts.static_order())
        return sorted_params

    def _generate_combinations(self):
        self.combinations = []
        param_names = self._topological_sort_params()
        combinations = [{}]

        for param_name in param_names:
            new_combinations = []
            for combination in combinations:
                possible_values = self._get_possible_values(param_name, combination)
                for value in possible_values:
                    new_combination = combination.copy()
                    new_combination[param_name] = value
                    new_combinations.append(new_combination)
            combinations = new_combinations

        self.combinations = combinations

    def _remove_duplications(self):
        unique_combinations = []
        seen = set()

        for combo in self.combinations:
            items = tuple(sorted(combo.items()))
            if items not in seen:
                seen.add(items)
                unique_combinations.append(combo)

        self.combinations = unique_combinations

    def init_directories(self):
        assert not self.dir_experiments.exists(), "The runs directory already exists"
        self.dir_experiments.mkdir(exist_ok=True, parents=True)

        for i, config in enumerate(self.combinations):
            folder_path = self.dir_experiments / f"{i:06d}"
            folder_path.mkdir(exist_ok=True)
            with open(folder_path / "configurations.json", "w") as f:
                json.dump(config, f, indent=4)

        settings_path = self.dir_experiments / "settings.json"

        with open(settings_path, "w") as f:
            json.dump(self.params, f)

        # ---- create markdown table of settings ----
        settings_md = self.dir_experiments / "settings.md"
        with open(settings_md, "w") as f:
            f.write("# Experiment Settings\n\n")
            f.write("| Parameter | Values | Dependencies |\n")
            f.write("|-----------|---------|-------------|\n")
            for param, details in self.params.items():
                values = str(details["values"]).replace("|", "\\|").replace("'", "")
                depends = (
                    str(details["depends_on"]).replace("|", "\\|").replace("'", "")
                    if details["depends_on"]
                    else "None"
                )
                f.write(f"| {param} | {values} | {depends} |\n")

        print(
            f"> created {len(self.combinations)} directories in {self.dir_experiments}"
        )


if __name__ == "__main__":
    experiment_dir = Path("experiments")

    params = {
        # -------- meta --------
        "network": {
            "values": ["VGG"],  # SimpleConvNet / SimpleFlexNet / VGG
            "depends_on": None,
        },
        "vgg_variant": {
            "values": ["16"],  # 16 / 11 / 6
            "depends_on": {"network": ["VGG"]},
        },
        "fully_flex": {
            "values": [
                True
            ],  # True / False; replace even the max pooling layers with flex
            "depends_on": {"network": ["VGG"], "use_flex": True},
        },
        "learning_rate": {
            "values": [1e-3],  # 1e-3 / 1e-4 / 5e-4
            "depends_on": None,
        },
        "dropout": {
            "values": [0.2],  # 0.2 / 0.5
            "depends_on": {"network": ["VGG"]},
        },
        "batch_size": {
            "values": [16],  # 80 / 320; 16, 64, 256
            "depends_on": None,
        },
        "validate_every_n_batch": {
            "values": [16],  # 16, 64, 256
            "depends_on": None,
        },
        "dataset": {
            "values": [
                "cifar10"
            ],  # cifar10 / imagenet100 / cifar10-down-50 / cifar10-random-small-100
            "depends_on": None,
        },
        "optimizer": {
            "values": ["ADAM"],  # "ADAM" / "SGD"
            "depends_on": None,
        },
        # -------- vgg --------
        "use_flex": {
            "values": [True, False],  # True / False
            "depends_on": {"network": ["VGG"]},
        },
        # -------- flex --------
        "joint_mechanism": {
            "values": [False, "CHANNELWISE_MAXPOOL"],  # "CHANNELWISE_MAXPOOL" / "None"
            "depends_on": {"use_flex": True},
        },
        "logits_mechanism": {
            "values": ["THRESHOLD"],  # SpatialAttentionBlock
            "depends_on": {"use_flex": True, "joint_mechanism": False},
        },
        "masking_mechanism": {
            "values": [
                "SIGMOID_MUL",
                "SIGMOID_HARD",
                "STE_FSBS",
                "StochasticRoundFSBS",
            ],
            "depends_on": {"use_flex": True, "joint_mechanism": False},
        },
        "sigmoid_mul_factor": {
            "values": [1, 50],
            "depends_on": {"masking_mechanism": "SIGMOID_MUL"},
        },
        "logits_use_batchnorm": {
            "values": [False],  # makes no massive difference / flex has no batchnorm
            "depends_on": {"use_flex": True, "joint_mechanism": False},
        },
        "num_spatial_attention_block": {
            "values": [1],
            "depends_on": {"logits_mechanism": "SpatialAttentionBlock"},
        },
        # -------- simple conv --------
        "custom_activation": {  # for simple conv
            "values": ["SharpScaledSigmoid", "NoneScaledSigmoid", "STE_FSBS"],
            "depends_on": {"network": "SimpleConvNet"},
        },
    }

    init = GridSearch(dir_experiments=experiment_dir, params=params)
    init.init_directories()
