"""
checking / searching and fixing the runs after training / testing is done
print out the results in terminal
"""

import json, shutil, os

from pathlib import Path
from collections import Counter
from src.analysis.run_batch_processor import RunBatchProcessor


# -------- [ step 1 ] --------
class RunChecker(RunBatchProcessor):
    def __init__(self, runs_folder: Path):
        super().__init__(runs_folder)
        """attributes: run_names, runs_folder"""

    # ======= [ search ] =======
    def search_runs(self, criteria: dict):
        """
        use some criteria to search for run names
        --------
        criteria = { "use_flex": True, "logits_mechanism": "THRESHOLD" }
        """
        # -------- [ searching ] --------
        fit_runs = []
        for run_name in self.run_names:
            run_config = self._read_config(run_name)
            if criteria.keys() <= run_config.keys():
                if all([run_config[key] == criteria[key] for key in criteria.keys()]):
                    fit_runs.append(run_name)
        return fit_runs

    # ======= [ fixers ] =======
    def move_results(self):
        """fix nested results folder"""
        for run_name in self.run_names:
            outer_results_path = self.runs_folder / run_name / "results"
            inner_results_path = outer_results_path / "results"
            if outer_results_path.exists() and inner_results_path.exists():
                print(f"- Moving results folder for {run_name}")
                for file in inner_results_path.glob("*"):
                    destination_file = outer_results_path / file.name
                    if destination_file.exists():
                        destination_file.unlink()  # Remove the file if it already exists
                    shutil.move(str(file), str(outer_results_path))
                inner_results_path.rmdir()

        return self

    # ======= [ checkers ] =======
    def check_missing_results(self):
        run_folders = [d for d in self.runs_folder.iterdir() if d.is_dir()]
        needs_rerun = []

        if not run_folders:
            print("No run folders found in the experiment directory.")
            return

        # Get the set of all files and directories in the first run folder's results directory
        reference_results = set()
        for root, dirs, files in os.walk(run_folders[0] / "results"):
            for name in dirs:
                reference_results.add(
                    Path(root).relative_to(run_folders[0] / "results") / name
                )
            for name in files:
                reference_results.add(
                    Path(root).relative_to(run_folders[0] / "results") / name
                )

        # Check each run folder against the reference
        for run_folder in run_folders:
            current_results = set()
            for root, dirs, files in os.walk(run_folder / "results"):
                for name in dirs:
                    current_results.add(
                        Path(root).relative_to(run_folder / "results") / name
                    )
                for name in files:
                    current_results.add(
                        Path(root).relative_to(run_folder / "results") / name
                    )

            missing_items = reference_results - current_results
            if missing_items:
                print(f"Missing items in {run_folder.name}: {missing_items}")
                needs_rerun.append(run_folder.name)

        # -------- [ save the runs that need rerun ] --------
        if len(needs_rerun):
            with open("registry_flex_analysis", "w") as f:
                f.write("\n".join(needs_rerun))
            print(f"- Total number of runs that need rerun: {len(needs_rerun)}")
            print(f"- Saved in registry_flex_analysis")
        else:
            print("- All runs have the desired result files")

    def check_runs_fits_planned(self):
        """check the runs in the experiment folder fits with that in the run_search.py script"""
        # -------- [ check the num of folders in the experiment folder ] --------
        print(f"Number of folders in the experiment folder: {len(self.run_names)}")

        # -------- [ check num of runs (existing) ] --------
        configs = [self._read_config(name) for name in self.run_names]
        unique_configs = []
        for config in configs:
            if config not in unique_configs:
                unique_configs.append(config)
        print(f"- Total number of unique runs: {len(unique_configs)} / {len(configs)}")

        # -------- [ check all runs are getting a pth file that looks like the rest of the runs ] --------
        run_checkpoints = [
            self._find_checkpoint(run_name) for run_name in self.run_names
        ]
        if any([i is None for i in run_checkpoints]):
            indices = [i for i, x in enumerate(run_checkpoints) if x is None]
            [print(f"- Warning: {len(indices)} runs are missing checkpoints")]
            [print(f"- Missing checkpoint: {self.run_names[i]}") for i in indices]

        else:
            count = Counter([i.name for i in run_checkpoints])
            count = {
                k: v
                for k, v in sorted(
                    count.items(), key=lambda item: item[1], reverse=True
                )
            }
            print(f"- Checkpoint files: {count}")

            majority = count.keys().__iter__().__next__()

            if any([i.name != majority for i in run_checkpoints]):
                indices = [
                    i for i, x in enumerate(run_checkpoints) if x.name != majority
                ]
                [
                    print(
                        f"- Warning: {len(indices)} runs have different checkpoint files"
                    )
                ]
                [print(f"- Checkpoint file: {self.run_names[i]}") for i in indices]

        return self

    # -------- [ helpers ] --------
    def _read_config(self, run_name: str):
        config_file = self.runs_folder / run_name / "configurations.json"
        assert config_file.exists(), f"{config_file} does not exist"
        with open(config_file, "r") as f:
            config = json.load(f)
        return config

    def _find_checkpoint(self, run_name: str):
        checkpoint_file = (self.runs_folder / run_name).rglob("*.pth")
        checkpoint_file = list(checkpoint_file)
        if not len(checkpoint_file):
            return None
        return checkpoint_file[0]

    def _collect_exist_runs_config(self):
        runs = [self._read_config(run_name) for run_name in self.run_names]
        with open("_detected_runs.json", "w") as f:
            json.dump(runs, f)

    @staticmethod
    def _compare_difference(json_1, json_2):
        "compare two json files and return the differences"

        with open(json_1, "r") as f:
            data_1 = json.load(f)
        with open(json_2, "r") as f:
            data_2 = json.load(f)

        for item in data_1:
            item.pop("annealing_factor", None)

        for item in data_2:
            item.pop("annealing_factor", None)

        differences = []
        for i in data_1:
            if i not in data_2:
                differences.append(i)

        for i in data_2:
            if i not in data_1 and i not in differences:
                differences.append(i)
        return differences


if __name__ == "__main__":
    pass
