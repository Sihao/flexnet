#!/Users/donyin/miniconda3/envs/imperial/bin/python

"""
when the training and analysis are done, we have a folder structure like this:
----------------------------------------------------------------------------------------------------------------
experiments:
    run_1:
        configurations.json
        results:
            results file 1
            results file 2
            ...
        ...
    run_2:
        ...
----------------------------------------------------------------------------------------------------------------
this script serves as a base to batch process all the runs
"""

from pathlib import Path


class RunBatchProcessor:
    def __init__(self, runs_folder: Path):
        assert runs_folder.exists(), f"{runs_folder} does not exist"
        self.runs_folder = runs_folder
        self.run_names = [run_name.name for run_name in self.runs_folder.glob("*")]
        self.run_names = [
            run_name for run_name in self.run_names if "." not in run_name
        ]
        self.run_names = sorted(self.run_names)


if __name__ == "__main__":
    pass
