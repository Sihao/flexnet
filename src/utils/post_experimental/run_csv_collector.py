"""
gather results from the runs folder after training and analysis is done for making linear regression plots / other analysis

- takes all information
- save it to a single csv file in a defined folder
- all the images that share a common name will be saved in the same folder with their names changed to the run_name
"""

from pathlib import Path
import json, pandas, shutil
from src.analysis.run_batch_processor import RunBatchProcessor


class RunCsvCollector(RunBatchProcessor):
    def __init__(self, runs_folder: Path):
        super().__init__(runs_folder)
        """attributes: run_names, runs_folder"""

        # filter the runs without a runsult folder with a warning
        runs_no_results = [
            run
            for run in self.run_names
            if not (self.runs_folder / run / "results").exists()
        ]

        if len(runs_no_results) > 0:
            print(f"Warning: the following runs have no results folder:")
            print(runs_no_results)

        self.run_names = [
            run
            for run in self.run_names
            if (self.runs_folder / run / "results").exists()
        ]

    def add_external_labels(self, labels: dict):
        self.external_labels = labels

    def main(self, save_folder: Path):
        self._gather_images(save_folder)
        self._gather_jsons(save_folder)

    # ------- [ gatherers ] -------
    def _gather_images(self, save_folder: Path):
        save_folder.mkdir(parents=True, exist_ok=True)
        unique_images = set()

        for run_name in self.run_names:
            results_folder = self.runs_folder / run_name / "results"
            assert (
                results_folder.exists()
            ), f"results folder does not exist for run {run_name}"

            images = list(results_folder.glob("*.png"))
            assert len(images) > 0, f"no images found in {results_folder}"

            for image in images:
                unique_images.add(image.name.strip(".png"))

        for name in unique_images:
            (save_folder / name).mkdir(parents=True, exist_ok=True)

        for run_name in self.run_names:
            results_folder = self.runs_folder / run_name / "results"
            images = list(results_folder.glob("*.png"))
            for image in images:
                shutil.copy(
                    str(image),
                    str(save_folder / image.name.strip(".png") / (run_name + ".png")),
                )

        return self

    def _gather_jsons(self, save_folder: Path):
        """get all jsons and turn into a dataframe"""
        save_folder.mkdir(parents=True, exist_ok=True)
        dataframe = []

        # ---- results jsons ----
        for run_name in self.run_names:
            results_folder = self.runs_folder / run_name / "results"
            assert (
                results_folder.exists()
            ), f"results folder does not exist for run {run_name}"

            jsons = list(results_folder.rglob("*.json"))
            assert len(jsons) > 0, f"no jsons found in {results_folder}"

            run_results = {}

            for json_path in jsons:
                with open(json_path, "r") as loader:
                    json_content = json.load(loader)

                for key, value in json_content.items():
                    if not isinstance(value, list):
                        run_results.update({key: value})

            if hasattr(self, "external_labels"):
                if run_name in self.external_labels.keys():
                    run_results["Manual Label"] = self.external_labels[run_name]

            # get the configurations.json
            configurations_path = self.runs_folder / run_name / "configurations.json"
            with open(configurations_path, "r") as loader:
                configurations = json.load(loader)
            run_results.update(configurations)

            dataframe.append(run_results)

        # ---- save to csv ----
        dataframe = pandas.DataFrame(dataframe)

        if "Manual Label" in dataframe.columns:  # move to front
            dataframe = dataframe[
                ["Manual Label"]
                + [col for col in dataframe.columns if col != "Manual Label"]
            ]

        dataframe.to_csv(save_folder / "results.csv", index=False)
        return self
