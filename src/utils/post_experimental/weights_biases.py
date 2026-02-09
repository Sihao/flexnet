from time import sleep

from tqdm import tqdm
import wandb, json, os
from pathlib import Path
from natsort import natsorted
from src.analysis.run_loader import RunLoader
from metricdb import MetricDB
import pandas


def process_one_folder(
    run_loader: RunLoader, project_name: str, upload_initial_n_epochs: int = None
):
    # https://docs.wandb.ai/ref/python/init
    run = wandb.init(
        project=project_name,
        config=run_loader.config,
        tags="",
        name=run_loader.run_folder.name,
        id=run_loader.run_folder.name,
        resume=True,
    )

    # ---- [ identify log files ] ----
    run_loader.logger.save_as_pandas_dataframe(save_dir="logs.csv")

    # read the pandas dataframe
    dataframe = pandas.read_csv("logs.csv")

    # ---- [ log each row ] ----
    for index, row in tqdm(dataframe.iterrows(), total=len(dataframe)):
        epoch = row["Epoch"]
        if upload_initial_n_epochs is not None and epoch > upload_initial_n_epochs:
            break
        run.log(row.to_dict())

    wandb.finish()


if __name__ == "__main__":
    loader = RunLoader(
        "/Users/donyin/Dropbox/root-dir/flex-layer/runs/experiment/000000",
        whether_load_checkpoint=False,
        whether_instantiate_model=False,
    )
    process_one_folder(loader)
