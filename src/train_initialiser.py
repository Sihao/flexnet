"""
The main script for training the image model in a classification tasks
"""

from src.training.train import main_training_loop
from src.training.dataset_select import get_dataset_obj
from src.training.dataset_subset import create_random_subset
from src.analysis.run_loader import RunLoader


def get_num_batches_per_log(dataset, batch_size, logs_per_epoch):
    """
    Calculate the number of batches between each log.

    Args:
        dataset: The dataset to be used for training.
        batch_size (int): The number of samples per batch.
        logs_per_epoch (int): The desired number of logs per epoch.

    Returns:
        int: The number of batches between each log, minimum of 1.
    """
    num_batches = len(dataset) // batch_size
    num_batches_per_log = num_batches // logs_per_epoch
    num_batches_per_log = max(1, num_batches_per_log)
    return num_batches_per_log


def continue_training(
    run_loader: RunLoader, epochs: int = 50, logs_per_epoch: int = 36
):
    """
    dev mode: use a small subset of the dataset for faster debugging
    """
    # ---------------- print init  ----------------
    print("Configurations")
    print(run_loader.config)

    # -------- setting up the dataset/dataloader ---------
    from src.training.dataset_select import get_dataloader, get_dataset_obj

    use_ffcv = run_loader.config.get("use_ffcv", False)
    dataset_train = None
    dataset_valid = None
    dataloader_train = None
    dataloader_valid = None

    if use_ffcv:
        print("[bold green]Using FFCV Data Loading[/bold green]")
        dataloader_train = get_dataloader(
            run_loader.config["dataset"],
            "TRAIN",
            run_loader.config["batch_size"],
            run_loader.config,
        )
        dataloader_valid = get_dataloader(
            run_loader.config["dataset"],
            "TEST",
            run_loader.config["batch_size"],
            run_loader.config,
        )

        # Calculate log frequency for FFCV
        # FFCV loader has length (number of batches)
        num_batches = len(dataloader_train)
        num_batches_per_log = max(1, num_batches // logs_per_epoch)
        log_every_n_batch = num_batches_per_log

    else:
        dataset_train = get_dataset_obj(run_loader.config["dataset"], "TRAIN")
        dataset_valid = get_dataset_obj(run_loader.config["dataset"], "TEST")

        # -------- check if datasets are empty ---------
        if len(dataset_train) == 0:
            print(f"[bold red]training dataset is empty[/bold red]")
            # ... error messages ...
            raise ValueError(
                f"training dataset '{run_loader.config['dataset']}' is empty - contact Don"
            )

        # -------- compute log frequency --------
        log_every_n_batch = get_num_batches_per_log(
            dataset_train, run_loader.config["batch_size"], logs_per_epoch
        )

    # ---------------- setting up the weighted loss function ----------------
    main_training_loop(
        epochs=epochs,
        run_loader=run_loader,
        dataset_train=dataset_train,
        dataset_valid=dataset_valid,
        log_every_n_batch=log_every_n_batch,
        dataloader_train=dataloader_train,
        dataloader_valid=dataloader_valid,
    )


if __name__ == "__main__":
    """
    only use grid search handler to run this; avoid switch between scripts
    e.g., to run with first combination:

    python -m search run_with 0
    """
    dataset_train = get_dataset_obj("cifar10", "TRAIN")
    num = get_num_batches_per_log(dataset_train, batch_size=320, logs_per_epoch=36)
    print(len(dataset_train))
    print(num)
