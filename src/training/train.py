import numpy as np
from torch import clamp
import torch, warnings, random
from torch.utils.data import DataLoader
from sklearn.metrics import balanced_accuracy_score
from src.utils.device import check_cuda_memory_usage
from src.training.historical_methods.anneal import get_annealing_factor, compute_tau
from src.utils.device import select_device

warnings.filterwarnings("ignore")


def get_accuracy(logits, labels):
    _, predicted = torch.max(logits, 1)
    accuracy = (predicted == labels).float().mean().item()
    return accuracy


def get_balanced_accuracy(logits, labels):
    _, predicted = torch.max(logits, 1)
    predicted_np = predicted.cpu().numpy()
    labels_np = labels.cpu().numpy()
    balanced_accuracy = balanced_accuracy_score(labels_np, predicted_np)
    return balanced_accuracy


def main_training_loop(
    epochs,
    run_loader,
    dataset_train=None,
    dataset_valid=None,
    log_every_n_batch=100,
    dataloader_train=None,
    dataloader_valid=None,
):
    batch_size = run_loader.config["batch_size"]

    if dataloader_train is None:
        dataloader_train = DataLoader(
            dataset_train, batch_size=batch_size, shuffle=True, num_workers=2
        )
    if dataloader_valid is None:
        dataloader_valid = DataLoader(
            dataset_valid, batch_size=batch_size, shuffle=True, num_workers=2
        )

    init_epoch, final_epoch = (
        run_loader.current_epoch,
        run_loader.current_epoch + epochs,
    )
    init_epoch, final_epoch = init_epoch + 1, final_epoch + 1

    logger_store_dict = {}

    for epoch in range(init_epoch, final_epoch):
        logger_store_dict.update({"Epoch": epoch})
        run_loader.current_epoch = epoch

        for i, train_batch in enumerate(dataloader_train):
            # --------------------[training step]--------------------
            run_loader.model.train()
            images, labels = train_batch
            images, labels = images.to(run_loader.device), labels.to(run_loader.device)

            run_loader.optimizer.zero_grad()
            logits = run_loader.model(images)
            loss_main = run_loader.criterion(logits, labels)
            run_loader.current_loss = loss_main.item()

            loss = loss_main

            loss.backward()
            run_loader.optimizer.step()

            # --------------------[annealing step]--------------------
            if (
                run_loader.config.get("use_flex")
                and run_loader.config.get("masking_mechanism") == "SIGMOID_SMOOTHED"
            ):
                # running_iteration = logger.get_iterations("Running Tau")
                # running_tau = compute_tau(running_iteration, run_loader.config["tau"], annealing_factor)
                # logger.log({"Running Tau": running_tau})
                # run_loader.model.update_tau(running_tau)
                pass

            # --------------------[end training step]--------------------
            if i % log_every_n_batch == 0 or i == 0 or i == len(dataloader_train) - 1:
                train_accuracy = get_accuracy(logits, labels)
                train_accuracy_balanced = get_balanced_accuracy(logits, labels)
                if i == 0:
                    _, predicted = torch.max(logits, 1)
                    with open("debug_predictions.txt", "a") as f:
                        f.write(f"Batch {i}\n")
                        f.write(f"LABELS: {labels[:20].tolist()}\n")
                        f.write(f"PREDS:  {predicted[:20].tolist()}\n")
                logger_store_dict.update({"Train Loss": loss.item()})
                logger_store_dict.update({"Train Accuracy": train_accuracy})
                logger_store_dict.update(
                    {"Train Accuracy Balanced": train_accuracy_balanced}
                )

                if run_loader.config.get("use_flex"):
                    global binariness, conv_ratio
                    binariness = run_loader.model.check_homogeneity()
                    conv_ratio = run_loader.model.check_conv_ratio()
                    logger_store_dict.update(
                        {
                            f"Binariness {idx}": value
                            for idx, value in enumerate(binariness)
                        }
                    )
                    logger_store_dict.update(
                        {
                            f"Conv Ratio {idx}": value
                            for idx, value in enumerate(conv_ratio)
                        }
                    )

                run_loader.logger.log(logger_store_dict)

        # ======== [ validate every epoch ] ========
        valid_batch = next(iter(dataloader_valid))
        run_loader.model.eval()

        # -------- [show in terminal every n batch] --------
        if torch.cuda.is_available():
            memory_usage = check_cuda_memory_usage()
            logger_store_dict.update({"CUDA Memory Usage": memory_usage})

        # run_loader.logger.show_last_row()

        # -------- [loggers] --------
        with torch.no_grad():
            images, labels = valid_batch
            images, labels = images.to(run_loader.device), labels.to(run_loader.device)

            logits = run_loader.model(images)
            loss = run_loader.criterion(logits, labels)

            valid_accuracy = get_accuracy(logits, labels)
            valid_accuracy_balanced = get_balanced_accuracy(logits, labels)

            logger_store_dict.update({"Valid Loss": loss.item()})
            logger_store_dict.update({"Valid Accuracy": valid_accuracy})
            logger_store_dict.update(
                {"Valid Accuracy Balanced": valid_accuracy_balanced}
            )

        # -------- [some derived measures] --------
        with torch.no_grad():
            # valid accuracy balanced - train accuracy balanced
            # bva = run_loader.logger.get_moving_average("Valid Accuracy Balanced")
            # bta = run_loader.logger.get_moving_average("Train Accuracy Balanced")
            # lv = run_loader.logger.get_moving_average("Valid Loss")
            # lt = run_loader.logger.get_moving_average("Train Loss")
            # logger_store_dict.update({"Balanced V Accuracy - T Accuracy": bva - bta})
            # logger_store_dict.update({"Loss Valid - Train": lv - lt})

            if run_loader.config.get("use_flex"):
                logger_store_dict.update({"Mean Binariness": np.mean(binariness)})
                logger_store_dict.update({"Mean Conv Ratio": np.mean(conv_ratio)})

        # ======== [tests & evaluations] ========
        run_loader.logger.log(logger_store_dict)
        run_loader.save_checkpoint()

        if hasattr(run_loader, "scheduler") and run_loader.scheduler:
            run_loader.scheduler.step()

    # run_loader.logger.on_end()
