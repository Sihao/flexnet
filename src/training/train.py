import numpy as np
import signal
import sys
from torch import clamp
import torch, warnings, random
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from sklearn.metrics import balanced_accuracy_score
from src.utils.device import check_cuda_memory_usage, is_ddp, is_main_process
from src.training.historical_methods.anneal import get_annealing_factor, compute_tau
from src.utils.device import select_device

warnings.filterwarnings("ignore")

# --- Graceful shutdown on SIGTERM (sent by SLURM wall-time handler) ---
_shutdown_requested = False


def _sigterm_handler(signum, frame):
    global _shutdown_requested
    _shutdown_requested = True
    print("\n[SIGTERM] Graceful shutdown requested. Will exit after current epoch completes.")


signal.signal(signal.SIGTERM, _sigterm_handler)


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


@torch.no_grad()
def run_full_validation(model, dataloader_valid, run_loader, epoch):
    """Full sweep over the entire validation set (not a single batch).

    Writes logs/full_val_epoch<N>.json in the same schema as
    scripts/eval_full_val.py, so the iso-accuracy selector / cron can consume
    it. Runs inline on the training GPU; rank-0 only (call site guards DDP).
    """
    import json
    import time
    from pathlib import Path

    model.eval()
    device = run_loader.device
    criterion = torch.nn.CrossEntropyLoss(reduction="sum")
    n = top1 = top5 = 0
    total_loss = 0.0
    t0 = time.time()
    for images, labels in dataloader_valid:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        total_loss += criterion(logits, labels).item()
        _, pred5 = logits.topk(5, dim=1)
        correct = pred5.eq(labels.view(-1, 1).expand_as(pred5))
        top1 += correct[:, 0].sum().item()
        top5 += correct.any(dim=1).sum().item()
        n += labels.size(0)
    if n == 0:
        print(f"[full-val] epoch {epoch}: empty val loader, skipping")
        return

    result = {
        "experiment_name": run_loader.run_folder.parent.name,
        "run_name": run_loader.run_folder.name,
        "epoch": epoch,
        "config": run_loader.config,
        "n": n,
        "top1": top1 / n,
        "top5": top5 / n,
        "loss": total_loss / n,
        "elapsed_s": time.time() - t0,
    }
    out_dir = Path(run_loader.run_folder) / "logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"full_val_epoch{epoch}.json"
    tmp_path = out_dir / f".full_val_epoch{epoch}.json.tmp"
    tmp_path.write_text(json.dumps(result, indent=2))
    tmp_path.replace(out_path)  # atomic
    print(
        f"[full-val] epoch {epoch}: top1={result['top1']:.4f} "
        f"top5={result['top5']:.4f} loss={result['loss']:.4f} "
        f"(n={n}, {result['elapsed_s']:.1f}s) -> {out_path.name}",
        flush=True,
    )


def main_training_loop(
    target_epoch,
    run_loader,
    dataset_train=None,
    dataset_valid=None,
    log_every_n_batch=100,
    dataloader_train=None,
    dataloader_valid=None,
):
    batch_size = run_loader.config["batch_size"]
    ddp = is_ddp()
    rank0 = is_main_process()

    # --- Build data loaders with DistributedSampler when using DDP ---
    train_sampler = None
    if dataloader_train is None:
        if ddp and dataset_train is not None:
            train_sampler = DistributedSampler(dataset_train, shuffle=True)
            dataloader_train = DataLoader(
                dataset_train, batch_size=batch_size, sampler=train_sampler,
                num_workers=2, pin_memory=True,
            )
        else:
            dataloader_train = DataLoader(
                dataset_train, batch_size=batch_size, shuffle=True, num_workers=2
            )
    if dataloader_valid is None:
        dataloader_valid = DataLoader(
            dataset_valid, batch_size=batch_size, shuffle=True, num_workers=2
        )

    # --- Wrap model in DDP ---
    model = run_loader.model
    if ddp:
        local_rank = int(torch.cuda.current_device())
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    init_epoch = run_loader.current_epoch + 1
    final_epoch = target_epoch + 1

    if init_epoch >= final_epoch:
        print(f"[train] Already at epoch {run_loader.current_epoch}, "
              f"target is {target_epoch}. Nothing to do.")
        return

    logger_store_dict = {}

    for epoch in range(init_epoch, final_epoch):
        logger_store_dict.update({"Epoch": epoch})
        run_loader.current_epoch = epoch

        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        # In DDP, model.module is the unwrapped model for flex-specific calls
        raw_model = model.module if ddp else model

        for i, train_batch in enumerate(dataloader_train):
            # --------------------[training step]--------------------
            model.train()
            images, labels = train_batch
            images, labels = images.to(run_loader.device), labels.to(run_loader.device)

            run_loader.optimizer.zero_grad()
            logits = model(images)
            loss_main = run_loader.criterion(logits, labels)
            run_loader.current_loss = loss_main.item()

            loss = loss_main

            loss.backward()

            # --------------------[gradient clipping]--------------------
            # The flex-conv mechanism is prone to gradient explosions under the
            # WarmupCosine schedule (valid loss spiking to 1e33, acc collapsing
            # to 0). Clip grad norm to stabilise; configurable via
            # `grad_clip_norm` (default 1.0, applied to both flex and vanilla
            # so the recipe stays identical). Set to 0/None to disable.
            grad_clip = run_loader.config.get("grad_clip_norm", 1.0)
            if grad_clip and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            run_loader.optimizer.step()

            # --------------------[annealing step]--------------------
            if (
                run_loader.config.get("use_flex")
                and run_loader.config.get("masking_mechanism") == "SIGMOID_SMOOTHED"
            ):
                # Annealing for SIGMOID_SMOOTHED is not currently active.
                # When enabled, call raw_model.update_tau(tau) here.
                _ = None

            # --------------------[end training step]--------------------
            if i % log_every_n_batch == 0 or i == 0 or i == len(dataloader_train) - 1:
                train_accuracy = get_accuracy(logits, labels)
                train_accuracy_balanced = get_balanced_accuracy(logits, labels)
                if rank0 and i == 0:
                    _, predicted = torch.max(logits, 1)
                    with open("debug_predictions.txt", "a") as f:
                        f.write(f"Batch {i}\n")
                        f.write(f"LABELS: {labels[:20].tolist()}\n")
                        f.write(f"PREDS:  {predicted[:20].tolist()}\n")

                if rank0:
                    logger_store_dict.update({"Train Loss": loss.item()})
                    logger_store_dict.update({"Train Accuracy": train_accuracy})
                    logger_store_dict.update(
                        {"Train Accuracy Balanced": train_accuracy_balanced}
                    )

                    if run_loader.config.get("use_flex"):
                        global binariness, conv_ratio
                        binariness = raw_model.check_homogeneity()
                        conv_ratio = raw_model.check_conv_ratio()
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
        model.eval()

        if rank0:
            # -------- [show in terminal every n batch] --------
            if torch.cuda.is_available():
                memory_usage = check_cuda_memory_usage()
                logger_store_dict.update({"CUDA Memory Usage": memory_usage})

        # -------- [loggers] --------
        with torch.no_grad():
            images, labels = valid_batch
            images, labels = images.to(run_loader.device), labels.to(run_loader.device)

            logits = model(images)
            loss = run_loader.criterion(logits, labels)

            valid_accuracy = get_accuracy(logits, labels)
            valid_accuracy_balanced = get_balanced_accuracy(logits, labels)

            if rank0:
                logger_store_dict.update({"Valid Loss": loss.item()})
                logger_store_dict.update({"Valid Accuracy": valid_accuracy})
                logger_store_dict.update(
                    {"Valid Accuracy Balanced": valid_accuracy_balanced}
                )

        # -------- [some derived measures] --------
        with torch.no_grad():
            if rank0 and run_loader.config.get("use_flex"):
                logger_store_dict.update({"Mean Binariness": np.mean(binariness)})
                logger_store_dict.update({"Mean Conv Ratio": np.mean(conv_ratio)})

        # ======== [tests & evaluations — rank 0 only] ========
        if rank0:
            run_loader.logger.log(logger_store_dict)
            run_loader.save_checkpoint()

            # -------- [full validation sweep every eval_every_n_epochs] --------
            # Writes logs/full_val_epoch<N>.json consumed by the iso-accuracy
            # selector / cron. Uses the same full val loader (whole val set).
            eval_every = run_loader.config.get("eval_every_n_epochs")
            if eval_every and eval_every > 0 and epoch % eval_every == 0:
                try:
                    run_full_validation(raw_model, dataloader_valid, run_loader, epoch)
                except Exception as e:  # never let eval crash training
                    print(f"[full-val] epoch {epoch} failed: {e}", flush=True)
                model.train()

        if hasattr(run_loader, "scheduler") and run_loader.scheduler:
            run_loader.scheduler.step()

        # Synchronize all processes at epoch boundary
        if ddp:
            torch.distributed.barrier()

        # --- Graceful exit on SIGTERM (SLURM wall-time) ---
        if _shutdown_requested:
            if rank0:
                print(f"[SIGTERM] Exiting after epoch {epoch}. Checkpoint saved.")
            if ddp:
                torch.distributed.destroy_process_group()
            sys.exit(0)
