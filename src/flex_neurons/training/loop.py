"""
Pure training loop -- no plotting, no analysis, no brain scores.

Returns a history dict with per-epoch metrics so callers can do whatever
they want with the numbers.
"""

import pathlib
import torch

from .schedulers import step_scheduler

try:
    from tqdm import tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False


def _accuracy(logits, labels):
    """Top-1 accuracy for a single batch."""
    _, predicted = torch.max(logits, 1)
    return (predicted == labels).float().mean().item()


def get_accuracy(logits, labels):
    """Public alias for top-1 accuracy; compatible with legacy callers."""
    _, predicted = torch.max(logits, 1)
    return (predicted == labels).float().mean().item()


def get_balanced_accuracy(logits, labels):
    """Balanced accuracy using sklearn; handles class imbalance."""
    from sklearn.metrics import balanced_accuracy_score
    _, predicted = torch.max(logits, 1)
    return balanced_accuracy_score(labels.cpu().numpy(), predicted.cpu().numpy())


def _epoch_bar(loader, desc, epoch, epochs):
    """Wrap loader in tqdm if available, else return as-is."""
    if _HAS_TQDM:
        return tqdm(loader, desc=f"[{epoch}/{epochs}] {desc}", leave=False)
    return loader


def evaluate(model, dataloader_valid, criterion, device, ddp=False):
    """Full sweep over the entire validation set.

    Unlike sampling a single batch, this iterates every batch in
    `dataloader_valid` and accumulates sample-weighted totals, so the
    returned metrics are exact over the whole validation set rather than
    quantized to 1/batch_size.

    When `ddp` is True, loss/accuracy totals are summed across all ranks
    via `torch.distributed.all_reduce`, and predictions/labels are gathered
    across ranks via `all_gather_object` so balanced accuracy is computed
    once, on rank 0, over the full distributed validation set. When `ddp`
    is False (single process / CPU), everything is computed directly on
    this process without touching `torch.distributed`.

    Returns:
        dict with keys "valid_loss", "valid_acc", "valid_acc_balanced".
    """
    from sklearn.metrics import balanced_accuracy_score

    was_training = model.training
    model.eval()

    loss_sum = 0.0
    correct = 0
    sample_count = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader_valid:
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            batch_size = labels.size(0)
            _, predicted = torch.max(logits, 1)

            loss_sum += loss.item() * batch_size
            correct += (predicted == labels).sum().item()
            sample_count += batch_size

            all_preds.append(predicted.cpu())
            all_labels.append(labels.cpu())

    if was_training:
        model.train()

    if ddp:
        totals = torch.tensor(
            [loss_sum, float(correct), float(sample_count)],
            dtype=torch.float64,
            device=device,
        )
        torch.distributed.all_reduce(totals, op=torch.distributed.ReduceOp.SUM)
        loss_sum, correct, sample_count = totals.tolist()

    valid_loss = loss_sum / max(sample_count, 1)
    valid_acc = correct / max(sample_count, 1)

    preds_cat = torch.cat(all_preds) if all_preds else torch.empty(0, dtype=torch.long)
    labels_cat = torch.cat(all_labels) if all_labels else torch.empty(0, dtype=torch.long)

    valid_acc_balanced = float("nan")
    if ddp:
        # Balanced accuracy needs every rank's predictions in one place;
        # gather them and compute once on rank 0, mirroring the
        # rank0-only logging contract used elsewhere in training.
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
        gathered_preds = [None] * world_size
        gathered_labels = [None] * world_size
        torch.distributed.all_gather_object(gathered_preds, preds_cat)
        torch.distributed.all_gather_object(gathered_labels, labels_cat)
        if rank == 0:
            full_preds = torch.cat(gathered_preds).numpy()
            full_labels = torch.cat(gathered_labels).numpy()
            if full_labels.size > 0:
                valid_acc_balanced = balanced_accuracy_score(full_labels, full_preds)
    elif labels_cat.numel() > 0:
        valid_acc_balanced = balanced_accuracy_score(labels_cat.numpy(), preds_cat.numpy())

    return {
        "valid_loss": valid_loss,
        "valid_acc": valid_acc,
        "valid_acc_balanced": valid_acc_balanced,
    }


def train(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    criterion,
    epochs: int,
    ckpt_dir: pathlib.Path,
    device: str = "cuda",
    *,
    log_interval: int = 100,
    save_every: int = 1,
) -> dict:
    """Run training. Returns history dict with keys train_loss train_acc val_loss val_acc
    val_acc_balanced per epoch. Validation runs a full sweep of val_loader (via evaluate())
    at the end of each epoch, not a single sampled batch."""

    # --- input validation ---
    if epochs < 1:
        raise ValueError(f"epochs must be >= 1, got {epochs}")

    ckpt_dir = pathlib.Path(ckpt_dir)
    try:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ValueError(f"Cannot create ckpt_dir {ckpt_dir!r}: {exc}") from exc

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_acc_balanced": [],
    }

    model.to(device)

    for epoch in range(1, epochs + 1):
        # ================================================================
        # Training phase
        # ================================================================
        model.train()
        running_loss = 0.0
        running_acc = 0.0
        n_batches = 0

        loader = _epoch_bar(train_loader, "train", epoch, epochs)

        for batch_idx, batch in enumerate(loader):
            images, labels = batch
            try:
                images = images.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                logits = model(images)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

            except RuntimeError as exc:
                if "out of memory" in str(exc).lower():
                    raise RuntimeError(
                        f"CUDA OOM at epoch {epoch}, batch {batch_idx}"
                    ) from exc
                raise

            batch_loss = loss.item()
            batch_acc = _accuracy(logits, labels)
            running_loss += batch_loss
            running_acc += batch_acc
            n_batches += 1

            if batch_idx % log_interval == 0:
                if _HAS_TQDM and hasattr(loader, "set_postfix"):
                    loader.set_postfix(loss=f"{batch_loss:.4f}", acc=f"{batch_acc:.4f}")
                else:
                    print(
                        f"  epoch {epoch}/{epochs}  batch {batch_idx}"
                        f"  loss={batch_loss:.4f}  acc={batch_acc:.4f}"
                    )

        epoch_train_loss = running_loss / max(n_batches, 1)
        epoch_train_acc = running_acc / max(n_batches, 1)

        # ================================================================
        # Validation phase -- full sweep over the entire validation set
        # (not a single sampled batch), via the module-level evaluate().
        # ================================================================
        if val_loader is None:
            epoch_val_loss = float("nan")
            epoch_val_acc = float("nan")
            epoch_val_acc_balanced = float("nan")
        else:
            try:
                val_metrics = evaluate(model, val_loader, criterion, device, ddp=False)
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower():
                    raise RuntimeError(
                        f"CUDA OOM during validation at epoch {epoch}"
                    ) from exc
                raise

            epoch_val_loss = val_metrics["valid_loss"]
            epoch_val_acc = val_metrics["valid_acc"]
            epoch_val_acc_balanced = val_metrics["valid_acc_balanced"]

        # ================================================================
        # Record metrics
        # ================================================================
        history["train_loss"].append(epoch_train_loss)
        history["train_acc"].append(epoch_train_acc)
        history["val_loss"].append(epoch_val_loss)
        history["val_acc"].append(epoch_val_acc)
        history["val_acc_balanced"].append(epoch_val_acc_balanced)

        print(
            f"epoch {epoch}/{epochs}"
            f"  train_loss={epoch_train_loss:.4f}  train_acc={epoch_train_acc:.4f}"
            f"  val_loss={epoch_val_loss:.4f}  val_acc={epoch_val_acc:.4f}"
            f"  val_acc_balanced={epoch_val_acc_balanced:.4f}"
        )

        # ================================================================
        # Scheduler step (per epoch, matching existing src/training/train.py)
        # ================================================================
        # epoch_val_loss is the metric ReduceLROnPlateau schedulers watch
        # (build_scheduler constructs them with mode="min"); non-plateau
        # schedulers (CosineAnnealingLR, SequentialLR) ignore it and step
        # on their own internal epoch counter. See step_scheduler().
        step_scheduler(scheduler, epoch_val_loss)

        # ================================================================
        # Checkpoint
        # ================================================================
        if save_every > 0 and epoch % save_every == 0:
            ckpt_path = ckpt_dir / f"epoch_{epoch}.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": epoch_train_loss,
                    "val_loss": epoch_val_loss,
                },
                ckpt_path,
            )

    return history
