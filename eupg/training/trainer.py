"""Unified training loop for standard and TabNet-like PyTorch models."""

from __future__ import annotations

from dataclasses import dataclass
from contextlib import nullcontext
from typing import Callable, Optional, Sequence, Tuple, Union

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from eupg.utils import get_logger

logger = get_logger(__name__)


@torch.no_grad()
def global_param_l2_norm(model: nn.Module) -> float:
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        return 0.0
    # .pow(2).sum() per param is a fused op; summing scalars on the same device
    # costs one kernel launch per param but avoids a large concat (parameters_to_vector).
    sq = sum(p.detach().float().pow(2).sum() for p in params)
    return float(sq.sqrt().item())


@dataclass
class TrainResult:
    model: nn.Module
    best_model: Optional[nn.Module]
    best_val_loss: float
    stopped_early: bool
    epochs_ran: int


@torch.no_grad()
def eval_loss(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: Union[str, torch.device],
    forward_fn: Optional[Callable[[nn.Module, torch.Tensor], torch.Tensor]] = None,
) -> float:
    """
    Safe eval:
      - uses no_grad
      - restores original train/eval mode
      - supports custom forward (e.g., TabNet with idxs)
    """
    device = torch.device(device)
    was_training = model.training
    model.eval()

    total = torch.zeros((), device=device)
    n_batches = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = forward_fn(model, x) if forward_fn is not None else model(x)
        loss = criterion(logits, y)
        total.add_(loss)
        n_batches += 1

    if was_training:
        model.train()

    # Single device sync per eval pass.
    return float(total.item()) / max(n_batches, 1)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    max_epochs: int,
    device: Union[str, torch.device] = "cuda",
    verbose_epoch: int = 10,
    patience: Optional[int] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    scheduler_step_per_batch: bool = True,
    metric_fn: Optional[Callable[[nn.Module, DataLoader], float]] = None,
    metric_name: str = "acc",
    report_each_epoch: bool = True,
    forward_fn: Optional[Callable[[nn.Module, torch.Tensor], torch.Tensor]] = None,
    use_amp: bool = False,
    grad_clip: Optional[float] = None,
    show_progress: bool = True,
) -> TrainResult:
    """
    Unified trainer for:
      - standard models (forward_fn=None)
      - TabNet-like models (provide forward_fn that uses idxs)

    Improvements vs original:
      - single implementation (no duplication)
      - correct val_loss handling (computed whenever needed)
      - safe eval that restores mode
      - optional AMP, grad clipping, scheduler
      - always returns the final epoch model (no best-checkpoint restore)
    """
    device = torch.device(device)
    model = model.to(device)

    amp_enabled = bool(use_amp) and device.type == "cuda"
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    except Exception:
        # Backward compatibility with older torch versions.
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    stopped_early = False
    epoch = -1  # guard against max_epochs=0 so TrainResult.epochs_ran is always valid

    # Whether we need validation this run
    has_val = val_loader is not None

    # Build the autocast context factory once — avoids hasattr + object alloc every batch.
    if amp_enabled and hasattr(torch, "amp"):
        def _amp_ctx():
            return torch.amp.autocast("cuda")
    elif amp_enabled:
        def _amp_ctx():  # type: ignore[misc]
            return torch.cuda.amp.autocast()
    else:
        _amp_ctx = nullcontext  # type: ignore[assignment]

    epoch_iter = range(int(max_epochs))
    if show_progress:
        epoch_iter = tqdm(epoch_iter, desc="Epochs", leave=False)

    for epoch in epoch_iter:
        model.train()
        # Accumulate loss as a GPU tensor to avoid a device sync (loss.item()) every batch.
        running = torch.zeros((), device=device)
        n_batches = 0

        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with _amp_ctx():
                logits = forward_fn(model, x) if forward_fn is not None else model(x)
                loss = criterion(logits, y)

            scaler.scale(loss).backward()

            if grad_clip is not None:
                # unscale before clipping when using AMP
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))

            scaler.step(optimizer)
            scaler.update()

            if scheduler is not None and bool(scheduler_step_per_batch):
                scheduler.step()

            running.add_(loss.detach())
            n_batches += 1

        # Single GPU→CPU sync per epoch instead of one per batch.
        train_loss = float(running.item()) / max(n_batches, 1)

        if scheduler is not None and (not bool(scheduler_step_per_batch)):
            scheduler.step()

        # Validation (only when asked, and only if val exists)
        do_val = has_val and (
            report_each_epoch
            or epoch == 0
            or ((epoch + 1) % max(1, int(verbose_epoch)) == 0)
            or (patience is not None)
        )
        val_loss = None
        if do_val:
            val_loss = eval_loss(model, val_loader, criterion, device=device, forward_fn=forward_fn)

        if report_each_epoch and val_loader is not None:
            train_eval_loss = eval_loss(model, train_loader, criterion, device=device, forward_fn=forward_fn)
            # Reuse val_loss already computed above — avoids a second full pass over val_loader.
            test_eval_loss = val_loss  # type: ignore[assignment]
            if metric_fn is not None:
                train_metric = float(metric_fn(model, train_loader))
                test_metric = float(metric_fn(model, val_loader))
                logger.info(
                    f"[epoch {epoch + 1:03d}/{int(max_epochs)}] "
                    f"train_loss={train_eval_loss:.4f} test_loss={test_eval_loss:.4f} "
                    f"train_{metric_name}={train_metric:.4f} test_{metric_name}={test_metric:.4f}"
                )
            else:
                logger.info(
                    f"[epoch {epoch + 1:03d}/{int(max_epochs)}] "
                    f"train_loss={train_eval_loss:.4f} test_loss={test_eval_loss:.4f}"
                )

        # Optional logging
        if show_progress and do_val:
            epoch_iter.set_postfix(train_loss=f"{train_loss:.4f}", val_loss=f"{val_loss:.4f}")

        # Keep API surface compatible, but do not early-stop or track best checkpoints.
        # `patience` is accepted for backward compatibility and ignored.

    # Always return final model state.
    return TrainResult(
        model=model,
        best_model=None,
        best_val_loss=float("nan"),
        stopped_early=stopped_early,
        epochs_ran=(epoch + 1),
    )
