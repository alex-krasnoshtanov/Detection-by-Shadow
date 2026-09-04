"""Training. Handles both the held-out-validation and the train-on-everything
regimes, because the project needed both for different reasons.

With a validation split you get an honest number and early stopping, which is
how the architecture was developed. Without one you get 15% more data and no
way to tell when to stop, which is how the final submission was produced -- a
fixed epoch budget with a cosine schedule, trained three times from different
seeds. ``docs/experiments.md`` has the numbers for both.
"""

from __future__ import annotations

import json
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from shadow_detection.config import TrainConfig
from shadow_detection.data import (
    Sample,
    ShadowDataset,
    TargetStats,
    stratification_labels,
)
from shadow_detection.features import NUM_FEATURES
from shadow_detection.geometry import FrameSize, iou, reconstruct
from shadow_detection.model import ShadowNet


@dataclass(slots=True)
class TrainResult:
    """Everything one seed produced."""

    seed: int
    checkpoint: Path
    history: dict[str, list[float]] = field(default_factory=dict)
    best_val_loss: float | None = None
    best_epoch: int | None = None
    seconds: float = 0.0


def pick_device(prefer: str | None = None) -> torch.device:
    """Resolve the compute device, honouring an explicit request."""
    if prefer:
        return torch.device(prefer)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


def _loss_fn(
    cfg: TrainConfig,
    outputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    batch: dict[str, torch.Tensor],
    cross_entropy: nn.Module,
    smooth_l1: nn.Module,
) -> torch.Tensor:
    side_logits, regression, direction_logits = outputs
    return (
        cfg.weight_side * cross_entropy(side_logits, batch["side"])
        + cfg.weight_regression * smooth_l1(regression, batch["regression"])
        + cfg.weight_direction * cross_entropy(direction_logits, batch["direction"])
    )


def _to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def train_one_seed(
    cfg: TrainConfig,
    samples: list[Sample],
    target_stats: TargetStats,
    seed: int,
    device: torch.device,
) -> TrainResult:
    """Train a single model and return where its weights landed."""
    seed_everything(seed)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = cfg.output_dir / f"model_seed{seed}.pt"

    if cfg.val_split > 0:
        labels = stratification_labels(samples)
        distinct = len(set(labels))
        held_out = int(len(samples) * cfg.val_split)
        if held_out < distinct:
            # sklearn's own message here talks about "the number of classes",
            # which is opaque unless you already know the split is stratified
            # across side x direction.
            raise ValueError(
                f"val_split={cfg.val_split} holds out only {held_out} of {len(samples)} "
                f"samples, but the split is stratified across {distinct} (side, direction) "
                f"groups and needs at least one of each. Raise val_split, or set it to 0 "
                f"to train on everything."
            )
        train_samples, val_samples = train_test_split(
            samples,
            test_size=cfg.val_split,
            random_state=seed,
            stratify=labels,
        )
    else:
        train_samples, val_samples = samples, []

    train_loader = DataLoader(
        ShadowDataset(train_samples, target_stats, cfg.input_size, augment=True),
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
        persistent_workers=cfg.num_workers > 0,
    )
    val_loader = (
        DataLoader(
            ShadowDataset(val_samples, target_stats, cfg.input_size, augment=False),
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=device.type == "cuda",
        )
        if val_samples
        else None
    )

    model = ShadowNet(num_features=NUM_FEATURES, dropout=cfg.dropout).to(device)
    optimizer = optim.AdamW(
        model.parameter_groups(cfg.lr, cfg.backbone_lr), weight_decay=cfg.weight_decay
    )
    if cfg.scheduler == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=1e-6)
    else:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    cross_entropy = nn.CrossEntropyLoss()
    smooth_l1 = nn.SmoothL1Loss()

    history: dict[str, list[float]] = {
        "train_loss": [],
        "val_loss": [],
        "val_side_acc": [],
        "val_direction_acc": [],
        "val_mean_iou": [],
    }
    best_val_loss = float("inf")
    best_epoch: int | None = None
    epochs_without_improvement = 0
    started = time.time()

    print(f"seed {seed}: {len(train_samples)} train, {len(val_samples)} val, {cfg.epochs} epochs")

    # A bar per epoch, only when someone is actually watching. Redirected output
    # (CI, a log file, the test suite) gets the plain per-epoch lines instead.
    show_progress = sys.stdout.isatty()

    for epoch in range(cfg.epochs):
        model.train()
        running, seen = 0.0, 0
        batches = tqdm(
            train_loader,
            desc=f"seed {seed} · epoch {epoch + 1}/{cfg.epochs}",
            unit="batch",
            leave=False,
            disable=not show_progress,
        )
        for raw_batch in batches:
            batch = _to_device(raw_batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                loss = _loss_fn(
                    cfg, model(batch["image"], batch["features"]), batch, cross_entropy, smooth_l1
                )
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += loss.item() * batch["image"].size(0)
            seen += batch["image"].size(0)
            if show_progress:
                batches.set_postfix(loss=f"{running / max(seen, 1):.4f}", refresh=False)
        batches.close()

        train_loss = running / max(seen, 1)
        history["train_loss"].append(train_loss)

        if val_loader is None:
            scheduler.step()
            # Every epoch, not every tenth: with no validation split this line
            # is the only signal that anything is happening, and 40 lines is
            # not a lot to read.
            print(
                f"  ep {epoch + 1:3d}/{cfg.epochs} | loss {train_loss:.4f} "
                f"| lr {optimizer.param_groups[-1]['lr']:.2e} "
                f"| {time.time() - started:.0f}s elapsed"
            )
            continue

        val_loss, side_acc, direction_acc, mean_iou = _evaluate(
            model, val_loader, cfg, device, use_amp, cross_entropy, smooth_l1, target_stats
        )
        history["val_loss"].append(val_loss)
        history["val_side_acc"].append(side_acc)
        history["val_direction_acc"].append(direction_acc)
        history["val_mean_iou"].append(mean_iou)
        # Only ReduceLROnPlateau consumes a metric. Handing a loss to a cosine
        # schedule would be read as an epoch index, which silently jumps the
        # learning rate to wherever that "epoch" falls on the curve.
        if cfg.scheduler == "plateau":
            scheduler.step(val_loss)
        else:
            scheduler.step()

        improved = val_loss < best_val_loss
        if improved:
            best_val_loss, best_epoch = val_loss, epoch
            epochs_without_improvement = 0
            torch.save(model.state_dict(), checkpoint)
        else:
            epochs_without_improvement += 1

        if epoch % 5 == 0 or improved:
            print(
                f"  ep {epoch + 1:3d}/{cfg.epochs} | train {train_loss:.4f} val {val_loss:.4f} "
                f"| IoU {mean_iou:.4f} | side {side_acc:.3f} dir {direction_acc:.3f}"
                f"{' *best*' if improved else ''}"
            )

        if epochs_without_improvement >= cfg.patience:
            print(f"  early stop at epoch {epoch}")
            break

    if val_loader is None:
        torch.save(model.state_dict(), checkpoint)

    elapsed = time.time() - started
    print(f"  saved {checkpoint} ({elapsed:.0f}s)")
    return TrainResult(
        seed=seed,
        checkpoint=checkpoint,
        history=history,
        best_val_loss=None if val_loader is None else best_val_loss,
        best_epoch=best_epoch,
        seconds=elapsed,
    )


@torch.no_grad()
def _evaluate(
    model: ShadowNet,
    loader: DataLoader,
    cfg: TrainConfig,
    device: torch.device,
    use_amp: bool,
    cross_entropy: nn.Module,
    smooth_l1: nn.Module,
    target_stats: TargetStats,
) -> tuple[float, float, float, float]:
    """Returns ``(loss, side accuracy, direction accuracy, mean IoU)``.

    Mean IoU is the point of this function. The training loss is a weighted sum
    over five outputs, which is fine for optimisation and useless for judging
    the model -- and the hackathon runs reported only the loss and the two
    accuracies, so the decomposed architecture never got a local IoU at all.
    That left the whole v2-to-v4 improvement visible only on a leaderboard.
    Reassembling the boxes here costs nothing and makes the number directly
    comparable to the 0.4675 the direct-regression model managed and the 0.4295
    of predicting the per-edge mean.
    """
    model.eval()
    total_loss, seen, side_correct, direction_correct = 0.0, 0, 0, 0
    iou_total = 0.0

    for raw_batch in loader:
        batch = _to_device(raw_batch, device)
        with torch.amp.autocast("cuda", enabled=use_amp):
            outputs = model(batch["image"], batch["features"])
            loss = _loss_fn(cfg, outputs, batch, cross_entropy, smooth_l1)
        side_logits, regression, direction_logits = outputs
        count = batch["image"].size(0)

        total_loss += loss.item() * count
        side_correct += (side_logits.argmax(1) == batch["side"]).sum().item()
        direction_correct += (direction_logits.argmax(1) == batch["direction"]).sum().item()
        iou_total += _batch_iou(
            predicted_sides=side_logits.argmax(1).cpu().numpy(),
            predicted_regression=regression.float().cpu().numpy(),
            true_sides=batch["side"].cpu().numpy(),
            true_regression=batch["regression"].float().cpu().numpy(),
            target_stats=target_stats,
            frame=cfg.frame,
        )
        seen += count

    return (
        total_loss / seen,
        side_correct / seen,
        direction_correct / seen,
        iou_total / seen,
    )


def _batch_iou(
    predicted_sides: np.ndarray,
    predicted_regression: np.ndarray,
    true_sides: np.ndarray,
    true_regression: np.ndarray,
    target_stats: TargetStats,
    frame: FrameSize,
) -> float:
    """Sum of per-sample IoU over a batch.

    Both boxes are reconstructed through the same standardisation, so this is
    the metric the challenge scored rather than a proxy for it.
    """
    total = 0.0
    for i in range(len(predicted_sides)):
        predicted = reconstruct(
            side=int(predicted_sides[i]),
            frame=frame,
            **target_stats.denormalize_vector(predicted_regression[i]),
        )
        truth = reconstruct(
            side=int(true_sides[i]),
            frame=frame,
            **target_stats.denormalize_vector(true_regression[i]),
        )
        total += iou(predicted, truth)
    return total


def train(
    cfg: TrainConfig,
    samples: list[Sample],
    device: torch.device | None = None,
) -> list[TrainResult]:
    """Train one model per seed and write the run manifest.

    The manifest (``run.json``) plus ``target_stats.json`` are what make a
    checkpoint directory self-contained: inference needs the standardisation
    constants and nothing else from the training data.
    """
    device = device or pick_device()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    target_stats = TargetStats.from_samples(samples)
    print(f"target statistics over {len(samples)} samples:\n{target_stats}")
    target_stats.save(cfg.output_dir / "target_stats.json")

    results = [train_one_seed(cfg, samples, target_stats, seed, device) for seed in cfg.seeds]

    manifest = {
        "config": cfg.to_dict(),
        "device": str(device),
        "num_samples": len(samples),
        "seeds": list(cfg.seeds),
        "results": [
            {
                "seed": r.seed,
                "checkpoint": r.checkpoint.name,
                "best_val_loss": r.best_val_loss,
                "best_epoch": r.best_epoch,
                "seconds": round(r.seconds, 1),
                "history": r.history,
            }
            for r in results
        ],
    }
    (cfg.output_dir / "run.json").write_text(json.dumps(manifest, indent=2) + "\n", "utf-8")
    print(f"wrote {cfg.output_dir / 'run.json'}")
    return results
