"""Behavior cloning trainer for baseline models.

Trains MLP or GRU baselines on tensorized replay data using
masked cross-entropy loss over legal actions. Supports:
- Per-turn training (MLP) and sequence training (GRU)
- Early stopping on validation loss
- Checkpoint saving (best + periodic)
- Optional Weights & Biases logging
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.models.baseline_mlp import BaselineMLP, BaselineGRU, masked_cross_entropy

logger = logging.getLogger(__name__)


# ── Dataset wrappers for PyTorch DataLoader ─────────────────────────────


class TurnDataset(Dataset):
    """PyTorch Dataset wrapping pre-tensorized turn data.

    Each item is a single turn observation with tensors for
    own_team, opponent_team, field, context, legal_mask, action, game_result.
    """

    def __init__(self, data: list[dict[str, np.ndarray]]) -> None:
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = self.data[idx]
        return {k: torch.from_numpy(np.array(v)) for k, v in item.items()}


class SequenceDataset(Dataset):
    """PyTorch Dataset wrapping pre-tensorized battle sequences.

    Each item is a full battle trajectory with shape (seq_len, ...).
    """

    def __init__(self, data: list[dict[str, np.ndarray]], max_turns: int = 20) -> None:
        self.data = data
        self.max_turns = max_turns

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = self.data[idx]
        return {k: torch.from_numpy(np.array(v)) for k, v in item.items()}


def collate_turns(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Collate function for turn-level data."""
    result: dict[str, torch.Tensor] = {}
    for key in batch[0]:
        result[key] = torch.stack([item[key] for item in batch], dim=0)
    return result


def collate_sequences(
    batch: list[dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    """Collate function for sequence data with padding."""
    keys = batch[0].keys()
    result: dict[str, torch.Tensor] = {}

    # Find max sequence length in this batch
    seq_lens = []
    for item in batch:
        # seq_len is stored as a scalar
        if "seq_len" in item:
            seq_lens.append(item["seq_len"].item())
        else:
            # Infer from tensor shapes
            for k, v in item.items():
                if v.dim() >= 1:
                    seq_lens.append(v.shape[0])
                    break

    max_len = max(seq_lens) if seq_lens else 1

    for key in keys:
        if key == "seq_len":
            result[key] = torch.tensor(seq_lens, dtype=torch.long)
            continue

        tensors = [item[key] for item in batch]

        # Pad sequences to max_len if any tensor has a sequence dimension
        if tensors[0].dim() >= 1 and any(t.shape[0] != max_len for t in tensors):
            # Use -1 for action padding so ignore_index=-1 filters them out;
            # use 0 for everything else.
            pad_value = -1 if key == "action" else 0
            padded = []
            for t in tensors:
                if t.shape[0] < max_len:
                    pad_shape = list(t.shape)
                    pad_shape[0] = max_len - t.shape[0]
                    padding = torch.full(pad_shape, pad_value, dtype=t.dtype)
                    t = torch.cat([t, padding], dim=0)
                padded.append(t[:max_len])
            tensors = padded

        result[key] = torch.stack(tensors, dim=0)

    return result


# ── Training metrics ────────────────────────────────────────────────────


@dataclass
class EpochMetrics:
    """Metrics for a single training epoch."""

    epoch: int = 0
    train_loss: float = 0.0
    val_loss: float = 0.0
    train_accuracy: float = 0.0
    val_accuracy: float = 0.0
    train_top3_accuracy: float = 0.0
    val_top3_accuracy: float = 0.0
    learning_rate: float = 0.0
    epoch_time: float = 0.0
    examples_per_sec: float = 0.0


@dataclass
class TrainingResult:
    """Complete training result."""

    best_val_loss: float = float("inf")
    best_epoch: int = 0
    total_epochs: int = 0
    epoch_metrics: list[EpochMetrics] = field(default_factory=list)
    best_checkpoint_path: str = ""
    final_train_accuracy: float = 0.0
    final_val_accuracy: float = 0.0


# ── Trainer ─────────────────────────────────────────────────────────────


class BCTrainer:
    """Behavior cloning trainer for baseline models.

    Handles the training loop, validation, early stopping,
    and checkpoint management.
    """

    def __init__(
        self,
        model: nn.Module,
        train_data: list[dict[str, np.ndarray]],
        val_data: list[dict[str, np.ndarray]],
        checkpoint_dir: str | Path = "checkpoints",
        # Training hyperparameters
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        batch_size: int = 256,
        max_epochs: int = 50,
        early_stopping_patience: int = 5,
        # Architecture mode
        sequence_mode: bool = False,
        # Device
        device: str | torch.device = "cpu",
        # Optional W&B
        use_wandb: bool = False,
        wandb_project: str = "pokemon-battle-model",
        wandb_run_name: str = "",
        seed: int = 42,
    ) -> None:
        self.model = model.to(device)
        self.device = torch.device(device)
        self.sequence_mode = sequence_mode
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Set seed
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Create datasets and dataloaders
        if sequence_mode:
            train_dataset = SequenceDataset(train_data)
            val_dataset = SequenceDataset(val_data)
            collate_fn = collate_sequences
        else:
            train_dataset = TurnDataset(train_data)
            val_dataset = TurnDataset(val_data)
            collate_fn = collate_turns

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=0,
            pin_memory=self.device.type == "cuda",
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0,
            pin_memory=self.device.type == "cuda",
        )

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        # Scheduler
        total_steps = max_epochs * len(self.train_loader)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps,
            eta_min=1e-6,
        )

        self.max_epochs = max_epochs
        self.early_stopping_patience = early_stopping_patience

        # W&B
        self.use_wandb = use_wandb
        self._wandb_run = None
        if use_wandb:
            try:
                import wandb
                self._wandb_run = wandb.init(
                    project=wandb_project,
                    name=wandb_run_name or f"bc_baseline_{int(time.time())}",
                    config={
                        "architecture": type(model).__name__,
                        "learning_rate": learning_rate,
                        "batch_size": batch_size,
                        "max_epochs": max_epochs,
                        "train_size": len(train_data),
                        "val_size": len(val_data),
                        "sequence_mode": sequence_mode,
                    },
                )
            except ImportError:
                logger.warning("wandb not available, disabling logging")
                self.use_wandb = False

    def train(self) -> TrainingResult:
        """Run the full training loop.

        Returns:
            TrainingResult with metrics from all epochs.
        """
        result = TrainingResult()
        best_val_loss = float("inf")
        patience_counter = 0

        logger.info(
            "Starting BC training: %d train batches, %d val batches, "
            "max %d epochs, patience %d",
            len(self.train_loader),
            len(self.val_loader),
            self.max_epochs,
            self.early_stopping_patience,
        )

        for epoch in range(1, self.max_epochs + 1):
            epoch_start = time.time()

            # Train
            train_metrics = self._train_epoch()

            # Validate
            val_metrics = self._validate()

            epoch_time = time.time() - epoch_start
            lr = self.optimizer.param_groups[0]["lr"]

            metrics = EpochMetrics(
                epoch=epoch,
                train_loss=train_metrics["loss"],
                val_loss=val_metrics["loss"],
                train_accuracy=train_metrics["accuracy"],
                val_accuracy=val_metrics["accuracy"],
                train_top3_accuracy=train_metrics["top3_accuracy"],
                val_top3_accuracy=val_metrics["top3_accuracy"],
                learning_rate=lr,
                epoch_time=epoch_time,
                examples_per_sec=train_metrics["total_examples"] / max(epoch_time, 0.001),
            )
            result.epoch_metrics.append(metrics)

            logger.info(
                "Epoch %d/%d: train_loss=%.4f val_loss=%.4f "
                "train_acc=%.3f val_acc=%.3f val_top3=%.3f "
                "lr=%.2e time=%.1fs",
                epoch, self.max_epochs,
                metrics.train_loss, metrics.val_loss,
                metrics.train_accuracy, metrics.val_accuracy,
                metrics.val_top3_accuracy,
                lr, epoch_time,
            )

            # W&B logging
            if self.use_wandb and self._wandb_run:
                import wandb
                wandb.log({
                    "epoch": epoch,
                    "train/loss": metrics.train_loss,
                    "train/accuracy": metrics.train_accuracy,
                    "train/top3_accuracy": metrics.train_top3_accuracy,
                    "val/loss": metrics.val_loss,
                    "val/accuracy": metrics.val_accuracy,
                    "val/top3_accuracy": metrics.val_top3_accuracy,
                    "learning_rate": lr,
                    "epoch_time": epoch_time,
                    "examples_per_sec": metrics.examples_per_sec,
                })

            # Checkpoint
            self._save_checkpoint(epoch, metrics.val_loss, is_best=False)

            # Early stopping
            if metrics.val_loss < best_val_loss:
                best_val_loss = metrics.val_loss
                patience_counter = 0
                result.best_val_loss = best_val_loss
                result.best_epoch = epoch
                self._save_checkpoint(epoch, metrics.val_loss, is_best=True)
                result.best_checkpoint_path = str(
                    self.checkpoint_dir / "best_model.pt"
                )
            else:
                patience_counter += 1
                if patience_counter >= self.early_stopping_patience:
                    logger.info(
                        "Early stopping at epoch %d (patience=%d)",
                        epoch, self.early_stopping_patience,
                    )
                    break

        result.total_epochs = len(result.epoch_metrics)
        if result.epoch_metrics:
            last = result.epoch_metrics[-1]
            result.final_train_accuracy = last.train_accuracy
            result.final_val_accuracy = last.val_accuracy

        if self.use_wandb and self._wandb_run:
            import wandb
            wandb.finish()

        return result

    def _train_epoch(self) -> dict[str, float]:
        """Run one training epoch."""
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_top3_correct = 0
        total_examples = 0

        for batch in self.train_loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}

            own_team = batch["own_team"]
            opp_team = batch["opponent_team"]
            field_feat = batch["field"]
            context = batch["context"]
            legal_mask = batch["legal_mask"]
            action = batch["action"]

            # Forward
            if self.sequence_mode and isinstance(self.model, BaselineGRU):
                logits = self.model(
                    own_team, opp_team, field_feat, context,
                    legal_mask=legal_mask,
                    seq_len=batch.get("seq_len"),
                )
            else:
                logits = self.model(
                    own_team, opp_team, field_feat, context,
                    legal_mask=legal_mask,
                )

            # Compute loss (ignore padding actions with index -1 or 0)
            loss = masked_cross_entropy(logits, action, legal_mask, ignore_index=-1)

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()

            # Metrics
            with torch.no_grad():
                flat_logits = logits.reshape(-1, logits.shape[-1])
                flat_actions = action.reshape(-1)
                valid = flat_actions >= 0

                if valid.any():
                    preds = flat_logits[valid].argmax(dim=-1)
                    correct = (preds == flat_actions[valid]).sum().item()
                    total_correct += correct

                    # Top-3 accuracy
                    top3_preds = flat_logits[valid].topk(
                        min(3, flat_logits.shape[-1]), dim=-1
                    ).indices
                    top3_correct = (
                        top3_preds == flat_actions[valid].unsqueeze(-1)
                    ).any(dim=-1).sum().item()
                    total_top3_correct += top3_correct

                    total_examples += valid.sum().item()

            total_loss += loss.item() * max(valid.sum().item(), 1)

        n = max(total_examples, 1)
        return {
            "loss": total_loss / n,
            "accuracy": total_correct / n,
            "top3_accuracy": total_top3_correct / n,
            "total_examples": total_examples,
        }

    @torch.no_grad()
    def _validate(self) -> dict[str, float]:
        """Run validation."""
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_top3_correct = 0
        total_examples = 0

        for batch in self.val_loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}

            own_team = batch["own_team"]
            opp_team = batch["opponent_team"]
            field_feat = batch["field"]
            context = batch["context"]
            legal_mask = batch["legal_mask"]
            action = batch["action"]

            if self.sequence_mode and isinstance(self.model, BaselineGRU):
                logits = self.model(
                    own_team, opp_team, field_feat, context,
                    legal_mask=legal_mask,
                    seq_len=batch.get("seq_len"),
                )
            else:
                logits = self.model(
                    own_team, opp_team, field_feat, context,
                    legal_mask=legal_mask,
                )

            loss = masked_cross_entropy(logits, action, legal_mask, ignore_index=-1)

            flat_logits = logits.reshape(-1, logits.shape[-1])
            flat_actions = action.reshape(-1)
            valid = flat_actions >= 0

            if valid.any():
                preds = flat_logits[valid].argmax(dim=-1)
                correct = (preds == flat_actions[valid]).sum().item()
                total_correct += correct

                top3_preds = flat_logits[valid].topk(
                    min(3, flat_logits.shape[-1]), dim=-1
                ).indices
                top3_correct = (
                    top3_preds == flat_actions[valid].unsqueeze(-1)
                ).any(dim=-1).sum().item()
                total_top3_correct += top3_correct

                total_examples += valid.sum().item()

            total_loss += loss.item() * max(valid.sum().item(), 1)

        n = max(total_examples, 1)
        return {
            "loss": total_loss / n,
            "accuracy": total_correct / n,
            "top3_accuracy": total_top3_correct / n,
        }

    def _save_checkpoint(
        self, epoch: int, val_loss: float, is_best: bool = False
    ) -> None:
        """Save a model checkpoint."""
        state = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_loss": val_loss,
            "model_class": type(self.model).__name__,
        }

        if is_best:
            path = self.checkpoint_dir / "best_model.pt"
            torch.save(state, path)
            logger.info("Saved best checkpoint at epoch %d (val_loss=%.4f)", epoch, val_loss)

        # Also save periodic checkpoints
        path = self.checkpoint_dir / f"checkpoint_epoch_{epoch:03d}.pt"
        torch.save(state, path)

        # Clean up old checkpoints (keep last 3 + best)
        checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_epoch_*.pt"))
        if len(checkpoints) > 3:
            for old_ckpt in checkpoints[:-3]:
                old_ckpt.unlink()
