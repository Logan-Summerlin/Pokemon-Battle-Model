"""Transformer trainer for behavior cloning with auxiliary heads.

Extends the BC training pipeline to support:
- Composite loss: policy + auxiliary hidden-info + value head
- Mixed precision training (bf16/fp16)
- Gradient accumulation
- Warmup + cosine annealing schedule
- Per-component loss tracking
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

from src.models.battle_transformer import (
    BattleTransformer,
    TransformerConfig,
    TransformerOutput,
    compute_total_loss,
)
from src.environment.action_space import NUM_ACTIONS

logger = logging.getLogger(__name__)


# ── Dataset ──────────────────────────────────────────────────────────────


class TransformerSequenceDataset(Dataset):
    """Dataset wrapping pre-tensorized battle sequences with auxiliary targets.

    Each item is a dict of numpy arrays with keys:
        own_team, opponent_team, field, context, legal_mask, action, game_result,
        seq_len, and auxiliary target keys (item_targets, speed_targets, etc.)
    """

    def __init__(self, data: list[dict[str, np.ndarray]]) -> None:
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = self.data[idx]
        return {k: torch.from_numpy(np.array(v)) for k, v in item.items()}


def collate_transformer(
    batch: list[dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    """Collate function for transformer sequence data with padding."""
    keys = batch[0].keys()
    result: dict[str, torch.Tensor] = {}

    # Find max sequence length
    seq_lens = []
    for item in batch:
        if "seq_len" in item:
            seq_lens.append(item["seq_len"].item())
        else:
            for k, v in item.items():
                if v.dim() >= 2:  # (seq, ...)
                    seq_lens.append(v.shape[0])
                    break

    max_len = max(seq_lens) if seq_lens else 1

    for key in keys:
        if key == "seq_len":
            result[key] = torch.tensor(seq_lens, dtype=torch.long)
            continue

        tensors = [item[key] for item in batch]

        # Pad sequences to max_len if needed
        if tensors[0].dim() >= 1 and any(t.shape[0] != max_len for t in tensors):
            pad_value = -1 if "target" in key or key == "action" else 0
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


# ── Metrics ──────────────────────────────────────────────────────────────


@dataclass
class TransformerEpochMetrics:
    """Metrics for a single training epoch."""

    epoch: int = 0
    train_loss: float = 0.0
    val_loss: float = 0.0
    train_policy_loss: float = 0.0
    val_policy_loss: float = 0.0
    train_aux_loss: float = 0.0
    val_aux_loss: float = 0.0
    train_value_loss: float = 0.0
    val_value_loss: float = 0.0
    train_accuracy: float = 0.0
    val_accuracy: float = 0.0
    train_top3_accuracy: float = 0.0
    val_top3_accuracy: float = 0.0
    learning_rate: float = 0.0
    epoch_time: float = 0.0
    examples_per_sec: float = 0.0
    # Auxiliary head metrics
    aux_item_accuracy: float = 0.0
    aux_speed_accuracy: float = 0.0
    aux_role_accuracy: float = 0.0


@dataclass
class TransformerTrainingResult:
    """Complete training result."""

    best_val_loss: float = float("inf")
    best_epoch: int = 0
    total_epochs: int = 0
    epoch_metrics: list[TransformerEpochMetrics] = field(default_factory=list)
    best_checkpoint_path: str = ""
    config: dict[str, Any] = field(default_factory=dict)


# ── Learning rate scheduler with warmup ──────────────────────────────────


class WarmupCosineScheduler:
    """Cosine annealing with linear warmup."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr: float = 1e-6,
    ) -> None:
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.base_lrs = [pg["lr"] for pg in optimizer.param_groups]
        self._step = 0

    def step(self) -> None:
        self._step += 1
        lr = self._get_lr()
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr

    def _get_lr(self) -> float:
        if self._step < self.warmup_steps:
            # Linear warmup
            return self.base_lrs[0] * self._step / max(self.warmup_steps, 1)
        else:
            # Cosine decay
            progress = (self._step - self.warmup_steps) / max(
                self.total_steps - self.warmup_steps, 1
            )
            progress = min(progress, 1.0)
            import math
            return self.min_lr + 0.5 * (self.base_lrs[0] - self.min_lr) * (
                1 + math.cos(math.pi * progress)
            )

    def get_lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]


# ── Trainer ──────────────────────────────────────────────────────────────


class TransformerTrainer:
    """Trainer for BattleTransformer with multi-loss support.

    Handles:
    - Policy loss (masked cross-entropy)
    - Auxiliary hidden-info loss (item, speed, role, move families)
    - Value loss (win probability)
    - Mixed precision training
    - Gradient accumulation
    - Warmup + cosine schedule
    - Early stopping & checkpointing
    """

    def __init__(
        self,
        model: BattleTransformer,
        config: TransformerConfig,
        train_data: list[dict[str, np.ndarray]],
        val_data: list[dict[str, np.ndarray]],
        checkpoint_dir: str | Path = "checkpoints/transformer",
        # Training hyperparameters
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        batch_size: int = 64,
        max_epochs: int = 50,
        early_stopping_patience: int = 5,
        warmup_steps: int = 1000,
        min_lr: float = 1e-6,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        # Precision
        mixed_precision: bool = True,
        # Device
        device: str | torch.device = "cpu",
        # Optional W&B
        use_wandb: bool = False,
        wandb_project: str = "pokemon-battle-model",
        wandb_run_name: str = "",
        seed: int = 42,
    ) -> None:
        self.model = model.to(device)
        self.config = config
        self.device = torch.device(device)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.max_epochs = max_epochs
        self.early_stopping_patience = early_stopping_patience

        # Set seed
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Create datasets
        train_dataset = TransformerSequenceDataset(train_data)
        val_dataset = TransformerSequenceDataset(val_data)

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_transformer,
            num_workers=0,
            pin_memory=self.device.type == "cuda",
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_transformer,
            num_workers=0,
            pin_memory=self.device.type == "cuda",
        )

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
        )

        # Scheduler with warmup
        total_steps = max_epochs * len(self.train_loader) // gradient_accumulation_steps
        self.scheduler = WarmupCosineScheduler(
            self.optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            min_lr=min_lr,
        )

        # Mixed precision
        self.use_amp = mixed_precision and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda") if self.use_amp else None

        # W&B
        self.use_wandb = use_wandb
        self._wandb_run = None
        if use_wandb:
            try:
                import wandb
                self._wandb_run = wandb.init(
                    project=wandb_project,
                    name=wandb_run_name or f"transformer_{int(time.time())}",
                    config={
                        "model": "BattleTransformer",
                        "num_layers": config.num_layers,
                        "hidden_dim": config.hidden_dim,
                        "num_heads": config.num_heads,
                        "learning_rate": learning_rate,
                        "batch_size": batch_size,
                        "max_epochs": max_epochs,
                        "train_size": len(train_data),
                        "val_size": len(val_data),
                        "aux_weight": config.auxiliary_loss_weight,
                        "value_weight": config.value_loss_weight,
                        "params": model.count_parameters(),
                    },
                )
            except ImportError:
                logger.warning("wandb not available, disabling logging")
                self.use_wandb = False

        logger.info(
            "TransformerTrainer initialized: %d params, %d train batches, "
            "%d val batches, device=%s, amp=%s",
            model.count_parameters(),
            len(self.train_loader),
            len(self.val_loader),
            self.device,
            self.use_amp,
        )

    def _extract_aux_targets(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor] | None:
        """Extract auxiliary targets from batch if available."""
        aux_keys = ["item_targets", "speed_targets", "role_targets",
                     "move_family_targets"]
        aux_targets = {}
        for key in aux_keys:
            if key in batch:
                aux_targets[key] = batch[key]

        return aux_targets if aux_targets else None

    def _forward_step(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, float], dict[str, Any]]:
        """Run a single forward step and compute loss.

        Returns:
            (loss, loss_components, extra_info)
        """
        own_team = batch["own_team"]
        opp_team = batch["opponent_team"]
        field_feat = batch["field"]
        context = batch["context"]
        legal_mask = batch["legal_mask"]
        action = batch["action"]
        game_result = batch.get("game_result")
        seq_len = batch.get("seq_len")

        # Extract auxiliary targets
        aux_targets = self._extract_aux_targets(batch)

        # Forward pass
        output = self.model(
            own_team, opp_team, field_feat, context,
            legal_mask=legal_mask,
            seq_len=seq_len,
            return_auxiliary=(aux_targets is not None),
            return_value=(game_result is not None and self.config.use_value_head),
        )

        if isinstance(output, torch.Tensor):
            # Baseline-compatible mode
            from src.models.baseline_mlp import masked_cross_entropy
            loss = masked_cross_entropy(output, action, legal_mask, ignore_index=-1)
            return loss, {"policy": loss.item(), "total": loss.item()}, {"logits": output}

        # For sequence models, get the action target for the last step
        is_sequence = own_team.dim() == 4
        if is_sequence:
            batch_size = own_team.shape[0]
            if seq_len is not None:
                last_idx = (seq_len - 1).clamp(min=0)
            else:
                last_idx = torch.full((batch_size,), action.shape[1] - 1, device=action.device)

            # Get last step's action
            action_last = torch.gather(action, dim=1, index=last_idx.unsqueeze(1)).squeeze(1)

            # Get last step's legal mask
            if legal_mask.dim() == 3:
                legal_last = torch.gather(
                    legal_mask, dim=1,
                    index=last_idx.unsqueeze(1).unsqueeze(2).expand(-1, 1, NUM_ACTIONS),
                ).squeeze(1)
            else:
                legal_last = legal_mask

            # Get last step's game result
            if game_result is not None and game_result.dim() == 2:
                game_result_last = torch.gather(
                    game_result, dim=1, index=last_idx.unsqueeze(1)
                ).squeeze(1)
            elif game_result is not None:
                game_result_last = game_result
            else:
                game_result_last = None

            # Get last step's aux targets
            if aux_targets is not None:
                aux_last = {}
                for key, val in aux_targets.items():
                    if val.dim() == 2:
                        # (batch, seq) -> (batch,) -- per-slot
                        # Actually: (batch, seq, 6) or (batch, seq, 6, 10)
                        pass
                    if val.dim() >= 2:
                        # Get last step
                        idx = last_idx
                        if val.dim() == 2:
                            aux_last[key] = torch.gather(
                                val, dim=1, index=idx.unsqueeze(1)
                            ).squeeze(1)
                        elif val.dim() == 3:
                            aux_last[key] = torch.gather(
                                val, dim=1,
                                index=idx.unsqueeze(1).unsqueeze(2).expand(-1, 1, val.shape[2]),
                            ).squeeze(1)
                        elif val.dim() == 4:
                            aux_last[key] = torch.gather(
                                val, dim=1,
                                index=idx.unsqueeze(1).unsqueeze(2).unsqueeze(3).expand(
                                    -1, 1, val.shape[2], val.shape[3]
                                ),
                            ).squeeze(1)
                    else:
                        aux_last[key] = val
                aux_targets = aux_last
        else:
            action_last = action
            legal_last = legal_mask
            game_result_last = game_result

        # Compute total loss
        loss, loss_dict = compute_total_loss(
            output, action_last, legal_last,
            aux_targets=aux_targets,
            game_result=game_result_last,
            config=self.config,
        )

        return loss, loss_dict, {"logits": output.policy_logits, "output": output}

    def _train_epoch(self) -> dict[str, float]:
        """Run one training epoch."""
        self.model.train()
        total_loss = 0.0
        total_policy_loss = 0.0
        total_aux_loss = 0.0
        total_value_loss = 0.0
        total_correct = 0
        total_top3_correct = 0
        total_examples = 0
        num_batches = 0

        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(self.train_loader):
            batch = {k: v.to(self.device) for k, v in batch.items()}

            # Forward with optional AMP
            if self.use_amp:
                with torch.amp.autocast("cuda"):
                    loss, loss_dict, extra = self._forward_step(batch)
                    loss = loss / self.gradient_accumulation_steps
                self.scaler.scale(loss).backward()
            else:
                loss, loss_dict, extra = self._forward_step(batch)
                loss = loss / self.gradient_accumulation_steps
                loss.backward()

            # Gradient accumulation step
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.optimizer.step()

                self.scheduler.step()
                self.optimizer.zero_grad()

            # Accumulate metrics
            total_loss += loss_dict.get("total", 0.0)
            total_policy_loss += loss_dict.get("policy", 0.0)
            total_aux_loss += loss_dict.get("auxiliary", 0.0)
            total_value_loss += loss_dict.get("value", 0.0)
            num_batches += 1

            # Action accuracy
            logits = extra["logits"]
            action = batch["action"]
            seq_len = batch.get("seq_len")

            with torch.no_grad():
                # For sequence data, get last step's action
                if action.dim() == 2:
                    if seq_len is not None:
                        last_idx = (seq_len - 1).clamp(min=0)
                    else:
                        last_idx = torch.full(
                            (action.shape[0],), action.shape[1] - 1,
                            device=action.device
                        )
                    flat_actions = torch.gather(
                        action, dim=1, index=last_idx.unsqueeze(1)
                    ).squeeze(1)
                else:
                    flat_actions = action

                valid = flat_actions >= 0
                if valid.any():
                    preds = logits[valid].argmax(dim=-1)
                    total_correct += (preds == flat_actions[valid]).sum().item()

                    top3 = logits[valid].topk(min(3, logits.shape[-1]), dim=-1).indices
                    total_top3_correct += (
                        top3 == flat_actions[valid].unsqueeze(-1)
                    ).any(dim=-1).sum().item()

                    total_examples += valid.sum().item()

        n = max(num_batches, 1)
        ne = max(total_examples, 1)
        return {
            "loss": total_loss / n,
            "policy_loss": total_policy_loss / n,
            "aux_loss": total_aux_loss / n,
            "value_loss": total_value_loss / n,
            "accuracy": total_correct / ne,
            "top3_accuracy": total_top3_correct / ne,
            "total_examples": total_examples,
        }

    @torch.no_grad()
    def _validate(self) -> dict[str, float]:
        """Run validation."""
        self.model.eval()
        total_loss = 0.0
        total_policy_loss = 0.0
        total_aux_loss = 0.0
        total_value_loss = 0.0
        total_correct = 0
        total_top3_correct = 0
        total_examples = 0
        num_batches = 0

        # Auxiliary accuracy tracking
        aux_correct: dict[str, int] = {"item": 0, "speed": 0, "role": 0}
        aux_total: dict[str, int] = {"item": 0, "speed": 0, "role": 0}

        for batch in self.val_loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}

            if self.use_amp:
                with torch.amp.autocast("cuda"):
                    loss, loss_dict, extra = self._forward_step(batch)
            else:
                loss, loss_dict, extra = self._forward_step(batch)

            total_loss += loss_dict.get("total", 0.0)
            total_policy_loss += loss_dict.get("policy", 0.0)
            total_aux_loss += loss_dict.get("auxiliary", 0.0)
            total_value_loss += loss_dict.get("value", 0.0)
            num_batches += 1

            # Action accuracy
            logits = extra["logits"]
            action = batch["action"]
            seq_len = batch.get("seq_len")

            if action.dim() == 2:
                if seq_len is not None:
                    last_idx = (seq_len - 1).clamp(min=0)
                else:
                    last_idx = torch.full(
                        (action.shape[0],), action.shape[1] - 1,
                        device=action.device
                    )
                flat_actions = torch.gather(
                    action, dim=1, index=last_idx.unsqueeze(1)
                ).squeeze(1)
            else:
                flat_actions = action

            valid = flat_actions >= 0
            if valid.any():
                preds = logits[valid].argmax(dim=-1)
                total_correct += (preds == flat_actions[valid]).sum().item()

                top3 = logits[valid].topk(min(3, logits.shape[-1]), dim=-1).indices
                total_top3_correct += (
                    top3 == flat_actions[valid].unsqueeze(-1)
                ).any(dim=-1).sum().item()

                total_examples += valid.sum().item()

            # Auxiliary accuracy
            if "output" in extra and isinstance(extra["output"], TransformerOutput):
                output = extra["output"]
                if output.auxiliary_preds is not None:
                    for head_name, target_key in [
                        ("item_logits", "item_targets"),
                        ("speed_logits", "speed_targets"),
                        ("role_logits", "role_targets"),
                    ]:
                        short_name = head_name.split("_")[0]
                        if head_name in output.auxiliary_preds and target_key in batch:
                            pred = output.auxiliary_preds[head_name]  # (batch, 6, classes)
                            target = batch[target_key]  # (batch, seq, 6) or (batch, 6)
                            # Get last step if sequence
                            if target.dim() == 3 and seq_len is not None:
                                last_idx2 = (seq_len - 1).clamp(min=0)
                                target = torch.gather(
                                    target, dim=1,
                                    index=last_idx2.unsqueeze(1).unsqueeze(2).expand(
                                        -1, 1, target.shape[2]
                                    ),
                                ).squeeze(1)
                            elif target.dim() == 3:
                                target = target[:, -1]

                            flat_pred = pred.reshape(-1, pred.shape[-1])
                            flat_target = target.reshape(-1)
                            valid_aux = flat_target >= 0
                            if valid_aux.any():
                                aux_preds = flat_pred[valid_aux].argmax(dim=-1)
                                aux_correct[short_name] += (
                                    aux_preds == flat_target[valid_aux]
                                ).sum().item()
                                aux_total[short_name] += valid_aux.sum().item()

        n = max(num_batches, 1)
        ne = max(total_examples, 1)
        result = {
            "loss": total_loss / n,
            "policy_loss": total_policy_loss / n,
            "aux_loss": total_aux_loss / n,
            "value_loss": total_value_loss / n,
            "accuracy": total_correct / ne,
            "top3_accuracy": total_top3_correct / ne,
        }

        # Add auxiliary accuracies
        for name in ["item", "speed", "role"]:
            if aux_total[name] > 0:
                result[f"aux_{name}_accuracy"] = aux_correct[name] / aux_total[name]
            else:
                result[f"aux_{name}_accuracy"] = 0.0

        return result

    def train(self) -> TransformerTrainingResult:
        """Run the full training loop."""
        result = TransformerTrainingResult(
            config={
                "num_layers": self.config.num_layers,
                "hidden_dim": self.config.hidden_dim,
                "num_heads": self.config.num_heads,
                "params": self.model.count_parameters(),
            }
        )
        best_val_loss = float("inf")
        patience_counter = 0

        logger.info(
            "Starting transformer training: %d params, %d train batches, "
            "%d val batches, max %d epochs",
            self.model.count_parameters(),
            len(self.train_loader),
            len(self.val_loader),
            self.max_epochs,
        )

        for epoch in range(1, self.max_epochs + 1):
            epoch_start = time.time()

            train_metrics = self._train_epoch()
            val_metrics = self._validate()

            epoch_time = time.time() - epoch_start
            lr = self.scheduler.get_lr()

            metrics = TransformerEpochMetrics(
                epoch=epoch,
                train_loss=train_metrics["loss"],
                val_loss=val_metrics["loss"],
                train_policy_loss=train_metrics["policy_loss"],
                val_policy_loss=val_metrics["policy_loss"],
                train_aux_loss=train_metrics["aux_loss"],
                val_aux_loss=val_metrics["aux_loss"],
                train_value_loss=train_metrics["value_loss"],
                val_value_loss=val_metrics["value_loss"],
                train_accuracy=train_metrics["accuracy"],
                val_accuracy=val_metrics["accuracy"],
                train_top3_accuracy=train_metrics["top3_accuracy"],
                val_top3_accuracy=val_metrics["top3_accuracy"],
                learning_rate=lr,
                epoch_time=epoch_time,
                examples_per_sec=train_metrics["total_examples"] / max(epoch_time, 0.001),
                aux_item_accuracy=val_metrics.get("aux_item_accuracy", 0.0),
                aux_speed_accuracy=val_metrics.get("aux_speed_accuracy", 0.0),
                aux_role_accuracy=val_metrics.get("aux_role_accuracy", 0.0),
            )
            result.epoch_metrics.append(metrics)

            logger.info(
                "Epoch %d/%d: loss=%.4f val_loss=%.4f "
                "acc=%.3f val_acc=%.3f top3=%.3f "
                "pol=%.4f aux=%.4f val=%.4f "
                "lr=%.2e time=%.1fs",
                epoch, self.max_epochs,
                metrics.train_loss, metrics.val_loss,
                metrics.train_accuracy, metrics.val_accuracy,
                metrics.val_top3_accuracy,
                metrics.val_policy_loss, metrics.val_aux_loss,
                metrics.val_value_loss,
                lr, epoch_time,
            )

            if val_metrics.get("aux_item_accuracy", 0) > 0:
                logger.info(
                    "  Aux heads: item=%.3f speed=%.3f role=%.3f",
                    metrics.aux_item_accuracy,
                    metrics.aux_speed_accuracy,
                    metrics.aux_role_accuracy,
                )

            # W&B logging
            if self.use_wandb and self._wandb_run:
                import wandb
                log_dict = {
                    "epoch": epoch,
                    "train/loss": metrics.train_loss,
                    "train/policy_loss": metrics.train_policy_loss,
                    "train/aux_loss": metrics.train_aux_loss,
                    "train/value_loss": metrics.train_value_loss,
                    "train/accuracy": metrics.train_accuracy,
                    "train/top3_accuracy": metrics.train_top3_accuracy,
                    "val/loss": metrics.val_loss,
                    "val/policy_loss": metrics.val_policy_loss,
                    "val/aux_loss": metrics.val_aux_loss,
                    "val/value_loss": metrics.val_value_loss,
                    "val/accuracy": metrics.val_accuracy,
                    "val/top3_accuracy": metrics.val_top3_accuracy,
                    "val/aux_item_accuracy": metrics.aux_item_accuracy,
                    "val/aux_speed_accuracy": metrics.aux_speed_accuracy,
                    "val/aux_role_accuracy": metrics.aux_role_accuracy,
                    "learning_rate": lr,
                    "epoch_time": epoch_time,
                }
                wandb.log(log_dict)

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

        if self.use_wandb and self._wandb_run:
            import wandb
            wandb.finish()

        return result

    def _save_checkpoint(
        self, epoch: int, val_loss: float, is_best: bool = False
    ) -> None:
        """Save a model checkpoint."""
        state = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_loss": val_loss,
            "model_class": "BattleTransformer",
            "config": {
                "num_layers": self.config.num_layers,
                "hidden_dim": self.config.hidden_dim,
                "num_heads": self.config.num_heads,
                "dropout": self.config.dropout,
                "species_vocab_size": self.config.species_vocab_size,
                "moves_vocab_size": self.config.moves_vocab_size,
                "items_vocab_size": self.config.items_vocab_size,
                "abilities_vocab_size": self.config.abilities_vocab_size,
                "types_vocab_size": self.config.types_vocab_size,
                "status_vocab_size": self.config.status_vocab_size,
                "weather_vocab_size": self.config.weather_vocab_size,
                "terrain_vocab_size": self.config.terrain_vocab_size,
                "auxiliary_loss_weight": self.config.auxiliary_loss_weight,
                "use_value_head": self.config.use_value_head,
                "value_loss_weight": self.config.value_loss_weight,
            },
        }

        if is_best:
            path = self.checkpoint_dir / "best_model.pt"
            torch.save(state, path)
            logger.info("Saved best checkpoint at epoch %d (val_loss=%.4f)", epoch, val_loss)

        # Periodic checkpoint
        path = self.checkpoint_dir / f"checkpoint_epoch_{epoch:03d}.pt"
        torch.save(state, path)

        # Clean up old checkpoints (keep last 3 + best)
        checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_epoch_*.pt"))
        if len(checkpoints) > 3:
            for old_ckpt in checkpoints[:-3]:
                old_ckpt.unlink()
