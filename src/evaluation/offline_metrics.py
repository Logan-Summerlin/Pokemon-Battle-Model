"""Offline evaluation metrics for Phase 4 models.

Computes metrics on held-out test data without running live battles:
- Action prediction accuracy (top-1, top-3)
- Negative log-likelihood (NLL) on test set
- Expected calibration error (ECE) for auxiliary heads
- Per-phase breakdown (early/mid/late game)
- Auxiliary head accuracy metrics
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.environment.action_space import NUM_ACTIONS
from src.models.battle_transformer import (
    BattleTransformer,
    TransformerOutput,
)
from src.training.transformer_trainer import (
    TransformerSequenceDataset,
    collate_transformer,
)

logger = logging.getLogger(__name__)


@dataclass
class OfflineMetrics:
    """Complete offline evaluation results."""

    # Action prediction
    top1_accuracy: float = 0.0
    top3_accuracy: float = 0.0
    action_nll: float = 0.0

    # Per-phase accuracy
    early_game_accuracy: float = 0.0   # turns 1-10
    mid_game_accuracy: float = 0.0     # turns 11-25
    late_game_accuracy: float = 0.0    # turns 26+

    # Auxiliary head metrics
    item_accuracy: float = 0.0
    speed_accuracy: float = 0.0
    role_accuracy: float = 0.0
    move_family_f1: float = 0.0

    # Calibration
    policy_ece: float = 0.0
    item_ece: float = 0.0
    speed_ece: float = 0.0

    # Value head
    value_accuracy: float = 0.0  # Binary accuracy for win prediction
    value_brier: float = 0.0     # Brier score

    # Dataset info
    num_examples: int = 0
    num_battles: int = 0

    def to_dict(self) -> dict[str, float]:
        return {k: v for k, v in self.__dict__.items()}

    def summary(self) -> str:
        lines = [
            "=== Offline Evaluation Results ===",
            f"Num examples: {self.num_examples}",
            f"Num battles: {self.num_battles}",
            "",
            "Action Prediction:",
            f"  Top-1 accuracy: {self.top1_accuracy:.4f}",
            f"  Top-3 accuracy: {self.top3_accuracy:.4f}",
            f"  NLL: {self.action_nll:.4f}",
            "",
            "Per-Phase Accuracy:",
            f"  Early (1-10):  {self.early_game_accuracy:.4f}",
            f"  Mid (11-25):   {self.mid_game_accuracy:.4f}",
            f"  Late (26+):    {self.late_game_accuracy:.4f}",
            "",
            "Auxiliary Head Accuracy:",
            f"  Item:  {self.item_accuracy:.4f}",
            f"  Speed: {self.speed_accuracy:.4f}",
            f"  Role:  {self.role_accuracy:.4f}",
            "",
            "Calibration (ECE):",
            f"  Policy: {self.policy_ece:.4f}",
            "",
            "Value Head:",
            f"  Accuracy: {self.value_accuracy:.4f}",
            f"  Brier:    {self.value_brier:.4f}",
        ]
        return "\n".join(lines)


def compute_ece(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Compute Expected Calibration Error.

    Args:
        probs: (N,) predicted probabilities for the chosen class
        labels: (N,) binary correctness indicators (1 if correct, 0 if not)
        n_bins: Number of calibration bins

    Returns:
        ECE value (lower is better).
    """
    if len(probs) == 0:
        return 0.0

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (probs > lo) & (probs <= hi)
        if mask.sum() == 0:
            continue
        bin_acc = labels[mask].mean()
        bin_conf = probs[mask].mean()
        ece += mask.sum() / len(probs) * abs(bin_acc - bin_conf)

    return float(ece)


@torch.no_grad()
def evaluate_model(
    model: BattleTransformer | nn.Module,
    test_data: list[dict[str, np.ndarray]],
    device: str | torch.device = "cpu",
    batch_size: int = 64,
) -> OfflineMetrics:
    """Run comprehensive offline evaluation.

    Args:
        model: Trained model to evaluate
        test_data: List of tensorized battle dicts
        device: Device to run on
        batch_size: Batch size for evaluation

    Returns:
        OfflineMetrics with all computed metrics.
    """
    device = torch.device(device)
    model = model.to(device)
    model.eval()

    dataset = TransformerSequenceDataset(test_data)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_transformer, num_workers=0,
    )

    # Accumulators
    all_probs: list[np.ndarray] = []      # predicted prob for correct action
    all_correct: list[bool] = []           # top-1 correct
    all_top3_correct: list[bool] = []
    all_nll: list[float] = []
    phase_correct: dict[str, list[bool]] = {"early": [], "mid": [], "late": []}

    # Auxiliary accumulators
    aux_correct: dict[str, list[bool]] = {"item": [], "speed": [], "role": []}

    # Value accumulators
    value_probs: list[float] = []
    value_labels: list[float] = []

    is_transformer = isinstance(model, BattleTransformer)

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}

        own_team = batch["own_team"]
        opp_team = batch["opponent_team"]
        field_feat = batch["field"]
        context = batch["context"]
        legal_mask = batch["legal_mask"]
        action = batch["action"]
        game_result = batch.get("game_result")
        seq_len = batch.get("seq_len")

        # Forward pass
        if is_transformer:
            output = model(
                own_team, opp_team, field_feat, context,
                legal_mask=legal_mask, seq_len=seq_len,
                return_auxiliary=True, return_value=True,
            )
            if isinstance(output, TransformerOutput):
                logits = output.policy_logits
            else:
                logits = output
        else:
            logits = model(own_team, opp_team, field_feat, context, legal_mask=legal_mask)

        # Get last step's action and turn number
        is_seq = action.dim() == 2
        if is_seq:
            batch_size_b = action.shape[0]
            if seq_len is not None:
                last_idx = (seq_len - 1).clamp(min=0)
            else:
                last_idx = torch.full((batch_size_b,), action.shape[1] - 1, device=device)

            flat_actions = torch.gather(
                action, dim=1, index=last_idx.unsqueeze(1)
            ).squeeze(1)

            # Get turn number from context (context[..., 0] = turn_number / 100)
            if context.dim() == 3:
                turn_nums = torch.gather(
                    context[:, :, 0], dim=1, index=last_idx.unsqueeze(1)
                ).squeeze(1) * 100
            else:
                turn_nums = context[:, 0] * 100
        else:
            flat_actions = action
            turn_nums = context[:, 0] * 100

        valid = flat_actions >= 0
        if not valid.any():
            continue

        valid_logits = logits[valid]
        valid_actions = flat_actions[valid]
        valid_turns = turn_nums[valid]

        # Softmax for probabilities
        probs = F.softmax(valid_logits, dim=-1)

        # Top-1 accuracy
        preds = valid_logits.argmax(dim=-1)
        correct = (preds == valid_actions)
        all_correct.extend(correct.cpu().tolist())

        # Top-3 accuracy
        top3 = valid_logits.topk(min(3, logits.shape[-1]), dim=-1).indices
        top3_correct = (top3 == valid_actions.unsqueeze(-1)).any(dim=-1)
        all_top3_correct.extend(top3_correct.cpu().tolist())

        # NLL
        nll = F.cross_entropy(valid_logits, valid_actions, reduction="none")
        all_nll.extend(nll.cpu().tolist())

        # Predicted probability of correct action (for ECE)
        correct_probs = probs[range(len(valid_actions)), valid_actions]
        all_probs.extend(correct_probs.float().cpu().tolist())

        # Per-phase breakdown
        for i in range(len(valid_turns)):
            turn = valid_turns[i].item()
            c = correct[i].item()
            if turn <= 10:
                phase_correct["early"].append(c)
            elif turn <= 25:
                phase_correct["mid"].append(c)
            else:
                phase_correct["late"].append(c)

        # Auxiliary head metrics
        if is_transformer and isinstance(output, TransformerOutput) and output.auxiliary_preds is not None:
            for head_name, target_key, short in [
                ("item_logits", "item_targets", "item"),
                ("speed_logits", "speed_targets", "speed"),
                ("role_logits", "role_targets", "role"),
            ]:
                if head_name in output.auxiliary_preds and target_key in batch:
                    pred = output.auxiliary_preds[head_name]  # (batch, 6, classes)
                    target = batch[target_key]
                    if target.dim() == 3 and seq_len is not None:
                        li = (seq_len - 1).clamp(min=0)
                        target = torch.gather(
                            target, dim=1,
                            index=li.unsqueeze(1).unsqueeze(2).expand(-1, 1, target.shape[2]),
                        ).squeeze(1)
                    elif target.dim() == 3:
                        target = target[:, -1]

                    flat_p = pred.reshape(-1, pred.shape[-1])
                    flat_t = target.reshape(-1)
                    valid_aux = flat_t >= 0
                    if valid_aux.any():
                        aux_preds_val = flat_p[valid_aux].argmax(dim=-1)
                        aux_c = (aux_preds_val == flat_t[valid_aux]).cpu().tolist()
                        aux_correct[short].extend(aux_c)

        # Value head metrics
        if (is_transformer and isinstance(output, TransformerOutput)
                and output.value_logits is not None and game_result is not None):
            vl = output.value_logits
            if game_result.dim() == 2 and seq_len is not None:
                li = (seq_len - 1).clamp(min=0)
                gr = torch.gather(game_result, dim=1, index=li.unsqueeze(1)).squeeze(1)
            elif game_result.dim() == 2:
                gr = game_result[:, -1]
            else:
                gr = game_result

            valid_val = (gr != 0.5)
            if valid_val.any():
                vp = torch.sigmoid(vl[valid_val])
                value_probs.extend(vp.cpu().tolist())
                value_labels.extend(gr[valid_val].cpu().tolist())

    # Compute final metrics
    metrics = OfflineMetrics()
    metrics.num_examples = len(all_correct)
    metrics.num_battles = len(test_data)

    if all_correct:
        metrics.top1_accuracy = np.mean(all_correct)
        metrics.top3_accuracy = np.mean(all_top3_correct)
        metrics.action_nll = np.mean(all_nll)

        # ECE
        metrics.policy_ece = compute_ece(
            np.array(all_probs),
            np.array(all_correct, dtype=np.float64),
        )

    # Per-phase
    for phase_name, key in [
        ("early_game_accuracy", "early"),
        ("mid_game_accuracy", "mid"),
        ("late_game_accuracy", "late"),
    ]:
        if phase_correct[key]:
            setattr(metrics, phase_name, np.mean(phase_correct[key]))

    # Auxiliary
    for name in ["item", "speed", "role"]:
        if aux_correct[name]:
            setattr(metrics, f"{name}_accuracy", np.mean(aux_correct[name]))

    # Value
    if value_probs:
        vp_arr = np.array(value_probs)
        vl_arr = np.array(value_labels)
        metrics.value_accuracy = np.mean((vp_arr > 0.5) == (vl_arr > 0.5))
        metrics.value_brier = np.mean((vp_arr - vl_arr) ** 2)

    return metrics
