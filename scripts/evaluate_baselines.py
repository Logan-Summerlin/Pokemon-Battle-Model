#!/usr/bin/env python3
"""Evaluate baseline models on held-out replay data.

Computes offline metrics (action accuracy, top-3 accuracy, NLL)
for trained models and compares against random/heuristic baselines.

Usage:
    python scripts/evaluate_baselines.py --data-dir data/processed --checkpoint checkpoints/baseline/best_model.pt

This is part of Phase 3 evaluation.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F

from src.data.dataset import load_processed_dataset
from src.data.tensorizer import BattleVocabularies
from src.environment.action_space import NUM_ACTIONS, ACTION_NAMES
from src.models.baseline_mlp import BaselineMLP, BaselineGRU, create_baseline_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _flatten_sequences(
    sequences: list[dict[str, np.ndarray]],
) -> list[dict[str, np.ndarray]]:
    """Flatten battle sequences into individual turns."""
    turns: list[dict[str, np.ndarray]] = []
    for seq in sequences:
        seq_len = int(seq.get("seq_len", seq["own_team"].shape[0]))
        for t in range(seq_len):
            action = int(seq["action"][t])
            if action < 0:
                continue
            turn = {
                "own_team": seq["own_team"][t],
                "opponent_team": seq["opponent_team"][t],
                "field": seq["field"][t],
                "context": seq["context"][t],
                "legal_mask": seq["legal_mask"][t],
                "action": seq["action"][t],
                "game_result": seq["game_result"][t],
            }
            turns.append(turn)
    return turns


def evaluate_random_baseline(turns: list[dict[str, np.ndarray]]) -> dict[str, float]:
    """Evaluate random legal action baseline."""
    correct = 0
    top3_correct = 0
    total_nll = 0.0
    total = 0

    for turn in turns:
        legal_mask = turn["legal_mask"]
        action = int(turn["action"])
        if action < 0:
            continue

        num_legal = int(legal_mask.sum())
        if num_legal == 0:
            continue

        # Random uniform over legal actions
        prob = 1.0 / num_legal
        total_nll += -np.log(max(prob, 1e-10))

        # Random baseline accuracy: 1/num_legal
        correct += 1.0 / num_legal
        top3_correct += min(3, num_legal) / num_legal
        total += 1

    n = max(total, 1)
    return {
        "accuracy": correct / n,
        "top3_accuracy": top3_correct / n,
        "nll": total_nll / n,
        "total_turns": total,
    }


def evaluate_most_common_baseline(
    turns: list[dict[str, np.ndarray]],
) -> dict[str, float]:
    """Evaluate most-common-action baseline (always picks the most frequent legal action)."""
    # First pass: count action frequencies
    action_counts: Counter[int] = Counter()
    for turn in turns:
        action = int(turn["action"])
        if action >= 0:
            action_counts[action] += 1

    # Second pass: predict the most common legal action
    correct = 0
    total = 0

    for turn in turns:
        action = int(turn["action"])
        if action < 0:
            continue

        legal_mask = turn["legal_mask"]
        legal_indices = [i for i in range(NUM_ACTIONS) if legal_mask[i] > 0]

        if not legal_indices:
            continue

        # Pick the most globally common legal action
        best = max(legal_indices, key=lambda i: action_counts.get(i, 0))
        if best == action:
            correct += 1
        total += 1

    n = max(total, 1)
    return {
        "accuracy": correct / n,
        "total_turns": total,
    }


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    turns: list[dict[str, np.ndarray]],
    device: str = "cpu",
    batch_size: int = 512,
) -> dict[str, float]:
    """Evaluate a trained model on individual turns."""
    model.eval()
    model.to(device)

    correct = 0
    top3_correct = 0
    total_nll = 0.0
    total = 0

    # Per-action-type accuracy
    per_action_correct: dict[int, int] = {}
    per_action_total: dict[int, int] = {}

    # Process in batches
    for start in range(0, len(turns), batch_size):
        batch_turns = turns[start : start + batch_size]

        own_team = torch.tensor(
            np.stack([t["own_team"] for t in batch_turns]), dtype=torch.float32
        ).to(device)
        opp_team = torch.tensor(
            np.stack([t["opponent_team"] for t in batch_turns]), dtype=torch.float32
        ).to(device)
        field = torch.tensor(
            np.stack([t["field"] for t in batch_turns]), dtype=torch.float32
        ).to(device)
        context = torch.tensor(
            np.stack([t["context"] for t in batch_turns]), dtype=torch.float32
        ).to(device)
        legal_mask = torch.tensor(
            np.stack([t["legal_mask"] for t in batch_turns]), dtype=torch.float32
        ).to(device)
        actions = torch.tensor(
            np.stack([t["action"] for t in batch_turns]), dtype=torch.long
        ).to(device)

        # For GRU models, add a sequence dimension (treat each turn as length-1 seq)
        is_gru = isinstance(model, BaselineGRU)
        if is_gru:
            own_team = own_team.unsqueeze(1)
            opp_team = opp_team.unsqueeze(1)
            field = field.unsqueeze(1)
            context = context.unsqueeze(1)
            legal_mask_in = legal_mask.unsqueeze(1)
        else:
            legal_mask_in = legal_mask

        # Forward
        logits = model(own_team, opp_team, field, context, legal_mask=legal_mask_in)
        if is_gru:
            logits = logits.squeeze(1)  # Remove sequence dim

        # Filter valid actions
        valid = actions >= 0
        if not valid.any():
            continue

        valid_logits = logits[valid]
        valid_actions = actions[valid]
        valid_masks = legal_mask[valid]

        # Top-1 accuracy
        preds = valid_logits.argmax(dim=-1)
        correct += (preds == valid_actions).sum().item()

        # Top-3 accuracy
        top3 = valid_logits.topk(min(3, NUM_ACTIONS), dim=-1).indices
        top3_hit = (top3 == valid_actions.unsqueeze(-1)).any(dim=-1)
        top3_correct += top3_hit.sum().item()

        # NLL (with legal masking)
        log_probs = F.log_softmax(valid_logits, dim=-1)
        nll = -log_probs.gather(1, valid_actions.unsqueeze(-1)).squeeze(-1)
        total_nll += nll.sum().item()

        # Per-action stats
        for a, p in zip(valid_actions.cpu().tolist(), preds.cpu().tolist()):
            per_action_total[a] = per_action_total.get(a, 0) + 1
            if a == p:
                per_action_correct[a] = per_action_correct.get(a, 0) + 1

        total += valid.sum().item()

    n = max(total, 1)

    # Per-action breakdown
    per_action_acc = {}
    for action_idx in sorted(per_action_total.keys()):
        t = per_action_total[action_idx]
        c = per_action_correct.get(action_idx, 0)
        name = ACTION_NAMES[action_idx] if action_idx < len(ACTION_NAMES) else f"action_{action_idx}"
        per_action_acc[name] = {"accuracy": c / t, "count": t}

    return {
        "accuracy": correct / n,
        "top3_accuracy": top3_correct / n,
        "nll": total_nll / n,
        "total_turns": total,
        "per_action": per_action_acc,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate baseline models")
    parser.add_argument("--data-dir", type=str, default="data/processed")
    parser.add_argument("--test-split", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--arch", type=str, default="mlp", choices=["mlp", "gru"])
    parser.add_argument("--hidden-dims", type=str, default="512,256")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--output", type=str, default=None, help="Save results to JSON")
    args = parser.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    # Load data
    logger.info("Loading evaluation data from %s...", args.data_dir)
    sequences, vocabs = load_processed_dataset(
        args.data_dir, split_file=args.test_split
    )
    logger.info("Loaded %d battle sequences", len(sequences))

    # Flatten for per-turn evaluation
    turns = _flatten_sequences(sequences)
    logger.info("Flattened to %d turns", len(turns))

    if not turns:
        logger.error("No valid turns found!")
        sys.exit(1)

    results: dict[str, Any] = {}

    # Random baseline
    logger.info("Evaluating random baseline...")
    random_results = evaluate_random_baseline(turns)
    results["random"] = random_results
    logger.info(
        "  Random: acc=%.3f top3=%.3f nll=%.3f",
        random_results["accuracy"],
        random_results["top3_accuracy"],
        random_results["nll"],
    )

    # Most-common baseline
    logger.info("Evaluating most-common-action baseline...")
    common_results = evaluate_most_common_baseline(turns)
    results["most_common"] = common_results
    logger.info("  Most-common: acc=%.3f", common_results["accuracy"])

    # Trained model
    if args.checkpoint:
        logger.info("Loading model from %s...", args.checkpoint)
        hidden_dims = [int(x) for x in args.hidden_dims.split(",")]
        model = create_baseline_model(
            architecture=args.arch,
            hidden_dims=hidden_dims,
        )

        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        logger.info(
            "Loaded checkpoint from epoch %d (val_loss=%.4f)",
            checkpoint.get("epoch", "?"),
            checkpoint.get("val_loss", float("nan")),
        )

        logger.info("Evaluating trained model...")
        model_results = evaluate_model(model, turns, device=device, batch_size=args.batch_size)
        results["model"] = model_results
        logger.info(
            "  Model: acc=%.3f top3=%.3f nll=%.3f",
            model_results["accuracy"],
            model_results["top3_accuracy"],
            model_results["nll"],
        )

        # Per-action breakdown
        logger.info("  Per-action accuracy:")
        for action_name, stats in model_results["per_action"].items():
            logger.info(
                "    %-15s  acc=%.3f  (n=%d)",
                action_name, stats["accuracy"], stats["count"],
            )

    # Summary
    logger.info("=" * 60)
    logger.info("EVALUATION SUMMARY")
    logger.info("=" * 60)
    logger.info("  %-20s  %-8s  %-8s  %-8s", "Method", "Top-1", "Top-3", "NLL")
    logger.info("  " + "-" * 50)
    for method, r in results.items():
        logger.info(
            "  %-20s  %-8.3f  %-8.3f  %-8.3f",
            method,
            r.get("accuracy", 0),
            r.get("top3_accuracy", 0),
            r.get("nll", 0),
        )

    # Save results
    if args.output:
        # Remove non-serializable items
        serializable = {}
        for method, r in results.items():
            serializable[method] = {
                k: v for k, v in r.items()
                if isinstance(v, (int, float, str, dict))
            }
        with open(args.output, "w") as f:
            json.dump(serializable, f, indent=2)
        logger.info("Results saved to %s", args.output)


if __name__ == "__main__":
    main()
