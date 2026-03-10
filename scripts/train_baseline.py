#!/usr/bin/env python3
"""Train a simple BC baseline model (MLP or GRU) on processed battle data.

Usage:
    python scripts/train_baseline.py --data-dir data/processed --arch mlp
    python scripts/train_baseline.py --data-dir data/processed --arch gru --sequence-mode

This is Phase 3, Step 3.3 of the implementation plan.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from src.data.dataset import load_processed_dataset
from src.data.tensorizer import BattleVocabularies
from src.models.baseline_mlp import create_baseline_model
from src.training.bc_trainer import BCTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _flatten_sequences(
    sequences: list[dict[str, np.ndarray]],
) -> list[dict[str, np.ndarray]]:
    """Flatten battle sequences into individual turns for MLP training.

    Each sequence has shape (seq_len, ...) - we split into individual turns.
    Only include turns with valid actions (action >= 0).
    """
    turns: list[dict[str, np.ndarray]] = []
    for seq in sequences:
        seq_len = int(seq.get("seq_len", seq["own_team"].shape[0]))
        for t in range(seq_len):
            action = int(seq["action"][t])
            if action < 0:
                continue  # Skip turns without actions

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


def main() -> None:
    parser = argparse.ArgumentParser(description="Train BC baseline model")
    parser.add_argument(
        "--data-dir", type=str, default="data/processed",
        help="Directory with processed battle data",
    )
    parser.add_argument(
        "--train-split", type=str, default=None,
        help="Path to train split manifest (JSON list of battle IDs)",
    )
    parser.add_argument(
        "--val-split", type=str, default=None,
        help="Path to val split manifest",
    )
    parser.add_argument(
        "--arch", type=str, default="mlp", choices=["mlp", "gru"],
        help="Model architecture",
    )
    parser.add_argument(
        "--hidden-dims", type=str, default="512,256",
        help="Hidden layer dimensions (comma-separated)",
    )
    parser.add_argument(
        "--gru-hidden-dim", type=int, default=256,
        help="GRU hidden dimension",
    )
    parser.add_argument(
        "--gru-layers", type=int, default=1,
        help="Number of GRU layers",
    )
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument(
        "--sequence-mode", action="store_true",
        help="Use sequence-level training (for GRU)",
    )
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/baseline")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--wandb", action="store_true", help="Enable W&B logging")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Device selection
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    logger.info("Using device: %s", device)

    # Load data
    logger.info("Loading data from %s...", args.data_dir)
    data_dir = Path(args.data_dir)

    # Load all processed sequences
    all_sequences, vocabs = load_processed_dataset(data_dir)
    logger.info("Loaded %d battle sequences", len(all_sequences))

    if not all_sequences:
        logger.error("No data loaded! Check data directory: %s", args.data_dir)
        sys.exit(1)

    # Split data
    if args.train_split and args.val_split:
        # Use provided split manifests
        with open(args.train_split) as f:
            train_ids = set(json.load(f))
        with open(args.val_split) as f:
            val_ids = set(json.load(f))

        # Need to filter sequences by ID - but sequences don't store IDs directly
        # Load separately
        train_data, _ = load_processed_dataset(data_dir, split_file=args.train_split)
        val_data, _ = load_processed_dataset(data_dir, split_file=args.val_split)
    else:
        # Default 80/20 split
        np.random.seed(args.seed)
        indices = np.random.permutation(len(all_sequences))
        split_idx = int(0.8 * len(indices))
        train_indices = indices[:split_idx]
        val_indices = indices[split_idx:]

        train_data = [all_sequences[i] for i in train_indices]
        val_data = [all_sequences[i] for i in val_indices]

    logger.info("Train: %d sequences, Val: %d sequences", len(train_data), len(val_data))

    # For MLP: flatten sequences into individual turns
    if not args.sequence_mode:
        logger.info("Flattening sequences to individual turns for MLP training...")
        train_data = _flatten_sequences(train_data)
        val_data = _flatten_sequences(val_data)
        logger.info("Train: %d turns, Val: %d turns", len(train_data), len(val_data))

    # Create model
    hidden_dims = [int(x) for x in args.hidden_dims.split(",")]
    model = create_baseline_model(
        architecture=args.arch,
        hidden_dims=hidden_dims,
        gru_hidden_dim=args.gru_hidden_dim,
        gru_num_layers=args.gru_layers,
        dropout=args.dropout,
    )

    param_count = sum(p.numel() for p in model.parameters())
    logger.info(
        "Created %s model with %d parameters (%.2f MB)",
        type(model).__name__, param_count, param_count * 4 / 1e6,
    )

    # Train
    trainer = BCTrainer(
        model=model,
        train_data=train_data,
        val_data=val_data,
        checkpoint_dir=args.checkpoint_dir,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        early_stopping_patience=args.patience,
        sequence_mode=args.sequence_mode,
        device=device,
        use_wandb=args.wandb,
        seed=args.seed,
    )

    start = time.time()
    result = trainer.train()
    elapsed = time.time() - start

    # Report
    logger.info("=" * 60)
    logger.info("Training complete!")
    logger.info("  Total epochs: %d", result.total_epochs)
    logger.info("  Best epoch: %d", result.best_epoch)
    logger.info("  Best val loss: %.4f", result.best_val_loss)
    logger.info("  Final train accuracy: %.3f", result.final_train_accuracy)
    logger.info("  Final val accuracy: %.3f", result.final_val_accuracy)
    logger.info("  Best checkpoint: %s", result.best_checkpoint_path)
    logger.info("  Total training time: %.1f seconds", elapsed)
    logger.info("=" * 60)

    # Save training summary
    summary = {
        "architecture": args.arch,
        "hidden_dims": hidden_dims,
        "total_epochs": result.total_epochs,
        "best_epoch": result.best_epoch,
        "best_val_loss": result.best_val_loss,
        "final_train_accuracy": result.final_train_accuracy,
        "final_val_accuracy": result.final_val_accuracy,
        "best_checkpoint": result.best_checkpoint_path,
        "training_time_seconds": elapsed,
        "device": device,
        "param_count": param_count,
        "epoch_history": [
            {
                "epoch": m.epoch,
                "train_loss": m.train_loss,
                "val_loss": m.val_loss,
                "train_acc": m.train_accuracy,
                "val_acc": m.val_accuracy,
                "val_top3_acc": m.val_top3_accuracy,
            }
            for m in result.epoch_metrics
        ],
    }

    summary_path = Path(args.checkpoint_dir) / "training_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Training summary saved to %s", summary_path)


if __name__ == "__main__":
    main()
