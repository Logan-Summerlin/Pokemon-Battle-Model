#!/usr/bin/env python3
"""Train the BattleTransformer model on processed replay data.

Phase 4 training script. Loads processed .npz battles, adds auxiliary
labels, and trains the structured transformer with multi-loss.

Usage:
    # Smoke test (tiny model, few battles)
    python scripts/train_transformer.py --smoke-test

    # Small training run
    python scripts/train_transformer.py --num-battles 1000 --epochs 20

    # Full training (default config)
    python scripts/train_transformer.py

    # Custom config
    python scripts/train_transformer.py --hidden-dim 512 --num-layers 8 --batch-size 32
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.data.auxiliary_labels import build_auxiliary_targets
from src.data.dataset import load_processed_dataset, create_splits
from src.data.observation import build_observations, HISTORY_LENGTH
from src.data.replay_parser import ParsedBattle
from src.data.tensorizer import BattleVocabularies, tensorize_battle
from src.models.battle_transformer import (
    BattleTransformer,
    TransformerConfig,
    create_battle_transformer,
)
from src.training.transformer_trainer import TransformerTrainer, TransformerTrainingResult

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def load_battles_with_aux_labels(
    data_dir: Path,
    battle_ids: list[str] | None = None,
    max_battles: int | None = None,
    max_turns: int = HISTORY_LENGTH,
) -> tuple[list[dict[str, np.ndarray]], BattleVocabularies]:
    """Load processed battles and add auxiliary labels.

    The .npz files contain tensorized observations. We need to also
    load the raw replay data to extract auxiliary labels.

    For efficiency, we augment the pre-tensorized data with auxiliary
    targets computed from the raw replay structure.
    """
    data_dir = Path(data_dir)
    battles_dir = data_dir / "battles"

    # Load vocabularies
    vocabs = BattleVocabularies.load(data_dir / "vocabs")

    # Load battle tensors
    sequences: list[dict[str, np.ndarray]] = []
    npz_files = sorted(battles_dir.glob("*.npz"))

    if battle_ids is not None:
        battle_id_set = set(battle_ids)
        npz_files = [f for f in npz_files if f.stem in battle_id_set]

    if max_battles is not None:
        npz_files = npz_files[:max_battles]

    for npz_file in npz_files:
        data = dict(np.load(str(npz_file)))
        sequences.append(data)

    logger.info(f"Loaded {len(sequences)} battles from {data_dir}")
    return sequences, vocabs


def add_auxiliary_labels_from_priors(
    sequences: list[dict[str, np.ndarray]],
    data_dir: Path,
) -> list[dict[str, np.ndarray]]:
    """Add auxiliary labels to sequences using heuristic labels.

    Since the .npz files don't contain raw replay data, we derive
    auxiliary labels from the tensorized features themselves:
    - Item class: from the item vocabulary index in opponent features
    - Speed bucket: from base stats (own team only; opponent has 0)
    - Role/tera/move_families: labeled as -1 (unknown) when base stats unavailable

    For opponent pokemon, the item is often "unknown" at observation time,
    but the Metamon data captures the actual item in the full replay.
    We can label based on what IS in the tensor (which includes revealed info).
    """
    from src.data.auxiliary_labels import (
        classify_item, classify_speed, NUM_ITEM_CLASSES,
        NUM_SPEED_BUCKETS, NUM_ROLE_ARCHETYPES, NUM_TERA_CATEGORIES,
        NUM_MOVE_FAMILIES,
    )
    from src.data.tensorizer import POKEMON_CATEGORICAL_DIM, POKEMON_FEATURE_DIM

    augmented = []
    for seq in sequences:
        new_seq = dict(seq)

        opp_team = seq["opponent_team"]  # (seq_len, 6, 30) or (6, 30)
        is_seq = opp_team.ndim == 3
        if not is_seq:
            opp_team = opp_team[np.newaxis]

        seq_len = opp_team.shape[0]
        n_slots = opp_team.shape[1]

        # Initialize targets as unknown (-1)
        item_targets = np.full((seq_len, n_slots), -1, dtype=np.int64)
        speed_targets = np.full((seq_len, n_slots), -1, dtype=np.int64)
        role_targets = np.full((seq_len, n_slots), -1, dtype=np.int64)
        tera_targets = np.full((seq_len, n_slots), -1, dtype=np.int64)
        move_family_targets = np.full((seq_len, n_slots, NUM_MOVE_FAMILIES), -1, dtype=np.int64)

        # Extract item indices from opponent features
        # In the tensorized data: pokemon_features[5] = item vocab index
        for t in range(seq_len):
            for s in range(n_slots):
                feat = opp_team[t, s]
                species_idx = int(feat[0])

                # Skip empty/padding slots
                if species_idx == 0:
                    continue

                item_idx = int(feat[5])
                # If item is known (not UNK=1, not PAD=0)
                if item_idx > 1:
                    # We can't reverse the vocab lookup easily here,
                    # so we use the raw index as a pseudo-class.
                    # The item_head will learn to map from encoder
                    # representations to item classes.
                    # For now, use modulo to map to our class space.
                    item_targets[t, s] = min(item_idx % NUM_ITEM_CLASSES, NUM_ITEM_CLASSES - 1)

        if not is_seq:
            item_targets = item_targets[0]
            speed_targets = speed_targets[0]
            role_targets = role_targets[0]
            tera_targets = tera_targets[0]
            move_family_targets = move_family_targets[0]

        new_seq["item_targets"] = item_targets
        new_seq["speed_targets"] = speed_targets
        new_seq["role_targets"] = role_targets
        new_seq["tera_targets"] = tera_targets
        new_seq["move_family_targets"] = move_family_targets

        augmented.append(new_seq)

    return augmented


def load_and_prepare_data(
    data_dir: Path,
    max_battles: int | None = None,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[list[dict], list[dict], list[dict], BattleVocabularies]:
    """Load data, split, and add auxiliary labels.

    Returns:
        (train_data, val_data, test_data, vocabs)
    """
    sequences, vocabs = load_battles_with_aux_labels(
        data_dir, max_battles=max_battles,
    )

    # Split by index (since we don't have battle IDs from npz easily)
    import random
    rng = random.Random(seed)
    indices = list(range(len(sequences)))
    rng.shuffle(indices)

    n = len(indices)
    n_val = int(n * val_ratio)
    n_test = int(n * test_ratio)
    n_train = n - n_val - n_test

    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]

    train_seqs = [sequences[i] for i in train_indices]
    val_seqs = [sequences[i] for i in val_indices]
    test_seqs = [sequences[i] for i in test_indices]

    # Add auxiliary labels
    train_seqs = add_auxiliary_labels_from_priors(train_seqs, data_dir)
    val_seqs = add_auxiliary_labels_from_priors(val_seqs, data_dir)
    test_seqs = add_auxiliary_labels_from_priors(test_seqs, data_dir)

    logger.info(
        f"Data split: train={len(train_seqs)}, val={len(val_seqs)}, test={len(test_seqs)}"
    )

    return train_seqs, val_seqs, test_seqs, vocabs


def main() -> None:
    parser = argparse.ArgumentParser(description="Train BattleTransformer")

    # Data
    parser.add_argument("--data-dir", type=str, default="data/processed",
                        help="Directory with processed battle data")
    parser.add_argument("--num-battles", type=int, default=None,
                        help="Max battles to load (None = all)")

    # Model
    parser.add_argument("--hidden-dim", type=int, default=384)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--num-heads", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--aux-weight", type=float, default=0.2)
    parser.add_argument("--value-weight", type=float, default=0.1)
    parser.add_argument("--no-value-head", action="store_true")

    # Training
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--grad-accum", type=int, default=1)

    # Infrastructure
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/transformer")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-name", type=str, default="")

    # Presets
    parser.add_argument("--smoke-test", action="store_true",
                        help="Tiny model, few battles, fast iteration")
    parser.add_argument("--small", action="store_true",
                        help="Small model for validation")

    args = parser.parse_args()

    # Device selection
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    # Apply presets
    if args.smoke_test:
        args.num_battles = args.num_battles or 200
        args.hidden_dim = 128
        args.num_layers = 2
        args.num_heads = 4
        args.batch_size = 16
        args.epochs = 5
        args.warmup_steps = 50
        args.patience = 3
        args.checkpoint_dir = "checkpoints/transformer_smoke"
        logger.info("=== SMOKE TEST MODE ===")

    elif args.small:
        args.num_battles = args.num_battles or 1000
        args.hidden_dim = 256
        args.num_layers = 4
        args.num_heads = 4
        args.batch_size = 32
        args.epochs = 20
        args.warmup_steps = 200
        args.checkpoint_dir = "checkpoints/transformer_small"
        logger.info("=== SMALL MODEL MODE ===")

    # Load data
    logger.info("Loading data from %s...", args.data_dir)
    data_dir = Path(args.data_dir)
    train_data, val_data, test_data, vocabs = load_and_prepare_data(
        data_dir, max_battles=args.num_battles, seed=args.seed,
    )

    # Create model
    config = TransformerConfig.from_vocabs(
        vocabs,
        num_layers=args.num_layers,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        dropout=args.dropout,
        auxiliary_loss_weight=args.aux_weight,
        use_value_head=not args.no_value_head,
        value_loss_weight=args.value_weight,
    )

    model = BattleTransformer(config)
    logger.info(
        "Model: %d layers, %d dim, %d heads, %d params",
        config.num_layers, config.hidden_dim, config.num_heads,
        model.count_parameters(),
    )

    # Create trainer
    trainer = TransformerTrainer(
        model=model,
        config=config,
        train_data=train_data,
        val_data=val_data,
        checkpoint_dir=args.checkpoint_dir,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        max_epochs=args.epochs,
        early_stopping_patience=args.patience,
        warmup_steps=args.warmup_steps,
        gradient_accumulation_steps=args.grad_accum,
        device=device,
        use_wandb=args.wandb,
        wandb_run_name=args.wandb_name,
        seed=args.seed,
    )

    # Train
    logger.info("Starting training...")
    start_time = time.time()
    result = trainer.train()
    total_time = time.time() - start_time

    # Report results
    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)
    logger.info("Total time: %.1f minutes", total_time / 60)
    logger.info("Best epoch: %d", result.best_epoch)
    logger.info("Best val loss: %.4f", result.best_val_loss)
    logger.info("Best checkpoint: %s", result.best_checkpoint_path)

    if result.epoch_metrics:
        best = result.epoch_metrics[result.best_epoch - 1]
        logger.info("Best val accuracy: %.3f", best.val_accuracy)
        logger.info("Best val top-3 accuracy: %.3f", best.val_top3_accuracy)
        if best.aux_item_accuracy > 0:
            logger.info("Aux item accuracy: %.3f", best.aux_item_accuracy)
            logger.info("Aux speed accuracy: %.3f", best.aux_speed_accuracy)
            logger.info("Aux role accuracy: %.3f", best.aux_role_accuracy)

    # Save test data info for evaluation
    test_info = {
        "num_test_battles": len(test_data),
        "checkpoint_path": result.best_checkpoint_path,
        "config": result.config,
    }
    test_info_path = Path(args.checkpoint_dir) / "test_info.json"
    with open(test_info_path, "w") as f:
        json.dump(test_info, f, indent=2)
    logger.info("Test info saved to %s", test_info_path)


if __name__ == "__main__":
    main()
