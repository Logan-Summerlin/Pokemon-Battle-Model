#!/usr/bin/env python3
"""Process raw replay files into tensorized training data.

Reads .json.lz4 files from data/raw/, parses them into observations,
tensorizes, and saves processed data with train/val/test splits.

Uses streaming processing to avoid loading all battles into memory at once.

Usage:
    python scripts/process_dataset.py
    python scripts/process_dataset.py --input-dir data/raw --output-dir data/processed
    python scripts/process_dataset.py --max-battles 1000
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Process raw replays into training data")
    parser.add_argument(
        "--input-dir", type=str, default="data/raw",
        help="Directory containing raw .json.lz4 files",
    )
    parser.add_argument(
        "--output-dir", type=str, default="data/processed",
        help="Output directory for processed data",
    )
    parser.add_argument(
        "--splits-dir", type=str, default="data/splits",
        help="Output directory for split manifests",
    )
    parser.add_argument(
        "--max-battles", type=int, default=None,
        help="Maximum number of battles to process",
    )
    parser.add_argument(
        "--max-turns", type=int, default=20,
        help="Maximum turns per battle sequence",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for splits",
    )
    args = parser.parse_args()

    import numpy as np

    from src.data.dataset import create_splits
    from src.data.observation import build_observations
    from src.data.priors import MetagamePriors
    from src.data.replay_parser import iter_battles_from_directory
    from src.data.tensorizer import BattleVocabularies, tensorize_battle

    # Set up output directories
    output_dir = Path(args.output_dir)
    battles_dir = output_dir / "battles"
    battles_dir.mkdir(parents=True, exist_ok=True)

    vocabs = BattleVocabularies()
    priors = MetagamePriors()

    metadata: dict = {
        "num_battles": 0,
        "num_turns": 0,
        "num_wins": 0,
        "num_losses": 0,
        "avg_turns": 0.0,
        "elo_distribution": {},
        "battle_ids": [],
    }
    elo_counts: dict[str, int] = {}

    # Stream battles one at a time to avoid OOM
    logger.info(f"Processing battles from {args.input_dir} (streaming)...")
    for battle in iter_battles_from_directory(
        args.input_dir,
        max_battles=args.max_battles,
    ):
        # Build observations
        observations = build_observations(battle)
        if not observations:
            continue

        # Tensorize
        seq = tensorize_battle(
            observations, vocabs, build_vocab=True, max_turns=args.max_turns
        )
        if not seq:
            continue

        # Save as npz
        battle_file = battles_dir / f"{battle.battle_id}.npz"
        np.savez_compressed(str(battle_file), **seq)

        # Update priors
        priors.update_from_battle(battle)

        # Update metadata
        metadata["num_battles"] += 1
        metadata["num_turns"] += len(observations)
        metadata["battle_ids"].append(battle.battle_id)

        if battle.won:
            metadata["num_wins"] += 1
        else:
            metadata["num_losses"] += 1

        elo_bucket = str((battle.player_elo // 100) * 100)
        elo_counts[elo_bucket] = elo_counts.get(elo_bucket, 0) + 1

        if metadata["num_battles"] % 1000 == 0:
            logger.info(
                f"  Processed {metadata['num_battles']} battles "
                f"({metadata['num_turns']} turns)..."
            )

    if metadata["num_battles"] == 0:
        logger.error("No valid battles processed!")
        sys.exit(1)

    metadata["elo_distribution"] = elo_counts
    metadata["avg_turns"] = metadata["num_turns"] / metadata["num_battles"]

    # Save vocabularies
    vocabs.freeze_all()
    vocabs.save(output_dir / "vocabs")

    # Save metadata
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(
        f"Saved {metadata['num_battles']} battles to {output_dir} "
        f"({metadata['num_turns']} total turns)"
    )

    # Save priors
    logger.info("Saving metagame priors...")
    priors.save(output_dir / "priors.json")

    # Create splits
    logger.info("Creating train/val/test splits...")
    battle_ids = metadata["battle_ids"]
    splits = create_splits(
        battle_ids,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        seed=args.seed,
        output_dir=args.splits_dir,
    )

    # Print summary
    logger.info("\n=== Processing Summary ===")
    logger.info(f"Total battles: {metadata['num_battles']}")
    logger.info(f"Total turns: {metadata['num_turns']}")
    logger.info(f"Average turns per battle: {metadata['avg_turns']:.1f}")
    logger.info(f"Wins: {metadata['num_wins']}, Losses: {metadata['num_losses']}")
    logger.info(f"Train: {len(splits['train'])}, Val: {len(splits['val'])}, Test: {len(splits['test'])}")
    logger.info(f"Vocabulary sizes:")
    logger.info(f"  Species: {vocabs.species.size}")
    logger.info(f"  Moves: {vocabs.moves.size}")
    logger.info(f"  Items: {vocabs.items.size}")
    logger.info(f"  Abilities: {vocabs.abilities.size}")

    # Print Elo distribution
    logger.info("Elo distribution:")
    for bucket, count in sorted(metadata.get("elo_distribution", {}).items()):
        logger.info(f"  {bucket}+: {count}")

    # Print top species
    top_species = priors.get_top_species(20)
    logger.info("Top 20 species:")
    for species, count in top_species:
        logger.info(f"  {species}: {count}")


if __name__ == "__main__":
    main()
