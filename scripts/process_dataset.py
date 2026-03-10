#!/usr/bin/env python3
"""Process raw replay files into tensorized training data.

Reads .json.lz4 files from data/raw/, parses them into observations,
tensorizes, and saves processed data with train/val/test splits.

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

    from src.data.dataset import create_splits, save_processed_battles
    from src.data.priors import build_priors_from_battles
    from src.data.replay_parser import load_battles_from_directory
    from src.data.tensorizer import BattleVocabularies

    # Load battles
    logger.info(f"Loading battles from {args.input_dir}...")
    battles = load_battles_from_directory(
        args.input_dir,
        max_battles=args.max_battles,
    )
    logger.info(f"Loaded {len(battles)} valid battles")

    if not battles:
        logger.error("No valid battles found!")
        sys.exit(1)

    # Build vocabularies and process
    vocabs = BattleVocabularies()
    logger.info("Processing battles...")
    metadata = save_processed_battles(
        battles, args.output_dir, vocabs, max_turns=args.max_turns,
    )

    # Build and save priors
    logger.info("Building metagame priors...")
    priors = build_priors_from_battles(battles)
    priors.save(Path(args.output_dir) / "priors.json")

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
