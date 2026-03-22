#!/usr/bin/env python3
"""AutoResearch Experiment Launcher.

Wraps scripts/train_phase4.py with:
- Automatic experiment registration
- Budget enforcement (max epochs, max wall-time)
- Result extraction and leaderboard update
- Parent comparison (delta from parent experiment)

Usage:
    python Autoresearch/run_experiment.py \
        --name "window_size_5" \
        --parent anchor \
        --tier 1 \
        --config-override max_window=5 \
        --budget-epochs 10 \
        --budget-minutes 30
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAIN_SCRIPT = PROJECT_ROOT / "scripts" / "train_phase4.py"
REGISTRY_PATH = PROJECT_ROOT / "Autoresearch" / "experiment_registry.json"
RESULTS_DIR = PROJECT_ROOT / "Autoresearch" / "results"
NOTES_DIR = PROJECT_ROOT / "Autoresearch" / "notes"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Default P8-Lean config (anchor baseline)
DEFAULT_CONFIG = {
    "num_layers": 3,
    "hidden_dim": 224,
    "num_heads": 4,
    "ffn_multiplier": 3,
    "species_embedding_dim": 48,
    "move_embedding_dim": 24,
    "item_embedding_dim": 16,
    "ability_embedding_dim": 16,
    "type_embedding_dim": 12,
    "max_window": 2,
    "aux_weight": 0.2,
    "no_value_head": True,
    "prune_dead_features": True,
    "dropout": 0.1,
    "batch_size": 64,
    "epochs": 30,
    "lr": 1e-4,
    "weight_decay": 0.01,
    "warmup_steps": 300,
    "grad_accum": 1,
    "num_battles": 50000,
    "seed": 42,
}

# Tier budget limits
TIER_BUDGETS = {
    1: {"max_epochs": 10, "max_minutes": 30},
    2: {"max_epochs": 30, "max_minutes": 120},
    3: {"max_epochs": 50, "max_minutes": 240},
}


def load_registry() -> list[dict]:
    """Load experiment registry."""
    if REGISTRY_PATH.exists():
        with open(REGISTRY_PATH) as f:
            return json.load(f)
    return []


def save_registry(registry: list[dict]) -> None:
    """Save experiment registry."""
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REGISTRY_PATH, "w") as f:
        json.dump(registry, f, indent=2)


def find_experiment(registry: list[dict], name: str) -> dict | None:
    """Find an experiment by name or ID."""
    for exp in registry:
        if exp.get("name") == name or exp.get("experiment_id") == name:
            return exp
    return None


def next_experiment_id(registry: list[dict]) -> str:
    """Generate next experiment ID."""
    max_num = 0
    for exp in registry:
        eid = exp.get("experiment_id", "")
        parts = eid.split("-")
        if len(parts) >= 2:
            try:
                max_num = max(max_num, int(parts[-1]))
            except ValueError:
                pass
    return f"AR-{max_num + 1:03d}"


def parse_config_overrides(overrides: list[str] | None) -> dict:
    """Parse key=value config overrides."""
    if not overrides:
        return {}
    result = {}
    for item in overrides:
        if "=" not in item:
            logger.warning(f"Ignoring malformed override: {item}")
            continue
        key, value = item.split("=", 1)
        # Try to parse as number
        try:
            value = int(value)
        except ValueError:
            try:
                value = float(value)
            except ValueError:
                pass
        result[key.strip()] = value
    return result


def build_train_command(config: dict, checkpoint_dir: str, report_path: str) -> list[str]:
    """Build the train_phase4.py command from config."""
    cmd = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--mode", "full",
        "--data-dir", config.get("data_dir", "data/processed"),
        "--num-battles", str(config["num_battles"]),
        "--num-layers", str(config["num_layers"]),
        "--hidden-dim", str(config["hidden_dim"]),
        "--num-heads", str(config["num_heads"]),
        "--ffn-multiplier", str(config["ffn_multiplier"]),
        "--species-embedding-dim", str(config["species_embedding_dim"]),
        "--move-embedding-dim", str(config["move_embedding_dim"]),
        "--item-embedding-dim", str(config["item_embedding_dim"]),
        "--ability-embedding-dim", str(config["ability_embedding_dim"]),
        "--type-embedding-dim", str(config["type_embedding_dim"]),
        "--max-window", str(config["max_window"]),
        "--aux-weight", str(config["aux_weight"]),
        "--dropout", str(config["dropout"]),
        "--batch-size", str(config["batch_size"]),
        "--epochs", str(config["epochs"]),
        "--lr", str(config["lr"]),
        "--weight-decay", str(config["weight_decay"]),
        "--warmup-steps", str(config["warmup_steps"]),
        "--grad-accum", str(config["grad_accum"]),
        "--seed", str(config["seed"]),
        "--checkpoint-dir", checkpoint_dir,
        "--report-path", report_path,
    ]

    if config.get("no_value_head"):
        cmd.append("--no-value-head")
    if config.get("prune_dead_features"):
        cmd.append("--prune-dead-features")
    if config.get("num_workers") is not None:
        cmd.extend(["--num-workers", str(config["num_workers"])])

    return cmd


def extract_results(report_path: str) -> dict | None:
    """Extract key metrics from training report."""
    try:
        with open(report_path) as f:
            report = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None

    test = report.get("test_results", {})
    final = report.get("final_results", {})

    return {
        "test_top1_accuracy": test.get("test_accuracy"),
        "test_top3_accuracy": test.get("test_top3_accuracy"),
        "test_loss": test.get("test_loss"),
        "test_nll": test.get("test_nll") or test.get("test_policy_loss"),
        "test_ece": test.get("expected_calibration_error"),
        "per_action_accuracy": test.get("per_action_accuracy"),
        "best_epoch": final.get("best_epoch"),
        "best_val_loss": final.get("best_val_loss"),
        "wall_time_min": final.get("total_wall_time_min"),
        "parameter_count": report.get("training_config", {}).get("parameter_count"),
    }


def compare_to_parent(results: dict, parent: dict) -> dict:
    """Compute delta between experiment results and parent."""
    delta = {}
    for key in ["test_top1_accuracy", "test_top3_accuracy", "test_loss", "test_nll"]:
        if results.get(key) is not None and parent.get("metrics", {}).get(key) is not None:
            delta[key] = round(results[key] - parent["metrics"][key], 4)
    return delta


def main():
    parser = argparse.ArgumentParser(description="AutoResearch Experiment Launcher")
    parser.add_argument("--name", required=True, help="Experiment name")
    parser.add_argument("--parent", default="anchor", help="Parent experiment name/ID")
    parser.add_argument("--tier", type=int, choices=[1, 2, 3], default=1)
    parser.add_argument("--description", default="", help="Experiment description")
    parser.add_argument("--config-override", nargs="*", help="Config overrides as key=value")
    parser.add_argument("--budget-epochs", type=int, default=None, help="Override max epochs")
    parser.add_argument("--budget-minutes", type=int, default=None, help="Override max wall time")
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--dry-run", action="store_true", help="Print command without executing")
    args = parser.parse_args()

    registry = load_registry()

    # Check for duplicate name
    if find_experiment(registry, args.name):
        logger.error(f"Experiment '{args.name}' already exists in registry")
        return 1

    # Find parent
    parent = find_experiment(registry, args.parent)
    if not parent and args.parent != "anchor":
        logger.warning(f"Parent '{args.parent}' not found — using defaults")

    # Build config
    config = dict(DEFAULT_CONFIG)
    if parent and parent.get("config_changes"):
        config.update(parent["config_changes"])
    overrides = parse_config_overrides(args.config_override)
    config.update(overrides)
    config["data_dir"] = args.data_dir

    # Apply tier budget
    tier_budget = TIER_BUDGETS[args.tier]
    if args.budget_epochs:
        config["epochs"] = min(config["epochs"], args.budget_epochs)
    else:
        config["epochs"] = min(config["epochs"], tier_budget["max_epochs"])

    max_minutes = args.budget_minutes or tier_budget["max_minutes"]

    # Generate experiment ID
    experiment_id = next_experiment_id(registry)

    # Paths
    checkpoint_dir = str(PROJECT_ROOT / "checkpoints" / f"autoresearch_{args.name}")
    report_path = str(RESULTS_DIR / f"{args.name}_report.json")

    # Register experiment
    experiment = {
        "experiment_id": experiment_id,
        "parent_id": parent["experiment_id"] if parent else "none",
        "name": args.name,
        "description": args.description,
        "tier": args.tier,
        "config_changes": overrides,
        "full_config": config,
        "metrics": {},
        "delta_from_parent": {},
        "decision": None,
        "notes": "",
        "timestamp": datetime.now(UTC).isoformat(),
        "seed": config["seed"],
        "checkpoint_path": checkpoint_dir,
        "report_path": report_path,
        "budget": {
            "max_epochs": config["epochs"],
            "max_minutes": max_minutes,
        },
    }

    registry.append(experiment)
    save_registry(registry)
    logger.info(f"Registered experiment {experiment_id}: {args.name}")

    # Build command
    cmd = build_train_command(config, checkpoint_dir, report_path)

    logger.info(f"Command: {' '.join(cmd)}")

    if args.dry_run:
        print("\nDry run — command printed above. No training executed.")
        return 0

    # Run with wall-time budget
    logger.info(f"Starting training (budget: {config['epochs']} epochs, {max_minutes} min)...")
    start = time.time()

    try:
        result = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            timeout=max_minutes * 60,
        )
        success = result.returncode == 0
    except subprocess.TimeoutExpired:
        logger.warning(f"Training exceeded wall-time budget of {max_minutes} minutes")
        success = False

    elapsed_min = (time.time() - start) / 60

    # Extract results
    results = extract_results(report_path)
    if results:
        experiment["metrics"] = results
        experiment["metrics"]["wall_time_min"] = round(elapsed_min, 2)
        if parent:
            experiment["delta_from_parent"] = compare_to_parent(results, parent)

    # Update registry
    for i, exp in enumerate(registry):
        if exp["experiment_id"] == experiment_id:
            registry[i] = experiment
            break
    save_registry(registry)

    # Print summary
    print("\n" + "=" * 60)
    print(f"EXPERIMENT {experiment_id}: {args.name}")
    print("=" * 60)
    if results:
        acc = results.get("test_top1_accuracy")
        top3 = results.get("test_top3_accuracy")
        print(f"  Top-1 accuracy: {acc * 100:.2f}%" if acc else "  Top-1: N/A")
        print(f"  Top-3 accuracy: {top3 * 100:.2f}%" if top3 else "  Top-3: N/A")
        if experiment.get("delta_from_parent"):
            d = experiment["delta_from_parent"]
            for k, v in d.items():
                sign = "+" if v > 0 else ""
                print(f"  Delta {k}: {sign}{v:.4f}")
    else:
        print("  No results extracted (training may have failed)")
    print(f"  Wall time: {elapsed_min:.1f} min")
    print(f"  Status: {'SUCCESS' if success else 'FAILED'}")
    print("=" * 60)

    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
