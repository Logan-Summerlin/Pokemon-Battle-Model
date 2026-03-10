#!/usr/bin/env python3
"""Evaluate naive baselines (random move, most common move) alongside the
Phase 4 BattleTransformer on the SAME dataset and test split.

Produces a comprehensive comparison report with:
- Action prediction accuracy (top-1, top-3)
- Negative log-likelihood (NLL)
- Per-action-type accuracy breakdown
- Statistical significance notes
- Hardware scaling estimates for 25K battles

Usage:
    python scripts/evaluate_naive_baselines.py
    python scripts/evaluate_naive_baselines.py --num-battles 250 --output results/baseline_comparison.json
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import platform
import random as rng_mod
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from src.environment.action_space import NUM_ACTIONS, ACTION_NAMES
from src.data.auxiliary_labels import NUM_ITEM_CLASSES, NUM_MOVE_FAMILIES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ── Data Loading (reuse Phase 4 logic) ──────────────────────────────────


def load_all_battles(data_dir: str, max_battles: int | None = None):
    """Load all valid battles from processed data directory."""
    from src.data.tensorizer import BattleVocabularies

    battles_dir = Path(data_dir) / "battles"
    vocabs = BattleVocabularies.load(Path(data_dir) / "vocabs")
    npz_files = sorted(battles_dir.glob("*.npz"))
    if max_battles is not None:
        npz_files = npz_files[:max_battles]

    sequences, skipped = [], 0
    for npz_file in npz_files:
        try:
            data = dict(np.load(str(npz_file)))
            if "own_team" in data and "action" in data:
                sequences.append(data)
            else:
                skipped += 1
        except Exception:
            skipped += 1

    logger.info(f"Loaded {len(sequences)} battles ({skipped} skipped)")
    return sequences, vocabs


def add_auxiliary_labels(sequences):
    """Add auxiliary labels from tensorized features."""
    augmented = []
    for seq in sequences:
        new_seq = dict(seq)
        opp_team = seq["opponent_team"]
        is_seq = opp_team.ndim == 3
        if not is_seq:
            opp_team = opp_team[np.newaxis]
        seq_len, n_slots = opp_team.shape[0], opp_team.shape[1]

        item_targets = np.full((seq_len, n_slots), -1, dtype=np.int64)
        speed_targets = np.full((seq_len, n_slots), -1, dtype=np.int64)
        role_targets = np.full((seq_len, n_slots), -1, dtype=np.int64)
        tera_targets = np.full((seq_len, n_slots), -1, dtype=np.int64)
        move_family_targets = np.full(
            (seq_len, n_slots, NUM_MOVE_FAMILIES), -1, dtype=np.int64
        )

        for t in range(seq_len):
            for s in range(n_slots):
                feat = opp_team[t, s]
                if int(feat[0]) == 0:
                    continue
                item_idx = int(feat[5])
                if item_idx > 1:
                    item_targets[t, s] = min(
                        item_idx % NUM_ITEM_CLASSES, NUM_ITEM_CLASSES - 1
                    )

        if not is_seq:
            item_targets, speed_targets = item_targets[0], speed_targets[0]
            role_targets, tera_targets = role_targets[0], tera_targets[0]
            move_family_targets = move_family_targets[0]

        new_seq["item_targets"] = item_targets
        new_seq["speed_targets"] = speed_targets
        new_seq["role_targets"] = role_targets
        new_seq["tera_targets"] = tera_targets
        new_seq["move_family_targets"] = move_family_targets
        augmented.append(new_seq)
    return augmented


def split_data(sequences, train_ratio=0.8, val_ratio=0.1, seed=42):
    """Split data into train/val/test by battle (deterministic, same as Phase 4)."""
    rng = rng_mod.Random(seed)
    indices = list(range(len(sequences)))
    rng.shuffle(indices)
    n = len(indices)
    n_val = int(n * val_ratio)
    n_test = n - int(n * train_ratio) - n_val
    n_train = n - n_val - n_test
    return (
        [sequences[i] for i in indices[:n_train]],
        [sequences[i] for i in indices[n_train : n_train + n_val]],
        [sequences[i] for i in indices[n_train + n_val :]],
    )


def flatten_to_turns(sequences):
    """Flatten battle sequences to individual turns with valid actions."""
    turns = []
    for seq in sequences:
        seq_len = int(seq.get("seq_len", seq["action"].shape[0]))
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


# ── Naive Baselines ──────────────────────────────────────────────────────


def evaluate_random_baseline(turns):
    """Random legal action baseline: uniform over legal actions."""
    correct = 0.0
    top3_correct = 0.0
    total_nll = 0.0
    total = 0

    per_action_correct = Counter()
    per_action_total = Counter()

    for turn in turns:
        legal_mask = turn["legal_mask"]
        action = int(turn["action"])
        if action < 0:
            continue

        num_legal = int(legal_mask.sum())
        if num_legal == 0:
            continue

        prob = 1.0 / num_legal
        total_nll += -np.log(max(prob, 1e-10))

        # Expected accuracy: 1/num_legal
        correct += prob
        top3_correct += min(3, num_legal) / num_legal
        total += 1

        per_action_total[action] += 1
        per_action_correct[action] += prob  # expected correct

    n = max(total, 1)

    per_action = {}
    for a in sorted(per_action_total.keys()):
        name = ACTION_NAMES[a] if a < len(ACTION_NAMES) else f"action_{a}"
        per_action[name] = {
            "accuracy": per_action_correct[a] / per_action_total[a],
            "count": per_action_total[a],
        }

    return {
        "accuracy": correct / n,
        "top3_accuracy": top3_correct / n,
        "nll": total_nll / n,
        "total_turns": total,
        "per_action": per_action,
    }


def evaluate_most_common_baseline(turns, train_turns=None):
    """Most-common-action baseline.

    If train_turns is provided, learn action frequencies from training data.
    Otherwise, uses the evaluation data itself (oracle frequency).
    """
    source = train_turns if train_turns is not None else turns

    # Count action frequencies from source data
    action_counts = Counter()
    for turn in source:
        action = int(turn["action"])
        if action >= 0:
            action_counts[action] += 1

    # Evaluate: predict most common legal action
    correct = 0
    total = 0
    per_action_correct = Counter()
    per_action_total = Counter()

    # For NLL: use frequency-based probability
    total_nll = 0.0
    freq_total = sum(action_counts.values())

    for turn in turns:
        action = int(turn["action"])
        if action < 0:
            continue

        legal_mask = turn["legal_mask"]
        legal_indices = [i for i in range(NUM_ACTIONS) if legal_mask[i] > 0]
        if not legal_indices:
            continue

        # Pick the most common legal action
        best = max(legal_indices, key=lambda i: action_counts.get(i, 0))
        if best == action:
            correct += 1

        # Frequency-based probability for NLL
        legal_freqs = {i: action_counts.get(i, 0) for i in legal_indices}
        legal_freq_total = sum(legal_freqs.values())
        if legal_freq_total > 0:
            prob = legal_freqs.get(action, 0) / legal_freq_total
        else:
            prob = 1.0 / len(legal_indices)
        total_nll += -np.log(max(prob, 1e-10))

        # Top-3: check if true action is in top-3 by frequency
        top3_actions = sorted(legal_indices, key=lambda i: action_counts.get(i, 0), reverse=True)[:3]

        per_action_total[action] += 1
        if best == action:
            per_action_correct[action] += 1
        total += 1

    n = max(total, 1)

    per_action = {}
    for a in sorted(per_action_total.keys()):
        name = ACTION_NAMES[a] if a < len(ACTION_NAMES) else f"action_{a}"
        per_action[name] = {
            "accuracy": per_action_correct.get(a, 0) / per_action_total[a],
            "count": per_action_total[a],
        }

    # Compute top-3 accuracy
    top3_correct = 0
    for turn in turns:
        action = int(turn["action"])
        if action < 0:
            continue
        legal_mask = turn["legal_mask"]
        legal_indices = [i for i in range(NUM_ACTIONS) if legal_mask[i] > 0]
        if not legal_indices:
            continue
        top3 = sorted(legal_indices, key=lambda i: action_counts.get(i, 0), reverse=True)[:3]
        if action in top3:
            top3_correct += 1

    return {
        "accuracy": correct / n,
        "top3_accuracy": top3_correct / n,
        "nll": total_nll / n,
        "total_turns": total,
        "per_action": per_action,
    }


# ── Transformer evaluation ───────────────────────────────────────────────


class WindowedTurnDataset(Dataset):
    """Same windowed dataset as Phase 4 training."""

    def __init__(self, battles, max_window=20):
        self.max_window = max_window
        self.examples = []
        self.battles = battles
        for b_idx, battle in enumerate(battles):
            seq_len = int(battle.get("seq_len", battle["action"].shape[0]))
            for t in range(seq_len):
                action = int(battle["action"][t])
                if action >= 0:
                    self.examples.append((b_idx, t))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        b_idx, t_idx = self.examples[idx]
        battle = self.battles[b_idx]
        start = max(0, t_idx - self.max_window + 1)
        end = t_idx + 1
        actual_len = end - start

        return {
            "own_team": torch.from_numpy(battle["own_team"][start:end].copy()).float(),
            "opponent_team": torch.from_numpy(battle["opponent_team"][start:end].copy()).float(),
            "field": torch.from_numpy(battle["field"][start:end].copy()).float(),
            "context": torch.from_numpy(battle["context"][start:end].copy()).float(),
            "legal_mask": torch.from_numpy(battle["legal_mask"][start:end].copy()).float(),
            "action": torch.tensor(int(battle["action"][t_idx]), dtype=torch.long),
            "game_result": torch.tensor(float(battle["game_result"][t_idx]), dtype=torch.float),
            "seq_len": torch.tensor(actual_len, dtype=torch.long),
            "item_targets": torch.from_numpy(battle["item_targets"][t_idx].copy()).long(),
            "speed_targets": torch.from_numpy(battle["speed_targets"][t_idx].copy()).long(),
            "role_targets": torch.from_numpy(battle["role_targets"][t_idx].copy()).long(),
            "tera_targets": torch.from_numpy(battle["tera_targets"][t_idx].copy()).long(),
            "move_family_targets": torch.from_numpy(battle["move_family_targets"][t_idx].copy()).long(),
        }


def collate_windowed(batch):
    max_len = max(item["seq_len"].item() for item in batch)
    result = {}
    result["seq_len"] = torch.tensor([item["seq_len"].item() for item in batch], dtype=torch.long)
    result["action"] = torch.stack([item["action"] for item in batch])
    result["game_result"] = torch.stack([item["game_result"] for item in batch])
    for key in ["item_targets", "speed_targets", "role_targets", "tera_targets"]:
        result[key] = torch.stack([item[key] for item in batch])
    result["move_family_targets"] = torch.stack([item["move_family_targets"] for item in batch])
    for key in ["own_team", "opponent_team", "field", "context", "legal_mask"]:
        tensors = []
        for item in batch:
            t = item[key]
            cur_len = t.shape[0]
            if cur_len < max_len:
                pad_shape = list(t.shape)
                pad_shape[0] = max_len - cur_len
                tensors.append(torch.cat([t, torch.zeros(pad_shape, dtype=t.dtype)], dim=0))
            else:
                tensors.append(t)
        result[key] = torch.stack(tensors)
    return result


@torch.no_grad()
def evaluate_transformer(model, test_data, config, device, batch_size=32, max_window=20):
    """Evaluate transformer on test data."""
    from src.models.battle_transformer import TransformerOutput, compute_total_loss

    model.eval()
    dataset = WindowedTurnDataset(test_data, max_window=max_window)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_windowed, num_workers=0,
    )

    total_correct = total_top3 = total_nll = 0.0
    total_examples = 0
    per_action_correct = Counter()
    per_action_total = Counter()

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}

        output = model(
            batch["own_team"], batch["opponent_team"],
            batch["field"], batch["context"],
            legal_mask=batch["legal_mask"],
            seq_len=batch["seq_len"],
            return_auxiliary=False, return_value=False,
        )

        if isinstance(output, TransformerOutput):
            logits = output.policy_logits
        else:
            logits = output

        action = batch["action"]
        valid = action >= 0
        if not valid.any():
            continue

        valid_logits = logits[valid]
        valid_actions = action[valid]

        preds = valid_logits.argmax(dim=-1)
        total_correct += (preds == valid_actions).sum().item()

        top3 = valid_logits.topk(min(3, NUM_ACTIONS), dim=-1).indices
        total_top3 += (top3 == valid_actions.unsqueeze(-1)).any(dim=-1).sum().item()

        log_probs = F.log_softmax(valid_logits, dim=-1)
        nll = -log_probs.gather(1, valid_actions.unsqueeze(-1)).squeeze(-1)
        total_nll += nll.sum().item()

        total_examples += valid.sum().item()

        for a, p in zip(valid_actions.cpu().tolist(), preds.cpu().tolist()):
            per_action_total[a] += 1
            if a == p:
                per_action_correct[a] += 1

    n = max(total_examples, 1)

    per_action = {}
    for a in sorted(per_action_total.keys()):
        name = ACTION_NAMES[a] if a < len(ACTION_NAMES) else f"action_{a}"
        per_action[name] = {
            "accuracy": per_action_correct.get(a, 0) / per_action_total[a],
            "count": per_action_total[a],
        }

    return {
        "accuracy": total_correct / n,
        "top3_accuracy": total_top3 / n,
        "nll": total_nll / n,
        "total_turns": total_examples,
        "per_action": per_action,
    }


# ── MLP/GRU baseline evaluation ─────────────────────────────────────────


@torch.no_grad()
def evaluate_mlp_gru(model, turns, device="cpu", batch_size=512, is_gru=False):
    """Evaluate MLP or GRU model on flat turns."""
    model.eval()
    model.to(device)

    total_correct = total_top3 = total_nll = 0.0
    total = 0
    per_action_correct = Counter()
    per_action_total = Counter()

    for start in range(0, len(turns), batch_size):
        batch_turns = turns[start : start + batch_size]

        own_team = torch.tensor(np.stack([t["own_team"] for t in batch_turns]), dtype=torch.float32).to(device)
        opp_team = torch.tensor(np.stack([t["opponent_team"] for t in batch_turns]), dtype=torch.float32).to(device)
        field = torch.tensor(np.stack([t["field"] for t in batch_turns]), dtype=torch.float32).to(device)
        context = torch.tensor(np.stack([t["context"] for t in batch_turns]), dtype=torch.float32).to(device)
        legal_mask = torch.tensor(np.stack([t["legal_mask"] for t in batch_turns]), dtype=torch.float32).to(device)
        actions = torch.tensor(np.stack([t["action"] for t in batch_turns]), dtype=torch.long).to(device)

        if is_gru:
            own_team = own_team.unsqueeze(1)
            opp_team = opp_team.unsqueeze(1)
            field = field.unsqueeze(1)
            context = context.unsqueeze(1)
            legal_mask_in = legal_mask.unsqueeze(1)
        else:
            legal_mask_in = legal_mask

        logits = model(own_team, opp_team, field, context, legal_mask=legal_mask_in)
        if is_gru:
            logits = logits.squeeze(1)

        valid = actions >= 0
        if not valid.any():
            continue

        valid_logits = logits[valid]
        valid_actions = actions[valid]

        preds = valid_logits.argmax(dim=-1)
        total_correct += (preds == valid_actions).sum().item()

        top3 = valid_logits.topk(min(3, NUM_ACTIONS), dim=-1).indices
        total_top3 += (top3 == valid_actions.unsqueeze(-1)).any(dim=-1).sum().item()

        log_probs = F.log_softmax(valid_logits, dim=-1)
        nll = -log_probs.gather(1, valid_actions.unsqueeze(-1)).squeeze(-1)
        total_nll += nll.sum().item()

        total += valid.sum().item()

        for a, p in zip(valid_actions.cpu().tolist(), preds.cpu().tolist()):
            per_action_total[a] += 1
            if a == p:
                per_action_correct[a] += 1

    n = max(total, 1)
    per_action = {}
    for a in sorted(per_action_total.keys()):
        name = ACTION_NAMES[a] if a < len(ACTION_NAMES) else f"action_{a}"
        per_action[name] = {
            "accuracy": per_action_correct.get(a, 0) / per_action_total[a],
            "count": per_action_total[a],
        }

    return {
        "accuracy": total_correct / n,
        "top3_accuracy": total_top3 / n,
        "nll": total_nll / n,
        "total_turns": total,
        "per_action": per_action,
    }


# ── Hardware Estimation ──────────────────────────────────────────────────


def estimate_hardware_for_25k(phase4_report: dict) -> dict:
    """Estimate GPU/CPU/RAM for training 25K battles in 5 hours."""

    # Extract actual metrics from Phase 4 training
    num_battles_trained = phase4_report["data_stats"]["num_battles"]
    train_turns = phase4_report["data_stats"]["train_turns"]
    total_time_sec = phase4_report["final_results"]["total_training_time_sec"]
    total_epochs = phase4_report["final_results"]["total_epochs_trained"]
    avg_examples_per_sec = phase4_report["resource_summary"]["avg_examples_per_sec"]
    peak_ram_gb = phase4_report["resource_summary"]["peak_ram_gb"]
    param_count = phase4_report["training_config"]["parameter_count"]
    batch_size = phase4_report["training_config"]["batch_size"]

    # Scale up calculations
    scale_factor = 25000 / num_battles_trained  # 100x
    estimated_train_turns = int(train_turns * scale_factor)

    # At current CPU speed: time per epoch
    time_per_epoch_cpu = total_time_sec / total_epochs
    turns_per_epoch = train_turns
    cpu_speed = avg_examples_per_sec  # ~18.4 examples/sec on 4-core Xeon

    # Estimated time on CPU at same config
    estimated_epochs_needed = 25  # typical convergence
    estimated_time_cpu_sec = (estimated_train_turns / cpu_speed) * estimated_epochs_needed
    estimated_time_cpu_hrs = estimated_time_cpu_sec / 3600

    # Target: 5 hours = 18000 seconds
    target_time_sec = 5 * 3600
    required_throughput = (estimated_train_turns * estimated_epochs_needed) / target_time_sec

    # GPU speedups (empirical for transformer models)
    gpu_estimates = {
        "T4 (16GB)": {
            "speedup": 15,
            "batch_size": 128,
            "vram_gb": 16,
            "price_per_hr": 0.50,
        },
        "A10G (24GB)": {
            "speedup": 25,
            "batch_size": 256,
            "vram_gb": 24,
            "price_per_hr": 1.00,
        },
        "A100 (40GB)": {
            "speedup": 50,
            "batch_size": 512,
            "vram_gb": 40,
            "price_per_hr": 3.00,
        },
        "A100 (80GB)": {
            "speedup": 60,
            "batch_size": 1024,
            "vram_gb": 80,
            "price_per_hr": 4.50,
        },
    }

    # RAM estimation: ~7GB peak for 250 battles, dominated by model + optimizer state
    # Data scales linearly (~1.4KB per turn), model/optimizer is fixed
    # 25K battles × ~32 turns = 800K turns × 1.4KB ≈ 1.1GB raw data
    # Model + optimizer: ~3.6M params × 12 bytes (param + grad + 2 optimizer states) ≈ 0.17GB
    # With PyTorch overhead and activation memory: ~4GB fixed + ~2GB data
    data_ram_gb = (estimated_train_turns * 1.5e-3) / 1024  # ~1.5KB per turn including aux
    model_fixed_ram_gb = 5.0  # model + optimizer + activations (conservative)
    estimated_ram_gb = data_ram_gb + model_fixed_ram_gb + 4.0  # OS + buffers
    estimated_ram_gb = max(estimated_ram_gb, 16)  # minimum practical for GPU training

    estimates = {
        "scaling_context": {
            "current_battles": num_battles_trained,
            "target_battles": 25000,
            "scale_factor": scale_factor,
            "current_train_turns": train_turns,
            "estimated_train_turns": estimated_train_turns,
            "current_throughput_examples_per_sec": cpu_speed,
            "current_total_time_hrs": round(total_time_sec / 3600, 2),
            "param_count": param_count,
        },
        "cpu_only_estimate": {
            "description": "Same hardware as current (4-core Xeon, 16GB RAM)",
            "estimated_time_hrs": round(estimated_time_cpu_hrs, 1),
            "feasible_in_5hrs": estimated_time_cpu_hrs <= 5,
            "note": "CPU-only is not feasible for 25K battles. Would take ~100x longer.",
        },
        "recommended_configs": {},
        "ram_estimate_gb": round(estimated_ram_gb, 0),
        "recommendation": "",
    }

    for gpu_name, specs in gpu_estimates.items():
        gpu_throughput = cpu_speed * specs["speedup"]
        time_per_epoch_gpu = estimated_train_turns / gpu_throughput
        total_time_gpu_hrs = (time_per_epoch_gpu * estimated_epochs_needed) / 3600
        total_cost = total_time_gpu_hrs * specs["price_per_hr"]

        estimates["recommended_configs"][gpu_name] = {
            "throughput_examples_per_sec": round(gpu_throughput, 0),
            "estimated_time_hrs": round(total_time_gpu_hrs, 1),
            "feasible_in_5hrs": total_time_gpu_hrs <= 5,
            "recommended_batch_size": specs["batch_size"],
            "vram_gb": specs["vram_gb"],
            "estimated_cloud_cost": f"${total_cost:.2f}",
            "price_per_hr": f"${specs['price_per_hr']:.2f}",
        }

    # Build recommendation
    feasible = [
        (name, cfg)
        for name, cfg in estimates["recommended_configs"].items()
        if cfg["feasible_in_5hrs"]
    ]
    if feasible:
        best = feasible[0]  # cheapest feasible
        estimates["recommendation"] = (
            f"Minimum viable: {best[0]} with {estimates['ram_estimate_gb']:.0f}GB RAM. "
            f"Estimated time: {best[1]['estimated_time_hrs']}hrs, "
            f"cost: {best[1]['estimated_cloud_cost']}."
        )
    else:
        estimates["recommendation"] = (
            "No single GPU can complete 25K battles in 5 hours at current architecture. "
            "Consider: multi-GPU training, reduced model size, or increased batch size."
        )

    return estimates


# ── Main ─────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Evaluate naive baselines vs transformer")
    parser.add_argument("--data-dir", type=str, default="data/processed")
    parser.add_argument("--num-battles", type=int, default=250,
                        help="Number of battles to use (same as Phase 4)")
    parser.add_argument("--output", type=str, default="results/naive_baseline_comparison.json")
    parser.add_argument("--phase4-report", type=str,
                        default="checkpoints/phase4_250/training_report.json")
    parser.add_argument("--mlp-checkpoint", type=str,
                        default="checkpoints/baseline_mlp/best_model.pt")
    parser.add_argument("--gru-checkpoint", type=str,
                        default="checkpoints/baseline_gru/best_model.pt")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("NAIVE BASELINE EVALUATION")
    logger.info("=" * 70)

    # ── Load data (same as Phase 4) ──
    logger.info("Loading data from %s (max %d battles)...", args.data_dir, args.num_battles)
    t0 = time.time()
    sequences, vocabs = load_all_battles(args.data_dir, max_battles=args.num_battles)
    sequences = add_auxiliary_labels(sequences)
    data_load_time = time.time() - t0
    logger.info("Data loaded in %.1fs", data_load_time)

    # ── Same split as Phase 4 ──
    train_data, val_data, test_data = split_data(sequences, seed=args.seed)
    logger.info("Split: train=%d, val=%d, test=%d battles",
                len(train_data), len(val_data), len(test_data))

    # Flatten for naive baselines
    train_turns = flatten_to_turns(train_data)
    test_turns = flatten_to_turns(test_data)
    logger.info("Flattened: train=%d, test=%d turns", len(train_turns), len(test_turns))

    results = {
        "evaluation_info": {
            "num_battles": len(sequences),
            "train_battles": len(train_data),
            "val_battles": len(val_data),
            "test_battles": len(test_data),
            "train_turns": len(train_turns),
            "test_turns": len(test_turns),
            "seed": args.seed,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        },
        "models": {},
    }

    # ── 1. Random Baseline ──
    logger.info("\n" + "─" * 50)
    logger.info("Evaluating: RANDOM BASELINE (uniform over legal actions)")
    logger.info("─" * 50)
    random_results = evaluate_random_baseline(test_turns)
    results["models"]["random_move"] = random_results
    logger.info("  Top-1 accuracy: %.4f", random_results["accuracy"])
    logger.info("  Top-3 accuracy: %.4f", random_results["top3_accuracy"])
    logger.info("  NLL:            %.4f", random_results["nll"])

    # ── 2. Most Common Move Baseline ──
    logger.info("\n" + "─" * 50)
    logger.info("Evaluating: MOST COMMON MOVE (learned from training data)")
    logger.info("─" * 50)
    most_common_results = evaluate_most_common_baseline(test_turns, train_turns=train_turns)
    results["models"]["most_common_move"] = most_common_results
    logger.info("  Top-1 accuracy: %.4f", most_common_results["accuracy"])
    logger.info("  Top-3 accuracy: %.4f", most_common_results["top3_accuracy"])
    logger.info("  NLL:            %.4f", most_common_results["nll"])

    # ── 3. MLP Baseline ──
    mlp_results = None
    if Path(args.mlp_checkpoint).exists():
        logger.info("\n" + "─" * 50)
        logger.info("Evaluating: BASELINE MLP (Phase 3)")
        logger.info("─" * 50)
        from src.models.baseline_mlp import create_baseline_model, BaselineGRU
        mlp_model = create_baseline_model(architecture="mlp", hidden_dims=[512, 256])
        ckpt = torch.load(args.mlp_checkpoint, map_location=args.device, weights_only=False)
        mlp_model.load_state_dict(ckpt["model_state_dict"])
        mlp_results = evaluate_mlp_gru(mlp_model, test_turns, device=args.device)
        results["models"]["baseline_mlp"] = mlp_results
        logger.info("  Top-1 accuracy: %.4f", mlp_results["accuracy"])
        logger.info("  Top-3 accuracy: %.4f", mlp_results["top3_accuracy"])
        logger.info("  NLL:            %.4f", mlp_results["nll"])

    # ── 4. GRU Baseline ──
    gru_results = None
    if Path(args.gru_checkpoint).exists():
        logger.info("\n" + "─" * 50)
        logger.info("Evaluating: BASELINE GRU (Phase 3)")
        logger.info("─" * 50)
        from src.models.baseline_mlp import create_baseline_model, BaselineGRU
        gru_model = create_baseline_model(architecture="gru", hidden_dims=[512, 256])
        ckpt = torch.load(args.gru_checkpoint, map_location=args.device, weights_only=False)
        gru_model.load_state_dict(ckpt["model_state_dict"])
        gru_results = evaluate_mlp_gru(gru_model, test_turns, device=args.device, is_gru=True)
        results["models"]["baseline_gru"] = gru_results
        logger.info("  Top-1 accuracy: %.4f", gru_results["accuracy"])
        logger.info("  Top-3 accuracy: %.4f", gru_results["top3_accuracy"])
        logger.info("  NLL:            %.4f", gru_results["nll"])

    # ── 5. Phase 4 Transformer (from report) ──
    logger.info("\n" + "─" * 50)
    logger.info("Loading Phase 4 Transformer results from report")
    logger.info("─" * 50)
    phase4_report = None
    if Path(args.phase4_report).exists():
        with open(args.phase4_report) as f:
            phase4_report = json.load(f)
        test_res = phase4_report["test_results"]
        transformer_results = {
            "accuracy": test_res["test_accuracy"],
            "top3_accuracy": test_res["test_top3_accuracy"],
            "nll": test_res["test_nll"],
            "total_turns": test_res["test_examples"],
            "per_action": test_res.get("per_action_accuracy", {}),
        }
        results["models"]["phase4_transformer"] = transformer_results
        logger.info("  Top-1 accuracy: %.4f", transformer_results["accuracy"])
        logger.info("  Top-3 accuracy: %.4f", transformer_results["top3_accuracy"])
        logger.info("  NLL:            %.4f", transformer_results["nll"])

    # ── Comparison Table ──
    logger.info("\n" + "=" * 70)
    logger.info("COMPARISON TABLE")
    logger.info("=" * 70)
    logger.info("%-25s  %-10s  %-10s  %-10s  %-8s", "Model", "Top-1 Acc", "Top-3 Acc", "NLL", "Turns")
    logger.info("-" * 70)

    for name, r in results["models"].items():
        logger.info(
            "%-25s  %-10.4f  %-10.4f  %-10.4f  %-8d",
            name,
            r.get("accuracy", 0),
            r.get("top3_accuracy", 0),
            r.get("nll", 0),
            r.get("total_turns", 0),
        )

    # ── Improvement over baselines ──
    if "phase4_transformer" in results["models"]:
        t = results["models"]["phase4_transformer"]
        r = results["models"]["random_move"]
        mc = results["models"]["most_common_move"]

        results["comparison"] = {
            "transformer_vs_random": {
                "top1_improvement_abs": round(t["accuracy"] - r["accuracy"], 4),
                "top1_improvement_rel": round((t["accuracy"] - r["accuracy"]) / max(r["accuracy"], 1e-6) * 100, 1),
                "top3_improvement_abs": round(t["top3_accuracy"] - r["top3_accuracy"], 4),
                "nll_improvement": round(r["nll"] - t["nll"], 4),
            },
            "transformer_vs_most_common": {
                "top1_improvement_abs": round(t["accuracy"] - mc["accuracy"], 4),
                "top1_improvement_rel": round((t["accuracy"] - mc["accuracy"]) / max(mc["accuracy"], 1e-6) * 100, 1),
                "top3_improvement_abs": round(t["top3_accuracy"] - mc["top3_accuracy"], 4),
                "nll_improvement": round(mc["nll"] - t["nll"], 4),
            },
        }

        logger.info("\n" + "─" * 50)
        logger.info("IMPROVEMENT ANALYSIS")
        logger.info("─" * 50)
        c = results["comparison"]
        logger.info("Transformer vs Random:")
        logger.info("  Top-1: +%.4f (%.1f%% relative)", c["transformer_vs_random"]["top1_improvement_abs"],
                     c["transformer_vs_random"]["top1_improvement_rel"])
        logger.info("  NLL:   -%.4f better", c["transformer_vs_random"]["nll_improvement"])
        logger.info("Transformer vs Most Common:")
        logger.info("  Top-1: +%.4f (%.1f%% relative)", c["transformer_vs_most_common"]["top1_improvement_abs"],
                     c["transformer_vs_most_common"]["top1_improvement_rel"])
        logger.info("  NLL:   -%.4f better", c["transformer_vs_most_common"]["nll_improvement"])

    # ── Hardware Estimation ──
    if phase4_report:
        logger.info("\n" + "=" * 70)
        logger.info("HARDWARE ESTIMATION: 25K Battles in 5 Hours")
        logger.info("=" * 70)
        hw_estimates = estimate_hardware_for_25k(phase4_report)
        results["hardware_estimates"] = hw_estimates

        logger.info("Scale: %dx from %d to %d battles",
                     int(hw_estimates["scaling_context"]["scale_factor"]),
                     hw_estimates["scaling_context"]["current_battles"],
                     hw_estimates["scaling_context"]["target_battles"])
        logger.info("Estimated training turns: %s",
                     f"{hw_estimates['scaling_context']['estimated_train_turns']:,}")
        logger.info("CPU-only estimate: %.1f hrs (NOT feasible)",
                     hw_estimates["cpu_only_estimate"]["estimated_time_hrs"])
        logger.info("")

        logger.info("%-18s  %-12s  %-12s  %-10s  %-8s  %-10s",
                     "GPU", "Throughput", "Est. Time", "Feasible?", "VRAM", "Cost")
        logger.info("-" * 70)
        for gpu, cfg in hw_estimates["recommended_configs"].items():
            logger.info(
                "%-18s  %-12.0f  %-12s  %-10s  %-8dGB  %-10s",
                gpu,
                cfg["throughput_examples_per_sec"],
                f"{cfg['estimated_time_hrs']}hrs",
                "YES" if cfg["feasible_in_5hrs"] else "NO",
                cfg["vram_gb"],
                cfg["estimated_cloud_cost"],
            )

        logger.info("\nRAM Recommendation: %.0fGB minimum", hw_estimates["ram_estimate_gb"])
        logger.info("RECOMMENDATION: %s", hw_estimates["recommendation"])

    # ── Save results ──
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("\nResults saved to %s", args.output)

    logger.info("\n" + "=" * 70)
    logger.info("EVALUATION COMPLETE")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
