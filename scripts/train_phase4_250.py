#!/usr/bin/env python3
"""Phase 4: Train BattleTransformer on 250 real replay battles.

This script:
1. Loads 250 pre-processed real replay battles from data/processed/battles/
2. Splits into train (200) / calibration-val (25) / test (25)
3. Trains the Phase 4 BattleTransformer with auxiliary hidden-info heads
4. Records detailed CPU, RAM, GPU, and time metrics per epoch
5. Evaluates on held-out test set with calibration analysis
6. Saves comprehensive training report

Usage:
    python scripts/train_phase4_250.py
    python scripts/train_phase4_250.py --epochs 30 --hidden-dim 256
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import os
import platform
import resource as resource_mod
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from src.data.tensorizer import BattleVocabularies
from src.data.auxiliary_labels import NUM_ITEM_CLASSES, NUM_MOVE_FAMILIES
from src.models.battle_transformer import (
    BattleTransformer,
    TransformerConfig,
    TransformerOutput,
    compute_total_loss,
    TOKENS_PER_STEP,
)
from src.environment.action_space import NUM_ACTIONS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ── Resource monitoring ──────────────────────────────────────────────────


def get_system_info() -> dict:
    """Collect static system information."""
    info: dict[str, Any] = {}

    # CPU info
    try:
        with open("/proc/cpuinfo") as f:
            cores_seen = set()
            for line in f:
                if "model name" in line:
                    info["cpu_model"] = line.split(":")[1].strip()
                if "processor" in line:
                    cores_seen.add(line.split(":")[1].strip())
            info["cpu_physical_cores"] = len(cores_seen)
    except Exception:
        info["cpu_model"] = platform.processor() or "unknown"
        info["cpu_physical_cores"] = 0

    info["cpu_count"] = os.cpu_count() or 0

    # RAM info
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if "MemTotal" in line:
                    kb = int(line.split()[1])
                    info["ram_total_gb"] = round(kb / 1024 / 1024, 2)
                elif "MemAvailable" in line:
                    kb = int(line.split()[1])
                    info["ram_available_gb"] = round(kb / 1024 / 1024, 2)
    except Exception:
        info["ram_total_gb"] = 0.0
        info["ram_available_gb"] = 0.0

    # GPU info
    info["gpu_name"] = "None"
    info["gpu_memory_gb"] = 0.0
    info["gpu_count"] = 0
    if torch.cuda.is_available():
        info["gpu_count"] = torch.cuda.device_count()
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_memory_gb"] = round(
            torch.cuda.get_device_properties(0).total_mem / 1024**3, 2
        )
        info["cuda_version"] = torch.version.cuda or "unknown"

    # System
    info["python_version"] = sys.version.split()[0]
    info["pytorch_version"] = torch.__version__
    info["cuda_available"] = torch.cuda.is_available()
    info["os_info"] = f"{platform.system()} {platform.release()}"
    info["architecture"] = platform.machine()

    return info


def get_resource_snapshot() -> dict:
    """Get current resource usage snapshot."""
    snapshot: dict[str, Any] = {}

    # RAM from /proc/self/status
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    snapshot["ram_used_gb"] = round(int(line.split()[1]) / 1024 / 1024, 4)
                elif line.startswith("VmPeak:"):
                    snapshot["ram_peak_gb"] = round(int(line.split()[1]) / 1024 / 1024, 4)
                elif line.startswith("VmSize:"):
                    snapshot["vm_size_gb"] = round(int(line.split()[1]) / 1024 / 1024, 4)
    except Exception:
        snapshot["ram_used_gb"] = 0.0
        snapshot["ram_peak_gb"] = 0.0

    # CPU usage
    try:
        ru = resource_mod.getrusage(resource_mod.RUSAGE_SELF)
        snapshot["cpu_user_time_sec"] = round(ru.ru_utime, 3)
        snapshot["cpu_sys_time_sec"] = round(ru.ru_stime, 3)
        snapshot["cpu_total_time_sec"] = round(ru.ru_utime + ru.ru_stime, 3)
        snapshot["max_rss_gb"] = round(ru.ru_maxrss / 1024 / 1024, 4)  # in KB on Linux
    except Exception:
        pass

    # CPU utilization from /proc/self/stat
    try:
        with open("/proc/self/stat") as f:
            fields = f.read().split()
            utime = int(fields[13])  # user mode jiffies
            stime = int(fields[14])  # kernel mode jiffies
            snapshot["proc_utime_jiffies"] = utime
            snapshot["proc_stime_jiffies"] = stime
    except Exception:
        pass

    # System-wide CPU load
    try:
        with open("/proc/loadavg") as f:
            parts = f.read().split()
            snapshot["load_avg_1min"] = float(parts[0])
            snapshot["load_avg_5min"] = float(parts[1])
    except Exception:
        pass

    # GPU memory
    if torch.cuda.is_available():
        snapshot["gpu_memory_allocated_gb"] = round(torch.cuda.memory_allocated() / 1024**3, 4)
        snapshot["gpu_memory_reserved_gb"] = round(torch.cuda.memory_reserved() / 1024**3, 4)
        snapshot["gpu_max_memory_allocated_gb"] = round(torch.cuda.max_memory_allocated() / 1024**3, 4)
    else:
        snapshot["gpu_memory_allocated_gb"] = 0.0

    return snapshot


# ── Windowed turn dataset ────────────────────────────────────────────────


class WindowedTurnDataset(Dataset):
    """Dataset that unrolls battles into per-turn examples with causal history.

    Each example is a window of consecutive turns [t-W+1 .. t] from a battle,
    where the target is the action at turn t.
    """

    def __init__(
        self,
        battles: list[dict[str, np.ndarray]],
        max_window: int = 20,
    ) -> None:
        self.max_window = max_window
        self.examples: list[tuple[int, int]] = []
        self.battles = battles

        for b_idx, battle in enumerate(battles):
            seq_len = int(battle.get("seq_len", battle["action"].shape[0]))
            for t in range(seq_len):
                action = int(battle["action"][t])
                if action >= 0:
                    self.examples.append((b_idx, t))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        b_idx, t_idx = self.examples[idx]
        battle = self.battles[b_idx]

        start = max(0, t_idx - self.max_window + 1)
        end = t_idx + 1
        actual_len = end - start

        own_team = battle["own_team"][start:end]
        opp_team = battle["opponent_team"][start:end]
        field_feat = battle["field"][start:end]
        context = battle["context"][start:end]
        legal_mask = battle["legal_mask"][start:end]

        action = int(battle["action"][t_idx])
        game_result = float(battle["game_result"][t_idx])

        item_targets = battle["item_targets"][t_idx]
        speed_targets = battle["speed_targets"][t_idx]
        role_targets = battle["role_targets"][t_idx]
        tera_targets = battle["tera_targets"][t_idx]
        move_family_targets = battle["move_family_targets"][t_idx]

        return {
            "own_team": torch.from_numpy(own_team.copy()).float(),
            "opponent_team": torch.from_numpy(opp_team.copy()).float(),
            "field": torch.from_numpy(field_feat.copy()).float(),
            "context": torch.from_numpy(context.copy()).float(),
            "legal_mask": torch.from_numpy(legal_mask.copy()).float(),
            "action": torch.tensor(action, dtype=torch.long),
            "game_result": torch.tensor(game_result, dtype=torch.float),
            "seq_len": torch.tensor(actual_len, dtype=torch.long),
            "item_targets": torch.from_numpy(item_targets.copy()).long(),
            "speed_targets": torch.from_numpy(speed_targets.copy()).long(),
            "role_targets": torch.from_numpy(role_targets.copy()).long(),
            "tera_targets": torch.from_numpy(tera_targets.copy()).long(),
            "move_family_targets": torch.from_numpy(move_family_targets.copy()).long(),
        }


def collate_windowed(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Collate windowed turn examples with right-padding."""
    max_len = max(item["seq_len"].item() for item in batch)

    result: dict[str, torch.Tensor] = {}
    seq_lens = torch.tensor([item["seq_len"].item() for item in batch], dtype=torch.long)
    result["seq_len"] = seq_lens

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
                padding = torch.zeros(pad_shape, dtype=t.dtype)
                t = torch.cat([t, padding], dim=0)
            tensors.append(t)
        result[key] = torch.stack(tensors)

    return result


# ── Learning rate scheduler ──────────────────────────────────────────────


class WarmupCosineScheduler:
    """Cosine annealing with linear warmup."""

    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.base_lrs = [pg["lr"] for pg in optimizer.param_groups]
        self._step = 0

    def step(self):
        self._step += 1
        lr = self._get_lr()
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr

    def _get_lr(self):
        if self._step < self.warmup_steps:
            return self.base_lrs[0] * self._step / max(self.warmup_steps, 1)
        progress = (self._step - self.warmup_steps) / max(self.total_steps - self.warmup_steps, 1)
        progress = min(progress, 1.0)
        return self.min_lr + 0.5 * (self.base_lrs[0] - self.min_lr) * (1 + math.cos(math.pi * progress))

    def get_lr(self):
        return self.optimizer.param_groups[0]["lr"]


# ── Training loop ────────────────────────────────────────────────────────


def forward_step(
    model: BattleTransformer,
    batch: dict[str, torch.Tensor],
    config: TransformerConfig,
) -> tuple[torch.Tensor, dict[str, float], torch.Tensor]:
    """Forward pass for windowed turn data."""
    own_team = batch["own_team"]
    opp_team = batch["opponent_team"]
    field_feat = batch["field"]
    context = batch["context"]
    legal_mask_seq = batch["legal_mask"]
    action = batch["action"]
    game_result = batch["game_result"]
    seq_len = batch["seq_len"]

    output = model(
        own_team, opp_team, field_feat, context,
        legal_mask=legal_mask_seq,
        seq_len=seq_len,
        return_auxiliary=True,
        return_value=config.use_value_head,
    )

    logits = output.policy_logits

    last_idx = (seq_len - 1).clamp(min=0)
    legal_last = torch.gather(
        legal_mask_seq, dim=1,
        index=last_idx.unsqueeze(1).unsqueeze(2).expand(-1, 1, NUM_ACTIONS),
    ).squeeze(1)

    aux_targets = {
        "item_targets": batch["item_targets"],
        "speed_targets": batch["speed_targets"],
        "role_targets": batch["role_targets"],
        "tera_targets": batch["tera_targets"],
        "move_family_targets": batch["move_family_targets"],
    }

    loss, loss_dict = compute_total_loss(
        output, action, legal_last,
        aux_targets=aux_targets,
        game_result=game_result,
        config=config,
    )

    return loss, loss_dict, logits


def train_epoch(
    model, loader, optimizer, scheduler, config, device,
    grad_accum=1, max_grad_norm=1.0,
) -> dict[str, float]:
    """Run one training epoch with resource tracking."""
    model.train()
    total_loss = total_policy = total_aux = total_value = 0.0
    total_correct = total_top3 = total_examples = n_batches = 0

    optimizer.zero_grad()

    for batch_idx, batch in enumerate(loader):
        batch = {k: v.to(device) for k, v in batch.items()}
        loss, loss_dict, logits = forward_step(model, batch, config)
        (loss / grad_accum).backward()

        if (batch_idx + 1) % grad_accum == 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        total_loss += loss_dict.get("total", 0.0)
        total_policy += loss_dict.get("policy", 0.0)
        total_aux += loss_dict.get("auxiliary", 0.0)
        total_value += loss_dict.get("value", 0.0)
        n_batches += 1

        action = batch["action"]
        with torch.no_grad():
            valid = action >= 0
            if valid.any():
                preds = logits[valid].argmax(dim=-1)
                total_correct += (preds == action[valid]).sum().item()
                top3 = logits[valid].topk(min(3, NUM_ACTIONS), dim=-1).indices
                total_top3 += (top3 == action[valid].unsqueeze(-1)).any(dim=-1).sum().item()
                total_examples += valid.sum().item()

    n = max(n_batches, 1)
    ne = max(total_examples, 1)
    return {
        "loss": total_loss / n, "policy_loss": total_policy / n,
        "aux_loss": total_aux / n, "value_loss": total_value / n,
        "accuracy": total_correct / ne, "top3_accuracy": total_top3 / ne,
        "total_examples": total_examples,
    }


@torch.no_grad()
def validate(model, loader, config, device) -> dict[str, float]:
    """Run validation with auxiliary accuracy tracking."""
    model.eval()
    total_loss = total_policy = total_aux = total_value = 0.0
    total_correct = total_top3 = total_examples = n_batches = 0
    aux_correct = {"item": 0, "speed": 0, "role": 0}
    aux_total = {"item": 0, "speed": 0, "role": 0}

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        loss, loss_dict, logits = forward_step(model, batch, config)

        total_loss += loss_dict.get("total", 0.0)
        total_policy += loss_dict.get("policy", 0.0)
        total_aux += loss_dict.get("auxiliary", 0.0)
        total_value += loss_dict.get("value", 0.0)
        n_batches += 1

        action = batch["action"]
        valid = action >= 0
        if valid.any():
            preds = logits[valid].argmax(dim=-1)
            total_correct += (preds == action[valid]).sum().item()
            top3 = logits[valid].topk(min(3, NUM_ACTIONS), dim=-1).indices
            total_top3 += (top3 == action[valid].unsqueeze(-1)).any(dim=-1).sum().item()
            total_examples += valid.sum().item()

        # Auxiliary accuracy
        output = model(
            batch["own_team"], batch["opponent_team"], batch["field"], batch["context"],
            legal_mask=batch["legal_mask"], seq_len=batch["seq_len"],
            return_auxiliary=True, return_value=False,
        )
        if isinstance(output, TransformerOutput) and output.auxiliary_preds is not None:
            for head, tgt_key, short in [
                ("item_logits", "item_targets", "item"),
                ("speed_logits", "speed_targets", "speed"),
                ("role_logits", "role_targets", "role"),
            ]:
                if head in output.auxiliary_preds and tgt_key in batch:
                    pred = output.auxiliary_preds[head]
                    target = batch[tgt_key]
                    flat_pred = pred.reshape(-1, pred.shape[-1])
                    flat_target = target.reshape(-1)
                    valid_aux = flat_target >= 0
                    if valid_aux.any():
                        aux_preds = flat_pred[valid_aux].argmax(dim=-1)
                        aux_correct[short] += (aux_preds == flat_target[valid_aux]).sum().item()
                        aux_total[short] += valid_aux.sum().item()

    n = max(n_batches, 1)
    ne = max(total_examples, 1)
    result = {
        "loss": total_loss / n, "policy_loss": total_policy / n,
        "aux_loss": total_aux / n, "value_loss": total_value / n,
        "accuracy": total_correct / ne, "top3_accuracy": total_top3 / ne,
        "total_examples": total_examples,
    }
    for name in ["item", "speed", "role"]:
        result[f"aux_{name}_accuracy"] = aux_correct[name] / aux_total[name] if aux_total[name] > 0 else 0.0
    return result


# ── Data loading ─────────────────────────────────────────────────────────


def load_battles(data_dir: Path, num_battles: int = 250) -> tuple[list[dict], Any]:
    """Load specified number of real battles from processed data directory."""
    battles_dir = data_dir / "battles"
    vocabs = BattleVocabularies.load(data_dir / "vocabs")
    npz_files = sorted(battles_dir.glob("*.npz"))

    # Sample deterministically
    import random
    rng = random.Random(42)
    if len(npz_files) > num_battles:
        npz_files = rng.sample(npz_files, num_battles)
        npz_files.sort()

    sequences, skipped = [], 0
    total_turns = 0
    for npz_file in npz_files:
        try:
            data = dict(np.load(str(npz_file)))
            if "own_team" in data and "action" in data:
                sequences.append(data)
                sl = int(data.get("seq_len", data["action"].shape[0]))
                total_turns += sl
            else:
                skipped += 1
        except Exception:
            skipped += 1

    logger.info(f"Loaded {len(sequences)} battles ({skipped} skipped), {total_turns} total turns")
    return sequences, vocabs


def add_auxiliary_labels(sequences: list[dict]) -> list[dict]:
    """Add auxiliary labels derived from tensorized features."""
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
        move_family_targets = np.full((seq_len, n_slots, NUM_MOVE_FAMILIES), -1, dtype=np.int64)

        for t in range(seq_len):
            for s in range(n_slots):
                feat = opp_team[t, s]
                if int(feat[0]) == 0:
                    continue
                # Item: feature index 5 is item vocab index
                item_idx = int(feat[5])
                if item_idx > 1:  # >1 means not padding/unknown
                    item_targets[t, s] = min(item_idx % NUM_ITEM_CLASSES, NUM_ITEM_CLASSES - 1)

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


def split_data(sequences: list[dict], train_ratio=0.80, val_ratio=0.10, seed=42):
    """Split data into train/calibration-val/test by battle."""
    import random
    rng = random.Random(seed)
    indices = list(range(len(sequences)))
    rng.shuffle(indices)
    n = len(indices)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    return (
        [sequences[i] for i in indices[:n_train]],
        [sequences[i] for i in indices[n_train:n_train + n_val]],
        [sequences[i] for i in indices[n_train + n_val:]],
    )


# ── Test set evaluation ──────────────────────────────────────────────────


@torch.no_grad()
def evaluate_test_set(model, test_data, config, device, batch_size=32, max_window=20):
    """Comprehensive evaluation on held-out test set."""
    model.eval()
    dataset = WindowedTurnDataset(test_data, max_window=max_window)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        collate_fn=collate_windowed, num_workers=0)

    total_loss = total_policy = total_aux = total_value = 0.0
    total_correct = total_top3 = total_examples = n_batches = 0
    action_type_correct = np.zeros(NUM_ACTIONS)
    action_type_total = np.zeros(NUM_ACTIONS)
    all_probs, all_correct_list = [], []
    all_entropies = []

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        loss, loss_dict, logits = forward_step(model, batch, config)

        total_loss += loss_dict.get("total", 0.0)
        total_policy += loss_dict.get("policy", 0.0)
        total_aux += loss_dict.get("auxiliary", 0.0)
        total_value += loss_dict.get("value", 0.0)
        n_batches += 1

        action = batch["action"]
        valid = action >= 0
        if valid.any():
            preds = logits[valid].argmax(dim=-1)
            targets = action[valid]
            total_correct += (preds == targets).sum().item()
            top3 = logits[valid].topk(min(3, NUM_ACTIONS), dim=-1).indices
            total_top3 += (top3 == targets.unsqueeze(-1)).any(dim=-1).sum().item()
            total_examples += valid.sum().item()

            for a in range(NUM_ACTIONS):
                mask_a = targets == a
                if mask_a.any():
                    action_type_total[a] += mask_a.sum().item()
                    action_type_correct[a] += ((preds == targets) & mask_a).sum().item()

            probs = F.softmax(logits[valid], dim=-1)
            max_probs = probs.max(dim=-1).values
            is_correct = (preds == targets).float()
            all_probs.extend(max_probs.float().cpu().tolist())
            all_correct_list.extend(is_correct.float().cpu().tolist())

            # Entropy of predictions (mask -inf before computing)
            masked_logits = logits[valid].clone()
            masked_logits[masked_logits == float("-inf")] = 0.0
            safe_probs = F.softmax(masked_logits, dim=-1)
            log_safe = torch.log(safe_probs + 1e-10)
            entropy = -(safe_probs * log_safe).sum(dim=-1)
            all_entropies.extend(entropy.float().cpu().tolist())

    n = max(n_batches, 1)
    ne = max(total_examples, 1)

    action_names = [
        "move1", "move2", "move3", "move4",
        "tera_move1", "tera_move2", "tera_move3", "tera_move4",
        "switch1", "switch2", "switch3", "switch4", "switch5",
    ]
    per_action_acc = {}
    for i in range(NUM_ACTIONS):
        if action_type_total[i] > 0:
            per_action_acc[action_names[i]] = {
                "accuracy": round(action_type_correct[i] / action_type_total[i], 4),
                "count": int(action_type_total[i]),
                "fraction": round(action_type_total[i] / ne, 4),
            }

    # Calibration analysis
    calibration = {}
    if all_probs:
        probs_arr = np.array(all_probs)
        correct_arr = np.array(all_correct_list)
        bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
        for i in range(len(bins) - 1):
            mask = (probs_arr >= bins[i]) & (probs_arr < bins[i + 1])
            if mask.any():
                calibration[f"{bins[i]:.1f}-{bins[i+1]:.1f}"] = {
                    "mean_confidence": round(float(probs_arr[mask].mean()), 4),
                    "mean_accuracy": round(float(correct_arr[mask].mean()), 4),
                    "count": int(mask.sum()),
                    "calibration_error": round(abs(float(probs_arr[mask].mean()) - float(correct_arr[mask].mean())), 4),
                }

    # Expected Calibration Error (ECE)
    ece = 0.0
    if all_probs:
        probs_arr = np.array(all_probs)
        correct_arr = np.array(all_correct_list)
        for i in range(len(bins) - 1):
            mask = (probs_arr >= bins[i]) & (probs_arr < bins[i + 1])
            if mask.any():
                bin_acc = correct_arr[mask].mean()
                bin_conf = probs_arr[mask].mean()
                ece += abs(bin_acc - bin_conf) * mask.sum() / len(probs_arr)

    return {
        "test_loss": round(total_loss / n, 4),
        "test_policy_loss": round(total_policy / n, 4),
        "test_aux_loss": round(total_aux / n, 4),
        "test_value_loss": round(total_value / n, 4),
        "test_accuracy": round(total_correct / ne, 4),
        "test_top3_accuracy": round(total_top3 / ne, 4),
        "test_examples": total_examples,
        "test_nll": round(total_policy / n, 4),
        "expected_calibration_error": round(ece, 4),
        "mean_prediction_entropy": round(float(np.mean(all_entropies)) if all_entropies else 0.0, 4),
        "per_action_accuracy": per_action_acc,
        "calibration": calibration,
    }


# ── Checkpointing ───────────────────────────────────────────────────────


def save_checkpoint(model, optimizer, config, epoch, val_loss, checkpoint_dir, is_best=False):
    """Save model checkpoint."""
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss": val_loss,
        "model_class": "BattleTransformer",
        "config": {
            "num_layers": config.num_layers,
            "hidden_dim": config.hidden_dim,
            "num_heads": config.num_heads,
            "dropout": config.dropout,
            "species_vocab_size": config.species_vocab_size,
            "moves_vocab_size": config.moves_vocab_size,
            "items_vocab_size": config.items_vocab_size,
            "abilities_vocab_size": config.abilities_vocab_size,
            "types_vocab_size": config.types_vocab_size,
            "status_vocab_size": config.status_vocab_size,
            "weather_vocab_size": config.weather_vocab_size,
            "terrain_vocab_size": config.terrain_vocab_size,
            "auxiliary_loss_weight": config.auxiliary_loss_weight,
            "use_value_head": config.use_value_head,
            "value_loss_weight": config.value_loss_weight,
        },
    }
    checkpoint_dir = Path(checkpoint_dir)
    if is_best:
        torch.save(state, checkpoint_dir / "best_model.pt")
        logger.info(f"  Saved best checkpoint (val_loss={val_loss:.4f})")

    path = checkpoint_dir / f"checkpoint_epoch_{epoch:03d}.pt"
    torch.save(state, path)

    # Keep last 3 + best
    checkpoints = sorted(checkpoint_dir.glob("checkpoint_epoch_*.pt"))
    if len(checkpoints) > 3:
        for old in checkpoints[:-3]:
            old.unlink()


# ── Main ─────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 4: Train BattleTransformer on 250 real battles")
    parser.add_argument("--data-dir", type=str, default="data/processed")
    parser.add_argument("--num-battles", type=int, default=250)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--aux-weight", type=float, default=0.2)
    parser.add_argument("--value-weight", type=float, default=0.1)
    parser.add_argument("--max-window", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/phase4_250")
    parser.add_argument("--report-path", type=str, default="checkpoints/phase4_250/training_report.json")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    overall_start = time.time()

    logger.info("=" * 72)
    logger.info("PHASE 4: BattleTransformer Training (250 Real Battles)")
    logger.info("=" * 72)

    # ── System info ──
    sys_info = get_system_info()
    logger.info(f"CPU:      {sys_info['cpu_model']} ({sys_info['cpu_count']} logical cores)")
    logger.info(f"RAM:      {sys_info['ram_total_gb']} GB total, {sys_info.get('ram_available_gb', '?')} GB available")
    logger.info(f"GPU:      {sys_info['gpu_name']} ({sys_info['gpu_memory_gb']} GB)")
    logger.info(f"PyTorch:  {sys_info['pytorch_version']}")
    logger.info(f"Device:   {device}")

    # ── Load data ──
    logger.info("-" * 72)
    logger.info("LOADING DATA")
    logger.info("-" * 72)
    data_dir = Path(args.data_dir)
    load_start = time.time()
    snap_before_load = get_resource_snapshot()

    sequences, vocabs = load_battles(data_dir, num_battles=args.num_battles)
    load_time = time.time() - load_start
    snap_after_load = get_resource_snapshot()

    logger.info(f"Data loaded in {load_time:.2f}s")
    logger.info(f"RAM after load: {snap_after_load['ram_used_gb']:.3f} GB "
                f"(delta: {snap_after_load['ram_used_gb'] - snap_before_load['ram_used_gb']:.3f} GB)")

    # ── Split: 80/10/10 ──
    train_seqs, val_seqs, test_seqs = split_data(sequences, seed=args.seed)
    logger.info(f"Split: train={len(train_seqs)}, calibration/val={len(val_seqs)}, test={len(test_seqs)}")
    del sequences
    gc.collect()

    # Compute per-split turn counts
    train_turns = sum(int(s.get("seq_len", s["action"].shape[0])) for s in train_seqs)
    val_turns = sum(int(s.get("seq_len", s["action"].shape[0])) for s in val_seqs)
    test_turns = sum(int(s.get("seq_len", s["action"].shape[0])) for s in test_seqs)
    logger.info(f"Turns: train={train_turns}, val={val_turns}, test={test_turns}")

    # ── Auxiliary labels ──
    logger.info("Adding auxiliary labels...")
    aux_start = time.time()
    train_seqs = add_auxiliary_labels(train_seqs)
    val_seqs = add_auxiliary_labels(val_seqs)
    test_seqs = add_auxiliary_labels(test_seqs)
    aux_time = time.time() - aux_start
    logger.info(f"Auxiliary labels added in {aux_time:.2f}s")

    # ── Create datasets ──
    train_dataset = WindowedTurnDataset(train_seqs, max_window=args.max_window)
    val_dataset = WindowedTurnDataset(val_seqs, max_window=args.max_window)
    logger.info(f"Train examples: {len(train_dataset)}, Val examples: {len(val_dataset)}")

    data_stats = {
        "num_battles": len(train_seqs) + len(val_seqs) + len(test_seqs),
        "train_battles": len(train_seqs),
        "calibration_val_battles": len(val_seqs),
        "test_battles": len(test_seqs),
        "train_turns": train_turns,
        "calibration_val_turns": val_turns,
        "test_turns": test_turns,
        "train_examples": len(train_dataset),
        "val_examples": len(val_dataset),
        "max_window": args.max_window,
        "data_load_time_sec": round(load_time, 2),
        "aux_label_time_sec": round(aux_time, 2),
    }

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_windowed, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate_windowed, num_workers=0)

    # ── Create model ──
    logger.info("-" * 72)
    logger.info("MODEL SETUP")
    logger.info("-" * 72)
    config = TransformerConfig.from_vocabs(
        vocabs, num_layers=args.num_layers, hidden_dim=args.hidden_dim,
        num_heads=args.num_heads, dropout=args.dropout,
        auxiliary_loss_weight=args.aux_weight,
        use_value_head=True, value_loss_weight=args.value_weight,
    )
    model = BattleTransformer(config).to(device)
    param_count = model.count_parameters()

    # Model architecture summary
    logger.info(f"Architecture: {config.num_layers}L / {config.hidden_dim}d / {config.num_heads}H")
    logger.info(f"Parameters:   {param_count:,} trainable")
    logger.info(f"Aux weight:   {config.auxiliary_loss_weight}")
    logger.info(f"Value weight: {config.value_loss_weight}")
    logger.info(f"Dropout:      {config.dropout}")

    # Memory estimate
    param_memory_mb = param_count * 4 / 1024 / 1024  # float32
    logger.info(f"Param memory: ~{param_memory_mb:.1f} MB")

    training_config = {
        "num_layers": config.num_layers,
        "hidden_dim": config.hidden_dim,
        "num_heads": config.num_heads,
        "dropout": config.dropout,
        "parameter_count": param_count,
        "param_memory_mb": round(param_memory_mb, 2),
        "batch_size": args.batch_size,
        "max_epochs": args.epochs,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "warmup_steps": args.warmup_steps,
        "patience": args.patience,
        "grad_accumulation": args.grad_accum,
        "aux_loss_weight": args.aux_weight,
        "value_loss_weight": args.value_weight,
        "use_value_head": True,
        "max_window": args.max_window,
        "device": device,
        "seed": args.seed,
    }

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                   weight_decay=args.weight_decay, betas=(0.9, 0.999))
    total_steps = args.epochs * len(train_loader) // args.grad_accum
    scheduler = WarmupCosineScheduler(optimizer, warmup_steps=args.warmup_steps,
                                       total_steps=total_steps)

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # ── Training loop ──
    logger.info("=" * 72)
    logger.info(f"STARTING TRAINING ({len(train_loader)} batches/epoch, {total_steps} total steps)")
    logger.info("=" * 72)

    train_start = time.time()
    best_val_loss = float("inf")
    patience_counter = 0
    epoch_records = []
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        snap_epoch_start = get_resource_snapshot()

        train_metrics = train_epoch(model, train_loader, optimizer, scheduler, config, device,
                                     grad_accum=args.grad_accum)
        val_metrics = validate(model, val_loader, config, device)

        epoch_time = time.time() - epoch_start
        snap_epoch_end = get_resource_snapshot()
        lr = scheduler.get_lr()

        # Epoch resource usage
        epoch_cpu_delta = snap_epoch_end.get("cpu_total_time_sec", 0) - snap_epoch_start.get("cpu_total_time_sec", 0)
        cpu_utilization = epoch_cpu_delta / max(epoch_time, 0.001) * 100

        epoch_record = {
            "epoch": epoch,
            # Losses
            "train_loss": round(train_metrics["loss"], 4),
            "val_loss": round(val_metrics["loss"], 4),
            "train_policy_loss": round(train_metrics["policy_loss"], 4),
            "val_policy_loss": round(val_metrics["policy_loss"], 4),
            "train_aux_loss": round(train_metrics["aux_loss"], 4),
            "val_aux_loss": round(val_metrics["aux_loss"], 4),
            "train_value_loss": round(train_metrics["value_loss"], 4),
            "val_value_loss": round(val_metrics["value_loss"], 4),
            # Accuracy
            "train_accuracy": round(train_metrics["accuracy"], 4),
            "val_accuracy": round(val_metrics["accuracy"], 4),
            "train_top3_accuracy": round(train_metrics["top3_accuracy"], 4),
            "val_top3_accuracy": round(val_metrics["top3_accuracy"], 4),
            # Auxiliary
            "val_aux_item_accuracy": round(val_metrics.get("aux_item_accuracy", 0.0), 4),
            "val_aux_speed_accuracy": round(val_metrics.get("aux_speed_accuracy", 0.0), 4),
            "val_aux_role_accuracy": round(val_metrics.get("aux_role_accuracy", 0.0), 4),
            # LR
            "learning_rate": lr,
            # Resource usage
            "epoch_wall_time_sec": round(epoch_time, 2),
            "examples_per_sec": round(train_metrics["total_examples"] / max(epoch_time, 0.001), 1),
            "cpu_time_this_epoch_sec": round(epoch_cpu_delta, 2),
            "cpu_utilization_pct": round(cpu_utilization, 1),
            "ram_used_gb": snap_epoch_end.get("ram_used_gb", 0.0),
            "ram_peak_gb": snap_epoch_end.get("ram_peak_gb", 0.0),
            "gpu_memory_allocated_gb": snap_epoch_end.get("gpu_memory_allocated_gb", 0.0),
            "load_avg_1min": snap_epoch_end.get("load_avg_1min", 0.0),
        }
        epoch_records.append(epoch_record)

        logger.info(
            f"Epoch {epoch:2d}/{args.epochs}: "
            f"loss={train_metrics['loss']:.4f} val_loss={val_metrics['loss']:.4f} "
            f"acc={train_metrics['accuracy']:.3f} val_acc={val_metrics['accuracy']:.3f} "
            f"top3={val_metrics['top3_accuracy']:.3f} "
            f"lr={lr:.2e} time={epoch_time:.1f}s "
            f"RAM={snap_epoch_end.get('ram_used_gb', 0):.2f}GB "
            f"CPU={cpu_utilization:.0f}%"
        )
        if val_metrics.get("aux_item_accuracy", 0) > 0:
            logger.info(f"  Aux: item={val_metrics['aux_item_accuracy']:.3f} "
                        f"speed={val_metrics['aux_speed_accuracy']:.3f} "
                        f"role={val_metrics['aux_role_accuracy']:.3f}")

        save_checkpoint(model, optimizer, config, epoch, val_metrics["loss"], checkpoint_dir)
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            patience_counter = 0
            save_checkpoint(model, optimizer, config, epoch, val_metrics["loss"], checkpoint_dir, is_best=True)
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                logger.info(f"Early stopping at epoch {epoch} (patience={args.patience})")
                break

    total_train_time = time.time() - train_start
    best_epoch = min(epoch_records, key=lambda m: m["val_loss"])["epoch"]
    logger.info("=" * 72)
    logger.info(f"TRAINING COMPLETE in {total_train_time:.1f}s ({total_train_time/60:.1f} min)")
    logger.info(f"Best epoch: {best_epoch}, Best val_loss: {best_val_loss:.4f}")

    # ── Load best model for test evaluation ──
    best_path = checkpoint_dir / "best_model.pt"
    if best_path.exists():
        ckpt = torch.load(str(best_path), map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        logger.info(f"Loaded best model from epoch {ckpt['epoch']}")

    # ── Test evaluation ──
    logger.info("=" * 72)
    logger.info("OUT-OF-SAMPLE TEST SET EVALUATION")
    logger.info("=" * 72)
    eval_start = time.time()
    test_results = evaluate_test_set(model, test_seqs, config, device,
                                     batch_size=args.batch_size, max_window=args.max_window)
    eval_time = time.time() - eval_start

    logger.info(f"Test accuracy:      {test_results['test_accuracy']:.4f}")
    logger.info(f"Test top-3 accuracy:{test_results['test_top3_accuracy']:.4f}")
    logger.info(f"Test NLL:           {test_results['test_nll']:.4f}")
    logger.info(f"Test ECE:           {test_results['expected_calibration_error']:.4f}")
    logger.info(f"Mean entropy:       {test_results['mean_prediction_entropy']:.4f}")
    logger.info(f"Test examples:      {test_results['test_examples']}")

    if test_results.get("per_action_accuracy"):
        logger.info("Per-action accuracy:")
        for name, stats in sorted(test_results["per_action_accuracy"].items()):
            logger.info(f"  {name:12s}: {stats['accuracy']:.3f} (n={stats['count']:4d}, {stats['fraction']:.1%})")

    if test_results.get("calibration"):
        logger.info("Calibration (confidence bins):")
        for bin_name, stats in test_results["calibration"].items():
            logger.info(f"  [{bin_name}]: conf={stats['mean_confidence']:.3f} "
                        f"acc={stats['mean_accuracy']:.3f} "
                        f"err={stats['calibration_error']:.3f} "
                        f"(n={stats['count']})")

    # ── Build comprehensive report ──
    final_snap = get_resource_snapshot()
    total_wall_time = time.time() - overall_start

    report = {
        "phase": "Phase 4 - BattleTransformer Training (250 Real Battles)",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),

        "system_info": sys_info,
        "training_config": training_config,
        "data_stats": data_stats,

        "epoch_metrics": epoch_records,

        "test_results": test_results,

        "final_results": {
            "best_epoch": best_epoch,
            "best_val_loss": round(best_val_loss, 4),
            "total_epochs_trained": len(epoch_records),
            "best_checkpoint": str(best_path),
            "total_training_time_sec": round(total_train_time, 2),
            "total_training_time_min": round(total_train_time / 60, 2),
            "test_evaluation_time_sec": round(eval_time, 2),
            "total_wall_time_sec": round(total_wall_time, 2),
            "total_wall_time_min": round(total_wall_time / 60, 2),
        },

        "resource_summary": {
            "peak_ram_gb": final_snap.get("ram_peak_gb", 0.0),
            "final_ram_gb": final_snap.get("ram_used_gb", 0.0),
            "total_cpu_user_time_sec": final_snap.get("cpu_user_time_sec", 0.0),
            "total_cpu_sys_time_sec": final_snap.get("cpu_sys_time_sec", 0.0),
            "total_cpu_time_sec": final_snap.get("cpu_total_time_sec", 0.0),
            "max_rss_gb": final_snap.get("max_rss_gb", 0.0),
            "gpu_used": torch.cuda.is_available(),
            "gpu_name": sys_info.get("gpu_name", "None"),
            "gpu_memory_gb": sys_info.get("gpu_memory_gb", 0.0),
            "gpu_peak_memory_allocated_gb": final_snap.get("gpu_max_memory_allocated_gb", 0.0),
            "avg_epoch_wall_time_sec": round(
                sum(r["epoch_wall_time_sec"] for r in epoch_records) / max(len(epoch_records), 1), 2
            ),
            "avg_examples_per_sec": round(
                sum(r["examples_per_sec"] for r in epoch_records) / max(len(epoch_records), 1), 1
            ),
            "avg_cpu_utilization_pct": round(
                sum(r["cpu_utilization_pct"] for r in epoch_records) / max(len(epoch_records), 1), 1
            ),
        },
    }

    report_path = Path(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"\nReport saved to {report_path}")

    # ── Final summary ──
    logger.info("=" * 72)
    logger.info("PHASE 4 TRAINING SUMMARY")
    logger.info("=" * 72)
    logger.info(f"Model:         {config.num_layers}L/{config.hidden_dim}d/{config.num_heads}H ({param_count:,} params)")
    logger.info(f"Data:          {data_stats['num_battles']} battles ({data_stats['train_examples']} train examples)")
    logger.info(f"Split:         {data_stats['train_battles']} train / {data_stats['calibration_val_battles']} cal-val / {data_stats['test_battles']} test")
    logger.info(f"Best val loss: {best_val_loss:.4f} (epoch {best_epoch})")
    logger.info(f"Test accuracy: {test_results['test_accuracy']:.4f}")
    logger.info(f"Test top-3:    {test_results['test_top3_accuracy']:.4f}")
    logger.info(f"Test ECE:      {test_results['expected_calibration_error']:.4f}")
    logger.info(f"Total time:    {total_wall_time/60:.1f} min ({total_wall_time:.0f}s)")
    logger.info(f"Peak RAM:      {final_snap.get('ram_peak_gb', 0.0):.2f} GB")
    logger.info(f"CPU time:      {final_snap.get('cpu_total_time_sec', 0.0):.1f}s")
    if torch.cuda.is_available():
        logger.info(f"GPU peak mem:  {final_snap.get('gpu_max_memory_allocated_gb', 0.0):.3f} GB")
    logger.info("=" * 72)


if __name__ == "__main__":
    main()
