#!/usr/bin/env python3
"""Summarize Phase 4/P8 dataset statistics relevant to training-speed tradeoffs."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze processed battle tensors for speed/feature tradeoffs")
    p.add_argument("--processed-dir", type=str, default="data/processed/battles")
    p.add_argument("--output", type=str, default="docs/artifacts/p8_efficiency_stats.json")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    files = sorted(glob.glob(str(Path(args.processed_dir) / "*.npz")))
    if not files:
        raise FileNotFoundError(f"No .npz files in {args.processed_dir}")

    seq_lens: list[int] = []
    legal_counts: list[float] = []
    action_counts = np.zeros(13, dtype=np.int64)

    total_slots = 0
    unknown_item = 0
    unknown_ability = 0
    unknown_tera = 0
    terastallized_flag = 0
    field_binary_nonzero = 0

    for fp in files:
        battle = np.load(fp)
        seq_len = int(battle["seq_len"])
        seq_lens.append(seq_len)

        actions = battle["action"][:seq_len]
        for a in actions:
            if a >= 0:
                action_counts[int(a)] += 1

        legal_mask = battle["legal_mask"][:seq_len]
        legal_counts.extend(legal_mask.sum(axis=1).tolist())

        for key in ("own_team", "opponent_team"):
            x = battle[key][:seq_len]
            total_slots += int(np.prod(x.shape[:2]))
            unknown_item += int((x[:, :, 26] > 0.5).sum())
            unknown_ability += int((x[:, :, 27] > 0.5).sum())
            unknown_tera += int((x[:, :, 28] > 0.5).sum())
            terastallized_flag += int((x[:, :, 29] > 0.5).sum())

        field = battle["field"][:seq_len]
        field_binary_nonzero += int((field[:, 2:] != 0).sum())

    seq = np.array(seq_lens)
    legal = np.array(legal_counts)
    action_total = int(action_counts.sum())

    windows = [20, 16, 12, 10, 8]
    window_stats = {}
    for w in windows:
        tokens = w * 14
        window_stats[str(w)] = {
            "token_count": tokens,
            "attention_token_sq": tokens * tokens,
            "attention_vs_w20": (tokens * tokens) / float((20 * 14) ** 2),
        }

    out = {
        "num_battles": len(files),
        "total_turn_examples": int(seq.sum()),
        "seq_len": {
            "mean": float(seq.mean()),
            "p50": float(np.percentile(seq, 50)),
            "p90": float(np.percentile(seq, 90)),
            "p95": float(np.percentile(seq, 95)),
            "max": int(seq.max()),
        },
        "fraction_battles_at_or_below_window": {
            str(w): float((seq <= w).mean()) for w in windows
        },
        "legal_actions_per_turn": {
            "mean": float(legal.mean()),
            "p10": float(np.percentile(legal, 10)),
            "p90": float(np.percentile(legal, 90)),
        },
        "action_mix": {
            "move_fraction": float(action_counts[:8].sum() / action_total),
            "switch_fraction": float(action_counts[8:].sum() / action_total),
            "tera_move_fraction": float(action_counts[4:8].sum() / action_total),
            "raw_counts": action_counts.tolist(),
        },
        "sparsity_signals": {
            "unknown_item_rate": float(unknown_item / total_slots),
            "unknown_ability_rate": float(unknown_ability / total_slots),
            "unknown_tera_rate": float(unknown_tera / total_slots),
            "terastallized_flag_rate": float(terastallized_flag / total_slots),
            "field_binary_nonzero_rate": float(field_binary_nonzero / (seq.sum() * 16)),
        },
        "window_attention_cost_proxy": window_stats,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
