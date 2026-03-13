#!/usr/bin/env python3
"""Build 5 synthetic P8-Lean inference scenarios from replay tensors.

This script mines real replay turns from `data/processed/battles/*.npz` and
creates a compact JSON scenario file for inference testing.

Selection strategy (hard-optimal labels):
- Use turns where the legal mask has exactly one legal action.
- These are forced-choice states, so there is one overwhelmingly-correct choice.
- Keep scenarios from different battles and different turn buckets where possible.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build synthetic P8-Lean scenarios")
    parser.add_argument("--data-dir", default="data/processed", help="Processed dataset root")
    parser.add_argument(
        "--output",
        default="data/synthetic/p8_lean_scenarios.json",
        help="Output scenario JSON file",
    )
    parser.add_argument("--num-scenarios", type=int, default=5)
    return parser.parse_args()


def turn_bucket(turn_norm: float) -> str:
    # Context turn number is normalized by 100 in tensorization.
    t = turn_norm * 100.0
    if t < 7:
        return "early"
    if t < 15:
        return "mid"
    return "late"


def main() -> int:
    args = parse_args()
    battles_dir = Path(args.data_dir) / "battles"
    if not battles_dir.exists():
        raise FileNotFoundError(f"Missing battles directory: {battles_dir}")

    candidates: list[dict] = []
    for npz_path in sorted(battles_dir.glob("*.npz")):
        data = np.load(npz_path)
        seq_len = int(data["seq_len"])
        for t in range(seq_len):
            legal_mask = data["legal_mask"][t]
            legal_indices = np.where(legal_mask > 0.5)[0].tolist()
            action = int(data["action"][t])
            if action < 0:
                continue
            if len(legal_indices) != 1:
                continue

            context = data["context"][t]
            field = data["field"][t]
            candidates.append(
                {
                    "battle_id": npz_path.stem,
                    "turn_index": t,
                    "turn_bucket": turn_bucket(float(context[0])),
                    "forced_switch": bool(context[3] > 0.5),
                    "weather_active": int(field[0]) > 0,
                    "action": action,
                    "legal_indices": legal_indices,
                    "own_team": data["own_team"][t].tolist(),
                    "opponent_team": data["opponent_team"][t].tolist(),
                    "field": field.tolist(),
                    "context": context.tolist(),
                    "legal_mask": legal_mask.tolist(),
                    "source_note": "real replay turn with only one legal action",
                }
            )

    if len(candidates) < args.num_scenarios:
        raise RuntimeError(
            f"Need at least {args.num_scenarios} forced-choice candidates; found {len(candidates)}"
        )

    selected: list[dict] = []
    seen_battles: set[str] = set()
    bucket_order = ["early", "mid", "late"]

    for bucket in bucket_order:
        for c in candidates:
            if len(selected) >= args.num_scenarios:
                break
            if c["battle_id"] in seen_battles:
                continue
            if c["turn_bucket"] != bucket:
                continue
            selected.append(c)
            seen_battles.add(c["battle_id"])

    for c in candidates:
        if len(selected) >= args.num_scenarios:
            break
        if c["battle_id"] in seen_battles:
            continue
        selected.append(c)
        seen_battles.add(c["battle_id"])

    scenarios = []
    for i, c in enumerate(selected[: args.num_scenarios], start=1):
        scenarios.append(
            {
                "scenario_id": f"p8_forced_{i:02d}",
                "family": "forced_mechanics",
                "label_type": "hard_optimal",
                "source": {
                    "battle_id": c["battle_id"],
                    "turn_index": c["turn_index"],
                    "source_note": c["source_note"],
                },
                "expected_action": c["action"],
                "acceptable_actions": c["legal_indices"],
                "metadata": {
                    "turn_bucket": c["turn_bucket"],
                    "forced_switch": c["forced_switch"],
                    "weather_active": c["weather_active"],
                    "num_legal": len(c["legal_indices"]),
                },
                "inputs": {
                    "own_team": c["own_team"],
                    "opponent_team": c["opponent_team"],
                    "field": c["field"],
                    "context": c["context"],
                    "legal_mask": c["legal_mask"],
                },
            }
        )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"scenarios": scenarios}, indent=2), encoding="utf-8")
    print(f"Wrote {len(scenarios)} scenarios -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
