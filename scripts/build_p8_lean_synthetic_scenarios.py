#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.environment.action_space import ACTION_NAMES

TARGET_SCENARIOS = [
    ("s1_priority_attack", 0, "Use move slot 1 to secure immediate tempo/KO line"),
    ("s2_coverage_attack", 1, "Use move slot 2 as the key coverage option"),
    ("s3_setup_or_tech_attack", 2, "Use move slot 3 for the identified tactical line"),
    ("s4_cleanup_attack", 3, "Use move slot 4 as the best cleanup option"),
    ("s5_forced_switch", 8, "Switch to bench slot 2 in a forced-switch state"),
]
TARGET_ACTIONS = {x[1] for x in TARGET_SCENARIOS}
MAX_CANDIDATES_PER_ACTION = 120


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build synthetic P8-Lean scenarios")
    p.add_argument("--data-dir", default="data/processed")
    p.add_argument("--output", default="data/synthetic/p8_lean_scenarios.json")
    return p.parse_args()


def turn_bucket(turn_norm: float) -> str:
    t = turn_norm * 100.0
    return "early" if t < 7 else ("mid" if t < 15 else "late")


def one_hot(action: int, size: int = 13) -> list[float]:
    m = [0.0] * size
    m[action] = 1.0
    return m


def main() -> int:
    args = parse_args()
    battles_dir = Path(args.data_dir) / "battles"
    if not battles_dir.exists():
        raise FileNotFoundError(battles_dir)

    by_action: dict[int, list[dict]] = {a: [] for a in TARGET_ACTIONS}

    for npz_path in sorted(battles_dir.glob("*.npz")):
        d = np.load(npz_path)
        L = int(d["seq_len"])
        for t in range(L):
            a = int(d["action"][t])
            if a not in TARGET_ACTIONS:
                continue
            lm = d["legal_mask"][t]
            if lm[a] <= 0.5:
                continue
            if len(by_action[a]) >= MAX_CANDIDATES_PER_ACTION:
                continue

            by_action[a].append(
                {
                    "battle_id": npz_path.stem,
                    "turn_index": t,
                    "turn_bucket": turn_bucket(float(d["context"][t][0])),
                    "forced_switch": bool(d["context"][t][3] > 0.5),
                    "weather_active": bool(d["field"][t][0] > 0),
                    "num_legal_original": int(np.sum(lm > 0.5)),
                    "own_team": d["own_team"][t].tolist(),
                    "opponent_team": d["opponent_team"][t].tolist(),
                    "field": d["field"][t].tolist(),
                    "context": d["context"][t].tolist(),
                    "original_legal_mask": lm.tolist(),
                }
            )
        if all(len(by_action[a]) >= MAX_CANDIDATES_PER_ACTION for a in TARGET_ACTIONS):
            break

    scenarios = []
    seen_battles: set[str] = set()

    for sid, a, desc in TARGET_SCENARIOS:
        cands = by_action.get(a, [])
        if not cands:
            raise RuntimeError(f"No candidates for {a}:{ACTION_NAMES[a]}")
        cands.sort(key=lambda c: (c["battle_id"] in seen_battles, -c["num_legal_original"], c["turn_bucket"]))
        c = cands[0]
        seen_battles.add(c["battle_id"])

        scenarios.append(
            {
                "scenario_id": sid,
                "family": "forced_mechanics",
                "label_type": "hard_optimal",
                "scenario_description": desc,
                "source": {
                    "battle_id": c["battle_id"],
                    "turn_index": c["turn_index"],
                    "source_note": "replay-grounded state with synthetic one-hot legality patch",
                },
                "expected_action": a,
                "expected_action_name": ACTION_NAMES[a],
                "acceptable_actions": [a],
                "metadata": {
                    "turn_bucket": c["turn_bucket"],
                    "forced_switch": c["forced_switch"],
                    "weather_active": c["weather_active"],
                    "num_legal_original": c["num_legal_original"],
                    "num_legal_synthetic": 1,
                },
                "inputs": {
                    "own_team": c["own_team"],
                    "opponent_team": c["opponent_team"],
                    "field": c["field"],
                    "context": c["context"],
                    "legal_mask": one_hot(a),
                    "original_legal_mask": c["original_legal_mask"],
                },
            }
        )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"scenarios": scenarios}, indent=2), encoding="utf-8")
    print(f"Wrote {len(scenarios)} scenarios -> {out}")
    for s in scenarios:
        print(f"- {s['scenario_id']}: {s['expected_action_name']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
