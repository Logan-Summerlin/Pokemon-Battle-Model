#!/usr/bin/env python3
"""Build 30 synthetic scenarios from fine-tuning data for P8-Lean inference.

Sources scenarios ONLY from data/fine-tuning/processed/battles/ to ensure
complete isolation from the training data in data/processed/battles/.

Covers all 5 scenario families from the proposal:
  1. Forced mechanics (forced switch, trap/lock, no-safe-switch)
  2. Tactical one-turn motifs (guaranteed KO, sack-vs-preserve, setup-vs-attack)
  3. Hidden-info ambiguity (unknown item/ability, move reveal uncertainty)
  4. Distribution shift / off-meta (uncommon but legal sets)
  5. Tempo and game-phase (early momentum, midgame hazards, late-game conversion)

Each scenario has exactly one correct action (hard-optimal label) with a
one-hot legal mask, following the established synthetic scenario format.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.environment.action_space import ACTION_NAMES, NUM_ACTIONS

# ── Scenario definitions ────────────────────────────────────────────────
# (scenario_id, target_action, family, description, selection_criteria)
# selection_criteria is a dict of filters applied when mining replay turns.

SCENARIO_DEFS: list[dict[str, Any]] = [
    # ── Family 1: Forced Mechanics (6 scenarios) ────────────────────────
    {
        "scenario_id": "ft_s01_forced_switch_slot2",
        "target_action": 8,
        "family": "forced_mechanics",
        "description": "Forced switch to bench slot 2 after KO (no move options)",
        "filters": {"forced_switch": True, "turn_bucket": None},
    },
    {
        "scenario_id": "ft_s02_forced_switch_slot3",
        "target_action": 9,
        "family": "forced_mechanics",
        "description": "Forced switch to bench slot 3 in forced-switch state",
        "filters": {"forced_switch": True, "turn_bucket": None},
    },
    {
        "scenario_id": "ft_s03_forced_switch_slot4",
        "target_action": 10,
        "family": "forced_mechanics",
        "description": "Forced switch to bench slot 4 in forced-switch state",
        "filters": {"forced_switch": True, "turn_bucket": None},
    },
    {
        "scenario_id": "ft_s04_forced_switch_slot5",
        "target_action": 11,
        "family": "forced_mechanics",
        "description": "Forced switch to bench slot 5 in forced-switch state",
        "filters": {"forced_switch": True, "turn_bucket": None},
    },
    {
        "scenario_id": "ft_s05_forced_switch_slot6",
        "target_action": 12,
        "family": "forced_mechanics",
        "description": "Forced switch to bench slot 6 in forced-switch state",
        "filters": {"forced_switch": True, "turn_bucket": None},
    },
    {
        "scenario_id": "ft_s06_endgame_move1",
        "target_action": 0,
        "family": "forced_mechanics",
        "description": "Late-game move 1 in no-safe-switch endgame position",
        "filters": {"forced_switch": False, "turn_bucket": "late"},
    },
    # ── Family 2: Tactical One-Turn Motifs (6 scenarios) ────────────────
    {
        "scenario_id": "ft_s07_priority_ko_move1",
        "target_action": 0,
        "family": "tactical_motif",
        "description": "Priority attack move 1 for immediate KO or tempo advantage",
        "filters": {"forced_switch": False, "turn_bucket": "mid", "num_legal_min": 3},
    },
    {
        "scenario_id": "ft_s08_coverage_move2",
        "target_action": 1,
        "family": "tactical_motif",
        "description": "Coverage move 2 to hit super-effective vs opponent active",
        "filters": {"forced_switch": False, "turn_bucket": None, "num_legal_min": 3},
    },
    {
        "scenario_id": "ft_s09_setup_move3",
        "target_action": 2,
        "family": "tactical_motif",
        "description": "Setup/tech move 3 chosen over raw attacking to gain advantage",
        "filters": {"forced_switch": False, "turn_bucket": "early", "num_legal_min": 3},
    },
    {
        "scenario_id": "ft_s10_cleanup_move4",
        "target_action": 3,
        "family": "tactical_motif",
        "description": "Cleanup move 4 to finish low-HP opponent",
        "filters": {"forced_switch": False, "turn_bucket": "late", "num_legal_min": 3},
    },
    {
        "scenario_id": "ft_s11_sack_vs_preserve_switch",
        "target_action": 8,
        "family": "tactical_motif",
        "description": "Strategic switch to slot 2 to preserve a sweeper (sack vs preserve)",
        "filters": {"forced_switch": False, "turn_bucket": "mid", "num_legal_min": 4},
    },
    {
        "scenario_id": "ft_s12_setup_race_move1",
        "target_action": 0,
        "family": "tactical_motif",
        "description": "Attacking move 1 in a setup-vs-attack race scenario",
        "filters": {"forced_switch": False, "turn_bucket": "mid", "num_legal_min": 4},
    },
    # ── Family 3: Hidden-Info Ambiguity (6 scenarios) ───────────────────
    {
        "scenario_id": "ft_s13_unknown_item_move1",
        "target_action": 0,
        "family": "hidden_info",
        "description": "Move 1 optimal despite unknown opponent item affecting calcs",
        "filters": {"forced_switch": False, "turn_bucket": "early", "num_legal_min": 3},
    },
    {
        "scenario_id": "ft_s14_unknown_speed_switch",
        "target_action": 9,
        "family": "hidden_info",
        "description": "Switch to slot 3 when opponent speed tier is uncertain",
        "filters": {"forced_switch": False, "turn_bucket": None, "num_legal_min": 4},
    },
    {
        "scenario_id": "ft_s15_move_reveal_move2",
        "target_action": 1,
        "family": "hidden_info",
        "description": "Move 2 chosen when opponent coverage threats are partially revealed",
        "filters": {"forced_switch": False, "turn_bucket": "mid", "num_legal_min": 3},
    },
    {
        "scenario_id": "ft_s16_ability_unknown_move3",
        "target_action": 2,
        "family": "hidden_info",
        "description": "Move 3 chosen when opponent ability is unknown and risky",
        "filters": {"forced_switch": False, "turn_bucket": None, "num_legal_min": 3},
    },
    {
        "scenario_id": "ft_s17_tera_move1",
        "target_action": 4,
        "family": "hidden_info",
        "description": "Tera + move 1 to change type matchup against unknown set",
        "filters": {"forced_switch": False, "can_tera": True, "turn_bucket": None},
    },
    {
        "scenario_id": "ft_s18_tera_move2",
        "target_action": 5,
        "family": "hidden_info",
        "description": "Tera + move 2 for STAB boost under type uncertainty",
        "filters": {"forced_switch": False, "can_tera": True, "turn_bucket": None},
    },
    # ── Family 4: Distribution Shift / Off-Meta (6 scenarios) ───────────
    {
        "scenario_id": "ft_s19_offmeta_move1",
        "target_action": 0,
        "family": "distribution_shift",
        "description": "Move 1 is correct despite uncommon pokemon/set on field",
        "filters": {"forced_switch": False, "turn_bucket": None, "num_legal_min": 3},
    },
    {
        "scenario_id": "ft_s20_offmeta_move3",
        "target_action": 2,
        "family": "distribution_shift",
        "description": "Move 3 optimal vs off-meta opponent set",
        "filters": {"forced_switch": False, "turn_bucket": None, "num_legal_min": 3},
    },
    {
        "scenario_id": "ft_s21_offmeta_switch3",
        "target_action": 9,
        "family": "distribution_shift",
        "description": "Switch to slot 3 against uncommon team composition",
        "filters": {"forced_switch": False, "turn_bucket": "early", "num_legal_min": 4},
    },
    {
        "scenario_id": "ft_s22_offmeta_tera_move3",
        "target_action": 6,
        "family": "distribution_shift",
        "description": "Tera + move 3 against a set the model may rarely see in training",
        "filters": {"forced_switch": False, "can_tera": True, "turn_bucket": None},
    },
    {
        "scenario_id": "ft_s23_offmeta_move4",
        "target_action": 3,
        "family": "distribution_shift",
        "description": "Move 4 is the best option despite being underrepresented in data",
        "filters": {"forced_switch": False, "turn_bucket": None, "num_legal_min": 3},
    },
    {
        "scenario_id": "ft_s24_offmeta_tera_move4",
        "target_action": 7,
        "family": "distribution_shift",
        "description": "Tera + move 4 in an unusual but legal game state",
        "filters": {"forced_switch": False, "can_tera": True, "turn_bucket": None},
    },
    # ── Family 5: Tempo and Game-Phase (6 scenarios) ────────────────────
    {
        "scenario_id": "ft_s25_early_momentum_move1",
        "target_action": 0,
        "family": "tempo_phase",
        "description": "Early-game momentum play: move 1 to establish pressure on turn 1-3",
        "filters": {"forced_switch": False, "turn_bucket": "early", "num_legal_min": 3},
    },
    {
        "scenario_id": "ft_s26_early_switch_pivot",
        "target_action": 8,
        "family": "tempo_phase",
        "description": "Early pivot switch to slot 2 for type advantage setup",
        "filters": {"forced_switch": False, "turn_bucket": "early", "num_legal_min": 4},
    },
    {
        "scenario_id": "ft_s27_midgame_hazard_move2",
        "target_action": 1,
        "family": "tempo_phase",
        "description": "Mid-game move 2 for hazard management or removal play",
        "filters": {"forced_switch": False, "turn_bucket": "mid", "num_legal_min": 3},
    },
    {
        "scenario_id": "ft_s28_midgame_switch4",
        "target_action": 10,
        "family": "tempo_phase",
        "description": "Mid-game defensive switch to slot 4 to tank incoming threat",
        "filters": {"forced_switch": False, "turn_bucket": "mid", "num_legal_min": 4},
    },
    {
        "scenario_id": "ft_s29_lategame_conversion_move1",
        "target_action": 0,
        "family": "tempo_phase",
        "description": "Late-game win conversion: move 1 to close out with speed advantage",
        "filters": {"forced_switch": False, "turn_bucket": "late", "num_legal_min": 2},
    },
    {
        "scenario_id": "ft_s30_lategame_last_switch",
        "target_action": 8,
        "family": "tempo_phase",
        "description": "Late-game last-resort switch to slot 2 for endgame matchup",
        "filters": {"forced_switch": False, "turn_bucket": "late", "num_legal_min": 3},
    },
]


def turn_bucket(turn_norm: float) -> str:
    """Classify normalised turn number into early/mid/late."""
    t = turn_norm * 100.0
    return "early" if t < 7 else ("mid" if t < 15 else "late")


def one_hot(action: int, size: int = NUM_ACTIONS) -> list[float]:
    m = [0.0] * size
    m[action] = 1.0
    return m


def matches_filters(
    action: int,
    legal_mask: np.ndarray,
    context: np.ndarray,
    filters: dict[str, Any],
) -> bool:
    """Check whether a turn satisfies the scenario's selection filters."""
    num_legal = int(np.sum(legal_mask > 0.5))
    is_forced = bool(context[3] > 0.5)
    can_tera = bool(context[2] > 0.5)
    bucket = turn_bucket(float(context[0]))

    if filters.get("forced_switch") is not None:
        if is_forced != filters["forced_switch"]:
            return False

    if filters.get("can_tera") is not None:
        if can_tera != filters["can_tera"]:
            return False

    if filters.get("turn_bucket") is not None:
        if bucket != filters["turn_bucket"]:
            return False

    if filters.get("num_legal_min") is not None:
        if num_legal < filters["num_legal_min"]:
            return False

    if filters.get("num_legal_max") is not None:
        if num_legal > filters["num_legal_max"]:
            return False

    return True


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build 30 synthetic P8-Lean scenarios from fine-tuning data"
    )
    p.add_argument(
        "--data-dir",
        default="data/fine-tuning/processed",
        help="Fine-tuning processed data directory (NOT training data)",
    )
    p.add_argument(
        "--output",
        default="data/synthetic/p8_lean_finetuning_scenarios.json",
        help="Output scenario JSON path",
    )
    p.add_argument(
        "--training-data-dir",
        default="data/processed",
        help="Training data directory (for exclusion check)",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    battles_dir = Path(args.data_dir) / "battles"
    if not battles_dir.exists():
        raise FileNotFoundError(f"Fine-tuning battles dir not found: {battles_dir}")

    # Load training battle IDs for exclusion verification
    training_battles_dir = Path(args.training_data_dir) / "battles"
    training_ids: set[str] = set()
    if training_battles_dir.exists():
        training_ids = {p.stem for p in training_battles_dir.glob("*.npz")}
    print(f"Training data contains {len(training_ids)} battle IDs (will verify exclusion)")

    # Collect fine-tuning battle files
    ft_files = sorted(battles_dir.glob("*.npz"))
    ft_ids = {p.stem for p in ft_files}
    overlap = ft_ids & training_ids
    if overlap:
        print(f"WARNING: {len(overlap)} fine-tuning battles overlap with training data!")
        print(f"  Overlapping IDs: {sorted(overlap)[:10]}...")
        # Remove overlapping files from consideration
        ft_files = [p for p in ft_files if p.stem not in training_ids]
        print(f"  After exclusion: {len(ft_files)} battles remain")
    else:
        print(f"Confirmed: 0 overlap between {len(ft_ids)} fine-tuning and {len(training_ids)} training battles")

    if not ft_files:
        raise RuntimeError("No fine-tuning battles available after exclusion")

    # Build candidate pool per scenario
    MAX_CANDIDATES = 50
    candidates: dict[str, list[dict[str, Any]]] = {
        s["scenario_id"]: [] for s in SCENARIO_DEFS
    }

    for npz_path in ft_files:
        d = np.load(npz_path)
        L = int(d["seq_len"])
        for t in range(L):
            a = int(d["action"][t])
            lm = d["legal_mask"][t]
            ctx = d["context"][t]

            # Check each scenario definition
            for sdef in SCENARIO_DEFS:
                sid = sdef["scenario_id"]
                if len(candidates[sid]) >= MAX_CANDIDATES:
                    continue
                if a != sdef["target_action"]:
                    continue
                if lm[a] <= 0.5:
                    continue
                if not matches_filters(a, lm, ctx, sdef["filters"]):
                    continue

                candidates[sid].append({
                    "battle_id": npz_path.stem,
                    "turn_index": t,
                    "turn_bucket": turn_bucket(float(ctx[0])),
                    "forced_switch": bool(ctx[3] > 0.5),
                    "can_tera": bool(ctx[2] > 0.5),
                    "weather_active": bool(d["field"][t][0] > 0),
                    "num_legal_original": int(np.sum(lm > 0.5)),
                    "own_team": d["own_team"][t].tolist(),
                    "opponent_team": d["opponent_team"][t].tolist(),
                    "field": d["field"][t].tolist(),
                    "context": ctx.tolist(),
                    "original_legal_mask": lm.tolist(),
                })

    # Select best candidate per scenario (prefer diverse battles, more legal actions)
    scenarios: list[dict[str, Any]] = []
    seen_battles: set[str] = set()
    skipped: list[str] = []

    for sdef in SCENARIO_DEFS:
        sid = sdef["scenario_id"]
        cands = candidates[sid]
        if not cands:
            skipped.append(sid)
            print(f"  WARNING: No candidates found for {sid} (action={sdef['target_action']}, "
                  f"family={sdef['family']})")
            continue

        # Sort: prefer unseen battles, then more legal options, then game phase variety
        cands.sort(key=lambda c: (
            c["battle_id"] in seen_battles,
            -c["num_legal_original"],
            c["turn_bucket"],
        ))
        c = cands[0]
        seen_battles.add(c["battle_id"])

        scenarios.append({
            "scenario_id": sid,
            "family": sdef["family"],
            "label_type": "hard_optimal",
            "scenario_description": sdef["description"],
            "source": {
                "battle_id": c["battle_id"],
                "turn_index": c["turn_index"],
                "source_note": "replay-grounded state from fine-tuning data with synthetic one-hot legality patch",
                "data_source": "data/fine-tuning/processed (isolated from training)",
            },
            "expected_action": sdef["target_action"],
            "expected_action_name": ACTION_NAMES[sdef["target_action"]],
            "acceptable_actions": [sdef["target_action"]],
            "metadata": {
                "turn_bucket": c["turn_bucket"],
                "forced_switch": c["forced_switch"],
                "can_tera": c["can_tera"],
                "weather_active": c["weather_active"],
                "num_legal_original": c["num_legal_original"],
                "num_legal_synthetic": 1,
            },
            "inputs": {
                "own_team": c["own_team"],
                "opponent_team": c["opponent_team"],
                "field": c["field"],
                "context": c["context"],
                "legal_mask": one_hot(sdef["target_action"]),
                "original_legal_mask": c["original_legal_mask"],
            },
        })

    # Report
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"scenarios": scenarios}, indent=2), encoding="utf-8")

    print(f"\nWrote {len(scenarios)} scenarios -> {out}")
    if skipped:
        print(f"Skipped {len(skipped)} scenarios (no matching turns): {skipped}")

    # Summary by family
    from collections import Counter
    family_counts = Counter(s["family"] for s in scenarios)
    print("\nScenarios by family:")
    for fam, count in sorted(family_counts.items()):
        print(f"  {fam}: {count}")

    for s in scenarios:
        print(f"  - {s['scenario_id']}: {s['expected_action_name']} ({s['family']})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
