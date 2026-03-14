#!/usr/bin/env python3
"""Generate comprehensive turn-by-turn battle decision reports.

Parses raw battle replay files and produces a detailed Markdown report
showing each turn's state, available moves, available switches, and
the action taken by the player.

Usage:
    python scripts/generate_battle_report.py \
        --input-dir data/fine-tuning/raw \
        --output docs/BATTLE_DECISION_REPORT.md \
        --max-turns 30 \
        --num-battles 10
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Metamon action mapping
ACTION_NAMES = {
    "0": "Move 1", "1": "Move 2", "2": "Move 3", "3": "Move 4",
    "4": "Switch to bench slot 1", "5": "Switch to bench slot 2",
    "6": "Switch to bench slot 3", "7": "Switch to bench slot 4",
    "8": "Switch to bench slot 5",
    "9": "Tera + Move 1", "10": "Tera + Move 2",
    "11": "Tera + Move 3", "12": "Tera + Move 4",
}


def format_pokemon_summary(poke: dict, is_own: bool = True) -> str:
    """Format a pokemon dict into a readable summary."""
    name = poke.get("name", "Unknown")
    hp = poke.get("hp_pct", 1.0)
    hp_pct = f"{hp * 100:.0f}%" if isinstance(hp, float) else f"{hp}%"
    types = poke.get("types", "")
    item = poke.get("item", "")
    ability = poke.get("ability", "")
    status = poke.get("status", "")
    tera = poke.get("tera_type", "")

    parts = [f"**{name}** (HP: {hp_pct})"]
    if types:
        parts.append(f"Type: {types}")
    if item and item not in ("unknownitem", "unknown"):
        parts.append(f"Item: {item}")
    if ability and ability not in ("unknownability", "unknown"):
        parts.append(f"Ability: {ability}")
    if status:
        parts.append(f"Status: {status}")
    if tera:
        parts.append(f"Tera: {tera}")

    # Stat boosts
    boosts = []
    for stat in ["atk", "spa", "def", "spd", "spe"]:
        val = poke.get(f"{stat}_boost", 0)
        if val != 0:
            sign = "+" if val > 0 else ""
            boosts.append(f"{stat.upper()}{sign}{val}")
    if boosts:
        parts.append(f"Boosts: {', '.join(boosts)}")

    return " | ".join(parts)


def format_move(move: dict) -> str:
    """Format a move dict into a readable string."""
    name = move.get("name", "Unknown")
    mtype = move.get("move_type", move.get("type", ""))
    category = move.get("category", "")
    bp = move.get("base_power", 0)
    pp_cur = move.get("current_pp", -1)
    pp_max = move.get("max_pp", -1)
    priority = move.get("priority", 0)

    parts = [f"**{name}**"]
    if mtype:
        parts.append(f"[{mtype}]")
    if category:
        parts.append(f"({category})")
    if bp and bp > 0:
        parts.append(f"BP:{bp}")
    if priority and priority != 0:
        parts.append(f"Priority:{priority}")
    if pp_cur >= 0 and pp_max >= 0:
        parts.append(f"PP:{pp_cur}/{pp_max}")

    return " ".join(parts)


def describe_action(action_str: str, state: dict) -> str:
    """Describe what an action actually does in context."""
    action_label = ACTION_NAMES.get(str(action_str), f"Action {action_str}")
    action_idx = int(action_str)

    active = state.get("player_active_pokemon", {})
    moves = active.get("moves", [])
    switches = state.get("available_switches", [])

    if action_idx <= 3:
        # Move
        if action_idx < len(moves):
            move = moves[action_idx]
            return f"{action_label}: Use **{move.get('name', '???')}**"
        return action_label
    elif action_idx <= 8:
        # Switch
        bench_idx = action_idx - 4
        if bench_idx < len(switches):
            target = switches[bench_idx]
            return f"{action_label}: Switch to **{target.get('name', '???')}**"
        return action_label
    elif action_idx <= 12:
        # Tera move
        move_idx = action_idx - 9
        if move_idx < len(moves):
            move = moves[move_idx]
            return f"{action_label}: Terastallize + Use **{move.get('name', '???')}**"
        return action_label
    return action_label


def generate_battle_report(battle_data: dict, filename: str) -> str:
    """Generate a comprehensive turn-by-turn report for a single battle."""
    states = battle_data.get("states", [])
    actions = battle_data.get("actions", [])

    # Parse filename for metadata
    basename = filename.replace(".json.lz4", "").replace(".json", "")
    parts = basename.split("_")
    result = parts[-1] if parts else "?"
    elo = parts[1] if len(parts) > 1 else "?"

    lines = []
    lines.append(f"### Battle: `{filename}`")
    lines.append(f"- **Result**: {result}")
    lines.append(f"- **Elo**: {elo}")
    lines.append(f"- **Total Turns**: {len(states)}")
    lines.append("")

    # Extract teams from first state + team preview
    if states:
        first_state = states[0]
        player_active = first_state.get("player_active_pokemon", {})
        opp_active = first_state.get("opponent_active_pokemon", {})
        switches = first_state.get("available_switches", [])
        opp_preview = first_state.get("opponent_teampreview", [])

        lines.append("#### Player's Team")
        lines.append(f"1. {format_pokemon_summary(player_active)} *(Lead)*")
        for i, sw in enumerate(switches):
            lines.append(f"{i+2}. {format_pokemon_summary(sw)}")
        lines.append("")

        lines.append("#### Opponent's Team (from Team Preview)")
        if opp_preview:
            for i, opp in enumerate(opp_preview):
                if isinstance(opp, str):
                    lines.append(f"{i+1}. **{opp}**")
                elif isinstance(opp, dict):
                    lines.append(f"{i+1}. **{opp.get('name', '???')}**")
        else:
            lines.append(f"1. {format_pokemon_summary(opp_active)} *(Lead)*")
        lines.append("")

    # Turn-by-turn
    lines.append("#### Turn-by-Turn Decisions")
    lines.append("")

    num_action_turns = min(len(states), len(actions))
    for turn_idx in range(num_action_turns):
        state = states[turn_idx]
        action = str(actions[turn_idx])

        player_active = state.get("player_active_pokemon", {})
        opp_active = state.get("opponent_active_pokemon", {})
        moves = player_active.get("moves", [])
        switches = state.get("available_switches", [])
        weather = state.get("weather", "")
        terrain = state.get("battle_field", "")
        forced_switch = state.get("forced_switch", False)
        can_tera = state.get("can_tera", False)
        player_prev = state.get("player_prev_move", {})
        opp_prev = state.get("opponent_prev_move", {})
        player_conds = state.get("player_conditions", "")
        opp_conds = state.get("opponent_conditions", "")

        lines.append(f"---")
        lines.append(f"**Turn {turn_idx + 1}**" + (" *(Forced Switch)*" if forced_switch else ""))
        lines.append("")

        # Field state
        field_info = []
        if weather:
            field_info.append(f"Weather: {weather}")
        if terrain:
            field_info.append(f"Terrain/Field: {terrain}")
        if player_conds:
            field_info.append(f"Player side: {player_conds}")
        if opp_conds:
            field_info.append(f"Opponent side: {opp_conds}")
        if field_info:
            lines.append(f"*Field*: {' | '.join(field_info)}")
            lines.append("")

        # Previous moves
        if player_prev and player_prev.get("name"):
            lines.append(f"*Last turn*: Player used {player_prev.get('name', '?')} | Opponent used {opp_prev.get('name', '?') if opp_prev else '?'}")
            lines.append("")

        # Active Pokemon
        lines.append(f"**Player Active**: {format_pokemon_summary(player_active)}")
        lines.append(f"**Opponent Active**: {format_pokemon_summary(opp_active, is_own=False)}")
        lines.append("")

        # Available moves
        if moves and not forced_switch:
            lines.append("**Available Moves:**")
            for i, move in enumerate(moves):
                marker = " <-- CHOSEN" if int(action) == i else ""
                tera_marker = ""
                if can_tera and int(action) == i + 9:
                    tera_marker = " <-- CHOSEN (with Tera)"
                lines.append(f"  - Slot {i+1}: {format_move(move)}{marker}{tera_marker}")
            if can_tera:
                lines.append(f"  - *Terastallize available* (can use Tera + any move)")
            lines.append("")

        # Available switches
        if switches:
            lines.append("**Available Switches:**")
            for i, sw in enumerate(switches):
                sw_action = i + 4  # Metamon switch actions are 4-8
                marker = " <-- CHOSEN" if int(action) == sw_action else ""
                hp = sw.get("hp_pct", 1.0)
                hp_str = f"{hp * 100:.0f}%" if isinstance(hp, float) else f"{hp}%"
                status = sw.get("status", "")
                status_str = f" [{status}]" if status else ""
                lines.append(f"  - Bench {i+1}: **{sw.get('name', '???')}** (HP: {hp_str}){status_str}{marker}")
            lines.append("")

        # Action taken
        action_desc = describe_action(action, state)
        lines.append(f"**Decision**: {action_desc}")
        lines.append("")

    # Final state (if there's one more state than actions - the terminal state)
    if len(states) > num_action_turns:
        final_state = states[-1]
        won = final_state.get("battle_won", False)
        lost = final_state.get("battle_lost", False)
        if won:
            lines.append("**Battle Result: WIN**")
        elif lost:
            lines.append("**Battle Result: LOSS**")
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate battle decision reports")
    parser.add_argument("--input-dir", type=str, default="data/fine-tuning/raw")
    parser.add_argument("--output", type=str, default="docs/BATTLE_DECISION_REPORT.md")
    parser.add_argument("--max-turns", type=int, default=30, help="Max turns threshold")
    parser.add_argument("--num-battles", type=int, default=10, help="Number of battles to report")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    import lz4.frame

    input_dir = Path(args.input_dir)
    files = sorted(input_dir.glob("*.json*"))

    # Load all battles and filter by turn count
    logger.info(f"Scanning {len(files)} battles for those under {args.max_turns} turns...")
    short_battles = []

    for filepath in files:
        try:
            if filepath.name.endswith(".json.lz4"):
                with lz4.frame.open(str(filepath), "rb") as f:
                    data = json.loads(f.read().decode("utf-8"))
            else:
                with open(filepath, "r") as f:
                    data = json.load(f)

            num_turns = len(data.get("states", []))
            if 5 <= num_turns < args.max_turns:
                short_battles.append((filepath.name, data, num_turns))
        except Exception as e:
            logger.warning(f"Error reading {filepath.name}: {e}")

    logger.info(f"Found {len(short_battles)} battles under {args.max_turns} turns")

    # Sort by turn count for variety, then pick first num_battles
    # Aim for variety: pick battles with different turn lengths
    short_battles.sort(key=lambda x: x[2])

    # Sample evenly across the turn-count range
    import random
    rng = random.Random(args.seed)

    if len(short_battles) > args.num_battles:
        # Pick evenly spaced + some randomness
        step = len(short_battles) / args.num_battles
        selected = []
        for i in range(args.num_battles):
            idx = min(int(i * step), len(short_battles) - 1)
            selected.append(short_battles[idx])
        short_battles = selected
    else:
        short_battles = short_battles[:args.num_battles]

    # Generate report
    report_lines = []
    report_lines.append("# Battle Decision Report — Fine-Tuning Data Analysis")
    report_lines.append("")
    report_lines.append(f"*Generated from {args.input_dir} — {len(short_battles)} battles with < {args.max_turns} turns*")
    report_lines.append("")
    report_lines.append("This report provides a comprehensive turn-by-turn account of player decisions")
    report_lines.append("in Gen 9 OU battles. For each turn, we list the active Pokemon, available moves")
    report_lines.append("(with type, category, base power, PP), available switches, field conditions,")
    report_lines.append("and the decision actually taken by the player. This data is crucial for")
    report_lines.append("understanding the decision space that the P8-Lean imitation learning model")
    report_lines.append("must learn to navigate.")
    report_lines.append("")
    report_lines.append("## Table of Contents")
    report_lines.append("")
    for i, (fname, data, nt) in enumerate(short_battles):
        basename = fname.replace(".json.lz4", "").replace(".json", "")
        result = basename.split("_")[-1] if "_" in basename else "?"
        report_lines.append(f"{i+1}. [{basename}](#{i+1}-battle-{i+1}) — {nt} turns, {result}")
    report_lines.append("")

    for i, (fname, data, nt) in enumerate(short_battles):
        logger.info(f"Generating report for battle {i+1}/{len(short_battles)}: {fname} ({nt} turns)")
        report_lines.append(f"## {i+1}. Battle {i+1}")
        report_lines.append("")
        report_lines.append(generate_battle_report(data, fname))
        report_lines.append("")
        report_lines.append("---")
        report_lines.append("")

    # Summary statistics
    report_lines.append("## Summary Statistics")
    report_lines.append("")
    total_turns = sum(nt for _, _, nt in short_battles)
    wins = sum(1 for fname, _, _ in short_battles if "WIN" in fname)
    losses = sum(1 for fname, _, _ in short_battles if "LOSS" in fname)
    avg_turns = total_turns / len(short_battles) if short_battles else 0
    report_lines.append(f"- **Battles analyzed**: {len(short_battles)}")
    report_lines.append(f"- **Total turns**: {total_turns}")
    report_lines.append(f"- **Average turns per battle**: {avg_turns:.1f}")
    report_lines.append(f"- **Wins**: {wins}")
    report_lines.append(f"- **Losses**: {losses}")
    report_lines.append("")

    # Write report
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(report_lines))

    logger.info(f"Report written to {output_path} ({len(report_lines)} lines)")


if __name__ == "__main__":
    main()
