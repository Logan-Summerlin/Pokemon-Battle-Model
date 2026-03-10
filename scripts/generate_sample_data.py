#!/usr/bin/env python3
"""Generate synthetic sample battles in Metamon format.

Creates realistic-looking Gen 9 OU battle data for pipeline validation.
The generated data follows the Metamon UniversalState JSON format so
the full data pipeline can be tested end-to-end.

Usage:
    python scripts/generate_sample_data.py
    python scripts/generate_sample_data.py --num-battles 100 --output-dir data/raw
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Metagame data for realistic generation ────────────────────────────────

OU_POKEMON = [
    ("Great Tusk", "Ground/Fighting", 115, 131, 131, 53, 63, 87),
    ("Gholdengo", "Steel/Ghost", 87, 60, 95, 133, 91, 84),
    ("Kingambit", "Dark/Steel", 100, 135, 120, 60, 85, 50),
    ("Dragapult", "Dragon/Ghost", 88, 120, 75, 100, 75, 142),
    ("Iron Valiant", "Fairy/Fighting", 74, 130, 90, 120, 60, 116),
    ("Heatran", "Fire/Steel", 91, 90, 106, 130, 106, 77),
    ("Toxapex", "Poison/Water", 50, 63, 152, 53, 142, 35),
    ("Corviknight", "Flying/Steel", 98, 87, 105, 53, 85, 67),
    ("Garganacl", "Rock", 100, 100, 130, 45, 90, 35),
    ("Gliscor", "Ground/Flying", 75, 95, 125, 45, 75, 95),
    ("Iron Moth", "Fire/Poison", 80, 70, 60, 140, 110, 110),
    ("Zamazenta", "Fighting", 92, 130, 115, 80, 115, 138),
    ("Roaring Moon", "Dragon/Dark", 105, 139, 71, 55, 101, 119),
    ("Clefable", "Fairy", 95, 70, 73, 95, 90, 60),
    ("Skeledirge", "Fire/Ghost", 104, 75, 100, 110, 75, 66),
    ("Slowking-Galar", "Poison/Psychic", 95, 65, 80, 110, 110, 30),
    ("Ting-Lu", "Dark/Ground", 155, 110, 125, 55, 80, 45),
    ("Dondozo", "Water", 150, 100, 115, 65, 65, 35),
    ("Annihilape", "Fighting/Ghost", 110, 115, 80, 50, 90, 90),
    ("Samurott-Hisui", "Water/Dark", 90, 108, 80, 100, 65, 85),
    ("Ogerpon-Wellspring", "Water/Grass", 80, 120, 84, 60, 96, 110),
    ("Landorus-Therian", "Ground/Flying", 89, 145, 90, 105, 80, 91),
    ("Iron Treads", "Ground/Steel", 90, 112, 120, 72, 70, 106),
    ("Raging Bolt", "Electric/Dragon", 125, 73, 91, 137, 89, 75),
    ("Kyurem", "Dragon/Ice", 125, 130, 90, 130, 90, 95),
    ("Alomomola", "Water", 165, 75, 80, 40, 45, 65),
    ("Tornadus-Therian", "Flying", 79, 100, 80, 110, 90, 121),
    ("Darkrai", "Dark", 70, 90, 90, 135, 90, 125),
    ("Volcarona", "Bug/Fire", 85, 60, 65, 135, 105, 100),
    ("Tapu Lele", "Psychic/Fairy", 70, 85, 75, 130, 115, 95),
]

COMMON_MOVES = {
    "Great Tusk": ["Headlong Rush", "Close Combat", "Ice Spinner", "Rapid Spin", "Knock Off", "Stealth Rock"],
    "Gholdengo": ["Make It Rain", "Shadow Ball", "Thunderbolt", "Nasty Plot", "Recover", "Trick"],
    "Kingambit": ["Sucker Punch", "Iron Head", "Kowtow Cleave", "Swords Dance", "Low Kick"],
    "Dragapult": ["Shadow Ball", "Draco Meteor", "U-turn", "Flamethrower", "Hex", "Thunder Wave"],
    "Heatran": ["Magma Storm", "Flash Cannon", "Earth Power", "Stealth Rock", "Toxic", "Taunt"],
    "Toxapex": ["Scald", "Recover", "Haze", "Toxic Spikes", "Knock Off", "Baneful Bunker"],
    "Corviknight": ["Brave Bird", "U-turn", "Roost", "Defog", "Body Press", "Iron Defense"],
    "Garganacl": ["Salt Cure", "Recover", "Stealth Rock", "Body Press", "Iron Defense", "Earthquake"],
    "Gliscor": ["Earthquake", "Facade", "Swords Dance", "Roost", "Knock Off", "Toxic"],
    "Clefable": ["Moonblast", "Soft-Boiled", "Stealth Rock", "Thunder Wave", "Knock Off", "Calm Mind"],
}

COMMON_ITEMS = [
    "Leftovers", "Choice Scarf", "Choice Band", "Choice Specs",
    "Heavy-Duty Boots", "Life Orb", "Assault Vest", "Rocky Helmet",
    "Eviolite", "Focus Sash", "Booster Energy", "Air Balloon",
    "Black Sludge", "Covert Cloak", "Loaded Dice", "Sitrus Berry",
]

COMMON_ABILITIES = [
    "Intimidate", "Regenerator", "Natural Cure", "Pressure",
    "Levitate", "Flash Fire", "Protosynthesis", "Quark Drive",
    "Guts", "Sturdy", "Magic Guard", "Good as Gold",
    "Supreme Overlord", "Unaware", "Prankster", "Beast Boost",
]

TERA_TYPES = [
    "Normal", "Fire", "Water", "Electric", "Grass", "Ice",
    "Fighting", "Poison", "Ground", "Flying", "Psychic",
    "Bug", "Rock", "Ghost", "Dragon", "Dark", "Steel", "Fairy",
]

WEATHER_OPTIONS = ["", "", "", "", "RainDance", "SunnyDay", "Sandstorm", "Snow"]
TERRAIN_OPTIONS = ["", "", "", "", "Electric Terrain", "Grassy Terrain", "Psychic Terrain", "Misty Terrain"]

STATUS_OPTIONS = ["", "", "", "", "", "brn", "par", "tox", "psn", "slp"]


def random_move(name: str = "") -> dict:
    if not name:
        all_moves = []
        for moves in COMMON_MOVES.values():
            all_moves.extend(moves)
        name = random.choice(all_moves) if all_moves else "Tackle"

    categories = {"Physical": 80, "Special": 80, "Status": 0}
    cat = random.choice(list(categories.keys()))
    bp = random.choice([0, 40, 60, 70, 80, 90, 100, 120]) if cat != "Status" else 0

    return {
        "name": name,
        "move_type": random.choice(TERA_TYPES),
        "category": cat,
        "base_power": bp,
        "accuracy": random.choice([100, 100, 100, 95, 90, 85, 80]),
        "priority": random.choice([0, 0, 0, 0, 0, 1, -1]),
        "current_pp": random.randint(1, 32),
        "max_pp": 32,
    }


def random_pokemon(species: str | None = None) -> dict:
    if species is None:
        entry = random.choice(OU_POKEMON)
    else:
        entry = next((e for e in OU_POKEMON if e[0] == species), random.choice(OU_POKEMON))

    name, types, base_hp, base_atk, base_def, base_spa, base_spd, base_spe = entry

    # Pick moves
    species_moves = COMMON_MOVES.get(name, [])
    if species_moves:
        n_moves = min(4, len(species_moves))
        move_names = random.sample(species_moves, n_moves)
    else:
        move_names = [f"Move{i}" for i in range(4)]
    moves = [random_move(m) for m in move_names]

    return {
        "name": name,
        "hp_pct": round(random.uniform(0.0, 1.0), 3),
        "types": types,
        "item": random.choice(COMMON_ITEMS),
        "ability": random.choice(COMMON_ABILITIES),
        "lvl": 100,
        "status": random.choice(STATUS_OPTIONS),
        "effect": "",
        "moves": moves,
        "atk_boost": random.choice([0, 0, 0, 0, 1, 2, -1]),
        "spa_boost": random.choice([0, 0, 0, 0, 1, 2, -1]),
        "def_boost": random.choice([0, 0, 0, 0, 1, -1]),
        "spd_boost": random.choice([0, 0, 0, 0, 1, -1]),
        "spe_boost": random.choice([0, 0, 0, 0, 1, -1]),
        "accuracy_boost": 0,
        "evasion_boost": 0,
        "base_atk": base_atk,
        "base_spa": base_spa,
        "base_def": base_def,
        "base_spd": base_spd,
        "base_spe": base_spe,
        "base_hp": base_hp,
        "tera_type": random.choice(TERA_TYPES),
        "base_species": name,
    }


def generate_battle(battle_id: int, rng: random.Random) -> tuple[dict, str]:
    """Generate a synthetic battle.

    Returns (battle_dict, filename).
    """
    # Pick teams (6 pokemon each, from OU pool)
    player_species = rng.sample([p[0] for p in OU_POKEMON], min(6, len(OU_POKEMON)))
    opponent_species = rng.sample([p[0] for p in OU_POKEMON], min(6, len(OU_POKEMON)))

    num_turns = rng.randint(5, 40)
    won = rng.random() > 0.5
    elo = rng.choice([1500, 1550, 1600, 1650, 1700, 1750, 1800, 1850, 1900])

    states = []
    actions = []

    for t in range(num_turns):
        is_last = t == num_turns - 1
        active_idx = min(t % len(player_species), len(player_species) - 1)
        opp_active_idx = min(t % len(opponent_species), len(opponent_species) - 1)

        player_active = random_pokemon(player_species[active_idx])
        player_active["hp_pct"] = max(0.05, 1.0 - (t * 0.03) + rng.uniform(-0.1, 0.1))
        if is_last and not won:
            player_active["hp_pct"] = 0.0

        opponent_active = random_pokemon(opponent_species[opp_active_idx])
        opponent_active["hp_pct"] = max(0.05, 1.0 - (t * 0.025) + rng.uniform(-0.1, 0.1))
        if is_last and won:
            opponent_active["hp_pct"] = 0.0

        # Available switches (bench pokemon)
        bench = [
            random_pokemon(sp) for sp in player_species
            if sp != player_species[active_idx]
        ]
        for b in bench:
            b["hp_pct"] = max(0.0, round(rng.uniform(0.3, 1.0), 3))

        # Opponent team preview
        opp_preview = [random_pokemon(sp) for sp in opponent_species]

        # Previous moves
        player_prev = random_move() if t > 0 else None
        opponent_prev = random_move() if t > 0 else None

        opp_remaining = max(1, len(opponent_species) - (t // 8))
        forced_switch = rng.random() < 0.05
        can_tera = t < num_turns // 2

        state = {
            "format": "gen9ou",
            "player_active_pokemon": player_active,
            "opponent_active_pokemon": opponent_active,
            "available_switches": bench[:5],
            "player_prev_move": player_prev,
            "opponent_prev_move": opponent_prev,
            "opponents_remaining": opp_remaining,
            "player_conditions": "",
            "opponent_conditions": "",
            "weather": rng.choice(WEATHER_OPTIONS),
            "battle_field": rng.choice(TERRAIN_OPTIONS),
            "forced_switch": forced_switch,
            "can_tera": can_tera,
            "battle_won": won and is_last,
            "battle_lost": (not won) and is_last,
            "opponent_teampreview": opp_preview,
        }

        # Add some side conditions occasionally
        if rng.random() < 0.3:
            state["player_conditions"] = rng.choice(["Stealth Rock", "Spikes:2", ""])
        if rng.random() < 0.3:
            state["opponent_conditions"] = rng.choice(["Stealth Rock", "Toxic Spikes:1", ""])

        states.append(state)

        # Generate action
        if forced_switch:
            action = f"switch {rng.randint(1, min(5, len(bench)))}"
        else:
            action = rng.choice(["move 1", "move 2", "move 3", "move 4", "switch 1", "switch 2"])

        actions.append(action)

    battle_data = {"states": states, "actions": actions}

    # Generate filename in Metamon format
    date = f"{rng.randint(1,28):02d}-{rng.randint(1,12):02d}-{rng.randint(2022,2026)}"
    result = "WIN" if won else "LOSS"
    filename = f"battle-gen9ou-{battle_id}_{elo}_Player_vs_Opponent_{date}_{result}.json"

    return battle_data, filename


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic sample battles")
    parser.add_argument("--num-battles", type=int, default=100,
                        help="Number of battles to generate (default: 100)")
    parser.add_argument("--output-dir", type=str, default="data/raw",
                        help="Output directory (default: data/raw)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    logger.info(f"Generating {args.num_battles} synthetic battles...")
    for i in range(args.num_battles):
        battle_data, filename = generate_battle(i, rng)
        filepath = Path(args.output_dir) / filename
        with open(filepath, "w") as f:
            json.dump(battle_data, f)

        if (i + 1) % 100 == 0:
            logger.info(f"  Generated {i + 1}/{args.num_battles}")

    logger.info(f"Done! {args.num_battles} battles saved to {args.output_dir}")


if __name__ == "__main__":
    main()
