#!/usr/bin/env python3
"""Generate synthetic sample battles in Metamon format.

Creates realistic-looking Gen 3 OU battle data for pipeline validation.
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

# ── Gen 3 OU metagame data for realistic generation ──────────────────────

# (species, types, base_hp, base_atk, base_def, base_spa, base_spd, base_spe)
OU_POKEMON = [
    ("Tyranitar", "Rock/Dark", 100, 134, 110, 95, 100, 61),
    ("Salamence", "Dragon/Flying", 95, 135, 80, 110, 80, 100),
    ("Metagross", "Steel/Psychic", 80, 135, 130, 95, 90, 70),
    ("Swampert", "Water/Ground", 100, 110, 90, 85, 90, 60),
    ("Skarmory", "Steel/Flying", 65, 80, 140, 40, 70, 70),
    ("Blissey", "Normal", 255, 10, 10, 75, 105, 55),
    ("Gengar", "Ghost/Poison", 60, 65, 60, 130, 75, 110),
    ("Starmie", "Water/Psychic", 60, 75, 85, 100, 85, 115),
    ("Jirachi", "Steel/Psychic", 100, 100, 100, 100, 100, 100),
    ("Celebi", "Psychic/Grass", 100, 100, 100, 100, 100, 100),
    ("Suicune", "Water", 100, 75, 115, 90, 115, 85),
    ("Aerodactyl", "Rock/Flying", 80, 105, 65, 60, 75, 130),
    ("Dugtrio", "Ground", 35, 80, 50, 50, 70, 120),
    ("Milotic", "Water", 95, 60, 79, 100, 125, 81),
    ("Magneton", "Electric/Steel", 50, 60, 95, 120, 70, 70),
    ("Snorlax", "Normal", 160, 110, 65, 65, 110, 30),
    ("Heracross", "Bug/Fighting", 80, 125, 75, 40, 95, 85),
    ("Flygon", "Ground/Dragon", 80, 100, 80, 80, 80, 100),
    ("Claydol", "Ground/Psychic", 60, 70, 105, 70, 120, 75),
    ("Forretress", "Bug/Steel", 75, 90, 140, 60, 60, 40),
    ("Zapdos", "Electric/Flying", 90, 90, 85, 125, 90, 100),
    ("Jolteon", "Electric", 65, 65, 60, 110, 95, 130),
    ("Weezing", "Poison", 65, 90, 120, 85, 70, 60),
    ("Dusclops", "Ghost", 40, 70, 130, 60, 130, 25),
    ("Breloom", "Grass/Fighting", 60, 130, 80, 60, 60, 70),
    ("Hariyama", "Fighting", 144, 120, 60, 40, 60, 50),
    ("Moltres", "Fire/Flying", 90, 100, 90, 125, 85, 90),
    ("Vaporeon", "Water", 130, 65, 60, 110, 95, 65),
    ("Alakazam", "Psychic", 55, 50, 45, 135, 85, 120),
    ("Gyarados", "Water/Flying", 95, 125, 79, 60, 100, 81),
]

COMMON_MOVES = {
    "Tyranitar": ["Rock Slide", "Earthquake", "Crunch", "Dragon Dance", "Pursuit", "Focus Punch"],
    "Salamence": ["Earthquake", "Dragon Claw", "Fire Blast", "Dragon Dance", "Rock Slide", "Hidden Power"],
    "Metagross": ["Meteor Mash", "Earthquake", "Explosion", "Rock Slide", "Agility", "Pursuit"],
    "Swampert": ["Earthquake", "Ice Beam", "Surf", "Protect", "Toxic", "Roar"],
    "Skarmory": ["Spikes", "Whirlwind", "Drill Peck", "Rest", "Toxic", "Protect"],
    "Blissey": ["Soft-Boiled", "Toxic", "Ice Beam", "Seismic Toss", "Aromatherapy", "Thunder Wave"],
    "Gengar": ["Thunderbolt", "Ice Punch", "Hypnosis", "Will-O-Wisp", "Substitute", "Focus Punch"],
    "Starmie": ["Surf", "Thunderbolt", "Ice Beam", "Rapid Spin", "Recover", "Psychic"],
    "Jirachi": ["Body Slam", "Fire Punch", "Psychic", "Calm Mind", "Wish", "Protect"],
    "Celebi": ["Psychic", "Giga Drain", "Leech Seed", "Recover", "Calm Mind", "Baton Pass"],
}

COMMON_ITEMS = [
    "Leftovers", "Choice Band", "Lum Berry", "Liechi Berry",
    "Salac Berry", "Petaya Berry", "Focus Band", "Macho Brace",
    "Shell Bell", "White Herb", "Mental Herb", "Sitrus Berry",
    "Chesto Berry", "Scope Lens", "King's Rock", "Brightpowder",
]

COMMON_ABILITIES = [
    "Intimidate", "Sand Stream", "Clear Body", "Torrent",
    "Keen Eye", "Natural Cure", "Levitate", "Serene Grace",
    "Pressure", "Water Absorb", "Guts", "Thick Fat",
    "Inner Focus", "Synchronize", "Sturdy", "Magnet Pull",
]

# Gen 3 types (no Fairy)
GEN3_TYPES = [
    "Normal", "Fire", "Water", "Electric", "Grass", "Ice",
    "Fighting", "Poison", "Ground", "Flying", "Psychic",
    "Bug", "Rock", "Ghost", "Dragon", "Dark", "Steel",
]

# Gen 3 weather: permanent from abilities, no terrains
WEATHER_OPTIONS = ["", "", "", "", "RainDance", "SunnyDay", "Sandstorm"]

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
        "move_type": random.choice(GEN3_TYPES),
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
    elo = rng.choice([1300, 1350, 1400, 1450, 1500, 1550, 1600, 1650, 1700])

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

        # Previous moves
        player_prev = random_move() if t > 0 else None
        opponent_prev = random_move() if t > 0 else None

        opp_remaining = max(1, len(opponent_species) - (t // 8))
        forced_switch = rng.random() < 0.05

        state = {
            "format": "gen3ou",
            "player_active_pokemon": player_active,
            "opponent_active_pokemon": opponent_active,
            "available_switches": bench[:5],
            "player_prev_move": player_prev,
            "opponent_prev_move": opponent_prev,
            "opponents_remaining": opp_remaining,
            "player_conditions": "",
            "opponent_conditions": "",
            "weather": rng.choice(WEATHER_OPTIONS),
            "battle_field": "",
            "forced_switch": forced_switch,
            "can_tera": False,
            "battle_won": won and is_last,
            "battle_lost": (not won) and is_last,
            "opponent_teampreview": [],
        }

        # Add Gen 3 side conditions (Spikes only, no Stealth Rock/Toxic Spikes)
        if rng.random() < 0.3:
            state["player_conditions"] = rng.choice(["Spikes:1", "Spikes:2", "Spikes:3", ""])
        if rng.random() < 0.3:
            state["opponent_conditions"] = rng.choice(["Spikes:1", "Spikes:2", ""])

        states.append(state)

        # Generate action (9 actions: move 1-4, switch 1-5)
        if forced_switch:
            action = f"switch {rng.randint(1, min(5, len(bench)))}"
        else:
            action = rng.choice(["move 1", "move 2", "move 3", "move 4", "switch 1", "switch 2"])

        actions.append(action)

    battle_data = {"states": states, "actions": actions}

    # Generate filename in Metamon format
    date = f"{rng.randint(1,28):02d}-{rng.randint(1,12):02d}-{rng.randint(2022,2026)}"
    result = "WIN" if won else "LOSS"
    filename = f"battle-gen3ou-{battle_id}_{elo}_Player_vs_Opponent_{date}_{result}.json"

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

    logger.info(f"Generating {args.num_battles} synthetic Gen 3 OU battles...")
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
