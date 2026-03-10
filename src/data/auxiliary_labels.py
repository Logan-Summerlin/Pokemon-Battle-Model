"""Auxiliary label extraction for hidden-info prediction head.

Extracts training labels for the auxiliary head from replay data.
Labels are derived from information that was EVENTUALLY revealed
during the battle — never from omniscient data.

Label types:
    - Item class: Categorizes opponent items into ~50 classes
    - Speed bucket: Ordinal speed category (very fast -> very slow)
    - Role archetype: Sweeper, wall, pivot, etc. (inferred from moves/stats)
    - Tera category: Offensive, defensive, STAB, coverage
    - Move family presence: Priority, recovery, hazards, status, etc.

Key constraint (Hidden Information Doctrine):
    If an item was revealed on turn 15, we use that label for turns 1-14
    of THAT game. If it was never revealed, we use -1 (unknown/masked).
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from src.data.observation import MAX_TEAM_SIZE, UNKNOWN
from src.data.replay_parser import ParsedBattle, ParsedPokemon, ParsedTurnState

logger = logging.getLogger(__name__)


# ── Item classification ──────────────────────────────────────────────────

# Map individual items to broader item classes for the auxiliary head.
# These classes group items by competitive function.
ITEM_CLASS_MAP: dict[str, int] = {}
ITEM_CLASSES: list[str] = [
    "heavydutyboots",       # 0
    "leftovers",            # 1
    "lifeorb",              # 2
    "choiceband",           # 3
    "choicespecs",          # 4
    "choicescarf",          # 5
    "assaultvest",          # 6
    "focussash",            # 7
    "rockyhelmet",          # 8
    "blacksludge",          # 9
    "eviolite",             # 10
    "loadeddice",           # 11
    "clearamulet",          # 12
    "covertcloak",          # 13
    "safetygoggles",        # 14
    "shedshell",            # 15
    "airballoon",           # 16
    "weaknesspolicy",       # 17
    "sitrusberry",          # 18
    "lumberry",             # 19
    "expertbelt",           # 20
    "whiteherb",            # 21
    "redcard",              # 22
    "ejectbutton",          # 23
    "ejectpack",            # 24
    "mentalherb",           # 25
    "powerherb",            # 26
    "throatspray",          # 27
    "mirrorherb",           # 28
    "boosterenergy",        # 29
    "flameorb",             # 30
    "toxicorb",             # 31
    "stickybarb",           # 32
    "lightclay",            # 33
    "heatrock",             # 34
    "damprock",             # 35
    "smoothrock",           # 36
    "icyrock",              # 37
    "terrainextender",      # 38
    "muscleband",           # 39
    "wiseglasses",          # 40
    "scopelens",            # 41
    "metronome",            # 42
    "shellbell",            # 43
    "custapberry",          # 44
    "aguavberry",           # 45  (pinch berries)
    "earthplate",           # 46  (type-boosting plates)
    "colburberry",          # 47  (resist berries)
    "other",                # 48
    "noitem",               # 49
]

# Build reverse mapping
for i, item_name in enumerate(ITEM_CLASSES):
    ITEM_CLASS_MAP[item_name] = i

# Type-boosting plates/incenses map to earthplate class
_TYPE_PLATES = [
    "earthplate", "meadowplate", "dracoplate", "fistplate", "flameplate",
    "icicleplate", "insectplate", "ironplate", "mindplate", "pixieplate",
    "skyplate", "splashplate", "spookyplate", "stoneplate", "toxicplate",
    "zapplate", "charcoal", "mysticwater", "magnet", "miracleseed",
    "nevermeltice", "blackbelt", "poisonbarb", "sharpbeak", "silkscarf",
    "silverpowder", "softsand", "spelltag", "twistedspoon", "dragonfang",
    "blackglasses", "hardstone", "metalcoat", "oddincense", "seaincense",
    "waveincense", "roseincense", "rockincense",
]
for plate in _TYPE_PLATES:
    if plate not in ITEM_CLASS_MAP:
        ITEM_CLASS_MAP[plate] = ITEM_CLASS_MAP["earthplate"]

# Resist berries map to colburberry class
_RESIST_BERRIES = [
    "colburberry", "babiriberry", "chartiberry", "chilanberry",
    "chopleberry", "cobaberry", "habanberry", "kasibberry", "kebiaberry",
    "occaberry", "passhoberry", "payapaberry", "rindoberry", "roseliberry",
    "shucaberry", "tangaberry", "wacanberry", "yacheberry",
]
for berry in _RESIST_BERRIES:
    if berry not in ITEM_CLASS_MAP:
        ITEM_CLASS_MAP[berry] = ITEM_CLASS_MAP["colburberry"]

# Pinch berries
_PINCH_BERRIES = [
    "aguavberry", "figyberry", "iapapaberry", "magoberry", "wikiberry",
]
for berry in _PINCH_BERRIES:
    if berry not in ITEM_CLASS_MAP:
        ITEM_CLASS_MAP[berry] = ITEM_CLASS_MAP["aguavberry"]

NUM_ITEM_CLASSES = len(ITEM_CLASSES)  # 50


def classify_item(item_name: str) -> int:
    """Map an item name to its class index."""
    if not item_name or item_name.lower() in ("", "none", "unknown", "unknownitem"):
        return ITEM_CLASS_MAP["noitem"]
    normalized = item_name.lower().replace(" ", "").replace("-", "").replace("_", "")
    return ITEM_CLASS_MAP.get(normalized, ITEM_CLASS_MAP["other"])


# ── Speed bucket classification ──────────────────────────────────────────

# Speed buckets based on base speed stat thresholds (Gen 9 OU meta)
# Very fast (>=115), Fast (95-114), Medium (70-94), Slow (45-69), Very slow (<45)
SPEED_THRESHOLDS = [115, 95, 70, 45]  # Boundaries between buckets
NUM_SPEED_BUCKETS = 5

SPEED_BUCKET_NAMES = ["very_fast", "fast", "medium", "slow", "very_slow"]


def classify_speed(base_spe: int) -> int:
    """Map base speed stat to a speed bucket index."""
    if base_spe >= SPEED_THRESHOLDS[0]:
        return 0  # very fast
    elif base_spe >= SPEED_THRESHOLDS[1]:
        return 1  # fast
    elif base_spe >= SPEED_THRESHOLDS[2]:
        return 2  # medium
    elif base_spe >= SPEED_THRESHOLDS[3]:
        return 3  # slow
    else:
        return 4  # very slow


# ── Role archetype classification ────────────────────────────────────────

NUM_ROLE_ARCHETYPES = 8
ROLE_NAMES = [
    "physical_sweeper",    # 0
    "special_sweeper",     # 1
    "mixed_attacker",      # 2
    "physical_wall",       # 3
    "special_wall",        # 4
    "pivot",               # 5
    "hazard_setter",       # 6
    "support",             # 7
]

# Known pivoting moves
_PIVOT_MOVES = {"uturn", "voltswitch", "flipturm", "flipturn", "teleport", "partingshot", "batonpass"}

# Known hazard-setting moves
_HAZARD_MOVES = {"stealthrock", "spikes", "toxicspikes", "stickyweb", "cometpunch"}

# Recovery moves
_RECOVERY_MOVES = {
    "recover", "roost", "softboiled", "moonlight", "morningsun", "synthesis",
    "slackoff", "shoreup", "milkdrink", "wish", "rest", "strengthsap",
}

# Status/support moves
_SUPPORT_MOVES = {
    "willowisp", "thunderwave", "toxic", "aromatherapy", "healbell",
    "defog", "rapidspin", "haze", "whirlwind", "roar", "yawn",
    "encore", "taunt", "trick", "knockoff", "reflect", "lightscreen",
    "auroraveil",
}

# Setup moves
_SETUP_MOVES = {
    "swordsdance", "nastyplot", "calmmind", "dragondance", "bulkup",
    "irondefense", "amnesia", "cosmicpower", "shellsmash", "quiverdance",
    "coil", "curse", "agility", "autotomize", "shiftgear",
    "tidyup", "victorydance",
}


def _normalize_move(name: str) -> str:
    """Normalize move name for matching."""
    return name.lower().replace(" ", "").replace("-", "").replace("_", "")


def classify_role(
    poke: ParsedPokemon,
    moves_used: list[str] | None = None,
) -> int:
    """Classify a pokemon's competitive role based on stats and moves.

    Uses base stats + known move pool to infer role archetype.
    """
    if poke is None:
        return 7  # support as default

    # Normalize moves
    all_moves: set[str] = set()
    if moves_used:
        all_moves.update(_normalize_move(m) for m in moves_used)
    for m in poke.moves:
        if m.name:
            all_moves.update([_normalize_move(m.name)])

    # Check for pivot moves
    if all_moves & _PIVOT_MOVES:
        return 5  # pivot

    # Check for hazard setter
    if all_moves & _HAZARD_MOVES:
        return 6  # hazard_setter

    # Check attack stats
    atk = poke.base_atk
    spa = poke.base_spa
    def_stat = poke.base_def
    spd = poke.base_spd
    hp = poke.base_hp

    # Bulk score
    phys_bulk = hp * def_stat
    spec_bulk = hp * spd
    total_bulk = phys_bulk + spec_bulk

    # Check for heavy recovery/support
    support_count = len(all_moves & (_RECOVERY_MOVES | _SUPPORT_MOVES))

    # If many support moves and low offensive stats -> wall/support
    if support_count >= 2:
        if total_bulk > 50000 or (hp > 80 and (def_stat > 100 or spd > 100)):
            if def_stat > spd:
                return 3  # physical_wall
            else:
                return 4  # special_wall

    if support_count >= 3:
        return 7  # support

    # Check for setup sweepers
    has_setup = bool(all_moves & _SETUP_MOVES)

    # Offensive classification
    if atk > spa + 20:
        if has_setup or atk >= 100:
            return 0  # physical_sweeper
    elif spa > atk + 20:
        if has_setup or spa >= 100:
            return 1  # special_sweeper
    else:
        if (atk >= 80 and spa >= 80) or has_setup:
            return 2  # mixed_attacker

    # Fallback: use bulk vs offense
    offense = max(atk, spa)
    if offense > 90:
        return 0 if atk >= spa else 1
    elif total_bulk > 50000:
        return 3 if def_stat > spd else 4
    else:
        return 7  # support


# ── Tera category classification ─────────────────────────────────────────

NUM_TERA_CATEGORIES = 4
TERA_CATEGORY_NAMES = ["offensive", "defensive", "stab", "coverage"]

# Defensive tera types (resist common attacking types)
_DEFENSIVE_TERA_TYPES = {
    "steel", "fairy", "water", "poison", "ghost", "flying",
}

# Common STAB tera types are same as pokemon type - need type info for this


def classify_tera(
    tera_type: str,
    pokemon_types: str,
) -> int:
    """Classify tera type usage category.

    0 = offensive (boosts attacking type)
    1 = defensive (adds resistances)
    2 = STAB (matches existing type)
    3 = coverage (new offensive type)
    """
    if not tera_type or tera_type.lower() in ("", "unknown", "none"):
        return -1  # Unknown

    tera_lower = tera_type.lower()
    types_lower = pokemon_types.lower() if pokemon_types else ""

    # Check if it matches existing type (STAB tera)
    if tera_lower in types_lower:
        return 2  # STAB

    # Check if defensive type
    if tera_lower in _DEFENSIVE_TERA_TYPES:
        return 1  # defensive

    # Otherwise it's coverage/offensive
    return 3  # coverage


# ── Move family classification ───────────────────────────────────────────

NUM_MOVE_FAMILIES = 10
MOVE_FAMILY_NAMES = [
    "priority",         # 0: Quick Attack, Mach Punch, Sucker Punch, etc.
    "recovery",         # 1: Recover, Roost, etc.
    "hazard_setup",     # 2: Stealth Rock, Spikes, etc.
    "hazard_removal",   # 3: Defog, Rapid Spin
    "status_move",      # 4: Will-O-Wisp, Thunder Wave, Toxic, etc.
    "setup_boost",      # 5: Swords Dance, Nasty Plot, etc.
    "pivot_move",       # 6: U-turn, Volt Switch, etc.
    "screen_move",      # 7: Reflect, Light Screen, etc.
    "phazing_move",     # 8: Whirlwind, Roar, Dragon Tail
    "trick_move",       # 9: Trick, Switcheroo, Knock Off
]

_PRIORITY_MOVES = {
    "extremespeed", "machpunch", "bulletpunch", "aquajet", "iceshard",
    "shadowsneak", "suckerpunch", "quickattack", "accelerock", "fakeout",
    "firstimpression", "grassyglide", "jetpunch",
}

_HAZARD_REMOVAL_MOVES = {"defog", "rapidspin", "courtchange", "tidyup", "mortalspin"}

_SCREEN_MOVES = {"reflect", "lightscreen", "auroraveil"}

_PHAZING_MOVES = {"whirlwind", "roar", "dragontail", "circlethrow", "haze"}

_TRICK_MOVES = {"trick", "switcheroo", "knockoff", "thief", "covet"}


def classify_move_families(moves: list[str]) -> list[int]:
    """Classify which move families are present in a moveset.

    Returns:
        List of 10 binary values (0/1) indicating family presence.
    """
    families = [0] * NUM_MOVE_FAMILIES
    if not moves:
        return families

    normalized = {_normalize_move(m) for m in moves if m}

    if normalized & _PRIORITY_MOVES:
        families[0] = 1
    if normalized & _RECOVERY_MOVES:
        families[1] = 1
    if normalized & _HAZARD_MOVES:
        families[2] = 1
    if normalized & _HAZARD_REMOVAL_MOVES:
        families[3] = 1
    if normalized & _SUPPORT_MOVES:
        families[4] = 1
    if normalized & _SETUP_MOVES:
        families[5] = 1
    if normalized & _PIVOT_MOVES:
        families[6] = 1
    if normalized & _SCREEN_MOVES:
        families[7] = 1
    if normalized & _PHAZING_MOVES:
        families[8] = 1
    if normalized & _TRICK_MOVES:
        families[9] = 1

    return families


# ── End-of-game label extraction ─────────────────────────────────────────


def extract_opponent_labels(
    battle: ParsedBattle,
) -> dict[str, dict[str, Any]]:
    """Extract auxiliary labels for each opponent pokemon from a battle.

    Scans the full battle to find eventually-revealed information about
    opponent pokemon. Only uses information that was visible at some
    point during the battle (items shown via Knock Off, abilities
    activated, moves used, etc.)

    Returns:
        Dict mapping opponent species -> label dict with:
            item_class: int (0-49) or -1
            speed_bucket: int (0-4) or -1
            role: int (0-7) or -1
            tera_category: int (0-3) or -1
            move_families: list[int] of length 10, or all -1 if unknown
    """
    labels: dict[str, dict[str, Any]] = {}
    revealed_moves: dict[str, list[str]] = {}
    revealed_items: dict[str, str] = {}
    revealed_abilities: dict[str, str] = {}
    pokemon_data: dict[str, ParsedPokemon] = {}  # Store last seen data

    for turn in battle.turns:
        opp = turn.opponent_active
        if opp is None:
            continue
        species = opp.name or opp.base_species
        if not species:
            continue

        # Store pokemon data (may contain base stats in Metamon format)
        pokemon_data[species] = opp

        # Track revealed moves
        if species not in revealed_moves:
            revealed_moves[species] = []
        for m in opp.moves:
            if m.name and m.name not in revealed_moves[species]:
                revealed_moves[species].append(m.name)
        if turn.opponent_prev_move and turn.opponent_prev_move.name:
            mn = turn.opponent_prev_move.name
            if mn not in revealed_moves.get(species, []):
                revealed_moves.setdefault(species, []).append(mn)

        # Track revealed items
        if opp.item and opp.item.lower() not in ("", "unknown", "none", "unknownitem"):
            revealed_items[species] = opp.item

        # Track revealed abilities
        if opp.ability and opp.ability.lower() not in ("", "unknown", "none", "unknownability"):
            revealed_abilities[species] = opp.ability

    # Also check team preview pokemon
    for turn in battle.turns:
        for preview_poke in turn.opponent_teampreview:
            species = preview_poke.name or preview_poke.base_species
            if species and species not in pokemon_data:
                pokemon_data[species] = preview_poke

    # Build labels for each opponent pokemon
    for species, poke in pokemon_data.items():
        label: dict[str, Any] = {
            "item_class": -1,
            "speed_bucket": -1,
            "role": -1,
            "tera_category": -1,
            "move_families": [-1] * NUM_MOVE_FAMILIES,
        }

        # Item label: only if revealed during battle
        if species in revealed_items:
            label["item_class"] = classify_item(revealed_items[species])

        # Speed bucket: from base stats (available in Metamon data)
        if poke.base_spe > 0:
            label["speed_bucket"] = classify_speed(poke.base_spe)

        # Role: infer from stats + moves
        moves = revealed_moves.get(species, [])
        # Also include moves from pokemon data
        for m in poke.moves:
            if m.name and m.name not in moves:
                moves.append(m.name)
        if poke.base_atk > 0 or poke.base_spa > 0 or moves:
            label["role"] = classify_role(poke, moves)

        # Tera category: only if we have type info
        if poke.tera_type and poke.tera_type.lower() not in ("", "unknown", "none"):
            label["tera_category"] = classify_tera(poke.tera_type, poke.types)

        # Move families: from all known moves
        if moves:
            label["move_families"] = classify_move_families(moves)

        labels[species] = label

    return labels


def build_auxiliary_targets(
    battle: ParsedBattle,
    max_team_size: int = MAX_TEAM_SIZE,
) -> dict[str, np.ndarray]:
    """Build auxiliary target tensors for a full battle.

    Returns arrays shaped for per-turn training:
        item_targets: (num_turns, 6) int64 - item class per opponent slot
        speed_targets: (num_turns, 6) int64
        role_targets: (num_turns, 6) int64
        tera_targets: (num_turns, 6) int64
        move_family_targets: (num_turns, 6, 10) int64

    Labels are constant across all turns (we use end-of-battle knowledge
    applied retroactively, as specified by the Hidden Information Doctrine
    for label sourcing).
    """
    # Extract labels from the full battle
    opp_labels = extract_opponent_labels(battle)

    num_turns = len(battle.turns)

    item_targets = np.full((num_turns, max_team_size), -1, dtype=np.int64)
    speed_targets = np.full((num_turns, max_team_size), -1, dtype=np.int64)
    role_targets = np.full((num_turns, max_team_size), -1, dtype=np.int64)
    tera_targets = np.full((num_turns, max_team_size), -1, dtype=np.int64)
    move_family_targets = np.full(
        (num_turns, max_team_size, NUM_MOVE_FAMILIES), -1, dtype=np.int64
    )

    for t, turn in enumerate(battle.turns):
        # Build opponent ordering consistent with observation builder
        opp_species_order: list[str] = []

        if turn.opponent_active:
            sp = turn.opponent_active.name or turn.opponent_active.base_species
            if sp:
                opp_species_order.append(sp)

        for preview_poke in turn.opponent_teampreview:
            sp = preview_poke.name or preview_poke.base_species
            if sp and sp not in opp_species_order:
                opp_species_order.append(sp)

        # Fill targets for each slot
        for slot_idx, species in enumerate(opp_species_order[:max_team_size]):
            if species in opp_labels:
                lab = opp_labels[species]
                item_targets[t, slot_idx] = lab["item_class"]
                speed_targets[t, slot_idx] = lab["speed_bucket"]
                role_targets[t, slot_idx] = lab["role"]
                tera_targets[t, slot_idx] = lab["tera_category"]
                families = lab["move_families"]
                move_family_targets[t, slot_idx] = np.array(families, dtype=np.int64)

    return {
        "item_targets": item_targets,
        "speed_targets": speed_targets,
        "role_targets": role_targets,
        "tera_targets": tera_targets,
        "move_family_targets": move_family_targets,
    }


def build_turn_auxiliary_targets(
    battle: ParsedBattle,
    turn_index: int,
    max_team_size: int = MAX_TEAM_SIZE,
) -> dict[str, np.ndarray]:
    """Build auxiliary targets for a single turn.

    Returns:
        Dict with per-slot target arrays (without turn dimension).
    """
    full = build_auxiliary_targets(battle, max_team_size)
    return {
        "item_targets": full["item_targets"][turn_index],
        "speed_targets": full["speed_targets"][turn_index],
        "role_targets": full["role_targets"][turn_index],
        "tera_targets": full["tera_targets"][turn_index],
        "move_family_targets": full["move_family_targets"][turn_index],
    }
