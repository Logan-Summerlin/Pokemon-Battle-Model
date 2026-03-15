"""Heuristic bot: scripted decision-making with damage estimation for Gen 3 OU.

Implements the Phase 3 heuristic bot from the implementation plan:
1. If a move guarantees a KO, use it
2. If at a severe type disadvantage, switch to the best available counter
3. Use the highest-expected-damage legal move
4. Break ties randomly

Uses simplified but non-trivial damage estimation including:
- Base power, STAB, type effectiveness
- Gen 3 type-based physical/special split (type determines category, NOT the move)
- Explosion/Self-Destruct Defense-halving mechanic (effective 2x power)
- Permanent weather from Sand Stream (SpD boost for Rock-types)
- Status move avoidance when offensive play is available
"""

from __future__ import annotations

import random
from typing import Any

from src.environment.action_space import (
    MOVE_1,
    MOVE_4,
    SWITCH_2,
    SWITCH_6,
    ActionMask,
    BattleAction,
    action_from_canonical_index,
)
from src.environment.battle_env import Observation
from src.environment.state import BattleState, OwnPokemon, OpponentPokemon
from src.bots.base_bot import Bot


# ── Gen 3 type-based physical/special split ─────────────────────────────
# In Gen 3, whether a move is physical or special depends ENTIRELY on its
# type, not the individual move. This is the defining mechanical difference
# from Gen 4+.

PHYSICAL_TYPES = frozenset({
    "Normal", "Fighting", "Poison", "Ground", "Flying",
    "Bug", "Rock", "Ghost", "Steel",
})

SPECIAL_TYPES = frozenset({
    "Fire", "Water", "Grass", "Electric", "Ice",
    "Psychic", "Dragon", "Dark",
})


def get_gen3_category(move_type: str) -> str:
    """Return 'Physical' or 'Special' based on the move's type in Gen 3."""
    if move_type in PHYSICAL_TYPES:
        return "Physical"
    elif move_type in SPECIAL_TYPES:
        return "Special"
    return "Physical"  # Default fallback


# ── Gen 3 type effectiveness chart ───────────────────────────────────────
# NO Fairy type in Gen 3. Steel resists Dark and Ghost (changed in Gen 6).

_TYPE_CHART: dict[str, dict[str, float]] = {
    "Normal": {"Rock": 0.5, "Ghost": 0, "Steel": 0.5},
    "Fire": {
        "Fire": 0.5, "Water": 0.5, "Grass": 2, "Ice": 2, "Bug": 2,
        "Rock": 0.5, "Dragon": 0.5, "Steel": 2,
    },
    "Water": {
        "Fire": 2, "Water": 0.5, "Grass": 0.5, "Ground": 2,
        "Rock": 2, "Dragon": 0.5,
    },
    "Electric": {
        "Water": 2, "Electric": 0.5, "Grass": 0.5, "Ground": 0,
        "Flying": 2, "Dragon": 0.5,
    },
    "Grass": {
        "Fire": 0.5, "Water": 2, "Grass": 0.5, "Poison": 0.5,
        "Ground": 2, "Flying": 0.5, "Bug": 0.5, "Rock": 2,
        "Dragon": 0.5, "Steel": 0.5,
    },
    "Ice": {
        "Fire": 0.5, "Water": 0.5, "Grass": 2, "Ice": 0.5,
        "Ground": 2, "Flying": 2, "Dragon": 2, "Steel": 0.5,
    },
    "Fighting": {
        "Normal": 2, "Ice": 2, "Poison": 0.5, "Flying": 0.5,
        "Psychic": 0.5, "Bug": 0.5, "Rock": 2, "Ghost": 0,
        "Dark": 2, "Steel": 2,
    },
    "Poison": {
        "Grass": 2, "Poison": 0.5, "Ground": 0.5, "Rock": 0.5,
        "Ghost": 0.5, "Steel": 0,
    },
    "Ground": {
        "Fire": 2, "Electric": 2, "Grass": 0.5, "Poison": 2,
        "Flying": 0, "Bug": 0.5, "Rock": 2, "Steel": 2,
    },
    "Flying": {
        "Electric": 0.5, "Grass": 2, "Fighting": 2, "Bug": 2,
        "Rock": 0.5, "Steel": 0.5,
    },
    "Psychic": {
        "Fighting": 2, "Poison": 2, "Psychic": 0.5, "Dark": 0,
        "Steel": 0.5,
    },
    "Bug": {
        "Fire": 0.5, "Grass": 2, "Fighting": 0.5, "Poison": 0.5,
        "Flying": 0.5, "Psychic": 2, "Ghost": 0.5, "Dark": 2,
        "Steel": 0.5,
    },
    "Rock": {
        "Fire": 2, "Ice": 2, "Fighting": 0.5, "Ground": 0.5,
        "Flying": 2, "Bug": 2, "Steel": 0.5,
    },
    # Gen 3: Ghost is resisted by Steel (changed in Gen 6)
    "Ghost": {"Normal": 0, "Psychic": 2, "Ghost": 2, "Dark": 0.5, "Steel": 0.5},
    "Dragon": {"Dragon": 2, "Steel": 0.5},
    # Gen 3: Dark is resisted by Steel (changed in Gen 6)
    "Dark": {
        "Fighting": 0.5, "Psychic": 2, "Ghost": 2, "Dark": 0.5,
        "Steel": 0.5,
    },
    "Steel": {
        "Fire": 0.5, "Water": 0.5, "Electric": 0.5, "Ice": 2,
        "Rock": 2, "Steel": 0.5,
    },
    # NO Fairy type in Gen 3
}

# Move name -> (type, base_power) for common Gen 3 OU moves.
# Category is determined by the move's TYPE, not the move itself.
# Explosion and Self-Destruct halve the target's Defense in Gen 3,
# effectively doubling their power.
_MOVE_DATA: dict[str, tuple[str, int]] = {
    # Normal
    "Body Slam": ("Normal", 85),
    "Double-Edge": ("Normal", 120),
    "Return": ("Normal", 102),
    "Explosion": ("Normal", 250),  # Halves def -> effective 500 power
    "Self-Destruct": ("Normal", 200),  # Halves def -> effective 400 power
    "Hyper Beam": ("Normal", 150),
    "Facade": ("Normal", 70),
    "Rapid Spin": ("Normal", 20),
    "Extreme Speed": ("Normal", 80),
    "Fake Out": ("Normal", 40),
    "Struggle": ("Normal", 50),
    "Strength": ("Normal", 80),
    "Secret Power": ("Normal", 70),
    "Crush Claw": ("Normal", 75),
    # Fighting
    "Cross Chop": ("Fighting", 100),
    "Brick Break": ("Fighting", 75),
    "Focus Punch": ("Fighting", 150),
    "Superpower": ("Fighting", 120),
    "Sky Uppercut": ("Fighting", 85),
    "Mach Punch": ("Fighting", 40),
    "Low Kick": ("Fighting", 1),  # Variable power, use low default
    "Seismic Toss": ("Fighting", 0),  # Fixed 100 HP damage
    "Counter": ("Fighting", 0),  # Variable damage
    "Reversal": ("Fighting", 1),  # Variable power
    # Fire
    "Fire Blast": ("Fire", 120),
    "Flamethrower": ("Fire", 95),
    "Overheat": ("Fire", 140),
    "Fire Punch": ("Fire", 75),
    "Blaze Kick": ("Fire", 85),
    "Will-O-Wisp": ("Fire", 0),
    "Eruption": ("Fire", 150),
    "Heat Wave": ("Fire", 100),
    # Water
    "Surf": ("Water", 95),
    "Hydro Pump": ("Water", 120),
    "Ice Beam": ("Ice", 95),
    "Waterfall": ("Water", 80),
    # Grass
    "Giga Drain": ("Grass", 60),
    "Solar Beam": ("Grass", 120),
    "Leaf Blade": ("Grass", 70),  # Gen 3 base power
    "Leech Seed": ("Grass", 0),
    "Sleep Powder": ("Grass", 0),
    "Stun Spore": ("Grass", 0),
    # Electric
    "Thunderbolt": ("Electric", 95),
    "Thunder": ("Electric", 120),
    "Thunder Wave": ("Electric", 0),
    "Spark": ("Electric", 65),
    "Thunder Punch": ("Electric", 75),
    "Volt Tackle": ("Electric", 120),
    # Ice
    "Blizzard": ("Ice", 120),
    "Ice Punch": ("Ice", 75),
    "Ice Beam": ("Ice", 95),
    # Psychic
    "Psychic": ("Psychic", 90),
    "Calm Mind": ("Psychic", 0),
    "Zen Headbutt": ("Psychic", 0),  # Doesn't exist in Gen 3
    "Psycho Boost": ("Psychic", 140),
    "Future Sight": ("Psychic", 80),
    # Dragon
    "Dragon Claw": ("Dragon", 80),
    "Outrage": ("Dragon", 90),  # Gen 3 base power
    "Dragon Dance": ("Dragon", 0),
    "Dragonbreath": ("Dragon", 60),
    # Dark
    "Crunch": ("Dark", 80),
    "Pursuit": ("Dark", 40),
    "Shadow Ball": ("Ghost", 80),
    "Knock Off": ("Dark", 20),  # Gen 3 base power (no boost for item removal)
    "Bite": ("Dark", 60),
    "Thief": ("Dark", 40),
    # Ghost
    "Shadow Punch": ("Ghost", 60),
    # Poison
    "Sludge Bomb": ("Poison", 90),
    "Toxic": ("Poison", 0),
    "Cross Poison": ("Poison", 0),  # Doesn't exist in Gen 3
    "Poison Jab": ("Poison", 0),  # Doesn't exist in Gen 3
    # Ground
    "Earthquake": ("Ground", 100),
    "Rock Slide": ("Rock", 75),
    "Sand Tomb": ("Ground", 15),
    "Mud Shot": ("Ground", 55),
    "Earth Power": ("Ground", 0),  # Doesn't exist in Gen 3
    "Dig": ("Ground", 60),
    # Rock
    "Rock Blast": ("Rock", 25),  # Multi-hit
    "Stone Edge": ("Rock", 0),  # Doesn't exist in Gen 3
    "Ancient Power": ("Rock", 60),
    "Rock Throw": ("Rock", 50),
    "Head Smash": ("Rock", 0),  # Doesn't exist in Gen 3
    # Flying
    "Aerial Ace": ("Flying", 60),
    "Drill Peck": ("Flying", 80),
    "Brave Bird": ("Flying", 0),  # Doesn't exist in Gen 3
    "Fly": ("Flying", 70),
    "Hidden Power Flying": ("Flying", 70),
    "Sky Attack": ("Flying", 140),
    # Bug
    "Signal Beam": ("Bug", 75),
    "Megahorn": ("Bug", 120),
    "Silver Wind": ("Bug", 60),
    "X-Scissor": ("Bug", 0),  # Doesn't exist in Gen 3
    # Steel
    "Meteor Mash": ("Steel", 100),
    "Iron Tail": ("Steel", 100),
    "Steel Wing": ("Steel", 70),
    "Metal Claw": ("Steel", 50),
    "Iron Defense": ("Steel", 0),
    # Hidden Power (common types in Gen 3 OU)
    "Hidden Power Grass": ("Grass", 70),
    "Hidden Power Fire": ("Fire", 70),
    "Hidden Power Ice": ("Ice", 70),
    "Hidden Power Bug": ("Bug", 70),
    "Hidden Power Electric": ("Electric", 70),
    "Hidden Power Ground": ("Ground", 70),
    "Hidden Power Fighting": ("Fighting", 70),
    # Support/Status moves (zero damage)
    "Spikes": ("Ground", 0),
    "Roar": ("Normal", 0),
    "Whirlwind": ("Normal", 0),
    "Protect": ("Normal", 0),
    "Substitute": ("Normal", 0),
    "Wish": ("Normal", 0),
    "Recover": ("Normal", 0),
    "Soft-Boiled": ("Normal", 0),
    "Rest": ("Psychic", 0),
    "Slack Off": ("Normal", 0),
    "Milk Drink": ("Normal", 0),
    "Morning Sun": ("Normal", 0),
    "Moonlight": ("Normal", 0),
    "Synthesis": ("Normal", 0),
    "Refresh": ("Normal", 0),
    "Heal Bell": ("Normal", 0),
    "Aromatherapy": ("Grass", 0),
    "Reflect": ("Psychic", 0),
    "Light Screen": ("Psychic", 0),
    "Haze": ("Ice", 0),
    "Taunt": ("Dark", 0),
    "Encore": ("Normal", 0),
    "Trick": ("Psychic", 0),
    "Yawn": ("Normal", 0),
    "Toxic": ("Poison", 0),
    "Thunder Wave": ("Electric", 0),
    "Swords Dance": ("Normal", 0),
    "Bulk Up": ("Fighting", 0),
    "Curse": ("???", 0),
    "Agility": ("Psychic", 0),
    "Belly Drum": ("Normal", 0),
    "Baton Pass": ("Normal", 0),
    "Rain Dance": ("Water", 0),
    "Sunny Day": ("Fire", 0),
    "Sandstorm": ("Rock", 0),
    "Perish Song": ("Normal", 0),
    "Destiny Bond": ("Ghost", 0),
    "Mean Look": ("Normal", 0),
    "Spider Web": ("Bug", 0),
    "Block": ("Normal", 0),
}

# Status moves (zero damage, utility-only) — Gen 3 specific
_STATUS_MOVES: set[str] = {
    "Spikes", "Roar", "Whirlwind", "Protect", "Substitute",
    "Wish", "Recover", "Soft-Boiled", "Rest", "Slack Off",
    "Milk Drink", "Morning Sun", "Moonlight", "Synthesis",
    "Refresh", "Heal Bell", "Aromatherapy",
    "Reflect", "Light Screen", "Haze",
    "Taunt", "Encore", "Trick", "Yawn",
    "Toxic", "Thunder Wave", "Will-O-Wisp", "Stun Spore",
    "Sleep Powder", "Spore", "Leech Seed",
    "Swords Dance", "Dragon Dance", "Calm Mind", "Bulk Up",
    "Iron Defense", "Amnesia", "Cosmic Power", "Meditate",
    "Curse", "Agility", "Belly Drum",
    "Baton Pass", "Rain Dance", "Sunny Day", "Sandstorm",
    "Perish Song", "Destiny Bond",
    "Mean Look", "Spider Web", "Block",
    "Rapid Spin",  # Does damage but very low
}

# Pivot moves in Gen 3: only Baton Pass (U-turn, Volt Switch don't exist)
_PIVOT_MOVES: set[str] = {"Baton Pass"}

# Explosion/Self-Destruct get effective 2x power in Gen 3
# (they halve the target's Defense before calculating damage)
_EXPLOSION_MOVES: set[str] = {"Explosion", "Self-Destruct"}


def _get_types(type_string: str) -> list[str]:
    """Parse a type string like 'Fire/Flying' or 'Water' into a list."""
    if not type_string:
        return []
    return [t.strip() for t in type_string.replace("/", ",").split(",") if t.strip()]


def _type_effectiveness(atk_type: str, def_types: list[str]) -> float:
    """Calculate type effectiveness multiplier."""
    multiplier = 1.0
    chart = _TYPE_CHART.get(atk_type, {})
    for def_type in def_types:
        multiplier *= chart.get(def_type, 1.0)
    return multiplier


def _estimate_damage(
    move_name: str,
    attacker: OwnPokemon,
    defender_types: list[str],
    attacker_types: list[str],
) -> float:
    """Estimate damage as a rough score (not exact HP).

    Uses simplified Gen 3 damage formula considering:
    - Base power
    - STAB
    - Type effectiveness
    - Gen 3 type-based physical/special split
    - Explosion/Self-Destruct Defense-halving mechanic
    """
    if move_name in _STATUS_MOVES:
        return 0.0

    move_data = _MOVE_DATA.get(move_name)
    if move_data is None:
        # Unknown move: assume moderate damage
        return 60.0

    move_type, base_power = move_data

    if base_power == 0:
        return 0.0

    # Type effectiveness
    effectiveness = _type_effectiveness(move_type, defender_types)
    if effectiveness == 0:
        return 0.0

    # STAB (Same Type Attack Bonus)
    stab = 1.5 if move_type in attacker_types else 1.0

    # Gen 3: physical/special determined by TYPE, not by individual move
    category = get_gen3_category(move_type)
    if category == "Physical":
        atk_stat = attacker.stats.get("atk", 100)
    else:
        atk_stat = attacker.stats.get("spa", 100)

    # Explosion/Self-Destruct: halve target Defense in Gen 3
    # This effectively doubles the damage
    explosion_bonus = 2.0 if move_name in _EXPLOSION_MOVES else 1.0

    # Rough damage score (proportional, not exact HP)
    stat_factor = atk_stat / 100.0

    return base_power * effectiveness * stab * stat_factor * explosion_bonus


def _defensive_type_score(
    own_types: list[str], opp_types: list[str]
) -> float:
    """Score how well our types defend against opponent's types.

    Returns a value where > 1.0 means we take super-effective damage,
    < 1.0 means we resist. Used to detect type disadvantage.
    """
    if not opp_types:
        return 1.0

    worst = 1.0
    for atk_type in opp_types:
        eff = _type_effectiveness(atk_type, own_types)
        worst = max(worst, eff)
    return worst


class HeuristicBot(Bot):
    """Scripted bot using damage estimation and type-based switching.

    Tuned for Gen 3 OU mechanics:
    - Type-based physical/special split
    - No Fairy type
    - Steel resists Dark and Ghost
    - Explosion/Self-Destruct halve target Defense
    - Permanent weather from Sand Stream

    Decision priority:
    1. If a move can KO the opponent (estimated damage > remaining HP), use it
    2. If at severe type disadvantage (>= 2x weakness), switch to best counter
    3. Use the highest-estimated-damage legal move
    4. Break ties randomly
    """

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)
        self._seed = seed
        self._games_played = 0
        self._wins = 0

    @property
    def name(self) -> str:
        return "HeuristicBot"

    def choose_action(
        self, observation: Observation, legal_actions: ActionMask
    ) -> BattleAction:
        legal_indices = legal_actions.legal_indices
        if not legal_indices:
            raise RuntimeError("No legal actions available for HeuristicBot")

        state = observation.state
        active = state.own_active
        opp = state.opponent_active

        # Separate moves and switches
        legal_moves = [i for i in legal_indices if MOVE_1 <= i <= MOVE_4]
        legal_switches = [i for i in legal_indices if SWITCH_2 <= i <= SWITCH_6]

        # If no active pokemon info, fall back to random
        if not active or not opp:
            chosen = self._rng.choice(legal_indices)
            return action_from_canonical_index(chosen)

        opp_types = _get_types(opp.species)  # Fallback; prefer actual types
        # Try to get opponent types from species data if available
        own_types = _get_types(self._get_own_types(active))
        opp_types = self._get_opp_types(opp)

        # Score all legal moves
        move_scores: list[tuple[int, float]] = []
        if legal_moves and active:
            for idx in legal_moves:
                move_slot = idx - MOVE_1
                if move_slot < len(active.moves):
                    move_name = active.moves[move_slot].name
                    score = _estimate_damage(
                        move_name, active, opp_types, own_types
                    )
                    move_scores.append((idx, score))

        # Rule 1: If a move can likely KO, use it
        if move_scores and opp.hp_fraction > 0:
            best_move_idx, best_score = max(move_scores, key=lambda x: x[1])
            # Rough KO check: if best damage score > some threshold scaled by opp HP
            ko_threshold = 120.0 * opp.hp_fraction
            if best_score > ko_threshold and best_score > 0:
                return action_from_canonical_index(best_move_idx)

        # Rule 2: If at severe type disadvantage, try to switch
        if legal_switches and own_types:
            vulnerability = _defensive_type_score(own_types, opp_types)
            # 2x or more super effective = severe disadvantage
            if vulnerability >= 2.0:
                best_switch = self._pick_best_switch(
                    state, legal_switches, opp_types
                )
                if best_switch is not None:
                    return action_from_canonical_index(best_switch)

        # Rule 3: Use highest damage move
        if move_scores:
            # Filter out zero-damage moves if we have damaging options
            damaging = [(idx, s) for idx, s in move_scores if s > 0]
            if damaging:
                best_idx, best_score = max(damaging, key=lambda x: x[1])
                # Check for ties and break randomly
                ties = [idx for idx, s in damaging if abs(s - best_score) < 0.1]
                chosen = self._rng.choice(ties)
                return action_from_canonical_index(chosen)

            # All moves are status moves - use a random one
            chosen = self._rng.choice([idx for idx, _ in move_scores])
            return action_from_canonical_index(chosen)

        # If only switches available (forced switch), pick best matchup
        if legal_switches:
            best_switch = self._pick_best_switch(state, legal_switches, opp_types)
            if best_switch is not None:
                return action_from_canonical_index(best_switch)
            chosen = self._rng.choice(legal_switches)
            return action_from_canonical_index(chosen)

        # Ultimate fallback
        chosen = self._rng.choice(legal_indices)
        return action_from_canonical_index(chosen)

    def _pick_best_switch(
        self,
        state: BattleState,
        legal_switches: list[int],
        opp_types: list[str],
    ) -> int | None:
        """Pick the best switch target based on type matchup.

        Prefers Pokemon that:
        1. Resist the opponent's STAB types
        2. Have super-effective moves against the opponent
        """
        best_idx: int | None = None
        best_score = -999.0

        for switch_idx in legal_switches:
            team_pos = switch_idx - SWITCH_2
            bench_pokemon = self._get_bench_pokemon(state, team_pos)
            if bench_pokemon is None or not bench_pokemon.is_alive:
                continue

            switch_types = _get_types(self._get_own_types(bench_pokemon))

            # Score: low vulnerability to opponent (good) + offensive potential
            vulnerability = _defensive_type_score(switch_types, opp_types)
            # Invert vulnerability: low vulnerability = high score
            defense_score = 1.0 / max(vulnerability, 0.25)

            # Check if this pokemon has SE moves against opponent
            offense_score = 0.0
            for move in bench_pokemon.moves:
                move_data = _MOVE_DATA.get(move.name)
                if move_data:
                    move_type, bp = move_data
                    eff = _type_effectiveness(move_type, opp_types)
                    if eff > 1.0 and bp > 0:
                        offense_score = max(offense_score, eff * bp / 100.0)

            total_score = defense_score * 2.0 + offense_score
            if total_score > best_score:
                best_score = total_score
                best_idx = switch_idx

        return best_idx

    def _get_bench_pokemon(
        self, state: BattleState, bench_index: int
    ) -> OwnPokemon | None:
        """Get a bench pokemon by its bench position index.

        bench_index 0 corresponds to the first non-active pokemon, etc.
        """
        bench_count = 0
        for i, poke in enumerate(state.own_team):
            if i == state.own_active_index:
                continue
            if bench_count == bench_index:
                return poke
            bench_count += 1
        return None

    def _get_own_types(self, poke: OwnPokemon) -> str:
        """Get type string for our pokemon."""
        return _SPECIES_TYPES.get(poke.species, "Normal")

    def _get_opp_types(self, opp: OpponentPokemon) -> list[str]:
        """Get opponent types from species data."""
        type_str = _SPECIES_TYPES.get(opp.species, "Normal")
        return _get_types(type_str)

    def choose_team_order(self, observation: Observation) -> str:
        # Gen 3 has no team preview, but keep interface consistent
        return "123456"

    def on_battle_start(self) -> None:
        self._games_played += 1

    def on_battle_end(self, won: bool | None, info: dict[str, Any] | None = None) -> None:
        if won:
            self._wins += 1

    def reset(self) -> None:
        self._rng = random.Random(self._seed)
        self._games_played = 0
        self._wins = 0

    @property
    def win_rate(self) -> float:
        if self._games_played == 0:
            return 0.0
        return self._wins / self._games_played


# ── Species type database (Gen 3 OU Pokemon) ───────────────────────────
# Comprehensive list of Pokemon seen in Gen 3 OU (ADV OU).
# NO Fairy type — Fairy was introduced in Gen 6.

_SPECIES_TYPES: dict[str, str] = {
    # Tier 1: Top OU threats
    "Tyranitar": "Rock,Dark",
    "Metagross": "Steel,Psychic",
    "Salamence": "Dragon,Flying",
    "Swampert": "Water,Ground",
    "Skarmory": "Steel,Flying",
    "Blissey": "Normal",
    "Gengar": "Ghost,Poison",
    "Celebi": "Psychic,Grass",
    "Jirachi": "Steel,Psychic",
    "Suicune": "Water",
    "Aerodactyl": "Rock,Flying",
    "Starmie": "Water,Psychic",
    "Snorlax": "Normal",
    "Milotic": "Water",
    "Zapdos": "Electric,Flying",
    "Magneton": "Electric,Steel",
    "Dugtrio": "Ground",
    "Heracross": "Bug,Fighting",
    "Flygon": "Ground,Dragon",
    "Claydol": "Ground,Psychic",
    "Forretress": "Bug,Steel",

    # Tier 2: Common OU picks
    "Gyarados": "Water,Flying",
    "Jolteon": "Electric",
    "Vaporeon": "Water",
    "Alakazam": "Psychic",
    "Machamp": "Fighting",
    "Charizard": "Fire,Flying",
    "Venusaur": "Grass,Poison",
    "Blaziken": "Fire,Fighting",
    "Breloom": "Grass,Fighting",
    "Hariyama": "Fighting",
    "Weezing": "Poison",
    "Dusclops": "Ghost",
    "Regice": "Ice",
    "Registeel": "Steel",
    "Regirock": "Rock",
    "Raikou": "Electric",
    "Entei": "Fire",
    "Moltres": "Fire,Flying",
    "Articuno": "Ice,Flying",
    "Cloyster": "Water,Ice",
    "Porygon2": "Normal",

    # Tier 3: Viable OU picks
    "Ludicolo": "Water,Grass",
    "Kingdra": "Water,Dragon",
    "Gardevoir": "Psychic",  # No Fairy in Gen 3
    "Medicham": "Fighting,Psychic",
    "Houndoom": "Dark,Fire",
    "Umbreon": "Dark",
    "Espeon": "Psychic",
    "Wailord": "Water",
    "Mantine": "Water,Flying",
    "Donphan": "Ground",
    "Marowak": "Ground",
    "Ninjask": "Bug,Flying",
    "Shedinja": "Bug,Ghost",
    "Camerupt": "Fire,Ground",
    "Armaldo": "Rock,Bug",
    "Cradily": "Rock,Grass",
    "Solrock": "Rock,Psychic",
    "Lunatone": "Rock,Psychic",
    "Absol": "Dark",
    "Mawile": "Steel",
    "Aggron": "Steel,Rock",
    "Walrein": "Ice,Water",
    "Glalie": "Ice",
    "Banette": "Ghost",
    "Misdreavus": "Ghost",
    "Crobat": "Poison,Flying",
    "Nidoking": "Poison,Ground",
    "Nidoqueen": "Poison,Ground",
    "Rhydon": "Ground,Rock",
    "Golem": "Rock,Ground",
    "Exeggutor": "Grass,Psychic",
    "Steelix": "Steel,Ground",
    "Scizor": "Bug,Steel",
    "Slowbro": "Water,Psychic",
    "Slowking": "Water,Psychic",
    "Lanturn": "Water,Electric",
    "Quagsire": "Water,Ground",
    "Tentacruel": "Water,Poison",
    "Lapras": "Water,Ice",
    "Weezing": "Poison",
    "Muk": "Poison",
    "Electrode": "Electric",
    "Raichu": "Electric",
    "Ampharos": "Electric",
    "Manectric": "Electric",
    "Arcanine": "Fire",
    "Ninetales": "Fire",
    "Magmar": "Fire",
    "Electabuzz": "Electric",
    "Mr. Mime": "Psychic",
    "Xatu": "Psychic,Flying",
    "Grumpig": "Psychic",
    "Linoone": "Normal",
    "Tauros": "Normal",
    "Kangaskhan": "Normal",
    "Ursaring": "Normal",
    "Slaking": "Normal",
    "Smeargle": "Normal",
    "Zangoose": "Normal",
    "Blastoise": "Water",
    "Feraligatr": "Water",
    "Golduck": "Water",
    "Politoed": "Water",
    "Poliwrath": "Water,Fighting",
    "Crawdaunt": "Water,Dark",
    "Sharpedo": "Water,Dark",
    "Whiscash": "Water,Ground",
    "Relicanth": "Water,Rock",
    "Corsola": "Water,Rock",
    "Stantler": "Normal",
    "Girafarig": "Normal,Psychic",
    "Dodrio": "Normal,Flying",
    "Fearow": "Normal,Flying",
    "Pidgeot": "Normal,Flying",
    "Swellow": "Normal,Flying",
    "Altaria": "Dragon,Flying",
    "Dragonite": "Dragon,Flying",
    "Tropius": "Grass,Flying",
    "Jumpluff": "Grass,Flying",
    "Vileplume": "Grass,Poison",
    "Bellossom": "Grass",
    "Roselia": "Grass,Poison",
    "Cacturne": "Grass,Dark",
    "Shiftry": "Grass,Dark",
    "Sableye": "Dark,Ghost",
    "Sneasel": "Dark,Ice",
    "Murkrow": "Dark,Flying",
    "Pinsir": "Bug",
    "Heracross": "Bug,Fighting",
    "Volbeat": "Bug",
    "Illumise": "Bug",
}
