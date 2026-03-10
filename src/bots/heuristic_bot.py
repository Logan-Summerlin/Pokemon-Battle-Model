"""Heuristic bot: scripted decision-making with damage estimation.

Implements the Phase 3 heuristic bot from the implementation plan:
1. If a move guarantees a KO, use it
2. If at a severe type disadvantage, switch to the best available counter
3. Use the highest-expected-damage legal move
4. Break ties randomly

Uses simplified but non-trivial damage estimation including:
- Base power, STAB, type effectiveness
- Physical vs special split (uses correct attacking/defending stats)
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


# ── Type effectiveness chart ───────────────────────────────────────────

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
        "Dark": 2, "Steel": 2, "Fairy": 0.5,
    },
    "Poison": {
        "Grass": 2, "Poison": 0.5, "Ground": 0.5, "Rock": 0.5,
        "Ghost": 0.5, "Steel": 0, "Fairy": 2,
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
        "Steel": 0.5, "Fairy": 0.5,
    },
    "Rock": {
        "Fire": 2, "Ice": 2, "Fighting": 0.5, "Ground": 0.5,
        "Flying": 2, "Bug": 2, "Steel": 0.5,
    },
    "Ghost": {"Normal": 0, "Psychic": 2, "Ghost": 2, "Dark": 0.5},
    "Dragon": {"Dragon": 2, "Steel": 0.5, "Fairy": 0},
    "Dark": {
        "Fighting": 0.5, "Psychic": 2, "Ghost": 2, "Dark": 0.5,
        "Fairy": 0.5,
    },
    "Steel": {
        "Fire": 0.5, "Water": 0.5, "Electric": 0.5, "Ice": 2,
        "Rock": 2, "Steel": 0.5, "Fairy": 2,
    },
    "Fairy": {
        "Fire": 0.5, "Poison": 0.5, "Fighting": 2, "Dragon": 2,
        "Dark": 2, "Steel": 0.5,
    },
}

# Move name -> (type, category, base_power) for common Gen 9 OU moves
_MOVE_DATA: dict[str, tuple[str, str, int]] = {
    "Earthquake": ("Ground", "Physical", 100),
    "Close Combat": ("Fighting", "Physical", 120),
    "Flare Blitz": ("Fire", "Physical", 120),
    "Ice Beam": ("Ice", "Special", 90),
    "Thunderbolt": ("Electric", "Special", 90),
    "Surf": ("Water", "Special", 90),
    "Psychic": ("Psychic", "Special", 90),
    "Moonblast": ("Fairy", "Special", 95),
    "Knock Off": ("Dark", "Physical", 65),
    "U-turn": ("Bug", "Physical", 70),
    "Volt Switch": ("Electric", "Special", 70),
    "Scald": ("Water", "Special", 80),
    "Body Slam": ("Normal", "Physical", 85),
    "Brave Bird": ("Flying", "Physical", 120),
    "Head Smash": ("Rock", "Physical", 150),
    "Stone Edge": ("Rock", "Physical", 100),
    "Iron Head": ("Steel", "Physical", 80),
    "Bullet Punch": ("Steel", "Physical", 40),
    "Mach Punch": ("Fighting", "Physical", 40),
    "Aqua Jet": ("Water", "Physical", 40),
    "Ice Shard": ("Ice", "Physical", 40),
    "Extreme Speed": ("Normal", "Physical", 80),
    "Sucker Punch": ("Dark", "Physical", 70),
    "Shadow Ball": ("Ghost", "Special", 80),
    "Dark Pulse": ("Dark", "Special", 80),
    "Draco Meteor": ("Dragon", "Special", 130),
    "Overheat": ("Fire", "Special", 130),
    "Leaf Storm": ("Grass", "Special", 130),
    "Hydro Pump": ("Water", "Special", 110),
    "Fire Blast": ("Fire", "Special", 110),
    "Thunder": ("Electric", "Special", 110),
    "Blizzard": ("Ice", "Special", 110),
    "Focus Blast": ("Fighting", "Special", 120),
    "Hurricane": ("Flying", "Special", 110),
    "Outrage": ("Dragon", "Physical", 120),
    "Play Rough": ("Fairy", "Physical", 90),
    "Icicle Crash": ("Ice", "Physical", 85),
    "Wild Charge": ("Electric", "Physical", 90),
    "Zen Headbutt": ("Psychic", "Physical", 80),
    "Crunch": ("Dark", "Physical", 80),
    "Poison Jab": ("Poison", "Physical", 80),
    "Waterfall": ("Water", "Physical", 80),
    "Seed Bomb": ("Grass", "Physical", 80),
    "X-Scissor": ("Bug", "Physical", 80),
    "Energy Ball": ("Grass", "Special", 90),
    "Sludge Bomb": ("Poison", "Special", 90),
    "Aura Sphere": ("Fighting", "Special", 80),
    "Flash Cannon": ("Steel", "Special", 80),
    "Flamethrower": ("Fire", "Special", 90),
    "Ice Punch": ("Ice", "Physical", 75),
    "Thunder Punch": ("Electric", "Physical", 75),
    "Fire Punch": ("Fire", "Physical", 75),
    "Dragon Claw": ("Dragon", "Physical", 80),
    "Shadow Claw": ("Ghost", "Physical", 70),
    "Brick Break": ("Fighting", "Physical", 75),
    "Rock Slide": ("Rock", "Physical", 75),
    "Iron Tail": ("Steel", "Physical", 100),
    "Giga Drain": ("Grass", "Special", 75),
    "Dragon Pulse": ("Dragon", "Special", 85),
    "Power Gem": ("Rock", "Special", 80),
    "Air Slash": ("Flying", "Special", 75),
    "Acrobatics": ("Flying", "Physical", 55),
    "Liquidation": ("Water", "Physical", 85),
    "Superpower": ("Fighting", "Physical", 120),
    "Sacred Sword": ("Fighting", "Physical", 90),
    "Gunk Shot": ("Poison", "Physical", 120),
    "Cross Poison": ("Poison", "Physical", 70),
    "Rapid Spin": ("Normal", "Physical", 50),
    "Struggle": ("Normal", "Physical", 50),
    "Hex": ("Ghost", "Special", 65),
    "Poltergeist": ("Ghost", "Physical", 110),
    "Astral Barrage": ("Ghost", "Special", 120),
    "Behemoth Blade": ("Steel", "Physical", 100),
    "Photon Geyser": ("Psychic", "Special", 100),
    "Spectral Thief": ("Ghost", "Physical", 90),
    "Bitter Blade": ("Fire", "Physical", 90),
    "Make It Rain": ("Steel", "Special", 120),
    "Population Bomb": ("Normal", "Physical", 20),
    "Rage Fist": ("Ghost", "Physical", 50),
    "Kowtow Cleave": ("Dark", "Physical", 85),
    "Headlong Rush": ("Ground", "Physical", 120),
    "Collision Course": ("Fighting", "Physical", 100),
    "Electro Drift": ("Electric", "Special", 100),
    "Triple Arrows": ("Fighting", "Physical", 90),
    "Torch Song": ("Fire", "Special", 80),
    "Ivy Cudgel": ("Grass", "Physical", 100),
    "Tachyon Cutter": ("Steel", "Special", 50),
    "Psyblade": ("Psychic", "Physical", 80),
    "Blood Moon": ("Normal", "Special", 140),
    "Matcha Gotcha": ("Grass", "Special", 80),
    "Syrup Bomb": ("Grass", "Special", 60),
}

# Status moves (zero damage, utility-only)
_STATUS_MOVES: set[str] = {
    "Stealth Rock", "Spikes", "Toxic Spikes", "Sticky Web", "Defog",
    "Recover", "Roost", "Wish", "Protect", "Substitute",
    "Swords Dance", "Dragon Dance", "Nasty Plot", "Calm Mind",
    "Will-O-Wisp", "Thunder Wave", "Toxic", "Glare", "Stun Spore",
    "Sleep Powder", "Spore", "Haze", "Whirlwind", "Roar",
    "Trick", "Switcheroo", "Taunt", "Encore", "Disable",
    "Leech Seed", "Pain Split", "Destiny Bond", "Perish Song",
    "Heal Bell", "Aromatherapy", "Court Change", "Parting Shot",
    "Teleport", "Baton Pass", "Flip Turn", "Shed Tail",
    "Slack Off", "Soft-Boiled", "Milk Drink", "Morning Sun",
    "Moonlight", "Synthesis", "Shore Up", "Strength Sap",
    "Curse", "Bulk Up", "Iron Defense", "Cosmic Power",
    "Quiver Dance", "Shell Smash", "Shift Gear", "Coil",
    "Agility", "Autotomize", "Rock Polish", "Tidy Up",
    "Trick Room", "Tailwind", "Rain Dance", "Sunny Day",
    "Sandstorm", "Snowscape", "Reflect", "Light Screen",
    "Aurora Veil", "Rapid Spin",  # Rapid Spin does damage but low
}

# Pivot moves that switch out after use
_PIVOT_MOVES: set[str] = {"U-turn", "Volt Switch", "Flip Turn", "Parting Shot", "Teleport"}


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

    Uses simplified damage formula considering:
    - Base power
    - STAB
    - Type effectiveness
    - Physical/special stat advantage (rough)
    """
    if move_name in _STATUS_MOVES:
        return 0.0

    move_data = _MOVE_DATA.get(move_name)
    if move_data is None:
        # Unknown move: assume moderate damage
        return 60.0

    move_type, category, base_power = move_data

    if base_power == 0:
        return 0.0

    # Type effectiveness
    effectiveness = _type_effectiveness(move_type, defender_types)
    if effectiveness == 0:
        return 0.0

    # STAB (Same Type Attack Bonus)
    stab = 1.5 if move_type in attacker_types else 1.0

    # Use actual stats if available for physical/special split
    if category == "Physical":
        atk_stat = attacker.stats.get("atk", 100)
    else:
        atk_stat = attacker.stats.get("spa", 100)

    # Rough damage score (not exact HP damage, but proportional)
    # Normalize by a baseline stat of 100
    stat_factor = atk_stat / 100.0

    return base_power * effectiveness * stab * stat_factor


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
        # In live battles we see types from the switch-in message
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
        # Estimate KO threshold based on opponent HP fraction
        # If opp is at low HP and we have a strong move, go for it
        if move_scores and opp.hp_fraction > 0:
            best_move_idx, best_score = max(move_scores, key=lambda x: x[1])
            # Rough KO check: if best damage score > some threshold scaled by opp HP
            # A score of ~150+ at full HP is roughly a KO-range hit
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
            # team_pos 0 = team slot index 1 (since 0 is active), etc.
            # We need to find the right bench pokemon
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
                    move_type, _, bp = move_data
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
        The mapping depends on which pokemon is currently active.
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
        """Get type string for our pokemon.

        Uses species name as a fallback type hint since we track types
        in the state data through the request/switch messages.
        """
        # In Showdown state, types aren't stored directly on OwnPokemon.
        # We use the species name to look up types from our known data.
        # This is a simplification - in the real bot, types would come
        # from the Pokedex or the state tracker.
        return _SPECIES_TYPES.get(poke.species, "Normal")

    def _get_opp_types(self, opp: OpponentPokemon) -> list[str]:
        """Get opponent types from species data."""
        type_str = _SPECIES_TYPES.get(opp.species, "Normal")
        return _get_types(type_str)

    def choose_team_order(self, observation: Observation) -> str:
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


# ── Species type database (common Gen 9 OU Pokemon) ─────────────────

_SPECIES_TYPES: dict[str, str] = {
    # Top OU Pokemon with their types
    "Great Tusk": "Ground,Fighting",
    "Iron Valiant": "Fairy,Fighting",
    "Gholdengo": "Steel,Ghost",
    "Kingambit": "Dark,Steel",
    "Dragapult": "Dragon,Ghost",
    "Heatran": "Fire,Steel",
    "Landorus-Therian": "Ground,Flying",
    "Landorus": "Ground,Flying",
    "Garchomp": "Dragon,Ground",
    "Clefable": "Fairy",
    "Toxapex": "Poison,Water",
    "Ferrothorn": "Grass,Steel",
    "Corviknight": "Flying,Steel",
    "Skeledirge": "Fire,Ghost",
    "Garganacl": "Rock",
    "Gliscor": "Ground,Flying",
    "Rotom-Wash": "Electric,Water",
    "Rotom-Heat": "Electric,Fire",
    "Zamazenta": "Fighting",
    "Zamazenta-Crowned": "Fighting,Steel",
    "Ogerpon": "Grass",
    "Ogerpon-Wellspring": "Grass,Water",
    "Ogerpon-Hearthflame": "Grass,Fire",
    "Ogerpon-Cornerstone": "Grass,Rock",
    "Raging Bolt": "Electric,Dragon",
    "Iron Crown": "Steel,Psychic",
    "Iron Boulder": "Rock,Psychic",
    "Iron Moth": "Fire,Poison",
    "Iron Treads": "Ground,Steel",
    "Iron Hands": "Fighting,Electric",
    "Iron Bundle": "Ice,Water",
    "Roaring Moon": "Dragon,Dark",
    "Flutter Mane": "Ghost,Fairy",
    "Sandy Shocks": "Electric,Ground",
    "Scream Tail": "Fairy,Psychic",
    "Brute Bonnet": "Grass,Dark",
    "Walking Wake": "Water,Dragon",
    "Gouging Fire": "Fire,Dragon",
    "Iron Leaves": "Grass,Psychic",
    "Ting-Lu": "Dark,Ground",
    "Chien-Pao": "Dark,Ice",
    "Chi-Yu": "Dark,Fire",
    "Wo-Chien": "Dark,Grass",
    "Dragonite": "Dragon,Flying",
    "Tyranitar": "Rock,Dark",
    "Blissey": "Normal",
    "Chansey": "Normal",
    "Dondozo": "Water",
    "Tatsugiri": "Dragon,Water",
    "Palafin": "Water",
    "Palafin-Hero": "Water",
    "Annihilape": "Fighting,Ghost",
    "Ceruledge": "Fire,Ghost",
    "Armarouge": "Fire,Psychic",
    "Baxcalibur": "Dragon,Ice",
    "Meowscarada": "Grass,Dark",
    "Quaquaval": "Water,Fighting",
    "Cinderace": "Fire",
    "Greninja": "Water,Dark",
    "Slowking-Galar": "Poison,Psychic",
    "Slowbro": "Water,Psychic",
    "Slowking": "Water,Psychic",
    "Pelipper": "Water,Flying",
    "Torkoal": "Fire",
    "Hippowdon": "Ground",
    "Amoonguss": "Grass,Poison",
    "Clodsire": "Poison,Ground",
    "Ditto": "Normal",
    "Volcarona": "Bug,Fire",
    "Weavile": "Dark,Ice",
    "Scizor": "Bug,Steel",
    "Magnezone": "Electric,Steel",
    "Gengar": "Ghost,Poison",
    "Alakazam": "Psychic",
    "Sneasler": "Fighting,Poison",
    "Samurott-Hisui": "Water,Dark",
    "Lilligant-Hisui": "Grass,Fighting",
    "Zoroark-Hisui": "Normal,Ghost",
    "Goodra-Hisui": "Steel,Dragon",
    "Enamorus-Therian": "Fairy,Flying",
    "Enamorus": "Fairy,Flying",
    "Tornadus-Therian": "Flying",
    "Thundurus-Therian": "Electric,Flying",
    "Thundurus": "Electric,Flying",
    "Tapu Koko": "Electric,Fairy",
    "Tapu Lele": "Psychic,Fairy",
    "Tapu Fini": "Water,Fairy",
    "Tapu Bulu": "Grass,Fairy",
    "Urshifu": "Fighting,Dark",
    "Urshifu-Rapid-Strike": "Fighting,Water",
    "Kyurem": "Dragon,Ice",
    "Moltres-Galar": "Dark,Flying",
    "Articuno-Galar": "Psychic,Flying",
    "Zapdos-Galar": "Fighting,Flying",
    "Melmetal": "Steel",
    "Alomomola": "Water",
    "Skarmory": "Steel,Flying",
    "Mandibuzz": "Dark,Flying",
    "Talonflame": "Fire,Flying",
    "Hawlucha": "Fighting,Flying",
    "Ribombee": "Bug,Fairy",
    "Grimmsnarl": "Dark,Fairy",
    "Hydreigon": "Dark,Dragon",
    "Excadrill": "Ground,Steel",
    "Bisharp": "Dark,Steel",
    "Azumarill": "Water,Fairy",
    "Mimikyu": "Ghost,Fairy",
    "Lycanroc-Dusk": "Rock",
    "Ninetales-Alola": "Ice,Fairy",
    "Arcanine": "Fire",
    "Arcanine-Hisui": "Fire,Rock",
    "Primarina": "Water,Fairy",
    "Decidueye-Hisui": "Grass,Fighting",
    "Typhlosion-Hisui": "Fire,Ghost",
    "Basculegion": "Water,Ghost",
    "Overqwil": "Dark,Poison",
    "Kleavor": "Bug,Rock",
    "Ursaluna": "Ground,Normal",
    "Ursaluna-Bloodmoon": "Ground,Normal",
    "Empoleon": "Water,Steel",
    "Infernape": "Fire,Fighting",
    "Lucario": "Fighting,Steel",
    "Blaziken": "Fire,Fighting",
    "Swampert": "Water,Ground",
    "Gyarados": "Water,Flying",
    "Milotic": "Water",
    "Weezing-Galar": "Poison,Fairy",
    "Muk-Alola": "Poison,Dark",
    "Ninetales": "Fire",
    "Mamoswine": "Ice,Ground",
    "Flygon": "Ground,Dragon",
    "Salamence": "Dragon,Flying",
    "Metagross": "Steel,Psychic",
    "Latios": "Dragon,Psychic",
    "Latias": "Dragon,Psychic",
    "Serperior": "Grass",
    "Reuniclus": "Psychic",
    "Conkeldurr": "Fighting",
    "Krookodile": "Ground,Dark",
    "Chandelure": "Ghost,Fire",
    "Haxorus": "Dragon",
    "Mienshao": "Fighting",
    "Breloom": "Grass,Fighting",
    "Rillaboom": "Grass",
    "Toxtricity": "Electric,Poison",
    "Dragalge": "Poison,Dragon",
    "Cloyster": "Water,Ice",
    "Espeon": "Psychic",
    "Umbreon": "Dark",
    "Sylveon": "Fairy",
    "Jolteon": "Electric",
    "Vaporeon": "Water",
    "Flareon": "Fire",
    "Leafeon": "Grass",
    "Glaceon": "Ice",
}
