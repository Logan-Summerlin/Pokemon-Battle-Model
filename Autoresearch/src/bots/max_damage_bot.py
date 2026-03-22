"""Max-damage heuristic bot: always picks the highest-damage move.

A simple scripted bot that serves as a second baseline and stress-tests
the action/legality system. Falls back to random legal action for switches.
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
from src.bots.base_bot import Bot


# Rough base power lookup for common Gen 3 OU moves.
# Moves not listed default to 80 (approximate average).
# This is intentionally simple — a real heuristic bot (Phase 3) will use
# full damage calculation.
_COMMON_MOVE_POWER: dict[str, int] = {
    # Physical moves (category determined by type in Gen 3)
    "Earthquake": 100,
    "Rock Slide": 75,
    "Meteor Mash": 100,
    "Shadow Ball": 80,
    "Sludge Bomb": 90,
    "Megahorn": 120,
    "Return": 102,
    "Body Slam": 85,
    "Double-Edge": 120,
    "Hyper Beam": 150,
    "Explosion": 250,
    "Self-Destruct": 200,
    "Cross Chop": 100,
    "Brick Break": 75,
    "Focus Punch": 150,
    "Aerial Ace": 60,
    "Hidden Power": 70,
    "Facade": 70,
    "Quick Attack": 40,
    "Extreme Speed": 80,
    "Mach Punch": 40,
    "Fake Out": 40,
    "Rapid Spin": 20,
    # Special moves (category determined by type in Gen 3)
    "Surf": 95,
    "Hydro Pump": 120,
    "Ice Beam": 95,
    "Blizzard": 120,
    "Thunderbolt": 95,
    "Thunder": 120,
    "Flamethrower": 95,
    "Fire Blast": 120,
    "Overheat": 140,
    "Psychic": 90,
    "Calm Mind": 0,
    "Crunch": 80,
    "Pursuit": 40,
    "Dragon Claw": 80,
    "Outrage": 90,
    "Giga Drain": 60,
    "Solar Beam": 120,
    "Thunderpunch": 75,
    "Ice Punch": 75,
    "Fire Punch": 75,
    # Status moves
    "Spikes": 0,
    "Toxic": 0,
    "Will-O-Wisp": 0,
    "Thunder Wave": 0,
    "Swords Dance": 0,
    "Dragon Dance": 0,
    "Calm Mind": 0,
    "Bulk Up": 0,
    "Agility": 0,
    "Curse": 0,
    "Belly Drum": 0,
    "Baton Pass": 0,
    "Wish": 0,
    "Protect": 0,
    "Substitute": 0,
    "Recover": 0,
    "Rest": 0,
    "Sleep Talk": 0,
    "Roar": 0,
    "Whirlwind": 0,
    "Haze": 0,
    "Reflect": 0,
    "Light Screen": 0,
    "Aromatherapy": 0,
    "Heal Bell": 0,
    "Sing": 0,
    "Lovely Kiss": 0,
    "Hypnosis": 0,
    "Spore": 0,
    "Leech Seed": 0,
    "Encore": 0,
    "Taunt": 0,
    "Struggle": 50,
}


class MaxDamageBot(Bot):
    """Bot that picks the highest-estimated-damage legal move.

    Uses a simple power lookup (no type effectiveness or stats).
    For forced switches, randomly picks from legal switch targets.
    """

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)
        self._seed = seed
        self._games_played = 0
        self._wins = 0

    @property
    def name(self) -> str:
        return "MaxDamageBot"

    def choose_action(
        self, observation: Observation, legal_actions: ActionMask
    ) -> BattleAction:
        legal_indices = legal_actions.legal_indices
        if not legal_indices:
            raise RuntimeError("No legal actions available for MaxDamageBot")

        # Separate moves and switches
        legal_moves = [i for i in legal_indices if MOVE_1 <= i <= MOVE_4]
        legal_switches = [i for i in legal_indices if SWITCH_2 <= i <= SWITCH_6]

        # If we have legal moves, pick the highest-power one
        if legal_moves:
            best_idx = self._pick_best_move(observation, legal_moves)
            return action_from_canonical_index(best_idx)

        # If only switches available (forced switch), pick randomly
        if legal_switches:
            chosen = self._rng.choice(legal_switches)
            return action_from_canonical_index(chosen)

        # Fallback: any legal action
        chosen = self._rng.choice(legal_indices)
        return action_from_canonical_index(chosen)

    def _pick_best_move(
        self, observation: Observation, legal_move_indices: list[int]
    ) -> int:
        """Pick the legal move with the highest estimated base power."""
        best_power = -1
        best_idx = legal_move_indices[0]

        active = observation.state.own_active
        if not active:
            return best_idx

        for idx in legal_move_indices:
            move_slot = idx - MOVE_1
            if move_slot < len(active.moves):
                move_name = active.moves[move_slot].name
                power = _COMMON_MOVE_POWER.get(move_name, 80)
                if power > best_power:
                    best_power = power
                    best_idx = idx

        return best_idx

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
