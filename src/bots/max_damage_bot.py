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


# Rough base power lookup for common Gen 9 OU moves.
# Moves not listed default to 80 (approximate average).
# This is intentionally simple — a real heuristic bot (Phase 3) will use
# full damage calculation.
_COMMON_MOVE_POWER: dict[str, int] = {
    "Earthquake": 100,
    "Close Combat": 120,
    "Flare Blitz": 120,
    "Ice Beam": 90,
    "Thunderbolt": 90,
    "Surf": 90,
    "Psychic": 90,
    "Moonblast": 95,
    "Knock Off": 65,
    "U-turn": 70,
    "Volt Switch": 70,
    "Stealth Rock": 0,
    "Spikes": 0,
    "Toxic Spikes": 0,
    "Sticky Web": 0,
    "Rapid Spin": 50,
    "Defog": 0,
    "Recover": 0,
    "Roost": 0,
    "Wish": 0,
    "Protect": 0,
    "Substitute": 0,
    "Swords Dance": 0,
    "Dragon Dance": 0,
    "Nasty Plot": 0,
    "Calm Mind": 0,
    "Will-O-Wisp": 0,
    "Thunder Wave": 0,
    "Toxic": 0,
    "Scald": 80,
    "Body Slam": 85,
    "Brave Bird": 120,
    "Head Smash": 150,
    "Stone Edge": 100,
    "Iron Head": 80,
    "Bullet Punch": 40,
    "Mach Punch": 40,
    "Aqua Jet": 40,
    "Ice Shard": 40,
    "Extreme Speed": 80,
    "Sucker Punch": 70,
    "Shadow Ball": 80,
    "Dark Pulse": 80,
    "Draco Meteor": 130,
    "Overheat": 130,
    "Leaf Storm": 130,
    "Hydro Pump": 110,
    "Fire Blast": 110,
    "Thunder": 110,
    "Blizzard": 110,
    "Focus Blast": 120,
    "Hurricane": 110,
    "Outrage": 120,
    "Play Rough": 90,
    "Icicle Crash": 85,
    "Wild Charge": 90,
    "Zen Headbutt": 80,
    "Crunch": 80,
    "Poison Jab": 80,
    "Waterfall": 80,
    "Seed Bomb": 80,
    "X-Scissor": 80,
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
