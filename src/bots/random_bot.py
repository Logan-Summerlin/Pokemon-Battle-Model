"""Random legal bot: uniformly selects from legal actions.

Serves as the absolute floor for evaluation and as a stress test
for the legality mask system.
"""

from __future__ import annotations

import random
from typing import Any

from src.environment.action_space import ActionMask, BattleAction, action_from_canonical_index
from src.environment.battle_env import Observation
from src.bots.base_bot import Bot


class RandomBot(Bot):
    """Bot that uniformly selects a random legal action each turn."""

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)
        self._seed = seed
        self._games_played = 0
        self._wins = 0

    @property
    def name(self) -> str:
        return "RandomBot"

    def choose_action(
        self, observation: Observation, legal_actions: ActionMask
    ) -> BattleAction:
        """Uniformly sample from the legal action set."""
        legal_indices = legal_actions.legal_indices
        if not legal_indices:
            raise RuntimeError("No legal actions available for RandomBot")

        chosen_idx = self._rng.choice(legal_indices)
        return action_from_canonical_index(chosen_idx)

    def choose_team_order(self, observation: Observation) -> str:
        """Randomly shuffle team order."""
        order = list("123456")
        self._rng.shuffle(order)
        return "".join(order)

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

    def __repr__(self) -> str:
        return f"RandomBot(seed={self._seed}, games={self._games_played}, wins={self._wins})"
