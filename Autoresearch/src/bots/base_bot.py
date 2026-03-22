"""Base bot interface for Pokemon battle agents.

All bots implement the Bot protocol:
    choose_action(observation, legal_actions) -> action

This interface is used by the battle harness to pit bots against each other.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from src.environment.action_space import ActionMask, BattleAction
from src.environment.battle_env import Observation


class Bot(ABC):
    """Abstract base class for battle bots.

    Subclasses must implement:
    - choose_action: Select an action given the current observation and legal actions
    - choose_team_order: Select team order during team preview

    Bots receive only first-person observations (no hidden state).
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name for this bot."""
        ...

    @abstractmethod
    def choose_action(
        self, observation: Observation, legal_actions: ActionMask
    ) -> BattleAction:
        """Choose an action given the current game state.

        Args:
            observation: First-person observation of the battle state.
            legal_actions: Binary mask of legal actions.

        Returns:
            The chosen BattleAction. Must be in the legal action set.
        """
        ...

    def choose_team_order(self, observation: Observation) -> str:
        """Choose team order during team preview.

        Args:
            observation: Observation containing team preview info.

        Returns:
            Team order string (e.g., "123456" for default).
        """
        return "123456"  # Default order

    def on_battle_start(self) -> None:
        """Called when a new battle begins. Override for per-battle setup."""
        pass

    def on_battle_end(self, won: bool | None, info: dict[str, Any] | None = None) -> None:
        """Called when a battle ends. Override for logging/learning."""
        pass

    def reset(self) -> None:
        """Reset bot state between battles."""
        pass
