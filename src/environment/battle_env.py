"""BattleEnv: Gym-like environment wrapper for Pokemon Showdown battles.

Provides a step(action) -> observation, reward, done, info interface
for interacting with the Showdown simulator. This is the primary interface
that bots and training pipelines use.

Critical design constraint: The observation must reconstruct only what
the acting player knew at decision time.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any

from src.environment.action_space import (
    ActionMask,
    BattleAction,
    action_from_canonical_index,
)
from src.environment.legality import get_legal_actions
from src.environment.protocol import BattleMessage, MessageType
from src.environment.showdown_client import ShowdownBattleStream, ShowdownClient, ShowdownConfig
from src.environment.state import BattleState, BattleStateTracker, GamePhase

logger = logging.getLogger(__name__)


@dataclass
class Observation:
    """The observation returned by BattleEnv.step().

    Contains only first-person information available to the acting player.
    """

    state: BattleState
    legal_actions: ActionMask
    turn: int = 0
    is_team_preview: bool = False
    raw_request: dict[str, Any] = field(default_factory=dict)

    @property
    def is_terminal(self) -> bool:
        return self.state.is_finished


@dataclass
class StepResult:
    """Result of a BattleEnv.step() call."""

    observation: Observation
    reward: float  # +1 for win, -1 for loss, 0 otherwise
    done: bool
    info: dict[str, Any] = field(default_factory=dict)


class BattleEnv:
    """Environment wrapper for a single Pokemon Showdown battle.

    Provides the standard RL environment interface:
    - reset() -> initial observation
    - step(action) -> observation, reward, done, info
    - get_legal_actions() -> action mask

    Usage:
        env = BattleEnv(client, room_id, player_id="p1")
        obs = await env.reset()
        while not obs.is_terminal:
            action = bot.choose_action(obs, obs.legal_actions)
            result = await env.step(action)
            obs = result.observation
    """

    def __init__(
        self,
        client: ShowdownClient,
        room_id: str,
        player_id: str = "p1",
    ) -> None:
        self.client = client
        self.room_id = room_id
        self.player_id = player_id

        self._tracker = BattleStateTracker(player_id)
        self._stream = ShowdownBattleStream(client, room_id)
        self._step_count = 0
        self._action_log: list[dict[str, Any]] = []

    @property
    def state(self) -> BattleState:
        return self._tracker.state

    @property
    def is_done(self) -> bool:
        return self.state.is_finished

    async def reset(self) -> Observation:
        """Initialize the environment and return the first observation.

        Receives messages until the first request (team preview or first turn).
        """
        messages = await self._stream.receive_until_request()
        self._tracker.process_messages(messages)

        legal = get_legal_actions(self.state)

        return Observation(
            state=self.state.snapshot(),
            legal_actions=legal,
            turn=self.state.turn,
            is_team_preview=self.state.phase == GamePhase.TEAM_PREVIEW,
            raw_request=self.state._current_request,
        )

    async def step(self, action: BattleAction | int) -> StepResult:
        """Take an action and return the next observation.

        Args:
            action: Either a BattleAction or a canonical action index.

        Returns:
            StepResult with observation, reward, done, and info.
        """
        if isinstance(action, int):
            action = action_from_canonical_index(action)

        # Validate legality
        legal = get_legal_actions(self.state)
        if not legal.is_legal(action.canonical_index):
            logger.warning(
                "Illegal action %s attempted. Legal: %s",
                action,
                legal.legal_indices,
            )
            # Fall back to first legal action
            if legal.any_legal:
                fallback_idx = legal.legal_indices[0]
                action = action_from_canonical_index(fallback_idx)
                logger.warning("Falling back to %s", action)
            else:
                raise RuntimeError("No legal actions available")

        # Log action
        self._step_count += 1
        self._action_log.append({
            "step": self._step_count,
            "turn": self.state.turn,
            "action": action.canonical_index,
            "command": action.to_showdown_command(),
        })

        # Send the action
        command = action.to_showdown_command()
        await self._stream.send_action(command)

        # Receive messages until next request or battle end
        messages = await self._stream.receive_until_request()
        self._tracker.process_messages(messages)

        # Compute reward
        reward = 0.0
        done = self.state.is_finished
        if done:
            if self.state.did_we_win is True:
                reward = 1.0
            elif self.state.did_we_win is False:
                reward = -1.0
            # Tie: reward = 0.0

        # Build observation
        new_legal = get_legal_actions(self.state) if not done else ActionMask()

        obs = Observation(
            state=self.state.snapshot(),
            legal_actions=new_legal,
            turn=self.state.turn,
            is_team_preview=self.state.phase == GamePhase.TEAM_PREVIEW,
            raw_request=self.state._current_request,
        )

        info: dict[str, Any] = {
            "step": self._step_count,
            "action_taken": action.canonical_index,
            "command_sent": command,
        }
        if done:
            info["winner"] = self.state.winner
            info["total_turns"] = self.state.turn
            info["action_log"] = list(self._action_log)

        return StepResult(
            observation=obs,
            reward=reward,
            done=done,
            info=info,
        )

    async def step_team_preview(self, order: str = "123456") -> Observation:
        """Handle team preview by sending team order.

        Args:
            order: Team order as a string of digits (e.g., "123456").

        Returns:
            The next observation (first turn of battle).
        """
        await self._stream.send_team_order(order)

        # Receive messages until next request
        messages = await self._stream.receive_until_request()
        self._tracker.process_messages(messages)

        legal = get_legal_actions(self.state)

        return Observation(
            state=self.state.snapshot(),
            legal_actions=legal,
            turn=self.state.turn,
            is_team_preview=self.state.phase == GamePhase.TEAM_PREVIEW,
            raw_request=self.state._current_request,
        )

    async def wait_for_update(self) -> StepResult:
        """Wait for the next state update without sending an action.

        Used when the player is in a "wait" state (e.g., the opponent is
        making a forced switch) and we need to receive the next request.
        """
        messages = await self._stream.receive_until_request()
        self._tracker.process_messages(messages)

        reward = 0.0
        done = self.state.is_finished
        if done:
            if self.state.did_we_win is True:
                reward = 1.0
            elif self.state.did_we_win is False:
                reward = -1.0

        new_legal = get_legal_actions(self.state) if not done else ActionMask()
        obs = Observation(
            state=self.state.snapshot(),
            legal_actions=new_legal,
            turn=self.state.turn,
            is_team_preview=self.state.phase == GamePhase.TEAM_PREVIEW,
            raw_request=self.state._current_request,
        )
        info: dict[str, Any] = {"waited": True}
        if done:
            info["winner"] = self.state.winner
            info["total_turns"] = self.state.turn

        return StepResult(observation=obs, reward=reward, done=done, info=info)

    def get_action_log(self) -> list[dict[str, Any]]:
        """Get the full action log for this battle."""
        return list(self._action_log)

    def get_replay_data(self) -> dict[str, Any]:
        """Get data needed to reconstruct this battle."""
        return {
            "room_id": self.room_id,
            "player_id": self.player_id,
            "action_log": self._action_log,
            "total_turns": self.state.turn,
            "winner": self.state.winner,
            "own_team": [p.species for p in self.state.own_team],
            "opponent_team": [p.species for p in self.state.opponent_team],
        }


class LocalBattleEnv:
    """A simulated battle environment for running games without a live server.

    This wraps the state tracking and action/legality logic to support
    replaying battles from log data or running simple simulations.
    Useful for testing the pipeline without a running Showdown server.
    """

    def __init__(self, player_id: str = "p1") -> None:
        self.player_id = player_id
        self._tracker = BattleStateTracker(player_id)
        self._step_count = 0

    @property
    def state(self) -> BattleState:
        return self._tracker.state

    def process_messages(self, messages: list[BattleMessage]) -> None:
        """Feed protocol messages into the state tracker."""
        self._tracker.process_messages(messages)

    def process_request(self, request_json: str) -> None:
        """Update state from a request JSON string."""
        self._tracker.update_from_request(request_json)

    def get_legal_actions(self) -> ActionMask:
        """Get the current legal action mask."""
        return get_legal_actions(self.state)

    def get_observation(self) -> Observation:
        """Get the current observation."""
        legal = self.get_legal_actions()
        return Observation(
            state=self.state.snapshot(),
            legal_actions=legal,
            turn=self.state.turn,
            is_team_preview=self.state.phase == GamePhase.TEAM_PREVIEW,
            raw_request=self.state._current_request,
        )
