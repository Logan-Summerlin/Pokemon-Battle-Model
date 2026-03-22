"""Model bot: wraps a trained neural network model as a Bot.

Used to evaluate trained models (MLP, GRU, or Transformer) against
scripted bots in the battle harness. Converts live Observation objects
to tensors, runs inference, and returns the chosen action.
"""

from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from src.bots.base_bot import Bot
from src.data.observation import (
    UNKNOWN,
    FieldObservation,
    PokemonObservation,
    TurnObservation,
)
from src.data.tensorizer import (
    BattleVocabularies,
    tensorize_turn,
)
from src.environment.action_space import (
    NUM_ACTIONS,
    ActionMask,
    BattleAction,
    action_from_canonical_index,
)
from src.environment.battle_env import Observation
from src.environment.state import BattleState, OwnPokemon, OpponentPokemon


def _own_pokemon_to_obs(poke: OwnPokemon) -> PokemonObservation:
    """Convert an OwnPokemon state to PokemonObservation for tensorization."""
    return PokemonObservation(
        species=poke.species,
        hp_fraction=poke.hp_fraction,
        status=poke.status,
        is_active=poke.active,
        is_fainted=poke.fainted,
        moves=[m.name for m in poke.moves if m.name],
        item=poke.item or UNKNOWN,
        ability=poke.ability or UNKNOWN,
        boosts=dict(poke.boosts),
        base_stats=dict(poke.stats) if poke.stats else {},
        types="",
        level=poke.level,
        is_own=True,
    )


def _opp_pokemon_to_obs(poke: OpponentPokemon) -> PokemonObservation:
    """Convert an OpponentPokemon state to PokemonObservation for tensorization."""
    return PokemonObservation(
        species=poke.species,
        hp_fraction=poke.hp_fraction,
        status=poke.status,
        is_active=poke.active,
        is_fainted=poke.fainted,
        moves=list(poke.revealed_moves),
        item=poke.item if poke.item != UNKNOWN else UNKNOWN,
        ability=poke.ability if poke.ability != UNKNOWN else UNKNOWN,
        boosts=dict(poke.boosts),
        base_stats={},
        types="",
        level=poke.level,
        is_own=False,
    )


def _state_to_turn_obs(
    state: BattleState,
    legal_actions: ActionMask,
) -> TurnObservation:
    """Convert a live BattleState to a TurnObservation for tensorization."""
    own_team = [_own_pokemon_to_obs(p) for p in state.own_team]
    opp_team = [_opp_pokemon_to_obs(p) for p in state.opponent_team]

    # Pad teams to 6
    while len(own_team) < 6:
        own_team.append(PokemonObservation(is_own=True, is_fainted=True, hp_fraction=0.0))
    while len(opp_team) < 6:
        opp_team.append(PokemonObservation(is_own=False))

    # Build field observation from state
    field_obs = FieldObservation(
        weather=state.field.weather,
        terrain=state.field.terrain,
        own_stealth_rock=state.own_side.stealth_rock,
        own_spikes=state.own_side.spikes,
        own_toxic_spikes=state.own_side.toxic_spikes,
        own_sticky_web=state.own_side.sticky_web,
        own_reflect=state.own_side.reflect > 0,
        own_light_screen=state.own_side.light_screen > 0,
        own_aurora_veil=state.own_side.aurora_veil > 0,
        own_tailwind=state.own_side.tailwind > 0,
        opp_stealth_rock=state.opponent_side.stealth_rock,
        opp_spikes=state.opponent_side.spikes,
        opp_toxic_spikes=state.opponent_side.toxic_spikes,
        opp_sticky_web=state.opponent_side.sticky_web,
        opp_reflect=state.opponent_side.reflect > 0,
        opp_light_screen=state.opponent_side.light_screen > 0,
        opp_aurora_veil=state.opponent_side.aurora_veil > 0,
        opp_tailwind=state.opponent_side.tailwind > 0,
    )

    # Get previous moves from turn history
    prev_player_move = ""
    prev_opponent_move = ""
    if state.turn_history:
        last = state.turn_history[-1]
        if last.our_action.startswith("move "):
            prev_player_move = last.our_action[5:]
        if last.opponent_action.startswith("move "):
            prev_opponent_move = last.opponent_action[5:]

    return TurnObservation(
        turn_number=state.turn,
        own_team=own_team[:6],
        opponent_team=opp_team[:6],
        field=field_obs,
        legal_action_mask=legal_actions.to_list(),
        forced_switch=False,
        opponents_remaining=state.opponent_alive_count,
        num_opponent_revealed=sum(1 for p in state.opponent_team if p.seen_in_battle),
        is_lead_turn=(state.turn <= 1),
        prev_player_move=prev_player_move,
        prev_opponent_move=prev_opponent_move,
    )


class ModelBot(Bot):
    """Bot that uses a trained model to select actions.

    Converts live observations to tensors, runs model inference,
    and selects the highest-scoring legal action.
    """

    def __init__(
        self,
        model: nn.Module,
        vocabs: BattleVocabularies,
        device: str | torch.device = "cpu",
        temperature: float = 1.0,
        seed: int | None = None,
        bot_name: str = "ModelBot",
    ) -> None:
        self._model = model
        self._vocabs = vocabs
        self._device = torch.device(device)
        self._temperature = temperature
        self._rng = random.Random(seed)
        self._seed = seed
        self._bot_name = bot_name
        self._games_played = 0
        self._wins = 0

        self._model.to(self._device)
        self._model.eval()

    @property
    def name(self) -> str:
        return self._bot_name

    @torch.no_grad()
    def choose_action(
        self, observation: Observation, legal_actions: ActionMask
    ) -> BattleAction:
        legal_indices = legal_actions.legal_indices
        if not legal_indices:
            raise RuntimeError("No legal actions available for ModelBot")

        # Convert live state to TurnObservation
        turn_obs = _state_to_turn_obs(observation.state, legal_actions)

        # Tensorize
        tensor_dict = tensorize_turn(turn_obs, self._vocabs, build_vocab=False)

        # Convert to torch tensors with batch dim
        own_team = torch.from_numpy(tensor_dict["own_team"]).unsqueeze(0).to(self._device)
        opp_team = torch.from_numpy(tensor_dict["opponent_team"]).unsqueeze(0).to(self._device)
        field = torch.from_numpy(tensor_dict["field"]).unsqueeze(0).to(self._device)
        context = torch.from_numpy(tensor_dict["context"]).unsqueeze(0).to(self._device)
        legal_mask = torch.from_numpy(tensor_dict["legal_mask"]).unsqueeze(0).to(self._device)

        # Run model - handle both baseline (returns Tensor) and transformer (returns TransformerOutput)
        output = self._model(own_team, opp_team, field, context, legal_mask=legal_mask)

        # Extract logits from output
        if isinstance(output, torch.Tensor):
            logits = output
        else:
            # TransformerOutput or similar
            logits = output.policy_logits if hasattr(output, "policy_logits") else output

        # Select action
        if self._temperature <= 0:
            # Greedy
            action_idx = logits[0].argmax().item()
        else:
            # Sample with temperature
            probs = torch.softmax(logits[0] / self._temperature, dim=-1)
            action_idx = torch.multinomial(probs, 1).item()

        # Validate the action is legal
        if not legal_actions.is_legal(action_idx):
            # Fallback to highest-scoring legal action
            masked_logits = logits[0].clone()
            for i in range(NUM_ACTIONS):
                if not legal_actions.is_legal(i):
                    masked_logits[i] = float("-inf")
            action_idx = masked_logits.argmax().item()

        return action_from_canonical_index(action_idx)

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


def load_transformer_bot(
    checkpoint_path: str,
    vocabs_dir: str,
    device: str = "cpu",
    temperature: float = 1.0,
    bot_name: str = "TransformerBot",
) -> ModelBot:
    """Load a BattleTransformer model from checkpoint and wrap as a bot.

    Args:
        checkpoint_path: Path to the .pt checkpoint file
        vocabs_dir: Path to the vocabularies directory
        device: Device to run on
        temperature: Sampling temperature (0 = greedy)
        bot_name: Name for the bot

    Returns:
        ModelBot wrapping the loaded transformer.
    """
    from src.models.battle_transformer import BattleTransformer, TransformerConfig

    vocabs = BattleVocabularies.load(vocabs_dir)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Reconstruct config from checkpoint
    ckpt_config = ckpt.get("config", {})
    config = TransformerConfig.from_vocabs(vocabs, **{
        k: v for k, v in ckpt_config.items()
        if hasattr(TransformerConfig, k)
    })

    model = BattleTransformer(config)
    model.load_state_dict(ckpt["model_state_dict"])

    return ModelBot(
        model=model,
        vocabs=vocabs,
        device=device,
        temperature=temperature,
        bot_name=bot_name,
    )
