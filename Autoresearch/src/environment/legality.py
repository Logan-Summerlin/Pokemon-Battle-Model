"""Legal action mask computation.

Computes which actions are legal at each decision point in a battle,
handling Gen 3 OU edge cases:
- Forced switches (after faint)
- Trapping (Shadow Tag, Arena Trap, Magnet Pull, trapping moves)
- Choice Band lock (only Choice Band exists in Gen 3; no Choice Specs/Scarf)
- Encore (locks into last used move)
- Disabled moves
- Struggle (when all moves have 0 PP or are disabled)
- No Terastallization, no Z-moves, no Mega Evolution in Gen 3
"""

from __future__ import annotations

from typing import Any

from src.environment.action_space import (
    MOVE_1,
    MOVE_4,
    SWITCH_2,
    SWITCH_6,
    ActionMask,
)
from src.environment.state import BattleState, GamePhase, OwnPokemon

# Abilities that prevent switching
TRAPPING_ABILITIES = frozenset({
    "shadowtag",
    "arenatrap",
    "magnetpull",
})

# Moves that trap the opponent (Gen 3 only — no Infestation, Magma Storm,
# Thousand Waves, or Sand Tomb trapping; Sand Tomb exists but doesn't trap)
TRAPPING_MOVES = frozenset({
    "bind",
    "clamp",
    "firespin",
    "whirlpool",
    "wrap",
})

# Ghost types are immune to trapping (except from Shadow Tag on non-Ghost)
GHOST_TYPES = frozenset({"Ghost"})


def get_legal_actions(state: BattleState) -> ActionMask:
    """Compute the legal action mask for the current game state.

    This is the primary entry point for legality checking. It examines the
    current request data and battle state to determine which actions are legal.

    Returns:
        ActionMask with True for each legal action index.
    """
    mask = ActionMask()
    request = state._current_request

    if not request or state.is_finished:
        return mask

    # Handle wait request (opponent still needs to act)
    if request.get("wait"):
        return mask

    # Handle team preview (not a move/switch decision)
    if request.get("teamPreview"):
        return _get_team_preview_actions(state)

    # Handle forced switch (after faint)
    if request.get("forceSwitch"):
        return _get_forced_switch_actions(state)

    # Normal turn: compute move and switch legality
    _compute_move_legality(state, mask)
    _compute_switch_legality(state, mask)

    # If nothing is legal (shouldn't happen, but safety), allow Struggle
    if not mask.any_legal:
        # Struggle is represented as move 1 when no moves are available
        mask.set_legal(MOVE_1)

    return mask


def _get_team_preview_actions(state: BattleState) -> ActionMask:
    """During team preview, all team orderings are valid.

    For simplicity in the action space, team preview is handled separately
    from in-battle actions. Returns an empty mask — team preview uses a
    different selection mechanism (team order).
    """
    # Team preview doesn't map to the standard action space.
    # The battle env handles this separately.
    return ActionMask()


def _get_forced_switch_actions(state: BattleState) -> ActionMask:
    """Compute legal switches when forced to switch (e.g., after faint).

    Only switch actions are legal; no moves can be used.
    """
    mask = ActionMask()
    request = state._current_request

    # Get available Pokemon from the request
    side = request.get("side", {})
    pokemon_list = side.get("pokemon", [])

    for i, poke_data in enumerate(pokemon_list):
        # Can't switch to the active Pokemon
        if poke_data.get("active", False):
            continue
        # Can't switch to fainted Pokemon
        condition = poke_data.get("condition", "")
        if condition == "0 fnt":
            continue

        # Map team index to switch action index
        # Team indices are 0-based; switch actions are for non-active slots
        # switch_index 0 = team slot 2 (index 1), etc.
        switch_action_idx = SWITCH_2 + (i - 1)
        if SWITCH_2 <= switch_action_idx <= SWITCH_6:
            mask.set_legal(switch_action_idx)

    return mask


def _compute_move_legality(state: BattleState, mask: ActionMask) -> None:
    """Determine which move actions are legal.

    Handles: disabled moves, 0 PP, choice lock, Encore, Taunt,
    Torment, Assault Vest, and other move restrictions.
    """
    request = state._current_request
    active_data = request.get("active", [{}])
    if not active_data:
        return

    active = active_data[0]
    moves = active.get("moves", [])

    if not moves:
        return

    # Check if all moves are unusable → Struggle
    all_disabled = True

    for i, move_data in enumerate(moves):
        if i > 3:
            break  # Max 4 moves

        # A move is legal if it's not disabled and has PP remaining
        disabled = move_data.get("disabled", False)
        pp = move_data.get("pp", 1)

        if not disabled and pp > 0:
            mask.set_legal(MOVE_1 + i)
            all_disabled = False

    # If all moves are disabled/0 PP, the only legal move is Struggle
    # Struggle is represented as move 1 (index 0) in our action space
    if all_disabled:
        mask.set_legal(MOVE_1)

    # Handle special move restrictions from the request.
    # The server already marks moves as disabled in the request data,
    # so we trust the server's legality determination for moves.
    # Our job is to correctly map this to the action mask.
    # Gen 3 has no Z-moves, Mega Evolution, Dynamax, or Terastallization.


def _compute_switch_legality(state: BattleState, mask: ActionMask) -> None:
    """Determine which switch actions are legal.

    Handles: trapping abilities, trapping moves, ingrain, no retreat,
    and other switch-prevention effects.
    """
    request = state._current_request

    # Check if the player is trapped (can't switch at all)
    if _is_trapped(state):
        return  # No switch actions legal

    # Get available Pokemon from the request
    side = request.get("side", {})
    pokemon_list = side.get("pokemon", [])

    for i, poke_data in enumerate(pokemon_list):
        # Can't switch to the active Pokemon
        if poke_data.get("active", False):
            continue
        # Can't switch to fainted Pokemon
        condition = poke_data.get("condition", "")
        if condition == "0 fnt":
            continue

        # Map team index to switch action index
        switch_action_idx = SWITCH_2 + (i - 1)
        if SWITCH_2 <= switch_action_idx <= SWITCH_6:
            mask.set_legal(switch_action_idx)


def _is_trapped(state: BattleState) -> bool:
    """Check if the active Pokemon is trapped and cannot switch.

    Reasons for being trapped:
    - Opponent has Shadow Tag (and we don't have Shadow Tag)
    - Opponent has Arena Trap (and we're grounded)
    - Opponent has Magnet Pull (and we're Steel type)
    - We're affected by a trapping move (Bind, Wrap, etc.)
    - We used Ingrain, No Retreat, or similar
    """
    request = state._current_request

    # The server tells us if we're trapped via the 'trapped' flag
    active_data = request.get("active", [{}])
    if active_data:
        active = active_data[0]
        if active.get("trapped") or active.get("maybeTrapped"):
            return True

    # Also check volatile statuses (Gen 3: no No Retreat, only partial trap)
    if state.own_active:
        trapping_volatiles = {"partiallytrapped", "trapped"}
        if state.own_active.volatiles & trapping_volatiles:
            return True

    return False


def validate_action_against_mask(
    action_index: int, mask: ActionMask
) -> tuple[bool, str]:
    """Check if a specific action is legal according to the mask.

    Returns:
        Tuple of (is_legal, reason_if_illegal).
    """
    if mask.is_legal(action_index):
        return True, ""

    from src.environment.action_space import ACTION_NAMES

    action_name = ACTION_NAMES[action_index] if action_index < len(ACTION_NAMES) else f"action {action_index}"

    # Provide specific reason
    if MOVE_1 <= action_index <= MOVE_4:
        return False, f"{action_name}: move disabled, 0 PP, or not available"
    elif SWITCH_2 <= action_index <= SWITCH_6:
        return False, f"{action_name}: target fainted, is active, or switching is blocked"
    else:
        return False, f"invalid action index: {action_index}"


def get_legal_actions_from_request(request: dict[str, Any]) -> ActionMask:
    """Compute legal action mask directly from a Showdown request JSON.

    This is a convenience function that creates a minimal BattleState
    from the request data for legality checking.
    """
    state = BattleState()
    state._current_request = request

    # Extract player id from request
    side = request.get("side", {})
    if side.get("id"):
        state.player_id = side["id"]
        state.opponent_id = "p2" if state.player_id == "p1" else "p1"

    # Set phase
    if request.get("teamPreview"):
        state.phase = GamePhase.TEAM_PREVIEW
    else:
        state.phase = GamePhase.BATTLE

    # Build minimal own_team from request for switch legality
    pokemon_list = side.get("pokemon", [])
    for i, poke_data in enumerate(pokemon_list):
        from src.environment.state import OwnPokemon

        poke = OwnPokemon()
        poke.active = poke_data.get("active", False)
        condition = poke_data.get("condition", "")
        if condition == "0 fnt":
            poke.fainted = True
            poke.current_hp = 0
        if poke.active:
            state.own_active_index = i
        state.own_team.append(poke)

    return get_legal_actions(state)
