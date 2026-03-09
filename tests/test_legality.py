"""Tests for the legality and action space system.

Covers:
- Legal action mask generation for all game situations
- Edge cases: trapping, choice lock, Encore, disabled moves, fainted forced-switch,
  Tera availability, Struggle, etc.
- Comparison of legality masks against expected behavior
- Action encoding/decoding round-trips

These tests use mock request data (the JSON the server sends) to verify
legality computation without needing a live server.
"""

import json

import pytest

from src.environment.action_space import (
    MOVE_1,
    MOVE_1_TERA,
    MOVE_2,
    MOVE_2_TERA,
    MOVE_3,
    MOVE_4,
    MOVE_4_TERA,
    NUM_ACTIONS,
    SWITCH_2,
    SWITCH_3,
    SWITCH_4,
    SWITCH_5,
    SWITCH_6,
    ActionMask,
    ActionType,
    BattleAction,
    action_from_canonical_index,
    action_from_showdown_choice,
)
from src.environment.legality import (
    get_legal_actions,
    get_legal_actions_from_request,
    validate_action_against_mask,
)
from src.environment.state import BattleState, BattleStateTracker, GamePhase


def _make_request(
    moves: list[dict] | None = None,
    pokemon: list[dict] | None = None,
    force_switch: bool = False,
    team_preview: bool = False,
    wait: bool = False,
    trapped: bool = False,
    can_tera: bool = False,
) -> dict:
    """Build a mock Showdown request JSON."""
    if moves is None:
        moves = [
            {"move": "Earthquake", "id": "earthquake", "pp": 16, "maxpp": 16, "disabled": False},
            {"move": "Swords Dance", "id": "swordsdance", "pp": 32, "maxpp": 32, "disabled": False},
            {"move": "Scale Shot", "id": "scaleshot", "pp": 32, "maxpp": 32, "disabled": False},
            {"move": "Iron Head", "id": "ironhead", "pp": 24, "maxpp": 24, "disabled": False},
        ]

    if pokemon is None:
        pokemon = [
            {"ident": "p1: Garchomp", "details": "Garchomp, L100, M", "condition": "350/350", "active": True},
            {"ident": "p1: Heatran", "details": "Heatran, L100, M", "condition": "300/300", "active": False},
            {"ident": "p1: Landorus", "details": "Landorus-Therian, L100, M", "condition": "319/319", "active": False},
            {"ident": "p1: Ferrothorn", "details": "Ferrothorn, L100, M", "condition": "0 fnt", "active": False},
            {"ident": "p1: Slowking", "details": "Slowking-Galar, L100, M", "condition": "394/394", "active": False},
            {"ident": "p1: Weavile", "details": "Weavile, L100, M", "condition": "281/281", "active": False},
        ]

    active: list[dict] = []
    if not force_switch and not team_preview and not wait:
        active_data: dict = {"moves": moves}
        if trapped:
            active_data["trapped"] = True
        if can_tera:
            active_data["canTerastallize"] = True
        active = [active_data]

    request: dict = {
        "side": {
            "id": "p1",
            "name": "TestPlayer",
            "pokemon": pokemon,
        },
    }
    if active:
        request["active"] = active
    if force_switch:
        request["forceSwitch"] = [True]
    if team_preview:
        request["teamPreview"] = True
    if wait:
        request["wait"] = True

    return request


class TestActionSpace:
    """Test the canonical action vocabulary."""

    def test_action_indices(self) -> None:
        assert MOVE_1 == 0
        assert MOVE_4 == 3
        assert MOVE_1_TERA == 4
        assert MOVE_4_TERA == 7
        assert SWITCH_2 == 8
        assert SWITCH_6 == 12
        assert NUM_ACTIONS == 13

    def test_action_from_index_moves(self) -> None:
        for i in range(4):
            action = action_from_canonical_index(i)
            assert action.action_type == ActionType.MOVE
            assert action.move_index == i

    def test_action_from_index_tera_moves(self) -> None:
        for i in range(4):
            action = action_from_canonical_index(MOVE_1_TERA + i)
            assert action.action_type == ActionType.MOVE_TERA
            assert action.move_index == i
            assert action.terastallize is True

    def test_action_from_index_switches(self) -> None:
        for i in range(5):
            action = action_from_canonical_index(SWITCH_2 + i)
            assert action.action_type == ActionType.SWITCH
            assert action.switch_index == i

    def test_action_from_index_invalid(self) -> None:
        with pytest.raises(ValueError):
            action_from_canonical_index(-1)
        with pytest.raises(ValueError):
            action_from_canonical_index(NUM_ACTIONS)

    def test_canonical_index_round_trip(self) -> None:
        for i in range(NUM_ACTIONS):
            action = action_from_canonical_index(i)
            assert action.canonical_index == i

    def test_showdown_command_moves(self) -> None:
        action = action_from_canonical_index(MOVE_1)
        assert action.to_showdown_command() == "/choose move 1"
        action = action_from_canonical_index(MOVE_4)
        assert action.to_showdown_command() == "/choose move 4"

    def test_showdown_command_tera(self) -> None:
        action = action_from_canonical_index(MOVE_2_TERA)
        cmd = action.to_showdown_command()
        assert "move 2" in cmd
        assert "terastallize" in cmd

    def test_showdown_command_switch(self) -> None:
        action = action_from_canonical_index(SWITCH_2)
        assert action.to_showdown_command() == "/choose switch 2"
        action = action_from_canonical_index(SWITCH_6)
        assert action.to_showdown_command() == "/choose switch 6"


class TestActionFromChoice:
    """Test parsing Showdown choice strings."""

    def test_move_choice(self) -> None:
        action = action_from_showdown_choice("move 1")
        assert action is not None
        assert action.action_type == ActionType.MOVE
        assert action.move_index == 0

    def test_move_tera_choice(self) -> None:
        action = action_from_showdown_choice("move 3 terastallize")
        assert action is not None
        assert action.terastallize is True
        assert action.move_index == 2

    def test_switch_choice(self) -> None:
        action = action_from_showdown_choice("switch 4")
        assert action is not None
        assert action.action_type == ActionType.SWITCH
        assert action.switch_index == 2

    def test_with_choose_prefix(self) -> None:
        action = action_from_showdown_choice("/choose move 2")
        assert action is not None
        assert action.move_index == 1

    def test_invalid_choice(self) -> None:
        assert action_from_showdown_choice("invalid") is None
        assert action_from_showdown_choice("") is None
        assert action_from_showdown_choice("move 0") is None
        assert action_from_showdown_choice("move 5") is None


class TestActionMask:
    """Test ActionMask operations."""

    def test_empty_mask(self) -> None:
        mask = ActionMask()
        assert mask.num_legal == 0
        assert not mask.any_legal

    def test_set_legal(self) -> None:
        mask = ActionMask()
        mask.set_legal(MOVE_1)
        mask.set_legal(SWITCH_3)
        assert mask.is_legal(MOVE_1)
        assert mask.is_legal(SWITCH_3)
        assert not mask.is_legal(MOVE_2)
        assert mask.num_legal == 2

    def test_from_list(self) -> None:
        mask = ActionMask.from_list([MOVE_1, MOVE_2, SWITCH_2])
        assert mask.num_legal == 3
        assert mask.legal_indices == [MOVE_1, MOVE_2, SWITCH_2]

    def test_all_moves(self) -> None:
        mask = ActionMask.all_moves()
        assert mask.num_legal == 4
        assert all(mask.is_legal(i) for i in range(MOVE_1, MOVE_4 + 1))

    def test_to_int_list(self) -> None:
        mask = ActionMask.from_list([MOVE_1, SWITCH_2])
        int_list = mask.to_int_list()
        assert len(int_list) == NUM_ACTIONS
        assert int_list[MOVE_1] == 1
        assert int_list[SWITCH_2] == 1
        assert int_list[MOVE_2] == 0


class TestLegalActionMask:
    """Test legal action mask generation from game state."""

    def test_normal_turn_all_moves_legal(self) -> None:
        """All 4 moves + non-fainted switches should be legal."""
        request = _make_request()
        mask = get_legal_actions_from_request(request)

        # All 4 moves legal
        for i in range(MOVE_1, MOVE_4 + 1):
            assert mask.is_legal(i), f"Move {i} should be legal"

        # Switches: Heatran(idx1), Landorus(idx2), Slowking(idx4), Weavile(idx5) alive
        # Ferrothorn(idx3) fainted
        assert mask.is_legal(SWITCH_2)  # Heatran
        assert mask.is_legal(SWITCH_3)  # Landorus
        assert not mask.is_legal(SWITCH_4)  # Ferrothorn - fainted
        assert mask.is_legal(SWITCH_5)  # Slowking
        assert mask.is_legal(SWITCH_6)  # Weavile

    def test_disabled_move(self) -> None:
        """A disabled move should not be legal."""
        moves = [
            {"move": "Earthquake", "pp": 16, "maxpp": 16, "disabled": True},
            {"move": "Swords Dance", "pp": 32, "maxpp": 32, "disabled": False},
            {"move": "Scale Shot", "pp": 32, "maxpp": 32, "disabled": False},
            {"move": "Iron Head", "pp": 24, "maxpp": 24, "disabled": False},
        ]
        request = _make_request(moves=moves)
        mask = get_legal_actions_from_request(request)

        assert not mask.is_legal(MOVE_1)
        assert mask.is_legal(MOVE_2)
        assert mask.is_legal(MOVE_3)
        assert mask.is_legal(MOVE_4)

    def test_zero_pp_move(self) -> None:
        """A move with 0 PP should not be legal."""
        moves = [
            {"move": "Earthquake", "pp": 0, "maxpp": 16, "disabled": False},
            {"move": "Swords Dance", "pp": 32, "maxpp": 32, "disabled": False},
            {"move": "Scale Shot", "pp": 0, "maxpp": 32, "disabled": False},
            {"move": "Iron Head", "pp": 24, "maxpp": 24, "disabled": False},
        ]
        request = _make_request(moves=moves)
        mask = get_legal_actions_from_request(request)

        assert not mask.is_legal(MOVE_1)
        assert mask.is_legal(MOVE_2)
        assert not mask.is_legal(MOVE_3)
        assert mask.is_legal(MOVE_4)

    def test_all_moves_disabled_means_struggle(self) -> None:
        """When all moves are disabled/0 PP, Struggle (move 1) should be legal."""
        moves = [
            {"move": "Earthquake", "pp": 0, "maxpp": 16, "disabled": False},
            {"move": "Swords Dance", "pp": 0, "maxpp": 32, "disabled": False},
            {"move": "Scale Shot", "pp": 0, "maxpp": 32, "disabled": False},
            {"move": "Iron Head", "pp": 0, "maxpp": 24, "disabled": False},
        ]
        request = _make_request(moves=moves)
        mask = get_legal_actions_from_request(request)

        # Struggle represented as move 1
        assert mask.is_legal(MOVE_1)
        # Switches should still be available
        assert mask.is_legal(SWITCH_2)

    def test_forced_switch_only_switches_legal(self) -> None:
        """During forced switch, only switch actions should be legal."""
        request = _make_request(force_switch=True)
        mask = get_legal_actions_from_request(request)

        # No moves
        for i in range(MOVE_1, MOVE_4_TERA + 1):
            assert not mask.is_legal(i)

        # Only non-fainted, non-active switches
        assert mask.is_legal(SWITCH_2)  # Heatran - alive
        assert mask.is_legal(SWITCH_3)  # Landorus - alive
        assert not mask.is_legal(SWITCH_4)  # Ferrothorn - fainted
        assert mask.is_legal(SWITCH_5)  # Slowking - alive
        assert mask.is_legal(SWITCH_6)  # Weavile - alive

    def test_forced_switch_all_fainted_except_one(self) -> None:
        """When only one non-fainted Pokemon left, it's the only switch."""
        pokemon = [
            {"ident": "p1: Garchomp", "details": "Garchomp, L100, M", "condition": "0 fnt", "active": False},
            {"ident": "p1: Heatran", "details": "Heatran, L100, M", "condition": "300/300", "active": False},
            {"ident": "p1: Landorus", "details": "Landorus-Therian, L100, M", "condition": "0 fnt", "active": False},
            {"ident": "p1: Ferrothorn", "details": "Ferrothorn, L100, M", "condition": "0 fnt", "active": False},
            {"ident": "p1: Slowking", "details": "Slowking-Galar, L100, M", "condition": "0 fnt", "active": False},
            {"ident": "p1: Weavile", "details": "Weavile, L100, M", "condition": "0 fnt", "active": False},
        ]
        request = _make_request(pokemon=pokemon, force_switch=True)
        mask = get_legal_actions_from_request(request)

        assert mask.num_legal == 1
        assert mask.is_legal(SWITCH_2)  # Heatran is the only one alive

    def test_trapped_cannot_switch(self) -> None:
        """When trapped (Shadow Tag, etc.), switch actions should be illegal."""
        request = _make_request(trapped=True)
        mask = get_legal_actions_from_request(request)

        # Moves should be legal
        assert mask.is_legal(MOVE_1)
        assert mask.is_legal(MOVE_2)

        # No switches
        for i in range(SWITCH_2, SWITCH_6 + 1):
            assert not mask.is_legal(i)

    def test_tera_moves_legal_when_can_tera(self) -> None:
        """Tera move variants should be legal when terastallization is available."""
        request = _make_request(can_tera=True)
        mask = get_legal_actions_from_request(request)

        # Normal moves legal
        for i in range(MOVE_1, MOVE_4 + 1):
            assert mask.is_legal(i)

        # Tera variants also legal
        for i in range(MOVE_1_TERA, MOVE_4_TERA + 1):
            assert mask.is_legal(i)

    def test_no_tera_moves_when_already_used(self) -> None:
        """Tera moves should be illegal when tera was already used."""
        request = _make_request(can_tera=False)
        mask = get_legal_actions_from_request(request)

        # Tera variants not legal
        for i in range(MOVE_1_TERA, MOVE_4_TERA + 1):
            assert not mask.is_legal(i)

    def test_wait_request_empty_mask(self) -> None:
        """Wait request means no actions available."""
        request = _make_request(wait=True)
        mask = get_legal_actions_from_request(request)
        assert mask.num_legal == 0

    def test_team_preview_empty_mask(self) -> None:
        """Team preview uses different mechanism; standard mask should be empty."""
        request = _make_request(team_preview=True)
        mask = get_legal_actions_from_request(request)
        # Team preview returns empty mask (handled separately)
        assert mask.num_legal == 0

    def test_choice_locked_one_move(self) -> None:
        """Choice item locks into one move; server marks others as disabled."""
        moves = [
            {"move": "Earthquake", "pp": 15, "maxpp": 16, "disabled": False},
            {"move": "Swords Dance", "pp": 32, "maxpp": 32, "disabled": True},
            {"move": "Scale Shot", "pp": 32, "maxpp": 32, "disabled": True},
            {"move": "Iron Head", "pp": 24, "maxpp": 24, "disabled": True},
        ]
        request = _make_request(moves=moves)
        mask = get_legal_actions_from_request(request)

        assert mask.is_legal(MOVE_1)
        assert not mask.is_legal(MOVE_2)
        assert not mask.is_legal(MOVE_3)
        assert not mask.is_legal(MOVE_4)

    def test_encore_locked_move(self) -> None:
        """Encore forces one move; server marks others as disabled."""
        moves = [
            {"move": "Earthquake", "pp": 15, "maxpp": 16, "disabled": True},
            {"move": "Swords Dance", "pp": 31, "maxpp": 32, "disabled": False},
            {"move": "Scale Shot", "pp": 32, "maxpp": 32, "disabled": True},
            {"move": "Iron Head", "pp": 24, "maxpp": 24, "disabled": True},
        ]
        request = _make_request(moves=moves)
        mask = get_legal_actions_from_request(request)

        assert not mask.is_legal(MOVE_1)
        assert mask.is_legal(MOVE_2)
        assert not mask.is_legal(MOVE_3)
        assert not mask.is_legal(MOVE_4)


class TestValidateAction:
    """Test action validation against a mask."""

    def test_valid_action(self) -> None:
        mask = ActionMask.from_list([MOVE_1, MOVE_2, SWITCH_2])
        is_legal, reason = validate_action_against_mask(MOVE_1, mask)
        assert is_legal
        assert reason == ""

    def test_invalid_move(self) -> None:
        mask = ActionMask.from_list([MOVE_1, SWITCH_2])
        is_legal, reason = validate_action_against_mask(MOVE_3, mask)
        assert not is_legal
        assert "disabled" in reason or "not legal" in reason

    def test_invalid_switch(self) -> None:
        mask = ActionMask.from_list([MOVE_1, MOVE_2])
        is_legal, reason = validate_action_against_mask(SWITCH_3, mask)
        assert not is_legal
        assert "fainted" in reason or "blocked" in reason

    def test_invalid_tera_when_base_illegal(self) -> None:
        mask = ActionMask.from_list([MOVE_2])  # Only move 2 legal
        is_legal, reason = validate_action_against_mask(MOVE_1_TERA, mask)
        assert not is_legal
