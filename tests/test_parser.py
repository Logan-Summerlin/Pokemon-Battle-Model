"""Tests for the Metamon replay parser.

Verifies:
- Filename metadata extraction
- JSON parsing into ParsedBattle/ParsedTurnState
- Battle validity checks
- Edge cases in data format
"""

import json
from pathlib import Path

import pytest

from src.data.replay_parser import (
    ParsedBattle,
    ParsedMove,
    ParsedPokemon,
    ParsedTurnState,
    load_battle_from_json,
    parse_filename_metadata,
)


# ── Fixtures ──────────────────────────────────────────────────────────────


def make_pokemon_dict(
    name: str = "Pikachu",
    hp_pct: float = 1.0,
    types: str = "Electric",
    item: str = "Light Ball",
    ability: str = "Static",
    lvl: int = 100,
    status: str = "",
    moves: list | None = None,
    base_atk: int = 55,
    base_spa: int = 50,
    base_def: int = 40,
    base_spd: int = 50,
    base_spe: int = 90,
    base_hp: int = 35,
    tera_type: str = "",
) -> dict:
    """Create a mock Pokemon dict in Metamon format."""
    if moves is None:
        moves = [
            {"name": "Thunderbolt", "move_type": "Electric", "category": "Special",
             "base_power": 90, "accuracy": 100, "priority": 0, "current_pp": 24, "max_pp": 24},
            {"name": "Volt Switch", "move_type": "Electric", "category": "Special",
             "base_power": 70, "accuracy": 100, "priority": 0, "current_pp": 32, "max_pp": 32},
        ]
    return {
        "name": name,
        "hp_pct": hp_pct,
        "types": types,
        "item": item,
        "ability": ability,
        "lvl": lvl,
        "status": status,
        "effect": "",
        "moves": moves,
        "atk_boost": 0,
        "spa_boost": 0,
        "def_boost": 0,
        "spd_boost": 0,
        "spe_boost": 0,
        "accuracy_boost": 0,
        "evasion_boost": 0,
        "base_atk": base_atk,
        "base_spa": base_spa,
        "base_def": base_def,
        "base_spd": base_spd,
        "base_spe": base_spe,
        "base_hp": base_hp,
        "tera_type": tera_type,
        "base_species": name,
    }


def make_state_dict(
    player_active: dict | None = None,
    opponent_active: dict | None = None,
    available_switches: list | None = None,
    weather: str = "",
    battle_field: str = "",
    battle_won: bool = False,
    battle_lost: bool = False,
    forced_switch: bool = False,
    can_tera: bool = True,
    opponents_remaining: int = 6,
    player_prev_move: dict | None = None,
    opponent_prev_move: dict | None = None,
    opponent_teampreview: list | None = None,
) -> dict:
    return {
        "format": "gen9ou",
        "player_active_pokemon": player_active or make_pokemon_dict("Pikachu"),
        "opponent_active_pokemon": opponent_active or make_pokemon_dict("Charizard"),
        "available_switches": available_switches or [
            make_pokemon_dict("Blastoise"),
            make_pokemon_dict("Venusaur"),
        ],
        "player_prev_move": player_prev_move,
        "opponent_prev_move": opponent_prev_move,
        "opponents_remaining": opponents_remaining,
        "player_conditions": "",
        "opponent_conditions": "",
        "weather": weather,
        "battle_field": battle_field,
        "forced_switch": forced_switch,
        "can_tera": can_tera,
        "battle_won": battle_won,
        "battle_lost": battle_lost,
        "opponent_teampreview": opponent_teampreview or [],
    }


def make_battle_json(
    num_turns: int = 5,
    actions: list | None = None,
    result_won: bool = True,
) -> dict:
    """Create a mock battle JSON dict."""
    states = []
    for t in range(num_turns):
        is_last = t == num_turns - 1
        state = make_state_dict(
            battle_won=result_won and is_last,
            battle_lost=(not result_won) and is_last,
        )
        states.append(state)

    if actions is None:
        actions = [f"move{i % 4}" for i in range(num_turns)]

    return {"states": states, "actions": actions}


# ── Tests: Filename parsing ───────────────────────────────────────────────


class TestFilenameParser:
    def test_standard_filename(self) -> None:
        filename = "battle-gen9ou-12345_1800_PlayerA_vs_PlayerB_15-03-2025_WIN.json.lz4"
        meta = parse_filename_metadata(filename)
        assert meta["battle_id"] == "battle-gen9ou-12345"
        assert meta["elo"] == "1800"
        assert meta["player"] == "PlayerA"
        assert meta["opponent"] == "PlayerB"
        assert meta["date"] == "15-03-2025"
        assert meta["result"] == "WIN"

    def test_loss_result(self) -> None:
        filename = "battle-123_1600_Alice_vs_Bob_01-01-2024_LOSS.json.lz4"
        meta = parse_filename_metadata(filename)
        assert meta["result"] == "LOSS"

    def test_underscore_in_names(self) -> None:
        filename = "battle-456_1700_Player_One_vs_Player_Two_20-06-2025_WIN.json"
        meta = parse_filename_metadata(filename)
        assert meta["player"] == "Player_One"
        assert meta["opponent"] == "Player_Two"

    def test_short_filename(self) -> None:
        filename = "battle123.json"
        meta = parse_filename_metadata(filename)
        assert meta["battle_id"] == "battle123"

    def test_json_extension(self) -> None:
        filename = "battle_1500_A_vs_B_01-01-2024_WIN.json"
        meta = parse_filename_metadata(filename)
        assert meta["result"] == "WIN"


# ── Tests: Battle parsing ─────────────────────────────────────────────────


class TestBattleParsing:
    def test_basic_battle_load(self) -> None:
        data = make_battle_json(num_turns=5)
        battle = load_battle_from_json(data, "test_battle.json")
        assert battle.num_turns == 5
        assert len(battle.actions) == 5
        assert battle.format == "gen9ou"

    def test_battle_validity(self) -> None:
        data = make_battle_json(num_turns=5)
        battle = load_battle_from_json(data, "test.json")
        assert battle.is_valid()

    def test_empty_battle_invalid(self) -> None:
        battle = ParsedBattle()
        assert not battle.is_valid()

    def test_single_turn_invalid(self) -> None:
        data = make_battle_json(num_turns=1)
        battle = load_battle_from_json(data)
        assert not battle.is_valid()

    def test_win_detection(self) -> None:
        data = make_battle_json(num_turns=3, result_won=True)
        battle = load_battle_from_json(data)
        assert battle.won

    def test_loss_detection(self) -> None:
        data = make_battle_json(num_turns=3, result_won=False)
        battle = load_battle_from_json(data)
        assert not battle.won

    def test_state_parsing(self) -> None:
        data = make_battle_json(num_turns=3)
        battle = load_battle_from_json(data)
        turn = battle.turns[0]

        assert turn.player_active is not None
        assert turn.player_active.name == "Pikachu"
        assert turn.opponent_active is not None
        assert turn.opponent_active.name == "Charizard"
        assert len(turn.available_switches) == 2

    def test_pokemon_stats(self) -> None:
        data = make_battle_json(num_turns=2)
        battle = load_battle_from_json(data)
        poke = battle.turns[0].player_active
        assert poke is not None
        assert poke.base_spe == 90
        assert poke.ability == "Static"
        assert poke.item == "Light Ball"

    def test_move_parsing(self) -> None:
        data = make_battle_json(num_turns=2)
        battle = load_battle_from_json(data)
        poke = battle.turns[0].player_active
        assert poke is not None
        assert len(poke.moves) == 2
        assert poke.moves[0].name == "Thunderbolt"
        assert poke.moves[0].base_power == 90

    def test_weather_parsing(self) -> None:
        state = make_state_dict(weather="RainDance")
        data = {"states": [state, make_state_dict()], "actions": ["move0", "move1"]}
        battle = load_battle_from_json(data)
        assert battle.turns[0].weather == "RainDance"

    def test_forced_switch(self) -> None:
        state = make_state_dict(forced_switch=True)
        data = {"states": [state, make_state_dict()], "actions": ["switch1", "move0"]}
        battle = load_battle_from_json(data)
        assert battle.turns[0].forced_switch

    def test_tera_availability(self) -> None:
        state = make_state_dict(can_tera=True)
        data = {"states": [state, make_state_dict()], "actions": ["move0", "move1"]}
        battle = load_battle_from_json(data)
        assert battle.turns[0].can_tera

    def test_opponent_teampreview(self) -> None:
        preview = [
            make_pokemon_dict("Garchomp"),
            make_pokemon_dict("Heatran"),
            make_pokemon_dict("Tapu Koko"),
        ]
        state = make_state_dict(opponent_teampreview=preview)
        data = {"states": [state, make_state_dict()], "actions": ["move0", "move1"]}
        battle = load_battle_from_json(data)
        assert len(battle.turns[0].opponent_teampreview) == 3
        assert battle.turns[0].opponent_teampreview[0].name == "Garchomp"

    def test_hp_fraction(self) -> None:
        active = make_pokemon_dict("Pikachu", hp_pct=0.5)
        state = make_state_dict(player_active=active)
        data = {"states": [state, make_state_dict()], "actions": ["move0", "move1"]}
        battle = load_battle_from_json(data)
        assert battle.turns[0].player_active.hp_pct == 0.5

    def test_status_condition(self) -> None:
        active = make_pokemon_dict("Pikachu", status="par")
        state = make_state_dict(player_active=active)
        data = {"states": [state, make_state_dict()], "actions": ["move0", "move1"]}
        battle = load_battle_from_json(data)
        assert battle.turns[0].player_active.status == "par"

    def test_actions_preserved(self) -> None:
        data = make_battle_json(num_turns=3, actions=["move0", "switch2", "move1"])
        battle = load_battle_from_json(data)
        assert battle.actions == ["move0", "switch2", "move1"]


# ── Tests: File I/O ───────────────────────────────────────────────────────


class TestFileIO:
    def test_load_json_file(self, tmp_path: Path) -> None:
        data = make_battle_json(num_turns=3)
        filepath = tmp_path / "test_battle.json"
        with open(filepath, "w") as f:
            json.dump(data, f)

        from src.data.replay_parser import load_battle_from_file
        battle = load_battle_from_file(filepath)
        assert battle.is_valid()
        assert battle.num_turns == 3

    @pytest.mark.skipif(
        not __import__("importlib").util.find_spec("lz4"),
        reason="lz4 not installed",
    )
    def test_load_lz4_file(self, tmp_path: Path) -> None:
        import lz4.frame

        data = make_battle_json(num_turns=4)
        filepath = tmp_path / "test_battle.json.lz4"
        with lz4.frame.open(str(filepath), "wb") as f:
            f.write(json.dumps(data).encode("utf-8"))

        from src.data.replay_parser import load_battle_from_file
        battle = load_battle_from_file(filepath)
        assert battle.is_valid()
        assert battle.num_turns == 4

    def test_load_from_directory(self, tmp_path: Path) -> None:
        for i in range(5):
            data = make_battle_json(num_turns=3)
            filepath = tmp_path / f"battle_{i}.json"
            with open(filepath, "w") as f:
                json.dump(data, f)

        from src.data.replay_parser import load_battles_from_directory
        battles = load_battles_from_directory(tmp_path)
        assert len(battles) == 5

    def test_max_battles_limit(self, tmp_path: Path) -> None:
        for i in range(10):
            data = make_battle_json(num_turns=3)
            filepath = tmp_path / f"battle_{i}.json"
            with open(filepath, "w") as f:
                json.dump(data, f)

        from src.data.replay_parser import load_battles_from_directory
        battles = load_battles_from_directory(tmp_path, max_battles=3)
        assert len(battles) == 3


# ── Tests: Edge cases ─────────────────────────────────────────────────────


class TestEdgeCases:
    def test_missing_fields(self) -> None:
        """Battle with minimal fields should still parse."""
        data = {
            "states": [
                {"format": "gen9ou"},
                {"format": "gen9ou"},
            ],
            "actions": ["move0", "move1"],
        }
        battle = load_battle_from_json(data)
        assert battle.num_turns == 2

    def test_empty_states(self) -> None:
        data = {"states": [], "actions": []}
        battle = load_battle_from_json(data)
        assert not battle.is_valid()

    def test_none_pokemon(self) -> None:
        state = make_state_dict()
        state["player_active_pokemon"] = None
        data = {"states": [state, make_state_dict()], "actions": ["move0", "move1"]}
        battle = load_battle_from_json(data)
        assert battle.turns[0].player_active is None

    def test_empty_moves_list(self) -> None:
        poke = make_pokemon_dict(moves=[])
        state = make_state_dict(player_active=poke)
        data = {"states": [state, make_state_dict()], "actions": ["move0", "move1"]}
        battle = load_battle_from_json(data)
        assert len(battle.turns[0].player_active.moves) == 0
