"""Tests for the Showdown protocol message parser.

Covers:
- Parsing individual message types (switch, move, damage, etc.)
- Parsing Pokemon identifiers, details, HP/status
- Parsing multi-line battle chunks
- Edge cases: malformed messages, missing fields, empty args
"""

import pytest

from src.environment.protocol import (
    BattleMessage,
    HPStatus,
    MessageType,
    PokemonDetails,
    PokemonIdent,
    parse_battle_chunk,
    parse_boost_message,
    parse_damage_message,
    parse_hp_status,
    parse_message,
    parse_move_message,
    parse_poke_message,
    parse_pokemon_details,
    parse_pokemon_ident,
    parse_side_condition_message,
    parse_switch_message,
    parse_weather_message,
)


class TestPokemonIdent:
    """Test Pokemon identifier parsing."""

    def test_basic_ident(self) -> None:
        result = parse_pokemon_ident("p1a: Pikachu")
        assert result is not None
        assert result.player == "p1"
        assert result.position == "a"
        assert result.name == "Pikachu"

    def test_p2_ident(self) -> None:
        result = parse_pokemon_ident("p2a: Charizard")
        assert result is not None
        assert result.player == "p2"
        assert result.name == "Charizard"

    def test_ident_with_spaces(self) -> None:
        result = parse_pokemon_ident("p1a: Mr. Mime")
        assert result is not None
        assert result.name == "Mr. Mime"

    def test_ident_with_forme(self) -> None:
        result = parse_pokemon_ident("p2a: Rotom-Wash")
        assert result is not None
        assert result.name == "Rotom-Wash"

    def test_invalid_ident(self) -> None:
        assert parse_pokemon_ident("invalid") is None
        assert parse_pokemon_ident("") is None

    def test_player_num(self) -> None:
        p1 = parse_pokemon_ident("p1a: Test")
        p2 = parse_pokemon_ident("p2a: Test")
        assert p1 is not None and p1.player_num == 1
        assert p2 is not None and p2.player_num == 2


class TestPokemonDetails:
    """Test Pokemon details parsing."""

    def test_basic_details(self) -> None:
        result = parse_pokemon_details("Pikachu, L50, M")
        assert result.species == "Pikachu"
        assert result.level == 50
        assert result.gender == "M"
        assert result.shiny is False

    def test_default_level(self) -> None:
        result = parse_pokemon_details("Pikachu")
        assert result.species == "Pikachu"
        assert result.level == 100

    def test_shiny(self) -> None:
        result = parse_pokemon_details("Charizard, L100, F, shiny")
        assert result.species == "Charizard"
        assert result.gender == "F"
        assert result.shiny is True

    def test_forme(self) -> None:
        result = parse_pokemon_details("Deoxys-Speed, L100, M")
        assert result.species == "Deoxys-Speed"
        assert result.base_species == "Deoxys"

    def test_genderless(self) -> None:
        result = parse_pokemon_details("Metagross, L100")
        assert result.species == "Metagross"
        assert result.gender == ""


class TestHPStatus:
    """Test HP/status parsing."""

    def test_full_hp(self) -> None:
        result = parse_hp_status("100/100")
        assert result.current_hp == 100
        assert result.max_hp == 100
        assert result.status == ""
        assert result.fraction == 1.0

    def test_partial_hp(self) -> None:
        result = parse_hp_status("50/100")
        assert result.current_hp == 50
        assert result.fraction == 0.5

    def test_hp_with_status(self) -> None:
        result = parse_hp_status("75/100 brn")
        assert result.current_hp == 75
        assert result.status == "brn"

    def test_fainted(self) -> None:
        result = parse_hp_status("0 fnt")
        assert result.current_hp == 0
        assert result.is_fainted is True

    def test_tox_status(self) -> None:
        result = parse_hp_status("200/300 tox")
        assert result.status == "tox"

    def test_hp_fraction_zero_max(self) -> None:
        result = HPStatus(current_hp=0, max_hp=0)
        assert result.fraction == 0.0


class TestParseMessage:
    """Test single message line parsing."""

    def test_turn_message(self) -> None:
        msg = parse_message("|turn|5")
        assert msg.msg_type == MessageType.TURN
        assert msg.args == ["5"]

    def test_switch_message(self) -> None:
        msg = parse_message("|switch|p1a: Pikachu|Pikachu, L50, M|100/100")
        assert msg.msg_type == MessageType.SWITCH
        assert len(msg.args) == 3

    def test_move_message(self) -> None:
        msg = parse_message("|move|p1a: Pikachu|Thunderbolt|p2a: Charizard")
        assert msg.msg_type == MessageType.MOVE
        assert len(msg.args) == 3

    def test_damage_message(self) -> None:
        msg = parse_message("|-damage|p2a: Charizard|50/100")
        assert msg.msg_type == MessageType.DAMAGE

    def test_message_with_kwargs(self) -> None:
        msg = parse_message("|-damage|p2a: Charizard|50/100|[from] Spikes")
        assert msg.msg_type == MessageType.DAMAGE
        assert "from" in msg.kwargs
        assert msg.kwargs["from"] == "Spikes"

    def test_win_message(self) -> None:
        msg = parse_message("|win|Player1")
        assert msg.msg_type == MessageType.WIN
        assert msg.args == ["Player1"]

    def test_weather_message(self) -> None:
        msg = parse_message("|-weather|RainDance")
        assert msg.msg_type == MessageType.WEATHER

    def test_boost_message(self) -> None:
        msg = parse_message("|-boost|p1a: Salamence|atk|2")
        assert msg.msg_type == MessageType.BOOST

    def test_unknown_message(self) -> None:
        msg = parse_message("|somethingweird|args")
        assert msg.msg_type == MessageType.UNKNOWN

    def test_empty_line(self) -> None:
        msg = parse_message("")
        assert msg.msg_type == MessageType.UNKNOWN

    def test_non_pipe_line(self) -> None:
        msg = parse_message("Some text without pipes")
        assert msg.msg_type == MessageType.UNKNOWN

    def test_status_message(self) -> None:
        msg = parse_message("|-status|p2a: Skarmory|par")
        assert msg.msg_type == MessageType.STATUS

    def test_faint_message(self) -> None:
        msg = parse_message("|faint|p1a: Pikachu")
        assert msg.msg_type == MessageType.FAINT

    def test_sidestart_message(self) -> None:
        msg = parse_message("|-sidestart|p1: Player1|move: Spikes")
        assert msg.msg_type == MessageType.SIDESTART

    def test_poke_message(self) -> None:
        msg = parse_message("|poke|p1|Pikachu, L50, M|item")
        assert msg.msg_type == MessageType.POKE


class TestParseBattleChunk:
    """Test parsing multi-line battle chunks."""

    def test_basic_chunk(self) -> None:
        chunk = """|turn|1
|switch|p1a: Pikachu|Pikachu, L100, M|100/100
|switch|p2a: Charizard|Charizard, L100, M|100/100"""
        messages = parse_battle_chunk(chunk)
        assert len(messages) == 3
        assert messages[0].msg_type == MessageType.TURN
        assert messages[1].msg_type == MessageType.SWITCH
        assert messages[2].msg_type == MessageType.SWITCH

    def test_chunk_with_empty_lines(self) -> None:
        chunk = """|turn|2

|move|p1a: Pikachu|Thunderbolt|p2a: Charizard

|-damage|p2a: Charizard|50/100"""
        messages = parse_battle_chunk(chunk)
        assert len(messages) == 3

    def test_empty_chunk(self) -> None:
        messages = parse_battle_chunk("")
        assert len(messages) == 0


class TestSpecializedParsers:
    """Test specialized message parsers."""

    def test_parse_switch_message(self) -> None:
        msg = parse_message("|switch|p1a: Pikachu|Pikachu, L50, M|100/100")
        data = parse_switch_message(msg)
        assert data["ident"].name == "Pikachu"
        assert data["details"].species == "Pikachu"
        assert data["details"].level == 50
        assert data["hp_status"].current_hp == 100

    def test_parse_move_message(self) -> None:
        msg = parse_message("|move|p1a: Pikachu|Thunderbolt|p2a: Charizard")
        data = parse_move_message(msg)
        assert data["source"].name == "Pikachu"
        assert data["move"] == "Thunderbolt"
        assert data["target"].name == "Charizard"

    def test_parse_damage_message(self) -> None:
        msg = parse_message("|-damage|p2a: Charizard|50/100")
        data = parse_damage_message(msg)
        assert data["target"].name == "Charizard"
        assert data["hp_status"].current_hp == 50

    def test_parse_boost_message(self) -> None:
        msg = parse_message("|-boost|p1a: Salamence|atk|2")
        data = parse_boost_message(msg)
        assert data["target"].name == "Salamence"
        assert data["stat"] == "atk"
        assert data["amount"] == 2

    def test_parse_weather_message(self) -> None:
        msg = parse_message("|-weather|RainDance")
        data = parse_weather_message(msg)
        assert data["weather"] == "RainDance"

    def test_parse_side_condition_message(self) -> None:
        msg = parse_message("|-sidestart|p1: Player1|move: Spikes")
        data = parse_side_condition_message(msg)
        assert data["player"] == "p1"
        assert data["condition"] == "move: Spikes"

    def test_parse_poke_message(self) -> None:
        msg = parse_message("|poke|p1|Pikachu, L50, M|item")
        data = parse_poke_message(msg)
        assert data["player"] == "p1"
        assert data["details"].species == "Pikachu"
        assert data["has_item"] is True

    def test_parse_poke_no_item(self) -> None:
        msg = parse_message("|poke|p2|Charizard, L100, M")
        data = parse_poke_message(msg)
        assert data["has_item"] is False
