"""Tests for the first-person battle state tracker.

Covers:
- Team preview processing
- Switch/drag state updates
- Move tracking and reveal handling
- Damage/heal HP tracking
- Status condition management
- Boost/unboost tracking
- Weather, terrain, trick room
- Side conditions (hazards, screens, tailwind)
- Item and ability reveals
- Volatile status tracking
- Win/tie detection
- First-person only: no hidden state leakage
"""

import json

import pytest

from src.environment.protocol import MessageType, parse_battle_chunk, parse_message
from src.environment.state import (
    UNKNOWN,
    BattleState,
    BattleStateTracker,
    GamePhase,
    OpponentPokemon,
    OwnPokemon,
)


def _make_tracker(player_id: str = "p1") -> BattleStateTracker:
    return BattleStateTracker(player_id=player_id)


def _process_chunk(tracker: BattleStateTracker, chunk: str) -> None:
    messages = parse_battle_chunk(chunk)
    tracker.process_messages(messages)


class TestTeamPreview:
    """Test team preview processing."""

    def test_poke_messages_build_opponent_team(self) -> None:
        tracker = _make_tracker()
        _process_chunk(tracker, """|poke|p2|Pikachu, L100, M
|poke|p2|Charizard, L100, M
|poke|p2|Blastoise, L100, F""")

        assert len(tracker.state.opponent_team) == 3
        assert tracker.state.opponent_team[0].species == "Pikachu"
        assert tracker.state.opponent_team[1].species == "Charizard"
        assert tracker.state.opponent_team[2].species == "Blastoise"

    def test_poke_messages_track_preview_species(self) -> None:
        tracker = _make_tracker()
        _process_chunk(tracker, """|poke|p1|Salamence, L100, M
|poke|p1|Metagross, L100
|poke|p2|Tyranitar, L100, M""")

        assert "Salamence" in tracker.state.preview_species
        assert "Metagross" in tracker.state.preview_species
        assert "Tyranitar" in tracker.state.opponent_preview_species

    def test_teampreview_sets_phase(self) -> None:
        tracker = _make_tracker()
        _process_chunk(tracker, "|teampreview")
        assert tracker.state.phase == GamePhase.TEAM_PREVIEW

    def test_opponent_has_unknown_fields(self) -> None:
        tracker = _make_tracker()
        _process_chunk(tracker, "|poke|p2|Salamence, L100, M")

        opp = tracker.state.opponent_team[0]
        assert opp.item == UNKNOWN
        assert opp.ability == UNKNOWN
        assert opp.revealed_moves == []


class TestSwitching:
    """Test switch message processing."""

    def test_opponent_switch_updates_active(self) -> None:
        tracker = _make_tracker()
        _process_chunk(tracker, """|poke|p2|Pikachu, L100, M
|poke|p2|Charizard, L100, M
|switch|p2a: Pikachu|Pikachu, L100, M|100/100""")

        assert tracker.state.opponent_active_index == 0
        assert tracker.state.opponent_active is not None
        assert tracker.state.opponent_active.species == "Pikachu"
        assert tracker.state.opponent_active.hp_fraction == 1.0
        assert tracker.state.opponent_active.active is True
        assert tracker.state.opponent_active.seen_in_battle is True

    def test_switch_clears_boosts(self) -> None:
        tracker = _make_tracker()
        _process_chunk(tracker, """|poke|p2|Pikachu, L100, M
|poke|p2|Charizard, L100, M
|switch|p2a: Pikachu|Pikachu, L100, M|100/100
|-boost|p2a: Pikachu|atk|2""")

        assert tracker.state.opponent_active.boosts.get("atk") == 2

        # Now switch — boosts should clear
        _process_chunk(tracker, "|switch|p2a: Charizard|Charizard, L100, M|100/100")
        # Old active (Pikachu) should have boosts cleared
        pikachu = tracker.state.opponent_team[0]
        assert pikachu.boosts == {}

    def test_switch_records_history(self) -> None:
        tracker = _make_tracker()
        _process_chunk(tracker, """|poke|p2|Pikachu, L100, M
|turn|1
|switch|p2a: Pikachu|Pikachu, L100, M|100/100""")

        assert len(tracker.state.turn_history) == 1
        assert "switch Pikachu" in tracker.state.turn_history[0].opponent_action


class TestMoveTracking:
    """Test move message processing."""

    def test_opponent_move_reveals_move(self) -> None:
        tracker = _make_tracker()
        _process_chunk(tracker, """|poke|p2|Pikachu, L100, M
|switch|p2a: Pikachu|Pikachu, L100, M|100/100
|move|p2a: Pikachu|Thunderbolt|p1a: Salamence""")

        opp = tracker.state.opponent_active
        assert "Thunderbolt" in opp.revealed_moves

    def test_move_doesnt_duplicate_reveals(self) -> None:
        tracker = _make_tracker()
        _process_chunk(tracker, """|poke|p2|Pikachu, L100, M
|switch|p2a: Pikachu|Pikachu, L100, M|100/100
|move|p2a: Pikachu|Thunderbolt|p1a: Salamence
|move|p2a: Pikachu|Thunderbolt|p1a: Salamence""")

        opp = tracker.state.opponent_active
        assert opp.revealed_moves.count("Thunderbolt") == 1


class TestDamageAndHealing:
    """Test damage and heal message processing."""

    def test_opponent_damage_updates_hp(self) -> None:
        tracker = _make_tracker()
        _process_chunk(tracker, """|poke|p2|Charizard, L100, M
|switch|p2a: Charizard|Charizard, L100, M|100/100
|-damage|p2a: Charizard|50/100""")

        assert tracker.state.opponent_active.hp_fraction == 0.5

    def test_opponent_faint(self) -> None:
        tracker = _make_tracker()
        _process_chunk(tracker, """|poke|p2|Charizard, L100, M
|switch|p2a: Charizard|Charizard, L100, M|100/100
|faint|p2a: Charizard""")

        assert tracker.state.opponent_active.fainted is True
        assert tracker.state.opponent_active.hp_fraction == 0.0

    def test_own_damage_from_request(self) -> None:
        tracker = _make_tracker()
        request = json.dumps({
            "side": {
                "id": "p1",
                "pokemon": [
                    {
                        "ident": "p1: Salamence",
                        "details": "Salamence, L100, M",
                        "condition": "200/350",
                        "active": True,
                        "stats": {"atk": 394, "def": 196},
                        "moves": ["earthquake", "dragondance"],
                        "item": "choiceband",
                        "ability": "Intimidate",
                    }
                ],
            }
        })
        tracker.update_from_request(request)

        poke = tracker.state.own_team[0]
        assert poke.current_hp == 200
        assert poke.max_hp == 350
        assert poke.item == "choiceband"
        assert poke.ability == "Intimidate"


class TestStatusConditions:
    """Test status condition tracking."""

    def test_opponent_status(self) -> None:
        tracker = _make_tracker()
        _process_chunk(tracker, """|poke|p2|Skarmory, L100
|switch|p2a: Skarmory|Skarmory, L100|100/100
|-status|p2a: Skarmory|par""")

        assert tracker.state.opponent_active.status == "par"

    def test_cure_status(self) -> None:
        tracker = _make_tracker()
        _process_chunk(tracker, """|poke|p2|Skarmory, L100
|switch|p2a: Skarmory|Skarmory, L100|100/100
|-status|p2a: Skarmory|par
|-curestatus|p2a: Skarmory|par""")

        assert tracker.state.opponent_active.status == ""


class TestBoosts:
    """Test stat boost tracking."""

    def test_boost(self) -> None:
        tracker = _make_tracker()
        _process_chunk(tracker, """|poke|p2|Salamence, L100, M
|switch|p2a: Salamence|Salamence, L100, M|100/100
|-boost|p2a: Salamence|atk|2""")

        assert tracker.state.opponent_active.boosts["atk"] == 2

    def test_unboost(self) -> None:
        tracker = _make_tracker()
        _process_chunk(tracker, """|poke|p2|Salamence, L100, M
|switch|p2a: Salamence|Salamence, L100, M|100/100
|-unboost|p2a: Salamence|def|1""")

        assert tracker.state.opponent_active.boosts["def"] == -1

    def test_boost_caps_at_6(self) -> None:
        tracker = _make_tracker()
        _process_chunk(tracker, """|poke|p2|Salamence, L100, M
|switch|p2a: Salamence|Salamence, L100, M|100/100
|-boost|p2a: Salamence|atk|3
|-boost|p2a: Salamence|atk|3
|-boost|p2a: Salamence|atk|3""")

        assert tracker.state.opponent_active.boosts["atk"] == 6

    def test_clearboost(self) -> None:
        tracker = _make_tracker()
        _process_chunk(tracker, """|poke|p2|Salamence, L100, M
|switch|p2a: Salamence|Salamence, L100, M|100/100
|-boost|p2a: Salamence|atk|2
|-clearboost|p2a: Salamence""")

        assert tracker.state.opponent_active.boosts == {}


class TestWeatherAndTerrain:
    """Test weather, terrain, and field condition tracking."""

    def test_weather_set(self) -> None:
        tracker = _make_tracker()
        _process_chunk(tracker, "|-weather|RainDance")
        assert tracker.state.field.weather == "RainDance"

    def test_weather_end(self) -> None:
        tracker = _make_tracker()
        _process_chunk(tracker, "|-weather|RainDance")
        _process_chunk(tracker, "|-weather|none")
        assert tracker.state.field.weather == ""

    def test_terrain_set(self) -> None:
        tracker = _make_tracker()
        _process_chunk(tracker, "|-fieldstart|move: Electric Terrain")
        assert tracker.state.field.terrain == "move: Electric Terrain"

    def test_terrain_end(self) -> None:
        tracker = _make_tracker()
        _process_chunk(tracker, "|-fieldstart|move: Electric Terrain")
        _process_chunk(tracker, "|-fieldend|move: Electric Terrain")
        assert tracker.state.field.terrain == ""

    def test_trick_room(self) -> None:
        tracker = _make_tracker()
        _process_chunk(tracker, "|-fieldstart|move: Trick Room")
        assert tracker.state.field.trick_room > 0

    def test_trick_room_end(self) -> None:
        tracker = _make_tracker()
        _process_chunk(tracker, "|-fieldstart|move: Trick Room")
        _process_chunk(tracker, "|-fieldend|move: Trick Room")
        assert tracker.state.field.trick_room == 0


class TestSideConditions:
    """Test side condition (hazards, screens) tracking."""

    # Note: Stealth Rock doesn't exist in Gen 3 (introduced Gen 4), but
    # the state tracker is generation-agnostic infrastructure, so we test
    # its ability to process these messages regardless.
    def test_stealth_rock(self) -> None:
        tracker = _make_tracker()
        _process_chunk(tracker, "|-sidestart|p1: Player1|move: Stealth Rock")
        assert tracker.state.own_side.stealth_rock is True

    def test_spikes_stacking(self) -> None:
        tracker = _make_tracker()
        _process_chunk(tracker, "|-sidestart|p2: Player2|move: Spikes")
        assert tracker.state.opponent_side.spikes == 1
        _process_chunk(tracker, "|-sidestart|p2: Player2|move: Spikes")
        assert tracker.state.opponent_side.spikes == 2
        _process_chunk(tracker, "|-sidestart|p2: Player2|move: Spikes")
        assert tracker.state.opponent_side.spikes == 3
        # Max 3 layers
        _process_chunk(tracker, "|-sidestart|p2: Player2|move: Spikes")
        assert tracker.state.opponent_side.spikes == 3

    # Note: Toxic Spikes doesn't exist in Gen 3 (introduced Gen 4), but
    # the state tracker is generation-agnostic infrastructure.
    def test_toxic_spikes_max_2(self) -> None:
        tracker = _make_tracker()
        _process_chunk(tracker, "|-sidestart|p1: Player1|move: Toxic Spikes")
        _process_chunk(tracker, "|-sidestart|p1: Player1|move: Toxic Spikes")
        _process_chunk(tracker, "|-sidestart|p1: Player1|move: Toxic Spikes")
        assert tracker.state.own_side.toxic_spikes == 2

    def test_reflect_screen(self) -> None:
        tracker = _make_tracker()
        _process_chunk(tracker, "|-sidestart|p1: Player1|move: Reflect")
        assert tracker.state.own_side.reflect > 0

    def test_screen_removal(self) -> None:
        tracker = _make_tracker()
        _process_chunk(tracker, "|-sidestart|p1: Player1|move: Light Screen")
        assert tracker.state.own_side.light_screen > 0
        _process_chunk(tracker, "|-sideend|p1: Player1|move: Light Screen")
        assert tracker.state.own_side.light_screen == 0

    def test_hazard_removal(self) -> None:
        tracker = _make_tracker()
        _process_chunk(tracker, "|-sidestart|p1: Player1|move: Stealth Rock")
        _process_chunk(tracker, "|-sideend|p1: Player1|move: Stealth Rock")
        assert tracker.state.own_side.stealth_rock is False

    # Note: Tailwind doesn't exist in Gen 3 (introduced Gen 4), but
    # the state tracker is generation-agnostic infrastructure.
    def test_tailwind(self) -> None:
        tracker = _make_tracker()
        _process_chunk(tracker, "|-sidestart|p1: Player1|move: Tailwind")
        assert tracker.state.own_side.tailwind > 0


class TestItemAndAbilityReveals:
    """Test item and ability reveal tracking."""

    def test_item_revealed(self) -> None:
        tracker = _make_tracker()
        _process_chunk(tracker, """|poke|p2|Metagross, L100
|switch|p2a: Metagross|Metagross, L100|100/100
|-item|p2a: Metagross|Leftovers""")

        assert tracker.state.opponent_active.item == "Leftovers"

    def test_item_consumed(self) -> None:
        tracker = _make_tracker()
        _process_chunk(tracker, """|poke|p2|Metagross, L100
|switch|p2a: Metagross|Metagross, L100|100/100
|-enditem|p2a: Metagross|Choice Band""")

        # Should show the item was consumed
        assert "Choice Band" in tracker.state.opponent_active.item

    def test_ability_revealed(self) -> None:
        tracker = _make_tracker()
        _process_chunk(tracker, """|poke|p2|Metagross, L100
|switch|p2a: Metagross|Metagross, L100|100/100
|-ability|p2a: Metagross|Clear Body""")

        assert tracker.state.opponent_active.ability == "Clear Body"


class TestVolatileStatuses:
    """Test volatile status tracking."""

    def test_volatile_start(self) -> None:
        tracker = _make_tracker()
        _process_chunk(tracker, """|poke|p2|Salamence, L100, M
|switch|p2a: Salamence|Salamence, L100, M|100/100
|-start|p2a: Salamence|confusion""")

        assert "confusion" in tracker.state.opponent_active.volatiles

    def test_volatile_end(self) -> None:
        tracker = _make_tracker()
        _process_chunk(tracker, """|poke|p2|Salamence, L100, M
|switch|p2a: Salamence|Salamence, L100, M|100/100
|-start|p2a: Salamence|confusion
|-end|p2a: Salamence|confusion""")

        assert "confusion" not in tracker.state.opponent_active.volatiles


class TestBattleEnd:
    """Test battle end detection."""

    def test_win_message(self) -> None:
        tracker = _make_tracker()
        _process_chunk(tracker, "|win|Player1")
        assert tracker.state.phase == GamePhase.FINISHED
        assert tracker.state.winner == "Player1"

    def test_tie_message(self) -> None:
        tracker = _make_tracker()
        _process_chunk(tracker, "|tie")
        assert tracker.state.phase == GamePhase.FINISHED
        assert tracker.state.winner == ""


class TestRequestProcessing:
    """Test request JSON processing."""

    def test_basic_request(self) -> None:
        tracker = _make_tracker()
        request = json.dumps({
            "side": {
                "id": "p1",
                "name": "TestPlayer",
                "pokemon": [
                    {
                        "ident": "p1: Salamence",
                        "details": "Salamence, L100, M",
                        "condition": "350/350",
                        "active": True,
                        "stats": {"atk": 394, "def": 196, "spa": 350, "spd": 196, "spe": 328},
                        "moves": [
                            {"move": "Earthquake", "id": "earthquake", "pp": 16, "maxpp": 16},
                            {"move": "Dragon Dance", "id": "dragondance", "pp": 32, "maxpp": 32},
                            {"move": "Dragon Claw", "id": "dragonclaw", "pp": 24, "maxpp": 24},
                            {"move": "Rock Slide", "id": "rockslide", "pp": 16, "maxpp": 16},
                        ],
                        "item": "choiceband",
                        "ability": "Intimidate",
                        "baseAbility": "Intimidate",
                    },
                    {
                        "ident": "p1: Metagross",
                        "details": "Metagross, L100",
                        "condition": "300/300",
                        "active": False,
                        "stats": {"atk": 405, "def": 394, "spa": 226, "spd": 216, "spe": 174},
                        "moves": [
                            {"move": "Meteor Mash", "id": "meteormash", "pp": 16, "maxpp": 16},
                        ],
                        "item": "leftovers",
                        "ability": "Clear Body",
                    },
                ],
            }
        })

        tracker.update_from_request(request)

        assert len(tracker.state.own_team) == 2
        assert tracker.state.own_team[0].species == "Salamence"
        assert tracker.state.own_team[0].active is True
        assert tracker.state.own_team[0].current_hp == 350
        assert tracker.state.own_team[0].max_hp == 350
        assert tracker.state.own_team[0].item == "choiceband"
        assert len(tracker.state.own_team[0].moves) == 4
        assert tracker.state.own_team[0].moves[0].name == "Earthquake"
        assert tracker.state.own_team[0].moves[0].pp == 16

        assert tracker.state.own_team[1].species == "Metagross"
        assert tracker.state.own_team[1].active is False

    def test_fainted_in_request(self) -> None:
        tracker = _make_tracker()
        request = json.dumps({
            "side": {
                "id": "p1",
                "pokemon": [
                    {
                        "ident": "p1: Pikachu",
                        "details": "Pikachu, L100, M",
                        "condition": "0 fnt",
                        "active": False,
                        "stats": {},
                        "moves": [],
                    }
                ],
            }
        })
        tracker.update_from_request(request)

        assert tracker.state.own_team[0].fainted is True
        assert tracker.state.own_team[0].current_hp == 0

    def test_wait_request(self) -> None:
        tracker = _make_tracker()
        request = json.dumps({"wait": True})
        tracker.update_from_request(request)
        # Should not crash or change state significantly

    def test_snapshot_is_independent(self) -> None:
        tracker = _make_tracker()
        _process_chunk(tracker, """|poke|p2|Pikachu, L100, M
|switch|p2a: Pikachu|Pikachu, L100, M|100/100""")

        snap = tracker.state.snapshot()
        # Modify original
        _process_chunk(tracker, "|-damage|p2a: Pikachu|50/100")

        # Snapshot should be unchanged
        assert snap.opponent_team[0].hp_fraction == 1.0
        assert tracker.state.opponent_team[0].hp_fraction == 0.5


class TestFirstPersonInvariant:
    """Verify the first-person information constraint.

    The most critical tests: ensure that our state tracking
    never exposes information the player shouldn't have.
    """

    def test_opponent_unrevealed_stays_unknown(self) -> None:
        tracker = _make_tracker()
        _process_chunk(tracker, """|poke|p2|Salamence, L100, M
|switch|p2a: Salamence|Salamence, L100, M|100/100""")

        opp = tracker.state.opponent_active
        assert opp.item == UNKNOWN
        assert opp.ability == UNKNOWN
        assert len(opp.revealed_moves) == 0

    def test_opponent_moves_revealed_incrementally(self) -> None:
        tracker = _make_tracker()
        _process_chunk(tracker, """|poke|p2|Salamence, L100, M
|switch|p2a: Salamence|Salamence, L100, M|100/100
|move|p2a: Salamence|Earthquake|p1a: Metagross""")

        opp = tracker.state.opponent_active
        assert opp.revealed_moves == ["Earthquake"]

        _process_chunk(tracker, "|move|p2a: Salamence|Dragon Dance|p2a: Salamence")
        assert opp.revealed_moves == ["Earthquake", "Dragon Dance"]

    def test_opponent_hp_is_fraction_only(self) -> None:
        """Opponent HP is only known as a fraction, not exact values."""
        tracker = _make_tracker()
        _process_chunk(tracker, """|poke|p2|Salamence, L100, M
|switch|p2a: Salamence|Salamence, L100, M|100/100
|-damage|p2a: Salamence|50/100""")

        opp = tracker.state.opponent_active
        assert opp.hp_fraction == 0.5
        # OpponentPokemon doesn't store exact HP values
        assert not hasattr(opp, "current_hp") or not hasattr(opp, "max_hp")

    def test_unseen_pokemon_not_marked_seen(self) -> None:
        tracker = _make_tracker()
        _process_chunk(tracker, """|poke|p2|Salamence, L100, M
|poke|p2|Metagross, L100
|switch|p2a: Salamence|Salamence, L100, M|100/100""")

        assert tracker.state.opponent_team[0].seen_in_battle is True
        assert tracker.state.opponent_team[1].seen_in_battle is False
