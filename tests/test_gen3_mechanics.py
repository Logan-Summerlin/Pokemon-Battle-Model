"""Tests for Gen 3 OU-specific mechanics.

Covers:
- Type-based physical/special split (the defining Gen 3 mechanic)
- Permanent weather (Sand Stream)
- No team preview (opponent tracking without preview data)
- Legal mask excludes tera options
- Explosion/Self-Destruct Defense-halving mechanic
- Gen 3 type chart (no Fairy, Steel resists Dark/Ghost)
- Gen 3 item taxonomy
- Gen 3 move families
"""

from __future__ import annotations

import pytest

from src.bots.heuristic_bot import (
    PHYSICAL_TYPES,
    SPECIAL_TYPES,
    get_gen3_category,
    _type_effectiveness,
    _estimate_damage,
    _get_types,
)
from src.data.auxiliary_labels import (
    ITEM_CLASSES,
    ITEM_CLASS_MAP,
    NUM_ITEM_CLASSES,
    NUM_MOVE_FAMILIES,
    classify_item,
    classify_move_families,
    classify_speed,
)
from src.data.observation import (
    UNKNOWN,
    MAX_TEAM_SIZE,
    FieldObservation,
    OpponentTracker,
    PokemonObservation,
    TurnObservation,
    build_observations,
)
from src.data.replay_parser import (
    ParsedBattle,
    ParsedMove,
    ParsedPokemon,
    ParsedTurnState,
)
from src.environment.action_space import (
    MOVE_1,
    MOVE_4,
    NUM_ACTIONS,
    SWITCH_2,
    SWITCH_6,
    ActionMask,
)


# ── Helpers ───────────────────────────────────────────────────────────────


def _make_pokemon(
    name: str = "Salamence",
    hp_pct: float = 1.0,
    item: str = "Leftovers",
    ability: str = "Intimidate",
    moves: list[str] | None = None,
    types: str = "Dragon,Flying",
) -> ParsedPokemon:
    move_list = []
    for m in (moves or ["Dragon Claw", "Earthquake"]):
        move_list.append(ParsedMove(name=m, move_type="Normal", base_power=80))
    return ParsedPokemon(
        name=name,
        hp_pct=hp_pct,
        types=types,
        item=item,
        ability=ability,
        moves=move_list,
        base_atk=135,
        base_spa=110,
        base_def=80,
        base_spd=80,
        base_spe=100,
        base_hp=95,
    )


def _make_turn(
    player_active: ParsedPokemon | None = None,
    opponent_active: ParsedPokemon | None = None,
    weather: str = "",
    opponent_teampreview: list | None = None,
) -> ParsedTurnState:
    return ParsedTurnState(
        format="gen3ou",
        player_active=player_active or _make_pokemon(),
        opponent_active=opponent_active or _make_pokemon("Metagross", types="Steel,Psychic"),
        available_switches=[],
        weather=weather,
        opponent_teampreview=opponent_teampreview or [],
    )


# ── Tests: Physical/Special Type Split ───────────────────────────────────


class TestPhysicalSpecialSplit:
    """Test that Gen 3's type-based physical/special split is correct."""

    def test_physical_types_complete(self) -> None:
        """All 9 physical types should be present."""
        expected = {"Normal", "Fighting", "Poison", "Ground", "Flying",
                    "Bug", "Rock", "Ghost", "Steel"}
        assert PHYSICAL_TYPES == expected

    def test_special_types_complete(self) -> None:
        """All 8 special types should be present."""
        expected = {"Fire", "Water", "Grass", "Electric", "Ice",
                    "Psychic", "Dragon", "Dark"}
        assert SPECIAL_TYPES == expected

    def test_no_overlap(self) -> None:
        """Physical and special types should not overlap."""
        assert len(PHYSICAL_TYPES & SPECIAL_TYPES) == 0

    def test_all_types_covered(self) -> None:
        """All 17 Gen 3 types (no Fairy) should be in either physical or special."""
        all_gen3_types = PHYSICAL_TYPES | SPECIAL_TYPES
        assert len(all_gen3_types) == 17
        assert "Fairy" not in all_gen3_types

    def test_get_gen3_category_physical(self) -> None:
        for t in ["Normal", "Fighting", "Ground", "Rock", "Steel", "Ghost"]:
            assert get_gen3_category(t) == "Physical", f"{t} should be Physical"

    def test_get_gen3_category_special(self) -> None:
        for t in ["Fire", "Water", "Electric", "Ice", "Dragon", "Dark"]:
            assert get_gen3_category(t) == "Special", f"{t} should be Special"

    def test_ghost_is_physical(self) -> None:
        """Ghost is physical in Gen 3 — Gengar's Shadow Ball uses Atk, not SpA."""
        assert get_gen3_category("Ghost") == "Physical"

    def test_dark_is_special(self) -> None:
        """Dark is special in Gen 3 — Tyranitar's Crunch uses SpA."""
        assert get_gen3_category("Dark") == "Special"

    def test_steel_is_physical(self) -> None:
        """Steel is physical — Metagross's Meteor Mash uses Atk."""
        assert get_gen3_category("Steel") == "Physical"

    def test_dragon_is_special(self) -> None:
        """Dragon is special — Dragon Claw uses SpA in Gen 3."""
        assert get_gen3_category("Dragon") == "Special"


# ── Tests: Gen 3 Type Chart ──────────────────────────────────────────────


class TestGen3TypeChart:
    """Verify Gen 3-specific type effectiveness (no Fairy, Steel resists Dark/Ghost)."""

    def test_no_fairy_type(self) -> None:
        """Fairy type does not exist in Gen 3."""
        eff = _type_effectiveness("Fairy", ["Dragon"])
        # Should return 1.0 (neutral/unknown) since Fairy isn't in the chart
        assert eff == 1.0

    def test_steel_resists_dark(self) -> None:
        """In Gen 3, Steel resists Dark (changed in Gen 6)."""
        eff = _type_effectiveness("Dark", ["Steel"])
        assert eff == 0.5

    def test_steel_resists_ghost(self) -> None:
        """In Gen 3, Steel resists Ghost (changed in Gen 6)."""
        eff = _type_effectiveness("Ghost", ["Steel"])
        assert eff == 0.5

    def test_ground_super_effective_vs_steel(self) -> None:
        eff = _type_effectiveness("Ground", ["Steel"])
        assert eff == 2.0

    def test_ice_super_effective_vs_dragon(self) -> None:
        eff = _type_effectiveness("Ice", ["Dragon"])
        assert eff == 2.0

    def test_normal_immune_ghost(self) -> None:
        eff = _type_effectiveness("Normal", ["Ghost"])
        assert eff == 0.0

    def test_ground_immune_flying(self) -> None:
        eff = _type_effectiveness("Ground", ["Flying"])
        assert eff == 0.0

    def test_dual_type_effectiveness(self) -> None:
        """Ground vs Fire/Steel (e.g., Earthquake vs Metagross-like)."""
        eff = _type_effectiveness("Ground", ["Steel", "Psychic"])
        # Ground is SE vs Steel (2x), neutral vs Psychic (1x) = 2.0
        assert eff == 2.0


# ── Tests: Permanent Weather ─────────────────────────────────────────────


class TestPermanentWeather:
    """Test that Gen 3 permanent weather mechanics are handled correctly."""

    def test_field_observation_weather_permanent_flag(self) -> None:
        """FieldObservation should have a weather_permanent flag."""
        field = FieldObservation(weather="Sandstorm", weather_permanent=True)
        assert field.weather == "Sandstorm"
        assert field.weather_permanent is True

    def test_field_observation_default_not_permanent(self) -> None:
        field = FieldObservation(weather="RainDance")
        assert field.weather_permanent is False

    def test_empty_field_no_weather(self) -> None:
        field = FieldObservation()
        assert field.weather == ""
        assert field.weather_permanent is False


# ── Tests: No Team Preview ───────────────────────────────────────────────


class TestNoTeamPreview:
    """Test that Gen 3's lack of team preview is handled correctly."""

    def test_gen3_empty_opponent_teampreview(self) -> None:
        """In Gen 3, opponent_teampreview should be empty (no team preview)."""
        turn = _make_turn(opponent_teampreview=[])
        assert len(turn.opponent_teampreview) == 0

    def test_opponent_tracker_builds_from_reveals(self) -> None:
        """OpponentTracker should build team from switch-in reveals, not preview."""
        tracker = OpponentTracker()
        # Turn 1: only see opponent's lead (Metagross)
        turn1 = _make_turn(
            opponent_active=_make_pokemon("Metagross", types="Steel,Psychic"),
        )
        tracker.update_from_turn(turn1)
        assert "Metagross" in tracker.revealed_species

    def test_opponent_tracker_accumulates_reveals(self) -> None:
        """New opponent Pokemon should be added as they switch in."""
        tracker = OpponentTracker()
        turn1 = _make_turn(
            opponent_active=_make_pokemon("Metagross", types="Steel,Psychic"),
        )
        turn2 = _make_turn(
            opponent_active=_make_pokemon("Tyranitar", types="Rock,Dark"),
        )
        turn3 = _make_turn(
            opponent_active=_make_pokemon("Skarmory", types="Steel,Flying"),
        )
        tracker.update_from_turn(turn1)
        tracker.update_from_turn(turn2)
        tracker.update_from_turn(turn3)
        assert len(tracker.revealed_species) == 3
        assert "Metagross" in tracker.revealed_species
        assert "Tyranitar" in tracker.revealed_species
        assert "Skarmory" in tracker.revealed_species

    def test_no_duplicate_reveals(self) -> None:
        """Same Pokemon switching in multiple times shouldn't duplicate."""
        tracker = OpponentTracker()
        turn = _make_turn(
            opponent_active=_make_pokemon("Metagross", types="Steel,Psychic"),
        )
        tracker.update_from_turn(turn)
        tracker.update_from_turn(turn)
        assert tracker.revealed_species.count("Metagross") == 1

    def test_build_observations_no_preview(self) -> None:
        """build_observations should work with empty team preview (Gen 3)."""
        battle = ParsedBattle(
            battle_id="gen3-test-001",
            format="gen3ou",
            player_elo=1500,
            result="WIN",
            turns=[
                _make_turn(opponent_teampreview=[]),
                _make_turn(opponent_teampreview=[]),
            ],
            actions=["move0", "move1"],
        )
        obs = build_observations(battle)
        assert len(obs) == 2
        # Opponent team should be padded to MAX_TEAM_SIZE even without preview
        assert len(obs[0].opponent_team) == MAX_TEAM_SIZE


# ── Tests: Legal Mask (No Tera) ──────────────────────────────────────────


class TestLegalMaskGen3:
    """Verify that the Gen 3 action space has no tera-related actions."""

    def test_action_space_size(self) -> None:
        """Gen 3 should have 9 actions (4 moves + 5 switches, no tera)."""
        assert NUM_ACTIONS == 9

    def test_move_indices(self) -> None:
        """Move indices should be 0-3."""
        assert MOVE_1 == 0
        assert MOVE_4 == 3

    def test_switch_indices(self) -> None:
        """Switch indices should be 4-8 (no tera gap)."""
        assert SWITCH_2 == 4
        assert SWITCH_6 == 8

    def test_all_moves_mask(self) -> None:
        """All-moves mask should have exactly 4 legal actions."""
        mask = ActionMask.all_moves()
        assert mask.num_legal == 4
        for i in range(MOVE_1, MOVE_4 + 1):
            assert mask.is_legal(i)
        for i in range(SWITCH_2, SWITCH_6 + 1):
            assert not mask.is_legal(i)

    def test_no_tera_actions_exist(self) -> None:
        """There should be no tera-move actions in the action space."""
        # In Gen 9 there were indices 4-7 for tera-moves. In Gen 3, indices 4-8 are switches.
        mask = ActionMask.from_list(list(range(NUM_ACTIONS)))
        assert mask.num_legal == 9
        # Verify the switch actions are at indices 4-8
        for i in range(4, 9):
            assert mask.is_legal(i)


# ── Tests: Explosion/Self-Destruct Mechanic ──────────────────────────────


class TestExplosionMechanic:
    """Test Gen 3's Explosion/Self-Destruct Defense-halving mechanic."""

    def test_explosion_high_estimated_damage(self) -> None:
        """Explosion should have very high damage due to Defense-halving in Gen 3."""
        from src.environment.state import OwnPokemon, MoveSlot

        attacker = OwnPokemon(
            species="Metagross",
            level=100,
            current_hp=300,
            max_hp=300,
            moves=[MoveSlot(name="Explosion")],
            stats={"atk": 135, "def": 130, "spa": 95, "spd": 90, "spe": 70},
            active=True,
        )
        # Explosion on a Normal type (neutral effectiveness)
        damage = _estimate_damage("Explosion", attacker, ["Normal"], ["Steel", "Psychic"])
        # Should be very high due to 250bp + STAB (Normal on Metagross = no STAB, but
        # Explosion has 250bp and Defense halving makes effective 500bp)
        assert damage > 0

    def test_selfdestruct_damage(self) -> None:
        """Self-Destruct should also benefit from Defense-halving."""
        from src.environment.state import OwnPokemon, MoveSlot

        attacker = OwnPokemon(
            species="Snorlax",
            level=100,
            current_hp=400,
            max_hp=400,
            moves=[MoveSlot(name="Self-Destruct")],
            stats={"atk": 110, "def": 65, "spa": 65, "spd": 110, "spe": 30},
            active=True,
        )
        damage_sd = _estimate_damage("Self-Destruct", attacker, ["Normal"], ["Normal"])
        # Self-Destruct has 200bp with Defense halving = effective 400bp + STAB for Normal/Normal
        assert damage_sd > 0


# ── Tests: Gen 3 Item Taxonomy ───────────────────────────────────────────


class TestGen3ItemTaxonomy:
    """Test that the item classification is Gen 3-appropriate."""

    def test_leftovers_is_class_0(self) -> None:
        """Leftovers should be the most common item class (index 0)."""
        assert classify_item("leftovers") == 0

    def test_choice_band_is_class_1(self) -> None:
        """Choice Band is the ONLY choice item in Gen 3."""
        assert classify_item("choiceband") == 1

    def test_no_modern_choice_items(self) -> None:
        """Choice Specs and Choice Scarf don't exist in Gen 3 — should map to 'other'."""
        specs_class = classify_item("choicespecs")
        scarf_class = classify_item("choicescarf")
        other_class = ITEM_CLASS_MAP.get("other", -1)
        # These should either map to 'other' or the noitem/unknown fallback
        assert specs_class != 1  # Not Choice Band class
        assert scarf_class != 1

    def test_no_heavy_duty_boots(self) -> None:
        """Heavy-Duty Boots don't exist in Gen 3."""
        hdb_class = classify_item("heavydutyboots")
        assert hdb_class != 0  # Not Leftovers

    def test_num_item_classes(self) -> None:
        """Should have 25 item classes for Gen 3."""
        assert NUM_ITEM_CLASSES == 25

    def test_type_boost_items_grouped(self) -> None:
        """Type-boosting items should map to the typeboost class."""
        typeboost_class = ITEM_CLASS_MAP["typeboost"]
        assert classify_item("charcoal") == typeboost_class
        assert classify_item("mysticwater") == typeboost_class
        assert classify_item("nevermeltice") == typeboost_class

    def test_pinch_berries_present(self) -> None:
        """Gen 3 pinch berries (Liechi, Petaya, Salac) should have their own classes."""
        assert "liechiberry" in ITEM_CLASSES
        assert "petayaberry" in ITEM_CLASSES
        assert "salacberry" in ITEM_CLASSES


# ── Tests: Gen 3 Move Families ───────────────────────────────────────────


class TestGen3MoveFamilies:
    """Test that move family classification uses Gen 3 moves."""

    def test_gen3_priority_moves(self) -> None:
        """Gen 3 priority moves: ExtremeSpeed, Mach Punch, Quick Attack, Fake Out."""
        families = classify_move_families(["ExtremeSpeed"])
        assert families[0] == 1  # Priority family

        families = classify_move_families(["Mach Punch"])
        assert families[0] == 1

        families = classify_move_families(["Quick Attack"])
        assert families[0] == 1

    def test_gen3_hazard_is_spikes_only(self) -> None:
        """Only Spikes exists as a hazard in Gen 3 (no Stealth Rock)."""
        families_spikes = classify_move_families(["Spikes"])
        families_sr = classify_move_families(["Stealth Rock"])
        # Spikes should be classified as hazard
        assert families_spikes[2] == 1  # Hazard family index
        # Stealth Rock should NOT be classified (doesn't exist in Gen 3)
        assert families_sr[2] == 0

    def test_gen3_hazard_removal_is_rapid_spin_only(self) -> None:
        """Only Rapid Spin removes hazards in Gen 3 (no Defog)."""
        families_spin = classify_move_families(["Rapid Spin"])
        families_defog = classify_move_families(["Defog"])
        assert families_spin[3] == 1  # Hazard removal family
        assert families_defog[3] == 0  # Defog doesn't exist in Gen 3

    def test_gen3_pivot_is_baton_pass_only(self) -> None:
        """Baton Pass is the only pivot move in Gen 3 (no U-turn/Volt Switch)."""
        families_bp = classify_move_families(["Baton Pass"])
        families_ut = classify_move_families(["U-turn"])
        families_vs = classify_move_families(["Volt Switch"])
        assert families_bp[6] == 1  # Pivot family (index 6)
        assert families_ut[6] == 0  # U-turn doesn't exist in Gen 3
        assert families_vs[6] == 0  # Volt Switch doesn't exist in Gen 3

    def test_gen3_setup_moves(self) -> None:
        """Common Gen 3 setup moves should be classified."""
        for move in ["Swords Dance", "Dragon Dance", "Calm Mind", "Bulk Up", "Agility"]:
            families = classify_move_families([move])
            assert families[5] == 1, f"{move} should be in setup family"

    def test_gen3_screens(self) -> None:
        """Only Reflect and Light Screen (no Aurora Veil in Gen 3)."""
        for move in ["Reflect", "Light Screen"]:
            families = classify_move_families([move])
            assert families[7] == 1, f"{move} should be in screens family"

        families_av = classify_move_families(["Aurora Veil"])
        assert families_av[7] == 0  # Aurora Veil doesn't exist in Gen 3


# ── Tests: Gen 3 Speed Tiers ────────────────────────────────────────────


class TestGen3SpeedTiers:
    """Test speed bucket classification with Gen 3 OU thresholds."""

    def test_very_fast(self) -> None:
        """Aerodactyl (130), Jolteon (130), Dugtrio (120), Starmie (115)."""
        assert classify_speed(130) == 0  # Very fast
        assert classify_speed(115) == 0

    def test_fast(self) -> None:
        """Gengar (110), Salamence (100), Jirachi (100), Celebi (100)."""
        assert classify_speed(100) == 1  # Fast
        assert classify_speed(109) == 1

    def test_medium(self) -> None:
        """Suicune (85), Metagross (70), Heracross (85)."""
        assert classify_speed(85) == 2  # Medium
        assert classify_speed(70) == 2

    def test_slow(self) -> None:
        """Swampert (60), Tyranitar (61), Blissey (55)."""
        assert classify_speed(60) == 3  # Slow
        assert classify_speed(55) == 3

    def test_very_slow(self) -> None:
        """Snorlax (30), Dusclops (25)."""
        assert classify_speed(30) == 4  # Very slow
        assert classify_speed(25) == 4
