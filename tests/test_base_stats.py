"""Tests for the base stats crosswalk module.

Covers:
- CSV loading and species name normalization
- Crosswalk lookup for standard and form-variant species
- Coverage of all species in the vocabulary
- Edge cases (empty species, unknown species)
"""

import json
from pathlib import Path

import pytest

from src.data.base_stats import (
    BaseStatsCrosswalk,
    _normalize_species_name,
    lookup_base_stats,
)


# ── Tests: Name normalization ───────────────────────────────────────────


class TestNormalization:
    def test_standard_species(self) -> None:
        assert _normalize_species_name("Bulbasaur", " ") == "bulbasaur"
        assert _normalize_species_name("Charizard", " ") == "charizard"
        assert _normalize_species_name("Pikachu", " ") == "pikachu"

    def test_multi_word_species(self) -> None:
        assert _normalize_species_name("Iron Valiant", " ") == "ironvaliant"
        assert _normalize_species_name("Great Tusk", " ") == "greattusk"
        assert _normalize_species_name("Raging Bolt", " ") == "ragingbolt"

    def test_regional_forms(self) -> None:
        assert _normalize_species_name("Raichu", "Alolan Raichu") == "raichualola"
        assert _normalize_species_name("Slowking", "Galarian Slowking") == "slowkinggalar"
        assert _normalize_species_name("Samurott", "Hisuian Samurott") == "samurotthisui"

    def test_therian_forms(self) -> None:
        assert _normalize_species_name("Landorus", "Therian Forme") == "landorustherian"
        assert _normalize_species_name("Tornadus", "Therian Forme") == "tornadustherian"

    def test_incarnate_is_default(self) -> None:
        assert _normalize_species_name("Landorus", "Incarnate Forme") == "landorus"

    def test_ogerpon_masks(self) -> None:
        assert _normalize_species_name("Ogerpon", "Teal Mask") == "ogerpon"
        assert _normalize_species_name("Ogerpon", "Wellspring Mask") == "ogerponwellspring"
        assert _normalize_species_name("Ogerpon", "Hearthflame Mask") == "ogerponhearthflame"
        assert _normalize_species_name("Ogerpon", "Cornerstone Mask") == "ogerponcornerstone"

    def test_rotom_forms(self) -> None:
        assert _normalize_species_name("Rotom", " ") == "rotom"
        assert _normalize_species_name("Rotom", "Wash Rotom") == "rotomwash"
        assert _normalize_species_name("Rotom", "Heat Rotom") == "rotomheat"

    def test_deoxys_forms(self) -> None:
        assert _normalize_species_name("Deoxys", "Normal Forme") == "deoxys"
        assert _normalize_species_name("Deoxys", "Speed Forme") == "deoxysspeed"
        assert _normalize_species_name("Deoxys", "Defense Forme") == "deoxysdefense"

    def test_special_characters(self) -> None:
        assert _normalize_species_name("Porygon-Z", " ") == "porygonz"
        assert _normalize_species_name("Mr. Mime", " ") == "mrmime"
        assert _normalize_species_name("Farfetch'd", " ") == "farfetchd"

    def test_paldean_tauros(self) -> None:
        assert _normalize_species_name("Tauros", "Aqua Breed") == "taurospaldeaaqua"
        assert _normalize_species_name("Tauros", "Blaze Breed") == "taurospaldeablaze"


# ── Tests: Crosswalk loading ────────────────────────────────────────────


class TestCrosswalkLoading:
    def test_load_from_default_csv(self) -> None:
        cw = BaseStatsCrosswalk.load()
        assert len(cw) > 1000  # Should have 1000+ species

    def test_lookup_standard_species(self) -> None:
        cw = BaseStatsCrosswalk.load()
        stats = cw.get("garchomp")
        assert stats["hp"] == 108
        assert stats["atk"] == 130
        assert stats["spe"] == 102

    def test_lookup_form_variant(self) -> None:
        cw = BaseStatsCrosswalk.load()
        stats = cw.get("landorustherian")
        assert stats["atk"] == 145
        assert stats["spe"] == 91

    def test_lookup_paradox_pokemon(self) -> None:
        cw = BaseStatsCrosswalk.load()
        stats = cw.get("ironvaliant")
        assert stats["atk"] == 130
        assert stats["spe"] == 116

    def test_lookup_returns_copy(self) -> None:
        """Modifying returned dict shouldn't affect internal state."""
        cw = BaseStatsCrosswalk.load()
        stats = cw.get("pikachu")
        stats["hp"] = 999
        assert cw.get("pikachu")["hp"] == 35

    def test_unknown_species_returns_empty(self) -> None:
        cw = BaseStatsCrosswalk.load()
        assert cw.get("nonexistentmon") == {}
        assert cw.get("") == {}

    def test_contains(self) -> None:
        cw = BaseStatsCrosswalk.load()
        assert "pikachu" in cw
        assert "nonexistent" not in cw

    def test_alias_coverage(self) -> None:
        """Aliases for forms not in CSV should be populated."""
        cw = BaseStatsCrosswalk.load()
        # Tera Ogerpon forms share stats with base mask forms
        assert cw.get("ogerponwellspringtera") == cw.get("ogerponwellspring")
        # Mimikyu busted shares stats with base
        assert cw.get("mimikyubusted") == cw.get("mimikyu")
        # Arceus type forms share base stats
        assert cw.get("arceusbug") == cw.get("arceus")


# ── Tests: Vocab coverage ──────────────────────────────────────────────


class TestVocabCoverage:
    def test_all_vocab_species_have_base_stats(self) -> None:
        """Every species in the frozen vocabulary should have base stats."""
        vocab_path = Path("data/processed/vocabs/species.json")
        if not vocab_path.exists():
            pytest.skip("Species vocabulary not found")

        with open(vocab_path) as f:
            vocab = json.load(f)

        cw = BaseStatsCrosswalk.load()
        missing = []
        for species in vocab["tokens"]:
            if not species or species == "unknown":
                continue
            if not cw.get(species):
                missing.append(species)

        assert missing == [], f"Species missing from crosswalk: {missing}"


# ── Tests: Convenience function ─────────────────────────────────────────


class TestLookupFunction:
    def test_lookup_base_stats(self) -> None:
        stats = lookup_base_stats("charizard")
        assert stats["hp"] == 78
        assert stats["spe"] == 100

    def test_lookup_normalizes_input(self) -> None:
        stats = lookup_base_stats("Charizard")
        assert stats["hp"] == 78
