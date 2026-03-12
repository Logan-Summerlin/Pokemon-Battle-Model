"""Pokemon base stats crosswalk.

Loads base stats from the CSV file and provides a lookup by species name
using the same normalized naming convention as the species vocabulary
(lowercase, no spaces/hyphens/punctuation, form suffixes appended).

Base stats are PUBLIC knowledge in Pokemon — knowing the opponent's species
(which is revealed on switch-in) is sufficient to know their base stats.
This does NOT violate the Hidden Information Doctrine.
"""

from __future__ import annotations

import csv
import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Path to the CSV relative to the project root
_DEFAULT_CSV_PATH = Path(__file__).resolve().parents[2] / "data" / "raw" / "pokedex" / "pokemon_base_stats.csv"

# Base stat keys in the order used by the observation/tensorizer pipeline
STAT_KEYS = ("hp", "atk", "def", "spa", "spd", "spe")

# Forms that map to the base species name (no suffix appended)
_DEFAULT_FORMS = frozenset({
    "",
    "normal forme",
    "ordinary form",
    "incarnate forme",
    "standard mode",
    "altered forme",
    "land forme",
    "aria forme",
    "midday form",
    "red-striped form",
    "plant cloak",
    "zero form",
    "normal form",
    "teal mask",
    "male",
    "keldeo ordinary",
    # Default forms for species where the base form has a specific name
    "ice face",           # Eiscue
    "full belly mode",    # Morpeko
    "hoopa confined",     # Hoopa
    "baile style",        # Oricorio
    "amped form",         # Toxtricity
    "hero of many battles",  # Zacian, Zamazenta
    "curly form",         # Tatsugiri
    "green plumage",      # Squawkabilly
    "family of three",    # Maushold
    "two-segment form",   # Dudunsparce
    "core form",          # Minior
})

# Manual overrides for forms with non-obvious normalization.
# Maps (lowercase_name, lowercase_form) -> showdown_id
_FORM_OVERRIDES: dict[tuple[str, str], str] = {
    # Regional breeds
    ("tauros", "combat breed"): "taurospaldeacombat",
    ("tauros", "blaze breed"): "taurospaldeablaze",
    ("tauros", "aqua breed"): "taurospaldeaaqua",
    # Gendered species with distinct forms
    ("basculegion", "female"): "basculegionf",
    ("indeedee", "female"): "indeedeef",
    ("nidoran", "female"): "nidoranf",
    ("nidoran", "male"): "nidoranm",
    # Oinkologne has no gender suffix in vocab — male is base
    ("oinkologne", "female"): "oinkolognef",
    # Mega evolutions with X/Y
    ("charizard", "mega charizard x"): "charizardmegax",
    ("charizard", "mega charizard y"): "charizardmegay",
    ("mewtwo", "mega mewtwo x"): "mewtwomegax",
    ("mewtwo", "mega mewtwo y"): "mewtwomegay",
    # Calyrex riders
    ("calyrex", "ice rider"): "calyrexicerider",
    ("calyrex", "shadow rider"): "calyrexshadowrider",
    # Ursaluna Bloodmoon
    ("ursaluna", "bloodmoon"): "ursalunabloodmoon",
    # Eiscue
    ("eiscue", "noice face"): "eiscuenoice",
    # Cramorant
    ("cramorant", "gulping form"): "cramorantgulping",
    ("cramorant", "gorging form"): "cramorantgorging",
    # Morpeko
    ("morpeko", "hangry mode"): "morpekohangry",
    # Sinistcha
    ("sinistcha", "masterpiece"): "sinistchamasterpiece",
    # Polteageist
    ("polteageist", "antique"): "polteageistantique",
    # Minior
    ("minior", "meteor form"): "miniormeteor",
    # Pikachu special forms
    ("pikachu", "world cap"): "pikachuworld",
    ("pikachu", "partner cap"): "pikachupartner",
    # Squawkabilly
    ("squawkabilly", "blue plumage"): "squawkabillyblue",
    # Maushold
    ("maushold", "family of four"): "mausholdfour",
    # Lycanroc
    ("lycanroc", "midnight form"): "lycanrocmidnight",
    ("lycanroc", "dusk form"): "lycanrocdusk",
    # Toxtricity
    ("toxtricity", "low key form"): "toxtricitylowkey",
    # Basculin
    ("basculin", "blue-striped form"): "basculinbluestriped",
    ("basculin", "white-striped form"): "basculinwhitestriped",
    # Oricorio
    ("oricorio", "pom-pom style"): "oricoriopompom",
    ("oricorio", "sensu style"): "oricoriosensu",
    ("oricorio", "pa'u style"): "oricoriopau",
    # Zarude
    ("zarude", "dada"): "zarudedada",
    # Gimmighoul
    ("gimmighoul", "roaming form"): "gimmighoulroaming",
    # Meloetta
    ("meloetta", "pirouette forme"): "meloettapirouette",
    # Hoopa
    ("hoopa", "hoopa unbound"): "hoopaunbound",
    ("hoopa", "hoopa confined"): "hoopa",
    # Deoxys special formes
    ("deoxys", "attack forme"): "deoxysattack",
    ("deoxys", "defense forme"): "deoxysdefense",
    ("deoxys", "speed forme"): "deoxysspeed",
    # Giratina
    ("giratina", "origin forme"): "giratinaorigin",
    # Shaymin
    ("shaymin", "sky forme"): "shayminsky",
    # Darmanitan
    ("darmanitan", "zen mode"): "darmanitanzen",
    # Terapagos
    ("terapagos", "terastal form"): "terapagosterastal",
    ("terapagos", "stellar form"): "terapagosstellar",
}


def _normalize_species_name(name: str, form: str) -> str:
    """Convert CSV (Name, Form) pair to the showdown-style species ID.

    Examples:
        ("Bulbasaur", " ") -> "bulbasaur"
        ("Landorus", "Therian Forme") -> "landorustherian"
        ("Iron Valiant", " ") -> "ironvaliant"
        ("Ogerpon", "Wellspring Mask") -> "ogerponwellspring"
        ("Slowking", "Galarian Slowking") -> "slowkinggalar"
    """
    name_lower = name.strip().lower()
    form_lower = form.strip().lower()

    # Check manual overrides first
    override = _FORM_OVERRIDES.get((name_lower, form_lower))
    if override is not None:
        return override

    # Base name: remove non-alphanumeric
    base = re.sub(r"[^a-z0-9]", "", name_lower)

    # Default forms → base name only
    if form_lower in _DEFAULT_FORMS:
        return base

    # Regional forms: "Alolan X", "Galarian X", "Hisuian X", "Paldean X"
    if "alolan" in form_lower:
        return base + "alola"
    if "galarian" in form_lower:
        return base + "galar"
    if "hisuian" in form_lower:
        return base + "hisui"
    if "paldean" in form_lower:
        return base + "paldea"

    # Ogerpon masks
    if "wellspring mask" in form_lower:
        return base + "wellspring"
    if "hearthflame mask" in form_lower:
        return base + "hearthflame"
    if "cornerstone mask" in form_lower:
        return base + "cornerstone"

    # Rotom forms: "Heat Rotom", "Wash Rotom", etc.
    for rotom_form in ("heat", "wash", "frost", "fan", "mow"):
        if rotom_form in form_lower and "rotom" in form_lower:
            return base + rotom_form

    # Kyurem forms: "Black Kyurem", "White Kyurem"
    if "black" in form_lower and base == "kyurem":
        return base + "black"
    if "white" in form_lower and base == "kyurem":
        return base + "white"

    # Mega evolutions (generic)
    if form_lower.startswith("mega"):
        return base + "mega"

    # Generic: extract form keyword by removing "Forme"/"Form"/"Mode" suffixes
    form_suffix = re.sub(r"\s*(forme?|mode|style|cloak|breed)\s*", "", form_lower).strip()
    form_suffix = re.sub(r"[^a-z0-9]", "", form_suffix)

    return base + form_suffix


class BaseStatsCrosswalk:
    """Lookup table mapping normalized species names to base stats.

    Usage:
        crosswalk = BaseStatsCrosswalk.load()
        stats = crosswalk.get("landorustherian")
        # stats = {"hp": 89, "atk": 145, "def": 90, "spa": 105, "spd": 80, "spe": 91}
    """

    def __init__(self, stats_by_species: dict[str, dict[str, int]]) -> None:
        self._stats = stats_by_species

    def get(self, species: str) -> dict[str, int]:
        """Look up base stats for a species. Returns empty dict if not found."""
        if not species:
            return {}
        # Normalize the query: lowercase, remove non-alphanumeric
        key = re.sub(r"[^a-z0-9]", "", species.lower())
        return dict(self._stats.get(key, {}))

    def __contains__(self, species: str) -> bool:
        key = re.sub(r"[^a-z0-9]", "", species.lower())
        return key in self._stats

    def __len__(self) -> int:
        return len(self._stats)

    @classmethod
    def load(cls, csv_path: str | Path | None = None) -> BaseStatsCrosswalk:
        """Load base stats from the CSV file.

        Args:
            csv_path: Path to pokemon_base_stats.csv. Uses default if None.

        Returns:
            BaseStatsCrosswalk instance.
        """
        path = Path(csv_path) if csv_path else _DEFAULT_CSV_PATH
        if not path.exists():
            logger.warning(f"Base stats CSV not found at {path}, returning empty crosswalk")
            return cls({})

        stats_by_species: dict[str, dict[str, int]] = {}
        duplicates = 0

        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                name = row.get("Name", "").strip().strip('"')
                form = row.get("Form", "").strip().strip('"')
                species_id = _normalize_species_name(name, form)

                stats = {
                    "hp": int(row.get("HP", 0)),
                    "atk": int(row.get("Attack", 0)),
                    "def": int(row.get("Defense", 0)),
                    "spa": int(row.get("Sp. Atk", 0)),
                    "spd": int(row.get("Sp. Def", 0)),
                    "spe": int(row.get("Speed", 0)),
                }

                if species_id in stats_by_species:
                    duplicates += 1
                stats_by_species[species_id] = stats

        # Add aliases for forms not in CSV that share stats with existing entries
        _STAT_ALIASES: dict[str, str] = {
            # Tera-activated Ogerpon forms share stats with base mask forms
            "ogerponwellspringtera": "ogerponwellspring",
            "ogerponhearthflametera": "ogerponhearthflame",
            "ogerponcornerstonetera": "ogerponcornerstone",
            "ogerpontealtera": "ogerpon",
            # Mimikyu busted shares stats with base
            "mimikyubusted": "mimikyu",
            # Cramorant gulping/gorging share stats with base
            "cramorantgulping": "cramorant",
            "cramorantgorging": "cramorant",
            # Zarude Dada shares stats with base Zarude
            "zarudedada": "zarude",
            # Maushold Family of Four shares stats with Family of Three
            "mausholdfour": "maushold",
            # Arceus type forms all share the same base stats
            **{f"arceus{t}": "arceus" for t in (
                "bug", "dark", "dragon", "electric", "fairy", "fighting",
                "fire", "flying", "ghost", "grass", "ground", "ice",
                "poison", "psychic", "rock", "steel", "water",
            )},
            # Zacian/Zamazenta crowned forms (different stats but alias base for vocab)
            "zaciancrownedsword": "zacian",
            "zamazentacrownedshield": "zamazenta",
            # Pikachu costume forms share base Pikachu stats
            "pikachuworld": "pikachu",
            "pikachupartner": "pikachu",
            # Polteageist Antique shares stats with base
            "polteageistantique": "polteageist",
            # Sinistcha Masterpiece shares stats with base
            "sinistchamasterpiece": "sinistcha",
            # Squawkabilly color variants share stats
            "squawkabillyblue": "squawkabilly",
            "squawkabillyyellow": "squawkabilly",
            "squawkabillywhite": "squawkabilly",
            # Tatsugiri form variants share stats
            "tatsugiridroopy": "tatsugiri",
            "tatsugiristretchy": "tatsugiri",
            # Dudunsparce Three-Segment shares stats
            "dudunsparcethreesegment": "dudunsparce",
            # Minior Meteor shares stats
            "miniormeteor": "minior",
        }
        for alias, source in _STAT_ALIASES.items():
            if alias not in stats_by_species and source in stats_by_species:
                stats_by_species[alias] = stats_by_species[source]

        logger.info(
            f"Loaded base stats for {len(stats_by_species)} species from {path}"
            f" ({duplicates} form duplicates resolved by last-wins)"
        )
        return cls(stats_by_species)


# Module-level singleton, lazily loaded
_crosswalk: BaseStatsCrosswalk | None = None


def get_base_stats_crosswalk() -> BaseStatsCrosswalk:
    """Get the module-level base stats crosswalk (lazy singleton)."""
    global _crosswalk
    if _crosswalk is None:
        _crosswalk = BaseStatsCrosswalk.load()
    return _crosswalk


def lookup_base_stats(species: str) -> dict[str, int]:
    """Convenience function: look up base stats for a species name.

    Args:
        species: Species name in any format (will be normalized).

    Returns:
        Dict with keys hp, atk, def, spa, spd, spe. Empty dict if unknown.
    """
    return get_base_stats_crosswalk().get(species)
