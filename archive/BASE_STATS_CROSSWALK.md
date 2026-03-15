# Base Stats Crosswalk Reference

This document details the name normalization logic, manual overrides, and stat aliases used by `src/data/base_stats.py` to map every Pokemon species from the CSV pokedex (`data/raw/pokedex/pokemon_base_stats.csv`) to the showdown-style species IDs used in the training vocabulary (`data/processed/vocabs/species.json`).

## Why This Exists

The CSV uses official Pokemon names with a separate Form column (e.g., `"Landorus"` + `"Therian Forme"`), while the Metamon replay dataset and our species vocabulary use lowercase, concatenated showdown IDs (e.g., `"landorustherian"`). The crosswalk bridges these two representations so that base stats can be looked up by the species ID that appears in training data.

Base stats are **public knowledge** in competitive Pokemon. Once a species is visible on the field (switch-in), any player can look up its stats. Providing these features for opponent pokemon does not violate the Hidden Information Doctrine.

## Coverage

- **CSV entries**: 1,215 rows (Gen 1-9, all forms)
- **Unique normalized species**: ~1,245 (after aliases)
- **Vocabulary species**: 539 (Gen 9 OU metagame)
- **Coverage**: 100% — every vocab species has base stats

## Normalization Pipeline

The function `_normalize_species_name(name, form)` converts a CSV `(Name, Form)` pair to a showdown ID through this priority chain:

1. **Manual override check** — exact `(name, form)` match in `_FORM_OVERRIDES`
2. **Base name extraction** — lowercase, remove non-alphanumeric (`"Iron Valiant"` -> `"ironvaliant"`)
3. **Default form check** — if the form is in `_DEFAULT_FORMS`, return just the base name
4. **Regional form rules** — detect "Alolan"/"Galarian"/"Hisuian"/"Paldean" keywords
5. **Specific form rules** — Ogerpon masks, Rotom forms, Kyurem forms, Mega evolutions
6. **Generic fallback** — strip "Forme"/"Form"/"Mode"/"Style"/"Cloak"/"Breed" suffixes, concatenate remainder

## Default Forms (27 entries)

These CSV form strings map to the base species name with no suffix appended. They represent the "primary" or "default" form of a species.

| Form String | Species | Rationale |
|---|---|---|
| `""` (empty/blank) | All standard Pokemon | No form variant |
| `"Normal Forme"` | Deoxys | Base Deoxys in Showdown |
| `"Ordinary Form"` | Keldeo | Base Keldeo |
| `"Incarnate Forme"` | Landorus, Tornadus, Thundurus, Enamorus | Base genies |
| `"Standard Mode"` | Darmanitan | Base Darmanitan |
| `"Altered Forme"` | Giratina | Base Giratina |
| `"Land Forme"` | Shaymin | Base Shaymin |
| `"Aria Forme"` | Meloetta | Base Meloetta |
| `"Midday Form"` | Lycanroc | Base Lycanroc |
| `"Red-Striped Form"` | Basculin | Base Basculin |
| `"Plant Cloak"` | Burmy, Wormadam | Base cloak form |
| `"Zero Form"` | Palafin | Base Palafin |
| `"Normal Form"` | Terapagos | Base Terapagos |
| `"Teal Mask"` | Ogerpon | Base Ogerpon |
| `"Male"` | Basculegion, Indeedee | Male is the base form |
| `"Keldeo Ordinary"` | Keldeo | Alternate default label |
| `"Ice Face"` | Eiscue | Default battle form |
| `"Full Belly Mode"` | Morpeko | Default mode |
| `"Hoopa Confined"` | Hoopa | Base Hoopa |
| `"Baile Style"` | Oricorio | Base Oricorio |
| `"Amped Form"` | Toxtricity | Base Toxtricity |
| `"Hero of Many Battles"` | Zacian, Zamazenta | Base box legends |
| `"Curly Form"` | Tatsugiri | Base Tatsugiri |
| `"Green Plumage"` | Squawkabilly | Base Squawkabilly |
| `"Family of Three"` | Maushold | Base Maushold |
| `"Two-Segment Form"` | Dudunsparce | Base Dudunsparce |
| `"Core Form"` | Minior | Base Minior |

## Manual Overrides (43 entries)

These are `(species, form)` pairs where the automatic normalization rules cannot produce the correct showdown ID. Each override maps directly to the expected vocabulary key.

### Regional Breeds (Paldean Tauros)
| CSV Name | CSV Form | Showdown ID | Why Override Needed |
|---|---|---|---|
| Tauros | Combat Breed | `taurospaldeacombat` | "paldea" prefix not in form string |
| Tauros | Blaze Breed | `taurospaldeablaze` | Same |
| Tauros | Aqua Breed | `taurospaldeaaqua` | Same |

### Gendered Species
| CSV Name | CSV Form | Showdown ID | Why Override Needed |
|---|---|---|---|
| Basculegion | Female | `basculegionf` | Suffix is `f`, not `female` |
| Indeedee | Female | `indeedeef` | Suffix is `f`, not `female` |
| Nidoran | Female | `nidoranf` | Suffix is `f` |
| Nidoran | Male | `nidoranm` | Suffix is `m` |
| Oinkologne | Female | `oinkolognef` | Suffix is `f` |

### Mega Evolutions with X/Y Variants
| CSV Name | CSV Form | Showdown ID | Why Override Needed |
|---|---|---|---|
| Charizard | Mega Charizard X | `charizardmegax` | Generic mega rule would drop X/Y |
| Charizard | Mega Charizard Y | `charizardmegay` | Same |
| Mewtwo | Mega Mewtwo X | `mewtwomegax` | Same |
| Mewtwo | Mega Mewtwo Y | `mewtwomegay` | Same |

### Calyrex Rider Forms
| CSV Name | CSV Form | Showdown ID | Why Override Needed |
|---|---|---|---|
| Calyrex | Ice Rider | `calyrexicerider` | Two-word suffix |
| Calyrex | Shadow Rider | `calyrexshadowrider` | Two-word suffix |

### Battle-Only / Alternate Forms
| CSV Name | CSV Form | Showdown ID | Why Override Needed |
|---|---|---|---|
| Ursaluna | Bloodmoon | `ursalunabloodmoon` | Direct mapping |
| Eiscue | Noice Face | `eiscuenoice` | Suffix is `noice`, not `noiceface` |
| Cramorant | Gulping Form | `cramorantgulping` | Strip "Form" suffix manually |
| Cramorant | Gorging Form | `cramorantgorging` | Same |
| Morpeko | Hangry Mode | `morpekohangry` | Strip "Mode" suffix manually |
| Darmanitan | Zen Mode | `darmanitanzen` | Strip "Mode" |

### Cosmetic / Rarity Variants
| CSV Name | CSV Form | Showdown ID | Why Override Needed |
|---|---|---|---|
| Sinistcha | Masterpiece | `sinistchamasterpiece` | Direct mapping |
| Polteageist | Antique | `polteageistantique` | Direct mapping |
| Pikachu | World Cap | `pikachuworld` | Suffix is `world`, not `worldcap` |
| Pikachu | Partner Cap | `pikachupartner` | Suffix is `partner`, not `partnercap` |

### Multi-Form Species
| CSV Name | CSV Form | Showdown ID | Why Override Needed |
|---|---|---|---|
| Squawkabilly | Blue Plumage | `squawkabillyblue` | Strip "Plumage" |
| Maushold | Family of Four | `mausholdfour` | Compress to `four` |
| Lycanroc | Midnight Form | `lycanrocmidnight` | Specific suffix |
| Lycanroc | Dusk Form | `lycanrocdusk` | Specific suffix |
| Toxtricity | Low Key Form | `toxtricitylowkey` | Two-word form compressed |
| Basculin | Blue-Striped Form | `basculinbluestriped` | Hyphen removal |
| Basculin | White-Striped Form | `basculinwhitestriped` | Hyphen removal |
| Oricorio | Pom-Pom Style | `oricoriopompom` | Hyphen removal |
| Oricorio | Sensu Style | `oricoriosensu` | Specific suffix |
| Oricorio | Pa'u Style | `oricoriopau` | Apostrophe removal |
| Minior | Meteor Form | `miniormeteor` | Specific suffix |
| Gimmighoul | Roaming Form | `gimmighoulroaming` | Specific suffix |
| Zarude | Dada | `zarudedada` | Direct mapping |

### Legendary / Mythical Formes
| CSV Name | CSV Form | Showdown ID | Why Override Needed |
|---|---|---|---|
| Meloetta | Pirouette Forme | `meloettapirouette` | Generic would strip "Forme" correctly but form string includes it |
| Hoopa | Hoopa Unbound | `hoopaunbound` | Species name repeated in form |
| Hoopa | Hoopa Confined | `hoopa` | Species name repeated in form; maps to base |
| Deoxys | Attack Forme | `deoxysattack` | Specific suffix |
| Deoxys | Defense Forme | `deoxysdefense` | Specific suffix |
| Deoxys | Speed Forme | `deoxysspeed` | Specific suffix |
| Giratina | Origin Forme | `giratinaorigin` | Specific suffix |
| Shaymin | Sky Forme | `shayminsky` | Specific suffix |
| Terapagos | Terastal Form | `terapagosterastal` | Specific suffix |
| Terapagos | Stellar Form | `terapagosstellar` | Specific suffix |

## Stat Aliases (31 entries)

These are species IDs present in the vocabulary but absent from the CSV because they represent in-battle form changes or cosmetic variants that share base stats with an existing entry. After loading the CSV, aliases are added by copying stats from the source species.

### Tera-Activated Ogerpon (4 aliases)
Ogerpon's tera-activated forms have distinct showdown IDs but identical base stats to their non-tera counterparts.

| Alias | Source | Stats (HP/Atk/Def/SpA/SpD/Spe) |
|---|---|---|
| `ogerponwellspringtera` | `ogerponwellspring` | 80/120/84/60/96/110 |
| `ogerponhearthflametera` | `ogerponhearthflame` | 80/120/84/60/96/110 |
| `ogerponcornerstonetera` | `ogerponcornerstone` | 80/120/84/60/96/110 |
| `ogerpontealtera` | `ogerpon` | 80/120/84/60/96/110 |

### In-Battle Form Changes (4 aliases)
These forms only appear during battle and share the base form's stats.

| Alias | Source | Rationale |
|---|---|---|
| `mimikyubusted` | `mimikyu` | Disguise broken — cosmetic only, same stats |
| `cramorantgulping` | `cramorant` | Gulp Missile form — same base stats |
| `cramorantgorging` | `cramorant` | Gorging form — same base stats |
| `miniormeteor` | `minior` | Meteor Form (shields up) — same in CSV |

### Cosmetic / Event Variants (4 aliases)
| Alias | Source | Rationale |
|---|---|---|
| `pikachuworld` | `pikachu` | Costume Pikachu, same stats |
| `pikachupartner` | `pikachu` | Partner Cap Pikachu, same stats |
| `polteageistantique` | `polteageist` | Antique form, purely cosmetic |
| `sinistchamasterpiece` | `sinistcha` | Masterpiece form, purely cosmetic |

### Zarude and Maushold (2 aliases)
| Alias | Source | Rationale |
|---|---|---|
| `zarudedada` | `zarude` | Dada form, same base stats |
| `mausholdfour` | `maushold` | Family of Four, same stats as Family of Three |

### Arceus Type Forms (17 aliases)
All Arceus type forms share the same 120/120/120/120/120/120 base stats. Each type form (`arceusbug`, `arceusdark`, `arceusdragon`, etc.) is aliased to the base `arceus` entry.

Types: Bug, Dark, Dragon, Electric, Fairy, Fighting, Fire, Flying, Ghost, Grass, Ground, Ice, Poison, Psychic, Rock, Steel, Water

### Box Legend Crowned Forms (2 aliases)
| Alias | Source | Note |
|---|---|---|
| `zaciancrownedsword` | `zacian` | Crowned Sword has different stats in reality, but the base "Hero of Many Battles" form is what appears in vocab |
| `zamazentacrownedshield` | `zamazenta` | Same rationale |

### Color / Shape Variants (4 aliases)
| Alias | Source | Rationale |
|---|---|---|
| `squawkabillyblue` | `squawkabilly` | Same stats across all plumage colors |
| `squawkabillyyellow` | `squawkabilly` | Same |
| `squawkabillywhite` | `squawkabilly` | Same |
| `dudunsparcethreesegment` | `dudunsparce` | Same stats regardless of segment count |

### Tatsugiri Variants (2 aliases)
| Alias | Source | Rationale |
|---|---|---|
| `tatsugiridroopy` | `tatsugiri` | All Tatsugiri forms share stats |
| `tatsugiristretchy` | `tatsugiri` | Same |

## Automatic Rule Coverage

The following form categories are handled by pattern-matching rules in `_normalize_species_name` rather than manual overrides:

| Rule | Example Input | Output | Species Count |
|---|---|---|---|
| Regional: Alolan | `("Ninetales", "Alolan Ninetales")` | `ninetalesalola` | ~18 |
| Regional: Galarian | `("Slowking", "Galarian Slowking")` | `slowkinggalar` | ~15 |
| Regional: Hisuian | `("Samurott", "Hisuian Samurott")` | `samurotthisui` | ~12 |
| Regional: Paldean | — | — | (handled by override for Tauros) |
| Ogerpon Masks | `("Ogerpon", "Wellspring Mask")` | `ogerponwellspring` | 3 |
| Rotom Forms | `("Rotom", "Wash Rotom")` | `rotomwash` | 5 |
| Kyurem Forms | `("Kyurem", "Black Kyurem")` | `kyuremblack` | 2 |
| Mega Evolutions | `("Venusaur", "Mega Venusaur")` | `venusaurmega` | ~40 |
| Therian Formes | `("Landorus", "Therian Forme")` | `landorustherian` | 4 |
| Generic fallback | `("Wormadam", "Sandy Cloak")` | `wormadamsandy` | Varies |

## Adding New Species

When new Pokemon or forms are added to the CSV or vocabulary:

1. Run the vocab coverage test: `pytest tests/test_base_stats.py::TestVocabCoverage -v`
2. If any species are missing, determine whether:
   - A **default form** entry is needed in `_DEFAULT_FORMS`
   - A **manual override** is needed in `_FORM_OVERRIDES`
   - A **stat alias** is needed in `_STAT_ALIASES` (for forms sharing stats but absent from CSV)
3. Add the appropriate entry and re-run the test
