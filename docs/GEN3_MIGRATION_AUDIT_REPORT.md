# Gen 3 Migration Audit Report

_Audit Date: March 15, 2026_

---

## Executive Summary

The Gen 3 OU migration proposal (M0–M8) has been **fully implemented for the core training pipeline**: replay download, parsing, observation construction, tensorization, model architecture, auxiliary labels, and testing. All 60,251 outdated Gen 9 OU data files (raw replays, processed tensors, and fine-tuning data) have been deleted.

However, **several environment, bot, config, and utility files still contain Gen 9 artifacts**. These do not affect offline training on Gen 3 replay data but must be cleaned up before running live Gen 3 battles, generating new sample data, or running analysis scripts.

---

## Migration Phase Verification

| Phase | Description | Status |
|-------|-------------|--------|
| M0 | Foundation (download pipeline, replay parser, vocabularies) | Done |
| M1 | Observation space (remove tera, no team preview, field features) | Done |
| M2 | Action space (13 → 9 actions) | Done |
| M3 | Tensorization (28-dim pokemon, 7-dim context) | Done |
| M4 | Model architecture (9-action policy head, Gen 3 configs) | Done |
| M5 | Environment & bots (heuristic bot, type-based split) | Partial |
| M6 | Training pipeline (auxiliary labels, training scripts) | Done |
| M7 | Testing (Gen 3-specific test suite) | Done |
| M8 | Evaluation (targets, archetype stratification) | Done |

---

## Remaining Gen 9 Artifacts

### HIGH Priority

#### `scripts/generate_sample_data.py`
**The entire file generates Gen 9 OU data.** Must be rewritten for Gen 3.

- Line 4: Docstring says "Gen 9 OU battle data"
- Lines 29–60: `OU_POKEMON` list contains Gen 9 Pokemon (Great Tusk, Gholdengo, Kingambit, Iron Valiant, Dragapult, etc.)
- Lines 62–73: `COMMON_MOVES` dict contains Gen 9 moves (Headlong Rush, Make It Rain, Kowtow Cleave, U-turn, Defog, etc.)
- Lines 75–80: `COMMON_ITEMS` list contains Gen 9 items (Heavy-Duty Boots, Choice Specs, Choice Scarf, Booster Energy, Covert Cloak, etc.)
- Lines 82–87: `COMMON_ABILITIES` list contains Gen 9 abilities (Protosynthesis, Quark Drive, Good as Gold, Supreme Overlord, etc.)
- Lines 89–93: `TERA_TYPES` array includes Fairy type and is used for tera_type generation
- Line 96: `TERRAIN_OPTIONS` includes terrains that don't exist in Gen 3
- Line 164: `"tera_type": random.choice(TERA_TYPES)`
- Lines 217, 232: `can_tera` logic in state generation
- Line 220: `"format": "gen9ou"` hardcoded
- Line 235: `"opponent_teampreview"` populated (Gen 3 has no team preview)
- Lines 240–242: Side conditions include Stealth Rock and Toxic Spikes (Gen 4+)
- Line 259: Filename pattern uses `gen9ou`

**Required changes:** Replace all Pokemon, moves, items, abilities with Gen 3 equivalents. Remove tera, terrain, Stealth Rock, Toxic Spikes. Set format to `gen3ou`. Empty `opponent_teampreview`. Add `weather_permanent` flag.

---

### MEDIUM Priority

#### `configs/environment/showdown.yaml`
- Line 21: `format: gen9ou` — should be `gen3ou`
- Line 23: `"Team Preview"` rule — Gen 3 has no team preview; remove this rule

#### `configs/training/bc.yaml`
- Line 41: `tags: ["bc", "gen9ou"]` — should be `["bc", "gen3ou"]`

#### `src/bots/model_bot.py`
Bridges live `BattleState` objects to `PokemonObservation` for inference. Still references tera fields from `state.py`.
- Line 49: `tera_type=poke.tera_type or UNKNOWN`
- Line 50: `terastallized=poke.terastallized`
- Line 70: `tera_type=poke.tera_type if poke.tera_type != UNKNOWN else UNKNOWN`
- Line 71: `terastallized=poke.terastallized`
- Line 132: `can_tera=state.can_terastallize`

**Note:** `PokemonObservation` no longer has `tera_type` or `terastallized` fields. These lines will raise `TypeError` if `model_bot.py` is used. This is a latent bug.

#### `src/bots/max_damage_bot.py`
- Line 25: Comment says "Gen 9 OU moves"
- Lines 29–93: `_COMMON_MOVE_POWER` dict contains Gen 9 moves not in Gen 3: Moonblast, U-turn, Volt Switch, Stealth Rock, Toxic Spikes, Sticky Web, Defog, Roost, Nasty Plot, Scald, Brave Bird, Head Smash, Stone Edge, Iron Head, Bullet Punch, Aqua Jet, Ice Shard, Sucker Punch, Dark Pulse, Draco Meteor, Leaf Storm, Focus Blast, Hurricane, Play Rough, Icicle Crash, Wild Charge, Zen Headbutt, Poison Jab, Seed Bomb, X-Scissor

**Required changes:** Replace move database with Gen 3 OU moves. Update comment.

#### `scripts/analyze_p8_dataset_efficiency.py`
- Line 35: `terastallized_flag = 0`
- Line 57: `x[:, :, 29]` — references feature index 29, but pokemon features are now 28 dims (indices 0–27). This will produce incorrect results or an index error.
- Line 104: Reports `terastallized_flag_rate`

**Required changes:** Remove terastallized analysis. Fix feature indexing for 28-dim vectors.

---

### LOW Priority

#### `src/environment/state.py`
Full terastallization state tracking remains wired. Harmless for Gen 3 (tera fields stay at defaults) but represents dead code.
- Lines 83–84: `OwnPokemon` has `tera_type: str = ""`, `terastallized: bool = False`
- Lines 120–121: `OpponentPokemon` has `tera_type: str = UNKNOWN`, `terastallized: bool = False`
- Lines 265–266: `BattleState` has `can_terastallize: bool = True`, `opponent_has_terastallized: bool = False`
- Lines 501–505: JSON parsing for `teraType` and `terastallized` fields
- Lines 887–901: `_handle_terastallize()` method
- Line 1000: `MessageType.TERASTALLIZE` handler registration

#### `src/environment/protocol.py`
- Line 73: `TERASTALLIZE = "-terastallize"` enum member
- Lines 423–432: `parse_terastallize_message()` function

#### `src/environment/showdown_client.py`
- Line 72: Example code uses `format="gen9ou"`
- Line 170: Docstring mentions `"gen9ou"` as example format

#### `scripts/run_phase1_exit_gate.py`
- Line 55: `BATTLE_FORMAT = "gen9randombattle"` — should be `gen3ou` or `gen3randombattle`

#### `scripts/train_phase4.py`
- Line 799: Help text mentions "terastallized flag" as a dead channel example

#### `tests/test_state.py`
- Lines 383–418: `TestTerastallization` class tests tera message handling
- Line 79: Asserts `opp.tera_type == UNKNOWN`
- Line 574: Asserts `opp.tera_type == UNKNOWN`

#### `tests/test_protocol.py`
- Lines 198–200: `test_terastallize_message()` tests tera message parsing
- Lines 280–284: `test_parse_terastallize_message()` tests structured data extraction

#### `tests/test_parser.py`
- Line 88: `can_tera` parameter in test helper
- Line 110: `"can_tera": can_tera` in synthetic state dict
- Lines 251–254: Test asserts `can_tera` is False

---

## Data Deletion Record

The following Gen 9 OU data was deleted in this audit:

| Directory | Files Deleted | Size |
|-----------|--------------|------|
| `data/raw/` | 30,001 `.json.lz4` files | 280 MB |
| `data/processed/battles/` | 29,648 `.npz` files | 118 MB |
| `data/fine-tuning/raw/` | 300 `.json.lz4` files | 2.7 MB |
| `data/fine-tuning/processed/battles/` | 311 `.npz` files | 1.4 MB |
| `data/fine-tuning/splits/` | 3 manifest files | 20 KB |
| **Total** | **60,263 files** | **~402 MB** |

To download Gen 3 OU replay data:
```bash
python scripts/download_replays.py --generation gen3ou --elo-threshold 1300
```

---

## Recommended Cleanup Order

1. **`scripts/generate_sample_data.py`** — Rewrite for Gen 3 (blocks sample data generation)
2. **`configs/environment/showdown.yaml`** — Fix format and rules (blocks live battles)
3. **`configs/training/bc.yaml`** — Fix tags (cosmetic but misleading)
4. **`src/bots/model_bot.py`** — Remove tera references (latent TypeError bug)
5. **`src/bots/max_damage_bot.py`** — Update move database for Gen 3
6. **`scripts/analyze_p8_dataset_efficiency.py`** — Fix feature indexing
7. **Environment layer** (`state.py`, `protocol.py`) — Remove tera dead code
8. **Tests** (`test_state.py`, `test_protocol.py`, `test_parser.py`) — Remove/update tera tests
9. **Remaining scripts and docs** — Update format strings and comments
