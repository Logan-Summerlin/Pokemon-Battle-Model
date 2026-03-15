# Gen 3 Migration Audit Report

_Initial Audit: March 15, 2026_
_Full Verification Audit: March 15, 2026_

---

## Executive Summary

The Gen 3 OU migration proposal (M0–M8) has been **fully implemented for the core training pipeline**: replay download, parsing, observation construction, tensorization, model architecture, auxiliary labels, and testing. The evaluation specification has been recalibrated for Gen 3. All 60,263 outdated Gen 9 OU data files (~402 MB) have been deleted.

**However, Phase M5 (Environment & Bots) is only partially complete.** Several environment, bot, config, and utility files still contain Gen 9 artifacts. These do not affect offline training on Gen 3 replay data but must be cleaned up before running live Gen 3 battles, generating sample data, or running certain analysis scripts. One latent `TypeError` bug exists in `model_bot.py` that will crash at runtime.

---

## Migration Phase Verification

| Phase | Description | Status | Confidence |
|-------|-------------|--------|------------|
| M0 | Foundation (download pipeline, replay parser, vocabularies) | **Done** | High |
| M1 | Observation space (remove tera, no team preview, field features) | **Done** | High |
| M2 | Action space (13 → 9 actions) | **Done** | High |
| M3 | Tensorization (28-dim pokemon, 7-dim context, 19-dim field) | **Done** | High |
| M4 | Model architecture (9-action policy head, Gen 3 configs, aux head) | **Done** | High |
| M5 | Environment & bots (heuristic bot, type-based split, configs) | **Partial** | Medium |
| M6 | Training pipeline (auxiliary labels, training scripts, process_dataset) | **Done** | High |
| M7 | Testing (Gen 3-specific test suite) | **Done** | High |
| M8 | Evaluation (targets, archetype stratification, evaluation spec) | **Done** | High |

---

## Detailed Phase-by-Phase Verification

### Phase M0: Foundation Changes — **DONE**

#### M0.1: Replay Download Pipeline
**File:** `scripts/download_replays.py`

| Requirement | Status | Evidence |
|-------------|--------|----------|
| Download target changed from `gen9ou.tar.gz` to `gen3ou.tar.gz` | Done | Line 44: `generation: str = "gen3ou"` default |
| Function parameterized by generation | Done | `download_replays_tar(generation="gen3ou", ...)` at line 43 |
| Elo threshold adjusted for Gen 3 | Done | Lines 37–40: `DEFAULT_ELO_THRESHOLDS = {"gen3ou": 1300, "gen9ou": 1500}` |
| CLI help text updated | Done | Lines 172–176: `default="gen3ou"`, help text lists both gens |
| Supports both Gen 3 and Gen 9 | Done | `--generation` flag with per-gen Elo defaults |

**Verdict:** Fully compliant with proposal.

#### M0.2: Replay Parser
**File:** `src/data/replay_parser.py`

| Requirement | Status | Evidence |
|-------------|--------|----------|
| Handles Gen 3 format (no team preview) | Done | Lines 8–13: Docstring documents Gen 3 differences |
| `can_tera` always False for Gen 3 | Done | Line 122: `can_tera: bool = False` default; line 282: parsed from data |
| Empty `opponent_teampreview` handled | Done | Line 125: defaults to empty list; `_parse_teampreview()` handles gracefully |
| Generation field on `ParsedBattle` | Done | Lines 152–157: `generation` property parses from format string |
| `has_team_preview` property | Done | Lines 160–166: Returns `gen >= 5` (Gen 3 = False) |
| Weather duration semantics noted | Done | Line 12: "Weather from abilities is permanent" |

**Verdict:** Fully compliant. Parser gracefully handles both Gen 3 and Gen 9 formats.

#### M0.3: Vocabularies
**File:** `src/data/tensorizer.py`, `data/processed/vocabs/`

| Requirement | Status | Evidence |
|-------------|--------|----------|
| `Vocabulary` class builds dynamically | Done | Lines 55–65: `add()` method; line 77: `freeze()` |
| Vocabulary files exist | Done | `data/processed/vocabs/` contains all 9 vocab JSON files |
| `process_dataset.py` saves gen-specific vocabs | Done | Lines 151–154: Saves to `vocabs/gen3ou/` and `vocabs/` |

**Verdict:** Vocabulary system is generation-agnostic and ready for Gen 3 data.

---

### Phase M1: Observation Space Changes — **DONE**

#### M1.1: Remove Terastallization from Observation
**File:** `src/data/observation.py`

| Requirement | Status | Evidence |
|-------------|--------|----------|
| `tera_type` removed from `PokemonObservation` | Done | Lines 36–58: No `tera_type` field in dataclass |
| `terastallized` removed from `PokemonObservation` | Done | Lines 36–58: No `terastallized` field |
| `can_tera` removed from `TurnObservation` | Done | Lines 96–123: No `can_tera` field |

**Verdict:** All tera fields cleanly removed from observation dataclasses.

#### M1.2: No Team Preview Handling (Critical Change)
**File:** `src/data/observation.py`

| Requirement | Status | Evidence |
|-------------|--------|----------|
| `OpponentTracker` class implemented | Done | Lines 297–383: Full implementation with `revealed_species`, `_last_known_state` |
| Revealed species tracked in switch-in order | Done | Lines 337–338: `if species not in self.revealed_species: self.revealed_species.append(species)` |
| `build_observations()` handles no-preview | Done | Lines 518–533: `else:` branch iterates `tracker.revealed_species` instead of `opponent_teampreview` |
| Unrevealed slots padded with empty obs | Done | Lines 535–538: Pads to `MAX_TEAM_SIZE` |
| `num_opponent_revealed` context feature added | Done | Line 117: Field in `TurnObservation`; line 574: `num_opponent_revealed=tracker.num_revealed` |
| `is_lead_turn` context feature added | Done | Line 119: Field in `TurnObservation`; line 575: `is_lead_turn=(t == 0)` |
| Current turn's active opponent tracked | Done | Lines 458–463: Active opponent added to `revealed_species` before building obs |

**Verdict:** No-team-preview handling is thorough and correct.

#### M1.3: Field Observation for Gen 3
**File:** `src/data/observation.py`

| Requirement | Status | Evidence |
|-------------|--------|----------|
| Terrain always empty for Gen 3 | Done | Lines 262–264: `if battle_field and not is_gen3: terrain = battle_field` |
| `weather_permanent` flag added | Done | Line 74: Field in `FieldObservation`; line 270: `weather_permanent=is_gen3 and bool(weather)` |
| Stealth Rock zeroed for Gen 3 | Done | Line 272: `if not is_gen3 else False` |
| Toxic Spikes zeroed for Gen 3 | Done | Line 276: `if not is_gen3 else 0` |
| Sticky Web zeroed for Gen 3 | Done | Line 278: `if not is_gen3 else False` |
| Aurora Veil zeroed for Gen 3 | Done | Lines 282, 292: `if not is_gen3 else False` |
| Tailwind zeroed for Gen 3 | Done | Lines 284, 293: `if not is_gen3 else False` |
| Spikes kept as-is (Gen 2+) | Done | Lines 274, 287: No Gen 3 guard on Spikes |
| Reflect/Light Screen kept as-is | Done | Lines 279–280, 290–291: No Gen 3 guard |

**Verdict:** Field observation correctly zeros out Gen 4+ features while preserving Gen 3 features.

---

### Phase M2: Action Space Changes — **DONE**

#### M2.1: Reduce Action Space from 13 to 9
**File:** `src/environment/action_space.py`

| Requirement | Status | Evidence |
|-------------|--------|----------|
| `NUM_ACTIONS = 9` | Done | Line 38 |
| No `MOVE_TERA` action type | Done | `ActionType` only has `MOVE` and `SWITCH` (lines 18–22) |
| No tera move constants | Done | Only `MOVE_1`–`MOVE_4` (0–3) and `SWITCH_2`–`SWITCH_6` (4–8) defined |
| `ACTION_NAMES` updated | Done | Lines 41–51: 9 names, no tera entries |
| `BattleAction` updated | Done | Lines 54–90: No tera logic in `canonical_index` or `to_showdown_command()` |
| `action_from_canonical_index()` updated | Done | Lines 93–104: Maps 0–3 to moves, 4–8 to switches |
| `action_from_showdown_choice()` updated | Done | Lines 107–143: No tera parsing |
| `ActionMask` uses 9 actions | Done | Line 154: `self._mask = [False] * NUM_ACTIONS` |

**Verdict:** Action space cleanly reduced to 9 with no tera remnants.

#### M2.2: Legal Mask Builder
**File:** `src/data/observation.py`

| Requirement | Status | Evidence |
|-------------|--------|----------|
| Mask size is 9 | Done | Line 394: `mask = [False] * NUM_ACTIONS` (NUM_ACTIONS=9) |
| No tera logic | Done | Lines 386–423: Only move (0–3) and switch (4+i) logic |
| Switch indices at 4+i | Done | Lines 403, 417: `mask[4 + i] = True` |

**Verdict:** Legal mask fully Gen 3 compliant.

---

### Phase M3: Tensorization Changes — **DONE**

#### M3.1: Pokemon Feature Vector (28 dims)
**File:** `src/data/tensorizer.py`

| Requirement | Status | Evidence |
|-------------|--------|----------|
| `POKEMON_BINARY_DIM = 5` (was 7) | Done | Line 174 |
| `POKEMON_FEATURE_DIM = 28` (was 30) | Done | Line 175: `9 + 14 + 5 = 28` |
| `is_unknown_tera` removed | Done | Lines 263–273: Only 5 binary features (active, fainted, own, unknown_item, unknown_ability) |
| `terastallized` removed | Done | Same evidence |

**Verdict:** Pokemon feature vector correctly reduced to 28 dims.

#### M3.2: Context Feature Vector (7 dims)
**File:** `src/data/tensorizer.py`

| Requirement | Status | Evidence |
|-------------|--------|----------|
| `CONTEXT_FEATURE_DIM = 7` | Done | Line 186 |
| `can_tera` removed | Done | Lines 361–374: No can_tera in context features |
| `num_opponent_revealed` added | Done | Line 364: `context[2] = obs.num_opponent_revealed / 6.0` |
| `is_lead_turn` added | Done | Line 366: `context[4] = float(obs.is_lead_turn)` |
| Context layout: turn_num(0), opp_remaining(1), num_opp_revealed(2), forced_switch(3), is_lead_turn(4), prev_player_move(5), prev_opp_move(6) | Done | Lines 362–374 |

**Verdict:** Context features correctly updated for Gen 3.

#### M3.3: Field Feature Tensorization
**File:** `src/data/tensorizer.py`

| Requirement | Status | Evidence |
|-------------|--------|----------|
| `FIELD_FEATURE_DIM = 19` | Done | Line 180: `2 + 17` (2 categorical + 17 binary including weather_permanent) |
| `weather_permanent` flag tensorized | Done | Line 306: `features[idx] = float(field.weather_permanent)` |
| Dead Gen 4+ features preserved but zeroed | Done | Lines 309–326: All side conditions tensorized (zeroed by observation layer for Gen 3) |

**Verdict:** Field tensorization correct with permanent weather flag.

#### M3.4: Metamon Action Mapping
**File:** `src/data/tensorizer.py`

| Requirement | Status | Evidence |
|-------------|--------|----------|
| Gen 3 action mapping (9 actions) | Done | Lines 384–387: `_METAMON_TO_CANONICAL = {0:0, 1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:8}` |
| Target action forced legal | Done | Lines 397–398: `if 0 <= action_idx < NUM_ACTIONS: legal_mask[action_idx] = 1.0` |

**Verdict:** Action mapping correct for Gen 3.

---

### Phase M4: Model Architecture Changes — **DONE**

#### M4.1: BattleTransformer
**File:** `src/models/battle_transformer.py`

| Requirement | Status | Evidence |
|-------------|--------|----------|
| Policy head outputs 9 actions | Done | Line 609: `nn.Linear(config.hidden_dim, NUM_ACTIONS)` where NUM_ACTIONS=9 |
| `PokemonEmbedding` handles 28-dim input | Done | Lines 236–304: Uses `POKEMON_CATEGORICAL_DIM`, `POKEMON_CONTINUOUS_DIM`, `POKEMON_BINARY_DIM` |
| `FieldEmbedding` with `prune_dead_features` | Done | Lines 319–327: Binary dim set to 0 when pruning, only uses weather/terrain embeddings |
| `ContextEmbedding` handles 7-dim input | Done | Lines 356–382: `cont_dim = 5` + 2 move embeddings; reads indices 0–4 as continuous, 5–6 as categorical |
| Default vocab sizes for Gen 3 | Done | Lines 60–67: Smaller defaults (species=200, moves=400, items=80, abilities=150) |
| `terrain_vocab_size = 5` | Done | Line 67 |
| 14 tokens per step | Done | Line 230: `TOKENS_PER_STEP = 14` |
| Input docstring updated | Done | Lines 14–19: "Input tokens (Gen 3 OU)" |

**Verdict:** Model architecture fully adapted for Gen 3.

#### M4.2: Auxiliary Head
**File:** `src/models/battle_transformer.py`

| Requirement | Status | Evidence |
|-------------|--------|----------|
| No tera category prediction | Done | Lines 635–697: No `tera_logits` in `AuxiliaryHead`; only item, threat_profile, move_family |
| `num_item_classes = 25` (Gen 3) | Done | Line 81 |
| `num_speed_buckets = 5` | Done | Line 82 |
| `num_role_archetypes = 8` | Done | Line 83 |
| `num_move_families = 10` | Done | Line 84 |
| Joint threat profile head (speed × role) | Done | Lines 655–661: `threat_classes = num_speed_buckets * num_role_archetypes`; marginalized via logsumexp |

**Verdict:** Auxiliary head correctly updated for Gen 3. No tera category.

#### M4.3: Gen 3 Config Profiles
**File:** `src/models/battle_transformer.py`

| Requirement | Status | Evidence |
|-------------|--------|----------|
| `p8_gen3()` class method | Done | Lines 149–184: 4L/256d/4H, smaller embeddings, `num_item_classes=25` |
| `p8_lean()` for Gen 3 | Done | Lines 187–221: 3L/192d/4H, `ffn_multiplier=3`, no value head |
| Smaller embedding dims for Gen 3 | Done | Both profiles: species=48, moves=24, items=12, abilities=12, types=12 |

**Verdict:** Config profiles match Gen 3 proposal specifications.

---

### Phase M5: Environment and Bot Changes — **PARTIAL**

#### M5.1: Showdown Server Configuration — **NOT DONE**
**File:** `configs/environment/showdown.yaml`

| Requirement | Status | Evidence |
|-------------|--------|----------|
| Format changed to `gen3ou` | **Not Done** | Line 21: Still `format: gen9ou` |
| Team Preview rule removed | **Not Done** | Line 23: Still has `"Team Preview"` rule |

#### M5.2: Heuristic Bot — **DONE**
**File:** `src/bots/heuristic_bot.py`

| Requirement | Status | Evidence |
|-------------|--------|----------|
| Type-based physical/special split | Done | Lines 41–58: `PHYSICAL_TYPES`, `SPECIAL_TYPES`, `get_gen3_category()` |
| Gen 3 type chart (no Fairy, Steel resists Dark/Ghost) | Done | Lines 64–130: Complete `_TYPE_CHART` with Gen 3 mechanics |
| Explosion/Self-Destruct Defense-halving | Done | Lines 326, 389–390: `_EXPLOSION_MOVES` set; `explosion_bonus = 2.0` in damage calc |
| Move database is Gen 3 | Done | Lines 136–300: `_MOVE_DATA` dict with Gen 3 OU moves (includes notes for non-existent moves with 0 power) |
| Species type database is Gen 3 | Done | Lines 623–762: `_SPECIES_TYPES` with ~140 Gen 3 Pokemon, no Fairy type |
| No tera logic | Done | No tera references anywhere in the file |
| Gen 3 docstring | Done | Lines 1–15: References Gen 3 OU mechanics |

**Verdict:** Heuristic bot fully migrated to Gen 3.

#### M5.3: Other Bots — **NOT DONE**

**`src/bots/model_bot.py` — CRITICAL BUG:**
- Lines 49–50: `tera_type=poke.tera_type or UNKNOWN`, `terastallized=poke.terastallized` — `PokemonObservation` no longer has these fields. **Will raise `TypeError` at runtime.**
- Lines 70–71: Same issue for opponent observation.
- Line 132: `can_tera=state.can_terastallize` — `TurnObservation` no longer has `can_tera` field. **Will raise `TypeError`.**

**`src/bots/max_damage_bot.py`:**
- Line 25: Comment says "Gen 9 OU moves"
- Lines 29–93: `_COMMON_MOVE_POWER` dict contains many Gen 4+ moves not in Gen 3 (Moonblast, U-turn, Volt Switch, Stealth Rock, Toxic Spikes, Sticky Web, Defog, Roost, Nasty Plot, Scald, Brave Bird, Head Smash, Stone Edge, Iron Head, Bullet Punch, Aqua Jet, Ice Shard, Sucker Punch, Dark Pulse, Draco Meteor, Leaf Storm, Focus Blast, Hurricane, Play Rough, Icicle Crash, Wild Charge, Zen Headbutt, Poison Jab, Seed Bomb, X-Scissor)
- Not blocking for offline training (used only in live battle evaluation), but will produce inaccurate damage estimates against Gen 3 teams.

#### M5.4: Legality Rules — **NOT VERIFIED**
**File:** `src/environment/legality.py`
- Legality rules are used only in live battles, not offline training. No specific Gen 3 legality updates were verified. The proposal requested removing tera-related legality checks and updating for Gen 3 move legality.

---

### Phase M6: Training Pipeline Changes — **DONE**

#### M6.1: Data Processing Script
**File:** `scripts/process_dataset.py`

| Requirement | Status | Evidence |
|-------------|--------|----------|
| Default generation set to `gen3ou` | Done | Line 39: `default="gen3ou"` |
| Vocab saved to generation-specific subdir | Done | Lines 151–154: `vocabs/gen3ou/` and `vocabs/` |
| Priors rebuilt from data | Done | Lines 121, 167: `priors.update_from_battle()` and `priors.save()` |
| Metadata includes generation | Done | Line 87: `"generation": generation` |
| Streaming processing for memory efficiency | Done | Line 100: `iter_battles_from_directory()` |

**Verdict:** Processing script fully parameterized for Gen 3.

#### M6.2: Training Scripts
**File:** `scripts/train_p8_lean.py`

| Requirement | Status | Evidence |
|-------------|--------|----------|
| Checkpoint dir uses gen3 naming | Done | Line 52: `default="checkpoints/phase4_gen3_p8_lean"` |
| Config matches Gen 3 P8-Lean profile | Done | Lines 166–179: 3L/224d/4H, FFN 3x, embeddings (48/24/16/16/12) |
| No value head | Done | Line 177: `"--no-value-head"` |
| Dead feature pruning enabled | Done | Line 178: `"--prune-dead-features"` |
| Default max window = 5 | Done | Line 42: `default=5` |

**Verdict:** Training wrapper fully configured for Gen 3.

#### M6.3: Auxiliary Label Extraction
**File:** `src/data/auxiliary_labels.py`

| Requirement | Status | Evidence |
|-------------|--------|----------|
| Gen 3 item taxonomy (25 classes) | Done | Lines 39–65: `ITEM_CLASSES` with 25 Gen 3 items |
| Item class map with type-boost grouping | Done | Lines 72–96: `_TYPE_BOOST_ITEMS`, `_PINCH_HP_BERRIES`, `_STATUS_CURE_BERRIES` grouped |
| `NUM_ITEM_CLASSES = 25` | Done | Line 98 |
| Gen 3 speed thresholds [110, 90, 65, 40] | Done | Line 117 |
| Role archetypes (8 classes, no tera role) | Done | Lines 139–149: No tera-related role |
| Pivot = Baton Pass only | Done | Line 152: `_PIVOT_MOVES = {"batonpass"}` |
| Hazard setup = Spikes only | Done | Line 155: `_HAZARD_MOVES = {"spikes"}` |
| Recovery moves Gen 3-appropriate | Done | Lines 158–161: No Roost, Shore Up, Strength Sap |
| Setup moves Gen 3-appropriate | Done | Lines 171–175: No Nasty Plot, Shell Smash, Quiver Dance |
| Move families (10 classes) | Done | Lines 262–274: Gen 3-appropriate |
| Priority moves: ExtremeSpeed, Mach Punch, Quick Attack, Fake Out only | Done | Lines 277–279: No Shadow Sneak, Sucker Punch, Bullet Punch |
| Hazard removal: Rapid Spin only | Done | Line 282: `_HAZARD_REMOVAL_MOVES = {"rapidspin"}` |
| Phazing: Whirlwind, Roar, Haze (no Dragon Tail/Circle Throw) | Done | Line 288 |
| No tera category in labels | Done | No tera references in the file |

**Verdict:** Auxiliary labels comprehensively updated for Gen 3.

---

### Phase M7: Testing Updates — **DONE**

#### M7.1: Test Fixtures Updated
**File:** `tests/test_gen3_mechanics.py`

| Requirement | Status | Evidence |
|-------------|--------|----------|
| Test Pokemon are Gen 3 (Salamence, Metagross) | Done | Lines 64–87: Helper uses Salamence default |
| Test format is `gen3ou` | Done | Line 97: `format="gen3ou"` |
| No tera in test fixtures | Done | No tera references in test helpers |

#### M7.2: Gen 3-Specific Tests

| Test Suite | Status | Evidence |
|------------|--------|----------|
| Physical/special type split (9 tests) | Done | Lines 109–157: `TestPhysicalSpecialSplit` |
| Gen 3 type chart (8 tests) | Done | Lines 162–202: `TestGen3TypeChart` — no Fairy, Steel resists Dark/Ghost |
| Permanent weather (3 tests) | Done | Lines 207–224: `TestPermanentWeather` |
| No team preview (5 tests) | Done | Lines 229–294: `TestNoTeamPreview` — tracker, accumulation, dedup, build_observations |
| Legal mask/no tera (5 tests) | Done | Lines 299–333: `TestLegalMaskGen3` — 9 actions, indices, no tera |
| Explosion/Self-Destruct mechanic (2 tests) | Done | Lines 338–376: `TestExplosionMechanic` |
| Gen 3 item taxonomy (7 tests) | Done | Lines 381–422: `TestGen3ItemTaxonomy` |
| Gen 3 move families (7 tests) | Done | Lines 427–480: `TestGen3MoveFamilies` — Spikes-only hazards, Rapid Spin-only removal, Baton Pass-only pivot |
| Gen 3 speed tiers (5 tests) | Done | Lines 485–512: `TestGen3SpeedTiers` |

**Verdict:** Comprehensive Gen 3-specific test suite with 51 test cases covering all key mechanics.

---

### Phase M8: Evaluation Recalibration — **DONE**

**File:** `docs/EVALUATION_SPEC.md`

| Requirement | Status | Evidence |
|-------------|--------|----------|
| Title updated to Gen 3 OU | Done | Line 1: "Evaluation Specification (Gen 3 OU)" |
| 9-action space noted | Done | Line 14: "9-action space: 4 moves + 5 switches (no Terastallization)" |
| Gen 3 archetypes (TSS, Baton Pass, etc.) | Done | Lines 86–98: TSS, Bulky Offense, Hyper Offense, Stall, Weather, Baton Pass chains |
| No Trick Room (replaced with Baton Pass) | Done | Line 96: "Trick Room does not exist in Gen 3" |
| Trapper sub-strategy documented | Done | Line 98: Dugtrio (Arena Trap), Magneton (Magnet Pull) |
| Speed bucket thresholds noted | Done | Line 65: "[110, 90, 65, 40]" |
| Spikes-only hazard acknowledged | Done | Line 128: "Spikes usage rate" as tracking metric |
| Lead matchup evaluation | Done | Lines 53, 129: Lead accuracy and lead win rate metrics |
| Scouting efficiency metric | Done | Line 78, line 141: Opponent team reveal curve |
| Weather exploitation metric | Done | Line 130: "Weather exploitation rate" |
| Physical/special accuracy diagnostic | Done | Line 145: "Physical/special damage calc accuracy" |
| Item prediction floors adjusted | Done | Line 75: "Gen 3's item distribution is heavily concentrated" |
| Gen 3 footer | Done | Line 181: "No Terastallization, no Stealth Rock, 9-action space" |

**Verdict:** Evaluation spec thoroughly recalibrated for Gen 3 OU.

---

## Remaining Gen 9 Artifacts

### CRITICAL — Runtime Bug

#### `src/bots/model_bot.py`
**Bug severity: Will crash with `TypeError` at runtime.**

`PokemonObservation` no longer has `tera_type` or `terastallized` fields, and `TurnObservation` no longer has `can_tera`. These lines will raise `TypeError` when `model_bot.py` is used for inference:

| Line | Code | Issue |
|------|------|-------|
| 49 | `tera_type=poke.tera_type or UNKNOWN` | `PokemonObservation.__init__()` got unexpected keyword argument `tera_type` |
| 50 | `terastallized=poke.terastallized` | `PokemonObservation.__init__()` got unexpected keyword argument `terastallized` |
| 70 | `tera_type=poke.tera_type if poke.tera_type != UNKNOWN else UNKNOWN` | Same crash |
| 71 | `terastallized=poke.terastallized` | Same crash |
| 132 | `can_tera=state.can_terastallize` | `TurnObservation.__init__()` got unexpected keyword argument `can_tera` |

**Impact:** Model cannot be used for live battle inference until fixed. Offline training is unaffected.

**Fix:** Remove the 5 lines above. The `tera_type`, `terastallized`, and `can_tera` kwargs should simply be deleted.

---

### HIGH Priority

#### `scripts/generate_sample_data.py`
**The entire file generates Gen 9 OU data.** Must be rewritten for Gen 3.

| Line(s) | Issue |
|---------|-------|
| 4 | Docstring says "Gen 9 OU battle data" |
| 29–60 | `OU_POKEMON` list contains Gen 9 Pokemon (Great Tusk, Gholdengo, Kingambit, Iron Valiant, Dragapult, etc.) |
| 62–73 | `COMMON_MOVES` dict contains Gen 9 moves (Headlong Rush, Make It Rain, Kowtow Cleave, U-turn, Defog, etc.) |
| 75–80 | `COMMON_ITEMS` list contains Gen 9 items (Heavy-Duty Boots, Choice Specs, Choice Scarf, Booster Energy, Covert Cloak, etc.) |
| 82–87 | `COMMON_ABILITIES` list contains Gen 9 abilities (Protosynthesis, Quark Drive, Good as Gold, Supreme Overlord, etc.) |
| 89–93 | `TERA_TYPES` array includes Fairy type and is used for tera_type generation |
| 96 | `TERRAIN_OPTIONS` includes terrains that don't exist in Gen 3 |
| 114 | `"move_type": random.choice(TERA_TYPES)` — uses Fairy type for random move types |
| 164 | `"tera_type": random.choice(TERA_TYPES)` — generates tera data |
| 217 | `can_tera = t < num_turns // 2` — tera logic in state generation |
| 220 | `"format": "gen9ou"` hardcoded |
| 232 | `"can_tera": can_tera` — tera in state dict |
| 235 | `"opponent_teampreview"` populated (Gen 3 has no team preview) |
| 240–242 | Side conditions include Stealth Rock and Toxic Spikes (Gen 4+) |
| 259 | Filename pattern uses `gen9ou` |

**Required changes:** Replace all Pokemon, moves, items, abilities with Gen 3 equivalents. Remove tera, terrain, Stealth Rock, Toxic Spikes. Set format to `gen3ou`. Empty `opponent_teampreview`. Add `weather_permanent` flag.

---

### MEDIUM Priority

#### `configs/environment/showdown.yaml`
| Line | Issue | Fix |
|------|-------|-----|
| 21 | `format: gen9ou` | Change to `gen3ou` |
| 23 | `"Team Preview"` rule | Remove (Gen 3 has no team preview) |

#### `configs/training/bc.yaml`
| Line | Issue | Fix |
|------|-------|-----|
| 41 | `tags: ["bc", "gen9ou"]` | Change to `["bc", "gen3ou"]` |

#### `src/bots/max_damage_bot.py`
| Line(s) | Issue | Fix |
|---------|-------|-----|
| 25 | Comment says "Gen 9 OU moves" | Update comment |
| 29–93 | Move database contains Gen 4+ moves not in Gen 3 | Replace with Gen 3 OU move database |

**Note:** Many moves listed don't exist in Gen 3 (Moonblast, U-turn, Volt Switch, Scald, Brave Bird, etc.) The bot will still function (defaulting to 80 power for unknown moves), but damage estimates will be inaccurate for Gen 3 battles.

#### `scripts/analyze_p8_dataset_efficiency.py`
| Line(s) | Issue | Fix |
|---------|-------|-----|
| 29 | `action_counts = np.zeros(13, ...)` | Change to `np.zeros(9, ...)` |
| 35 | `terastallized_flag = 0` — tracks dead feature | Remove |
| 56 | `x[:, :, 28]` — indexes dim 28 (was `is_unknown_tera`) | Remove (only 28 dims now, max index is 27) |
| 57 | `x[:, :, 29]` — indexes dim 29 (was `terastallized`) | **Index out of bounds** — will crash or produce wrong results |
| 95–97 | Action mix assumes 13 actions (indices 4–7 as "tera_move") | Fix for 9-action space |
| 103–104 | Reports `unknown_tera_rate` and `terastallized_flag_rate` | Remove |

---

### LOW Priority

#### `src/environment/state.py`
Full terastallization state tracking remains wired. Harmless for Gen 3 (tera fields stay at defaults) but represents dead code.

| Line(s) | Issue |
|---------|-------|
| 83–84 | `OwnPokemon` has `tera_type: str = ""`, `terastallized: bool = False` |
| 120–121 | `OpponentPokemon` has `tera_type: str = UNKNOWN`, `terastallized: bool = False` |
| 265–266 | `BattleState` has `can_terastallize: bool = True`, `opponent_has_terastallized: bool = False` |
| 501–505 | JSON parsing for `teraType` and `terastallized` fields |
| 887–901 | `_handle_terastallize()` method |
| 1000 | `MessageType.TERASTALLIZE` handler registration |

#### `src/environment/protocol.py`
| Line(s) | Issue |
|---------|-------|
| 73 | `TERASTALLIZE = "-terastallize"` enum member |
| 423–432 | `parse_terastallize_message()` function |

#### `src/environment/showdown_client.py`
| Line(s) | Issue |
|---------|-------|
| 72 | Example code uses `format="gen9ou"` |
| 170 | Docstring mentions `"gen9ou"` as example format |

#### `scripts/run_phase1_exit_gate.py`
| Line | Issue |
|------|-------|
| 55 | `BATTLE_FORMAT = "gen9randombattle"` — should be `gen3ou` or `gen3randombattle` |

#### `scripts/train_phase4.py`
| Line | Issue |
|------|-------|
| 799 | Help text mentions "terastallized flag" as a dead channel example |

#### `tests/test_state.py`
| Line(s) | Issue |
|---------|-------|
| 79 | Asserts `opp.tera_type == UNKNOWN` |
| 182 | Test data has `"teraType": "Dragon"` |
| 383–418 | `TestTerastallization` class tests tera message handling |
| 411, 417–418 | Tests tera disabling future tera |
| 483, 496 | Test data has `"teraType"` fields |
| 574 | Asserts `opp.tera_type == UNKNOWN` |

#### `tests/test_protocol.py`
| Line(s) | Issue |
|---------|-------|
| 29 | Imports `parse_terastallize_message` |
| 198–200 | `test_terastallize_message()` tests tera message parsing |
| 280–284 | `test_parse_terastallize_message()` tests structured data extraction |

#### `tests/test_parser.py`
| Line(s) | Issue |
|---------|-------|
| 88 | `can_tera` parameter in test helper |
| 110 | `"can_tera": can_tera` in synthetic state dict |
| 250–254 | Test correctly asserts `can_tera` is False (this test is actually Gen 3-correct) |

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

## Summary Statistics

| Category | Count |
|----------|-------|
| Proposal steps fully implemented | 23 of 26 |
| Proposal steps partially done | 3 of 26 (M5.1 config, M5.3 legality, M5.3 bots) |
| Files with Gen 9 artifacts remaining | 14 |
| Critical runtime bugs | 1 (`model_bot.py` TypeError) |
| High-priority files to rewrite | 1 (`generate_sample_data.py`) |
| Medium-priority files to update | 4 |
| Low-priority files (dead code) | 9 |
| Gen 3-specific test cases added | 51 (in `test_gen3_mechanics.py`) |

---

## Recommended Cleanup Order

1. **`src/bots/model_bot.py`** — **CRITICAL**: Remove tera/can_tera kwargs (latent TypeError crash). 5 lines to delete.
2. **`scripts/generate_sample_data.py`** — Rewrite for Gen 3 (blocks sample data generation for pipeline testing).
3. **`configs/environment/showdown.yaml`** — Fix format to `gen3ou`, remove Team Preview rule (blocks live battles).
4. **`configs/training/bc.yaml`** — Fix tags to `gen3ou` (cosmetic but misleading for experiment tracking).
5. **`src/bots/max_damage_bot.py`** — Update move database for Gen 3 (inaccurate baseline evaluation).
6. **`scripts/analyze_p8_dataset_efficiency.py`** — Fix 13→9 action count, remove tera feature indexing, fix out-of-bounds index.
7. **`scripts/run_phase1_exit_gate.py`** — Update battle format to `gen3ou` or `gen3randombattle`.
8. **Environment layer** (`state.py`, `protocol.py`, `showdown_client.py`) — Remove tera dead code.
9. **Tests** (`test_state.py`, `test_protocol.py`, `test_parser.py`) — Remove/update tera-specific tests.
10. **`scripts/train_phase4.py`** — Update help text reference to "terastallized flag".

---

## Hidden Information Doctrine Compliance

The migration preserves and strengthens the Hidden Information Doctrine:

| Rule | Status | Evidence |
|------|--------|----------|
| 1. Never train on omniscient features | **Compliant** | Opponent obs uses only revealed info; unrevealed slots are empty `PokemonObservation` |
| 2. Represent uncertainty with "unknown" markers | **Compliant** | `UNKNOWN` sentinel used for items, abilities; unrevealed opponent slots padded with defaults |
| 3. Metagame priors are soft hints | **Compliant** | Priors rebuilt from Gen 3 data; not leaked into observation space |
| 4. Separate hidden-state inference from move selection | **Compliant** | Auxiliary head remains separate from policy head; no tera category (correctly removed) |
| 5. No team preview makes doctrine more critical | **Enhanced** | `OpponentTracker` builds team incrementally; `num_opponent_revealed` gives explicit uncertainty signal |

---

## Conclusion

The Gen 3 migration is **substantially complete for offline training**. The core pipeline (data → observation → tensorization → model → training → testing → evaluation) is fully Gen 3-compliant and tested. The primary remaining work is cleaning up the live-battle environment layer (M5), which requires fixing 1 critical bug in `model_bot.py`, rewriting `generate_sample_data.py`, and updating config files. The 9 low-priority items are dead code that does not affect correctness.
