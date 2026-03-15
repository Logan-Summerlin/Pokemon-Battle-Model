# Proposal: Migrating the Pokemon Battle Model from Gen 9 OU to Gen 3 OU

_Created: March 15, 2026_

---

## Executive Summary

This document proposes migrating the Pokemon Battle Model pipeline from Gen 9 OU (Scarlet/Violet) to Gen 3 OU (Ruby/Sapphire/Emerald, also known as ADV). Gen 3 presents a more challenging hidden-information problem (no team preview), a simpler action space (no Terastallization), and a more compact metagame (~50-60 viable Pokemon). The Metamon project has demonstrated that Gens 1-4 are tractable for imitation learning and offline RL, with their `jakegrigsby/metamon-parsed-replays` dataset already containing `gen3ou` replay data.

This proposal covers every component that must change, organized as a step-by-step migration plan.

---

## Motivation

### Why Gen 3?

1. **Deeper hidden information**: No team preview means the opponent's entire team is unknown at battle start. This makes the hidden-information doctrine even more central to the model's success. Our architecture's auxiliary head for predicting opponent info becomes more valuable, not less.

2. **Simpler action space**: No Terastallization eliminates 4 action slots, reducing from 13 to 9 canonical actions. This simplifies the policy head and removes an entire category of strategic complexity.

3. **Research alignment with Metamon**: The Metamon paper (Grigsby et al., RLC 2025) specifically chose Gens 1-4 as their primary domain because these are "the most partially observed" competitive settings. Gen 3 replay data is available in their dataset, and their `UniversalState` abstraction already handles Gen 3 mechanics.

4. **Compact metagame**: ~50-60 viable OU Pokemon, ~100 commonly seen moves, and a much smaller item pool (dominated by Leftovers and Choice Band) yield smaller vocabularies and faster iteration.

5. **Interesting strategic depth**: Permanent weather (Tyranitar's Sand Stream), Spikes-only hazard game, the type-based physical/special split, and a rich lead metagame all provide meaningful strategic complexity despite the simpler mechanics.

---

## Scope of Changes

### Summary Table

| Component | Change Magnitude | Key Delta |
|-----------|-----------------|-----------|
| Action space | **Major** | 13 -> 9 actions (remove tera-moves) |
| Observation space | **Major** | Remove tera features; add opponent-unknown tracking for no team preview |
| Replay download | **Moderate** | `gen9ou.tar.gz` -> `gen3ou.tar.gz` |
| Replay parser | **Moderate** | Handle Gen 3 data format from Metamon; no team preview events |
| Vocabularies | **Rebuild** | New species/moves/items/abilities/types for Gen 3 |
| Field observation | **Moderate** | Remove terrain, Aurora Veil, Sticky Web, Toxic Spikes; add permanent weather flag |
| Auxiliary labels | **Moderate** | Remove tera category; update item classes; update move families |
| Transformer model | **Moderate** | Smaller action dim; remove tera features; adjust embedding sizes |
| Environment/Showdown | **Moderate** | Pin to Gen 3 OU format; update legality rules |
| Heuristic bot | **Moderate** | Update damage calc for type-based physical/special split |
| Training scripts | **Minor** | Update paths, hyperparameters, config defaults |
| Tests | **Moderate** | Update all test fixtures for Gen 3 data |

---

## Step-by-Step Migration Plan

### Phase M0: Foundation Changes (Data Layer)

#### Step M0.1: Update Replay Download Pipeline

**File:** `scripts/download_replays.py`

**Changes:**
- Change the HuggingFace download target from `gen9ou.tar.gz` to `gen3ou.tar.gz`
- Update the `download_gen9ou_tar()` function (rename to `download_gen3ou_tar()` or parameterize by generation)
- Verify Elo distribution in Gen 3 data — the 1500+ threshold may need adjustment since Gen 3 ladder population is smaller and older
- Update CLI help text and default values

**Recommendation:** Make the generation a configurable parameter so the pipeline supports both Gen 3 and Gen 9 datasets. This is a small change with high reuse value:
```python
def download_replays(generation: str = "gen3ou", cache_dir: str = "/tmp/metamon_cache") -> str:
    path = hf_hub_download(
        repo_id="jakegrigsby/metamon-parsed-replays",
        filename=f"{generation}.tar.gz",
        ...
    )
```

**Data volume consideration:** Gen 3 likely has fewer replays than Gen 9. Check the total available count — if under 50K, consider lowering the Elo threshold to 1300+ or using all available data. Metamon's dataset has been growing continuously with new scraped replays.

#### Step M0.2: Update Replay Parser for Gen 3 Format

**File:** `src/data/replay_parser.py`

**Changes:**
- Verify that the Metamon Gen 3 replay format matches the Gen 9 format structurally. The Metamon `UniversalState` abstracts across generations, so the `.json.lz4` file structure should be consistent. However:
  - `can_tera` will always be `false`
  - `tera_type` fields will be absent or empty
  - `opponent_teampreview` will be **empty** (no team preview in Gen 3)
  - Weather may have different duration semantics (permanent vs 5-turn)
  - There will be no terrain field data (terrains were introduced in Gen 6)
- Add a generation field to `ParsedBattle` if not already present
- Handle the case where `opponent_teampreview` is empty gracefully — this is the core "no team preview" change

**Key insight from Metamon:** Their `UniversalState` already handles Gen 3's physical/special type-based split at the move level, encoding each move's damage category as determined by the generation's rules. The `ParsedPokemon` move data should already contain the correct `category` field.

#### Step M0.3: Rebuild Vocabularies

**File:** `src/data/tensorizer.py`, `data/processed/vocabs/`

**Changes:**
- Run `process_dataset.py` on Gen 3 data to build new vocabulary files
- Expected vocabulary sizes will be significantly smaller:

| Vocabulary | Gen 9 OU (current) | Gen 3 OU (estimated) |
|-----------|--------------------|--------------------|
| Species | ~600 | ~100-150 |
| Moves | ~600 | ~200-300 |
| Items | ~200 | ~30-50 |
| Abilities | ~300 | ~80-120 |
| Types | ~200 (type combos) | ~150 (no Fairy combos) |
| Status | ~20 | ~15 |
| Weather | ~20 | ~10 |
| Terrain | ~20 | ~2 (none + padding only) |

- Update `TransformerConfig` default vocab sizes to match Gen 3 scale
- The `Vocabulary` class itself needs no changes — it builds dynamically from data

---

### Phase M1: Observation Space Changes

#### Step M1.1: Remove Terastallization from Observation

**File:** `src/data/observation.py`

**Changes to `PokemonObservation`:**
- Remove `tera_type: str = UNKNOWN` field (or keep for compatibility but always set to empty/unused)
- Remove `terastallized: bool = False` field
- The `tera_type` and `terastallized` fields in the binary feature vector become dead features

**Changes to `TurnObservation`:**
- Remove `can_tera: bool = False` (or always set to `False`)
- The `game_phase` field no longer needs `"team_preview"` as a valid state for Gen 3

**Changes to `build_observations()`:**
- When processing Gen 3 data, `can_tera` is always `False`
- The opponent tracker's `revealed_tera` tracking is unused but harmless

#### Step M1.2: Handle No Team Preview (Critical Change)

**File:** `src/data/observation.py`

This is the **most architecturally significant change** in the migration.

**Current behavior (Gen 9):** The `build_observations()` function populates `opponent_team_obs` from:
1. `turn.opponent_active` (the currently active opponent Pokemon)
2. `turn.opponent_teampreview` (all 6 opponent species visible from team preview)

This means in Gen 9, all 6 opponent token slots are populated from turn 1 with at least species information.

**Gen 3 behavior:** There is no team preview. The opponent team is built up incrementally:
- Turn 1: Only the opponent's lead is known (1 Pokemon visible)
- As the battle progresses, opponent Pokemon are revealed when switched in
- Some opponent Pokemon may never be revealed if they're never sent out

**Required changes to `build_observations()`:**

```python
# Replace the team preview section with revealed-pokemon tracking
class OpponentTracker:
    def __init__(self) -> None:
        ...
        # NEW: Track which opponent Pokemon have been revealed (switched in)
        self.revealed_species: list[str] = []  # Ordered by first appearance

    def update_from_turn(self, turn: ParsedTurnState) -> None:
        ...
        # Track new opponent Pokemon reveals
        if species and species not in self.revealed_species:
            self.revealed_species.append(species)
```

**Building the opponent team observation (Gen 3):**
```python
# Instead of iterating opponent_teampreview, iterate revealed_species
for species in tracker.revealed_species:
    if species == active_species:
        continue  # Already added as active
    # Find this pokemon's last known state from previous turns
    opponent_team_obs.append(
        _pokemon_to_opponent_observation(
            last_known_state[species], is_active=False, ...
        )
    )
# Remaining slots (unrevealed Pokemon) stay as empty PokemonObservation
```

**Hidden information impact:** In Gen 3, the number of opponent Pokemon remaining (`opponents_remaining`) becomes more critical context because the model cannot see the full opponent roster. The model must infer:
- How many opponent Pokemon have been revealed vs. how many remain hidden
- What types/roles the hidden Pokemon might fill based on team composition patterns

**New context features to consider adding:**
- `num_opponent_revealed: int` — how many distinct opponent Pokemon have been seen
- `num_opponent_remaining: int` — how many opponent Pokemon are still alive (already exists)
- The gap between these tells the model how many Pokemon it hasn't seen yet

#### Step M1.3: Update Field Observation for Gen 3

**File:** `src/data/observation.py`

**Remove or zero out features that don't exist in Gen 3:**

| Feature | Gen 9 | Gen 3 | Action |
|---------|-------|-------|--------|
| `terrain` | Electric/Grassy/Misty/Psychic | Does not exist | Always empty |
| `own_sticky_web` / `opp_sticky_web` | Yes | Does not exist (Gen 6) | Always 0 |
| `own_toxic_spikes` / `opp_toxic_spikes` | Yes (Gen 4) | Does not exist | Always 0 |
| `own_aurora_veil` / `opp_aurora_veil` | Yes (Gen 7) | Does not exist | Always 0 |
| Stealth Rock | Yes (Gen 4) | Does not exist | Always 0 |
| Weather | 5-turn (ability) / 8-turn (item) | **Permanent** (ability) | Keep weather field; add `weather_permanent: bool` flag |
| `own_spikes` / `opp_spikes` | 0-3 layers | 0-3 layers | Keep as-is |
| `own_reflect` / `opp_reflect` | Yes | Yes | Keep as-is |
| `own_light_screen` / `opp_light_screen` | Yes | Yes | Keep as-is |
| `own_tailwind` / `opp_tailwind` | Yes (Gen 4) | Does not exist | Always 0 |

**Recommended approach:** Rather than removing fields (which would change tensor dimensions and break compatibility), set them to their "absent" value (0 or empty). Enable `prune_dead_features` to handle the always-zero features. This approach:
- Minimizes code changes
- Lets the model naturally learn that these features are inactive
- Preserves the option to port back to Gen 9 later

**Alternative approach (cleaner but more work):** Create a `Gen3FieldObservation` dataclass with only the relevant fields, and adjust `FIELD_FEATURE_DIM` accordingly. This would reduce:
- Terrain: 1 categorical -> removed (save 1 dim)
- Stealth Rock, Toxic Spikes, Sticky Web, Aurora Veil, Tailwind: 5 binary per side -> removed (save 10 dims)
- Net field dim: from 18 to ~8

---

### Phase M2: Action Space Changes

#### Step M2.1: Reduce Action Space from 13 to 9

**File:** `src/environment/action_space.py`

**Current (Gen 9):**
```
0-3:  move 1-4
4-7:  move 1-4 + tera
8-12: switch to bench slot 0-4
```

**Gen 3:**
```
0-3:  move 1-4
4-8:  switch to bench slot 0-4
```

**Changes:**
- Remove `MOVE_1_TERA` through `MOVE_4_TERA` constants
- Remove `ActionType.MOVE_TERA`
- Update `NUM_ACTIONS = 9`
- Update `ACTION_NAMES` list
- Update `SWITCH_2` through `SWITCH_6` to indices 4-8
- Update `BattleAction.canonical_index` property
- Update `BattleAction.to_showdown_command()` (remove terastallize option)
- Update `action_from_canonical_index()` and `action_from_showdown_choice()`

**Metamon action mapping update in `tensorizer.py`:**
```python
# Metamon Gen 3: 0-3 = moves, 4-8 = switches (bench 0-4)
# Ours Gen 3:    0-3 = moves, 4-8 = switches (slot 2-6)
_METAMON_TO_CANONICAL_GEN3 = {
    0: 0, 1: 1, 2: 2, 3: 3,       # moves 1-4 -> moves 1-4
    4: 4, 5: 5, 6: 6, 7: 7, 8: 8, # switch bench 0-4 -> switch 2-6
}
```

#### Step M2.2: Update Legal Mask Builder

**File:** `src/data/observation.py`

**Changes to `_build_legal_mask()`:**
- Remove all tera-related logic (`if turn.can_tera: mask[4 + i] = True`)
- Adjust switch indices from `8 + i` to `4 + i`
- Mask size changes from 13 to 9

---

### Phase M3: Tensorization and Feature Changes

#### Step M3.1: Update Pokemon Feature Vector

**File:** `src/data/tensorizer.py`

**Current per-pokemon feature vector (30 dims):**
- 9 categorical: species(1) + moves(4) + item(1) + ability(1) + types(1) + status(1)
- 14 continuous: hp_frac(1) + boosts(7) + base_stats(6)
- 7 binary: is_active(1) + is_fainted(1) + is_own(1) + is_unknown_item(1) + is_unknown_ability(1) + is_unknown_tera(1) + terastallized(1)

**Gen 3 per-pokemon feature vector (28 dims):**
- 9 categorical: species(1) + moves(4) + item(1) + ability(1) + types(1) + status(1) *(unchanged)*
- 14 continuous: hp_frac(1) + boosts(7) + base_stats(6) *(unchanged)*
- 5 binary: is_active(1) + is_fainted(1) + is_own(1) + is_unknown_item(1) + is_unknown_ability(1)
  - Remove: `is_unknown_tera` (no tera in Gen 3)
  - Remove: `terastallized` (no tera in Gen 3)

**Update constants:**
```python
POKEMON_BINARY_DIM = 5  # was 7
POKEMON_FEATURE_DIM = 9 + 14 + 5  # = 28, was 30
```

**Alternative (minimal changes):** Keep the 30-dim vector and always set the last 2 binary features to 0. Use `prune_dead_features = True` to drop them. This is the current approach for the `terastallized` flag already.

#### Step M3.2: Update Context Feature Vector

**File:** `src/data/tensorizer.py`

**Current context (6 dims):**
- turn_number(1), opponents_remaining(1), can_tera(1), forced_switch(1), prev_player_move(1), prev_opponent_move(1)

**Gen 3 context (proposed 7 dims):**
- turn_number(1), opponents_remaining(1), **num_opponent_revealed(1)**, forced_switch(1), prev_player_move(1), prev_opponent_move(1), **is_lead_turn(1)**
  - Remove: `can_tera` (always false)
  - Add: `num_opponent_revealed` — critical for no-team-preview reasoning (normalized by 6)
  - Add: `is_lead_turn` — binary flag for turn 1 (lead matchup is a distinct sub-game in Gen 3)

**Update constant:** `CONTEXT_FEATURE_DIM = 7`

#### Step M3.3: Update Field Feature Tensorization

**File:** `src/data/tensorizer.py`

If taking the "clean" approach from Phase M1 Step 3, update `FIELD_FEATURE_DIM` and `tensorize_field()` to only encode Gen 3-relevant features. If taking the "preserve dimensions" approach, no changes needed here beyond ensuring dead features are zero.

**Recommend adding:** `weather_is_permanent: bool` binary feature — in Gen 3, ability-set weather is permanent, which is a major strategic difference the model should be aware of. (In practice, if only training on Gen 3, this will always be true for ability-set weather, but it's a useful signal.)

---

### Phase M4: Model Architecture Changes

#### Step M4.1: Update BattleTransformer

**File:** `src/models/battle_transformer.py`

**PolicyHead:** Update output dimension from `NUM_ACTIONS=13` to `NUM_ACTIONS=9`.

**PokemonEmbedding:** If reducing `POKEMON_BINARY_DIM`:
- Update `cont_binary_dim` calculation
- The `prune_dead_features` logic already handles this if we keep the 30-dim vector

**FieldEmbedding:** If reducing `FIELD_FEATURE_DIM`, update the input dimension.

**ContextEmbedding:** Update for new context dims:
```python
# Gen 3: turn_num(0), opp_remaining(1), num_opp_revealed(2),
#         forced_switch(3), prev_player_move(4), prev_opponent_move(5), is_lead_turn(6)
cont_dim = 5  # was 4 (removed can_tera, added num_opp_revealed, is_lead_turn)
```

**TransformerConfig defaults for Gen 3:**
```python
# Smaller vocabularies -> smaller embedding tables
species_vocab_size: int = 200    # was 600
moves_vocab_size: int = 400      # was 600
items_vocab_size: int = 80       # was 200
abilities_vocab_size: int = 150  # was 300
types_vocab_size: int = 150      # was 200 (no Fairy combos)
terrain_vocab_size: int = 5      # was 20 (effectively unused)

# No tera category in auxiliary head
num_tera_categories: int = 0     # was 4
```

#### Step M4.2: Update Auxiliary Head

**File:** `src/models/battle_transformer.py`

**Remove tera category prediction:**
- Remove `num_tera_categories` from `TransformerConfig` (or set to 0)
- Remove `tera_logits` from `AuxiliaryHead.forward()` output
- The `compute_auxiliary_loss()` function already skips missing keys, so removing `tera_logits` from the predictions dict is sufficient

**Update item classes for Gen 3:**
The current 50-class item taxonomy is heavily Gen 9-oriented. Gen 3 items require a new taxonomy:

```python
GEN3_ITEM_CLASSES = [
    "leftovers",        # 0 - by far the most common
    "choiceband",       # 1
    "lumberry",         # 2
    "liechiberry",      # 3 (pinch berries)
    "petayaberry",      # 4
    "salacberry",       # 5
    "magoberry",        # 6
    "sitrusberry",      # 7
    "focusband",        # 8
    "shellbell",        # 9
    "whiteherb",        # 10
    "mentalherb",       # 11
    "leppaberry",       # 12
    "cheriberry",       # 13 (status cure berries)
    "chestoberry",      # 14
    "soothebell",       # 15
    "nevermeltice",     # 16 (type-boosting items)
    "charcoal",         # 17
    "mysticwater",      # 18
    "magnet",           # 19
    "blackbelt",        # 20
    "twistedspoon",     # 21
    "metalcoat",        # 22
    "other",            # 23
    "noitem",           # 24
]
```

This reduces `NUM_ITEM_CLASSES` from 50 to ~25.

**Update speed buckets:** Gen 3 base speed distribution differs from Gen 9. Suggested thresholds based on Gen 3 OU metagame:
- Very fast (>=110): Aerodactyl (130), Starmie (115), Jolteon (130), Dugtrio (120)
- Fast (90-109): Gengar (110), Salamence (100), Celebi (100), Jirachi (100)
- Medium (65-89): Tyranitar (61->bump up), Suicune (85), Metagross (70)
- Slow (40-64): Swampert (60), Skarmory (70->medium), Snorlax (30->very slow)
- Very slow (<40): Blissey (55->slow)

Adjusted thresholds for Gen 3: `SPEED_THRESHOLDS = [110, 90, 65, 40]`

**Update move families:** Mostly compatible, but some moves in the current family lists don't exist in Gen 3:
- Priority: Remove `grassyglide`, `jetpunch`. Keep `extremespeed`, `machpunch`, `quickattack`, `fakeout`, `shadowsneak`(Gen 4, remove), `suckerpunch`(Gen 4, remove). Gen 3 priority moves: ExtremeSpeed, Mach Punch, Quick Attack, Fake Out
- Hazard setup: Remove `stealthrock`, `toxicspikes`, `stickyweb`. Keep only `spikes`
- Hazard removal: Remove `defog`, `courtchange`, `tidyup`, `mortalspin`. Keep only `rapidspin`
- Pivot moves: Remove `flipturn`(Gen 8). `uturn`(Gen 4), `voltswitch`(Gen 5) **do not exist in Gen 3**. Gen 3 pivot options: `batonpass` only
- Screens: Remove `auroraveil`. Keep `reflect`, `lightscreen`
- Setup: Remove `tidyup`, `victorydance`. Keep `swordsdance`, `dragondance`, `calmmind`, `bulkup`, `agility`, `curse`, `bellydrum`
- Phazing: Keep `whirlwind`, `roar`, `haze`. Remove `dragontail`(Gen 5), `circlethrow`(Gen 5)

#### Step M4.3: Update P8-Lean Config for Gen 3

**File:** `src/models/battle_transformer.py`

Create a `p8_lean_gen3()` class method:
```python
@classmethod
def p8_lean_gen3(cls, vocabs=None, **kwargs):
    """P8-Lean profile tuned for Gen 3 OU."""
    base = dict(
        num_layers=3,
        hidden_dim=192,         # Slightly smaller (fewer entities)
        num_heads=4,
        ffn_multiplier=3,
        use_value_head=False,
        prune_dead_features=True,
        species_embedding_dim=48,
        move_embedding_dim=24,
        item_embedding_dim=12,  # Fewer items
        ability_embedding_dim=12,  # Fewer abilities
        type_embedding_dim=12,
        max_seq_len=5,
        num_item_classes=25,    # Gen 3 item taxonomy
        num_tera_categories=0,  # No tera
    )
    ...
```

The smaller vocab sizes should lead to ~1.2-1.5M parameters (down from ~1.95M for Gen 9 P8-Lean), making training faster.

---

### Phase M5: Environment and Bot Changes

#### Step M5.1: Update Showdown Server Configuration

**File:** `src/environment/showdown_client.py`, `src/environment/battle_env.py`

**Changes:**
- Pin the Showdown server to a version that supports Gen 3 OU format
- Change the format string from `"gen9ou"` to `"gen3ou"`
- Remove team preview handling from the battle client
- Gen 3 battles start directly with lead selection, no preview phase

#### Step M5.2: Update Heuristic Bot for Gen 3 Mechanics

**File:** `src/bots/heuristic_bot.py`

**Critical change — type-based physical/special split:**

In Gen 3, whether a move is physical or special depends on its **type**, not the individual move. The heuristic bot's damage estimation must use:

```python
# Gen 3 physical types
PHYSICAL_TYPES = {"Normal", "Fighting", "Poison", "Ground", "Flying", "Bug", "Rock", "Ghost", "Steel"}

# Gen 3 special types
SPECIAL_TYPES = {"Fire", "Water", "Grass", "Electric", "Ice", "Psychic", "Dragon", "Dark"}

def get_damage_stats_gen3(move_type: str, attacker, defender):
    """Return (attack_stat, defense_stat) based on move type for Gen 3."""
    if move_type in PHYSICAL_TYPES:
        return attacker.atk, defender.def_stat
    else:
        return attacker.spa, defender.spd
```

This profoundly affects which Pokemon are viable and which moves are useful:
- Gyarados's Water moves use SpA (which is low), so it relies on physical moves like Earthquake and Hidden Power Flying
- Gengar's Shadow Ball is physical (Ghost type), but Gengar has high SpA/low Atk, so Shadow Ball is weak on Gengar
- Tyranitar's Crunch is special (Dark type), benefiting from SpA investment

**Other heuristic changes:**
- Remove tera-related logic from move selection
- Update type effectiveness chart (no Fairy type in Gen 3; Steel resists Dark and Ghost)
- Account for permanent weather in damage calculations (Sand Stream Tyranitar boosts SpD of Rock-types by 50% permanently)
- Explosion/Self-Destruct halve the target's Defense in Gen 3 (effective base powers of 500/400)

#### Step M5.3: Update Legality Rules

**File:** `src/environment/legality.py`

- Remove all tera-related legality checks
- Update for Gen 3 move legality (different TM/HM compatibility, move tutors)
- Handle Gen 3-specific mechanics: Choice Band lock (no Choice Specs/Scarf)
- Sleep Clause: Only one Pokemon can be put to sleep at a time (same as Gen 9)
- Species Clause: No duplicate species (same as Gen 9)
- Remove checks for items that don't exist (Heavy-Duty Boots, Life Orb, etc.)

---

### Phase M6: Training Pipeline Changes

#### Step M6.1: Update Data Processing Script

**File:** `scripts/process_dataset.py`

- Update default data paths for Gen 3
- Rebuild vocabularies from Gen 3 data
- Rebuild priors (`data/processed/priors.json`) from Gen 3 usage statistics
- Regenerate base stats crosswalk for Gen 3 Pokemon (use Gen 3 base stat values, which differ from later gens for some Pokemon)

#### Step M6.2: Update Training Scripts

**Files:** `scripts/train_phase4.py`, `scripts/train_p8_lean.py`

- Update default config values for Gen 3 model dimensions
- Update checkpoint directory names (e.g., `checkpoints/phase4_gen3_p8_lean/`)
- Consider reducing `max_window` — Gen 3 battles tend to be longer (more defensive meta), so either:
  - Increase window size to 10-15 (capture more context from longer battles)
  - Keep at 5 for compute efficiency and accept less context

**Hyperparameter considerations for Gen 3:**
- Smaller dataset may benefit from stronger regularization (higher dropout: 0.15-0.2)
- Auxiliary loss weight could be **increased** to 0.3 since hidden info is more critical in Gen 3
- Batch size may need reduction if dataset is significantly smaller
- Consider class-weighted loss for actions since the move distribution may be more skewed (Leftovers usage is dominant)

#### Step M6.3: Update Auxiliary Label Extraction

**File:** `src/data/auxiliary_labels.py`

- Replace `ITEM_CLASSES` and `ITEM_CLASS_MAP` with Gen 3 item taxonomy
- Update `SPEED_THRESHOLDS` for Gen 3 speed tiers
- Remove tera category classification entirely
- Update move family classifications for Gen 3 move pools (see M4.2)
- Role archetypes remain largely the same but some nuance:
  - "Pivot" in Gen 3 is more about Baton Pass than U-turn (which doesn't exist)
  - "Hazard setter" is Spikes only
  - Add consideration for "Trapper" role (Dugtrio with Arena Trap, Magneton with Magnet Pull)

---

### Phase M7: Testing Updates

#### Step M7.1: Update Test Fixtures

**Files:** All files in `tests/`

- Update test Pokemon, moves, items to Gen 3 examples:
  - Replace Garchomp/Dragapult -> Salamence/Metagross
  - Replace Heavy-Duty Boots -> Leftovers
  - Replace Terastallization tests -> simple move tests
  - Replace Stealth Rock tests -> Spikes tests
- Update action space tests for 9 actions instead of 13
- Add tests for no-team-preview opponent tracking
- Add tests for type-based physical/special classification

#### Step M7.2: New Tests for Gen 3-Specific Logic

- Test that the physical/special type split is correctly applied in damage estimation
- Test that permanent weather is handled correctly
- Test that opponent Pokemon are correctly tracked without team preview
- Test that the legal mask correctly excludes tera options
- Test that the Explosion/Self-Destruct Defense-halving mechanic is modeled

---

### Phase M8: Evaluation and Baseline Recalibration

#### Step M8.1: Update Evaluation Targets

**File:** `docs/EVALUATION_SPEC.md`

Gen 3 targets may differ from Gen 9:
- Win rate vs. random bot: still target >95%
- Win rate vs. heuristic bot: still target >70%
- Action accuracy may be **higher** on Gen 3 (smaller action space, more predictable meta)
- Hidden-info prediction accuracy may be **lower** initially (harder problem with no team preview)

**New Gen 3-specific evaluation dimensions:**
- Lead prediction accuracy: Given the opponent's first Pokemon, can the model predict likely teammates?
- Scouting efficiency: Does the model make switches to reveal opponent team composition at appropriate times?
- Weather exploitation: Does the model correctly play around permanent sandstorm?

#### Step M8.2: Update Archetype Stratification

Gen 3 OU archetypes differ from Gen 9:
- **TSS (Toxic/Spikes/Sandstorm):** The signature Gen 3 stall strategy
- **Bulky Offense:** The dominant balanced playstyle
- **Hyper Offense:** Dragon Dance sweepers, mixed attackers
- **Stall:** Full stall teams (Blissey/Skarmory/Milotic core)
- **Weather teams:** Rain Dance teams, Sunny Day teams (manual weather, not ability-based except Sand Stream)
- **Baton Pass teams:** Chain Baton Pass was a legitimate strategy in Gen 3

---

## Migration Order and Dependencies

```
M0.1 (Download) ──> M0.2 (Parser) ──> M0.3 (Vocabs)
                                            │
                    ┌───────────────────────┤
                    ▼                       ▼
              M1.1 (Obs/Tera)        M2.1 (Action Space)
              M1.2 (No Preview)      M2.2 (Legal Mask)
              M1.3 (Field)                 │
                    │                      │
                    ▼                      ▼
              M3.1 (Pokemon Feats)   M3.2 (Context Feats)
              M3.3 (Field Feats)
                    │
                    ▼
              M4.1 (Transformer)
              M4.2 (Aux Head)
              M4.3 (P8-Lean Config)
                    │
        ┌───────────┼───────────┐
        ▼           ▼           ▼
  M5.1 (Showdown) M5.2 (Bot)  M6.1 (Process Data)
  M5.3 (Legality)              M6.2 (Train Scripts)
                                M6.3 (Aux Labels)
                                     │
                                     ▼
                               M7 (Tests)
                                     │
                                     ▼
                               M8 (Evaluation)
```

**Estimated effort:** ~2-3 weeks of focused development for a working Gen 3 pipeline, assuming familiarity with the existing codebase. The largest time investments are:
1. Verifying the Metamon Gen 3 data format and parser compatibility (~3-4 days)
2. Implementing no-team-preview opponent tracking (~2-3 days)
3. Updating and testing the action space and legal mask (~2 days)
4. Rebuilding vocabularies, item taxonomy, and auxiliary labels (~2-3 days)
5. Running initial training experiments and debugging (~3-4 days)

---

## Risks and Mitigations

| Risk | Mitigation |
|------|-----------|
| Gen 3 replay data volume may be insufficient | Lower Elo threshold; use all available data; consider data augmentation via team/move reordering |
| Metamon Gen 3 replay format may differ from Gen 9 | Validate format early (M0.2); write format tests before proceeding |
| Permanent weather makes battles longer/more repetitive | May actually help — more turns = more training signal per battle |
| Smaller metagame may cause overfitting | Stronger regularization; monitor train/val gap closely; consider label smoothing |
| No team preview makes auxiliary head harder to train | Increase aux loss weight; consider curriculum learning (first train with more revealed info, then reduce) |
| Type-based physical/special split confuses damage estimation | Explicit encoding in observation space; ensure heuristic bot models it correctly for baseline comparison |
| Gen 3 community is smaller — fewer experts to validate model behavior | Leverage Smogon ADV OU resources; consider consulting competitive ADV players |

---

## Compatibility and Reversibility

This proposal recommends making the generation a configurable parameter where possible, rather than hard-coding Gen 3 everywhere. Key design patterns:

1. **Parameterize the download script** by generation string
2. **Keep the `PokemonObservation` dataclass broad** — unused fields (tera) are set to defaults
3. **Use `prune_dead_features`** to handle always-zero features rather than changing tensor dimensions
4. **Version vocabulary files** by generation (e.g., `vocabs/gen3/`, `vocabs/gen9/`)
5. **Make action space generation-aware** via a config flag

This approach allows switching between generations with minimal code changes, supporting future multi-generation experiments as demonstrated by the Metamon project.

---

## References

1. Grigsby, J., Xie, Y., Sasek, J., Zheng, S., & Zhu, Y. (2025). "Human-Level Competitive Pokemon via Scalable Offline Reinforcement Learning with Transformers." RLC 2025. [arXiv:2504.04395](https://arxiv.org/abs/2504.04395)
2. Metamon GitHub: [github.com/UT-Austin-RPL/metamon](https://github.com/UT-Austin-RPL/metamon)
3. Metamon Dataset: [huggingface.co/jakegrigsby/metamon-parsed-replays](https://huggingface.co/datasets/jakegrigsby/metamon-parsed-replays)
4. Smogon ADV OU Resources: [smogon.com/dex/rs/formats/ou/](https://www.smogon.com/dex/rs/formats/ou/)
5. Pokemon Showdown: [play.pokemonshowdown.com](https://play.pokemonshowdown.com)
