# Phase 4 Feature Analysis: Perfecting Imitation Learning

## Executive Summary

This document analyzes the current 384-dimensional feature set used by the Phase 4 transformer model and proposes **16 categories of feature enhancements** that would substantially improve behavior cloning performance. The analysis is grounded in how competitive Pokemon players actually make decisions, the information available in the Metamon replay format, and the Hidden Information Doctrine.

The highest-impact additions are:
1. **Move properties** (power, type, category, accuracy, priority) — currently discarded entirely
2. **Type effectiveness matchup features** — the single most fundamental competitive concept
3. **Opponent base stats from Pokedex lookup** — public knowledge, not hidden information
4. **Speed comparison features** — "do I outspeed?" drives nearly every decision
5. **Metagame prior integration** — already computed but never used in tensorization
6. **Damage estimation features** — what expert players mentally calculate every turn

---

## Current Feature Inventory

### Per-Pokemon Features (30 dims × 12 pokemon = 360 dims)

| Category | Features | Dims | Notes |
|----------|----------|------|-------|
| Categorical | species, move1-4, item, ability, types, status | 9 | Encoded as vocab indices |
| Continuous | hp_fraction, 7 stat boosts, 6 base stats | 14 | Normalized floats |
| Binary | is_active, is_fainted, is_own, unknown_item/ability/tera, terastallized | 7 | 0.0 or 1.0 |

### Field Features (18 dims)

| Category | Features | Dims |
|----------|----------|------|
| Categorical | weather, terrain | 2 |
| Binary/Continuous | 8 own-side + 8 opp-side conditions | 16 |

### Context Features (6 dims)

| Category | Features | Dims |
|----------|----------|------|
| Continuous | turn_number/100, opponents_remaining/6 | 2 |
| Binary | can_tera, forced_switch | 2 |
| Categorical | prev_player_move, prev_opponent_move | 2 |

**Total: 384 dims flat, or 14 tokens × hidden_dim in transformer**

---

## Feature Gap Analysis

### Gap 1: Move Properties Are Completely Discarded (CRITICAL)

**Current state:** The Metamon `ParsedMove` has 8 fields: `name`, `move_type`, `category`, `base_power`, `accuracy`, `priority`, `current_pp`, `max_pp`. During observation construction, **only `name` is kept**. All other properties are thrown away.

**Impact:** The model must learn from the name embedding alone that "Earthquake" is a 100BP Ground-type Physical move with 100% accuracy and 0 priority. While learned embeddings can theoretically capture this, it requires the model to memorize ~600 move entries and their properties purely from usage patterns — a massive waste of learning capacity.

**What competitive players know:** Players instantly evaluate moves by their power, type, accuracy, and priority. "Is this a strong STAB move?" and "Does this have priority?" are the first questions.

**Proposed features per move slot (7 dims × 4 moves = 28 additional dims per Pokemon):**

| Feature | Type | Normalization | Source |
|---------|------|---------------|--------|
| base_power | continuous | /150 (caps most moves to [0, 1]) | ParsedMove.base_power |
| accuracy | continuous | /100 | ParsedMove.accuracy |
| priority | continuous | /5 (range [-7, +5] → [-1.4, 1.0]) | ParsedMove.priority |
| category | categorical (3) | physical=0, special=1, status=2 | ParsedMove.category |
| move_type | categorical (18) | type vocab index | ParsedMove.move_type |
| pp_ratio | continuous | current_pp / max_pp | ParsedMove fields |
| is_stab | binary | 1 if move_type matches pokemon type | Derived |

**Implementation notes:**
- Carry `ParsedMove` objects through to `PokemonObservation` instead of just names
- Add move properties to `tensorize_pokemon()` after existing move name indices
- For opponent's unrevealed moves, these fields would be 0/PAD (consistent with unknown markers)
- STAB is derivable by comparing `move_type` to `pokemon.types` — not hidden information

**Estimated impact: HIGH.** This is the single largest information loss in the current pipeline. Every competitive decision involves reasoning about move properties.

---

### Gap 2: Type Effectiveness Matchup Features (CRITICAL)

**Current state:** The model receives type embeddings for each Pokemon but must learn all type effectiveness relationships purely from gameplay outcomes. There are 18 types × 18 types = 324 type matchups, many with 0.25x or 4x multipliers from dual typing.

**What competitive players know:** Type matchups are the absolute foundation of competitive Pokemon. "Is this super effective?" is evaluated instantly for every move consideration.

**Proposed features (per active matchup, ~12 dims):**

| Feature | Type | Description |
|---------|------|-------------|
| best_move_effectiveness | continuous | Max type effectiveness of our moves vs opponent active |
| worst_defensive_matchup | continuous | Worst type effectiveness of opponent's known moves vs us |
| move1_effectiveness | continuous | Type multiplier of move 1 vs opponent active (0, 0.25, 0.5, 1, 2, 4) |
| move2_effectiveness | continuous | Same for move 2 |
| move3_effectiveness | continuous | Same for move 3 |
| move4_effectiveness | continuous | Same for move 4 |
| our_weakness_count | continuous | Number of types we're weak to (normalized /6) |
| our_resistance_count | continuous | Number of types we resist (normalized /10) |
| opponent_weakness_count | continuous | Same for opponent |
| opponent_resistance_count | continuous | Same for opponent |
| has_super_effective_move | binary | Do we have at least one SE move? |
| opponent_has_se_move | binary | Does opponent have a revealed SE move vs us? |

**Implementation notes:**
- Requires a type effectiveness lookup table (18×18 matrix) — a static game data constant
- Dual typing handled by multiplying: `eff = type_chart[move_type][target_type1] * type_chart[move_type][target_type2]`
- For unrevealed opponent moves, `opponent_has_se_move` should use metagame priors (soft probability)
- This is NOT hidden information — type matchups are deterministic from visible types

**Estimated impact: VERY HIGH.** Type effectiveness is the #1 factor in competitive decision-making.

---

### Gap 3: Opponent Base Stats from Pokedex (CRITICAL)

**Current state:** Opponent base stats are **always 0** (6 dims of zeros per opponent Pokemon). The reasoning was the Hidden Information Doctrine — opponent stats are "unknown."

**Why this is wrong:** Base stats are species-intrinsic, publicly available Pokedex data. Once you know the opponent is "Dragapult" (which you always do from team preview), you know it has 142 base Speed, 120 base SpA, etc. **Every competitive player has this memorized.** Base stats ≠ exact stat values (which depend on EVs/IVs/nature — those ARE hidden), but base stats are universal knowledge.

**The distinction:**
- **Base stats** (PUBLIC): Dragapult always has 88/120/75/100/75/142 base stats. Known from species identification.
- **Exact stats** (HIDDEN): Dragapult's actual Speed stat depends on EVs (0-252), IVs (0-31), nature (0.9x, 1.0x, or 1.1x). These are hidden.
- **Base stats provide the range:** A 142 base Speed Pokemon will always be "very fast." The exact number varies by ~30% based on investment, but the tier is known.

**Proposed change:**
- Build a species → base_stats lookup table from Pokedex data (or derive from own-team data in replays)
- When tensorizing opponent Pokemon, fill in base stats from species identification instead of zeros
- This uses the same 6 dims already allocated (hp, atk, def, spa, spd, spe), just populated instead of zeroed

**Implementation notes:**
- Can be built from the training data: for each species seen on own teams, record its base stats
- Or use a static `pokedex.json` file
- Normalize the same way as own stats: divide by 255

**Estimated impact: VERY HIGH.** This gives the model fundamental knowledge about what each opponent Pokemon is capable of. Currently 36 dims per turn (6 opponent pokemon × 6 stats) are wasted zeros.

---

### Gap 4: Speed Comparison Features (HIGH)

**Current state:** No explicit speed comparison. The model has own base Speed stats and (currently zero) opponent base Speed stats, but must learn to compare them and factor in boosts, abilities, items, weather, and Trick Room.

**What competitive players think:** "Do I outspeed?" is one of the first questions every turn. It determines whether to attack or switch, whether priority matters, and whether you can revenge kill.

**Available but unused data:** `TurnAction.speed_order` records `"us_first"` or `"them_first"` from previous turns — directly observable information that is NOT tensorized.

**Proposed features (4-6 dims in context):**

| Feature | Type | Description |
|---------|------|-------------|
| prev_speed_order | binary | Did we move first last turn? (from TurnAction) |
| speed_advantage_estimate | continuous | base_spe_own / (base_spe_own + base_spe_opp) |
| speed_boosted | binary | Do we have positive speed boosts? |
| speed_debuffed | binary | Does opponent have negative speed boosts? |
| trick_room_active | binary | Is Trick Room active? (reverses speed) |
| priority_available | binary | Do we have a priority move? (from move properties) |

**Implementation notes:**
- `speed_order` is already tracked in `BattleState.turn_history` but never reaches observations
- Speed advantage estimate uses base stats (which we propose filling in for opponents via Gap 3)
- Priority availability is a derived feature from move properties (Gap 1)

**Estimated impact: HIGH.** Speed comparison is one of the most important strategic considerations, directly informing attack vs. switch decisions.

---

### Gap 5: Metagame Prior Features (HIGH)

**Current state:** `MetagamePriors` are computed from training data with rich distributions (item/ability/move probabilities per species), but they are stored separately and **never incorporated into tensorization**. The priors.py docstring says they're for "soft prior features" but the feature doesn't exist yet.

**What competitive players know:** Strong players have extensive metagame knowledge. When they see a Kingambit, they immediately think "probably Swords Dance + Sucker Punch + Iron Head, likely Black Glasses or Life Orb, Defiant ability."

**Proposed features per opponent Pokemon (6-10 dims):**

| Feature | Type | Description |
|---------|------|-------------|
| likely_item_confidence | continuous | Prior probability of most likely item |
| item_diversity | continuous | Entropy of item distribution (high = many viable items) |
| likely_ability_confidence | continuous | Prior probability of most likely ability |
| unrevealed_move_threat | continuous | Max base power of likely unrevealed moves |
| has_likely_priority | continuous | Prior probability that species carries a priority move |
| has_likely_recovery | continuous | Prior probability that species carries recovery |
| has_likely_hazards | continuous | Prior probability that species has hazard moves |
| setup_sweeper_prob | continuous | Prior probability that species is a setup sweeper |
| choice_item_prob | continuous | Prior probability of carrying a choice item |
| scarf_prob | continuous | Prior probability of Choice Scarf (speed matters!) |

**Implementation notes:**
- These are **soft hints**, not ground truth — fully consistent with Hidden Information Doctrine
- They represent metagame expectations, exactly what human players carry in their heads
- Should be computed at tensorization time using `MetagamePriors.get_item_distribution()` etc.
- Only apply to opponent Pokemon with unknown attributes
- Once an item/ability is revealed, the prior features should be zeroed out (truth replaces prior)

**Estimated impact: HIGH.** These features encode the kind of metagame knowledge that separates 1500-rated players from 1800-rated players.

---

### Gap 6: Damage Estimation Features (HIGH)

**Current state:** No damage calculation or estimation. The model must learn damage expectations purely from gameplay outcomes across thousands of games.

**What competitive players do:** Mental damage calculation is a core competitive skill. "Can I OHKO?", "Can they 2HKO me?", "Is this a guaranteed KO after Stealth Rock?" are constant considerations.

**Proposed features (4-8 dims per active matchup):**

| Feature | Type | Description |
|---------|------|-------------|
| best_move_damage_est | continuous | Estimated damage fraction of our best move vs opponent |
| can_ohko_estimate | binary | Does our best move likely OHKO? (est_damage > opp_hp_fraction) |
| opponent_best_damage_est | continuous | Estimated max damage from opponent's known moves |
| can_be_ohkod | binary | Can opponent's best known move likely OHKO us? |
| ko_after_hazards | binary | Can we KO after entry hazard chip? |
| turns_to_ko | continuous | Estimated turns to KO opponent (best_damage / opp_hp) normalized |

**Implementation notes:**
- Simplified damage formula: `damage_fraction ≈ ((2 * level / 5 + 2) * power * atk/def / 50 + 2) * STAB * effectiveness / target_max_hp`
- Use base stats as proxy for actual stats (exact EVs unknown for opponent)
- Normalize output to [0, 1] representing fraction of target HP
- This is an **estimate**, not exact — but even approximate damage knowledge is extremely valuable
- Can be computed during tensorization using available stats + move properties

**Estimated impact: HIGH.** Damage calculation is the quantitative backbone of competitive decisions.

---

### Gap 7: Volatile Status Effects (MEDIUM-HIGH)

**Current state:** `BattleState` tracks `volatiles` as a set of strings (confusion, substitute, leech seed, encore, taunt, etc.), but these are **NOT carried through to observations or tensorization**.

**Why this matters:**
- **Substitute** completely changes decision-making (status moves blocked, can't hit through it with some moves)
- **Confusion** adds 33% self-hit chance
- **Taunt** prevents status moves for 3 turns
- **Encore** locks into the same move
- **Leech Seed** provides passive drain
- **Perish Song** creates urgent switching pressure
- **Trapped** (Arena Trap, Shadow Tag, Magnet Pull) prevents switching

**Proposed features per active Pokemon (8-10 binary dims):**

| Feature | Type | Description |
|---------|------|-------------|
| has_substitute | binary | Protected by Substitute |
| is_confused | binary | Confused |
| is_taunted | binary | Can't use status moves |
| is_encored | binary | Locked into one move |
| is_leech_seeded | binary | Losing HP each turn |
| is_trapped | binary | Cannot switch out |
| has_perish_count | binary | Under Perish Song pressure |
| is_cursed | binary | Losing HP each turn (Ghost curse) |
| protect_used_last | binary | Used Protect last turn (will likely fail again) |
| is_yawned | binary | Will fall asleep next turn unless switching |

**Implementation notes:**
- `BattleState.own_pokemon.volatiles` and `opponent_pokemon.volatiles` already track these
- Need to carry them through `PokemonObservation` → tensorization
- Apply to both own and opponent active Pokemon
- For opponent, only volatiles visible in the battle log (all of them are)

**Estimated impact: MEDIUM-HIGH.** These change optimal play dramatically when present, though they're less frequent than core stats/moves.

---

### Gap 8: Entry Hazard Damage on Switch (MEDIUM-HIGH)

**Current state:** Hazards are tracked as binary/counts in field features, but there's no indication of how much damage each potential switch-in would take.

**What competitive players calculate:** Before every switch, players compute: "Stealth Rock does 25% to this Fire-type, plus 1 layer of Spikes does 12.5%, so my Volcarona takes 37.5% on switch-in. It only has 60% HP, so it'll be at 22.5% — too low."

**Proposed features per bench Pokemon (1 dim × 5 bench = 5 dims):**

| Feature | Type | Description |
|---------|------|-------------|
| switch_in_hazard_damage | continuous | Total % damage this Pokemon takes on switch-in |

**Calculation:**
- Stealth Rock: 12.5% × type effectiveness vs Rock (0.25x to 4x → 3.125% to 50%)
- Spikes: 12.5% × layers (1/2/3) → 12.5%, 16.67%, 25%
- Toxic Spikes: poisons or toxic-poisons on entry (type-dependent)
- Sticky Web: lowers Speed by 1 stage (not direct damage, but flag-worthy)

**Implementation notes:**
- Requires type effectiveness lookup (same table as Gap 2)
- Heavy-Duty Boots (item) negates all hazard damage — check if item is known to be boots
- Levitate/Flying type negates Spikes/Toxic Spikes
- Magic Guard negates all indirect damage

**Estimated impact: MEDIUM-HIGH.** Switch decisions are ~40% of competitive play, and hazard damage is the primary cost of switching.

---

### Gap 9: Action History Beyond Previous Turn (MEDIUM)

**Current state:** Only `prev_player_move` and `prev_opponent_move` (1 turn lookback) are in context features. The transformer does have sequence attention over previous turns' full observations, but explicit recent action features could help.

**What competitive players track:**
- "They've used Earthquake twice — might be Choice-locked"
- "They keep switching out when I set up — they have a phaser/roar"
- "They never attack this Pokemon — they probably don't have a good move for it"

**Proposed features (4-6 dims):**

| Feature | Type | Description |
|---------|------|-------------|
| opponent_used_same_move_twice | binary | Choice lock indicator |
| opponent_switch_count | continuous | Times opponent has switched (normalized /10) |
| own_switch_count | continuous | Times we've switched |
| turns_active_own | continuous | How many consecutive turns our active has been in |
| turns_active_opp | continuous | How many consecutive turns opponent's active has been in |
| opponent_revealed_move_count | continuous | Number of revealed moves for opp active (normalized /4) |

**Implementation notes:**
- Some of this is already in the GRU/transformer sequence context, but explicit features help the model attend to these patterns
- `opponent_used_same_move_twice` is a strong Choice item indicator
- `turns_active` helps with setup timing decisions

**Estimated impact: MEDIUM.** The transformer's sequence attention partially handles this, but explicit features reduce what the model needs to learn.

---

### Gap 10: Own Team Alive Count and Team Advantage (MEDIUM)

**Current state:** `opponents_remaining` is tracked, but there's no explicit `own_remaining` or team advantage feature.

**Proposed features (3 dims):**

| Feature | Type | Description |
|---------|------|-------------|
| own_remaining | continuous | Own alive Pokemon count / 6 |
| team_advantage | continuous | (own_remaining - opp_remaining) / 6, centered at 0 |
| total_hp_advantage | continuous | Sum(own_hp_fractions) - Sum(opp_hp_fractions), normalized |

**Implementation notes:**
- `own_remaining` can be counted from `is_fainted` flags in own team
- `total_hp_advantage` aggregates the overall health differential
- These are high-level strategic signals that help the model understand the game state

**Estimated impact: MEDIUM.** Useful for strategic decision-making (aggressive when ahead, conservative when behind).

---

### Gap 11: Screen/Weather/Terrain Duration (MEDIUM)

**Current state:** Screens (Reflect, Light Screen, Aurora Veil) are binary, and weather/terrain presence is binary. Duration is not encoded.

**Why duration matters:**
- Reflect halves physical damage for 5 turns. On turn 1 of Reflect, you play very differently than turn 5.
- Weather lasting 1 more turn vs 4 more turns affects whether to re-set it.
- Tailwind (4 turns of doubled speed) urgency changes with remaining duration.

**Proposed enhancement (replace binary with continuous for timed effects, ~8 dims adjusted):**

| Feature | Change | Description |
|---------|--------|-------------|
| reflect_turns | binary → continuous | Remaining turns / 5 |
| light_screen_turns | binary → continuous | Remaining turns / 5 |
| aurora_veil_turns | binary → continuous | Remaining turns / 5 |
| tailwind_turns | binary → continuous | Remaining turns / 4 |
| weather_turns | new continuous | Remaining turns / 8 |
| terrain_turns | new continuous | Remaining turns / 5 |
| trick_room_turns | new continuous | Remaining turns / 5 (currently not tracked at all!) |
| gravity_turns | new continuous | Remaining turns / 5 |

**Implementation notes:**
- `SideConditions` already stores turn counts for screens — this info exists but is collapsed to binary
- Weather/terrain turns are in `FieldState` but not in `FieldObservation`
- Trick Room is tracked as a turn counter in `FieldState` but completely absent from features

**Estimated impact: MEDIUM.** Duration-aware play is important for timing decisions.

---

### Gap 12: Game Phase Feature (LOW-MEDIUM)

**Current state:** `game_phase` exists in `TurnObservation` as "team_preview", "battle", or "finished" but is **not tensorized**.

**Why it matters:** Strategy changes dramatically between early game (hazard setup, scouting), mid game (trading, positioning), and late game (cleaning, endgame calculation).

**Proposed features (3 binary dims):**

| Feature | Type | Description |
|---------|------|-------------|
| is_early_game | binary | turn < 5 AND most Pokemon alive |
| is_mid_game | binary | 5 ≤ turn < 20 |
| is_late_game | binary | Few Pokemon remaining or turn ≥ 20 |

**Implementation notes:**
- Could also be a continuous "game progress" feature: `1 - (alive_both / 12)`
- The transformer's turn positional encoding partially captures this already

**Estimated impact: LOW-MEDIUM.** Turn number and remaining counts partially encode this already.

---

### Gap 13: Opponent Pokemon Visibility Features (LOW-MEDIUM)

**Current state:** No explicit tracking of which opponent Pokemon have been seen in battle vs. only known from team preview.

**Proposed features (2-3 dims):**

| Feature | Type | Description |
|---------|------|-------------|
| opponent_unseen_count | continuous | Number of opponent Pokemon never switched in / 6 |
| opponent_scouted | binary | Have all 6 opponent Pokemon been seen in battle? |
| opponent_active_is_new | binary | Is this the first time this Pokemon has been sent out? |

**Implementation notes:**
- `OpponentPokemon.seen_in_battle` already tracks this in `BattleState`
- Unseen Pokemon represent strategic uncertainty — the opponent may have a surprise counter

**Estimated impact: LOW-MEDIUM.** Useful for uncertainty quantification.

---

### Gap 14: Ability Interaction Features (LOW-MEDIUM)

**Current state:** Ability is a categorical embedding. No explicit modeling of ability effects.

**Key ability interactions that change play:**

| Ability | Effect | Importance |
|---------|--------|------------|
| Intimidate | -1 Atk on switch-in | Affects physical damage calculations |
| Levitate | Ground immunity | Changes type matchup entirely |
| Flash Fire | Fire immunity + boost | Changes type matchup |
| Water Absorb / Volt Absorb | Type immunity + heal | Changes type matchup |
| Mold Breaker | Ignores target abilities | Removes defensive interactions |
| Drought/Drizzle/Sand Stream/Snow | Auto-weather | Weather changes on switch |
| Swift Swim/Chlorophyll | Speed doubled in weather | Speed matchup changes |
| Protean/Libero | Type changes | Matchup becomes dynamic |

**Proposed features (3-4 binary dims for active Pokemon):**

| Feature | Type | Description |
|---------|------|-------------|
| ability_grants_immunity | binary | Active ability provides a type immunity |
| ability_sets_weather | binary | Switching this Pokemon changes weather |
| ability_boosts_speed | binary | Ability doubles speed under conditions |
| ability_affects_contact | binary | Ability punishes/modifies contact moves |

**Implementation notes:**
- Requires a small ability effect lookup table
- Most impactful for known opponent abilities (revealed ones)
- For unrevealed abilities, metagame priors (Gap 5) cover the uncertainty

**Estimated impact: LOW-MEDIUM.** Important in specific matchups but the embedding can learn some of this.

---

### Gap 15: Choice Lock Detection (LOW)

**Current state:** No explicit feature for detecting Choice item locks.

**What competitive players infer:** If an opponent has used the same move for 2+ consecutive turns, they're likely Choice-locked (Choice Band/Specs/Scarf). This makes them predictable and vulnerable to specific counters.

**Proposed features (2 dims):**

| Feature | Type | Description |
|---------|------|-------------|
| opponent_likely_choice_locked | binary | Opponent used same move 2+ turns in a row |
| opponent_choice_lock_turns | continuous | Consecutive turns using same move / 5 |

**Implementation notes:**
- Derivable from action history
- The transformer's attention over previous turns can theoretically detect this, but an explicit feature helps

**Estimated impact: LOW.** The transformer sequence attention should eventually learn this pattern.

---

### Gap 16: Tera Type Information (LOW)

**Current state:** Tera type is tracked but always `UNKNOWN` for opponents until used. Own tera type is represented as a categorical feature.

**Enhancement:** Once Tera is used (for either player), the defensive type matchup changes completely. The current binary `terastallized` flag doesn't capture the NEW type matchup.

**Proposed features (2 dims per terastallized Pokemon):**

| Feature | Type | Description |
|---------|------|-------------|
| tera_type_effectiveness_shift | continuous | Change in type vulnerability from Tera |
| tera_stab_coverage | binary | Does Tera type grant new STAB coverage? |

**Estimated impact: LOW.** Edge case that only matters after Tera has been used.

---

## Priority Matrix

| Priority | Feature Category | Added Dims (est.) | Implementation Effort | Impact |
|----------|-----------------|-------------------|----------------------|--------|
| **P0** | Move Properties (Gap 1) | 28/pokemon (own) | Medium | Very High |
| **P0** | Opponent Base Stats (Gap 3) | 0 (fill existing 0s) | Low | Very High |
| **P0** | Type Effectiveness (Gap 2) | 12 | Medium | Very High |
| **P1** | Speed Comparison (Gap 4) | 6 | Low | High |
| **P1** | Metagame Priors (Gap 5) | 10/opp pokemon | Medium | High |
| **P1** | Damage Estimation (Gap 6) | 6 | Medium-High | High |
| **P2** | Volatile Statuses (Gap 7) | 10/active pokemon | Low-Medium | Medium-High |
| **P2** | Hazard Switch Damage (Gap 8) | 5 | Medium | Medium-High |
| **P2** | Action History (Gap 9) | 6 | Low | Medium |
| **P2** | Team Advantage (Gap 10) | 3 | Low | Medium |
| **P3** | Duration Features (Gap 11) | 8 (replace existing) | Low | Medium |
| **P3** | Game Phase (Gap 12) | 3 | Low | Low-Medium |
| **P3** | Opponent Visibility (Gap 13) | 3 | Low | Low-Medium |
| **P3** | Ability Interactions (Gap 14) | 4 | Medium | Low-Medium |
| **P4** | Choice Lock (Gap 15) | 2 | Low | Low |
| **P4** | Tera Shift (Gap 16) | 2 | Low | Low |

---

## Hidden Information Doctrine Compliance

Every proposed feature is checked against the five rules:

| Rule | Feature Compliance |
|------|-------------------|
| 1. Never train on omniscient features | All features use only information available to the acting player at decision time |
| 2. Represent uncertainty explicitly | Unknown opponent moves/items still marked unknown; priors are soft probabilities |
| 3. Metagame priors as soft hints only | Prior features are probability distributions, not leaked ground truth |
| 4. Separate hidden-state inference from action | Auxiliary head handles hidden-info prediction; features are observations |
| 5. Evaluate rare-set robustness | Priors naturally have uncertainty for rare sets; damage estimates use base stats as proxies |

**Special note on opponent base stats (Gap 3):** Base stats are Pokedex data, intrinsic to the species, and universally known. They are NOT hidden information. The current practice of zeroing them out is overly conservative and discards information that every human player has. The distinction between base stats (public, species-level) and exact stats (private, set-dependent) is critical.

---

## Estimated Total Feature Dimension Impact

### Conservative (P0 + P1 only):
- Move properties: +28 per own pokemon × 6 = +168 (but these could be added only for active pokemon's moves for efficiency)
- Opponent base stats: +0 (fill existing zeros)
- Type effectiveness: +12
- Speed comparison: +6
- Metagame priors: +10 per opp pokemon × 6 = +60
- Damage estimation: +6

**Conservative total: ~252 additional dims** (before embedding), bringing flat input from 384 to ~636 dims.

### With transformer embeddings:
The transformer already uses embedding layers, so additional dims are projected into hidden_dim. The added features would:
- Enrich the `PokemonEmbedding` input from 30 → ~58 dims per pokemon
- Enrich the `ContextEmbedding` input from 6 → ~18 dims
- Add a potential `MatchupEmbedding` token (new token type for active matchup features)

The hidden_dim (384) and token count (14) can remain unchanged — the added information improves token quality rather than token quantity.

---

## Implementation Roadmap

### Phase 4A: Core Feature Enrichment (P0 features)
1. Add Pokedex base stats lookup table
2. Fill opponent base stats from species identification
3. Carry move properties through observation pipeline
4. Add type effectiveness lookup table
5. Compute type matchup features
6. Update `POKEMON_FEATURE_DIM` and `PokemonEmbedding`
7. Retrain and measure action prediction accuracy improvement

### Phase 4B: Strategic Feature Layer (P1 features)
1. Integrate metagame priors into tensorization
2. Add speed comparison features
3. Implement simplified damage estimation
4. Add features to context embedding
5. Ablation study: measure marginal contribution of each feature group

### Phase 4C: Tactical Features (P2-P3 features)
1. Add volatile status tracking to observations
2. Add hazard switch damage calculation
3. Extend action history features
4. Add duration features for timed effects
5. Final ablation sweep

---

## Ablation Study Plan

To measure the marginal value of each feature group, run the following ablation matrix:

| Experiment | Features | Expected Metric |
|-----------|----------|-----------------|
| Baseline | Current 384 dims | Top-1 ~35-40% |
| +MoveProps | + move properties | Top-1 +3-5% |
| +OppStats | + opponent base stats | Top-1 +2-4% |
| +TypeEff | + type effectiveness | Top-1 +3-5% |
| +Speed | + speed comparison | Top-1 +1-2% |
| +Priors | + metagame priors | Top-1 +2-3% |
| +Damage | + damage estimation | Top-1 +2-3% |
| All P0+P1 | All above combined | Top-1 +8-15% |

These are estimates based on the principle that features encoding information competitive players actively use should directly improve imitation accuracy.

---

## Conclusion

The current 384-dim feature set captures the *what* of the battle state (what Pokemon are present, what moves are known, what the HP levels are) but misses the *so what* — the derived strategic implications that competitive players compute instantly. By adding move properties, type matchups, opponent base stats, speed comparisons, and metagame priors, we bridge the gap between raw state observation and strategic understanding, directly improving the model's ability to imitate expert play.

The single most impactful change is filling in opponent base stats (Gap 3) because it's zero implementation cost (the dims already exist) and provides fundamental information about every opponent Pokemon. The second most impactful is adding move properties (Gap 1) because it eliminates the need for the model to memorize ~600 moves' properties from usage patterns alone.
