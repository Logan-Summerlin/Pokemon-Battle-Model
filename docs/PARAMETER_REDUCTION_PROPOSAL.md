# Parameter Reduction Proposal: P8 from ~3.6M to ~2M Parameters

## Executive Summary

The current P8 model (4L/256d/4H) has **3,639,611 trainable parameters**. This proposal presents a systematic plan to reduce the model to approximately **1.9M–2.1M parameters** while maintaining or improving action prediction accuracy. The strategy combines architectural rationalization, embedding compression, dead feature pruning, head simplification, and targeted feature enrichment that replaces parameters with information.

The core insight is this: **86.8% of parameters live in the transformer layers**, so the primary lever is reducing depth and/or width. But a naive reduction loses representational capacity. We compensate by (a) enriching the input features so the model needs to learn less from raw data, (b) pruning dead/redundant features, and (c) simplifying auxiliary heads that consume parameters without proportional benefit.

---

## Current Architecture Breakdown

### P8 (4 Layers / 256 Hidden / 4 Heads)

| Component | Parameters | % of Total |
|-----------|-----------|-----------|
| Embedding tables | 96,480 | 2.7% |
| Embedding projections | 104,960 | 2.9% |
| Positional embeddings | 2,560 | 0.1% |
| **Transformer layers** | **3,159,040** | **86.8%** |
| Policy head | 69,133 | 1.9% |
| Auxiliary head | 174,413 | 4.8% |
| Value head | 33,025 | 0.9% |
| **TOTAL** | **3,639,611** | **100%** |

### Per-Transformer-Layer Breakdown (256d)

| Sub-component | Parameters |
|---------------|-----------|
| Self-attention (Q,K,V,O projections) | 263,168 |
| LayerNorm (attention) | 512 |
| FFN (256 → 1024 → 256) | 525,568 |
| LayerNorm (FFN) | 512 |
| **Total per layer** | **789,760** |

**Key observation:** The FFN is 66.5% of each transformer layer. The standard 4x FFN multiplier is the single largest parameter consumer in the entire model.

---

## Proposed Target Architecture: "P8-Lean"

### Configuration: 3 Layers / 224 Hidden / 4 Heads

| Component | Current (P8) | Proposed (P8-Lean) | Savings |
|-----------|-------------|-------------------|---------|
| Layers | 4 | 3 | -1 layer |
| Hidden dim | 256 | 224 | -32 dims |
| Heads | 4 | 4 | same |
| FFN dim | 1024 | 672 (3x ratio) | -352 dims |
| Aux head | 5 sub-heads, H/2 intermediate | 3 sub-heads, H/4 intermediate | simplified |
| Value head | On | Off | removed |
| Dead features | Included | Pruned | cleaner |
| Embedding dims | Large | Compressed | smaller |

### Estimated Parameter Count: ~1,950,000

---

## Reduction Strategies (Ordered by Impact)

### Strategy 1: Reduce Transformer Depth (4 → 3 Layers)

**Savings: ~790K parameters (21.7% of total)**

The case for 3 layers:

1. **Diminishing returns at this data scale.** With ~190K training examples, 4 layers may overfit more than 3. The existing P8_10K_TRAINING_TIME_REDUCTION_PLAN.md already proposed "3L/224d/4H" as a "P8-fast" candidate with "<10% quality loss."

2. **Token-level representation is already rich.** The 14 tokens per turn have strong semantic structure (own pokemon, opponent pokemon, field, context). Unlike NLP where tokens need many layers to build phrase→sentence→meaning hierarchies, battle tokens arrive pre-structured. Three layers of self-attention are sufficient for: (a) cross-attending between own and opponent slots, (b) integrating field/context information, and (c) refining representations for the policy head.

3. **Empirical evidence from the scaling ladder.** The 2L/128d smoke test config (606K params) already achieves non-trivial accuracy. Going from 2→3 layers adds substantial capacity; the marginal value of the 4th layer is likely smaller.

4. **The windowed dataset helps.** With per-turn training examples (not just last-step), each layer gets trained on ~190K examples. Three well-trained layers should outperform four under-trained layers.

**Quality risk:** Low. The 3L/224d configuration is explicitly mentioned in existing docs as a viable reduced config.

---

### Strategy 2: Reduce Hidden Dimension (256 → 224)

**Savings: ~630K parameters (across all components)**

Why 224 specifically:

1. **Divisible by 4 heads** (224 / 4 = 56 per head). Clean head dimension alignment.

2. **Still comfortably above the representation bottleneck.** The maximum per-token input (pokemon embedding concatenation) is ~300 dims before projection. Projecting 300 → 224 is mild compression. Projecting 300 → 192 would be more aggressive.

3. **The key information to represent per token.** A pokemon token encodes: species identity, moveset, item, ability, type, HP, boosts, status, and ownership. These concepts likely need ~50-100 independent "feature directions" in embedding space. 224 dims provides comfortable headroom for this. The remaining capacity handles cross-token interactions learned through attention.

4. **Scale-appropriate for 10K battles.** At 190K training examples, a 224d model (per the scaling laws for transformers) is in a better data/param balance than 256d.

**Quality risk:** Low. The 32-dim reduction is modest (12.5%). With enriched input features (see Strategy 6), the model needs less internal capacity to represent the same information.

---

### Strategy 3: Reduce FFN Multiplier (4x → 3x)

**Savings: ~350K parameters (across 3 layers)**

This is one of the most impactful and least-discussed optimizations:

| Config | FFN dim | FFN params/layer | Total FFN (3L) |
|--------|---------|-----------------|----------------|
| 4x (standard) | 896 | 401,408 | 1,204,224 |
| 3x (proposed) | 672 | 301,056 | 903,168 |
| **Savings** | | **100,352** | **301,056** |

**Justification:**

1. **The 4x multiplier is a legacy default**, not a universal optimum. It was established in the original "Attention Is All You Need" paper for language modeling at massive scale. For a 14-token structured input domain with 190K examples, it is likely over-parameterized.

2. **Recent efficient transformer research** (e.g., MobileBERT, EfficientFormer) shows that 3x or even 2.5x FFN ratios work well for smaller models, sometimes with improved generalization due to reduced overfitting risk.

3. **The FFN's role in our model** is to add non-linear transformation capacity per-token. With structured, information-rich input tokens (not raw text subwords), each token already has strong semantic content. The FFN doesn't need as much capacity to "decode" meaning.

4. **Combined effect with depth reduction.** Removing 1 layer AND reducing the FFN ratio compounds the savings without compounding the quality loss, because each remaining layer becomes a higher proportion of a better-regularized model.

**Implementation:** Add a `ffn_multiplier` config parameter (default 4, set to 3) or simply pass `dim_feedforward=hidden_dim * 3` to `TransformerEncoderLayer`.

**Quality risk:** Low-to-moderate. Should be validated with ablation, but 3x FFN at 224d is well within safe territory.

---

### Strategy 4: Remove Value Head

**Savings: ~25,000 parameters + training compute**

The value head predicts win probability, but:

1. **It doesn't contribute to the primary metric** (action prediction top-1/top-3 accuracy). It's an auxiliary signal.

2. **Its regularization benefit is questionable.** The P8_10K plan already recommends `--no-value-head` for throughput-first runs and notes "usually low [quality risk] for top-1 action imitation."

3. **Game result labels are noisy.** With `game_result = 0.5` for "unknown" outcomes (common in replays without clear win/loss markers), many examples don't contribute to value loss at all.

4. **The parameter savings are modest** (~25K) but the compute savings per training step are meaningful, and it simplifies the loss landscape.

**Recommendation:** Remove the value head for the lean model. If value estimation is needed later (e.g., for RL fine-tuning in Phase 5+), it can be re-added and fine-tuned on a frozen encoder.

**Quality risk:** Very low for imitation learning accuracy.

---

### Strategy 5: Simplify Auxiliary Head

**Savings: ~100K parameters**

The current auxiliary head has **5 sub-heads**, each with a `Linear(H, H/2) + GELU + Dropout + Linear(H/2, K)` architecture. This is ~174K parameters total. We propose streamlining:

#### 5a. Remove the tera category sub-head

**Savings: ~33K parameters**

- Tera actions are only 2.45% of all actions in the dataset.
- Predicting opponent tera category (4 classes: offensive/defensive/STAB/coverage) from partial information is extremely difficult and noisy.
- The signal-to-noise ratio is poor: most opponent Pokemon haven't terastallized, so targets are predominantly masked (-1).
- The embedding already captures tera-relevant information through species and type embeddings.

#### 5b. Merge speed bucket and role archetype into a single "threat profile" head

**Savings: ~33K parameters**

Current: Two separate heads predict speed bucket (5 classes) and role archetype (8 classes).

Proposed: One combined "threat profile" head that predicts a single categorical over ~10 classes (e.g., "fast-sweeper", "slow-wall", "pivot", "hazard-setter", "bulky-attacker", "speed-control", "special-wall", "mixed-attacker", "support", "other"). This is more pokemetically meaningful — speed and role are highly correlated (fast pokemon are usually sweepers, slow pokemon are usually walls). A single head captures this correlation directly rather than forcing the model to predict two independently.

#### 5c. Reduce auxiliary hidden dimension from H/2 to H/4

**Savings: ~40K additional parameters**

The auxiliary heads don't need large intermediate representations. They operate on individual opponent slot tokens (already rich with H-dimensional contextual information from the encoder). A smaller bottleneck (H/4 = 56 dims at H=224) is sufficient for the classification tasks.

| Sub-head | Current | Proposed |
|----------|---------|----------|
| Item prediction (50 classes) | H→H/2→50 | H→H/4→50 |
| Threat profile (10 classes) | (H→H/2→5) + (H→H/2→8) | H→H/4→10 |
| Move family (10 families) | H→H/2→10 | H→H/4→10 |
| Tera category | H→H/2→4 | **Removed** |

**Estimated auxiliary params after simplification: ~40K** (down from 174K).

**Quality risk:** Low. The auxiliary head's purpose is to regularize the encoder representations, not to achieve high auxiliary accuracy. A smaller auxiliary head still provides the gradient signal needed for representation learning. The tera head provided minimal signal due to data sparsity.

---

### Strategy 6: Prune Dead and Redundant Input Features

**Savings: Minimal parameter savings, but improves information density and effective capacity.**

#### 6a. Remove `terastallized` binary flag

The `terastallized` flag has a **0.0% activation rate** in the processed P8 dataset. It is always zero. This is a genuinely dead feature — in the Metamon replay format, terastallization is not captured as a per-pokemon state flag in a way that populates this field.

**Implementation:** Set `prune_dead_features: True` in config, which already reduces `cont_binary_dim` by 1 in `PokemonEmbedding`.

#### 6b. Remove field binary side conditions (16 dims)

The field binary nonzero rate is **0.0%** in processed data. All 16 binary field features (own/opponent stealth_rock, spikes, toxic_spikes, sticky_web, reflect, light_screen, aurora_veil, tailwind) are consistently zero.

**Why this happens:** The Metamon parsed replay format stores side conditions in a way that the current `_build_field_observation` either can't access or that isn't populated in the 10K cohort. These 16 zero-dimensions pass through the field embedding projection, wasting `16 * H` parameters in the projection weight matrix (~3,584 at H=224).

**Implementation:** Set `prune_dead_features: True` in config, which already sets `binary_dim = 0` in `FieldEmbedding`. The field projection input shrinks from 32 → 16 (just weather + terrain embeddings).

**Important caveat:** If future data processing fixes the side condition extraction, these features become valuable. The prune flag should be considered a data-quality workaround, not a permanent architectural decision. If we ever fix the pipeline to populate hazard/screen data, we should re-enable these features — they're strategically important (as the Feature Analysis doc notes in Gap 8).

#### 6c. Compress the `accuracy` and `evasion` boosts

Accuracy and evasion boosts are extremely rare in competitive Gen 9 OU. Moves that modify accuracy/evasion are largely banned (Evasion Clause) or rarely used. These two features could be merged into a single "accuracy_modifier" = `own_accuracy_boost - opp_evasion_boost` (what actually matters for hit calculations), saving 1 dim per pokemon (12 dims total).

However, this saves essentially zero parameters (it's continuous input, not an embedding table), so the benefit is marginal input complexity reduction rather than parameter savings.

#### 6d. Evaluate the `is_own` binary flag for redundancy

Every own-team token has `is_own = 1.0` and every opponent token has `is_own = 0.0`. The `TokenTypeEmbedding` already distinguishes own vs. opponent tokens via a separate learned embedding (type 0 = own, type 1 = opponent). The `is_own` flag in the pokemon features is therefore **redundant with the token type embedding**.

Removing `is_own` saves 0 parameters (it's a binary input, not an embedding), but it reduces noise in the input vector by removing a perfectly-predictable feature that the model must learn to ignore or align with the token type signal.

**Recommendation:** Keep `is_own` for now but flag as a future simplification candidate. The token type embedding should be sufficient, but removing it requires careful validation.

---

### Strategy 7: Compress Embedding Tables

**Savings: ~20K–40K parameters**

Current embedding tables consume 96,480 parameters (2.7% of total). While small relative to the transformer, there are clear over-allocations:

#### 7a. Reduce species embedding: 64 → 48 dims

**Savings: ~10K parameters**

The species vocabulary is 541 entries (plus PAD and UNK). At 64 dims, we allocate 34,624 parameters. At 48 dims, we allocate 25,968 — a savings of 8,656.

**Justification:** Species identity is important but partially redundant with other features. Once you know a Pokemon's type, base stats, and moveset, the species identity provides diminishing additional information. A 48-dim embedding still has more than enough capacity to encode the ~100-200 competitively-relevant species distinctly (a 48-dim space can theoretically separate hundreds of thousands of points).

#### 7b. Reduce move embedding: 32 → 24 dims

**Savings: ~8K parameters per table**

There are two move embedding tables: one in `PokemonEmbedding` (4 move slots per pokemon) and one in `ContextEmbedding` (prev_player_move, prev_opponent_move). With 527 moves in the vocabulary:

- Current: 527 × 32 = 16,864 per table × 2 = 33,728 total
- Proposed: 527 × 24 = 12,648 per table × 2 = 25,296 total
- Savings: ~8,400

**Justification:** Move identity is supplemented by the move properties features proposed in the Feature Analysis (base_power, accuracy, priority, type, category). With these properties provided explicitly, the embedding needs to capture less — mainly the move's "strategic archetype" and edge-case interactions not captured by raw stats.

**Important synergy:** This reduction is most effective when paired with move property features (Feature Analysis Gap 1). The embedding carries less burden because the explicit features carry the quantitative information.

#### 7c. Reduce item and ability embeddings: 32 → 16 dims

**Savings: ~13K parameters**

- Items: 133 vocab × 16 (down from 32) = savings of 2,128
- Abilities: 240 vocab × 16 (down from 32) = savings of 3,840

For opponent Pokemon, items and abilities are frequently UNKNOWN (~49% and ~45% respectively). The embedding table entry for UNK_IDX carries no information. Reducing the embedding dimension reduces wasted capacity on the unknown-item representations that dominate half of all lookups.

#### 7d. Share move embedding between PokemonEmbedding and ContextEmbedding

**Savings: ~12K parameters**

Currently, `PokemonEmbedding.move_emb` and `ContextEmbedding.prev_move_emb` are separate `nn.Embedding(600, 32)` tables (or `nn.Embedding(530, 24)` after compression). They encode the same vocabulary and have the same semantic meaning — a move's identity. Sharing a single embedding table eliminates ~12K redundant parameters.

**Implementation:** Pass the move embedding table from `PokemonEmbedding` to `ContextEmbedding` during initialization, or create a shared embedding at the `BattleTransformerEncoder` level.

**Quality risk:** Very low. Weight sharing for the same vocabulary is standard practice (cf. input/output embedding tying in language models).

---

### Strategy 8: Feature Enrichment to Compensate for Capacity Reduction

This is the compensatory strategy: **add information to the input so the model needs fewer parameters to achieve the same accuracy.** Adding features to the input is essentially free in parameter cost (a few extra dimensions in the projection layer), but can dramatically reduce what the model needs to learn from data.

We prioritize the three highest-impact, lowest-cost additions from the Phase 4 Feature Analysis:

#### 8a. Fill Opponent Base Stats from Species (Gap 3 — Zero Implementation Cost)

**Current:** 6 base stat dims × 6 opponent pokemon = 36 dims of zeros per turn.
**Proposed:** Look up base stats from a species → stats mapping (built from own-team data in training set).

This is the single highest-ROI change in the entire proposal:
- **Zero additional parameters** (the dims already exist, just populated instead of zeroed)
- **Zero additional embedding cost**
- **Massive information gain**: The model currently has no idea whether the opponent's Dragapult is fast or slow, physical or special. With base stats filled in, it instantly knows Dragapult has 142 base Speed and 120 base SpAtk.
- **Fully compliant with Hidden Information Doctrine**: Base stats are species-level public Pokedex data, not hidden EV/IV information.

#### 8b. Add Move Properties (Gap 1 — ~5 dims per move, minor cost)

Add `base_power/150`, `accuracy/100`, `priority/5`, `category` (one-hot or ordinal: physical=0, special=0.5, status=1), and `is_stab` (binary) to each move slot.

**Cost:** 5 additional continuous/binary features per move × 4 moves = 20 extra dims per pokemon in the projection. At H=224, this adds `20 * 224 = 4,480` parameters to the pokemon projection layer — trivial.

**Benefit:** The model no longer needs to memorize that Earthquake is a 100BP Ground Physical move purely from seeing species+move name co-occurrence with actions. This directly reduces the information that move embeddings need to encode, validating the reduction from 32→24 dim move embeddings.

#### 8c. Provide Speed Comparison Signal (Gap 4 — ~2 dims, near-zero cost)

Add `speed_advantage_estimate = base_spe_own / (base_spe_own + base_spe_opp)` and `prev_speed_order` (binary: did we move first last turn?) to context features.

**Cost:** 2 extra dims in context, adding `2 * 224 = 448` parameters to the context projection.

**Benefit:** "Do I outspeed?" is one of the most important binary decisions in competitive Pokemon. Currently the model has no direct speed comparison signal and must learn to compare stats across token positions through attention — a wasteful use of attention capacity. An explicit signal frees up transformer capacity for other reasoning.

---

## Parameter Budget Summary

### P8-Lean Target Architecture

| Component | P8 Current | P8-Lean | Change |
|-----------|-----------|---------|--------|
| **Transformer** | | | |
| Layers | 4 | 3 | -1 |
| Hidden dim | 256 | 224 | -32 |
| FFN multiplier | 4x (1024) | 3x (672) | -1x |
| Transformer params | 3,159,040 | ~1,350,000 | **-1,809,040** |
| **Embeddings** | | | |
| Species emb | 38,400 (600×64) | 28,800 (600×48) | -9,600 |
| Move emb (×2) | 38,400 (2×600×32) | 14,400 (1×600×24 shared) | -24,000 |
| Item emb | 6,400 (200×32) | 3,200 (200×16) | -3,200 |
| Ability emb | 9,600 (300×32) | 4,800 (300×16) | -4,800 |
| Type emb | 3,200 (200×16) | 2,400 (200×12) | -800 |
| Status/weather/terrain | 480 | 480 | 0 |
| Emb table params | 96,480 | 54,080 | **-42,400** |
| **Projections** | | | |
| Pokemon proj | 77,824 | ~72,000* | -5,824 |
| Field proj | 8,960 | ~4,000** | -4,960 |
| Context proj | 37,376 | ~17,000*** | -20,376 |
| Proj params | 124,160 | ~93,000 | **-31,160** |
| **Heads** | | | |
| Policy head | 69,133 | ~53,600 | -15,533 |
| Auxiliary head | 174,413 | ~40,000 | **-134,413** |
| Value head | 33,025 | 0 | **-33,025** |
| Head params | 276,571 | ~93,600 | **-182,971** |
| **Position/Type** | | | |
| Token type + slot | 2,560 | 2,240 | -320 |
| **TOTAL** | **~3,640,000** | **~1,950,000** | **~-1,690,000** |

\* Pokemon projection increases slightly due to +20 move property dims but decreases due to smaller H and pruned terastallized flag. Net slightly smaller.

\** Field projection shrinks dramatically: input goes from 32 → 16 (pruning 16 dead binary dims).

\*** Context projection: no longer needs its own move embedding table (shared), and input dims change only marginally (+2 speed features).

### Estimated Total: ~1.95M parameters (46% reduction)

---

## Strategies That Were Considered But Rejected

### Rejected: Reducing to 2 layers

While 2L/256d/4H hits the ~2M target, 2 layers is too few for the cross-attention reasoning required. With 14 tokens per step and up to 20 steps of history (280 tokens), two layers provide only two rounds of information propagation. The model needs at minimum: (1) one layer to integrate within-group information (comparing own pokemon to each other, opponent pokemon to each other), and (2-3) layers to integrate across groups (own vs opponent matchups, field effects on matchups, context influences on action selection). Three layers is the minimum for this reasoning chain.

### Rejected: Reducing to 192 hidden dims

At 192d, the model drops below 1.7M parameters — overshooting the target. More importantly, 192 dims may bottleneck the pokemon embedding projection (301 input dims → 192 is >36% compression in a single linear layer), risking information loss. The 224d target maintains a gentler 25% compression ratio.

### Rejected: Removing the auxiliary head entirely

The auxiliary head provides valuable gradient signal for learning opponent representations. Without it, the encoder's opponent slot tokens only receive gradient through the policy head (via mean-pooling), which is an indirect and diluted signal. The auxiliary head provides direct, per-slot supervision that specifically shapes how the model represents hidden opponent information. This is worth the ~40K parameter cost.

### Rejected: Token count reduction (14 → 10)

Compressing 6 opponent bench tokens into 1-2 summary tokens (as suggested in the P8_10K plan) would reduce attention cost but risks losing the ability to make informed switching decisions. Switch actions are 35.9% of all actions in the dataset — the model needs per-slot bench information to decide which pokemon to switch to and when to switch away from a bad matchup. This optimization should only be explored after the lean architecture is validated.

### Rejected: Reducing number of attention heads (4 → 2)

At 224d with 4 heads, each head has 56 dimensions — already fairly small. Reducing to 2 heads (112 dims each) would give each head more capacity but fewer independent attention patterns. In our structured 14-token input, different heads likely specialize (e.g., one for own-opp matchup attention, one for field/context integration, one for bench evaluation, one for temporal patterns). Maintaining 4 heads preserves this specialization.

---

## Implementation Roadmap

### Phase A: Architecture Changes (No Data Pipeline Changes)

1. Add `ffn_multiplier` parameter to `TransformerConfig` (default 4, target 3).
2. Create `P8LeanConfig` factory method: 3L/224d/4H, FFN 3x, no value head.
3. Implement shared move embedding between PokemonEmbedding and ContextEmbedding.
4. Reduce embedding dimensions in config (species 48, moves 24, items 16, abilities 16, types 12).
5. Simplify auxiliary head: remove tera sub-head, merge speed+role into threat profile, reduce intermediate dim to H/4.
6. Enable `prune_dead_features` by default.
7. Validate parameter count matches ~1.95M target.
8. Run smoke test training to verify forward/backward pass works.

### Phase B: Feature Enrichment (Data Pipeline Changes)

1. Build species → base_stats lookup table from training data.
2. Populate opponent base stats from species identification in `_pokemon_to_opponent_observation`.
3. Add move properties (power, accuracy, priority, category, STAB) to `PokemonObservation` and `tensorize_pokemon`.
4. Add speed comparison features to context.
5. Update `POKEMON_FEATURE_DIM`, `CONTEXT_FEATURE_DIM`, and embedding input dimensions.
6. Reprocess dataset with enriched features.

### Phase C: Validation

1. Train P8-Lean on the same 10K battle dataset used for current P8.
2. Compare top-1 and top-3 action prediction accuracy against P8 baseline.
3. Run 3-seed stability check per the experiment plan protocol.
4. Measure training time improvement (expect ~2x faster per epoch due to smaller model + shorter attention).
5. Ablation: test each reduction independently to identify any single change that causes disproportionate quality loss.

---

## Ablation Matrix

To validate the proposal, run these configurations (3 seeds each):

| ID | Config | Est. Params | Purpose |
|----|--------|------------|---------|
| A0 | P8 baseline (4L/256d/4H) | 3.64M | Reference |
| A1 | 3L/256d/4H (depth only) | 2.85M | Isolate depth reduction |
| A2 | 3L/224d/4H (depth + width) | 2.22M | Isolate width reduction |
| A3 | A2 + FFN 3x | ~1.95M | Isolate FFN reduction |
| A4 | A3 + no value head | ~1.92M | Isolate value head removal |
| A5 | A4 + compressed embeddings | ~1.88M | Isolate embedding compression |
| A6 | A5 + simplified aux head | ~1.75M | Isolate aux simplification |
| A7 | A6 + enriched features | ~1.78M | Full lean + feature enrichment |

**Decision rule:** Accept P8-Lean (A7) if:
- Top-1 accuracy drop ≤ 1.0 absolute point vs A0
- Top-3 accuracy drop ≤ 0.8 absolute point vs A0
- Generalization gap increase ≤ 1.0 absolute point
- Seed std-dev increase ≤ 0.5 absolute point

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| 3L too shallow for long sequences | Low | Medium | Ablation A1 isolates this; fallback to 4L/208d |
| FFN 3x loses expressiveness | Low | Medium | Ablation A3 isolates this; fallback to 3.5x |
| Smaller embeddings lose species distinctions | Very Low | Low | 48d species emb still far exceeds the ~200 species that matter |
| Move properties add noise | Low | Low | Properties are ground truth, not estimates; unlikely to hurt |
| Opponent base stats leak information | Very Low | High | Base stats are public Pokedex data, not hidden EVs/IVs |
| Simplified aux head weakens representations | Low-Med | Medium | Ablation A6 isolates this; can restore if needed |

---

## Expected Outcomes

1. **Parameter count:** ~1.95M (46% reduction from 3.64M)
2. **Training speed:** ~2.0-2.5x faster per epoch (fewer layers, smaller dims, shorter FFN)
3. **Accuracy:** Within 1 top-1 point of P8 baseline, possibly improved due to better regularization + richer features
4. **Generalization:** Likely improved due to reduced overfitting risk with smaller model
5. **Memory:** ~46% less GPU memory for model parameters, enabling larger batch sizes or longer windows

---

## Conclusion

The path from 3.6M to ~2M parameters is achievable through a principled combination of:

1. **Depth reduction** (4→3 layers): the largest single lever, well-supported by existing analysis
2. **Width reduction** (256→224): moderate and safe
3. **FFN ratio reduction** (4x→3x): underutilized optimization with strong theoretical backing
4. **Embedding compression** (smaller dims + shared tables): low-risk with feature enrichment
5. **Head simplification** (remove value, streamline aux): removing low-value components
6. **Dead feature pruning**: cleaning up confirmed-zero inputs
7. **Feature enrichment** (opponent base stats, move properties, speed): compensating for reduced capacity with increased input quality

The key philosophical shift is: **instead of a large model that must learn everything from raw categorical indices, build a smaller model that receives richer, more informative features.** This is both more parameter-efficient and more aligned with how competitive Pokemon players actually process information — they don't memorize move IDs, they evaluate move power, type effectiveness, and speed comparisons directly.
