# P8-Lean Training Speed Optimization Proposal

## Objective

Reduce total training time for the P8-Lean BattleTransformer by **30%** without significant loss in action prediction accuracy or generalizability. Current conditions: ~10,000 battles (~170,000 per-turn training examples), P8-Lean config (3L/224d/4H, ~1.95M params), GPU-bound training, ~140s per epoch.

---

## Current Implementation Baseline

This section describes what the code **actually does today** based on a line-by-line audit of `scripts/train_phase4.py` and `src/models/battle_transformer.py`.

### Architecture: P8-Lean (3L/224d/4H)

Defined in `TransformerConfig.p8_lean()` (battle_transformer.py:150-178):

| Component | Value |
|-----------|-------|
| Transformer layers | 3 |
| Hidden dimension | 224 |
| Attention heads | 4 (56d per head) |
| FFN multiplier | 3x (672 FFN dim) |
| Tokens per turn step | 14 (6 own + 6 opp + 1 field + 1 context) |
| Parameters | ~1.95M |
| Value head | Off |
| `prune_dead_features` | True |
| `max_seq_len` | 5 |
| Embedding dims | species=48, move=24, item=16, ability=16, type=12 |

**Note:** The training script (`train_phase4.py`) does **not** use the `p8_lean()` factory. It constructs config via `TransformerConfig.from_vocabs()` with CLI args. This means the P8-Lean profile values (e.g., `max_seq_len=5`) are not automatically applied — they must be passed explicitly via flags like `--num-layers 3 --hidden-dim 224 --num-heads 4 --ffn-multiplier 3 --no-value-head --prune-dead-features`.

### Already-Implemented Infrastructure (No Changes Needed)

The training script already has the following features **fully wired and functional**:

| Feature | Status | Default on CUDA | CLI Flag |
|---------|--------|-----------------|----------|
| Multi-worker DataLoader | Implemented (line 918) | `min(8, cpu_count//2)` | `--num-workers` |
| Prefetch factor | Implemented (line 932) | 4 | `--prefetch-factor` |
| Persistent workers | Implemented (line 922) | True | `--persistent-workers` |
| Pin memory | Implemented (line 920) | True | `--pin-memory` |
| Non-blocking transfer | Implemented (line 924) | True | `--non-blocking-transfer` |
| AMP mixed precision | Implemented (line 903) | bf16 if supported, else fp16 | `--amp auto` |
| torch.compile | Implemented (line 1008) | Off | `--torch-compile` |
| Gradient accumulation | Implemented (line 391) | 1 | `--grad-accum` |
| Dead feature pruning | Implemented (line 1004) | Off | `--prune-dead-features` |
| Max window control | Implemented (line 965) | 20 | `--max-window` |

**These are not optimizations to propose — they already exist.** The question is which of them are being used and which are left on their default settings.

### Training Defaults (Full Mode)

From `train_phase4.py` lines 857-863:

| Setting | Default |
|---------|---------|
| Batch size | 32 |
| Max epochs | 30 |
| Warmup steps | 300 |
| Max window | 20 |
| Patience | 7 |
| `--prune-dead-features` | Not set (off by default) |
| `--torch-compile` | Not set (off by default) |

### Feature Vector State (tensorizer.py)

| Feature | Status | Notes |
|---------|--------|-------|
| Move embeddings | Shared (line 420) | Single `shared_move_emb` used by PokemonEmbedding + ContextEmbedding |
| `terastallized` flag | Prunable (lines 228-263) | 0% activation; removed when `prune_dead_features=True` |
| Field side conditions | Prunable (lines 284-305) | 0% activation; removed when `prune_dead_features=True` |
| Opponent base stats | Always zero (line 259) | `poke.base_stats.get(stat, 0)` returns 0 for opponents |
| `is_own` flag | Present (line 268) | Redundant with `TokenTypeEmbedding` |
| Accuracy/evasion boosts | Separate features (line 254) | Rarely non-zero in Gen 9 OU |
| `enable_nested_tensor` | False (line 444) | Prevents efficient BetterTransformer path |

---

## Confirmed Bugs and Waste

### Bug 1: Redundant Forward Pass in Validation (CRITICAL)

**Location:** `train_phase4.py` lines 452 and 470-475

The `validate()` function runs the model **twice per batch**:

```python
# Line 452: First forward pass — computes loss, policy logits, AND auxiliary preds
loss, loss_dict, logits = forward_step(model, batch, config)

# Lines 470-475: SECOND forward pass — recomputes everything just for aux accuracy
output = model(
    batch["own_team"], batch["opponent_team"], batch["field"], batch["context"],
    legal_mask=batch["legal_mask"], seq_len=batch["seq_len"],
    return_auxiliary=True, return_value=False,
)
```

The `forward_step()` function (line 336) already calls the model with `return_auxiliary=True`, computing all auxiliary predictions. But it only returns `logits` (the third return value, line 368), discarding the full `TransformerOutput`. The validation function then calls the model a second time solely to retrieve the auxiliary predictions it already computed and threw away.

**Impact:** Doubles the validation forward pass compute. Validation is ~25-30% of epoch time, so this wastes **~12-15% of total epoch time**.

**Fix:** Modify `forward_step()` to return the `TransformerOutput` object alongside the loss and logits. Then `validate()` can extract `output.auxiliary_preds` from the first pass.

### Waste 1: Ghost Auxiliary Target Arrays

**Location:** `train_phase4.py` lines 543-568

The `add_auxiliary_labels()` function creates five target arrays per battle:
- `item_targets` — **only one with actual labels** (lines 549-556)
- `speed_targets` — initialized to -1, **never populated** (line 544)
- `role_targets` — initialized to -1, **never populated** (line 545)
- `tera_targets` — initialized to -1, **never populated** (line 546)
- `move_family_targets` — initialized to -1, **never populated** (line 547)

These four ghost arrays are:
1. Created in memory for every battle (numpy allocation)
2. Copied in `WindowedTurnDataset.__getitem__` (lines 196-198, 212-213)
3. Converted to tensors via `torch.from_numpy().long()`
4. Stacked in `collate_windowed` (lines 228-230)
5. Transferred to GPU via `batch = {k: v.to(device) ...}` (line 386)
6. Passed to `compute_total_loss()` → `compute_auxiliary_loss()` where they hit `valid = flat_targets >= 0` → `valid.any()` → False → skipped

**Impact:** Every batch transfers ~4 tensors of shape `(batch_size, 6)` and `(batch_size, 6, 10)` to GPU for zero gradient contribution. Estimated ~2-3% of epoch time wasted on memory allocation, copying, transfer, and filtering.

### Waste 2: Numpy Copy + Conversion on Every Access

**Location:** `train_phase4.py` lines 201-213

`WindowedTurnDataset.__getitem__` calls `.copy()` + `torch.from_numpy()` on every field for every access:

```python
"own_team": torch.from_numpy(own_team.copy()).float(),
"opponent_team": torch.from_numpy(opp_team.copy()).float(),
...
```

With ~170K examples accessed every epoch, this is ~170K × 12 fields = ~2M numpy-copy-then-convert operations per epoch. The `.copy()` is necessary because numpy slice returns a view and `torch.from_numpy` can't handle non-contiguous arrays — but converting battles to tensors once at init time would eliminate this entirely.

**Impact:** Estimated ~3-5% of epoch time.

---

## Optimization Strategies

### Category A: Reduce Computational Work Per Forward Pass

#### A1. Reduce Max Window from 20 to 12 Turns

**Estimated speedup: 10-15%**

Self-attention cost is O(n^2) in sequence length. The transformer currently processes `window × 14` tokens.

| Window | Tokens | Attn FLOPs (relative) |
|--------|--------|----------------------|
| 20 | 280 | 1.00x |
| 15 | 210 | 0.56x |
| 12 | 168 | 0.36x |
| 10 | 140 | 0.25x |

**Why 12:** The P8-Lean config sets `max_seq_len=5` (line 164 of `battle_transformer.py`), meaning its designers expected a very short window. A window of 12 is generous relative to that intent while capturing the 6-10 turn patterns that dominate competitive decision-making. The windowed per-turn dataset still trains on every turn — only the lookback context shortens.

**Quality risk: Low.** The PHASE4_OPTIMIZATION_REPORT flags window=10 as "significant speedup; watch strategic degradation." Window=12 provides comfortable margin.

**Implementation:** `--max-window 12`

---

#### A2. Enable torch.compile

**Estimated speedup: 8-15%**

Already wired at line 1008 but **not enabled by default**. Fuses kernels, eliminates Python overhead, optimizes GPU execution. The model uses standard `nn.TransformerEncoderLayer`, `nn.Embedding`, `nn.Linear` — all well-supported.

First epoch incurs a one-time ~30-60s compilation cost, amortized over subsequent epochs.

**Quality risk: Zero.** Numerically identical computation.

**Implementation:** `--torch-compile`

---

#### A3. Enable Nested Tensors in Transformer Encoder

**Estimated speedup: 3-8%**

`enable_nested_tensor=False` is explicitly set at `battle_transformer.py` line 444. This prevents PyTorch from using the BetterTransformer fast path, which efficiently handles variable-length sequences with padding masks by avoiding computation on padding tokens.

With P8-Lean's windowed dataset, earlier turns in a battle have shorter windows (a turn at position 3 has only 3 turns of context, padded to max_window). Nested tensors skip computation on those padding positions.

**Quality risk: Zero** (same math). Requires testing to confirm the padding mask format (`src_key_padding_mask` bool tensor) is compatible with the nested tensor path.

**Implementation:** Change line 444 of `battle_transformer.py` from `enable_nested_tensor=False` to `enable_nested_tensor=True`.

---

### Category B: Eliminate Wasted Computation

#### B1. Fix Redundant Validation Forward Pass

**Estimated speedup: 12-15% of total epoch time**

As described in Bug 1 above. The fix requires modifying `forward_step()` to return the full `TransformerOutput` alongside loss and logits:

```python
# Current (line 368):
return loss, loss_dict, logits

# Proposed:
return loss, loss_dict, logits, output
```

Then `validate()` uses the returned `output.auxiliary_preds` instead of calling `model()` a second time.

**Quality risk: Zero.**

---

#### B2. Remove Ghost Auxiliary Target Arrays

**Estimated speedup: 2-3%**

As described in Waste 1. Stop creating, copying, and transferring `speed_targets`, `role_targets`, `tera_targets`, and `move_family_targets` since they are entirely -1 and contribute zero gradients.

Changes needed:
1. `add_auxiliary_labels()`: Only create `item_targets`.
2. `WindowedTurnDataset.__getitem__()`: Only copy `item_targets`.
3. `collate_windowed()`: Only stack `item_targets`.
4. `forward_step()`: Only pass `item_targets` in `aux_targets` dict.

**Quality risk: Zero.** These arrays already produce zero loss and zero gradients.

---

#### B3. Pre-convert Battle Arrays to Tensors at Init

**Estimated speedup: 3-5%**

As described in Waste 2. Convert `self.battles` entries from `dict[str, np.ndarray]` to `dict[str, torch.Tensor]` in `WindowedTurnDataset.__init__()`. Replace the per-access `.copy()` + `torch.from_numpy()` pattern with `tensor[start:end].clone()`.

**Quality risk: Zero.** Same data, different memory management. Slightly higher init-time memory cost (tensors use ~same memory as numpy arrays).

---

### Category C: Increase Effective Throughput

#### C1. Increase Batch Size

**Estimated speedup: 5-8%**

The full-mode default is batch_size=32 (line 860). With ~170K examples, that's ~5,310 batches per epoch, each incurring kernel launch and optimizer step overhead. Increasing batch size improves GPU utilization through better parallelism.

Two approaches (choose based on GPU memory):
- `--batch-size 64 --grad-accum 2 --lr 2e-4` (same effective batch, fits more GPUs)
- `--batch-size 128 --lr 4e-4` (requires more VRAM, fewer kernel launches)

Learning rate should scale linearly with effective batch size per standard practice.

**Quality risk: Low.** Linear LR scaling is well-studied. Verify val_loss within 0.02 of baseline.

---

#### C2. Pre-load Dataset to GPU Memory

**Estimated speedup: 2-4% (GPU training only)**

The full dataset is ~170K examples. At ~390 float32 values per example (own_team + opp_team + field + context + legal_mask), that's ~170K × 390 × 4 bytes ≈ 265 MB — easily fits in A100 80GB alongside the 1.95M-parameter model.

Pre-moving all battle tensors to GPU at init time eliminates all host-to-device transfer during training. This makes the already-implemented `pin_memory`, `non_blocking`, and `num_workers` infrastructure unnecessary (data is already on device).

**Quality risk: Zero.**

**Caveat:** Not feasible on consumer GPUs with < 8GB VRAM. Only implement behind a flag.

---

### Category D: Training Recipe Adjustments

#### D1. Reduce Warmup Steps (300 → 150)

**Estimated impact: 1-2% (slightly faster early convergence)**

The 300-step warmup was set for the full 6L/384d model. P8-Lean at 1.95M params stabilizes faster. With batch=32, 300 warmup steps = 5.6% of the first epoch learning at artificially suppressed learning rates.

**Quality risk: Very low.**

**Implementation:** `--warmup-steps 150`

---

#### D2. Reduce Early Stopping Patience (7 → 5)

**Estimated impact: Saves 0-2 tail epochs**

With patience=7, the model trains up to 7 epochs past its best checkpoint. P8-Lean at 1.95M params with 170K examples converges fast — the cosine schedule already decays LR, making late improvements unlikely.

**Quality risk: Low.**

**Implementation:** `--patience 5`

---

### Category E: Feature Engineering for Faster Convergence and Better Accuracy

These changes reduce the **number of epochs** needed to reach target accuracy (and may improve final accuracy), rather than speeding up individual epochs.

#### E1. Fill Opponent Base Stats from Species Lookup

**Estimated convergence improvement: 2-4 fewer epochs to same accuracy**
**Estimated final accuracy improvement: +2-4% top-1**

Currently, 36 dimensions per turn (6 opponent pokemon × 6 base stats) are **always zero** (`tensorizer.py` line 259: `poke.base_stats.get(stat, 0)` returns 0 for opponents with empty `base_stats` dicts).

Base stats are public Pokedex data. When you see "Dragapult," you know it has 142 base Speed, 120 base SpA. Every competitive player has this memorized. The Hidden Information Doctrine permits this — base stats are species-intrinsic public knowledge, not hidden EVs/IVs.

The model currently must learn approximate base stat knowledge from species embeddings and usage patterns across thousands of games. Providing base stats directly eliminates this burden.

**Implementation:**
1. Build a `species_vocab_idx → base_stats` lookup from own-team features in training data (where base stats are populated).
2. During data processing or at `tensorize_pokemon` time, fill opponent base stats from species identification.
3. Reprocess the dataset.

**Quality risk: Negative** (i.e., accuracy improves). The PARAMETER_REDUCTION_PROPOSAL and PHASE4_FEATURE_ANALYSIS both rate this as the single highest-ROI change.

---

#### E2. Remove `is_own` Binary Feature

**Estimated accuracy impact: Neutral to slightly positive**

The `is_own` flag (`tensorizer.py` line 268) is perfectly redundant with `TokenTypeEmbedding` (`battle_transformer.py` lines 350-367), which assigns different learned embeddings to own-team tokens (type 0) vs opponent-team tokens (type 1). The model must either learn to ignore this redundant signal or waste capacity aligning it with the token type signal.

**Quality risk: Very low.** Validate with ablation.

**Implementation:** Remove `is_own` from `tensorize_pokemon`. Adjust `POKEMON_BINARY_DIM`. Requires dataset reprocessing.

---

#### E3. Compress Accuracy/Evasion Stat Boosts

**Estimated accuracy impact: Neutral**

`tensorizer.py` line 254 loops over 7 boost stats including `accuracy` and `evasion`. These are nearly always zero in Gen 9 OU due to Evasion Clause. Merging into a single `accuracy_modifier` or dropping them saves 2 dims × 12 pokemon = 24 input dimensions of near-zero signal.

**Quality risk: Very low.**

---

#### E4. Add Speed Comparison Signal to Context

**Estimated accuracy impact: +1-2% top-1**

Once opponent base stats are filled (E1), a derived `speed_advantage = base_spe_own / (base_spe_own + base_spe_opp)` feature can be trivially added to context. "Do I outspeed?" is one of the most important binary decisions in competitive Pokemon.

**Implementation:** Add 1-2 continuous features to `CONTEXT_FEATURE_DIM` and `tensorize_turn()`. Requires dataset reprocessing.

**Quality risk: Zero.** Derived from already-available public information.

---

## Combined Optimization Plan

### Tier 1: Flag-Only Changes (No Code Edits)

These use already-implemented infrastructure via CLI flags:

| Change | Est. Speedup | Risk |
|--------|-------------|------|
| `--max-window 12` | 10-15% | Low |
| `--torch-compile` | 8-15% | Zero |
| `--batch-size 64 --grad-accum 2 --lr 2e-4` | 5-8% | Low |
| `--warmup-steps 150` | 1-2% | Very Low |
| `--patience 5` | Saves 0-2 epochs | Low |
| `--prune-dead-features` | 1-2% | Zero |

**Combined Tier 1 estimate: 25-35% per-epoch speedup** (effects partially overlap).

Recommended command:
```bash
python scripts/train_phase4.py \
  --mode full \
  --num-layers 3 --hidden-dim 224 --num-heads 4 --ffn-multiplier 3 \
  --no-value-head --prune-dead-features \
  --max-window 12 \
  --torch-compile \
  --batch-size 64 --grad-accum 2 --lr 2e-4 \
  --warmup-steps 150 --patience 5
```

### Tier 2: Targeted Code Fixes (~50 Lines Changed)

| Change | Est. Speedup | Lines Changed |
|--------|-------------|---------------|
| Fix redundant validation forward pass (B1) | 12-15% | ~15 |
| Remove ghost auxiliary targets (B2) | 2-3% | ~20 |
| Pre-convert numpy to tensors at init (B3) | 3-5% | ~30 |
| Enable nested tensors (A3) | 3-8% | 1 |

**Combined Tier 2 estimate: 15-25% additional speedup.**

### Tier 3: Data Pipeline Changes (Requires Dataset Reprocessing)

| Change | Effect | Risk |
|--------|--------|------|
| Fill opponent base stats (E1) | 2-4 fewer epochs + better accuracy | Improves accuracy |
| Remove `is_own` redundancy (E2) | Cleaner representation | Very Low |
| Compress accuracy/evasion boosts (E3) | Slightly smaller input | Very Low |
| Add speed comparison signal (E4) | +1-2% accuracy | Zero |

---

## Projected Combined Impact

### Conservative (Tier 1 only)
- Per-epoch time: 140s → ~100s (28% reduction)
- **Meets the 30% target with flag changes alone**

### Moderate (Tier 1 + Tier 2)
- Per-epoch time: 140s → ~75s (46% reduction)
- Reduced tail epochs via patience: save 1-2 epochs
- **Total time reduction: ~45-50%**

### Full (Tier 1 + Tier 2 + Tier 3)
- Per-epoch time: ~75s
- Epochs to convergence: ~15 (down from ~20, due to opponent base stats)
- **Total time reduction: ~55-60% with accuracy improvement**

---

## Redundant Features to Remove

| Feature | Location | Current Dims | Activation Rate | Verdict |
|---------|----------|-------------|-----------------|---------|
| `terastallized` flag | tensorizer.py:276 | 1 × 12 pokemon | 0.0% | **Remove** via `--prune-dead-features` |
| Field side conditions | tensorizer.py:310-327 | 16 | 0.0% | **Remove** via `--prune-dead-features` |
| `is_own` flag | tensorizer.py:268 | 1 × 12 pokemon | Deterministic | **Remove** (redundant with TokenTypeEmbedding) |
| Accuracy boost | tensorizer.py:254 | 1 × 12 pokemon | ~0.1% | **Merge** with evasion or drop |
| Evasion boost | tensorizer.py:254 | 1 × 12 pokemon | ~0.1% | **Merge** with accuracy or drop |
| Opponent base stats (zeros) | tensorizer.py:259 | 6 × 6 opp pokemon | 0.0% | **Fill** from Pokedex |

**Net effect:** Remove ~52 dead/redundant dims. Fill 36 zero-dims with high-value Pokedex data. Input becomes smaller AND more informative.

---

## Essential Features to Add

| Feature | Dims | Cost | Accuracy Impact |
|---------|------|------|-----------------|
| Opponent base stats from Pokedex | 0 (fill existing zeros) | Low (one-time reprocessing) | Very High (+2-4%) |
| Speed comparison signal | 1-2 (in context) | Very Low | High (+1-2%) |

---

## Risk Assessment

| Optimization | Risk | Mitigation |
|-------------|------|-----------|
| Window reduction (20→12) | Low | Fallback to 15; compare per-action accuracy for late-game switches |
| torch.compile | Zero | Numerically identical; disable flag if compilation fails |
| Batch size increase (32→64) | Low | Linear LR scaling; verify val_loss within 0.02 of baseline |
| Validation forward pass fix | Zero | Code-level optimization; same output |
| Remove ghost aux targets | Zero | Already produce zero gradients |
| Nested tensors | Zero | Smoke test first; revert line 444 if incompatible |
| Opponent base stats fill | Negative (improves accuracy) | Public Pokedex data; HID-compliant |

---

## Validation Protocol

1. **Baseline:** Run P8-Lean with current defaults. Record top-1, top-3, val_loss, epoch_time.
2. **Tier 1:** Apply flag changes only. Compare metrics.
3. **Tier 1 + 2:** Apply code fixes. Compare.
4. **Full stack:** Apply feature pipeline changes. Compare.

**Acceptance criteria** (from PHASE4_25K_COMPUTE_GENERALIZATION_EXPERIMENT.md):
- Top-1 accuracy drop ≤ 1.0 absolute point vs baseline
- Top-3 accuracy drop ≤ 0.8 absolute point vs baseline
- Generalization gap increase ≤ 1.0 absolute point
- Compute reduction ≥ 25% in wall-clock to best checkpoint

---

## Summary

The 30% target is achievable through **Tier 1 flag changes alone** — no code edits required. The most impactful levers are window reduction (10-15%), `torch.compile` (8-15%), and batch size increase (5-8%). These use infrastructure already built into `train_phase4.py`.

Tier 2 code fixes — particularly eliminating the redundant validation forward pass (12-15% of epoch time wasted) — push savings to ~45-50%.

Tier 3 feature changes (filling opponent base stats) don't speed up per-epoch time but reduce total epochs to convergence while improving final accuracy. This is the highest-ROI change for model quality and should be prioritized alongside the speed work.
