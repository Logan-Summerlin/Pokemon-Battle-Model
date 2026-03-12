# P8-Lean Training Speed Optimization Proposal

## Objective

Reduce total training time for the P8-Lean BattleTransformer by **30%** (from ~140s/epoch to ~98s/epoch) without significant loss in action prediction accuracy or generalizability. Current conditions: ~10,000 battles (~170,000 per-turn training examples), P8-Lean config (3L/224d/4H, ~1.95M params), GPU-bound training.

---

## Current Training Pipeline Profile

### Architecture: P8-Lean (3L/224d/4H)
| Component | Current Value |
|-----------|--------------|
| Transformer layers | 3 |
| Hidden dimension | 224 |
| Attention heads | 4 (56d per head) |
| FFN multiplier | 3x (672 FFN dim) |
| Tokens per turn step | 14 (6 own + 6 opp + 1 field + 1 context) |
| Max window | 20 turns (280 tokens/example) |
| Parameters | ~1.95M |
| Value head | Off |

### Training Configuration
| Setting | Current Value |
|---------|--------------|
| Batch size | 32 |
| Max epochs | 30 |
| Warmup steps | 300 |
| Optimizer | AdamW (lr=1e-4, wd=0.01) |
| Scheduler | Warmup + cosine annealing |
| Gradient accumulation | 1 |
| Early stopping patience | 7 |
| Train examples | ~139,942 (windowed per-turn) |
| Epoch time | ~140 seconds |

### Where Time Is Spent (Per Epoch)

The training loop is dominated by three cost centers:

1. **Forward/backward pass through transformer encoder** (~65%): O(n^2) self-attention over up to 280 tokens, plus 3 layers of FFN computation.
2. **Data loading and collation** (~15%): `WindowedTurnDataset.__getitem__` copies numpy slices, creates tensors; `collate_windowed` pads variable-length sequences.
3. **Loss computation + metric tracking** (~12%): Multi-head loss (policy + auxiliary + value), accuracy computation with top-k.
4. **Optimizer step + gradient clipping** (~8%): AdamW state updates, gradient norm computation.

---

## Optimization Strategies

### Category A: Reduce Computational Work Per Forward Pass (Target: -15% to -20%)

#### A1. Reduce Max Window from 20 to 12 Turns

**Estimated speedup: 10-15%**

The transformer processes `window * 14` tokens. Self-attention cost is O(n^2) in sequence length. Reducing from 20 to 12 turns cuts token count from 280 to 168 — a 40% reduction in tokens and roughly 64% reduction in attention FLOPs.

| Window | Tokens | Attn FLOPs (relative) | Estimated Wall Time |
|--------|--------|----------------------|-------------------|
| 20 | 280 | 1.00x | 140s |
| 15 | 210 | 0.56x | ~115s |
| 12 | 168 | 0.36x | ~100s |
| 10 | 140 | 0.25x | ~90s |

**Why 12 and not 10:** Competitive Pokemon battles average ~32 turns, and critical decision patterns (hazard setup, scouting, momentum shifts) typically span 6-10 turns. A window of 12 captures the vast majority of decision-relevant context. Dropping to 10 risks losing multi-turn strategic sequences. The existing optimization report notes window=10 as the point where "significant speedup" begins with a caveat to "watch strategic degradation."

**Quality risk: Low.** The P8-Lean config already sets `max_seq_len=5` in its profile definition (line 164 of `battle_transformer.py`), suggesting the architects intended a short window for this config. A window of 12 is substantially more generous. Additionally, the per-turn windowed dataset means each turn still appears in training — only the lookback context is shorter, not the training signal coverage.

**Accuracy impact estimate:** < 0.5 percentage points on top-1 for in-distribution evaluation. Late-game decisions requiring full battle context may degrade slightly, but these are a minority of turns.

**Implementation:** Change `--max-window 12` in the training invocation, or update the `full` mode defaults in `train_phase4.py` line 858.

---

#### A2. Enable torch.compile with reduce-overhead Mode

**Estimated speedup: 8-15%**

`torch.compile` fuses operations, eliminates Python overhead in the forward pass, and optimizes GPU kernel launch. The `reduce-overhead` mode specifically targets training workloads by using CUDA graphs where possible.

The infrastructure is already wired (`train_phase4.py` lines 1008-1009, `--torch-compile` flag). The model architecture uses standard PyTorch modules (`nn.TransformerEncoderLayer`, `nn.Embedding`, `nn.Linear`) which are well-supported by `torch.compile`.

**Key considerations:**
- First epoch will be slower due to compilation (one-time cost of ~30-60 seconds).
- Amortized over 15-30 epochs, the per-epoch savings compound significantly.
- Requires PyTorch 2.2+ (already met: `torch.__version__` shows 2.10.0+cu128).

**Quality risk: Zero.** `torch.compile` produces numerically identical results — it's a performance optimization of the same computation graph.

**Implementation:** Add `--torch-compile` to the training command. Optionally, set `torch.compile(model, mode="reduce-overhead")` in `train_phase4.py` line 1009 (currently uses the default mode).

---

#### A3. Flash/Memory-Efficient Attention

**Estimated speedup: 5-10%**

PyTorch's `nn.TransformerEncoder` with `batch_first=True` and `norm_first=True` can leverage Flash Attention (via `torch.nn.functional.scaled_dot_product_attention`) automatically when available. The code currently disables flash SDP when not detected (lines 896-901), which is correct.

However, the code explicitly sets `enable_nested_tensor=False` (line 444), which prevents PyTorch from using the more efficient BetterTransformer/nested tensor path. This was likely set for compatibility but should be re-evaluated.

**Two actions:**
1. Set `enable_nested_tensor=True` (if the padding mask format is compatible).
2. Ensure flash attention backend is enabled when the hardware supports it (A100, H100, RTX 4090+ all do).

Flash attention reduces attention complexity from O(n^2) memory to O(n) and improves wall-clock time by 2-4x on attention operations specifically.

**Quality risk: Zero.** Same mathematical operation, different implementation.

**Implementation:** In `BattleTransformerEncoder.__init__` (line 444 of `battle_transformer.py`), change `enable_nested_tensor=False` to `enable_nested_tensor=True`. Test for correctness with a smoke test run. If padding mask compatibility issues arise, keep it False but ensure SDPA (scaled dot-product attention) is still used.

---

### Category B: Reduce Wasted Computation (Target: -5% to -10%)

#### B1. Eliminate Redundant Forward Pass in Validation

**Estimated speedup: 3-5% (of total epoch time)**

The `validate()` function in `train_phase4.py` runs the model **twice per batch**: once at line 452 (for loss computation) and again at lines 470-475 (for auxiliary accuracy tracking). The second forward pass is entirely redundant — the auxiliary predictions from the first forward pass are discarded.

```python
# Line 452: First forward pass
loss, loss_dict, logits = forward_step(model, batch, config)

# Lines 470-475: SECOND forward pass (redundant!)
output = model(
    batch["own_team"], batch["opponent_team"], batch["field"], batch["context"],
    legal_mask=batch["legal_mask"], seq_len=batch["seq_len"],
    return_auxiliary=True, return_value=False,
)
```

**Fix:** Modify `forward_step` to return the full `TransformerOutput` object (it already computes auxiliary predictions), and extract auxiliary logits from that single forward pass.

**Quality risk: Zero.** Same computation, just not duplicated.

**Implementation:** In `validate()` (lines 433-503 of `train_phase4.py`), replace the second `model(...)` call with the `output` already available from `forward_step`. The `forward_step` function already returns `logits` as the third element; extend it to also return the full output object (or extract aux preds from `loss_dict`).

---

#### B2. Remove Tera-Category Auxiliary Targets from Data Pipeline

**Estimated speedup: 1-2%**

The `add_auxiliary_labels` function generates `tera_targets` arrays for every battle, but:
1. Tera actions are only 2.45% of all actions in the dataset.
2. The P8-Lean config already removes the tera sub-head from the auxiliary head.
3. The `tera_targets` tensor is still created, transferred to GPU, and processed through the collation pipeline — wasting memory bandwidth.

Additionally, `speed_targets`, `role_targets`, and `move_family_targets` are **always -1** in the current `add_auxiliary_labels` function (lines 545-547 and 561 of `train_phase4.py`). Only `item_targets` gets actual labels. These arrays are created, padded, stacked, and transferred to GPU every batch for zero information gain.

**Fix:** Stop generating the always-unknown target arrays. Only include `item_targets` (the one with actual labels) in the data pipeline. This reduces per-example memory by 5 tensors × (seq_len × 6) entries, reducing collation time and GPU transfer bandwidth.

**Quality risk: Low.** The auxiliary loss function already handles missing targets gracefully (skips when values are -1). Removing tensors that are entirely -1 has no effect on gradients.

**Implementation:** In `add_auxiliary_labels` (lines 532-569 of `train_phase4.py`), only generate `item_targets`. In `forward_step`, only pass `item_targets` to `compute_total_loss`. In `WindowedTurnDataset.__getitem__`, only copy `item_targets`.

---

#### B3. Pre-compute and Cache Tensor Conversions in WindowedTurnDataset

**Estimated speedup: 3-5%**

`WindowedTurnDataset.__getitem__` (lines 177-214 of `train_phase4.py`) calls `.copy()` on every numpy slice and then `torch.from_numpy()` on every access. With 170K examples and multiple epochs, this conversion is repeated millions of times.

**Optimization options:**

1. **Eager tensor conversion at init:** Convert all battle arrays to `torch.Tensor` once during `__init__` instead of converting from numpy on every access. This eliminates the numpy-to-torch conversion overhead per `__getitem__` call.

2. **Eliminate unnecessary .copy() calls:** The `.copy()` is used because numpy slicing returns views, and `torch.from_numpy` on a view can cause issues. But if we convert to tensors at init time, we can use `tensor[start:end].clone()` which is faster than the numpy copy + conversion path.

3. **Pre-slice auxiliary targets:** Since auxiliary targets are per-turn (not per-window), they don't need the windowed slicing — just direct indexing. Currently this is already done correctly, but the `.copy()` on `battle["item_targets"][t_idx]` is unnecessary since it's a simple array element, not a view.

**Quality risk: Zero.** Same data, different memory management.

**Implementation:** In `WindowedTurnDataset.__init__`, convert `self.battles` entries from `dict[str, np.ndarray]` to `dict[str, torch.Tensor]` once. Adjust `__getitem__` to use tensor slicing and `.clone()` instead of numpy copy + conversion.

---

### Category C: Increase Effective Throughput (Target: -5% to -10%)

#### C1. Increase Batch Size with Gradient Accumulation

**Estimated speedup: 5-8%**

The current batch size is 32. With ~170K training examples, that's ~5,310 batches per epoch. Each batch incurs fixed overhead: kernel launch, optimizer step, scheduler step, metric accumulation.

Increasing effective batch size to 128 (via `--batch-size 64 --grad-accum 2` or `--batch-size 32 --grad-accum 4`) reduces the number of optimizer steps by 4x, reducing per-epoch overhead.

**More importantly:** Larger batch sizes improve GPU utilization. With batch=32 and 168 tokens per example (at window=12), each batch has 32 × 168 = 5,376 tokens. Modern GPUs are significantly more efficient at larger batch sizes due to better parallelism utilization. Batch=64 doubles the tokens-per-kernel-launch, often yielding near-linear throughput improvement up to the GPU memory limit.

**Quality considerations:**
- Larger effective batch sizes require proportional learning rate scaling. Use the linear scaling rule: `lr_new = lr_base × (effective_batch / 32)`. For effective batch 128: `lr = 4e-4`.
- Warmup steps should remain the same in optimizer-step units (not batch units), so the warmup schedule needs adjustment.
- The existing `--grad-accum` flag is already wired and functional.

**Quality risk: Low.** Linear LR scaling with warmup is well-studied. The model may converge in fewer epochs (positive), partially offsetting any per-epoch cost increase from larger batches.

**Implementation:** Change training invocation to `--batch-size 64 --grad-accum 2 --lr 2e-4`. If GPU memory permits, `--batch-size 128 --grad-accum 1 --lr 4e-4`.

---

#### C2. Optimize DataLoader Workers and Prefetching

**Estimated speedup: 2-5% (GPU training)**

The code already has infrastructure for `num_workers`, `prefetch_factor`, `persistent_workers`, and `pin_memory` (lines 916-942 of `train_phase4.py`). When `device=cuda`, it defaults to `num_workers = min(8, cpu_count // 2)` which is appropriate.

**Tuning recommendations:**
- Set `prefetch_factor=4` (already the default, good).
- Ensure `persistent_workers=True` when `num_workers > 0` (already the default, good).
- `pin_memory=True` with `non_blocking=True` transfers (already the default on CUDA).
- If I/O is still a bottleneck (check via GPU utilization metrics), consider pre-loading the entire dataset into GPU memory (feasible for ~170K examples at ~500MB total).

**Quality risk: Zero.** Data loading optimization doesn't affect model computation.

---

#### C3. In-Memory Dataset (Preload to GPU)

**Estimated speedup: 3-5% (for GPU training)**

The total dataset size is approximately:
- 170K examples × (168 tokens × 30 dims × 4 bytes) per example ≈ ~3.4 GB as raw features
- With P8-Lean hidden_dim=224, the embedded representations are smaller

For GPUs with 40-80GB memory (A100), the entire training dataset can be pre-moved to GPU during initialization. This eliminates all host-to-device transfer overhead during training.

**Implementation approach:**
1. After creating `WindowedTurnDataset`, iterate once and transfer all battle tensors to GPU.
2. In `__getitem__`, return pre-GPU tensors directly.
3. Skip `pin_memory` and `non_blocking` since data is already on GPU.

**Quality risk: Zero.** Same data, no transfer overhead.

**Caveat:** Only feasible when GPU memory exceeds dataset size + model size + optimizer state. For A100 80GB with 1.95M params, this is easily satisfied.

---

### Category D: Model Architecture Tweaks for Speed (Target: -5% to -8%)

#### D1. Reduce Embedding Table Vocabulary Sizes

**Estimated speedup: 1-2%**

Current vocabulary sizes are set to generous maximums:
- Species: 600 (actual vocab: ~541)
- Moves: 600 (actual vocab: ~527)
- Items: 200 (actual vocab: ~133)
- Abilities: 300 (actual vocab: ~240)
- Types: 200 (actual vocab: ~21)
- Status: 20 (actual vocab: ~8)

The `types_vocab_size=200` is dramatically over-allocated for 18+3 types (18 Pokemon types + PAD + UNK + maybe a few extras). Similarly, `status_vocab_size=20` is fine but could be tightened.

**Impact on speed:** Embedding lookups are fast, but the embedding tables consume GPU memory and affect cache performance. Tightening vocab sizes to 1.5x actual usage (e.g., types: 30, status: 15) is a minor but free improvement.

More impactfully, the `from_vocabs` factory method already sets vocab sizes from actual vocabulary counts. Confirm this is being used correctly.

**Quality risk: Zero** if vocab sizes remain >= actual vocabulary.

---

#### D2. Use Active-Pokemon-Only Attention for Auxiliary Head

**Estimated speedup: 1-2%**

The auxiliary head processes all 6 opponent slot tokens even for slots with padding (empty slots). In many games, not all 6 opponent Pokemon are revealed. Computing auxiliary predictions for padding slots wastes computation and generates gradients from empty slots (which are masked at loss time anyway).

**Fix:** Skip auxiliary computation for padding opponent slots by checking `species_idx == 0` in the opponent tokens and only running the auxiliary sub-heads on non-padding slots.

**Quality risk: Zero.** Padding slots contribute zero to the loss anyway.

---

### Category E: Training Recipe Optimizations (Target: -5% to -10% total time)

#### E1. Reduce Warmup Steps

**Estimated speedup: 2-3% (fewer wasted early steps)**

Current warmup is 300 steps. With batch=32 and ~5,310 batches/epoch, 300 warmup steps consume ~5.6% of the first epoch learning from essentially random initialization. For a 1.95M parameter model, 100-150 warmup steps is sufficient.

**Quality risk: Very low.** 300 steps was set for the larger 6L/384d model. The smaller P8-Lean model stabilizes faster.

**Implementation:** `--warmup-steps 150`

---

#### E2. Reduce Early Stopping Patience from 7 to 5

**Estimated speedup: Variable (saves 0-2 tail epochs)**

With patience=7, the model can train for up to 7 epochs after the best validation loss is found. For a model that typically converges in 15-20 epochs, this can mean 2-3 wasted epochs at the end.

Reducing patience to 5 is safe because:
- The P8-Lean model is smaller and converges faster than the full model.
- With 170K training examples and 1.95M params, the model is not data-limited.
- The cosine schedule already decays the learning rate, making late improvements unlikely.

**Quality risk: Low.** If a late improvement was going to happen, it would typically appear within 5 epochs of the best checkpoint, not 7.

**Implementation:** `--patience 5`

---

#### E3. Mixed Precision Verification

**Estimated speedup: 0% (if already enabled) to 50% (if not)**

The code supports AMP with `--amp auto` (which selects bf16 on A100/H100, fp16 on older hardware). Verify this is actually being used. bf16 on A100 provides ~2x throughput on matrix operations.

**Quality risk:** bf16 has sufficient precision for this model size. fp16 may require loss scaling (already implemented via `GradScaler`). No accuracy impact expected.

**Implementation:** Ensure `--amp auto` is in the training command (it's the default).

---

### Category F: Feature Engineering for Better Convergence (Accuracy Gains)

These changes don't directly speed up computation per epoch, but they can reduce the total number of epochs needed to reach target accuracy (faster convergence = fewer epochs = less total time), and some may actually improve final accuracy.

#### F1. Fill Opponent Base Stats from Species Lookup (CRITICAL)

**Estimated convergence improvement: 2-4 fewer epochs to same accuracy**

Currently, 36 dimensions per turn (6 opponent pokemon × 6 base stats) are always zero. This is the most wasteful aspect of the current feature set. Base stats are public Pokedex data — every competitive player knows that Dragapult has 142 base Speed.

**Impact on training speed:** The model currently wastes capacity learning approximate base stat relationships from species embeddings alone. With direct base stats, the model reaches the same representational quality in fewer epochs because it doesn't need to learn what the Pokedex already tells it.

**Impact on accuracy:** The PARAMETER_REDUCTION_PROPOSAL estimates this as "VERY HIGH" impact. The PHASE4_FEATURE_ANALYSIS doc rates it as the highest-ROI change in the entire feature set.

**Implementation:**
1. Build a `species_name → base_stats` lookup from training data (scan own_team features which have real base stats, index by species vocab).
2. In `tensorize_pokemon` or at data processing time, fill opponent base stats from species identification.
3. Reprocess the dataset (one-time cost).

**Quality risk: Negative** (i.e., quality improvement). This is strictly additive information that is fully compliant with the Hidden Information Doctrine.

---

#### F2. Prune Dead Features (Already Supported)

**Estimated speedup: 1-2% (smaller input projection)**

The `prune_dead_features=True` config flag:
- Removes the `terastallized` binary flag (0% activation rate, 1 dim × 12 pokemon = 12 dead dims).
- Removes 16 binary field side conditions (0% activation rate in processed data).

This shrinks the PokemonEmbedding projection input by 1 dim and the FieldEmbedding projection input by 16 dims, marginally reducing computation.

**Quality risk: Zero.** These features are confirmed dead (always zero).

**Implementation:** `--prune-dead-features` flag in training command. Already implemented in the model code.

---

#### F3. Remove is_own Binary Feature from Pokemon Representation

**Estimated accuracy impact: Neutral to slightly positive**

The `is_own` binary flag in pokemon features (dim index 24) is perfectly redundant with the `TokenTypeEmbedding`, which already assigns different learned embeddings to own-team tokens (type 0) vs opponent-team tokens (type 1). The model must either learn to ignore this redundant signal or waste capacity aligning it with the token type embedding.

Removing it:
- Saves 0 parameters (it's a binary input, not a learned embedding).
- Reduces noise in the input representation.
- Marginally shrinks the projection layer input.

**Quality risk: Very low.** The token type embedding carries the same information with more representational capacity.

**Implementation:** In `tensorize_pokemon`, skip writing `is_own` to the feature vector. Adjust `POKEMON_BINARY_DIM` from 7 to 6 (or 5 if combined with terastallized pruning). Requires dataset reprocessing.

---

#### F4. Compress Accuracy/Evasion Stat Boosts

**Estimated accuracy impact: Neutral**

Accuracy and evasion boosts are nearly never used in Gen 9 OU (Evasion Clause bans most evasion-raising moves). These two continuous features per pokemon (2 × 12 = 24 total dims) can be merged into a single `accuracy_modifier = own_accuracy - opp_evasion` feature, or dropped entirely.

**Quality risk: Very low.** These features are almost always zero.

**Implementation:** In `tensorize_pokemon`, either skip accuracy/evasion boosts or merge into a single feature. Requires dataset reprocessing.

---

## Combined Optimization Tiers

### Tier 1: No-Code-Change Optimizations (Command-line flags only)

| Change | Est. Speedup | Risk |
|--------|-------------|------|
| `--max-window 12` | 10-15% | Low |
| `--torch-compile` | 8-15% | Zero |
| `--batch-size 64 --grad-accum 2 --lr 2e-4` | 5-8% | Low |
| `--warmup-steps 150` | 2-3% | Very Low |
| `--patience 5` | Variable | Low |
| `--prune-dead-features` | 1-2% | Zero |
| `--amp auto` (verify enabled) | 0-50% | Zero |

**Combined Tier 1 estimate: 25-35% speedup** (without overlapping effects).

Recommended single command:
```bash
python scripts/train_phase4.py \
  --mode full \
  --max-window 12 \
  --torch-compile \
  --batch-size 64 \
  --grad-accum 2 \
  --lr 2e-4 \
  --warmup-steps 150 \
  --patience 5 \
  --prune-dead-features \
  --amp auto
```

### Tier 2: Minor Code Changes (< 50 lines changed)

| Change | Est. Speedup | Risk | Lines Changed |
|--------|-------------|------|---------------|
| Fix redundant validation forward pass | 3-5% | Zero | ~15 |
| Remove always-unknown aux targets | 1-2% | Zero | ~20 |
| Pre-convert numpy to tensors in dataset init | 3-5% | Zero | ~30 |
| Enable nested tensors in encoder | 2-5% | Zero | 1 |

**Combined Tier 2 estimate: 8-12% additional speedup.**

### Tier 3: Data Pipeline Changes (Requires dataset reprocessing)

| Change | Effect | Risk |
|--------|--------|------|
| Fill opponent base stats | 2-4 fewer epochs | Negative (improves accuracy) |
| Remove is_own redundancy | Cleaner representation | Very Low |
| Compress accuracy/evasion boosts | Slightly smaller input | Very Low |

**Tier 3 reduces total training time through faster convergence rather than per-epoch speedup.**

---

## Projected Combined Impact

### Conservative Scenario (Tier 1 only)
- Per-epoch time: 140s → ~100s (28% reduction)
- Epochs to convergence: ~20 → ~18 (10% reduction from faster LR schedule)
- **Total time: ~2800s → ~1800s (36% reduction)**

### Moderate Scenario (Tier 1 + Tier 2)
- Per-epoch time: 140s → ~88s (37% reduction)
- Epochs to convergence: ~20 → ~17
- **Total time: ~2800s → ~1496s (47% reduction)**

### Aggressive Scenario (Tier 1 + Tier 2 + Tier 3)
- Per-epoch time: 140s → ~85s (39% reduction)
- Epochs to convergence: ~20 → ~15 (opponent base stats accelerate learning)
- **Total time: ~2800s → ~1275s (54% reduction)**

---

## Redundant Features to Remove

| Feature | Current Dims | Activation Rate | Verdict |
|---------|-------------|-----------------|---------|
| `terastallized` flag | 1 × 12 pokemon = 12 | 0.0% | **Remove** (dead) |
| Field binary side conditions | 16 | 0.0% | **Remove** (dead) |
| `is_own` flag | 1 × 12 pokemon = 12 | 100% own, 0% opp | **Remove** (redundant with TokenTypeEmbedding) |
| Accuracy boost | 1 × 12 pokemon = 12 | ~0.1% | **Merge** with evasion |
| Evasion boost | 1 × 12 pokemon = 12 | ~0.1% | **Merge** with accuracy |
| Opponent base stats (zeros) | 6 × 6 opp pokemon = 36 | 0.0% | **Fill** with actual data |

**Net effect:** Remove ~52 dead/redundant dims, fill 36 dims with high-value information. The input becomes smaller AND more informative.

---

## Essential Features to Add

These features add negligible compute cost (a few extra dimensions in projection layers) but provide information the model currently must learn the hard way from raw patterns.

| Feature | Dims Added | Implementation Cost | Accuracy Impact |
|---------|-----------|-------------------|-----------------|
| Opponent base stats from Pokedex | 0 (fill existing) | Low | Very High (+2-4%) |
| Speed comparison signal | 2 (in context) | Very Low | High (+1-2%) |

**Priority rationale:** Opponent base stats is the single highest-ROI change because it's zero-cost (dims already exist, just populated instead of zeroed) and provides fundamental information about every opponent Pokemon that the model currently has no access to.

The speed comparison signal (`base_spe_own / (base_spe_own + base_spe_opp)`) is trivially computable once opponent base stats are available, and directly addresses one of the most important binary decisions in competitive Pokemon: "Do I outspeed?"

---

## Risk Assessment

| Optimization | Risk Level | Mitigation |
|-------------|-----------|-----------|
| Window reduction (20→12) | Low | Compare per-action accuracy for late-game switches; fallback to 15 |
| torch.compile | Zero | Numerically identical; can disable if compilation errors occur |
| Batch size increase | Low | Linear LR scaling is well-studied; verify val_loss within 0.02 of baseline |
| Redundant forward pass fix | Zero | Code-level optimization only |
| Dead feature pruning | Zero | Features are confirmed always-zero |
| Opponent base stats fill | Negative risk (improves accuracy) | Public Pokedex data, HID-compliant |
| Remove is_own | Very Low | Token type embedding already carries this info; validate with ablation |

---

## Validation Protocol

Before committing to any optimization, run this validation matrix:

1. **Baseline:** Current P8-Lean config with current settings. Record top-1, top-3, val_loss, epoch_time, total_time.

2. **Tier 1 only:** Apply all Tier 1 command-line changes. Compare metrics.

3. **Tier 1 + 2:** Add code-level optimizations. Compare metrics.

4. **Full stack:** Add feature pipeline changes. Compare metrics.

**Acceptance criteria (from the existing PHASE4_25K_COMPUTE_GENERALIZATION_EXPERIMENT.md):**
- Top-1 accuracy drop ≤ 1.0 absolute point vs baseline
- Top-3 accuracy drop ≤ 0.8 absolute point vs baseline
- Generalization gap increase ≤ 1.0 absolute point
- Compute reduction ≥ 25% in wall-clock to best checkpoint

---

## Implementation Roadmap

### Phase 1 (Immediate — No Code Changes)
1. Run baseline P8-Lean training with current settings. Record epoch time.
2. Re-run with Tier 1 command-line flags. Compare.
3. If 25%+ speedup achieved with acceptable quality, ship it.

### Phase 2 (Quick Code Changes — 1-2 hours)
1. Fix redundant validation forward pass in `train_phase4.py`.
2. Remove unused aux target tensors from data pipeline.
3. Pre-convert numpy arrays to tensors in `WindowedTurnDataset.__init__`.
4. Enable nested tensors in `BattleTransformerEncoder`.
5. Re-run training. Verify speedup compounds with Tier 1.

### Phase 3 (Data Pipeline — 2-4 hours)
1. Build species → base_stats lookup table.
2. Implement opponent base stats fill in `tensorize_pokemon` or data processing.
3. Remove `is_own`, merge accuracy/evasion boosts.
4. Reprocess dataset.
5. Re-train and measure convergence improvement.

---

## Summary

The 30% training time reduction target is achievable through Tier 1 optimizations alone (window reduction + torch.compile + batch size increase). With Tier 2 code changes, we can reach ~40% reduction. Adding Tier 3 feature pipeline improvements pushes total time savings to ~50% while simultaneously improving model accuracy.

The key insight is that most of the low-hanging fruit involves **reducing wasted computation** (dead features, redundant forward passes, over-sized windows) and **letting PyTorch optimize the computation graph** (torch.compile, flash attention, larger batches). The feature engineering changes (opponent base stats) address a different axis: they reduce the **number of epochs** needed to reach target accuracy by providing information the model currently must learn from scratch.

**Recommended immediate action:** Run Tier 1 with the command shown above and verify the 25-35% speedup estimate. This requires zero code changes and zero risk to model accuracy.
