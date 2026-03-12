# P8-Lean Training Speed Optimization Proposal (Audited + Corrected)

## Objective

Reduce **wall-clock training time** for P8-Lean at 10,000 battles without degrading action-policy accuracy or generalizability.

This revision is a hard audit of the previous proposal against the **actual current pipeline** and removes outdated or unsupported recommendations.

---

## What the Current P8-Lean Pipeline Actually Is

### Entry point and defaults

The canonical P8-Lean launcher is `scripts/train_p8_lean.py`, which invokes `scripts/train_phase4.py` with P8-Lean architecture and training flags. Notably:

- P8-Lean architecture: `3L / 224d / 4H / ffn_multiplier=3`
- `--max-window` default is **5** (not 20)
- `--batch-size` default is **64** (not 32)
- `--no-value-head` and `--prune-dead-features` are always set
- `--warmup-steps` default 300, `--patience` default 7

So any proposal assuming the P8-Lean baseline is “window 20, batch 32, value head on” is outdated for this repo’s current P8-Lean path.

### Actual training core

`train_phase4.py` does the following:

- Loads tensorized battle `.npz` files and battle-level split (80/10/10)
- Adds auxiliary labels via `add_auxiliary_labels()`
- Expands each battle to per-turn examples using `WindowedTurnDataset`
- Trains `BattleTransformer` using masked policy loss + auxiliary loss (+ optional value loss)
- Evaluates val each epoch and test at end

### Model internals relevant to speed

In `src/models/battle_transformer.py`:

- Tokens per step are fixed at 14
- Transformer encoder uses `nn.TransformerEncoder`
- `enable_nested_tensor=False`
- Auxiliary head still computes item/speed/role/move-family logits

---

## Harsh Audit of the Previous Proposal

## Removed / corrected as incorrect or outdated

1. **“Reduce max window 20→12” as a primary P8-Lean lever**
   - Outdated for P8-Lean path. P8-Lean launcher already defaults to `max_window=5`.
   - Not a valid primary speed recommendation for the requested P8-Lean 10K setup.

2. **“Increase batch size 32→64” as a main speedup**
   - Outdated for P8-Lean path. Baseline is already batch 64.

3. **Claims that warmup/patience produce direct per-epoch speedups**
   - Incorrect framing. These can reduce total epochs sometimes, but do not improve step throughput.

4. **Feature-engineering items (opponent base stats, speed features, etc.) presented as training-speed optimizations**
   - These are representation/quality proposals, not direct throughput wins.
   - They require dataset schema changes/reprocessing and are outside “speed without quality loss” guarantees.

5. **Nested tensor claim as guaranteed win**
   - Not guaranteed. Benefit is workload/version-dependent, and can regress or break with certain masking/compile combinations.
   - Keep as optional benchmark item, not a committed correctness-safe recommendation.

---

## Valid, High-Confidence Speed Opportunities (No Accuracy Loss Expected)

## Tier 1 — Code-level compute waste removal (highest ROI)

### 1) Remove duplicate validation forward pass

`validate()` currently calls `forward_step()` and then does a second full model forward only to read auxiliary preds. This is redundant because `forward_step()` already performs a forward with `return_auxiliary=True`.

- Expected impact: large validation-time reduction, often double-digit % of total epoch wall time
- Quality risk: none (same predictions, less duplicate compute)

### 2) Remove ghost auxiliary targets from data path

`add_auxiliary_labels()` populates only `item_targets`; `speed_targets`, `role_targets`, `tera_targets`, and `move_family_targets` are effectively placeholder `-1` arrays in current flow.

They are still:
- materialized
- copied/stacked
- transferred to GPU
- traversed in loss code before being skipped

- Expected impact: small but real data+transfer savings
- Quality risk: none in current setting (these targets are inactive)

### 3) Eliminate per-sample numpy copy/convert churn

`WindowedTurnDataset.__getitem__()` repeatedly does `.copy()` + `torch.from_numpy()` for each field on every access.

Pre-converting once at dataset init (or caching tensors per battle) avoids repeated host-side overhead.

- Expected impact: small-to-moderate input pipeline speedup
- Quality risk: none

---

## Tier 2 — Runtime controls worth keeping (already supported)

### 4) `torch.compile` for long training runs

Already wired via `--torch-compile`. Use for multi-epoch runs where compile amortizes.

- Expected impact: hardware/version dependent (can help, sometimes neutral)
- Quality risk: none in objective, but must benchmark

### 5) Keep CUDA transfer path optimized

Current script already supports and defaults appropriately on CUDA:
- workers/prefetch/persistent workers
- pin memory
- non-blocking transfer
- AMP auto (bf16/fp16)

Recommendation: keep these enabled; tune `num_workers/prefetch` empirically per machine.

- Expected impact: moderate throughput gains when dataloader is bottleneck
- Quality risk: none

---

## Tier 3 — Optional benchmark items (not guaranteed)

### 6) Nested tensor fast-path (`enable_nested_tensor=True`)

Possible speedup but **not guaranteed**. Treat as benchmark-only due to PyTorch-version and mask-path sensitivity.

### 7) Validation cadence reduction (operational option)

Evaluating every epoch is expensive. If operationally acceptable, validating every N epochs can reduce wall time, but this changes early-stopping responsiveness and checkpoint granularity. Use cautiously.

---

## Revised Priority Order for P8-Lean @ 10K

1. Fix duplicate validation forward pass.
2. Remove ghost auxiliary target tensors from dataset/collate/device transfer/loss wiring.
3. Pre-convert/cached tensors in dataset to avoid repeated numpy copy+wrap.
4. Benchmark `--torch-compile` on your exact hardware.
5. Re-tune dataloader workers/prefetch only after (1)-(3) are done.
6. Treat nested tensor as optional experiment, not default recommendation.

---

## Proposals Explicitly Excluded from This Speed Plan

These may still be useful research directions, but are excluded here because they are not direct, high-confidence training-speed wins without quality risk:

- Window-size changes for P8-Lean baseline (already at window 5)
- Batch-size “increase to 64” (already baseline)
- Feature-schema changes requiring data reprocessing (opponent base stats fill, `is_own` removal, new engineered features)
- Convergence-only heuristics framed as throughput improvements (warmup/patience)

---

## Success Criteria for This Proposal

A candidate optimization stays in scope only if it is:

1. **Current-pipeline valid** (matches `train_p8_lean.py` → `train_phase4.py` reality)
2. **Wall-clock beneficial** for 10K-battle P8-Lean runs
3. **Accuracy/generalization preserving** by construction (or empirically neutral)
4. **Not relying on outdated baseline assumptions**
