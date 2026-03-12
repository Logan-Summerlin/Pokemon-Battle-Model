# Phase 4 (25K Battles) Compute-vs-Generalization Experiment Plan

## Why this experiment is the right fit for this repository

This repo’s Phase 4 training stack already supports battle subsampling (`--num-battles`), architecture scaling (`--num-layers`, `--hidden-dim`, `--num-heads`), history-length control (`--max-window`), and auxiliary/value-head toggles (`--aux-weight`, `--no-value-head`). That means we can run a controlled ablation matrix without changing core training code. The current trainer also reports validation and test metrics needed for quality checks and records runtime/resource telemetry for compute comparisons. The data pipeline uses battle-level train/val/test splitting and the windowed per-turn dataset, which should remain fixed to avoid leakage and to preserve training signal quality.

## Objective

Quantify how much training-time compute can be reduced in Phase 4 before materially degrading:

1. **Accuracy** (action prediction quality), and
2. **Generalizability** (performance stability across held-out distributions),

using a fixed sample of **25,000 battles**.

## Fixed experimental controls (hold constant across all runs)

- Dataset size: 25,000 processed battles (`--num-battles 25000`).
- Split protocol: battle-level 80/10/10 split with fixed seed per run.
- Training recipe baseline: AdamW + warmup cosine schedule + early stopping.
- Input construction: windowed per-turn examples (no reverting to last-step-only).
- Hidden-information doctrine: no omniscient leakage in opponent features.
- Evaluate each permutation with **3 random seeds** (e.g., 42/43/44).

These controls ensure each architecture permutation is compared on an apples-to-apples basis.

## Primary and secondary metrics

### Quality metrics (accuracy)

- Validation loss (total and policy component).
- Test action top-1 accuracy.
- Test top-3 accuracy.
- Per-action accuracy breakdown (moves, tera-moves, switches).

### Generalizability metrics

- **Generalization gap** = train top-1 − test top-1.
- **Temporal holdout delta** (optional but recommended): evaluate on later-date replays and measure drop from in-distribution test.
- **Seed variance**: std-dev of test top-1 across 3 seeds.
- **Calibration** (ECE from stored confidence/correctness bins if using existing report outputs).

### Compute metrics (training-time, not inference)

- Wall-clock time to best checkpoint.
- Total optimizer steps to early stop.
- Mean examples/sec.
- Peak RAM / GPU memory.
- Estimated training FLOP proxy:
  - use `params × steps × (window_tokens^2 attention term proxy)` where `window_tokens = max_window × 14`.

## Decision rule for “no meaningful loss”

A lower-compute permutation is acceptable if all conditions hold versus the strongest reference model:

- Test top-1 drop ≤ **1.0 absolute point**.
- Test top-3 drop ≤ **0.8 absolute point**.
- Generalization gap increase ≤ **1.0 absolute point**.
- Seed std-dev increase ≤ **0.5 absolute point**.
- Compute reduction ≥ **25%** in wall-clock to best checkpoint.

## 10 architecture permutations (ranked most compute-intensive → least)

All runs use 25K battles. Rank is based on dominant training-time cost drivers in this repo (layers, hidden dim, and window length).

| Rank | ID | Architecture / training settings | Relative training compute | Expected quality profile |
|---|---|---|---|---|
| 1 | P1 | **8L / 512d / 8H**, window 20, aux on (`0.2`), value on | 1.00x (reference max) | Best ceiling, slowest |
| 2 | P2 | **8L / 512d / 8H**, window 20, aux on, value **off** | 0.96x | Near P1, slightly less regularization |
| 3 | P3 | **6L / 512d / 8H**, window 20, aux on, value on | 0.82x | Often near-P1 with lower depth |
| 4 | P4 | **6L / 384d / 6H**, window 20, aux on, value on | 0.55x | Current practical full-mode anchor |
| 5 | P5 | **6L / 384d / 6H**, window 15, aux on, value on | 0.43x | Context reduction may mildly hurt long-horizon decisions |
| 6 | P6 | **6L / 384d / 6H**, window 10, aux on, value on | 0.31x | Significant speedup; watch strategic degradation |
| 7 | P7 | **4L / 384d / 6H**, window 15, aux on, value on | 0.29x | Shallower model with moderate context |
| 8 | P8 | **4L / 256d / 4H**, window 20, aux on, value on | 0.24x | Small-model baseline from scaling ladder |
| 9 | P9 | **4L / 256d / 4H**, window 10, aux on, value off | 0.14x | Strong compute savings, likely noticeable drop |
|10 | P10| **2L / 192d / 4H**, window 10, aux on, value off | 0.08x | Minimum-cost candidate; likely largest quality loss |

Notes:
- Relative compute here is a planning prior; use measured wall-clock and examples/sec for final ranking.
- Keep auxiliary head on in all but explicitly ablated runs because it is low overhead and representation-helpful in this codebase.

## Execution protocol (practical)

### Step 1: Create a fixed 25K battle cohort

- Prefer deterministic file ordering + fixed `--num-battles 25000`.
- Keep the same cohort for all permutations.

### Step 2: Run each permutation for 3 seeds

Template command:

```bash
python scripts/train_phase4.py \
  --mode full \
  --num-battles 25000 \
  --num-layers <L> \
  --hidden-dim <D> \
  --num-heads <H> \
  --max-window <W> \
  --aux-weight 0.2 \
  [--no-value-head if needed] \
  --batch-size 32 \
  --epochs 30 \
  --patience 7 \
  --seed <SEED> \
  --checkpoint-dir checkpoints/phase4_25k/<perm>/seed_<SEED> \
  --report-path checkpoints/phase4_25k/<perm>/seed_<SEED>/training_report.json
```

If GPU memory allows, increase physical batch size and use grad accumulation to keep effective batch comparable across runs.

### Step 3: Aggregate and score

For each permutation, compute:

- mean ± std over seeds for top-1/top-3/test loss,
- generalization gap statistics,
- time-to-best-checkpoint and total run time,
- compute-efficiency ratio: `(top1_mean / hours_to_best)` and `(top1_mean / FLOP_proxy)`.

### Step 4: Select frontier model

Construct a Pareto frontier on:

- x-axis: training compute (hours or FLOP proxy),
- y-axis: test top-1 (and optionally separate chart for generalization gap).

Choose the lowest-compute model that satisfies the “no meaningful loss” rule.

## Recommended expected outcome (hypothesis)

Most likely efficient sweet spot on 25K battles is either:

- **P4 (6L/384d, W=20)** for maximum robustness, or
- **P5/P6** if reduced window keeps quality within thresholds.

P8 is a good fallback if hard training-budget constraints dominate.

## Risks and mitigation

- **Risk:** Window reduction underfits long-range planning.
  - **Mitigation:** Compare per-action accuracy for switch decisions and late-turn contexts.
- **Risk:** Small models look good on IID split but fail temporally.
  - **Mitigation:** Include temporal holdout slice.
- **Risk:** Seed sensitivity at low compute.
  - **Mitigation:** Enforce 3-seed stability gate.

## Deliverables

1. One CSV/JSON summary table with all 10 permutations × 3 seeds.
2. Pareto plot (compute vs test accuracy).
3. Final recommendation memo identifying selected Phase 4 training configuration for scale-up.
