# P8 @ 10K Battles: Path from ~900s/Epoch → ~300s/Epoch (≤10% quality loss target)

## Scope

This memo combines:

1. The existing Phase 4 compute/generalization experiment framework.
2. The current P8 architecture/training stack.
3. New dataset-level evidence from processed replay tensors.

Goal: propose a **cumulative speed plan** to reach roughly **3x faster training per epoch** while constraining degradation in accuracy/generalizability.

---

## What the repository currently tells us

- P8 is defined as **4 layers / 256 hidden / 4 heads / window 20**, with auxiliary head and value head enabled. This is the lowest-cost model in the documented ladder that still preserves full 20-turn context.【F:docs/PHASE4_25K_COMPUTE_GENERALIZATION_EXPERIMENT.md†L75-L76】
- The trainer already supports the key ablation knobs you asked about (`--max-window`, architecture scaling, aux/value toggles, data loader and AMP controls).【F:docs/PHASE4_25K_COMPUTE_GENERALIZATION_EXPERIMENT.md†L5-L6】【F:scripts/train_phase4.py†L754-L818】
- The sequence pipeline is fixed to **14 tokens/turn** (6 own + 6 opponent + field + context), so time scales strongly with `window * 14` token length in attention blocks.【F:src/models/battle_transformer.py†L13-L21】【F:src/models/battle_transformer.py†L151-L152】

---

## Fresh evidence from the 10K processed battles

I added a small analysis script and generated a JSON artifact over the current processed cohort.

- Script: `scripts/analyze_p8_dataset_efficiency.py`.
- Output artifact: `docs/artifacts/p8_efficiency_stats.json`.

### Key findings

1. Effective dataset is ~10K battles / ~190K turn examples, with most sequences saturating the 20-turn cap (median/p95 = 20).【F:docs/artifacts/p8_efficiency_stats.json†L2-L11】
2. Legal action set is large (mean ~10.34 legal actions/turn), and switches are common (~35.9%), so preserving strong tactical decision quality (especially switching) remains essential.【F:docs/artifacts/p8_efficiency_stats.json†L19-L37】
3. Tera actions are rare (~2.45% of actions), so tera-specific complexity is a good target for selective simplification/curriculum rather than full removal.【F:docs/artifacts/p8_efficiency_stats.json†L30-L37】
4. Several input dimensions appear effectively dead in the processed tensors:
   - `terastallized` binary flag rate = 0.0.
   - Field binary side-condition block nonzero rate = 0.0.
   These are high-confidence feature-pruning opportunities with little/no expected quality loss if confirmed on future cohorts too.【F:docs/artifacts/p8_efficiency_stats.json†L38-L44】
5. Attention cost proxy drops sharply with window reduction: `W=20 → 12` is 0.36x attention-token-squared cost; `W=20 → 10` is 0.25x. This is your biggest lever, but also highest quality risk because most battles hit length 20.【F:docs/artifacts/p8_efficiency_stats.json†L46-L68】【F:docs/artifacts/p8_efficiency_stats.json†L12-L18】

---

## Recommended speed plan (ordered, cumulative)

## Tier 1 (Low risk, should be done first)

### 1) Prune dead/near-dead input features before embedding projection

**What to change**
- Remove permanently-zero field binary channels and the always-zero `terastallized` flag from tensorization/model input projection.
- Keep schema versioned so old checkpoints remain loadable.

**Why it helps**
- Cuts unnecessary memory bandwidth and matmul input width in embedding projection.
- Slightly reduces host→device transfer payload and collation cost.

**Quality risk**
- Very low if these channels remain zero in holdout cohorts.

**Expected speedup**
- ~1.05x to 1.12x epoch speedup (small alone, but “free”).

### 2) Remove or downweight value head for most runs

**What to change**
- Run with `--no-value-head` for throughput-first BC runs, or enable value head only in short fine-tuning stage.

**Why it helps**
- Cuts extra head forward/backward and associated loss computation.

**Quality risk**
- Usually low for top-1 action imitation (primary metric), but verify generalization gap and calibration.

**Expected speedup**
- ~1.03x to 1.08x.

### 3) Throughput tuning defaults for CUDA path

**What to change**
- Standardize P8 fast profile: AMP bf16/fp16 + non-blocking transfer + tuned workers/prefetch + persistent workers.
- Add optional `torch.compile` toggle for repeat runs.

**Why it helps**
- Improves hardware utilization without changing model semantics.

**Quality risk**
- None expected (numerical parity check required for AMP mode).

**Expected speedup**
- ~1.15x to 1.35x depending on hardware.

**Tier-1 cumulative target**: ~1.25x–1.50x with near-zero accuracy loss.

---

## Tier 2 (Medium risk, major speed gains)

### 4) Window curriculum instead of static full-window training

**What to change**
- Train early epochs with short context (`W=8` or `W=10`), then increase to `W=12` (and optionally brief `W=16`) for late-stage refinement.
- Keep final checkpoint from long-enough context stage.

**Why it helps**
- Most gradient steps happen in cheap regime where attention cost is drastically lower.
- Final-stage longer window recovers long-horizon behavior.

**Quality risk**
- Lower than permanently training at short window.

**Expected speedup**
- ~1.4x to 1.9x over always-`W=20`, depending on schedule.

### 5) Slight architecture trim from P8 to “P8-fast”

**Candidate config**
- 3 layers / 224 hidden / 4 heads, with same tokenization and aux head.

**Why it helps**
- Depth and width reductions compound with shorter-window curriculum.

**Quality risk**
- Moderate; should still stay within your <10% drop constraint if combined with strong regularization + early stop and brief longer-window finishing.

**Expected speedup**
- ~1.25x to 1.45x vs current P8 backbone.

**Tier-1 + Tier-2 cumulative target**: approximately **2.0x to 2.8x**.

---

## Tier 3 (Higher risk, high upside)

### 6) Token-count reduction per turn (14 → 10 or 8)

**What to change**
- Replace full 6-slot opponent bench tokens with compressed summary token(s) (revealed count, known threats, hazard sensitivity proxy).
- Optionally compress own bench similarly while preserving active mon detail.

**Why it helps**
- Attention cost depends on total tokens; reducing tokens/turn can rival window reduction while keeping temporal span.

**Quality risk**
- Medium-high; switching and sack decisions may suffer if bench detail is over-compressed.

**Expected speedup**
- 14→10 tokens yields about `(10/14)^2 ≈ 0.51x` attention cost at same window.

**Use strategy**
- Treat as separate ablation branch only after Tier 1/2 are quantified.

---

## Concrete recipe likely to hit ~300s/epoch

A pragmatic candidate stack:

1. Tier 1 fully applied (feature pruning + value-head off + tuned runtime).
2. Window curriculum: `W=8` for 50% epochs, `W=12` for 40%, `W=16` for final 10%.
3. P8-fast backbone (3L/224d/4H).

This combination plausibly reaches ~3x epoch speedup in aggregate while preserving policy quality better than static tiny-window training.

---

## Accuracy/generalization guardrails (must-pass)

Use the same philosophy as the Phase 4 decision rule:

- Top-1 drop ≤ 1.0 abs point preferred for “safe” rollout; hard cap aligned with your request: <10% relative degradation.
- Top-3 drop, generalization-gap growth, and seed variance tracked for every ablation candidate.
- Pay special attention to action slices where compression/window cuts usually hurt first:
  - forced switches,
  - late-game sequences,
  - low-frequency tera lines.

(Phase 4 already defines suitable metrics and 3-seed protocol for these checks.)【F:docs/PHASE4_25K_COMPUTE_GENERALIZATION_EXPERIMENT.md†L16-L25】【F:docs/PHASE4_25K_COMPUTE_GENERALIZATION_EXPERIMENT.md†L27-L68】

---

## Suggested ablation matrix (minimal but decisive)

1. **Baseline**: current P8 (W20, value on).
2. + Tier 1 only.
3. + Tier 1 + window curriculum.
4. + Tier 1 + window curriculum + P8-fast (3L/224d).
5. + optional token compression branch.

Run each with 3 seeds and compare compute-vs-quality frontier exactly as the experiment plan prescribes.【F:docs/PHASE4_25K_COMPUTE_GENERALIZATION_EXPERIMENT.md†L104-L136】

---

## Why these changes are aligned with Pokemon battle decision-making

Most important signals to preserve for policy quality:

1. **Action legality and switch constraints** (because legal set is dense/variable).【F:docs/artifacts/p8_efficiency_stats.json†L19-L25】
2. **Active matchup state** (HP, boosts, status, revealed moves/items/abilities).
3. **Bench resource state** (remaining healthy pivots/revenge options) — should be compressed carefully, not removed blindly.
4. **Recent tactical tempo** (previous moves, forced-switch status).

Least valuable in current processed tensors (for speed/feature budget):

- Persistently-zero channels identified above (safe prune candidates).【F:docs/artifacts/p8_efficiency_stats.json†L38-L44】
- Heavy modeling emphasis on tera-specific branches despite very low tera action frequency; better handled via targeted weighting/curriculum than large dedicated capacity.【F:docs/artifacts/p8_efficiency_stats.json†L30-L37】

---

## New files added in this memo update

- `scripts/analyze_p8_dataset_efficiency.py` (data-driven efficiency diagnostics).
- `docs/artifacts/p8_efficiency_stats.json` (generated stats from processed replays).
- `docs/P8_10K_TRAINING_TIME_REDUCTION_PLAN.md` (this plan).
