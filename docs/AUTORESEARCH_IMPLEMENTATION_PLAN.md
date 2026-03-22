# AutoResearch Implementation Plan for Pokemon Battle Model

## Critical Analysis & Alignment with Current IL Work

This document is the implementation plan for converting the Pokemon Battle Model repository
into an AutoResearch sandbox. It builds on the vision in `metamon_autoresearch_il_accuracy_a40_plan.md`
and critically evaluates that plan against the actual state of the codebase and training results.

---

## 1. Current State Assessment

### What We Have (March 2026)

**Best checkpoint:** P8-Lean 50K battles
| Metric | Value |
|--------|-------|
| Top-1 accuracy | **63.21%** |
| Top-3 accuracy | **89.27%** |
| Test policy loss | 1.0148 |
| Move accuracy | 72–75% |
| Switch accuracy | 37–48% |
| Parameters | 1.72M |
| Architecture | 3L / 224d / 4H / FFN 3x |
| Window size | 2 (minimal history) |
| Training time | 212 min on GTX 1650 |
| Throughput | ~1,520 examples/sec |
| Dataset | 50K battles, 694K train examples |

**Known weaknesses:**
- Switch prediction is dramatically worse than move prediction (37–48% vs 72–75%)
- Auxiliary speed and role heads show **0% accuracy** — they are not learning
- Window size of 2 is extremely short — the model has almost no battle history
- Trained on a GTX 1650 (4GB VRAM) with batch size 64 — severely constrained
- Calibration shows systematic overconfidence in the 0.4–0.8 range

### What the Plan Assumes vs Reality

| Plan Assumption | Reality | Gap |
|----------------|---------|-----|
| "50k-battle IL checkpoint" as anchor | **Exists** — P8-Lean 50K seed 42 | Aligned |
| ">1300 Elo" filter | Current data uses **1500+** with stratified bins 1000–1500 | Plan's threshold is **lower** than current; need to decide |
| Functioning codebase for loading/parsing | **Fully functional** pipeline | Aligned |
| Legal action reconstruction | **Complete** with legal mask | Aligned |
| Single A40 GPU | Currently GTX 1650; A40 = **12x more VRAM** (48GB vs 4GB) | Massive upgrade — unlocks larger batch sizes, models |
| Codex + Claude Code setup | Need to install on RunPod | New setup required |

---

## 2. Critical Analysis of the Plan

### What the Plan Gets Right

1. **Focus on data over architecture.** The plan correctly prioritizes data filtering, cleaning,
   and versioning before architecture changes. Our best gains likely come from data quality.

2. **Tiered experiment budgets.** The Tier 1/2/3 structure (triage → confirm → promote) is
   well-suited to A40 budget constraints and prevents wasted compute.

3. **History length as a key variable.** Our current window=2 is almost certainly too short.
   The plan flags this and it should be among the first experiments.

4. **Throughput matters.** On A40, we should see 5–10x throughput gains from larger batch
   sizes alone. The plan's emphasis on time-to-accuracy is correct.

5. **Narrow edit surface.** Restricting what agents can modify prevents chaos.

### What the Plan Gets Wrong or Misses

1. **Auxiliary heads are broken, not just suboptimal.** The plan treats aux heads as something
   to tune. In reality, speed and role auxiliary heads show **0% accuracy** across all epochs.
   This is a bug or architectural issue that needs debugging before any AutoResearch cycle
   can meaningfully use these metrics. **This is a pre-requisite, not an experiment.**

2. **Switch prediction gap is the biggest accuracy lever.** The plan mentions move-vs-switch
   diagnostics but doesn't emphasize that switches account for ~35% of all actions and are
   predicted at 37–48% accuracy vs 72–75% for moves. Closing even half this gap would add
   ~5 points to overall Top-1 accuracy. This deserves its own experiment track.

3. **The Elo threshold mismatch needs resolution.** The plan says ">1300" but our current
   pipeline uses 1500+ for all data and stratified 1000–1500. We need to decide:
   is the AutoResearch target 1300+ or 1500+? This affects the dataset version definition.

4. **Value head is disabled and untested.** The plan doesn't mention the value head, but it
   exists in the architecture. It should stay disabled for pure IL accuracy focus — but this
   should be explicitly noted.

5. **No mention of the existing evaluation spec.** `docs/EVALUATION_SPEC.md` defines detailed
   gate criteria, cross-archetype evaluation, and calibration targets. The AutoResearch
   evaluation protocol should build on this, not replace it.

6. **Dataset size is a variable.** We have 100K battles processed but only trained on 50K.
   Training on the full 100K is an obvious early experiment that the plan doesn't highlight.

7. **The plan's phases conflict with the project's existing phases.** The plan defines Phases
   0–5 for AutoResearch, but the project already has Phases 0–8. The AutoResearch work is
   a focused sub-effort within the existing Phase 4. Renaming to avoid confusion is important.

---

## 3. Revised AutoResearch Phases

To avoid confusion with the project's existing Phase 0–8 structure, AutoResearch phases
are prefixed with "AR-".

### AR-0: Benchmark Lockdown (Pre-requisite, ~2 hours)

**Goal:** Freeze the anchor checkpoint and create a reproducible evaluation baseline.

**Deliverables:**
1. Copy P8-Lean 50K checkpoint as `anchor_il_50k_gen3ou_seed42`
2. Create `autoresearch/eval_harness.py` — single-command evaluation that reports:
   - Top-1, Top-3 accuracy (overall, moves-only, switches-only)
   - Policy NLL / cross-entropy
   - Per-action accuracy breakdown
   - ECE calibration
   - Throughput (examples/sec)
   - Wall-clock time
3. Create `autoresearch/experiment_registry.json` — schema for tracking all runs
4. Create `autoresearch/leaderboard.py` — script to rank experiments
5. Freeze dataset version `gen3ou_100k_v1` with exact split manifests
6. Document the anchor's exact config in the registry as experiment #0

**Implementation details:**
```
autoresearch/
├── eval_harness.py          # One-command evaluation
├── experiment_registry.json # Machine-readable experiment log
├── leaderboard.py           # Rank experiments, generate markdown
├── run_experiment.py        # Bounded experiment launcher
├── configs/                 # Experiment configs (YAML)
│   └── anchor.yaml          # Frozen anchor config
├── notes/                   # Per-experiment markdown notes
│   └── 000_anchor.md
└── prompts/                 # Agent prompt templates
    ├── codex_template.md
    └── claude_code_template.md
```

### AR-1: Fix Known Issues (~4 hours)

**Goal:** Fix broken components before running experiments.

**Tasks:**
1. **Debug auxiliary speed/role heads** — investigate why accuracy = 0% for all epochs.
   Likely causes: label construction bug, loss masking issue, or gradient not flowing.
   Check `src/data/auxiliary_labels.py` and the aux loss computation in `compute_total_loss()`.
2. **Validate legal mask correctness** — confirm that the legal mask is 100% correct on
   the test set (no illegal actions ever predicted with nonzero probability).
3. **Verify data deduplication** — confirm no battle appears in multiple splits.
4. **Test full 100K dataset loading** — ensure the pipeline handles all 100K battles.

### AR-2: A40 Throughput Optimization (~4 hours)

**Goal:** Maximize training speed on A40 before changing the model.

**Experiments (all Tier 1 — short triage runs):**

| ID | Experiment | Expected Impact |
|----|-----------|----------------|
| AR2-01 | Batch size sweep: 64 → 128 → 256 → 512 → 1024 | 2–4x throughput |
| AR2-02 | Gradient accumulation: effective batch 256, 512, 1024 | Stability check |
| AR2-03 | DataLoader workers: 2, 4, 8, 12 | CPU-bound check |
| AR2-04 | bf16 vs fp16 vs fp32 on A40 | A40 has good bf16 |
| AR2-05 | torch.compile() on A40 | Potential 20–40% speedup |
| AR2-06 | Prefetch factor: 2, 4, 8 | Memory vs speed |

**Promotion criteria:** Same or better accuracy at higher throughput.

### AR-3: Data & Representation Experiments (~8 hours)

**Goal:** Improve accuracy through data quality and feature engineering.

**Tier 1 experiments:**

| ID | Experiment | Hypothesis |
|----|-----------|-----------|
| AR3-01 | Train on full 100K battles (vs 50K) | More data = better generalization |
| AR3-02 | Window size: 2 → 5 → 10 → 20 | History context improves switch prediction |
| AR3-03 | Elo-weighted loss (upweight 1500+) | Higher-skill turns are more informative |
| AR3-04 | Class-weighted loss (upweight switches) | Address switch accuracy gap |
| AR3-05 | Label smoothing (0.05, 0.1) | Reduce overconfidence |
| AR3-06 | Remove corrupted/ambiguous turns | Cleaner labels |
| AR3-07 | Separate move-vs-switch diagnostic head | Understand prediction errors |

**Priority ranking:**
1. AR3-02 (window size) — likely biggest single gain, moves from near-zero context to real history
2. AR3-01 (100K data) — low effort, guaranteed more signal
3. AR3-04 (class weighting) — direct attack on switch accuracy gap
4. AR3-03 (Elo weighting) — cleaner signal from better players

### AR-4: Architecture Tuning (~8 hours)

**Goal:** Find the optimal model size for A40 given data quality improvements.

**Tier 1 experiments:**

| ID | Experiment | Hypothesis |
|----|-----------|-----------|
| AR4-01 | Scale up: 4L/256d/4H (P8, 3.6M) | More capacity for longer context |
| AR4-02 | Scale up: 6L/384d/6H (P4, 11.5M) | A40 can handle this easily |
| AR4-03 | Wider but shallower: 2L/384d/6H | Test width vs depth |
| AR4-04 | Deeper but narrower: 6L/224d/4H | Test depth vs width |
| AR4-05 | FFN multiplier: 3x vs 4x vs 2x | MLP capacity |
| AR4-06 | Dropout sweep: 0.05, 0.1, 0.15, 0.2 | Regularization tuning |
| AR4-07 | Embedding dims sweep | Species/move embedding capacity |
| AR4-08 | Hierarchical action head (move-vs-switch then which) | Structural inductive bias |

**Dependencies:** Run AR-3 winners first, then tune architecture on best data config.

### AR-5: Consolidation & Final Training (~4 hours)

**Goal:** Produce the best checkpoint with confirmed improvements.

**Tasks:**
1. Combine best data, representation, architecture, and loss from AR-2 through AR-4
2. Run 3-seed confirmation (seeds 42, 43, 44)
3. Compute full test metrics including per-action accuracy and calibration
4. Write final experiment report with:
   - Full ranked leaderboard
   - Ablation summary (what contributed most)
   - Remaining headroom analysis
   - Recommendations for next steps

---

## 4. AutoResearch Harness Design

### Experiment Registry Schema

```json
{
  "experiment_id": "AR3-02a",
  "parent_id": "AR0-anchor",
  "name": "window_size_10",
  "description": "Increase window size from 2 to 10 turns",
  "tier": 1,
  "dataset_version": "gen3ou_100k_v1",
  "config_changes": {
    "max_window": 10
  },
  "metrics": {
    "val_top1_accuracy": null,
    "val_top3_accuracy": null,
    "val_loss": null,
    "test_top1_accuracy": null,
    "throughput_examples_sec": null,
    "wall_time_min": null,
    "gpu_hours": null
  },
  "decision": null,
  "notes": "",
  "timestamp": null,
  "seed": 42,
  "checkpoint_path": null
}
```

### Run Wrapper: `autoresearch/run_experiment.py`

This script wraps `scripts/train_phase4.py` with:
- Automatic experiment ID generation and registry update
- Budget enforcement (max epochs, max wall-time)
- Result extraction and leaderboard update
- Markdown note generation
- Parent comparison (delta from parent experiment)

```python
# Usage:
# python autoresearch/run_experiment.py \
#   --name "window_size_10" \
#   --parent AR0-anchor \
#   --tier 1 \
#   --config-override max_window=10 \
#   --budget-epochs 10 \
#   --budget-minutes 30
```

### Leaderboard: `autoresearch/leaderboard.py`

Reads the experiment registry and produces:
- Ranked table by validation Top-1 accuracy
- Time-to-accuracy chart
- Throughput comparison
- Promotion recommendations

### Evaluation Harness: `autoresearch/eval_harness.py`

Single-command evaluation of any checkpoint:
```bash
python autoresearch/eval_harness.py \
  --checkpoint checkpoints/phase4_p8_lean_50k/seed_42/best_model.pt \
  --data-dir data/processed \
  --output results/eval_AR0_anchor.json
```

Reports:
- Overall Top-1, Top-3 accuracy
- Move-only and switch-only accuracy
- Per-action accuracy (9 actions)
- Policy NLL
- ECE calibration (5 bins)
- Throughput (examples/sec)
- Auxiliary head accuracies (item, speed, role)

---

## 5. Agent Operating Model

### Claude Code Responsibilities

1. **Experiment planning:** Analyze leaderboard, propose next experiments, rank hypotheses
2. **Log interpretation:** Read training reports, identify trends, diagnose issues
3. **Code changes:** Implement model/data/loss modifications within approved edit surface
4. **Research notes:** Write structured per-experiment analysis
5. **Bug investigation:** Debug issues like the broken aux heads

### Codex Responsibilities

1. **Config generation:** Create experiment config files
2. **Script fixes:** Patch throughput issues, profiling utilities
3. **Boilerplate:** Experiment launch scripts, result aggregation
4. **Log parsing:** Extract metrics from training reports

### Approved Edit Surface

Agents may freely modify:
- `autoresearch/` — all files
- `configs/` — YAML configs
- `scripts/train_phase4.py` — training loop (with review)
- `src/models/battle_transformer.py` — model architecture (with review)
- `src/data/dataset.py` — data loading
- `src/data/auxiliary_labels.py` — aux label construction

Agents must NOT modify without explicit approval:
- `src/data/observation.py` — observation construction (affects data integrity)
- `src/data/tensorizer.py` — tensorization (affects data integrity)
- `src/data/replay_parser.py` — parsing (affects data integrity)
- Split manifests — evaluation integrity
- `docs/EVALUATION_SPEC.md` — benchmark definition

---

## 6. Experiment Loop Protocol

Each agent cycle follows this protocol:

```
1. READ    → Load leaderboard, identify current champion
2. HYPOTHESIZE → Select one hypothesis to test
3. PLAN    → Specify exact config change, expected effect, tier
4. IMPLEMENT → Modify only approved files
5. RUN     → Launch via run_experiment.py with budget
6. EVALUATE → Compare to parent on eval harness
7. RECORD  → Write experiment note with:
             - Change made
             - Metric delta (Top-1, Top-3, loss, throughput)
             - Time cost
             - Likely reason for result
             - Decision: KILL / RETRY / PROMOTE
8. UPDATE  → Update leaderboard
```

### Tier Budgets on A40

| Tier | Max Epochs | Max Wall Time | Purpose |
|------|-----------|---------------|---------|
| 1 | 5–10 | 30 min | Quick triage |
| 2 | 20–30 | 2 hours | Confirmation |
| 3 | 30–50 | 4 hours | Full promotion |

### Promotion Rules

A candidate replaces the champion only if:
- **Accuracy gate:** Top-1 accuracy improves by ≥0.5 percentage points, OR
- **Speed gate:** Matches accuracy within 0.3 points while reducing time-to-baseline by ≥20%
- **Stability gate:** For Tier 3, must be confirmed on ≥2 seeds with std < 1.0 point

---

## 7. Expected Accuracy Roadmap

Based on analysis of current weaknesses and available headroom:

| Phase | Expected Top-1 Gain | Cumulative Top-1 | Confidence |
|-------|---------------------|-------------------|------------|
| Anchor | — | 63.2% | Known |
| AR-1 (fix aux) | +0–0.5% | 63.2–63.7% | High |
| AR-3-02 (window 10+) | +2–5% | 65–68% | High |
| AR-3-01 (100K data) | +1–3% | 66–71% | Medium-High |
| AR-3-04 (switch weighting) | +1–3% | 67–74% | Medium |
| AR-4 (architecture) | +1–2% | 68–76% | Medium |
| AR-5 (combined) | +0–1% | 68–77% | Medium-Low |

**Realistic target:** 70–75% Top-1 accuracy within the A40 compute budget.
**Stretch target:** 77%+ if window size and data scaling compound well.

---

## 8. Key Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Aux heads remain broken after debug | Low | Medium | Disable aux loss, focus on policy only |
| Window size increase causes OOM | Medium | Low | Gradient accumulation, reduce batch size |
| 100K data doesn't help (ceiling) | Low | Medium | Focus on data quality over quantity |
| Agents make conflicting changes | Medium | High | Single-agent-per-file ownership, git branches |
| A40 throughput bottleneck is CPU | Medium | Medium | Profile first, optimize dataloader |
| Overfitting on larger models | Medium | Medium | Dropout sweep, early stopping, regularization |

---

## 9. Files to Create

| File | Purpose |
|------|---------|
| `autoresearch/eval_harness.py` | Unified evaluation command |
| `autoresearch/run_experiment.py` | Bounded experiment launcher |
| `autoresearch/leaderboard.py` | Experiment ranking |
| `autoresearch/experiment_registry.json` | Machine-readable log |
| `autoresearch/configs/anchor.yaml` | Frozen anchor config |
| `autoresearch/notes/000_anchor.md` | Anchor experiment note |
| `autoresearch/prompts/codex_template.md` | Codex agent prompt |
| `autoresearch/prompts/claude_code_template.md` | Claude Code agent prompt |

---

## 10. Immediate Next Steps

1. Create the `autoresearch/` directory structure
2. Implement `eval_harness.py` by wrapping existing evaluation logic from `train_phase4.py`
3. Register the P8-Lean 50K anchor as experiment #0
4. Debug auxiliary speed/role head 0% accuracy issue
5. Profile training pipeline on A40 (batch size sweep)
6. Run first experiment: window size 2 → 10 on 50K battles
7. Run second experiment: 100K battles with window=2 (isolate data scaling effect)
