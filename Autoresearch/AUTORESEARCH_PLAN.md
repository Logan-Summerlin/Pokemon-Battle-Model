# AutoResearch Implementation Plan — Pokemon Battle Model Phase 4

## Goal

Perfect imitation learning for **>1300 Elo Gen 3 OU battles** from the Metamon dataset. This is a focused sub-effort within the existing Phase 4 of the Pokemon Battle Model project.

---

## 1. Current State (March 2026)

### Anchor Checkpoint: P8-Lean 50K

| Metric | Value |
|--------|-------|
| Top-1 accuracy | **63.21%** |
| Top-3 accuracy | **89.27%** |
| Test policy loss | 1.0148 |
| Move accuracy | 72–75% |
| Switch accuracy | 37–48% |
| Parameters | ~1.72M |
| Architecture | 3L / 224d / 4H / FFN 3x |
| Context window | 2 turns (minimal history) |
| Training time | 212 min on GTX 1650 |
| Throughput | ~1,520 examples/sec |
| Dataset | 50K battles, 694K train examples |

### Known Weaknesses

1. **Switch prediction is dramatically worse than move prediction** (37–48% vs 72–75%)
2. **Auxiliary speed and role heads show 0% accuracy** — broken, not just suboptimal
3. **Context window of 2 turns is extremely short** — model has almost no battle history
4. **Trained on GTX 1650 (4GB VRAM)** with batch size 64 — severely constrained
5. **Calibration shows systematic overconfidence** in the 0.4–0.8 range
6. **Only 50K of >100K available battles used** — data scaling unexplored

### Key Decisions

- **Elo threshold**: >1300 (lower than the previous 1500+ filter — includes more data for better generalization)
- **Dataset size**: Use as many battles as necessary from the full Metamon Gen 3 OU dataset (>100K battles)
- **Context window**: 2 turns is reasonable baseline; optimal is likely 2–8 turns (key experimental variable)
- **Scope**: This is Phase 4 perfection — not Phase 5+ (synthetic fine-tuning, RL)

---

## 2. AutoResearch Phases

Prefixed "AR-" to distinguish from the parent project's Phase 0–8 structure.

### AR-0: Benchmark Lockdown (~2 hours)

**Goal:** Freeze the anchor checkpoint and create a reproducible evaluation baseline.

**Deliverables:**
1. Copy P8-Lean 50K checkpoint as `anchor_il_50k_gen3ou_seed42`
2. Run `eval_harness.py` on the anchor to produce the baseline metrics JSON
3. Register anchor as experiment #0 in `experiment_registry.json`
4. Freeze dataset version with exact split manifests
5. Document the anchor's exact config in `configs/anchor.yaml`

### AR-1: Fix Known Issues (~4 hours)

**Goal:** Fix broken components before running experiments.

**Tasks:**
1. **Debug auxiliary speed/role heads** — investigate why accuracy = 0% for all epochs. Likely causes: label construction bug in `src/data/auxiliary_labels.py`, loss masking issue in `compute_total_loss()`, or gradient not flowing through the aux head.
2. **Validate legal mask correctness** — confirm 100% correct on test set.
3. **Verify data deduplication** — confirm no battle appears in multiple splits.
4. **Test full dataset loading** — ensure the pipeline handles all available battles.

### AR-2: A40 Throughput Optimization (~4 hours)

**Goal:** Maximize training speed on A40 before changing the model.

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

| ID | Experiment | Hypothesis | Priority |
|----|-----------|-----------|----------|
| AR3-01 | Train on full dataset (all >1300 Elo battles) | More data = better generalization | 2 |
| AR3-02 | Window size: 2 → 3 → 5 → 8 | History context improves switch prediction | **1 (highest)** |
| AR3-03 | Elo-weighted loss (upweight 1500+) | Higher-skill turns are more informative | 4 |
| AR3-04 | Class-weighted loss (upweight switches) | Address switch accuracy gap | 3 |
| AR3-05 | Label smoothing (0.05, 0.1) | Reduce overconfidence / improve calibration | 5 |
| AR3-06 | Remove corrupted/ambiguous turns | Cleaner labels | 6 |
| AR3-07 | Separate move-vs-switch diagnostic head | Understand prediction errors | 7 |

**Priority rationale:**
1. **AR3-02 (window size)** — single highest expected gain. Current window=2 means the model has almost no battle context. Optimal is likely 2–8 turns.
2. **AR3-01 (full dataset)** — low effort, guaranteed more signal from using all available >1300 Elo data.
3. **AR3-04 (switch weighting)** — direct attack on the 37–48% switch accuracy gap.
4. **AR3-03 (Elo weighting)** — cleaner signal from higher-skill players.

### AR-4: Architecture Tuning (~8 hours)

**Goal:** Find optimal model size for A40 given best data config from AR-3.

| ID | Experiment | Hypothesis |
|----|-----------|-----------|
| AR4-01 | Scale to P8: 4L/256d/4H (3.6M params) | More capacity for longer context |
| AR4-02 | Scale to P4: 6L/384d/6H (11.5M params) | A40 can handle this easily |
| AR4-03 | Wider but shallower: 2L/384d/6H | Test width vs depth |
| AR4-04 | Deeper but narrower: 6L/224d/4H | Test depth vs width |
| AR4-05 | FFN multiplier: 2x vs 3x vs 4x | MLP capacity |
| AR4-06 | Dropout sweep: 0.05, 0.1, 0.15, 0.2 | Regularization tuning |
| AR4-07 | Embedding dims sweep | Species/move embedding capacity |
| AR4-08 | Hierarchical action head (move-vs-switch then which) | Structural inductive bias for switch prediction |

**Dependencies:** Run AR-3 winners first, then tune architecture on best data config.

### AR-5: Consolidation & Final Training (~4 hours)

**Goal:** Produce the best checkpoint with confirmed improvements.

1. Combine best data, representation, architecture, and loss from AR-2 through AR-4
2. Run 3-seed confirmation (seeds 42, 43, 44)
3. Compute full test metrics including per-action accuracy and calibration
4. Write final experiment report with:
   - Full ranked leaderboard
   - Ablation summary (what contributed most)
   - Remaining headroom analysis
   - Recommendations for Phase 5+ (synthetic fine-tuning, RL)

---

## 3. Expected Accuracy Roadmap

| Phase | Expected Top-1 Gain | Cumulative Top-1 | Confidence |
|-------|---------------------|-------------------|------------|
| Anchor | — | 63.2% | Known |
| AR-1 (fix aux) | +0–0.5% | 63.2–63.7% | High |
| AR-3-02 (window 3–8) | +2–5% | 65–68% | High |
| AR-3-01 (full dataset) | +1–3% | 66–71% | Medium-High |
| AR-3-04 (switch weighting) | +1–3% | 67–74% | Medium |
| AR-4 (architecture) | +1–2% | 68–76% | Medium |
| AR-5 (combined) | +0–1% | 68–77% | Medium-Low |

**Realistic target:** 70–75% Top-1 accuracy.
**Stretch target:** 77%+ if window size and data scaling compound well.

---

## 4. Key Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Aux heads remain broken after debug | Low | Medium | Disable aux loss, focus on policy only |
| Window size increase causes OOM | Medium | Low | Gradient accumulation, reduce batch size |
| Full dataset doesn't help (ceiling) | Low | Medium | Focus on data quality over quantity |
| Agents make conflicting changes | Medium | High | Single-agent-per-file ownership |
| A40 throughput bottleneck is CPU | Medium | Medium | Profile first, optimize DataLoader |
| Overfitting on larger models | Medium | Medium | Dropout sweep, early stopping |

---

## 5. Success Criteria

This AutoResearch effort is considered **successful** if:

1. **Top-1 accuracy >= 70%** on the test split of >1300 Elo Gen 3 OU battles
2. **Switch accuracy >= 55%** (up from 37–48%)
3. **Auxiliary heads functional** — speed and role heads above chance accuracy
4. **Calibration ECE < 0.10** (improved from current overconfidence)
5. **Reproducible** — results confirmed on >= 2 seeds with std < 1.0 point

The effort is considered **exceptional** if Top-1 accuracy reaches 75%+.
