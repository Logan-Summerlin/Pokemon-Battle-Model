# P8-Lean Inference Stress Test + RL Stage Proposal

_Date: 2026-03-12_

## 1) Repository-informed context

This proposal is grounded in the current Phase 4 training/evaluation stack and project roadmap:

- **P8-Lean training wrapper** already defines a compact imitation-learning profile (3L/224d/4H, short history window, aux on, value off) and routes through the common Phase 4 trainer (`scripts/train_p8_lean.py` → `scripts/train_phase4.py`).
- **Core model** (`src/models/battle_transformer.py`) supports policy logits with legality masking, auxiliary hidden-info predictions, optional value head, and sequence-window handling.
- **Inference in live battles** is already operational via `ModelBot` + `BattleEnv` + `BattleEvaluator` (`src/bots/model_bot.py`, `src/environment/battle_env.py`, `src/evaluation/battle_evaluator.py`).
- **Project plan alignment** explicitly recommends BC-first, targeted synthetic repair, then conditional RL advancement with strict gates, and only later self-play scaling (`docs/IMPLEMENTATION_PLAN.md`, `archive/pokemon_model_project_plan.md`).

---

## 2) Goals for the next stage

### Goal A: Validate inference behavior (not just offline top-k accuracy)
Test how the trained P8-Lean policy behaves in **synthetic battle scenarios** that isolate tactical and hidden-information edge cases.

### Goal B: Start a constrained RL stage
Initialize from best BC (P8-Lean) and run repeated battles against synthetic opponents to improve decision quality and robustness without breaking calibration/legality.

---

## 3) Proposal Part I — Synthetic Inference Test Program

## 3.1 Build a scenario taxonomy (first)
Create a scenario catalog with reproducible seeds and labels. Minimum families:

1. **Forced mechanics**
   - forced switch situations
   - trap/choice-lock/disable interactions
   - no-safe-switch endgames
2. **Tactical one-turn motifs**
   - guaranteed KO available
   - sack vs preserve decision
   - setup-vs-attack race
3. **Hidden-info ambiguity**
   - unknown item/ability affecting speed or damage assumptions
   - move reveal uncertainty (coverage threats)
4. **Distribution shift / off-meta legality**
   - legal but uncommon sets to test robustness
5. **Tempo and game-phase states**
   - early momentum pivots
   - midgame hazard management
   - late-game conversion positions

Output artifact: `scenario_manifest.jsonl` with fields:
`scenario_id, family, seed, team_a, team_b, initial_state_patch, expected_label_type, tags`.

## 3.2 Label policy quality with three label types
Use the same philosophy already documented for synthetic repair:

- **Hard-optimal** (forced move only)
- **Acceptable set** (2-3 strategically defensible actions)
- **Distributional target** (for ambiguity; score probabilistic alignment)

Scoring should avoid over-penalizing cases with multiple viable lines.

## 3.3 Define inference metrics per scenario family
Beyond top-1 accuracy:

- legality rate (must remain 100%)
- top-1/top-3 vs label policy
- probability mass on acceptable set
- entropy in ambiguous states (overconfidence detector)
- calibration (ECE/Brier) on hard-labeled subsets
- hidden-info head consistency (item/speed/role predictions by turn)

## 3.4 Add an inference harness script
Implement a new script (recommended path: `scripts/evaluate_synthetic_inference.py`) that:

1. Loads checkpoint + vocabs.
2. Loads scenario manifest.
3. Instantiates `ModelBot` with configurable temperature (0.0 greedy, >0 stochastic).
4. Runs scenarios in deterministic batches (fixed seed).
5. Produces:
   - aggregate metrics by family
   - per-scenario traces (`chosen_action`, legal set, logits/probs, aux preds)
   - failure clusters for triage.

## 3.5 Add failure clustering output
For each failed scenario, assign one primary cause:

- mechanics violation
- tactical ranking error
- ambiguity mismanagement (overcommit)
- calibration mismatch
- hidden-info miss

This cluster report is the bridge to RL curriculum design.

## 3.6 Promotion gate from inference testing to RL
Only start RL if all are met:

- legality 100%
- no catastrophic family (<35% acceptable-set hit rate) on core tactical families
- ambiguity calibration within threshold (e.g., ECE not worse than BC baseline by >0.02)
- hidden-info auxiliary quality non-regressing vs current held-out replay eval

---

## 4) Proposal Part II — RL Stage Against Synthetic Opponents

## 4.1 RL framing and constraints
Use **BC-initialized policy** (best P8-Lean checkpoint) as actor initialization; do not train from scratch.

Environment side:
- use existing `BattleEnv` + `BattleEvaluator` loop primitives
- define a synthetic-opponent pool (random legal, heuristic, scripted tactical specialists, historical frozen snapshots)
- rotate opponents by curriculum stage.

## 4.2 Start with conservative objective and short horizons
Recommended first objective:

- terminal reward (+1/-1), optional tiny shaping terms only after baseline stability
- clipped policy updates (PPO-style) or conservative offline-to-online hybrid
- maintain legality mask in policy head at all times

Reason: current plan warns about reward hacking and instability if shaping/RL is introduced too aggressively.

## 4.3 Training loop design (practical)
Per iteration:

1. Sample opponents from weighted pool.
2. Roll out N battles (e.g., 512–2048) with stochastic policy temperature.
3. Store trajectories with:
   - observations (first-person only)
   - legal masks
   - action logits/probs
   - rewards/outcomes
   - opponent identity + scenario tags
4. Compute policy/value losses.
5. Update model for K minibatch epochs.
6. Run evaluation battery after each outer iteration.
7. Keep best checkpoint by multi-metric gate (not win rate alone).

## 4.4 Opponent curriculum schedule
Stage progression:

- **Stage R1 (stability):** random + simple heuristic opponents.
- **Stage R2 (competence):** stronger heuristic + scripted tactical specialists.
- **Stage R3 (robustness):** mixed pool including off-meta and frozen historical policies.

Promotion requires passing per-stage gates on win rate + robustness + calibration.

## 4.5 Critical anti-collapse safeguards
1. **Replay mix anchoring**: keep a fraction of BC replay batches in each RL epoch to prevent forgetting.
2. **KL regularization to BC prior**: constrain large policy drift in early RL.
3. **Population checkpoints**: evaluate against older snapshots to detect cyclic/meta overfitting.
4. **Calibration watchdog**: stop/rollback if confidence rises while correctness drops.
5. **Family-level metrics**: block promotion if any key family regresses materially.

## 4.6 RL evaluation battery (must run every iteration)
- Win rate vs each opponent family (not just pooled mean)
- Synthetic gate suite (from Part I)
- Held-out replay action metrics (top-1/top-3/NLL/ECE)
- Hidden-info auxiliary metrics by turn
- Uncommon-set robustness tests

**Kill criteria** (from project-plan philosophy):
- in-distribution win improves but robustness drops significantly
- calibration worsens materially
- gains smaller than equivalent BC/synthetic repair effort

## 4.7 Minimal implementation plan (engineering tasks)

### Task A — New RL config + runner
- Add `configs/training/rl_stage1.yaml`.
- Add `scripts/train_rl_stage1.py` to orchestrate rollout/update/eval.

### Task B — Trajectory buffer
- Add `src/training/trajectory_buffer.py` for battle trajectories with legal masks and metadata.

### Task C — Opponent pool abstraction
- Add `src/bots/opponent_pool.py` with weighted sampling and stage-based curriculum.

### Task D — Evaluation integration
- Reuse `BattleEvaluator` + add synthetic inference harness invocation each iteration.

### Task E — Governance and reproducibility
- Save every iteration report JSON with:
  `opponent_mix, win_rates_by_family, calibration, hidden-info metrics, checkpoint_hash`.

---

## 5) 6-week execution timeline (conservative)

### Week 1
- Implement scenario manifest format + 100-200 synthetic scenarios.
- Implement synthetic inference evaluator script.

### Week 2
- Run P8-Lean checkpoint on scenario suite.
- Triage failures and finalize promotion gate thresholds.

### Week 3
- Implement RL runner skeleton + trajectory buffer + opponent pool (R1 only).

### Week 4
- Run short RL experiments (small battle budgets), tune stability controls (KL, replay mix).

### Week 5
- Expand to R2 opponents, run full evaluation battery every iteration.

### Week 6
- Robustness-focused R3 pass; produce go/no-go report for larger-scale RL/self-play.

---

## 6) Expected outputs

1. `synthetic_inference_report.json` (+ family breakdown CSV/markdown).
2. RL iteration reports per checkpoint.
3. Best RL checkpoint + regression comparison vs BC P8-Lean baseline.
4. Decision memo: continue RL scale-up vs return to feature/data improvements.

---

## 7) Why this fits the current codebase

- Reuses existing model forward/inference path (policy + aux, legality mask).
- Reuses existing bot/environment/evaluator battle infrastructure.
- Matches documented project sequence: BC → targeted synthetic validation/repair → conditional RL.
- Keeps first-person hidden-information constraints intact and measurable.

