# Project Plan: AutoResearch for High-Accuracy Gen 3 OU Move Prediction on a RunPod A40

## 1. Project definition

This project is a tightly scoped supervised-learning research program.

The objective is to use an AutoResearch-style agent loop to improve a Pokémon move-prediction model trained from Metamon-style replay data, with two goals only:

1. **Reach the highest possible per-turn move prediction accuracy** on **Gen 3 OU turns from players above 1300 Elo**.
2. **Reach that accuracy in the shortest possible training time** on a **single RunPod A40**.

This is **not** a general Pokémon agent project. It is **not** an end-to-end self-play reinforcement-learning project. It is **not** a ladder-optimization project. It is a focused imitation-learning and systems-optimization program.

The starting point is your existing **small imitation-learning model trained on 50,000 Gen 3 OU battles** from the Metamon dataset.

---

## 2. One-sentence mission

> Build an AutoResearch-driven supervised-learning pipeline that improves move-prediction accuracy on >1300 Elo Gen 3 OU turns faster and more reliably than manual iteration, starting from an existing 50k-battle imitation-learning baseline on a single A40.

---

## 3. Success criteria

The project succeeds only if it improves both **quality** and **speed** under a fixed evaluation protocol.

### Primary success metric
- **Top-1 move prediction accuracy** on a held-out validation and test set of **>1300 Elo Gen 3 OU turns**.

### Secondary quality metrics
- Top-3 accuracy.
- Cross-entropy / negative log-likelihood.
- Accuracy on:
  - move turns,
  - switch turns,
  - player decisionmaking

### Speed metrics
- Wall-clock time to reach:
  - baseline accuracy,
  - +1 percentage point above baseline,
  - 95% of best-observed accuracy,
  - best-observed accuracy under the fixed budget.
- GPU-hours consumed.
- Examples or turns processed per second.
- Training cost per 1-point improvement in Top-1 accuracy.

### Operational success metric
- The agentic loop must produce reproducible experiment logs, ranked experiment summaries, and a clear best-known configuration.

---

## 4. What is in scope

This plan includes only work that can improve supervised move prediction on the target distribution.

### In scope
- Dataset filtering and cleaning.
- Turn extraction and label construction.
- Feature engineering and tokenization.
- Model architecture changes for imitation learning.
- Loss design and class balancing.
- Curriculum learning for supervised training.
- Hyperparameter search.
- Throughput optimization on A40.
- Experiment automation with Codex and Claude Code agents.
- Reproducible evaluation and checkpoint ranking.

### Explicitly out of scope
- Self-play reinforcement learning.
- Offline RL.
- Reward engineering.
- Full battle-strength optimization.
- Synthetic scenario generation unless used strictly as supervised augmentation for move prediction.
- Massive autonomous repo-wide code rewriting.

This restriction matters. The fastest path to high move-prediction accuracy is to remove every distraction that does not directly improve supervised learning or throughput.

---

## 5. Working assumptions

The project assumes the following are already available or can be produced quickly:

- A functioning codebase that can load Metamon-style Gen 3 OU replay data.
- A trained **50k-battle imitation-learning checkpoint**.
- The ability to reconstruct the player-visible state and legal action space for each turn.
- Enough replay metadata to identify or approximate **>1300 Elo** battles.
- A RunPod instance with one **A40 GPU**.
- A working local or remote setup for both **Codex** and **Claude Code**.

---

## 6. Core research question

The project revolves around one practical question:

> On a single A40, what combination of data selection, representation, architecture, loss, and systems tuning yields the best Top-1 move prediction accuracy on >1300 Elo Gen 3 OU turns in the least wall-clock time?

That question should govern every experiment.

---

## 7. Baseline definition

Your current model is the anchor.

### Anchor artifact
- Name: `anchor_il_50k_adv1300`
- Training source: 50,000 Gen 3 OU battles from the Metamon dataset.
- Task: next legal action prediction from a player-visible battle state.
- Hardware reference: whatever trained the original model.

### First required baseline work
Before running AutoResearch, the team must freeze and document:

- exact input representation,
- exact label space,
- exact train/validation/test split,
- exact filtering logic,
- exact optimizer and schedule,
- exact batch size and gradient accumulation,
- exact evaluation script,
- exact random seeds.

If the anchor is not reproducible, AutoResearch will produce noise instead of progress.

---

## 8. Task formulation

The prediction task must be defined very carefully or the accuracy numbers will be misleading.

### Prediction target
At each decision point, predict the player’s actual selected legal action:
- one of the legal moves, or
- one of the legal switches.

### Recommended task variants
The project should support three evaluation variants:

1. **Unified action prediction**
   - One action vocabulary containing both moves and switches.
   - Best for direct real-world decision prediction.

2. **Move-only prediction**
   - Evaluate only turns where a move is legal and chosen.
   - Useful for isolating tactical move selection.

3. **Switch-aware conditional prediction**
   - First predict move vs switch.
   - Then predict which move or which switch.
   - Useful diagnostically, even if the main model stays unified.

### Recommendation
Use **unified action prediction** as the main benchmark and track the conditional decomposition as a diagnostic.

---

## 9. Dataset plan

The dataset will dominate the result. The first major advantage over the current baseline should come from building a cleaner target distribution.

### Target distribution
- Format: **Gen 3 OU / ADV OU**.
- Skill filter: **players above 1300 Elo**.
- Unit: player decision turn.
- Labels: legal action actually taken.

### Dataset splits
Use a strict split that prevents leakage.

Recommended hierarchy:
- train split,
- validation split,
- final test split,
- optional challenge split for rare or hard situations.

### Leakage controls
Do not allow leakage through:
- identical battles in multiple splits,
- mirrored battle states,
- duplicated replay ingestion,
- near-duplicate team compositions appearing excessively across splits,
- same replay parsed twice under different player perspectives without care.

### Filtering priorities
The first data pass should test:
- remove corrupted or incomplete replays,
- remove turns with ambiguous parsing,
- remove illegal or malformed action labels,
- restrict to >1300 Elo,
- optionally upweight replays from higher Elo buckets,
- optionally downweight repetitive low-information turns.

### Data versions
Every training run must declare a dataset version such as:
- `adv1300_v1_raw`
- `adv1300_v2_clean`
- `adv1300_v3_clean_dedup`
- `adv1300_v4_clean_dedup_weighted`

This is critical. In many supervised projects, the main gains come from data versioning, not architecture novelty.

---

## 10. Recommended data representation work

The earliest agent cycles should focus on representation quality before large model changes.

### Required state components
The representation should include, at minimum:
- current active Pokémon on both sides,
- HP, status, boosts, known moves, revealed team members,
- side conditions,
- field conditions,
- turn history summary,
- legal action mask,
- known residual damage / weather / sand context,
- trapped or forced-switch state,
- hidden-information placeholders for unknown opponent details.

### Early representation experiments
The first AutoResearch branch set should test:
- richer turn history vs shorter history,
- explicit legal-action masking vs post-hoc masking,
- separate embeddings for side state, active matchup, and history,
- normalized scalar features vs discrete bucketization,
- explicit indicator features for common ADV OU tactical patterns.

The key rule is to improve representation while keeping the benchmark fixed.

---

## 11. Model strategy

The model family should stay narrow.

### Recommended model family
Use a compact transformer-style architecture that can comfortably fit and train on one A40.

### Keep the search space small
The initial search space should be limited to:
- embedding size,
- number of layers,
- number of heads,
- context length,
- MLP width,
- dropout,
- normalization scheme,
- action head design,
- whether move/switch prediction uses one head or a hierarchical head.

### Avoid at the start
Do not begin with:
- giant architecture overhauls,
- multimodal complexity,
- MoE,
- RL integration,
- huge ensembling.

The highest probability wins will come from better data, better masking, better loss, and a well-tuned small-to-medium model.

---

## 12. Loss design

Raw cross-entropy is the default, but it should not be assumed optimal.

### Loss experiments to prioritize
1. Standard masked cross-entropy.
2. Class-weighted cross-entropy.
3. Focal-style loss for rare but important actions.
4. Auxiliary head for move-vs-switch classification.
5. Label smoothing at small values.
6. Elo-weighted loss so higher-skill turns matter more.

### Recommendation
Start with:
- masked cross-entropy,
- optional small label smoothing,
- optional auxiliary move-vs-switch head.

Only promote more complex losses if they improve validation accuracy and calibration together.

---

## 13. Training-speed strategy for A40

The A40 is capable, but the plan should assume that throughput matters as much as model quality.

### Speed objective
Every candidate configuration should be judged on:
- accuracy,
- time-to-accuracy,
- throughput,
- stability.

### Throughput optimization priorities
1. Precompute and cache parsed turn tensors.
2. Minimize Python overhead in data loading.
3. Use pinned memory and tuned worker counts.
4. Benchmark batch size and gradient accumulation.
5. Use mixed precision if numerically stable.
6. Profile dataloader bottlenecks before changing the model.
7. Save compact checkpoints and reduce logging overhead.
8. Keep evaluation efficient and incremental.

### Promotion rule
A model that is only slightly more accurate but much slower should not automatically win. The project cares about the **best accuracy reached fastest**, not just the single highest late-stage number.

---

## 14. AutoResearch operating model

The AutoResearch component should be bounded and disciplined.

### Principle
Agents are not allowed to roam the repository freely. They operate inside an experiment harness with a narrow editable surface and fixed success metrics.

### Agent roles

#### Codex role
Codex should handle:
- small code edits,
- config creation,
- log parsing,
- benchmark script improvements,
- profiling and throughput fixes,
- experiment launch scripts.

#### Claude Code role
Claude Code should handle:
- experiment design,
- ranking hypotheses,
- interpreting logs,
- writing structured research notes,
- identifying likely confounders,
- deciding which branches deserve confirmation runs.

### Human role
You remain responsible for:
- approving the search space,
- approving budget ceilings,
- promoting major branches,
- deciding when the benchmark definition changes,
- protecting the project from scope drift.

---

## 15. Agent loop design

Each cycle should be small, explicit, and reproducible.

### Standard experiment loop
1. Load current leaderboard of experiments.
2. Select one hypothesis.
3. Modify only approved files.
4. Launch a short bounded run.
5. Compare to the parent on the fixed validation harness.
6. Write a note with:
   - change made,
   - metric delta,
   - time delta,
   - likely reason,
   - whether to kill, retry, or promote.
7. Promote only if the result clears the gate.

### Approved edit surface
At the beginning, agents may edit only:
- config files,
- training schedule files,
- model definition files,
- data filtering files,
- experiment scripts,
- profiling utilities,
- result aggregation utilities.

They should not edit unrelated infrastructure, deployment code, or the benchmark definition without explicit approval.

---

## 16. Experiment budget structure

The project should use three experiment tiers.

### Tier 1: short triage runs
Purpose: eliminate weak ideas quickly.

Use for:
- checking whether a change obviously hurts,
- detecting throughput wins,
- rough ranking of candidate changes.

### Tier 2: medium confirmation runs
Purpose: verify that a promising change survives longer training.

Use for:
- best 10–20% of Tier 1 ideas,
- confirming accuracy gains are real,
- checking stability across seeds.

### Tier 3: full promotion runs
Purpose: establish a new best-known checkpoint.

Use for:
- only the strongest candidates,
- at least a small multi-seed confirmation,
- final model registration into the experiment leaderboard.

This tiering is essential on A40 because it prevents expensive dead-end runs.

---

## 17. Experiment ranking policy

The project needs a single composite ranking rule.

### Recommended primary ranking order
1. Validation Top-1 accuracy.
2. Test Top-1 accuracy if validation passes.
3. Time to reach baseline + fixed improvement threshold.
4. GPU-hours.
5. Stability across seeds.

### Suggested promotion gate
A candidate may replace the champion only if it:
- improves Top-1 accuracy by a meaningful margin, or
- matches Top-1 accuracy while materially reducing time-to-accuracy.

This prevents the system from chasing tiny accuracy gains that waste large amounts of compute.

---

## 18. Phase breakdown

## Phase 0: Benchmark lockdown
Goal: make the baseline reproducible and measurable.

Deliverables:
- frozen dataset version (subset of Metamon Replay dataset),
- frozen split file,
- reproducible baseline run,
- evaluation script,
- experiment table schema,
- champion checkpoint registry.

## Phase 1: Data cleanup and target sharpening
Goal: improve the training target before architecture expansion.

Tasks:
- enforce >1300 Elo filter for testing,
- clean parsing edge cases,
- deduplicate,
- test battle and turn weighting,
- measure class imbalance,
- create hard-situation slice reports.

## Phase 2: Throughput optimization
Goal: improve examples-per-second without harming correctness.

Tasks:
- cache features,
- optimize dataloader,
- benchmark precision modes,
- benchmark batch size and gradient accumulation,
- profile CPU/GPU utilization.

## Phase 3: Representation and loss search
Goal: improve quality with low-to-medium complexity changes.

Tasks:
- history-length ablations,
- masking improvements,
- auxiliary heads,
- class weighting,
- label smoothing,
- hierarchical action head tests.

## Phase 4: Architecture tuning
Goal: find the smallest model that reaches the best accuracy quickly.

Tasks:
- width/depth sweep,
- context-length sweep,
- dropout sweep,
- normalization and activation tests,
- parameter-efficiency vs accuracy study.

## Phase 5: Consolidation and final training
Goal: produce the champion model and its exact recipe.

Tasks:
- final confirmation runs,
- multi-seed validation,
- final test set measurement,
- best-checkpoint packaging,
- full experiment report.

---

## 19. Highest-priority early experiments

The first 15–25 agent cycles should focus on the most likely wins.

### Highest-value experiment buckets
1. **Better data filtering**
   - different Elo thresholds for training,
   - replay cleanup,
   - turn deduplication,
   - hard-turn weighting.
   - curriculum training stages

2. **Better masking and label handling**
   - ensure illegal actions are impossible,
   - improve move/switch decomposition diagnostics.

3. **History-length tuning**
   - test whether the current model lacks enough turn memory.

4. **A40 throughput optimization**
   - increase steps per hour before changing the architecture much.

5. **Small architecture sweep**
   - find whether the current model is under-capacity or over-capacity.

6. **Loss refinement**
   - class weights,
   - auxiliary move-vs-switch head,
   - small label smoothing.

These are much more likely to pay off than grander redesigns.

---

## 20. Concrete task breakdown

### Workstream A: data
- Build the replay dataset.
- Parse to per-turn decision records.
- Validate legal actions per record.
- Build split manifests.
- Build slice datasets for diagnostics.
- Version all outputs.

### Workstream B: training pipeline
- Freeze baseline trainer.
- Add fast config-driven experiment launching.
- Add mixed-precision options.
- Add gradient accumulation settings.
- Add checkpoint and metric naming conventions.

### Workstream C: evaluation
- Build one command that evaluates any checkpoint on:
  - full validation set,
  - full test set,
  - slice metrics,
  - throughput summary.
- Add champion comparison reports.

### Workstream D: profiling
- Measure dataloader speed.
- Measure GPU utilization.
- Benchmark caching strategies.
- Benchmark batch size / accumulation combinations.

### Workstream E: AutoResearch harness
- Add experiment registry.
- Add template prompts for Codex and Claude Code.
- Add run wrappers with bounded budgets.
- Add result summarization.
- Add auto-generated markdown notes per run.

### Workstream F: model search
- Run controlled sweeps over:
  - depth,
  - width,
  - context length,
  - loss variant,
  - history handling,
  - head design.

---

## 21. Implementation breakdown for RunPod A40

The deployment should be simple and reproducible.

### Base pod setup
- Launch one RunPod pod with an **A40**.
- Use a persistent volume for datasets, checkpoints, and logs.
- Install CUDA-compatible PyTorch and all project dependencies.
- Install Node.js if needed for Codex CLI.
- Install Git, tmux, htop, nvtop, and profiling tools.

### Recommended directory layout
```text
/workspace/project/
  data/
    raw/
    processed/
    manifests/
  checkpoints/
  logs/
  experiments/
  reports/
  scripts/
  configs/
  prompts/
```

### Minimal services to keep running
- one terminal session for training,
- one for evaluation,
- one for Codex,
- one for Claude Code,
- one for monitoring and log inspection.

Use tmux so sessions survive disconnects.

---

## 22. How to run Codex and Claude Code on the pod

The exact tools may evolve, but the operating pattern should stay the same.

### Codex operating pattern
Use Codex for bounded code-edit and experiment-launch tasks.

Recommended workflow:
1. SSH into the RunPod pod.
2. Start a tmux session.
3. Open the repo.
4. Give Codex a narrow instruction such as:
   - optimize dataloader throughput,
   - add a config for weighted loss,
   - generate a benchmark script,
   - patch a bug in the evaluation harness.
5. Review the diff before running.
6. Run the bounded experiment.
7. Save the result note.

### Claude Code operating pattern
Use Claude Code for experiment planning, higher-level analysis, and ranking decisions.

Recommended workflow:
1. Point Claude Code at the repo and experiment logs.
2. Ask it to inspect the latest leaderboard.
3. Ask it to propose the next 3–5 highest-value experiments.
4. Have it write a short rationale for each.
5. Use it to summarize why the winning branch likely improved.

### Best practice
Do not let both agents edit the same files simultaneously. Assign ownership by workstream per session.

---

## 23. Prompt structure for the agents

The prompts should be short and operational.

### Codex prompt template
- You are working on a supervised-learning system for Gen 3 OU move prediction.
- Goal: improve validation Top-1 accuracy or reduce time-to-accuracy on A40.
- You may edit only approved files.
- Keep changes small and localized.
- After edits, provide:
  - changed files,
  - expected effect,
  - risk,
  - exact command to run.

### Claude Code prompt template
- You are the research planner for a supervised move-prediction project.
- Your task is to analyze experiment history and propose the next most useful bounded experiments.
- Optimize for:
  - higher validation Top-1 accuracy,
  - faster time-to-accuracy,
  - low confounding risk.
- Output:
  - ranked experiments,
  - why each might work,
  - expected runtime,
  - promotion criteria.

---

## 24. Experiment logging standard

Every run should produce a machine-readable record and a short markdown summary.

### Required fields
- experiment id,
- parent experiment id,
- dataset version,
- config hash,
- model size,
- batch size,
- precision mode,
- train duration,
- validation metrics,
- test metrics,
- throughput,
- GPU-hours,
- notes,
- promote / kill decision.

This logging system is the backbone of the AutoResearch method.

---

## 25. Risks

### Risk 1: metric inflation from leakage
Mitigation:
- strict deduplication,
- fixed splits,
- hold-out test set,
- team/replay leak checks.

### Risk 2: agents generate noisy changes
Mitigation:
- narrow editable surface,
- one hypothesis per run,
- mandatory experiment notes,
- human approval for benchmark changes.

### Risk 3: throughput is limited by data loading, not compute
Mitigation:
- profile first,
- cache tensors,
- benchmark worker counts and storage layout.

### Risk 4: over-optimizing easy common actions
Mitigation:
- slice metrics,
- move vs switch diagnostics,
- hard-turn evaluation,
- class weighting experiments.

### Risk 5: project scope expands again
Mitigation:
- reject all RL and battle-strength work until the supervised benchmark is complete.

---

## 26. Final deliverables

At the end of the project, the output should be:

1. A reproducible **best-known Gen 3 OU >1300 Elo move-prediction checkpoint**.
2. A frozen dataset and split definition.
3. A ranked experiment table.
4. A short report describing what improved accuracy most.
5. A short report describing what improved training speed most.
6. A reusable AutoResearch harness for future supervised Pokémon imitation-learning work.

---

## 27. Practical next steps

The immediate next actions should be:

1. Freeze and benchmark the current 50k-battle anchor.
2. Build the reply dataset manifest and held-out splits.
3. Add a one-command evaluation harness for quality and throughput.
4. Profile the current training pipeline on A40.
5. Run the first short experiment slate on:
   - data cleanup,
   - masking correctness,
   - batch size / accumulation,
   - history length,
   - weighted loss.

That sequence gives the highest chance of fast gains with minimal wasted compute.
