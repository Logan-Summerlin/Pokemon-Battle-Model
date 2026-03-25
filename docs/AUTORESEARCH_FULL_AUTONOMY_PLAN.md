# Plan: Converting AutoResearch to Fully Autonomous Operation

## Goal

Transform the Pokemon Battle AutoResearch repository from a human-in-the-loop experiment system into a fully autonomous research loop, modeled after Karpathy's autoresearch. The user prompts a single agent once, and the agent runs experiments indefinitely until interrupted.

---

## 1. How Karpathy's AutoResearch Achieves Full Autonomy

Karpathy's design has three critical properties that enable autonomy:

| Property | How It Works | Why It Enables Autonomy |
|----------|-------------|------------------------|
| **Single file to edit** | Agent only modifies `train.py` | No coordination needed, no file conflicts, simple diffs |
| **Fixed time budget** | Every experiment takes exactly 5 minutes | Predictable loop timing, no runaway trains |
| **Binary success metric** | `val_bpb` went down → keep; didn't → revert | Decision is mechanical, no judgment calls |
| **Git as memory** | Branch advances on success, resets on failure | State management is built into git |
| **No config files** | Everything lives in `train.py` | No config/code synchronization issues |
| **Crash recovery** | Crashes are logged and skipped | Agent never gets permanently stuck |

The key insight: **the agent never needs to ask a question because every decision has a clear protocol.**

---

## 2. Gaps Between Current AutoResearch and Full Autonomy

### Current Design (Human-in-the-Loop)

```
Human decides direction → Claude Code designs experiment → Codex implements →
Human reviews → Training runs → Claude Code analyzes → Human decides next direction
```

### Problems for Autonomy

| Gap | Description | Severity |
|-----|-------------|----------|
| **Two agents required** | Claude Code and Codex have different roles; coordination requires human routing | HIGH |
| **Complex promotion rules** | "≥0.5% improvement OR speed gate OR stability gate" requires judgment | MEDIUM |
| **Multi-file edit surface** | Agent can edit 6+ files; more surface = more ways to break things | MEDIUM |
| **No automatic revert** | Failed experiments aren't automatically reverted via git | HIGH |
| **Tiered budgets require planning** | Agent must decide Tier 1/2/3 — adds a decision point | LOW |
| **Config + code separation** | Experiments defined in YAML configs AND code changes | MEDIUM |
| **No crash recovery protocol** | If training crashes, the loop stalls | HIGH |

---

## 3. The Autonomy Conversion Plan

### Phase A: Simplify to Single-Agent, Single-File

**Eliminate Codex entirely.** One agent, one file, one loop.

**Collapse the edit surface to `Autoresearch/run_experiment.py` only.**

Currently `run_experiment.py` wraps `scripts/train_phase4.py` with command-line arguments. The key insight: every experiment in the current plan (window size, batch size, data scaling, architecture changes) can be expressed as argument overrides to `train_phase4.py`. The agent doesn't need to edit the training script — it needs to call it with different arguments.

For architecture changes and loss modifications that can't be expressed as args, the agent edits `run_experiment.py` to add new argument-forwarding logic. This keeps all agent-authored code in one file.

**New `.claude/settings.json`:**
```json
{
  "permissions": {
    "allow": [
      "Read",
      "Glob",
      "Grep",
      "Edit(Autoresearch/run_experiment.py)",
      "Write(Autoresearch/results/**)",
      "Write(Autoresearch/notes/**)",
      "Bash(python Autoresearch/run_experiment.py *)",
      "Bash(python Autoresearch/eval_harness.py *)",
      "Bash(python Autoresearch/leaderboard.py *)",
      "Bash(python -m pytest *)",
      "Bash(grep *)",
      "Bash(tail *)",
      "Bash(git *)",
      "Bash(nvidia-smi)",
      "Bash(ls *)"
    ],
    "deny": [
      "Edit(src/**)",
      "Edit(scripts/**)",
      "Edit(tests/**)",
      "Edit(data/**)",
      "Bash(rm -rf *)"
    ]
  }
}
```

**If architecture changes are needed** (e.g., hierarchical action head), the agent should be allowed to edit `src/models/battle_transformer.py` and `scripts/train_phase4.py` as well. In this case, expand the allow list but keep the Karpathy-style revert discipline: git commit before each experiment, git reset if it fails.

**Recommended approach for maximum autonomy:** Allow editing `train.py` equivalent files (`scripts/train_phase4.py`, `src/models/battle_transformer.py`, `src/data/dataset.py`, `src/data/auxiliary_labels.py`) but enforce the git-revert protocol in the CLAUDE.md instructions.

### Phase B: Implement Automatic Git-Based State Management

Add the Karpathy-style branch/revert protocol to CLAUDE.md:

```markdown
## Git Protocol (Mandatory)

Before EVERY experiment:
1. `git add -A && git commit -m "EXP-{id}: {description}"`

After EVERY experiment:
- If Top-1 accuracy improved by ≥0.3%: KEEP the commit (branch advances)
- If accuracy is equal or worse: `git reset --hard HEAD~1` (revert to previous state)
- If crashed: `git reset --hard HEAD~1`, log as crash, move on

The branch always represents the current best configuration.
```

### Phase C: Fixed Time Budget Per Experiment

Replace the tiered budget system with a single fixed budget (like Karpathy's 5-minute rule):

```markdown
## Time Budget

Every experiment trains for a FIXED budget of 15 minutes wall clock time on A40.
This is approximately 5-8 epochs on 50K battles with batch_size=256.

Do not change this budget. It ensures:
- ~4 experiments per hour
- ~32 experiments in an 8-hour overnight run
- Fair comparison between all experiments

To enforce this, always pass: --budget-minutes 15
```

**Why 15 minutes instead of 5:** Pokemon battle training is more complex than Karpathy's character-level LM. The model needs enough epochs to show learning signal. 15 minutes is the sweet spot — long enough for meaningful results, short enough for rapid iteration.

**Alternative: adaptive tiering.** If you want to keep the tier system, automate the promotion path:
- All experiments start as Tier 1 (15 min)
- If Tier 1 shows ≥0.5% gain, auto-promote to Tier 2 (60 min) on next iteration
- If Tier 2 confirms (≥0.3% gain over anchor), auto-promote to Tier 3 (4 hr, multi-seed)

### Phase D: Binary Success Metric

Simplify the promotion rules to a mechanical decision:

```markdown
## Decision Protocol

After each experiment, compute the delta:
  delta = new_top1_accuracy - current_champion_top1_accuracy

Decision:
- delta > 0.003 (0.3 percentage points): KEEP ✓
- delta ≤ 0.003: DISCARD ✗

That's it. No speed gates, no stability gates for Tier 1/2.
For Tier 3 (final confirmation only): require 2/3 seeds to improve.
```

### Phase E: Crash Recovery

Add explicit crash handling to the experiment loop:

```markdown
## Crash Handling

If `run_experiment.py` crashes:
1. Read the last 50 lines of the training log
2. If it's a simple fix (typo, import error, OOM): fix and retry ONCE
3. If it crashes again or the idea is fundamentally broken:
   - Log as "crash" in the registry
   - `git reset --hard HEAD~1`
   - Move on to the next hypothesis
4. NEVER spend more than 10 minutes debugging a single crash
```

### Phase F: Autonomous Idea Generation

The hardest part of full autonomy: where do new experiment ideas come from?

**Build an idea queue into CLAUDE.md:**

```markdown
## Experiment Priority Queue

Work through these in order. Skip any that have already been tried
(check the registry). When the queue is exhausted, generate new
hypotheses from the results so far.

### Priority 1: Low-Hanging Fruit
1. Window size: 2 → 5 (biggest expected gain)
2. Window size: 5 → 10
3. Full 100K battles (vs 50K)
4. Batch size: 64 → 256 (A40 headroom)

### Priority 2: Loss Engineering
5. Class-weighted loss: upweight switch actions 2x
6. Class-weighted loss: upweight switch actions 3x
7. Label smoothing 0.05
8. Label smoothing 0.1
9. Aux weight sweep: 0.1, 0.2, 0.3, 0.5

### Priority 3: Architecture
10. 4L/256d/4H (P8 scale-up)
11. 6L/384d/6H (P4 scale-up)
12. FFN multiplier: 3x → 4x
13. Dropout: 0.1 → 0.15
14. Dropout: 0.1 → 0.05

### Priority 4: Advanced
15. Combine best window + best data + best architecture
16. Hierarchical action head (move-vs-switch classifier)
17. Elo-weighted loss (upweight 1500+)
18. Value head enabled (win prediction auxiliary)

### Priority 5: Agent-Generated
When the above are exhausted, analyze results and generate new hypotheses.
Look for patterns: which changes helped? Which interaction effects remain untested?
```

---

## 4. The Fully Autonomous CLAUDE.md

Here is the complete rewritten CLAUDE.md for fully autonomous operation:

```markdown
# Pokemon Battle AutoResearch — Autonomous Agent Instructions

## Your Mission

You are an autonomous ML researcher. Your goal: maximize Top-1 action
prediction accuracy on Gen 3 OU Pokemon battles. The anchor checkpoint
achieves 63.21%. Beat it.

You run experiments in a loop, forever, until the human stops you.
Do not ask for permission. Do not pause between experiments. Do not
wait for feedback. The experiment loop IS the program.

## Setup (One Time)

1. Verify data: `ls data/processed/battles/ | wc -l` (expect ~75K+ files)
2. Verify anchor: `python Autoresearch/eval_harness.py --checkpoint checkpoints/phase4_p8_lean_50k/seed_42/best_model.pt --data-dir data/processed --output Autoresearch/results/anchor.json`
3. Verify anchor reproduces: 63.21% Top-1, 89.27% Top-3
4. Create branch: `git checkout -b autoresearch/<date>`
5. Confirm setup, then BEGIN THE LOOP.

## The Experiment Loop

LOOP FOREVER:

1. READ the registry: `python Autoresearch/leaderboard.py`
2. PICK the next experiment from the priority queue (or generate one)
3. COMMIT: `git add -A && git commit -m "EXP-{id}: {description}"`
4. RUN: `python Autoresearch/run_experiment.py --name "{name}" --parent {parent} --tier 1 --config-override {overrides} --budget-minutes 15 > run.log 2>&1`
5. READ results: `grep "Top-1\|Top-3\|Switch" run.log`
6. If empty (crash): `tail -50 run.log`, attempt fix once, else log crash and revert
7. RECORD in registry and write note to `Autoresearch/notes/`
8. DECIDE:
   - If Top-1 improved by ≥0.3%: KEEP (branch advances)
   - If not: `git reset --hard HEAD~1` (revert)
9. GOTO 1

## NEVER STOP

Do not pause. Do not ask. The loop runs until interrupted.
If you run out of ideas, think harder. Re-read the model.
Re-read the loss function. Analyze per-action accuracy patterns.
Combine near-misses. Try radical changes.

~4 experiments/hour × 8 hours = ~32 experiments overnight.
Make every one count.
```

---

## 5. Implementation Steps

| Step | Action | Time |
|------|--------|------|
| 1 | Fix repo issues (pyproject.toml, vocabs, directories) per AUTORESEARCH_REPO_FIXES.md | 30 min |
| 2 | Rewrite CLAUDE.md to the autonomous version above | 15 min |
| 3 | Remove `.codex/` directory (single-agent model) | 5 min |
| 4 | Update `.claude/settings.json` to the autonomy-compatible permissions | 10 min |
| 5 | Add the experiment priority queue to CLAUDE.md | 15 min |
| 6 | Add git-revert protocol to `run_experiment.py` (auto-revert on failure) | 30 min |
| 7 | Add `--budget-minutes` enforcement to `run_experiment.py` | 15 min |
| 8 | Test the full loop locally with `--mode smoke` | 15 min |
| 9 | Push to GitHub | 5 min |
| 10 | Launch RunPod, clone, install, prompt Claude Code once | 20 min |

**Total conversion time: ~2.5 hours**

---

## 6. The Launch Prompt

Once everything is set up on RunPod, the single prompt to kick off autonomous research:

```
Read program.md and CLAUDE.md. This is a fully autonomous research session.
Set up the experiment branch, verify the anchor, and begin the experiment loop.
Run experiments continuously. Do not stop or ask for permission.
Target: beat the 63.21% Top-1 anchor accuracy on Gen 3 OU battle prediction.
Start with the priority queue in CLAUDE.md. Go.
```

Then close the laptop and go to sleep.

---

## 7. Key Differences from Karpathy's Design

| Aspect | Karpathy | Pokemon AutoResearch |
|--------|----------|---------------------|
| Domain | Character-level LM (GPT) | Pokemon battle prediction (behavior cloning) |
| Time budget | 5 min fixed | 15 min fixed |
| Single metric | val_bpb (lower is better) | Top-1 accuracy (higher is better) |
| Files to edit | `train.py` only | `run_experiment.py` only (or +train_phase4.py, battle_transformer.py for arch changes) |
| Data prep | `prepare.py` (fixed) | `process_dataset.py` (fixed, run once) |
| Eval | `evaluate_bpb()` in prepare.py | `eval_harness.py` (fixed) |
| State | git branch advance/revert | git branch advance/revert + experiment registry JSON |
| Agent | Claude Code (single) | Claude Code (single — Codex removed) |
| Experiments/hour | ~12 (5 min each) | ~4 (15 min each) |
| Overnight yield | ~100 experiments | ~32 experiments |

---

## 8. Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Agent gets stuck in a loop of bad ideas | Priority queue provides 18+ pre-planned experiments before agent-generated ones |
| Agent breaks the codebase | Git revert after every failed experiment; denied edit on data pipeline |
| Agent edits protected files | `.claude/settings.json` deny list enforces boundaries |
| RunPod pod terminated mid-run | Persistent volume preserves data; git commits preserve progress |
| Agent runs out of context window | Registry + notes provide persistent memory across context resets |
| Training OOM on large config | `run_experiment.py` catches OOM, logs crash, reverts |
| Agent stops autonomously | "NEVER STOP" directive in CLAUDE.md; no confirmation points in the loop |
