# Plan: Autonomous Single-Agent AutoResearch Operation

## Goal

Transform the Pokemon Battle AutoResearch system into a fully autonomous single-agent research loop. One Claude Code agent runs experiments indefinitely until interrupted, modeled after Karpathy's autoresearch. No human routing, no second agent, no coordination overhead.

---

## 1. How Karpathy's AutoResearch Achieves Full Autonomy

| Property | How It Works | Why It Enables Autonomy |
|----------|-------------|------------------------|
| **Single file to edit** | Agent only modifies `train.py` | No coordination needed, no file conflicts, simple diffs |
| **Fixed time budget** | Every experiment takes exactly 5 minutes | Predictable loop timing, no runaway trains |
| **Binary success metric** | `val_bpb` went down -> keep; didn't -> revert | Decision is mechanical, no judgment calls |
| **Git as memory** | Branch advances on success, resets on failure | State management is built into git |
| **No config files** | Everything lives in `train.py` | No config/code synchronization issues |
| **Crash recovery** | Crashes are logged and skipped | Agent never gets permanently stuck |

The key insight: **the agent never needs to ask a question because every decision has a clear protocol.**

---

## 2. Gaps Between Current AutoResearch and Full Autonomy

### Current Design (Human-in-the-Loop)

```
Human decides direction -> Claude Code designs experiment -> Codex implements ->
Human reviews -> Training runs -> Claude Code analyzes -> Human decides next direction
```

### Problems for Autonomy

| Gap | Description | Severity |
|-----|-------------|----------|
| **Two agents required** | Claude Code and Codex have different roles; coordination requires human routing | HIGH |
| **No automatic revert** | Failed experiments aren't automatically reverted via git | HIGH |
| **No crash recovery protocol** | If training crashes, the loop stalls | HIGH |
| **Config + code separation** | Experiments defined in YAML configs AND code changes | MEDIUM |
| **Codex files add complexity** | `.codex/AGENTS.md` and `.codex/config.toml` define a second agent that won't exist | MEDIUM |

---

## 3. The Single-Agent Autonomy Plan

### Phase A: Remove Codex and Consolidate to Claude Code

**Eliminate Codex entirely.** One agent, one loop, zero coordination overhead.

Claude Code is capable of everything Codex was responsible for: config generation, script fixes, log parsing, and metric extraction. Splitting these across two agents added coordination complexity with no real benefit for autonomous operation.

**Files to remove from the Autoresearch repository:**

| File | Reason for Removal |
|------|-------------------|
| `Autoresearch/.codex/AGENTS.md` | Codex agent instructions — no longer needed |
| `Autoresearch/.codex/config.toml` | Codex sandbox configuration — no longer needed |
| `Autoresearch/prompts/codex_template.md` | Codex prompt template — no longer needed |

**Files to update:**

| File | Change |
|------|--------|
| `Autoresearch/.claude/settings.json` | Expand permissions to cover all tasks Claude Code now owns (configs, run_experiment.py, etc.) |
| `Autoresearch/.claude/rules/memory.md` | Update to reflect single-agent model |
| `Autoresearch/AUTORESEARCH_PLAN.md` | Remove all Codex references from agent operating model |

**New `.claude/settings.json` for full autonomy:**

```json
{
  "permissions": {
    "allow": [
      "Read",
      "Glob",
      "Grep",
      "Edit(Autoresearch/**)",
      "Write(Autoresearch/**)",
      "Edit(configs/**)",
      "Edit(scripts/train_phase4.py)",
      "Edit(src/models/battle_transformer.py)",
      "Edit(src/data/dataset.py)",
      "Edit(src/data/auxiliary_labels.py)",
      "Bash(python scripts/train_phase4.py *)",
      "Bash(python Autoresearch/*.py *)",
      "Bash(python -m pytest *)",
      "Bash(pytest *)",
      "Bash(git status)",
      "Bash(git diff *)",
      "Bash(git add *)",
      "Bash(git commit *)",
      "Bash(git log *)",
      "Bash(git push *)",
      "Bash(git reset *)",
      "Bash(git checkout *)",
      "Bash(git pull *)",
      "Bash(ls *)",
      "Bash(nvidia-smi)",
      "Bash(python -c *)",
      "Bash(pip install *)",
      "Bash(cat Autoresearch/*)",
      "Bash(grep *)",
      "Bash(tail *)",
      "Bash(head *)",
      "Bash(wc *)"
    ],
    "deny": [
      "Edit(src/data/observation.py)",
      "Edit(src/data/tensorizer.py)",
      "Edit(src/data/replay_parser.py)",
      "Edit(data/splits/**)",
      "Edit(docs/EVALUATION_SPEC.md)",
      "Bash(rm -rf *)",
      "Bash(rm -r data/*)",
      "Bash(rm -r checkpoints/*)"
    ]
  }
}
```

Key additions vs current settings:
- `git reset`, `git checkout`, `git pull` — needed for the revert protocol
- `grep`, `tail`, `head`, `wc` — needed for log inspection and crash diagnosis
- All `Autoresearch/**` write access — Claude Code now owns configs, run_experiment.py, notes, everything

### Phase B: Implement Automatic Git-Based State Management

Add the Karpathy-style branch/revert protocol:

```markdown
## Git Protocol (Mandatory)

Before EVERY experiment:
1. `git add -A && git commit -m "EXP-{id}: {description}"`

After EVERY experiment:
- If the experiment meets the promotion rules: KEEP the commit (branch advances)
- If accuracy is equal or worse: `git reset --hard HEAD~1` (revert to previous state)
- If crashed: `git reset --hard HEAD~1`, log as crash, move on

The branch always represents the current best configuration.
```

### Phase C: Crash Recovery

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

### Phase D: Autonomous Idea Generation

Build an idea queue into CLAUDE.md that the agent works through:

```markdown
## Experiment Priority Queue

Work through these in order. Skip any that have already been tried
(check the registry). When the queue is exhausted, generate new
hypotheses from the results so far.

### Priority 1: Low-Hanging Fruit
1. Window size: 2 -> 5 (biggest expected gain)
2. Window size: 5 -> 10
3. Full 100K battles (vs 50K)
4. Batch size: 64 -> 256 (A40 headroom)

### Priority 2: Loss Engineering
5. Class-weighted loss: upweight switch actions 2x
6. Class-weighted loss: upweight switch actions 3x
7. Label smoothing 0.05
8. Label smoothing 0.1
9. Aux weight sweep: 0.1, 0.2, 0.3, 0.5

### Priority 3: Architecture
10. 4L/256d/4H (P8 scale-up)
11. 6L/384d/6H (P4 scale-up)
12. FFN multiplier: 3x -> 4x
13. Dropout: 0.1 -> 0.15
14. Dropout: 0.1 -> 0.05

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

## 4. Claude Code's Full Responsibility Set

With Codex removed, Claude Code owns the entire research loop:

| Responsibility | Previously | Now |
|---------------|-----------|-----|
| Experiment planning | Claude Code | Claude Code |
| Config generation | Codex | Claude Code |
| Code changes (model, data, loss) | Claude Code | Claude Code |
| Script fixes (run_experiment.py) | Codex | Claude Code |
| Log parsing and metric extraction | Codex | Claude Code |
| Result analysis and notes | Claude Code | Claude Code |
| Leaderboard updates | Claude Code | Claude Code |
| Bug investigation | Claude Code | Claude Code |
| Git state management (commit/revert) | Neither (manual) | Claude Code |

This is actually simpler than the dual-agent model — no coordination, no message passing, no conflict avoidance. The agent plans, implements, runs, evaluates, and records in a single tight loop.

---

## 5. Implementation Steps

| Step | Action | Time |
|------|--------|------|
| 1 | Fix repo issues (pyproject.toml, vocabs, directories) per AUTORESEARCH_REPO_FIXES.md | 30 min |
| 2 | Delete `Autoresearch/.codex/` directory (AGENTS.md + config.toml) | 2 min |
| 3 | Delete `Autoresearch/prompts/codex_template.md` if it exists | 1 min |
| 4 | Update `Autoresearch/.claude/settings.json` to the full-autonomy permissions above | 5 min |
| 5 | Update `Autoresearch/.claude/rules/memory.md` — remove all Codex references, add single-agent protocol | 10 min |
| 6 | Add "NEVER STOP" directive to `Autoresearch/CLAUDE.md` (from AUTORESEARCH_AGENT_MOTIVATION.md) | 10 min |
| 7 | Add git-revert protocol to `Autoresearch/CLAUDE.md` | 5 min |
| 8 | Add crash recovery instructions to `Autoresearch/CLAUDE.md` | 5 min |
| 9 | Add the experiment priority queue to `Autoresearch/CLAUDE.md` | 10 min |
| 10 | Add auto-revert logic to `run_experiment.py` (revert on failure, keep on success) | 30 min |
| 11 | Remove Codex references from `Autoresearch/AUTORESEARCH_PLAN.md` agent operating model section | 10 min |
| 12 | Test the full loop locally with `--mode smoke` | 15 min |
| 13 | Push to GitHub | 5 min |
| 14 | Launch RunPod, clone, install, prompt Claude Code once | 20 min |

**Total conversion time: ~2.5 hours**

---

## 6. The Launch Prompt

Once everything is set up on RunPod, the single prompt to kick off autonomous research:

```
Read Autoresearch/CLAUDE.md. This is a fully autonomous research session.

Set up the experiment branch, verify the anchor, and begin the experiment loop.
Run experiments continuously. Do not stop or ask for permission.

Work through the priority queue. Generate configs inline. Edit any approved file
as needed. Commit before every experiment, revert on failure, advance on success.

Target: beat the 63.21% Top-1 anchor accuracy on Gen 3 OU battle prediction.
Go.
```

Then close the laptop and go to sleep.

---

## 7. Key Differences from Karpathy's Design

| Aspect | Karpathy | Pokemon AutoResearch |
|--------|----------|---------------------|
| Domain | Character-level LM (GPT) | Pokemon battle prediction (behavior cloning) |
| Agent | Claude Code (single) | Claude Code (single) |
| Single metric | val_bpb (lower is better) | Top-1 accuracy (higher is better) |
| Files to edit | `train.py` only | `run_experiment.py`, `train_phase4.py`, `battle_transformer.py`, configs |
| Data prep | `prepare.py` (fixed) | `process_dataset.py` (fixed, run once) |
| Eval | `evaluate_bpb()` in prepare.py | `eval_harness.py` (fixed) |
| State | git branch advance/revert | git branch advance/revert + experiment registry JSON |

---

## 8. Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Agent gets stuck in a loop of bad ideas | Priority queue provides 18+ pre-planned experiments before agent-generated ones |
| Agent breaks the codebase | Git revert after every failed experiment; deny list on data pipeline files |
| Agent edits protected files | `.claude/settings.json` deny list enforces boundaries |
| RunPod pod terminated mid-run | Persistent volume preserves data; git commits preserve progress |
| Agent runs out of context window | Registry + notes provide persistent memory across context resets |
| Training OOM on large config | `run_experiment.py` catches OOM, logs crash, reverts |
| Agent stops autonomously | "NEVER STOP" directive in CLAUDE.md; no confirmation points in the loop |
