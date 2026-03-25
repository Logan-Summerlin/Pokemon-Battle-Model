# Plan: Autonomous Dual-Agent AutoResearch Operation

## Goal

Transform the Pokemon Battle AutoResearch system into a fully autonomous dual-agent research loop. Claude Code and Codex each run continuously in their respective roles, coordinating through a shared file-based communication protocol. Both agents follow the "NEVER STOP" directive independently, and neither requires human routing between them.

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
| **Two agents need coordination** | Claude Code and Codex have different roles; coordination currently requires human routing | HIGH |
| **No inter-agent communication** | Agents can't request work from each other or share findings | HIGH |
| **No automatic revert** | Failed experiments aren't automatically reverted via git | HIGH |
| **No crash recovery protocol** | If training crashes, the loop stalls | HIGH |
| **Config + code separation** | Experiments defined in YAML configs AND code changes | MEDIUM |

---

## 3. The Dual-Agent Autonomy Plan

### Phase A: File-Based Inter-Agent Communication Protocol

The core problem with two autonomous agents is coordination. Human routing is eliminated by introducing a **shared message queue** — a set of files both agents read and write.

**Directory structure:**

```
Autoresearch/comms/
├── task_queue.json          # Shared task queue (both agents read/write)
├── inbox_claude_code/       # Messages TO Claude Code FROM Codex
│   └── msg_001.json
├── inbox_codex/             # Messages TO Codex FROM Claude Code
│   └── msg_001.json
└── completed/               # Processed messages (audit trail)
    └── msg_001.json
```

**Message schema:**

```json
{
  "id": "msg_001",
  "from": "claude_code",
  "to": "codex",
  "timestamp": "2026-03-25T10:30:00Z",
  "type": "task_request",
  "priority": "high",
  "subject": "Generate config for window_size_10 experiment",
  "body": "Create Autoresearch/configs/ar3_02a_window10.yaml with max_window=10, all other params matching anchor.yaml",
  "context": {
    "parent_experiment": "AR0-anchor",
    "experiment_id": "AR3-02a",
    "hypothesis": "Increasing window from 2 to 10 will improve switch prediction"
  },
  "status": "pending"
}
```

**Message types:**

| Type | From | To | Purpose |
|------|------|----|---------|
| `task_request` | Claude Code | Codex | Ask Codex to generate configs, fix scripts, parse logs |
| `task_complete` | Codex | Claude Code | Report that a requested task is done, with results |
| `finding` | Either | Either | Share an observation (e.g., "OOM at batch 1024", "aux head still 0%") |
| `code_change_needed` | Codex | Claude Code | Codex identified a change needed in a file it can't edit |
| `experiment_result` | Claude Code | Codex | Share results so Codex can adjust future configs |
| `status_update` | Either | Either | "I'm running experiment X", "I'm idle and waiting for tasks" |

**Agent loop integration:**

Both agents add a communication check to the start of every loop iteration:

```
BEFORE each experiment iteration:
1. CHECK inbox for new messages
2. PROCESS any pending messages (fulfill task requests, read findings)
3. MOVE processed messages to completed/
4. PROCEED with experiment loop
```

This is non-blocking — if the inbox is empty, the agent proceeds immediately.

### Phase B: Defined Agent Roles for Autonomous Operation

**Claude Code** is the **research lead**:
1. Reads the leaderboard, selects the next experiment hypothesis
2. Sends task requests to Codex for config generation and script work
3. Makes model/data/loss code changes (files Codex can't edit)
4. Runs experiments via `run_experiment.py`
5. Analyzes results, writes experiment notes
6. Decides keep/discard based on the promotion rules
7. Updates the leaderboard and registry
8. Reads Codex's inbox for `code_change_needed` requests and fulfills them

**Codex** is the **research engineer**:
1. Reads Claude Code's task requests from its inbox
2. Generates experiment configs in `Autoresearch/configs/`
3. Improves `Autoresearch/run_experiment.py` (its primary edit target)
4. Parses training logs, extracts metrics
5. Sends `code_change_needed` messages when it identifies issues in read-only files
6. Sends `finding` messages when it discovers useful patterns in results
7. When idle (no pending tasks), proactively reviews recent experiment logs and sends analysis findings

**Key principle:** Neither agent blocks on the other. If Claude Code needs a config and Codex hasn't produced it yet, Claude Code generates a minimal inline config and continues. If Codex has no pending tasks, it reviews logs and sends findings rather than stopping.

### Phase C: Implement Automatic Git-Based State Management

Add the Karpathy-style branch/revert protocol to both agents' instructions:

```markdown
## Git Protocol (Mandatory for Both Agents)

Before EVERY experiment:
1. `git add -A && git commit -m "EXP-{id}: {description}"`

After EVERY experiment:
- If the experiment meets the promotion rules: KEEP the commit (branch advances)
- If accuracy is equal or worse: `git reset --hard HEAD~1` (revert to previous state)
- If crashed: `git reset --hard HEAD~1`, log as crash, move on

The branch always represents the current best configuration.

Git conflict resolution:
- Both agents work on the SAME branch
- Before committing, always `git pull --rebase` first
- If rebase conflicts occur, prefer the version with better metrics
- Communication files (comms/) should never conflict (unique message IDs)
```

### Phase D: Crash Recovery

Add explicit crash handling to the experiment loop:

```markdown
## Crash Handling

If `run_experiment.py` crashes:
1. Read the last 50 lines of the training log
2. If it's a simple fix (typo, import error, OOM): fix and retry ONCE
3. If it crashes again or the idea is fundamentally broken:
   - Log as "crash" in the registry
   - `git reset --hard HEAD~1`
   - Send a `finding` message to the other agent describing the crash
   - Move on to the next hypothesis
4. NEVER spend more than 10 minutes debugging a single crash
```

### Phase E: Autonomous Idea Generation

Build an idea queue into CLAUDE.md that Claude Code works through:

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

## 4. Communication Protocol in Detail

### Startup Sequence

When both agents are launched:

```
1. Claude Code starts first:
   - Reads CLAUDE.md and program.md
   - Verifies data and anchor checkpoint
   - Creates experiment branch: git checkout -b autoresearch/<date>
   - Creates Autoresearch/comms/ directory structure
   - Writes initial status_update to Codex inbox: "Session started, beginning with Priority 1 experiments"
   - Begins experiment loop

2. Codex starts (can be simultaneous or shortly after):
   - Reads AGENTS.md
   - Checks its inbox for messages
   - If task_request pending: fulfill it
   - If no tasks: review experiment_registry.json and recent notes, send findings
   - Enters its own loop: check inbox -> work -> check inbox -> work
```

### Communication Cadence

| Event | Claude Code Action | Codex Action |
|-------|-------------------|--------------|
| New experiment planned | Sends `task_request` to Codex for config | - |
| Config needed | Proceeds with inline config if Codex hasn't responded | Generates config when it sees the request |
| Experiment completes | Sends `experiment_result` to Codex | Reads result, updates analysis, may send `finding` |
| Code change needed in read-only file | - | Sends `code_change_needed` to Claude Code |
| Bug discovered | Sends `finding` to Codex | Sends `finding` to Claude Code |
| Agent idle | Generates new hypotheses from results | Reviews logs and sends analysis findings |

### Conflict Avoidance

The agents have **non-overlapping edit surfaces** by design:

| Files | Claude Code | Codex |
|-------|------------|-------|
| `src/models/battle_transformer.py` | Read/Write | Read only |
| `src/data/dataset.py` | Read/Write | Read only |
| `scripts/train_phase4.py` | Read/Write | Read only |
| `Autoresearch/run_experiment.py` | Read only | Read/Write |
| `Autoresearch/configs/` | Read only | Read/Write |
| `Autoresearch/comms/inbox_codex/` | Write | Read |
| `Autoresearch/comms/inbox_claude_code/` | Read | Write |
| `Autoresearch/notes/` | Read/Write | Read only |
| `Autoresearch/experiment_registry.json` | Read/Write | Read only |

This means git conflicts should be extremely rare. The only shared-write location is `Autoresearch/comms/completed/`, which uses unique message IDs and is append-only.

---

## 5. Implementation Steps

| Step | Action | Time |
|------|--------|------|
| 1 | Fix repo issues (pyproject.toml, vocabs, directories) per AUTORESEARCH_REPO_FIXES.md | 30 min |
| 2 | Create `Autoresearch/comms/` directory structure with inbox directories | 5 min |
| 3 | Add communication protocol instructions to `Autoresearch/CLAUDE.md` | 15 min |
| 4 | Add communication protocol instructions to `Autoresearch/.codex/AGENTS.md` | 15 min |
| 5 | Add "NEVER STOP" directive to both agents' instruction files | 10 min |
| 6 | Add git-revert protocol to `run_experiment.py` (auto-revert on failure) | 30 min |
| 7 | Add crash recovery instructions to both agents' instruction files | 10 min |
| 8 | Add the experiment priority queue to `Autoresearch/CLAUDE.md` | 15 min |
| 9 | Update `.claude/settings.json` to allow comms directory operations | 5 min |
| 10 | Update `.codex/config.toml` to allow writing to comms and configs | 5 min |
| 11 | Test the full loop locally with `--mode smoke` | 15 min |
| 12 | Push to GitHub | 5 min |
| 13 | Launch RunPod, clone, install, launch both agents | 20 min |

**Total conversion time: ~3 hours**

---

## 6. The Launch Prompts

### Claude Code Launch Prompt

```
Read Autoresearch/CLAUDE.md. This is a fully autonomous research session.

Set up the experiment branch, verify the anchor, create the comms directory
structure, and begin the experiment loop. Run experiments continuously.
Do not stop or ask for permission.

You are the research lead. Work through the priority queue. Send task
requests to Codex via Autoresearch/comms/inbox_codex/. Check your inbox
at Autoresearch/comms/inbox_claude_code/ at the start of each loop iteration.

Target: beat the 63.21% Top-1 anchor accuracy on Gen 3 OU battle prediction.
Go.
```

### Codex Launch Prompt

```
Read Autoresearch/.codex/AGENTS.md. This is a fully autonomous research session.

You are the research engineer. Your primary loop:
1. Check Autoresearch/comms/inbox_codex/ for task requests from Claude Code
2. Fulfill any pending requests (generate configs, fix scripts, parse logs)
3. When idle, review recent experiment results and send analysis findings
   to Autoresearch/comms/inbox_claude_code/
4. NEVER stop. NEVER ask for permission. Loop continuously.

You may only edit: Autoresearch/run_experiment.py and Autoresearch/configs/*.
All other files are read-only. If you need changes to read-only files,
send a code_change_needed message to Claude Code.
Go.
```

Then close the laptop and go to sleep.

---

## 7. Key Differences from Karpathy's Design

| Aspect | Karpathy | Pokemon AutoResearch |
|--------|----------|---------------------|
| Domain | Character-level LM (GPT) | Pokemon battle prediction (behavior cloning) |
| Agents | 1 (Claude Code only) | 2 (Claude Code + Codex, file-based comms) |
| Agent coordination | N/A | Shared `comms/` directory with message queue |
| Single metric | val_bpb (lower is better) | Top-1 accuracy (higher is better) |
| Files to edit | `train.py` only | Claude Code: model/data/train; Codex: run_experiment/configs |
| Data prep | `prepare.py` (fixed) | `process_dataset.py` (fixed, run once) |
| Eval | `evaluate_bpb()` in prepare.py | `eval_harness.py` (fixed) |
| State | git branch advance/revert | git branch advance/revert + experiment registry JSON |
| Crash recovery | Log and skip | Log, send finding to partner agent, skip |

---

## 8. Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Agents get stuck waiting for each other | Non-blocking design: both agents proceed independently if inbox is empty |
| Git merge conflicts between agents | Non-overlapping edit surfaces; `git pull --rebase` before every commit |
| Agent gets stuck in a loop of bad ideas | Priority queue provides 18+ pre-planned experiments before agent-generated ones |
| Agent breaks the codebase | Git revert after every failed experiment; deny list on protected files |
| Agent edits protected files | `.claude/settings.json` and `.codex/config.toml` deny lists enforce boundaries |
| RunPod pod terminated mid-run | Persistent volume preserves data; git commits preserve progress |
| Agent runs out of context window | Registry + notes + comms provide persistent memory across context resets |
| Training OOM on large config | `run_experiment.py` catches OOM, logs crash, reverts |
| Agent stops autonomously | "NEVER STOP" directive in both agents' instructions; no confirmation points in the loop |
| Message queue grows unbounded | Processed messages moved to `completed/`; agents only read `pending` status |
| One agent crashes permanently | The other continues independently — Claude Code can generate its own configs; Codex can still review and analyze |

---

## 9. Graceful Degradation

The dual-agent design is resilient because neither agent is strictly dependent on the other:

| Scenario | What Happens |
|----------|-------------|
| Codex dies, Claude Code survives | Claude Code generates inline configs, runs full loop solo. Slightly slower but fully functional. |
| Claude Code dies, Codex survives | Codex enters review-only mode: analyzes existing results, sends findings to inbox (read when Claude Code restarts). |
| Both agents running | Optimal: Claude Code focuses on research decisions and code changes, Codex handles configs and log analysis. |
| Communication directory missing | Both agents detect this and create it. No crash. |
| Stale messages in inbox | Messages include timestamps. Agents ignore messages older than 1 hour. |

This means the system is **at least as good as single-agent** and **better when both agents are active**.
