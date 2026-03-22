# AutoResearch Agent Coordination Guide

This document defines the operating model for Claude Code and Codex agents working on the Pokemon Battle Model AutoResearch pipeline.

---

## Agent Roles

### Claude Code — Research Lead

**Responsibilities:**
1. **Experiment planning**: Analyze the leaderboard, propose next experiments, rank hypotheses by expected value
2. **Log interpretation**: Read training reports, identify trends, diagnose convergence issues
3. **Code changes**: Implement model/data/loss modifications within the approved edit surface
4. **Research notes**: Write structured per-experiment analysis in `Autoresearch/notes/`
5. **Bug investigation**: Debug issues like the broken auxiliary speed/role heads
6. **Evaluation**: Run and interpret eval_harness.py results

**File ownership:**
- `Autoresearch/notes/` — experiment analysis
- `Autoresearch/experiment_registry.json` — experiment log
- `Autoresearch/eval_harness.py` — evaluation harness
- `src/models/battle_transformer.py` — model architecture
- `src/data/` — data pipeline (within approved edit surface)

### Codex — Implementation Support

**Responsibilities:**
1. **Config generation**: Create experiment YAML config files
2. **Script improvements**: Patch throughput issues, add profiling utilities, add CLI flags
3. **Boilerplate**: Experiment launch scripts, result aggregation, plotting
4. **Log parsing**: Extract metrics from training report JSONs
5. **Infrastructure**: DataLoader optimization, torch.compile integration, profiling

**File ownership:**
- `Autoresearch/configs/` — experiment configs
- `scripts/train_phase4.py` — training script patches
- `Autoresearch/run_experiment.py` — experiment launcher improvements

---

## Coordination Rules

### Rule 1: One Agent Per File

Never let both agents edit the same file simultaneously. The ownership table above defines who owns what. If you need to modify a file owned by the other agent, note the needed change in `Autoresearch/notes/` and let the appropriate agent handle it.

### Rule 2: Communication via Notes

Agents communicate through structured notes in `Autoresearch/notes/`. When you complete an experiment or find something the other agent needs to act on, write a note.

### Rule 3: Human in the Loop

The human decides experiment direction. The workflow is:
```
Human decides experiment direction
  → Claude Code designs experiment, writes hypothesis
    → Codex implements config/code changes (if needed)
      → Human reviews diff
        → Training runs
          → Claude Code analyzes results
            → Human decides next direction
```

### Rule 4: Budget Discipline

Never exceed tier budgets without human approval:
- **Tier 1**: 5–10 epochs, 30 min max
- **Tier 2**: 20–30 epochs, 2 hours max
- **Tier 3**: 30–50 epochs, 4 hours max

### Rule 5: Registry Before Running

Every experiment MUST be registered in `experiment_registry.json` BEFORE launching. No untracked experiments.

---

## Experiment Note Template

Every experiment note in `Autoresearch/notes/` must include:

```markdown
# Experiment [ID]: [Name]

## Hypothesis
[What we expected and why]

## Change Made
[Exact config/code diff from parent experiment]

## Results
| Metric | Parent | This | Delta |
|--------|--------|------|-------|
| Top-1 accuracy | X% | Y% | +/-Z% |
| Top-3 accuracy | X% | Y% | +/-Z% |
| Policy NLL | X | Y | +/-Z |
| Move accuracy | X% | Y% | +/-Z% |
| Switch accuracy | X% | Y% | +/-Z% |
| Throughput (ex/sec) | X | Y | +/-Z |
| Wall time (min) | X | Y | +/-Z |

## Analysis
[Why did this work or not? What did we learn?]

## Decision
[ ] KILL — no improvement, discard
[ ] RETRY — promising but needs adjustment
[ ] PROMOTE — confirmed improvement, adopt as new champion
```

---

## Restrictions for All Agents

### NEVER:
- Modify `src/data/observation.py`, `src/data/tensorizer.py`, or `src/data/replay_parser.py` without explicit human approval
- Delete or weaken existing tests
- Modify split manifests in `data/splits/`
- Train on omniscient features (violates Hidden Information Doctrine)
- Run experiments without registering them first
- Exceed tier budget without approval
- Use zeros to represent unknown information (use "unknown" markers)
- Push to branches other than the designated development branch

### ALWAYS:
- Run `pytest` after modifying model or data code
- Record every experiment in the registry
- Write a note for every experiment, even failed ones
- Compare results to the parent experiment, not just the anchor
- Report move accuracy and switch accuracy separately (not just overall Top-1)
- Use legal mask — the model must never predict illegal actions
- Commit work regularly with descriptive messages

---

## Priority Experiment Queue

Based on analysis of current weaknesses, run experiments in this order:

### Phase 1: Fix & Profile (AR-0, AR-1, AR-2)
1. Register anchor checkpoint
2. Debug auxiliary speed/role heads (0% accuracy)
3. Batch size sweep on A40: 64 → 128 → 256 → 512 → 1024
4. torch.compile() test
5. DataLoader worker count sweep

### Phase 2: Data & Representation (AR-3)
6. **Window size sweep: 2 → 3 → 5 → 8** (highest-priority experiment — optimal likely between 2 and 8 turns)
7. Full dataset training with best window size
8. Class-weighted loss for switches (address 37–48% vs 72–75% gap)
9. Elo-weighted loss (upweight >1500 Elo)
10. Label smoothing (0.05, 0.1)

### Phase 3: Architecture (AR-4)
11. Scale to P8 (4L/256d/4H) with best data config
12. Scale to P4 (6L/384d/6H) if P8 helps
13. Dropout sweep
14. Hierarchical action head (move-vs-switch then which)

### Phase 4: Consolidation (AR-5)
15. Combine best config, 3-seed confirmation
16. Full test evaluation
17. Write final report
