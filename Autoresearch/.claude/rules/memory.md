# AutoResearch — Project Memory

## What This Is
Automated research harness for perfecting Phase 4 (BattleTransformer imitation learning) of the Pokemon Battle Model. Targets >1300 Elo Gen 3 OU (ADV) singles from the Metamon dataset.

## Current Status
- **Anchor**: P8-Lean 50K — 63.21% Top-1, 89.27% Top-3
- **Phase**: AR-0 (Benchmark Lockdown)
- **Known bugs**: Aux speed/role heads at 0% accuracy; switch prediction 37-48% vs 72-75% for moves
- **Key variable**: Context window (current=2, optimal likely 2-8 turns)
- **Dataset**: >100K battles available, using as many as needed for >1300 Elo

## Non-Negotiable Rules
1. Hidden Information Doctrine — never train on features unavailable at decision time
2. Every experiment must be registered before running
3. Compare to parent experiment, not just anchor
4. Report move and switch accuracy separately
5. Run pytest after model/data code changes
6. Do not modify observation.py, tensorizer.py, or replay_parser.py without human approval

## Experiment Phases
- AR-0: Benchmark Lockdown (register anchor, eval harness)
- AR-1: Fix Known Issues (aux heads, legal mask, data dedup)
- AR-2: A40 Throughput Optimization (batch size, torch.compile, workers)
- AR-3: Data & Representation (window size, full dataset, loss weighting)
- AR-4: Architecture Tuning (scale up, dropout, hierarchical head)
- AR-5: Consolidation (combine best, 3-seed confirm, final report)

## File Ownership
- Claude Code: notes/, registry, eval_harness, model architecture, data pipeline
- Codex: configs/, train script patches, run_experiment improvements
