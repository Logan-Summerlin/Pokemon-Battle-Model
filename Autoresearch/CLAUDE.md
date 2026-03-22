# Pokemon Battle Model — AutoResearch

## What This Is

An automated research harness for perfecting Phase 4 (BattleTransformer imitation learning) of the Pokemon Battle Model project. The goal is to maximize action prediction accuracy on **>1300 Elo Gen 3 OU (ADV) singles** replay data from the Metamon dataset.

## Objective

**Benchmark:** Predict the moves and switches of >1300 Elo Gen 3 OU battles.

**Current anchor:** P8-Lean 50K — 63.21% Top-1 accuracy, 89.27% Top-3 accuracy.

**Target:** Maximize Top-1 and Top-3 action prediction accuracy through systematic experimentation with data, architecture, loss functions, and training hyperparameters.

## Architecture

- **BattleTransformer** (`src/models/battle_transformer.py`): 14 tokens/turn (6 own + 6 opponent + field + context), transformer encoder, candidate-action-scoring policy head
- **Auxiliary head**: predicts opponent hidden info (items, speed tier, role, move families) from encoder output
- **Action space**: 9 canonical actions (4 moves + 5 switches), legal mask applied before softmax
- **Observation**: per-pokemon 28 dims (9 categorical + 14 continuous + 5 binary), field 19 dims, context 7 dims
- **No team preview**: Gen 3 has no team preview — opponent team entirely unknown at battle start

## Dataset

- **Source**: Metamon (jakegrigsby/metamon-parsed-replays on HuggingFace)
- **Format**: Gen 3 OU (ADV) singles
- **Volume**: Full dataset is >100,000 battles. Use as many as necessary.
- **Elo filter**: >1300 Elo. Higher-Elo data is more informative for imitation learning.
- **Splits**: 80/10/10 by battle ID (no turn-level leakage)
- **Context window**: Current is 2 turns. Optimal is likely 2–8 turns — this is a key experimental variable.

## Hidden Information Doctrine (Non-Negotiable)

1. **Never** train on omniscient features unavailable at decision time
2. Represent uncertainty explicitly with "unknown" markers, not zeros
3. Metagame priors are soft hints, not leaked truth
4. Separate hidden-state inference (auxiliary head) from move selection (policy head)
5. If you are unsure whether a feature leaks hidden info, **do not use it**

## Key Files

### Core Source (from parent Pokemon-Battle-Model repo)
- `src/models/battle_transformer.py` — main model
- `src/data/observation.py` — observation construction (Hidden Info Doctrine enforced here)
- `src/data/tensorizer.py` — tensorization pipeline
- `src/data/dataset.py` — windowed dataset
- `src/data/auxiliary_labels.py` — auxiliary label construction
- `scripts/train_phase4.py` — training entry point
- `scripts/train_p8_lean.py` — P8-Lean wrapper script

### AutoResearch Harness
- `Autoresearch/eval_harness.py` — unified evaluation command
- `Autoresearch/run_experiment.py` — bounded experiment launcher
- `Autoresearch/leaderboard.py` — experiment ranking and comparison
- `Autoresearch/experiment_registry.json` — machine-readable experiment log
- `Autoresearch/configs/` — experiment YAML configs
- `Autoresearch/notes/` — per-experiment analysis notes
- `Autoresearch/results/` — evaluation output JSONs

## Approved Edit Surface

You **MAY** freely modify:
- `Autoresearch/` — all files (configs, notes, results, harness scripts)
- `configs/` — YAML configuration files
- `scripts/train_phase4.py` — training loop
- `src/models/battle_transformer.py` — model architecture
- `src/data/dataset.py` — data loading and windowed dataset
- `src/data/auxiliary_labels.py` — auxiliary label construction

You **MUST NOT** modify without explicit human approval:
- `src/data/observation.py` — observation construction (data integrity)
- `src/data/tensorizer.py` — tensorization (data integrity)
- `src/data/replay_parser.py` — replay parsing (data integrity)
- `data/splits/` — split manifests (evaluation integrity)
- `docs/EVALUATION_SPEC.md` — benchmark definition
- `tests/` — do not delete or weaken tests (adding new tests is fine)

## Experiment Loop Protocol

Every experiment must follow this cycle:

```
1. READ       → Load leaderboard, identify current champion
2. HYPOTHESIZE → Select one hypothesis to test, with expected effect
3. PLAN       → Specify exact config change, tier, budget
4. IMPLEMENT  → Modify only approved files
5. RUN        → Launch via run_experiment.py with tier budget
6. EVALUATE   → Compare to parent on eval_harness.py
7. RECORD     → Write experiment note with metric deltas
8. UPDATE     → Update leaderboard and registry
```

## Tier Budgets

| Tier | Max Epochs | Max Wall Time | Purpose |
|------|-----------|---------------|---------|
| 1 | 5–10 | 30 min | Quick triage — kill bad ideas fast |
| 2 | 20–30 | 2 hours | Confirmation — validate promising results |
| 3 | 30–50 | 4 hours | Full promotion — final training with multi-seed |

## Promotion Rules

A candidate replaces the champion ONLY if:
- **Accuracy gate:** Top-1 accuracy improves by >=0.5 percentage points, OR
- **Speed gate:** Matches accuracy within 0.3 points while reducing time-to-baseline by >=20%
- **Stability gate:** For Tier 3, must be confirmed on >=2 seeds with std < 1.0 point

## Known Issues to Fix First

1. **Auxiliary speed/role heads show 0% accuracy** — this is a bug, not a tuning issue. Debug `src/data/auxiliary_labels.py` and `compute_total_loss()` before running aux-head experiments.
2. **Switch prediction is 37–48% vs 72–75% for moves** — closing this gap is the highest-value lever for overall accuracy.
3. **Calibration shows systematic overconfidence in the 0.4–0.8 range**.

## Testing

```bash
pytest                              # all tests
pytest tests/test_transformer.py    # transformer-specific
```

Always run tests after modifying model or data code. Never commit code that breaks existing tests.

## Model Variants

| Variant | Config | Params | Script |
|---------|--------|--------|--------|
| P8 | 4L/256d/4H, W20, aux+value | 3.6M | `train_p8_1k.py` |
| P8-Lean | 3L/224d/4H/FFN3x, W2, aux only | ~1.95M | `train_p8_lean.py` |
| P4 | 6L/384d/6H, W20, aux+value | 11.5M | `train_p4_25k.py` |

## Gen 3 OU Key Mechanics

- No Terastallization (Gen 9 mechanic)
- No team preview — opponent team unknown at battle start
- Type-based physical/special split — move category determined by type
- Permanent weather — Sand Stream lasts indefinitely
- Spikes only — no Stealth Rock, Toxic Spikes, or Sticky Web
- No pivot moves — no U-turn, Volt Switch, Flip Turn
- No Choice Scarf/Specs — only Choice Band exists

## Tech Stack

Python 3.11+, PyTorch 2.4+, NumPy, Hydra, W&B, pytest
