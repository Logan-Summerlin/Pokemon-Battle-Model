# Pokemon Battle Model — Project Memory

## What This Is
A structured transformer-based AI agent for Gen 9 OU singles Pokemon battles on Pokemon Showdown.
Uses behavior cloning on human replay data with a hidden-information-aware architecture.

## Current Status (Phase 4 In Progress)
- **Phases 0–3 complete**: Scope/metrics, Showdown integration, replay parser, baselines (RandomBot, HeuristicBot, MLP/GRU BC)
- **Phase 4 active**: BattleTransformer model, compute/generalization experiments, P8-Lean optimization
- **Phases 5–8 not started**: Synthetic fine-tuning, evaluation harness expansion, offline RL, enhancements
- **Dataset**: 10K battles (321,601 turns) from Metamon (jakegrigsby/metamon-parsed-replays), Gen 9 OU, 1500+ Elo

## Model Variants (all use `scripts/train_phase4.py`)
| Variant | Config | Params | Wrapper |
|---------|--------|--------|---------|
| P8 | 4L/256d/4H, W20, aux+value | 3.6M | `train_p8_1k.py` |
| P8-Lean | 3L/224d/4H/FFN3x, W5, aux only | ~1.95M | `train_p8_lean.py` |
| P4 | 6L/384d/6H, W20, aux+value | 11.5M | `train_p4_25k.py` |

## Architecture
- **Input**: 14 tokens/turn (6 own pokemon + 6 opponent + field + context)
- **Per-pokemon features**: 30 dims (9 categorical + 14 continuous + 7 binary)
- **Action space**: 13 actions (4 moves + 4 tera-moves + 5 switches), legal mask before softmax
- **Heads**: Policy (candidate-action scoring), Auxiliary (item/speed/role/move-family prediction), optional Value
- **Hidden info doctrine**: Never leak omniscient features; use "unknown" markers; priors are soft hints only

## Directory Layout
```
docs/          → SCOPE.md, IMPLEMENTATION_PLAN.md, EVALUATION_SPEC.md, CHECKPOINT_CONVENTION.md,
                 PHASE4_ARCHITECTURE_AND_TRAINING.md (consolidated reference)
archive/       → Historical planning docs + superseded Phase 4 docs
configs/       → Hydra YAML (model/, training/, evaluation/, environment/)
src/
  environment/ → showdown_client.py, battle_env.py, action_space.py, legality.py, state.py, protocol.py
  data/        → replay_parser.py, observation.py, tensorizer.py, dataset.py, priors.py, auxiliary_labels.py
  models/      → baseline_mlp.py, battle_transformer.py
  training/    → bc_trainer.py, transformer_trainer.py
  bots/        → base_bot.py, random_bot.py, max_damage_bot.py, heuristic_bot.py, model_bot.py
  evaluation/  → battle_evaluator.py, offline_metrics.py
  synthetic/   → (stub, Phase 5)
scripts/       → train_phase4.py (core), train_p8_1k.py, train_p8_lean.py, train_p4_25k.py,
                 download_replays.py, process_dataset.py, evaluate_baselines.py, etc.
tests/         → 10 test files covering baselines, legality, parser, observation, tensorizer,
                 state, protocol, battle harness, transformer
data/processed/→ vocabs/, metadata.json, priors.json (battles/ gitignored)
checkpoints/   → baseline_mlp/, baseline_gru/, phase4_250/
```

## Key Conventions
- All bots implement `Bot` ABC: `choose_action(observation, legal_actions) -> BattleAction`
- Models output logits over NUM_ACTIONS=13, masked by legal_mask before softmax
- Training data: `.npz` tensors → `WindowedTurnDataset` → per-turn sliding window examples
- Split: 80/10/10 by battle ID (no leakage)
- Loss: masked cross-entropy (policy) + aux loss (0.2 weight)

## Known Data Issues
- `terastallized` flag always zero in processed data (dead feature)
- Field binary side conditions (16 dims) always zero (pipeline extraction issue)
- Aux labels: only item_targets populated; speed/role/tera/move-family are placeholder -1

## Tech Stack
Python 3.11+, PyTorch 2.2+, Hydra, W&B, websockets, numpy, pytest
