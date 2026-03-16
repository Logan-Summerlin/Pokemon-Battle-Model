# Pokemon Battle Model — Project Memory

## What This Is
A structured transformer-based AI agent for Gen 3 OU (ADV) singles Pokemon battles on Pokemon Showdown.
Uses behavior cloning on human replay data with a hidden-information-aware architecture.

## Current Status (Phase 4 In Progress)
- **Phases 0–3 complete**: Scope/metrics, Showdown integration, replay parser, baselines (RandomBot, HeuristicBot, MLP/GRU BC)
- **Phase 4 active**: BattleTransformer model, compute/generalization experiments, P8-Lean optimization
- **Phases 5–8 not started**: Synthetic fine-tuning, evaluation harness expansion, offline RL, enhancements
- **Dataset**: 100K battles from Metamon (jakegrigsby/metamon-parsed-replays), Gen 3 OU, stratified Elo sampling (all 1500+, equal bins 1000-1500)
- **Migration**: Fully migrated from Gen 9 OU to Gen 3 OU (March 2026)

## Model Variants (all use `scripts/train_phase4.py`)
| Variant | Config | Params | Wrapper |
|---------|--------|--------|---------|
| P8 | 4L/256d/4H, W20, aux+value | 3.6M | `train_p8_1k.py` |
| P8-Lean | 3L/224d/4H/FFN3x, W5, aux only | ~1.95M | `train_p8_lean.py` |
| P4 | 6L/384d/6H, W20, aux+value | 11.5M | `train_p4_25k.py` |

## Architecture
- **Input**: 14 tokens/turn (6 own pokemon + 6 opponent + field + context)
- **Per-pokemon features**: 28 dims (9 categorical + 14 continuous + 5 binary)
- **Field features**: 19 dims | **Context features**: 7 dims
- **Action space**: 9 actions (4 moves + 5 switches), legal mask before softmax (no Terastallization in Gen 3)
- **Heads**: Policy (candidate-action scoring), Auxiliary (item/speed/role/move-family prediction), optional Value
- **Hidden info doctrine**: Never leak omniscient features; use "unknown" markers; priors are soft hints only
- **No team preview**: Gen 3 has no team preview — opponent team entirely unknown at battle start

## Gen 3 OU Key Mechanics
- **No Terastallization** (Gen 9 mechanic)
- **No team preview** — opponent team unknown at battle start
- **Type-based physical/special split** — move category determined by type
- **Permanent weather** — Sand Stream lasts indefinitely
- **Spikes only** — no Stealth Rock, Toxic Spikes, or Sticky Web
- **No pivot moves** — no U-turn, Volt Switch, Flip Turn
- **No Choice Scarf/Specs** — only Choice Band exists

## Directory Layout
```
docs/          → SCOPE.md, IMPLEMENTATION_PLAN.md, EVALUATION_SPEC.md, CHECKPOINT_CONVENTION.md,
                 POKEMON_MODEL_PIPELINE_PLAN.md, COMPETITIVE_BATTLE_STRATEGY_GUIDE.md
archive/       → Historical planning docs + superseded Phase 4 docs + Gen 9 migration docs
configs/       → Hydra YAML (model/, training/, evaluation/, environment/)
src/
  environment/ → showdown_client.py, battle_env.py, action_space.py, legality.py, state.py, protocol.py
  data/        → replay_parser.py, observation.py, tensorizer.py, dataset.py, priors.py, auxiliary_labels.py, base_stats.py
  models/      → baseline_mlp.py, battle_transformer.py
  training/    → bc_trainer.py, transformer_trainer.py
  bots/        → base_bot.py, random_bot.py, max_damage_bot.py, heuristic_bot.py, model_bot.py
  evaluation/  → battle_evaluator.py, offline_metrics.py
  synthetic/   → (stub, Phase 5)
scripts/       → train_phase4.py (core), train_p8_1k.py, train_p8_lean.py, train_p4_25k.py,
                 download_replays.py, download_replays_stratified.py, process_dataset.py, evaluate_baselines.py, etc.
tests/         → test files covering baselines, legality, parser, observation, tensorizer,
                 state, protocol, battle harness, transformer, Gen 3 mechanics
data/processed/→ vocabs/, metadata.json, priors.json (battles/ gitignored)
checkpoints/   → baseline_mlp/, baseline_gru/, phase4_250/
```

## Key Conventions
- All bots implement `Bot` ABC: `choose_action(observation, legal_actions) -> BattleAction`
- Models output logits over NUM_ACTIONS=9, masked by legal_mask before softmax
- Training data: `.npz` tensors → `WindowedTurnDataset` → per-turn sliding window examples
- Split: 80/10/10 by battle ID (no leakage)
- Loss: masked cross-entropy (policy) + aux loss (0.2 weight)

## Tech Stack
Python 3.11+, PyTorch 2.2+, Hydra, W&B, websockets, numpy, pytest
