# Pokemon Battle Model - Repository Summary

## What This Is
A structured transformer-based AI agent for playing Gen 9 OU singles Pokemon battles on Pokemon Showdown. Uses behavior cloning on replay data with a hidden-information-aware architecture.

## Implementation Plan (8 Phases)
- **Phase 0** ✅: Scope, metrics, infrastructure (SCOPE.md, EVALUATION_SPEC.md)
- **Phase 1** ✅: Showdown server integration, BattleEnv, action space, legality, battle harness
- **Phase 2** ✅: Replay parser (Metamon dataset), observation constructor, tensorizer, 10K battles processed
- **Phase 3** ✅: Baselines (RandomBot, HeuristicBot, MLP/GRU BC, training pipeline)
- **Phase 4** ⏳: Main transformer model with hidden-info auxiliary head (CORE)
- **Phase 5-8** ⏳: Synthetic fine-tuning, evaluation harness, offline RL, enhancements

## Directory Layout
```
docs/                    → SCOPE.md, IMPLEMENTATION_PLAN.md, EVALUATION_SPEC.md, CHECKPOINT_CONVENTION.md
archive/                 → Historical planning documents
configs/                 → Hydra YAML configs (model/, training/, evaluation/, environment/)
src/
  environment/           → showdown_client.py, battle_env.py, action_space.py, legality.py, state.py, protocol.py
  data/                  → replay_parser.py, observation.py, tensorizer.py, dataset.py, priors.py
  models/                → baseline_mlp.py (MLP + GRU baselines)
  training/              → bc_trainer.py (behavior cloning trainer)
  bots/                  → base_bot.py, random_bot.py, max_damage_bot.py, heuristic_bot.py, model_bot.py
  evaluation/            → battle_evaluator.py
  synthetic/             → (stub, Phase 5)
tests/                   → 8 test files: test_baselines, test_legality, test_parser, test_observation, test_tensorizer, test_state, test_protocol, test_battle_harness
scripts/                 → download_replays.py, process_dataset.py, train_baseline.py, evaluate_baselines.py
data/processed/          → vocabs/, metadata.json, priors.json (battles/ is gitignored - too large)
checkpoints/             → (empty, for trained models)
```

## Key Architecture Decisions
- **Hidden Information Doctrine**: Never train on omniscient features. Opponent info uses explicit "unknown" markers. Metagame priors are soft hints only.
- **Action Space**: 13 canonical actions (4 moves + 4 tera-moves + 5 switches). Legal action masks applied at each turn.
- **Observation**: TurnObservation → tensorize_turn() → dict of numpy arrays. Per-pokemon features = 30 dims (9 categorical + 14 continuous + 7 binary). Total flat input = 384 dims.
- **Data**: Metamon dataset (jakegrigsby/metamon-parsed-replays), Gen 9 OU, 1500+ Elo. ParsedBattle → build_observations() → tensorize → .npz files.

## Tech Stack
Python 3.11+, PyTorch 2.2+, Hydra, W&B, websockets, numpy, pytest

## Dataset Stats
10,000 battles, 321,601 turns, 80/10/10 train/val/test split by battle ID

## Key Conventions
- All bots implement `Bot` ABC with `choose_action(observation, legal_actions) -> BattleAction`
- Models output logits over NUM_ACTIONS=13, masked by legal_mask before softmax
- BattleEvaluator runs bot-vs-bot games on local Showdown server
- Configs in configs/ are Hydra YAML (model/baseline_mlp.yaml, training/bc.yaml, etc.)
