# Pokemon Battle Model

A structured transformer model for Pokemon battle decision-making under partial observability.

## Project Overview

This project builds an AI agent that plays Gen 3 OU (ADV) singles on Pokemon Showdown. The model learns from human replay data via behavior cloning, using a candidate-action-scoring transformer architecture with an auxiliary hidden-information prediction head.

**Key design constraints:**
- The agent only ever sees what a real player would see. Hidden information (opponent items, abilities, EVs, unrevealed moves, unrevealed team members) is never leaked into the observation space.
- Gen 3 has **no team preview** — the opponent's team is entirely unknown at battle start, making hidden-information prediction central to the architecture.
- **9-action space** — 4 moves + 5 switches (no Terastallization in Gen 3).

## Current Status

- **Phases 0–3**: Complete (scope, Showdown integration, data pipeline, baselines)
- **Phase 4**: In progress (BattleTransformer model, compute experiments, P8-Lean optimization)
- **Phases 5–8**: Not started (synthetic fine-tuning, evaluation harness, offline RL, enhancements)

## Architecture

- **14 tokens/turn**: 6 own-team + 6 opponent-team + field + context
- **Per-pokemon features**: 28 dims (9 categorical + 14 continuous + 5 binary)
- **Field**: 19 dims | **Context**: 7 dims
- **Policy head**: candidate-action scoring with legal mask
- **Auxiliary head**: predicts opponent hidden info (item, speed tier, role, move families)
- **Optional value head**: predicts win probability

### Model Variants

| Variant | Config | Params | Wrapper Script |
|---------|--------|--------|----------------|
| P8 | 4L/256d/4H, W20, aux+value | 3.6M | `train_p8_1k.py` |
| P8-Lean | 3L/224d/4H/FFN3x, W5, aux only | ~1.95M | `train_p8_lean.py` |
| P4 | 6L/384d/6H, W20, aux+value | 11.5M | `train_p4_25k.py` |

## Project Structure

```
├── docs/                       # Active project documentation
│   ├── SCOPE.md                # Frozen scope decisions (Gen 3 OU)
│   ├── EVALUATION_SPEC.md      # Success metrics and evaluation protocol
│   ├── IMPLEMENTATION_PLAN.md  # Full 8-phase build order
│   ├── POKEMON_MODEL_PIPELINE_PLAN.md  # 4-stage training pipeline
│   ├── CHECKPOINT_CONVENTION.md# Model checkpoint naming scheme
│   └── COMPETITIVE_BATTLE_STRATEGY_GUIDE.md  # Gen 3 OU strategy reference
├── archive/                    # Historical and superseded documents
├── configs/                    # Hydra configuration files
│   ├── model/                  # Model architecture configs
│   ├── training/               # Training hyperparameters
│   ├── evaluation/             # Evaluation settings
│   └── environment/            # Showdown server config
├── src/                        # Source code
│   ├── environment/            # Showdown interface and battle env
│   ├── data/                   # Replay parsing and tensorization
│   ├── models/                 # Model definitions (baselines + transformer)
│   ├── training/               # Training loops
│   ├── bots/                   # Bot implementations
│   ├── evaluation/             # Evaluation harness
│   └── synthetic/              # Synthetic scenario factory (stub)
├── tests/                      # Unit and integration tests
├── scripts/                    # Training, evaluation, data scripts
├── data/                       # Processed tensors, vocabs, metadata
└── checkpoints/                # Model checkpoints
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

## Training

```bash
# P8 on 1K battles
python scripts/train_p8_1k.py --num-battles 1000 --seeds 42 --batch-size 32 --epochs 30

# P8-Lean on 10K battles
python scripts/train_p8_lean.py --num-battles 10000 --seeds 42 43 44 --batch-size 64

# Direct trainer with full control
python scripts/train_phase4.py --mode full --num-battles 10000 --num-layers 4 --hidden-dim 256 --num-heads 4 --max-window 20 --aux-weight 0.2 --seed 42
```

## Documentation

| Document | Purpose |
|----------|---------|
| [SCOPE.md](docs/SCOPE.md) | Frozen scope: format, info regime, data, architecture |
| [IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md) | Full 8-phase roadmap with exit gates |
| [EVALUATION_SPEC.md](docs/EVALUATION_SPEC.md) | Metrics, thresholds, evaluation protocol (Gen 3 calibrated) |
| [POKEMON_MODEL_PIPELINE_PLAN.md](docs/POKEMON_MODEL_PIPELINE_PLAN.md) | 4-stage training pipeline (BC → SFT → RL → repair) |
| [CHECKPOINT_CONVENTION.md](docs/CHECKPOINT_CONVENTION.md) | Checkpoint naming scheme and model variants |
| [COMPETITIVE_BATTLE_STRATEGY_GUIDE.md](docs/COMPETITIVE_BATTLE_STRATEGY_GUIDE.md) | Gen 3 OU competitive strategy reference |
