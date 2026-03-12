# Pokemon Battle Model

A structured transformer model for Pokemon battle decision-making under partial observability.

## Project Overview

This project builds an AI agent that plays Gen 9 OU singles on Pokemon Showdown. The model learns from human replay data via behavior cloning, using a candidate-action-scoring transformer architecture with an auxiliary hidden-information prediction head.

**Key design constraint:** The agent only ever sees what a real player would see. Hidden information (opponent items, abilities, EVs, unrevealed moves) is never leaked into the observation space.

## Current Status

- **Phases 0–3**: Complete (scope, Showdown integration, data pipeline, baselines)
- **Phase 4**: In progress (BattleTransformer model, compute experiments, P8-Lean optimization)
- **Phases 5–8**: Not started (synthetic fine-tuning, evaluation harness, offline RL, enhancements)

## Project Structure

```
├── docs/                       # Active project documentation
│   ├── SCOPE.md                # Frozen scope decisions
│   ├── EVALUATION_SPEC.md      # Success metrics and evaluation protocol
│   ├── IMPLEMENTATION_PLAN.md  # Full 8-phase build order
│   ├── CHECKPOINT_CONVENTION.md# Model checkpoint naming scheme
│   └── PHASE4_ARCHITECTURE_AND_TRAINING.md  # Consolidated Phase 4 reference
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

# See docs/PHASE4_ARCHITECTURE_AND_TRAINING.md for full training reference
```

## Documentation

| Document | Purpose |
|----------|---------|
| [SCOPE.md](docs/SCOPE.md) | Frozen scope: format, info regime, data, architecture |
| [IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md) | Full 8-phase roadmap with exit gates |
| [EVALUATION_SPEC.md](docs/EVALUATION_SPEC.md) | Metrics, thresholds, evaluation protocol |
| [PHASE4_ARCHITECTURE_AND_TRAINING.md](docs/PHASE4_ARCHITECTURE_AND_TRAINING.md) | Model variants, training commands, optimization |
| [CHECKPOINT_CONVENTION.md](docs/CHECKPOINT_CONVENTION.md) | Checkpoint naming scheme |
