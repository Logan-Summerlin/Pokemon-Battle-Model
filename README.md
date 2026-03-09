# Pokemon Battle Model

A structured transformer model for Pokemon battle decision-making under partial observability.

## Project Overview

This project builds an AI agent that plays Gen 9 OU singles on Pokemon Showdown. The model learns from human replay data via behavior cloning, using a candidate-action-scoring transformer architecture with an auxiliary hidden-information prediction head.

**Key design constraint:** The agent only ever sees what a real player would see. Hidden information (opponent items, abilities, EVs, unrevealed moves) is never leaked into the observation space.

## Project Structure

```
├── SCOPE.md                    # Frozen scope decisions
├── EVALUATION_SPEC.md          # Success metrics and evaluation protocol
├── CHECKPOINT_CONVENTION.md    # Model checkpoint naming scheme
├── IMPLEMENTATION_PLAN.md      # Full 8-phase build order
├── configs/                    # Hydra configuration files
│   ├── model/                  # Model architecture configs
│   ├── training/               # Training hyperparameters
│   └── evaluation/             # Evaluation settings
├── src/                        # Source code
│   ├── environment/            # Showdown interface and battle env
│   ├── data/                   # Replay parsing and tensorization
│   ├── models/                 # Model definitions
│   ├── training/               # Training loops
│   ├── bots/                   # Bot implementations
│   ├── evaluation/             # Evaluation harness
│   └── synthetic/              # Synthetic scenario factory
├── tests/                      # Unit and integration tests
├── scripts/                    # Training, evaluation, data scripts
├── data/                       # Raw replays, processed tensors, splits
└── checkpoints/                # Model checkpoints
```

## Setup

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -e ".[dev]"

# Run tests
pytest
```

## Current Status

**Phase 0: Research, Scoping, and Infrastructure** — Complete.

See [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) for the full roadmap.
