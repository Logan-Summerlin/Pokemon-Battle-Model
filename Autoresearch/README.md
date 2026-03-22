# AutoResearch: Pokemon Battle Model Phase 4 Optimization

Automated research harness for perfecting imitation learning on **>1300 Elo Gen 3 OU (ADV) singles** from the Metamon dataset.

## Quick Start

```bash
# 1. Setup (from parent Pokemon-Battle-Model repo)
pip install -e .

# 2. Evaluate anchor checkpoint
python Autoresearch/eval_harness.py \
    --checkpoint checkpoints/phase4_p8_lean_50k/seed_42/best_model.pt \
    --data-dir data/processed \
    --output Autoresearch/results/anchor.json

# 3. Run an experiment
python Autoresearch/run_experiment.py \
    --name "window_size_5" \
    --parent anchor \
    --tier 1 \
    --config-override max_window=5 \
    --budget-epochs 10

# 4. View leaderboard
python Autoresearch/leaderboard.py
```

## Directory Structure

```
Autoresearch/
├── CLAUDE.md                    # Agent instructions (Claude Code)
├── AGENTS.md                    # Agent coordination guide (Claude Code + Codex)
├── AUTORESEARCH_PLAN.md         # Full implementation plan
├── RUNPOD_SETUP_GUIDE.md        # Cloud setup guide
├── README.md                    # This file
├── eval_harness.py              # Unified evaluation command
├── run_experiment.py            # Bounded experiment launcher
├── leaderboard.py               # Experiment ranking
├── experiment_registry.json     # Machine-readable experiment log
├── .claude/
│   ├── settings.json            # Agent permission settings
│   └── rules/
│       └── memory.md            # Agent memory/context
├── configs/
│   └── anchor.yaml              # Frozen anchor configuration
├── notes/
│   └── 000_anchor.md            # Anchor experiment analysis
└── results/                     # Evaluation output JSONs
```

## Anchor Baseline

| Metric | Value |
|--------|-------|
| Top-1 accuracy | 63.21% |
| Top-3 accuracy | 89.27% |
| Architecture | 3L / 224d / 4H / FFN 3x (~1.72M params) |
| Context window | 2 turns |
| Dataset | 50K battles |

## Target

- **Realistic**: 70-75% Top-1 accuracy
- **Stretch**: 77%+ Top-1 accuracy
- **Switch accuracy**: >= 55% (up from 37-48%)

## Key Documents

- [CLAUDE.md](CLAUDE.md) — Full agent instructions, approved edit surface, hidden info doctrine
- [AGENTS.md](AGENTS.md) — Agent roles, coordination rules, experiment note template
- [AUTORESEARCH_PLAN.md](AUTORESEARCH_PLAN.md) — Detailed experiment plan with phases AR-0 through AR-5
- [RUNPOD_SETUP_GUIDE.md](RUNPOD_SETUP_GUIDE.md) — RunPod A40 setup and cost estimates
