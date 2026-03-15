# Checkpoint Naming Convention

_Updated: March 2026 (Gen 3 OU)_

All model checkpoints are saved to `checkpoints/` using the following naming scheme.

## Format

```
{model_type}-{phase}-{dataset_tag}-{epoch:03d}-{val_metric:.4f}-{timestamp}.pt
```

## Fields

| Field | Description | Examples |
|-------|-------------|---------|
| `model_type` | Architecture identifier | `mlp`, `gru`, `transformer` |
| `phase` | Training stage | `bc`, `synthetic`, `rl` |
| `dataset_tag` | Dataset version/size | `10k`, `200k`, `smoke5k` |
| `epoch` | Epoch number (zero-padded) | `001`, `025`, `050` |
| `val_metric` | Validation action NLL | `1.8432` |
| `timestamp` | ISO date | `20260315` |

## Examples

```
mlp-bc-10k-015-2.1034-20260401.pt
transformer-bc-10k-042-1.7821-20260415.pt
transformer-synthetic-10k-003-1.7512-20260420.pt
transformer-rl-200k-010-1.6998-20260501.pt
```

## Current Training Output Convention

The Phase 4 trainer (`scripts/train_phase4.py`) saves artifacts per-seed:
- `checkpoints/<run>/seed_<N>/best_model.pt` — best checkpoint by validation metric
- `checkpoints/<run>/seed_<N>/training_report.json` — training metrics
- `checkpoints/<run>/<variant>_benchmark_summary.json` — aggregate summary

## Special Tags

- `best` symlink always points to the best checkpoint by validation metric: `transformer-bc-10k-best.pt`
- `latest` symlink points to the most recent checkpoint: `transformer-bc-10k-latest.pt`

## Model Variants

| Variant | Config | Params | Wrapper Script |
|---------|--------|--------|----------------|
| P8 | 4L/256d/4H, W20, aux+value | 3.6M | `train_p8_1k.py` |
| P8-Lean | 3L/224d/4H/FFN3x, W5, aux only | ~1.95M | `train_p8_lean.py` |
| P4 | 6L/384d/6H, W20, aux+value | 11.5M | `train_p4_25k.py` |

## Registry

Active checkpoints and their evaluation results are tracked in Weights & Biases. The W&B artifact name matches the checkpoint filename.
