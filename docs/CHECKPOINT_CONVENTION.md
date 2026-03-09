# Checkpoint Naming Convention

All model checkpoints are saved to `checkpoints/` using the following naming scheme:

## Format

```
{model_type}-{phase}-{dataset_tag}-{epoch:03d}-{val_metric:.4f}-{timestamp}.pt
```

## Fields

| Field | Description | Examples |
|-------|-------------|---------|
| `model_type` | Architecture identifier | `mlp`, `gru`, `transformer` |
| `phase` | Training phase | `bc`, `synthetic`, `rl` |
| `dataset_tag` | Dataset version/size | `200k`, `500k`, `smoke5k` |
| `epoch` | Epoch number (zero-padded) | `001`, `025`, `050` |
| `val_metric` | Validation action NLL | `1.8432` |
| `timestamp` | ISO date | `20260315` |

## Examples

```
mlp-bc-200k-015-2.1034-20260401.pt
transformer-bc-200k-042-1.7821-20260415.pt
transformer-synthetic-200k-003-1.7512-20260420.pt
transformer-rl-500k-010-1.6998-20260501.pt
```

## Special Tags

- `best` symlink always points to the best checkpoint by validation metric: `transformer-bc-200k-best.pt`
- `latest` symlink points to the most recent checkpoint: `transformer-bc-200k-latest.pt`

## Registry

Active checkpoints and their evaluation results are tracked in Weights & Biases. The W&B artifact name matches the checkpoint filename.
