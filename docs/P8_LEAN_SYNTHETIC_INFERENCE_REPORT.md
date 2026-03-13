# P8 Lean Synthetic Inference Scenarios and Evaluation Pipeline

## What this adds

This report introduces a lightweight inference stress-test pipeline for the P8 Lean imitation-learning model:

1. `scripts/build_p8_lean_synthetic_scenarios.py`
   - mines replay-tensor states from `data/processed/battles/*.npz`
   - builds **5 synthetic scenarios** with a **hard-optimal** (single overwhelmingly-correct) choice
2. `scripts/evaluate_p8_lean_synthetic_inference.py`
   - loads a P8 Lean checkpoint
   - runs inference on the 5 synthetic scenarios
   - writes a JSON report with aggregate and per-scenario metrics

## Scenario design

The synthetic scenarios are grounded in replay data and use a strict criterion for “overwhelmingly correct”:

- each chosen state has exactly **one legal action** in replay tensors (`num_legal = 1`)
- label type: `hard_optimal`
- expected action = that sole legal action

This directly stress-tests inference-time legality handling and deterministic forced-choice decision quality.

## Files and outputs

- Scenario builder output:
  - `data/synthetic/p8_lean_scenarios.json`
- Inference evaluation output:
  - `reports/p8_lean_synthetic_inference_report.json`

## How to run

### 1) Build the synthetic scenarios

```bash
python scripts/build_p8_lean_synthetic_scenarios.py \
  --data-dir data/processed \
  --output data/synthetic/p8_lean_scenarios.json \
  --num-scenarios 5
```

### 2) Evaluate a P8 Lean checkpoint on scenarios

```bash
python scripts/evaluate_p8_lean_synthetic_inference.py \
  --checkpoint /path/to/p8_lean/best_model.pt \
  --scenario-file data/synthetic/p8_lean_scenarios.json \
  --output reports/p8_lean_synthetic_inference_report.json \
  --device cpu
```

## Report interpretation

The evaluation script reports:

- `top1_accuracy`: exact match to hard-optimal action
- `acceptable_set_hit_rate`: hit rate on acceptable set (same as top-1 here)
- `legality_rate`: whether predicted actions are legal under scenario masks
- `per_scenario` details:
  - expected action and probability
  - predicted action and probability
  - top-3 action probabilities

For these hard-optimal forced-choice scenarios, expected behavior is:

- legality rate = `1.0`
- top1 accuracy = `1.0`

Any failure indicates inference pipeline mismatch (checkpoint/config mismatch, tensor-shape issues, or masking drift).

## Notes on alignment with Phase 4 and P8 Lean inference proposal

- Uses the same BattleTransformer inference path used in Phase 4 training/eval (`src/models/battle_transformer.py`).
- Respects proposal guidance to include synthetic inference evaluation with reproducible scenario artifacts.
- Produces deterministic, inspectable JSON artifacts suitable for future expansion into broader scenario families.
