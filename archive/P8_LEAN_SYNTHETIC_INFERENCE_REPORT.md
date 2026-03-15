# P8 Lean Synthetic Inference Scenarios and Evaluation Pipeline

## What changed and why

The original synthetic set was too narrow (all scenarios collapsed to the same forced switch action). This revision creates **5 distinct replay-grounded synthetic scenarios** with one unambiguous correct action each.

- Source states are mined from real replay tensors (`data/processed/battles/*.npz`)
- Each scenario is then converted to a hard-optimal inference test by applying a one-hot legal mask on the target action
- This gives deterministic pass/fail behavior while keeping battle-state features realistic

## Scenario set (5 scenarios + correct choice)

The scenario builder now emits the following fixed set:

1. `s1_priority_attack`
   - **Correct choice:** action `0` (`move 1`)
2. `s2_coverage_attack`
   - **Correct choice:** action `1` (`move 2`)
3. `s3_setup_or_tech_attack`
   - **Correct choice:** action `2` (`move 3`)
4. `s4_cleanup_attack`
   - **Correct choice:** action `3` (`move 4`)
5. `s5_forced_switch`
   - **Correct choice:** action `8` (`switch 2`)

All are labeled `hard_optimal` with `acceptable_actions=[expected_action]`.

## Pipeline components

1. `scripts/build_p8_lean_synthetic_scenarios.py`
   - builds `data/synthetic/p8_lean_scenarios.json`
   - records source replay turn and synthetic mask details
2. `scripts/evaluate_p8_lean_synthetic_inference.py`
   - loads a P8 Lean/BattleTransformer checkpoint
   - runs model inference on each scenario
   - outputs metrics + traces to `reports/p8_lean_synthetic_inference_report.json`

## How to run

### 1) Build synthetic scenarios

```bash
python scripts/build_p8_lean_synthetic_scenarios.py \
  --data-dir data/processed \
  --output data/synthetic/p8_lean_scenarios.json
```

### 2) Run P8 Lean inference evaluation

```bash
python scripts/evaluate_p8_lean_synthetic_inference.py \
  --checkpoint /path/to/p8_lean/best_model.pt \
  --scenario-file data/synthetic/p8_lean_scenarios.json \
  --output reports/p8_lean_synthetic_inference_report.json \
  --device cpu
```

## Expected interpretation

Because each synthetic scenario has exactly one legal action in its test mask:

- expected `legality_rate = 1.0`
- expected `top1_accuracy = 1.0`
- any miss usually indicates inference pipeline issues (checkpoint/config mismatch, tensor-shape mismatch, or masking path bugs)
